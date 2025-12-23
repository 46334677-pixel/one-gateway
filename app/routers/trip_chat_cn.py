import json
import os
import re
import urllib.parse
import urllib.request
import logging
import uuid
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, ConfigDict, root_validator, validator, field_validator

from app.llm_cn import (
    ChatMessage,
    TripChatLLMInput,
    TripChatLLMDecision,
    SYSTEM_PROMPT,
    call_qwen_for_trip_chat,
)
from app.routers.trip import TripPlanRequest, TripPlanResponse, TripPlanMeta, trip_plan
from app.routers import kb  # 鏂板锛氱煡璇嗗簱
from app.utils.date_range import extract_cn_date_range, infer_cn_date_range_from_text

router = APIRouter(prefix="/cn/v1", tags=["trip_chat"])
logger = logging.getLogger(__name__)

# ---------- 楂樺痉宸ュ叿 ----------
def _http_get_json(url: str, params: Dict[str, Any], timeout: float = 5.0):
    try:
        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{qs}", timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


def _fetch_gaode_first(keyword: str, city: str) -> Optional[Dict[str, Any]]:
    key = os.getenv("GAODE_KEY") or os.getenv("AMAP_KEY")
    if not key or not keyword:
        return None
    data = _http_get_json(
        "https://restapi.amap.com/v3/place/text",
        {
            "key": key,
            "keywords": keyword,
            "city": city,
            "offset": 1,
            "page": 1,
            "extensions": "all",
        },
    )
    if not data or data.get("status") != "1":
        return None
    pois = data.get("pois") or []
    return pois[0] if pois else None


def _enrich_spot(spot: Dict[str, Any], city: str) -> Dict[str, Any]:
    poi = _fetch_gaode_first(spot.get("name"), city)
    photos = poi.get("photos") if poi else []
    photo = photos[0].get("url") if photos and photos[0].get("url") else None
    return {
        "name": spot.get("name", ""),
        "category": spot.get("category", ""),
        "desc": spot.get("brief_desc", ""),
        "photo": photo,
        "photos": [photo] if photo else [],
        "address": poi.get("address") if poi else "",
        "opentime": (poi.get("opentime") or poi.get("open_time") or poi.get("business_time") or "") if poi else "",
        "url": f"https://www.amap.com/place/{poi.get('id')}" if poi and poi.get("id") else "",
        "source": "gaode",
        "type": "鏅偣",
    }

# ---------- 璇锋眰/鍝嶅簲妯″瀷 ----------
class TripChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_id: Optional[str] = None
    history: List[ChatMessage] = []
    # FE minimal payload compatibility: accepts `text` and maps into `input_text`
    text: Optional[str] = None
    input_text: Optional[str] = None
    slots: Optional["SlotsInput"] = None
    current_slots: Optional[TripPlanRequest] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    origin: Optional[str] = None
    default_origin: Optional[str] = Field(
        None, description="鏍规嵁鐢ㄦ埛瀹氫綅鎺ㄦ柇鐨勯粯璁ゅ嚭鍙戝湴锛屼緥濡傦細姝︽眽 / 涓婃捣"
    )
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None

    @root_validator(pre=True)
    def _compat_text(cls, values):
        if not isinstance(values, dict):
            return values
        if not values.get("input_text") and values.get("text"):
            values["input_text"] = values.get("text")
        return values

    @validator("input_text")
    def _input_text_required(cls, v):
        s = "" if v is None else str(v)
        if not s.strip():
            raise ValueError("input_text is required")
        return s


class SlotsInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    date_range: Optional[Union[str, Dict[str, Any]]] = None

    @field_validator("date_range", mode="before")
    @classmethod
    def normalize_date_range(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            start = v.get("start_date") or v.get("start")
            end = v.get("end_date") or v.get("end")
            if start or end:
                return [start, end]
            return None
        return v


class TripChatResponse(BaseModel):
    reply: str
    history: List[ChatMessage]
    slots: Optional[TripPlanRequest] = None
    trip_plan: Optional[TripPlanResponse] = None
    resources: Optional[Dict[str, Any]] = None  # 鍥炬枃璧勬簮

# ---------- 宸ュ叿锛氱敤鎴疯疆娆′笌妲戒綅鐚滄祴 ----------
def _count_user_turns(history: List[ChatMessage]) -> int:
    return sum(1 for m in history if m.role == "user")

def _guess_destination(text: str) -> Optional[str]:
    # 1) 鈥滃幓/鍒?鎯冲幓 + 鍦板悕 + 锛堢帺锛塏澶?鏃モ€濅紭鍏堝尮閰?
    m = re.search(
        r"(?:鍘粅鍒皘鍘诲線|鎯冲幓|鎯冲幓鍒皘鍘讳竴瓒?"
        r"([\u4e00-\u9fa5A-Za-z]{2,10})"
        r"(?:鐜??\s*\d+\s*(?:澶﹟鏃?",
        text,
    )
    if m:
        dest = m.group(1)
        dest = re.sub(r"(甯倈鍖簗鍘縷鐜?$", "", dest)
        return dest

    # 2) 鈥滃ぇ闃?澶╂父/涓滀含涓夋棩娓糕€濊繖绫昏〃杈?
    m2 = re.search(
        r"([\u4e00-\u9fa5A-Za-z]{2,10})\s*(?:\d+\s*(?:澶﹟鏃?|[浜屼笁鍥涗簲鍏竷鍏節鍗乚+\s*(?:澶﹟鏃?)\s*娓?",
        text,
    )
    if m2:
        dest = m2.group(1)
        dest = re.sub(r"(甯倈鍖簗鍘?$", "", dest)
        return dest

    # 3) 閫€鍖栧埌鏈€鍩虹鐨勨€滃幓/鍒?鎯冲幓 + 鍦板悕鈥?
    m3 = re.search(r"(?:鍘粅鍒皘鍘诲線|鎯冲幓|鎯冲幓鍒皘鍘讳竴瓒?([\u4e00-\u9fa5A-Za-z]{2,10})", text)
    if m3:
        dest = m3.group(1)
        dest = re.sub(r"(甯倈鍖簗鍘?$", "", dest)
        return dest
    return None

def _guess_days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:澶﹟鏃?", text)
    if m:
        return max(1, min(int(m.group(1)), 10))
    return None

def _guess_origin(text: str) -> Optional[str]:
    m = re.search(r"浠?[\u4e00-\u9fa5A-Za-z]{2,10})鍑哄彂", text)
    return m.group(1) if m else None

def _guess_preferences(text: str) -> List[str]:
    prefs = []
    for kw in ["浜插瓙", "缇庨", "鑷劧", "娴峰矝", "娴?, "閰掑簵", "浼戦棽", "鍗氱墿棣?, "澶滄櫙", "璐墿"]:
        if kw in text and kw not in prefs:
            prefs.append(kw)
    return prefs

def _valid_date_range(dr: Optional[List[str]]) -> bool:
    if not dr or not isinstance(dr, list) or len(dr) != 2:
        return False
    try:
        import datetime as dt

        dt.date.fromisoformat(str(dr[0]))
        dt.date.fromisoformat(str(dr[1]))
        return True
    except Exception:
        return False

def _fill_defaults(slots: TripPlanRequest) -> TripPlanRequest:
    data = slots.dict()
    # 鏃ユ湡鑼冨洿鍏滃簳涓虹┖鍒楄〃
    if not data.get("date_range"):
        data["date_range"] = []

    # 棰勭畻妗ｄ綅榛樿
    if data.get("budget_level") is None:
        data["budget_level"] = "medium"

    # 鎴愪汉/鍎跨榛樿锛氫粎鍦?adults 涓?people_count 閮界己澶辨椂璁剧疆
    adults = data.get("adults")
    people_count = data.get("people_count")
    if adults is None and people_count is None:
        data["adults"] = 2
        if data.get("children") is None:
            data["children"] = 0

    return TripPlanRequest.parse_obj(data)

def _missing_fields(slots: TripPlanRequest) -> List[str]:
    missing = []
    if not slots.origin:
        missing.append("鍑哄彂鍦?)
    if not slots.destination:
        missing.append("鐩殑鍦?)
    if slots.adults is None and slots.people_count is None:
        missing.append("鎴愪汉浜烘暟")
    return missing

def _is_generic_ack(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return re.match(r"^(鎴戝凡(缁??(鏀跺埌|浜嗚В).{0,10}(闇€姹倈淇℃伅)?[浜嗐€傦紒!]?)$", t) is not None

def _needs_recommendations(text: str) -> bool:
    return bool(re.search(r"(鎺ㄨ崘|鐜╂硶|鐗硅壊椤圭洰|椤圭洰|婊戦洩|娓╂硥|鎬庝箞鐜?", text or ""))

def _count_actionable_items(text: str) -> int:
    lines = (text or "").splitlines()
    return sum(1 for line in lines if re.match(r"^\s*(?:\d+[\.\)銆乚|[-鈥)", line.strip()))

def _build_reco_list(dest: str) -> str:
    d = (dest or "").strip()
    if "闀跨櫧灞? in d:
        return "\n".join(
            [
                "浣犻棶鐨勨€滅壒鑹查」鐩?鐜╂硶鈥濓紝闀跨櫧灞卞父瑙佺殑楂樹环鍊间綋楠屾竻鍗曪細",
                "1锛夋粦闆細涓囪揪搴﹀亣鍖烘粦闆満锛堥厤濂楁垚鐔燂級锛屽寳鍧?瑗垮潯閮ㄥ垎瀛ｈ妭涔熸湁鐜╅洩椤圭洰",
                "2锛夋俯娉夛細搴﹀亣鍖烘俯娉?閰掑簵娓╂硥锛堟洿閫傚悎鏀炬澗锛?,
                "3锛夊ぉ姹?缁忓吀鏅尯锛氬寳鍧?瑗垮潯鐪嬪ぉ姹狅紙鍙楀ぉ姘斿奖鍝嶈緝澶э紝闇€鐪嬪綋澶╁紑鏀撅級",
                "4锛夊啲瀛ｉ檺瀹氾細闆惧噰銆侀洩鏅媿鐓с€佸啺鐎?鏋楁捣闆師绾胯矾锛堢湅姘旇薄鏉′欢锛?,
                "5锛夌編椋熸墦鍗★細閾侀攨鐐栥€佸北閲庤彍銆佸喎姘撮奔绛夊湴鏂圭壒鑹?,
            ]
        )
    return "\n".join(
        [
            f"{d} 鐨勭帺娉曞缓璁竻鍗曪紙鍏堢粰 5 鏉″彲鎵ц椤癸級锛?,
            "1锛夋牳蹇冨湴鏍?浠ｈ〃鎬ф櫙鍖烘墦鍗★紙浼樺厛瀹夋帓 1-2 涓級",
            "2锛夊煄甯傜編椋熻矾绾匡紙鏈湴蹇呭悆 + 澶滃競/灏忓悆琛楋級",
            "3锛夌壒鑹蹭綋楠岄」鐩紙瀛ｈ妭闄愬畾/涓婚涔愬洯/姘戜織浣撻獙锛?,
            "4锛夊懆杈逛竴鏃ユ父锛堣嚜鐒?鍙ら晣/婀栨捣灞辨按浠婚€夊叾涓€锛?,
            "5锛夎交鏉句紤闂叉椂娈碉紙鍜栧暋棣?娓╂硥/鍏洯/澶滄櫙锛?,
        ]
    )

def _build_followup_questions(has_dest: bool) -> str:
    if has_dest:
        return "\n".join(
            [
                "鍐嶇‘璁や袱涓偣锛岀粰浣犳妸瀹夋帓鍋氬緱鏇磋创鍚堬細",
                "A. 浣犳洿鍋忔埛澶栬繍鍔ㄨ繕鏄紤闂插害鍋囷紵",
                "B. 棰勭畻澶ф浠€涔堟。浣嶏紙缁忔祹/鑸掗€?楂樼锛夛紵",
            ]
        )
    return "\n".join(
        [
            "涓轰簡缁欎綘鏇村噯鐨勬帹鑽愶紝鍏堢‘璁?3 鐐癸細",
            "1锛夌洰鐨勫湴鏄摢锛?,
            "2锛夐璁″嚑澶╋紵",
            "3锛夊悓琛屼汉鏁颁笌棰勭畻妗ｄ綅锛?,
        ]
    )

# ---------- LLM 璋冪敤鍙婅В鏋?----------
def _build_llm_input(prompt: str, req: TripChatRequest, system_prompt: Optional[str] = None) -> TripChatLLMInput:
    lat = req.location_lat if req.location_lat is not None else req.user_lat
    lng = req.location_lng if req.location_lng is not None else req.user_lng
    return TripChatLLMInput(
        user_id=req.user_id or "",
        history=req.history,
        new_user_message=prompt,
        location_lat=lat,
        location_lng=lng,
        system_prompt=system_prompt,
    )

def _force_json_prompt(user_text: str) -> str:
    return f"""{user_text}
璇峰彧杈撳嚭 JSON锛堟棤 Markdown锛屾棤鑷劧璇█鍓嶅悗缂€锛夛紝鏍煎紡锛?
{{
  "summary": "涓€鍙ヨ瘽姒傛嫭",
  "spots": [
    {{"name": "鏅偣1", "category": "鍒嗙被/鏍囩", "brief_desc": "2-3 鍙ョ畝浠?}},
    {{"name": "鏅偣2", "category": "鍒嗙被/鏍囩", "brief_desc": "2-3 鍙ョ畝浠?}}
  ]
}}
鏃犳硶鐢熸垚鏃惰緭鍑? {{"summary": "鏃犳硶鐢熸垚", "spots": []}}
"""

def _retry_json_prompt(user_text: str) -> str:
    return f"""{user_text}
鍐嶆鎻愰啋锛氬彧杈撳嚭 JSON锛堟棤 Markdown锛夛紝鏍煎紡:
{{"summary":"...","spots":[{{"name":"...","category":"...","brief_desc":"..."}}]}}
鏃犳硶鐢熸垚鏃惰緭鍑?{{"summary":"鏃犳硶鐢熸垚","spots":[]}}
"""

def _clean_json_text(txt: str) -> str:
    clean = txt.strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE | re.DOTALL).strip()
    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        clean = clean[first : last + 1]
    return clean

def _try_parse_json(txt: str) -> Optional[Dict[str, Any]]:
    for candidate in [txt, _clean_json_text(txt)]:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

def _call_llm_for_json(req: TripChatRequest, system_prompt: Optional[str] = None):
    decision = call_qwen_for_trip_chat(
        _build_llm_input(_force_json_prompt(req.input_text), req, system_prompt)
    )
    reply_text = decision.reply or ""
    parsed = _try_parse_json(reply_text)
    if parsed is not None:
        return reply_text, parsed, decision
    decision2 = call_qwen_for_trip_chat(
        _build_llm_input(_retry_json_prompt(req.input_text), req, system_prompt)
    )
    reply_text2 = decision2.reply or reply_text
    parsed2 = _try_parse_json(reply_text2)
    return reply_text2, parsed2, decision2

# ---------- 鐭ヨ瘑搴撹浆鎹?----------
def _kb_to_resources(kb_items):
    tickets = []
    for it in kb_items:
        photos = it.get("photos") or []
        tickets.append(
            {
                "name": it.get("title") or it.get("name", ""),
                "category": "鏀荤暐绮鹃€?,
                "desc": it.get("summary", ""),
                "photo": photos[0] if photos else None,
                "photos": photos,
                "address": "",
                "opentime": "",
                "url": it.get("url"),
                "source": it.get("platform") or "kb",
                "type": "鏅偣",
            }
        )
    return {"tickets": tickets} if tickets else None

# ---------- 涓绘祦绋?----------
# ---------- 涓绘祦绋?----------
@router.post("/trip_chat", response_model=TripChatResponse)
def trip_chat(req: TripChatRequest, response: Response, request: Request) -> TripChatResponse:
    """
    涓枃 TripChat 涓绘祦绋嬶細
    - 缁撳悎 origin / default_origin 琛ュ厖鍑哄彂鍦版彁绀猴紱
    - 鍛戒腑鐭ヨ瘑搴撴椂鐢熸垚鍥炬枃 resources锛堜絾涓嶉樆姝㈠悗缁绋嬭鍒掞級锛?
    - 璋?LLM 鐢熸垚鏅偣 JSON锛堢洰鍓嶄富瑕佺敤浜庤祫婧愬睍绀猴紝鍙涓衡€滈棽鑱婂洖绛斺€濓級锛?
    - 灏濊瘯鏋勯€?TripPlanRequest 妲戒綅骞惰皟鐢?trip_plan 鐢熸垚琛岀▼锛?
    - 鍦?reply 涓€傚綋鍔犱笂鈥滆崏绋胯鏄庘€濓紝浣?trip_plan 濮嬬粓閫氳繃瀛楁杩斿洖缁欏墠绔€?
    """
    trace_id = getattr(request.state, "trace_id", None) or uuid.uuid4().hex
    response.headers["X-Trace-Id"] = trace_id
    date_hint_needed = False

    # ----- 鍑哄彂鍦版彁绀?-----
    origin_hint = ""
    origin_value = req.origin
    if not origin_value and req.current_slots and getattr(req.current_slots, "origin", None):
        origin_value = req.current_slots.origin

    if origin_value:
        # 鐢ㄦ埛宸茬粡鏄庣‘鍑哄彂鍩庡競
        origin_hint = (
            "绯荤粺琛ュ厖鑳屾櫙淇℃伅锛氱敤鎴峰凡缁忔湁鏄庣‘鐨勫嚭鍙戝煄甯傦紝"
            f"褰撳墠鍋囧畾鍑哄彂鍦颁负銆寋origin_value}銆嶃€?
            "鍦ㄨ鍒掕绋嬫椂锛岃浼樺厛浠ヨ繖涓煄甯備綔涓哄嚭鍙戝湴銆?
        )
    elif req.default_origin:
        # 娌℃湁鏄惧紡 origin锛岀敤 default_origin 浣滀负鎺ㄦ柇鍑虹殑鍑哄彂鍦?
        origin_hint = (
            "绯荤粺琛ュ厖鑳屾櫙淇℃伅锛氭牴鎹敤鎴锋渶杩戜竴娆″畾浣嶆帹鏂紝"
            f"鐢ㄦ埛澶ф鐜囦綅浜庛€寋req.default_origin}銆嶃€?
            "濡傛灉鐢ㄦ埛鍦ㄥ璇濅腑娌℃湁鐗瑰埆璇存槑鍑哄彂鍩庡競锛?
            "浣犲彲浠ユ殏鏃跺亣璁惧嚭鍙戝湴涓鸿繖閲岋紱"
            "涓€鏃︾敤鎴锋彁渚涗簡鏂扮殑鍑哄彂鍩庡競锛屼互鐢ㄦ埛鐨勬渶鏂拌鏄庝负鍑嗐€?
        )

    system_content = SYSTEM_PROMPT
    if origin_hint:
        system_content = SYSTEM_PROMPT + "\n\n" + origin_hint

    logger.info(
        "TripChat origin=%s default_origin=%s user_lat=%s user_lng=%s",
        origin_value,
        req.default_origin,
        req.user_lat,
        req.user_lng,
    )

    # ----- 1) 鐭ヨ瘑搴擄細涓嶅啀鏃╅€€锛屽彧鍏堣涓?resources -----
    kb_res = kb.kb_search(kb.KbSearch(query=req.input_text, destination=""))
    kb_items = kb_res.get("items") if kb_res else []
    resources_from_kb = _kb_to_resources(kb_items) if kb_items else None

    # ----- 2) 璋?LLM锛堝己绾︽潫 JSON锛屽け璐ラ噸璇曚竴娆★級锛屼繚鐣欏師濮嬪洖澶?-----
    reply_text, parsed_json, decision_used = _call_llm_for_json(req, system_content)
    reply_text = reply_text or ""

    # 鍙妸鈥滄湰杞敤鎴锋秷鎭€濆厛鍔犺繘 history锛岀敤浜庣粺璁?user_turns
    history_with_new_user = list(req.history) + [
        ChatMessage(role="user", content=req.input_text),
    ]
    user_turns = _count_user_turns(history_with_new_user)

    # ----- 3) 瑙ｆ瀽 slots_json锛屾垨鐢?current_slots -----
    slots_from_llm: Optional[TripPlanRequest] = None
    try:
        if getattr(decision_used, "slots_json", None):
            raw_slots = json.loads(decision_used.slots_json) or {}
            # 琛ュ厖鐢ㄦ埛鍧愭爣
            if req.location_lat is not None:
                raw_slots.setdefault("user_lat", req.location_lat)
            elif req.user_lat is not None:
                raw_slots.setdefault("user_lat", req.user_lat)
            if req.location_lng is not None:
                raw_slots.setdefault("user_lng", req.location_lng)
            elif req.user_lng is not None:
                raw_slots.setdefault("user_lng", req.user_lng)
            slots_from_llm = TripPlanRequest.parse_obj(raw_slots)
    except Exception:
        logger.exception("TripChat: failed to parse slots_json")
        slots_from_llm = None

    slots: Optional[TripPlanRequest] = slots_from_llm or req.current_slots
    if slots is None and req.slots is not None:
        try:
            raw_slots = req.slots.dict(exclude_none=True)
            # normalize date_range if provided as string/dict
            dr = raw_slots.get("date_range")
            if isinstance(dr, str):
                raw_slots["date_range"] = []
            elif isinstance(dr, dict):
                start = dr.get("start_date") or dr.get("start")
                end = dr.get("end_date") or dr.get("end")
                raw_slots["date_range"] = [start, end] if (start or end) else []
            slots = TripPlanRequest.parse_obj(raw_slots)
        except Exception:
            logger.exception("TripChat: failed to parse req.slots")
            slots = None

    # ----- 4) 鐚滄Ы浣嶏紙鐩殑鍦?澶╂暟/鍑哄彂鍦?鍋忓ソ锛夛紝骞朵笌 slots 铻嶅悎 -----
    guess_slots = TripPlanRequest()
    text_merge = req.input_text + "\n" + "\n".join(
        [m.content for m in req.history if m.role == "user"]
    )
    guess_slots.destination = _guess_destination(text_merge)
    days = _guess_days(text_merge)
    if days:
        # 鐩墠浠呯敤鈥滃ぉ鏁板瓨鍦ㄢ€濊繖涓俊鍙凤紝涓嶅己琛屾瀯閫?date_range
        guess_slots.date_range = []
    guess_slots.origin = _guess_origin(text_merge)
    guess_slots.preferences = _guess_preferences(text_merge)

    # 濡傛灉瀹屽叏娌℃湁 slots锛屼絾鐚滃埌浜嗙洰鐨勫湴锛屽垯鐢?guess_slots
    if slots is None and guess_slots.destination:
        slots = guess_slots

    # 鐢ㄧ寽娴嬬粨鏋滆ˉ鍏?slots 涓己澶辩殑閮ㄥ垎瀛楁锛堜笉瑕嗙洊宸叉湁鍊硷級
    if slots is not None:
        if not slots.destination and guess_slots.destination:
            slots.destination = guess_slots.destination
        if not slots.origin and guess_slots.origin:
            slots.origin = guess_slots.origin
        if not slots.preferences and guess_slots.preferences:
            slots.preferences = guess_slots.preferences
        # 杩涗竴姝ワ細鑻ヤ粛鏃?origin锛屽皾璇曠敤鏄惧紡 origin 鎴?default_origin 琛ヤ笂
        if not slots.origin:
            if origin_value:
                slots.origin = origin_value
            elif req.default_origin:
                slots.origin = req.default_origin

        # ----- 4.1) 纭畾鎬ф棩鏈熻В鏋愶細浼樺厛浜?LLM锛堝湪缂哄瓧娈靛垽瀹氫箣鍓嶅啓鍥?slots锛?-----
        if not _valid_date_range(slots.date_range):
            extracted = extract_cn_date_range(text_merge)
            if extracted:
                slots.date_range = [extracted["start_date"], extracted["end_date"]]
                logger.info(
                    "TripChat date_range extracted trace_id=%s date_range=%s",
                    trace_id,
                    slots.date_range,
                )
            else:
                inferred = infer_cn_date_range_from_text(text_merge)
                if inferred:
                    slots.date_range = [inferred["start_date"], inferred["end_date"]]
                    date_hint_needed = True
                    logger.info(
                        "TripChat date_range inferred trace_id=%s date_range=%s",
                        trace_id,
                        slots.date_range,
                    )

        # 鍙鏄€滆妭鍋囨棩/鍏冩棪/鍛ㄦ湯鈥濊繖绫昏〃杈句笖娌℃湁鏄庣‘鏃ユ湡鑼冨洿锛屽氨鍔犺交鎻愮ず锛堜笉闃绘柇鐢熸垚锛?        if not extract_cn_date_range(text_merge) and re.search(
            r"(鍏冩棪|鑺傚亣鏃鍛ㄦ湯|鏈懆鏈珅涓嬪懆鏈珅浜斾竴|鍔冲姩鑺倈鍥藉簡|鏄ヨ妭)", text_merge
        ):
            date_hint_needed = True

    # ----- 4.2) empty/generic reply fallback -----
    destination_value = slots.destination if slots is not None else None
    if _needs_recommendations(req.input_text):
        if destination_value:
            if _is_generic_ack(reply_text) or _count_actionable_items(reply_text) < 5:
                reply_text = _build_reco_list(destination_value) + "\n\n" + _build_followup_questions(True)
        else:
            reply_text = _build_followup_questions(False)
    elif _is_generic_ack(reply_text):
        reply_text = _build_followup_questions(bool(destination_value))

    # ----- 5) 璋?TripPlan锛氬彧瑕佹湁 slots 灏卞皾璇曡皟鐢紝璁?TripPlan 鑷繁鍒ゆ柇淇℃伅鏄惁鍏呭垎 -----
    trip_plan_result: Optional[TripPlanResponse] = None
    missing: List[str] = []

    if slots is not None:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        try:
            logger.info("TripChat: calling trip_plan with slots=%s", safe_slots.dict())
            trip_plan_result = trip_plan(safe_slots, request)

            if trip_plan_result is not None and date_hint_needed:
                msg = "鏈‘璁ゅ嚭琛屾棩鏈燂紝钀ヤ笟鏃堕棿/棰勭害璇蜂互瀹為檯鏃ユ湡鏍稿"
                if trip_plan_result.meta is None:
                    trip_plan_result.meta = TripPlanMeta(
                        trace_id=f"tc_{trace_id}",
                        quality_score=0,
                        warnings=[msg],
                        fixed=None,
                    )
                else:
                    warnings = list(trip_plan_result.meta.warnings or [])
                    if msg not in warnings:
                        warnings.insert(0, msg)
                    trip_plan_result.meta.warnings = warnings[:5]
        except Exception:
            logger.exception("TripChat: error when calling trip_plan")
            trip_plan_result = None

    final_mode = trip_plan_result.mode if trip_plan_result is not None else "no-trip-plan"
    logger.info(
        "TripChat trace_id=%s user_text=%s slots.date_range=%s missing_fields=%s final_mode=%s",
        trace_id,
        req.input_text,
        (slots.date_range if slots is not None else None),
        missing,
        final_mode,
    )

    # ----- 6) 鏍规嵁缂哄け瀛楁澧炲姞鈥滆崏绋胯鏄庘€濆墠缂€锛堟渶澶氳В閲婁竴娆★級 -----
    if trip_plan_result is not None and missing:
        # 鏂囨浜掓枼锛氬鏋滀粛缂衡€滃嚭琛屾棩鏈熲€濓紝涓嶈鍦ㄥ悓涓€鍥炲閲屽嚭鐜扳€滃凡鏀跺埌鏃ユ湡鈥濅箣绫昏〃杩?
        if "鍑鸿鏃ユ湡" in missing:
            reply_text = re.sub(r"^.*?(宸叉敹鍒皘鏀跺埌).*(鍑鸿)?鏃ユ湡.*?$", "", reply_text, flags=re.MULTILINE).strip()
        draft_phrase = "鎴戝厛鎸夌洰鍓嶄俊鎭嚭浜嗕竴鐗堣崏绋?
        already_notified = any(
            (m.role == "assistant" and draft_phrase in (m.content or ""))
            for m in req.history
        )
        missing_text = "銆?.join(missing)
        if not already_notified:
            prefix = f"{draft_phrase}锛堢己灏戯細{missing_text}锛夛紝璇疯ˉ鍏呭悗鎴戝啀浼樺寲銆?
        else:
            prefix = f"鐜板湪杩樼己锛歿missing_text}锛屾柟渚垮憡璇夋垜杩欎簺淇℃伅鍚楋紵"
        reply_text = prefix + "\n" + (reply_text or "")

    # ----- 7) 鏈€缁?history锛氭妸鏈疆 assistant 鍥炲鍔犺繘鍘?-----
    new_history = history_with_new_user + [
        ChatMessage(role="assistant", content=reply_text),
    ]

    # ----- 8) 缁勮鍝嶅簲锛歳esources_from_kb 鍜?trip_plan 閮戒竴璧疯繑鍥?-----
    return TripChatResponse(
        reply=reply_text or "杩欒竟鐜板湪鏈夌偣蹇欙紝浣犲彲浠ョ◢鍚庡啀璇曡瘯銆?,
        history=new_history,
        slots=slots,
        trip_plan=trip_plan_result,
        resources=resources_from_kb,
    )
