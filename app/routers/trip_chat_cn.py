import json
import os
import re
import urllib.parse
import urllib.request
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.llm_cn import (
    ChatMessage,
    TripChatLLMInput,
    TripChatLLMDecision,
    SYSTEM_PROMPT,
    call_qwen_for_trip_chat,
)
from app.routers.trip import TripPlanRequest, TripPlanResponse, trip_plan
from app.routers import kb  # æ°å¢ï¼ç¥è¯åº

router = APIRouter(prefix="/cn/v1", tags=["trip_chat"])
logger = logging.getLogger(__name__)

# ---------- é«å¾·å·¥å
· ----------
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
        "type": "æ¯ç¹",
    }

# ---------- è¯·æ±/ååºæ¨¡å ----------
class TripChatRequest(BaseModel):
    user_id: Optional[str] = None
    history: List[ChatMessage] = []
    input_text: str
    current_slots: Optional[TripPlanRequest] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    origin: Optional[str] = None
    default_origin: Optional[str] = Field(
        None, description="æ ¹æ®ç¨æ·å®ä½æ¨æ­çé»è®¤åºåå°ï¼ä¾å¦ï¼æ­¦æ± / ä¸æµ·"
    )
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None


class TripChatResponse(BaseModel):
    reply: str
    history: List[ChatMessage]
    slots: Optional[Dict[str, Any]] = None
    trip_plan: Optional[TripPlanResponse] = None
    resources: Optional[Dict[str, Any]] = None  # å¾æèµæº

# ---------- å·¥å
·ï¼ç¨æ·è½®æ¬¡ä¸æ§½ä½çæµ ----------
def _count_user_turns(history: List[ChatMessage]) -> int:
    return sum(1 for m in history if m.role == "user")

def _guess_destination(text: str) -> Optional[str]:
    m = re.search(r"(?:å»|å°|å»å¾|æ³å»|æ³å»å°|å»ä¸è¶)([\u4e00-\u9fa5A-Za-z]{2,10})", text)
    if m:
        return m.group(1)
    m2 = re.search(r"([\u4e00-\u9fa5A-Za-z]{2,10})\s*(?:\d+å¤©|æ¥)æ¸¸", text)
    if m2:
        return m2.group(1)
    return None

def _guess_days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:å¤©|æ¥)", text)
    if m:
        return max(1, min(int(m.group(1)), 10))
    return None

def _guess_origin(text: str) -> Optional[str]:
    m = re.search(r"ä»([\u4e00-\u9fa5A-Za-z]{2,10})åºå", text)
    return m.group(1) if m else None

def _guess_preferences(text: str) -> List[str]:
    prefs = []
    for kw in ["äº²å­", "ç¾é£", "èªç¶", "æµ·å²", "æµ·", "é
åº", "ä¼é²", "åç©é¦", "å¤æ¯", "è´­ç©"]:
        if kw in text and kw not in prefs:
            prefs.append(kw)
    return prefs

def _fill_defaults(slots: TripPlanRequest) -> TripPlanRequest:
    data = slots.dict()
    data.setdefault("adults", 2)
    data.setdefault("children", 0)
    data.setdefault("budget_level", "medium")
    data.setdefault("date_range", data.get("date_range") or [])
    return TripPlanRequest.parse_obj(data)

FIELD_LABELS_CN = {
    "origin": "åºåå°",
    "origin_city": "åºåå°",
    "destination": "ç®çå°",
    "destination_city": "ç®çå°",
    "date_range": "åºè¡æ¥æ",
    "start_date": "åºè¡æ¥æ",
    "end_date": "ç»ææ¥æ",
    "days": "è¡ç¨å¤©æ°",
    "adults": "æäººäººæ°",
    "adult_count": "æäººäººæ°",
    "people_count": "æäººäººæ°",
    "children": "å¿ç«¥äººæ°",
    "child_count": "å¿ç«¥äººæ°",
}


def _to_human_labels(missing_keys: List[str]) -> List[str]:
    return [FIELD_LABELS_CN.get(k, k) for k in missing_keys]


def _missing_fields(slots: TripPlanRequest) -> List[str]:
    missing: List[str] = []
    if not slots.origin:
        missing.append("origin")
    if not slots.destination:
        missing.append("destination")
    if not slots.date_range:
        missing.append("date_range")
    if slots.adults is None and slots.people_count is None:
        missing.append("adults")
    return missing


def _build_slots_payload(slots: Optional[TripPlanRequest]) -> Dict[str, Any]:
    """
    æé ç»æåç slots ä¿¡æ¯ï¼å
å«åå§å­æ®µãå·²å¡«/ç¼ºå¤±å­æ®µåä¸­ææ ç­¾ã
    """
    if not slots:
        return {}
    try:
        raw = slots.dict()
    except Exception:
        # å
åºï¼æ dict å¤ç
        raw = dict(slots) if hasattr(slots, "items") else {}
    filled = [k for k, v in raw.items() if v not in (None, "", [], {})]
    missing_keys = _missing_fields(slots)
    missing_labels = _to_human_labels(missing_keys)
    payload = dict(raw)
    payload.update(
        {
            "raw": raw or None,
            "filled": filled,
            "missing": missing_keys,
            "missing_labels": missing_labels,
        }
    )
    return payload


def _get_role(m: Any) -> str:
    """
    å
¼å®¹åå²æ¶æ¯å¯è½æ¯ Pydantic å¯¹è±¡æ dictï¼ç»ä¸è¿åå°å roleã
    """
    if isinstance(m, dict):
        return (m.get("role") or "").lower()
    return (getattr(m, "role", "") or "").lower()


def _get_content(m: Any) -> str:
    """
    å
¼å®¹åå²æ¶æ¯å¯è½æ¯ Pydantic å¯¹è±¡æ dictï¼ç»ä¸è¿å contentã
    """
    if isinstance(m, dict):
        return m.get("content") or ""
    return getattr(m, "content", "") or ""

# ---------- LLM è°ç¨åè§£æ ----------
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
è¯·åªè¾åº JSONï¼æ  Markdownï¼æ èªç¶è¯­è¨ååç¼ï¼ï¼æ ¼å¼ï¼
{{
  "summary": "ä¸å¥è¯æ¦æ¬",
  "spots": [
    {{"name": "æ¯ç¹1", "category": "åç±»/æ ç­¾", "brief_desc": "2-3 å¥ç®ä»"}},
    {{"name": "æ¯ç¹2", "category": "åç±»/æ ç­¾", "brief_desc": "2-3 å¥ç®ä»"}}
  ]
}}
æ æ³çææ¶è¾åº: {{"summary": "æ æ³çæ", "spots": []}}
"""

def _retry_json_prompt(user_text: str) -> str:
    return f"""{user_text}
åæ¬¡æéï¼åªè¾åº JSONï¼æ  Markdownï¼ï¼æ ¼å¼:
{{"summary":"...","spots":[{{"name":"...","category":"...","brief_desc":"..."}}]}}
æ æ³çææ¶è¾åº {{"summary":"æ æ³çæ","spots":[]}}
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

# ---------- ç¥è¯åºè½¬æ¢ ----------
def _kb_to_resources(kb_items):
    tickets = []
    for it in kb_items:
        photos = it.get("photos") or []
        tickets.append(
            {
                "name": it.get("title") or it.get("name", ""),
                "category": "æ»ç¥ç²¾é",
                "desc": it.get("summary", ""),
                "photo": photos[0] if photos else None,
                "photos": photos,
                "address": "",
                "opentime": "",
                "url": it.get("url"),
                "source": it.get("platform") or "kb",
                "type": "æ¯ç¹",
            }
        )
    return {"tickets": tickets} if tickets else None

# ---------- ä¸»æµç¨ ----------
@router.post("/trip_chat", response_model=TripChatResponse)
def trip_chat(req: TripChatRequest) -> TripChatResponse:
    origin_hint = ""
    origin_value = req.origin
    if not origin_value and req.current_slots and getattr(req.current_slots, "origin", None):
        origin_value = req.current_slots.origin
    if origin_value:
        origin_hint = f"???????????????{origin_value}????????????????????????"
    elif req.default_origin:
        origin_hint = (
            "???????????????????????????"
            f"?{req.default_origin}?????????????????????????"
            "????????????????"
            f"???????{req.default_origin}???????????????????????? "
            "????????????????????????"
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

    # 1) ç¥è¯åºä¼å

    kb_res = kb.kb_search(kb.KbSearch(query=req.input_text, destination=""))
    kb_items = kb_res.get("items") if kb_res else []
    resources_from_kb = _kb_to_resources(kb_items) if kb_items else None

    # 2) è° LLMï¼å¼ºçº¦æ JSONï¼å¤±è´¥éè¯ä¸æ¬¡ï¼ï¼ä¿çåå§åå¤
    reply_text, parsed_json, decision_used = _call_llm_for_json(req, system_content)

    new_history = list(req.history) + [
        ChatMessage(role="user", content=req.input_text),
        ChatMessage(role="assistant", content=reply_text),
    ]
    user_turns = _count_user_turns(new_history)

    # 3) è§£æ slots_jsonï¼æç¨ current_slots
    slots_from_llm: Optional[TripPlanRequest] = None
    try:
        if getattr(decision_used, "slots_json", None):
            raw_slots = json.loads(decision_used.slots_json)
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
        slots_from_llm = None
    slots: Optional[TripPlanRequest] = slots_from_llm or req.current_slots

    # 4) Guess slots (destination/days/origin/preferences)
    guess_slots = TripPlanRequest()
    text_merge = req.input_text + "\n" + "\n".join([
        _get_content(m) for m in req.history if _get_role(m) == "user"
    ])
    guess_slots.destination = _guess_destination(text_merge)
    days = _guess_days(text_merge)
    if days:
        guess_slots.date_range = []
    guess_slots.origin = _guess_origin(text_merge)
    guess_slots.preferences = _guess_preferences(text_merge)

    if slots is None and guess_slots.destination:
        slots = guess_slots

    # 5) If KB hits, return resources directly
    slots_struct = _build_slots_payload(slots)
    if resources_from_kb:
        return TripChatResponse(
            reply=reply_text,
            history=new_history,
            slots=slots_struct or slots,
            trip_plan=None,
            resources=resources_from_kb,
        )

    # 6) Decide whether to call TripPlan
    should_call_plan = False
    if slots and slots.destination:
        should_call_plan = True
    elif user_turns >= 3 and guess_slots.destination:
        slots = guess_slots
        should_call_plan = True

    trip_plan_result: Optional[TripPlanResponse] = None
    safe_slots: Optional[TripPlanRequest] = None
    missing_labels: List[str] = []
    missing: List[str] = []
    if should_call_plan and slots:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        missing_labels = _to_human_labels(missing)
        try:
            trip_plan_result = trip_plan(safe_slots)
                                    if missing:
                draft_phrase = "我先按目前信息出了一版草稿"
                already_notified = any(
                    (_get_role(m).startswith("assistant") and draft_phrase in _get_content(m))
                    for m in (req.history or [])
                )
                missing_text = "、".join(missing_labels)
                if not already_notified:
                    prefix = f"{draft_phrase}（缺少：{missing_text}），请补充后我再优化。"
                else:
                    prefix = f"现在还缺：{missing_text}，方便告诉我这些信息吗？"
                reply_text = prefix + "\n" + (reply_text or "")
        except Exception:
            trip_plan_result = None

    slots_for_payload = safe_slots or slots
    slots_struct = _build_slots_payload(slots_for_payload)

    if new_history and _get_role(new_history[-1]).startswith("assistant"):
        try:
            new_history[-1].content = reply_text
        except Exception:
            if isinstance(new_history[-1], dict):
                new_history[-1]["content"] = reply_text

    return TripChatResponse(
        reply=reply_text or "?????????????????",
        history=new_history,
        slots=slots_struct or slots_for_payload,
        trip_plan=trip_plan_result,
        resources=None,
    )