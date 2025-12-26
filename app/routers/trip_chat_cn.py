# -*- coding: utf-8 -*-
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
from app.routers import kb  # 新增：知识库
from app.utils.date_range import extract_cn_date_range, infer_cn_date_range_from_text

router = APIRouter(prefix="/cn/v1", tags=["trip_chat"])
logger = logging.getLogger(__name__)

# ---------- 高德工具 ----------
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
        "opentime": (
            (poi.get("opentime") or poi.get("open_time") or poi.get("business_time") or "")
            if poi
            else ""
        ),
        "url": f"https://www.amap.com/place/{poi.get('id')}" if poi and poi.get("id") else "",
        "source": "gaode",
        "type": "景点",
    }


# ---------- 请求/响应模型 ----------
class TripChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)

    # FE minimal payload compatibility: accepts `text` and maps into `input_text`
    text: Optional[str] = None
    input_text: Optional[str] = None

    slots: Optional["SlotsInput"] = None
    current_slots: Optional[TripPlanRequest] = None

    location_lat: Optional[float] = None
    location_lng: Optional[float] = None

    origin: Optional[str] = None
    default_origin: Optional[str] = Field(
        None, description="根据用户定位推断的默认出发地，例如：武汉/上海"
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
    date_range: Optional[Union[str, Dict[str, Any], List[Any]]] = None

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
    resources: Optional[Dict[str, Any]] = None  # 图文资源


# ---------- 工具：用户轮次与槽位猜测 ----------
def _count_user_turns(history: List[ChatMessage]) -> int:
    return sum(1 for m in history if m.role == "user")


def _guess_destination(text: str) -> Optional[str]:
    """
    粗略猜测目的地：
    - “去/到/想去/准备去 + 地名”
    - “XX三日游/XX2天游”
    """
    if not text:
        return None

    m = re.search(r"(?:去|到|想去|想去到|准备去)\s*([\u4e00-\u9fa5A-Za-z]{2,10})", text)
    if m:
        dest = m.group(1)
        dest = re.sub(r"(市|区|县|省)$", "", dest)
        return dest

    m2 = re.search(
        r"([\u4e00-\u9fa5A-Za-z]{2,10})\s*(?:\d+\s*(?:天|日)\s*(?:游|行程)|[二三四五六七八九十]\s*(?:天|日)\s*(?:游|行程))",
        text,
    )
    if m2:
        dest = m2.group(1)
        dest = re.sub(r"(市|区|县|省)$", "", dest)
        return dest

    return None


def _guess_days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:天|日)", text or "")
    if m:
        try:
            return max(1, min(int(m.group(1)), 10))
        except Exception:
            return None
    return None


def _guess_origin(text: str) -> Optional[str]:
    # 兼容：“从武汉出发 / 自武汉出发 / 从武汉启程”
    m = re.search(r"(?:从|自)\s*([\u4e00-\u9fa5A-Za-z]{2,15})\s*(?:出发|启程)", text or "")
    if not m:
        return None
    origin = m.group(1)
    origin = re.sub(r"(市|区|县|省)$", "", origin)
    return origin


def _guess_preferences(text: str) -> List[str]:
    text = text or ""
    prefs: List[str] = []
    keywords = ["亲子", "美食", "自然", "海岛", "海边", "酒店", "休闲", "博物馆", "夜景", "购物", "滑雪", "温泉"]
    for kw in keywords:
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

    # 日期范围兜底为空列表
    if not data.get("date_range"):
        data["date_range"] = []

    # 预算档位默认
    if data.get("budget_level") is None:
        data["budget_level"] = "medium"

    # 成人人数/总人数兜底：仅在 adults 与 people_count 均缺失时设置
    adults = data.get("adults")
    people_count = data.get("people_count")
    if adults is None and people_count is None:
        data["adults"] = 2
        if data.get("children") is None:
            data["children"] = 0

    return TripPlanRequest.parse_obj(data)


def _missing_fields(slots: TripPlanRequest) -> List[str]:
    missing: List[str] = []
    if not getattr(slots, "origin", None):
        missing.append("出发地")
    if not getattr(slots, "destination", None):
        missing.append("目的地")
    if getattr(slots, "adults", None) is None and getattr(slots, "people_count", None) is None:
        missing.append("成人人数")
    return missing


# QUICK_REPLIES: 根据用户意图在缺少 destination 时给出候选
def _dest_quick_replies(text: str) -> List[str]:
    t = text or ""
    if "东北" in t:
        return ["哈尔滨", "长春", "沈阳", "大连", "长白山"]
    if "新疆" in t:
        return ["乌鲁木齐", "喀什", "伊犁", "阿勒泰"]
    if "云南" in t:
        return ["昆明", "大理", "丽江", "香格里拉", "西双版纳"]
    return []


def _is_generic_ack(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t in {"好的", "收到", "明白", "了解", "OK", "ok"}:
        return True
    return bool(re.match(r"^(我已(经)?(收到|了解|明白).{0,10})$", t))


def _needs_recommendations(text: str) -> bool:
    return bool(re.search(r"(推荐|玩法|特色项目|项目|怎么?玩|攻略|安排|路线)", text or ""))


def _count_actionable_items(text: str) -> int:
    lines = (text or "").splitlines()
    return sum(1 for line in lines if re.match(r"^\s*(?:\d+[.)、]|[-*•])", line.strip()))


def _build_reco_list(dest: str) -> str:
    d = (dest or "").strip()
    if "长白山" in d:
        return "\n".join(
            [
                "你问的「特色项目/玩法」，长白山常见的高价值体验清单：",
                "1）滑雪：万达度假区滑雪场（可租装备/请教练）。",
                "2）温泉：度假区温泉或温泉酒店（更适合放松）。",
                "3）天池：经典景区，受天气影响大，建议预留弹性时间。",
                "4）冬季限定：雾凇、雪景拍照、冰雪项目（看气象条件与开放情况）。",
                "5）美食打卡：铁锅炖、山野菜、冷水鱼等地方特色。",
            ]
        )
    return "\n".join(
        [
            f"{d} 的玩法建议清单（先给 5 条可执行项）：",
            "1）核心地标：代表性景区/地标打卡（优先安排 1–2 个）。",
            "2）城市美食：本地必吃 + 夜市/小吃街。",
            "3）特色体验：季节限定/主题乐园/民俗体验等。",
            "4）周边一日游：自然/古镇/湖海山水任选其一。",
            "5）轻松休闲：咖啡馆/公园/夜景/温泉等。",
        ]
    )


def _build_followup_questions(has_dest: bool) -> str:
    if has_dest:
        return "\n".join(
            [
                "再确认两个点，我就能把安排做得更贴合：",
                "A. 你更偏户外运动还是休闲度假？",
                "B. 预算大概什么档位（经济/舒适/高端）？",
            ]
        )
    return "\n".join(
        [
            "为了给你更准确的推荐，先确认 3 点：",
            "1）目的地是哪里？",
            "2）预计几天？",
            "3）同行人数与预算档位？",
        ]
    )


# ---------- LLM 调用及解析 ----------
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
请只输出 JSON（无 Markdown，无自然语言前后缀），格式：
{{
  "summary": "一句话概括",
  "spots": [
    {{"name": "景点1", "category": "分类/标签", "brief_desc": "2-3 句简介"}},
    {{"name": "景点2", "category": "分类/标签", "brief_desc": "2-3 句简介"}}
  ]
}}
无法生成时输出：{{"summary": "无法生成", "spots": []}}
"""


def _retry_json_prompt(user_text: str) -> str:
    return f"""{user_text}
再次提醒：只输出 JSON（无 Markdown），格式：
{{"summary":"...","spots":[{{"name":"...","category":"...","brief_desc":"..."}}]}}
无法生成时输出：{{"summary":"无法生成","spots":[]}}
"""


def _clean_json_text(txt: str) -> str:
    clean = (txt or "").strip()
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


# ---------- 知识库转换 ----------
def _kb_to_resources(kb_items):
    tickets = []
    for it in kb_items:
        photos = it.get("photos") or []
        tickets.append(
            {
                "name": it.get("title") or it.get("name", ""),
                "category": "攻略精选",
                "desc": it.get("summary", ""),
                "photo": photos[0] if photos else None,
                "photos": photos,
                "address": "",
                "opentime": "",
                "url": it.get("url"),
                "source": it.get("platform") or "kb",
                "type": "景点",
            }
        )
    return {"tickets": tickets} if tickets else None


# ---------- 主流程 ----------
@router.post("/trip_chat", response_model=TripChatResponse)
def trip_chat(req: TripChatRequest, response: Response, request: Request) -> TripChatResponse:
    """
    中文 TripChat 主流程：
    - 结合 origin / default_origin 补充出发地提示；
    - 命中知识库时生成图文 resources（但不阻断后续行程规划）；
    - 调 LLM 生成景点 JSON（主要用于资源展示/闲聊回覆）；
    - 尝试构造 TripPlanRequest 槽位并调用 trip_plan 生成行程；
    - reply 中适当加入“草稿说明”，trip_plan 始终通过字段返回给前端。
    """
    trace_id = getattr(request.state, "trace_id", None) or uuid.uuid4().hex
    response.headers["X-Trace-Id"] = trace_id
    date_hint_needed = False

    # ----- 出发地提示 -----
    origin_hint = ""
    origin_value = req.origin
    if not origin_value and req.current_slots and getattr(req.current_slots, "origin", None):
        origin_value = req.current_slots.origin

    system_content = SYSTEM_PROMPT

    if origin_value:
        origin_hint = (
            "系统补充背景信息：用户已经有明确的出发城市，"
            f"当前假定出发地为“{origin_value}”。"
            "在规划行程时，请优先以该城市作为出发地。"
        )
    elif req.default_origin:
        origin_hint = (
            "系统补充背景信息：根据用户最近一次定位推断，"
            f"用户大概率位于“{req.default_origin}”。"
            "如果用户在对话中没有特别说明出发城市，可暂时以此作为出发地；"
            "一旦用户提供了新的出发城市，请以用户的最新说明为准。"
        )

    if origin_hint:
        system_content = SYSTEM_PROMPT + "\n\n" + origin_hint

    logger.info(
        "TripChat origin=%s default_origin=%s user_lat=%s user_lng=%s",
        origin_value,
        req.default_origin,
        req.user_lat,
        req.user_lng,
    )

    # ----- 1) 知识库：不再早退，只先记录 resources -----
    kb_res = kb.kb_search(kb.KbSearch(query=req.input_text, destination=""))
    kb_items = kb_res.get("items") if kb_res else []
    resources_from_kb = _kb_to_resources(kb_items) if kb_items else None

    # ----- 2) 调 LLM（强约束 JSON，失败重试一次），保留原始回复 -----
    reply_text, parsed_json, decision_used = _call_llm_for_json(req, system_content)
    reply_text = reply_text or ""
    _ = parsed_json  # 保留变量，避免未来扩展时误删

    # 只把“本轮用户消息”先加进 history，用于统计 user_turns
    history_with_new_user = list(req.history) + [ChatMessage(role="user", content=req.input_text)]
    user_turns = _count_user_turns(history_with_new_user)
    _ = user_turns  # 保留变量以避免未来逻辑改动时被误删

    # ----- 3) 解析 slots_json，或用 current_slots -----
    slots_from_llm: Optional[TripPlanRequest] = None
    try:
        if getattr(decision_used, "slots_json", None):
            raw_slots = json.loads(decision_used.slots_json) or {}
            # 补充用户坐标
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

    # ----- 4) 猜槽位（目的地/天数/出发地/偏好），并与 slots 融合 -----
    guess_slots = TripPlanRequest()
    text_merge = req.input_text + "\n" + "\n".join([m.content for m in req.history if m.role == "user"])

    guess_slots.destination = _guess_destination(text_merge)
    days = _guess_days(text_merge)
    if days:
        # 目前仅用“天数存在”作为信号，不强行构造 date_range
        guess_slots.date_range = []
    guess_slots.origin = _guess_origin(text_merge)
    guess_slots.preferences = _guess_preferences(text_merge)

    # 如果完全没有 slots，但猜到目的地，则使用 guess_slots
    if slots is None and guess_slots.destination:
        slots = guess_slots

    # 用猜测结果补齐 slots 中缺失字段（不覆盖已有值）
    if slots is not None:
        if not getattr(slots, "destination", None) and getattr(guess_slots, "destination", None):
            slots.destination = guess_slots.destination
        if not getattr(slots, "origin", None) and getattr(guess_slots, "origin", None):
            slots.origin = guess_slots.origin
        if not getattr(slots, "preferences", None) and getattr(guess_slots, "preferences", None):
            slots.preferences = guess_slots.preferences

        # 若仍无 origin，尝试用显式 origin 或 default_origin 补上
        if not getattr(slots, "origin", None):
            if origin_value:
                slots.origin = origin_value
            elif req.default_origin:
                slots.origin = req.default_origin

        # ----- 4.1) 确定性日期解析：优先在缺字段判定之前写回 slots -----
        if not _valid_date_range(getattr(slots, "date_range", None)):
            extracted = extract_cn_date_range(text_merge)
            if extracted:
                slots.date_range = [extracted["start_date"], extracted["end_date"]]
                logger.info("TripChat date_range extracted trace_id=%s date_range=%s", trace_id, slots.date_range)
            else:
                inferred = infer_cn_date_range_from_text(text_merge)
                if inferred:
                    slots.date_range = [inferred["start_date"], inferred["end_date"]]
                    date_hint_needed = True
                    logger.info("TripChat date_range inferred trace_id=%s date_range=%s", trace_id, slots.date_range)

        # 只要是“节假日/周末”等表达且没有明确日期范围，就加轻提示（不阻断生成）
        if not extract_cn_date_range(text_merge) and re.search(
            r"(元旦|节假日|周末|本周末|下周末|五一|劳动节|国庆|春节)",
            text_merge,
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

    # ----- 5) 调 TripPlan：只要有 slots 就尝试调用，让 TripPlan 自己判断信息是否充分 -----
    trip_plan_result: Optional[TripPlanResponse] = None
    missing: List[str] = []

    if slots is not None:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        try:
            logger.info("TripChat: calling trip_plan with slots=%s", safe_slots.dict())
            trip_plan_result = trip_plan(safe_slots, request)

            if trip_plan_result is not None and date_hint_needed:
                msg = "未确认出行日期，营业时间/预约请以实际日期核对"
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

    # ----- 6) 根据缺失字段增加“草稿说明”前缀（最多解释一次） -----
    if trip_plan_result is not None and missing:
        # 文案互斥：若仍缺“出行日期”，不要在同一回复里出现“已收到日期”等话术
        if "出行日期" in missing:
            reply_text = re.sub(r"^.*?(已收到|收到).*(出行)?日期.*?$", "", reply_text, flags=re.MULTILINE).strip()

        draft_phrase = "我先按目前信息出了一个草稿"
        already_notified = any(
            (m.role == "assistant" and draft_phrase in (m.content or ""))
            for m in req.history
        )

        missing_text = "、".join(missing)
        if not already_notified:
            prefix = f"{draft_phrase}（缺少：{missing_text}），你补充后我再优化。"
        else:
            prefix = f"现在还缺：{missing_text}，方便补充一下吗？"

        reply_text = prefix + "\n" + (reply_text or "")

    # ----- 7) 最终 history：把本轮 assistant 回复加进去 -----
    new_history = history_with_new_user + [ChatMessage(role="assistant", content=reply_text)]

    # QUICK_REPLIES: 若缺少目的地，返回候选给前端渲染 chips
    quick_replies: List[str] = []
    try:
        if slots is None or not getattr(slots, "destination", None):
            quick_replies = _dest_quick_replies(text_merge)
    except Exception:
        quick_replies = []

    resources_out: Dict[str, Any] = resources_from_kb or {}
    if quick_replies:
        if not isinstance(resources_out, dict):
            resources_out = {}
        resources_out = dict(resources_out)
        resources_out["quick_replies"] = quick_replies

    # ----- 8) 组装响应：resources_from_kb 和 trip_plan 一起返回 -----
    return TripChatResponse(
        reply=reply_text or "这边现在有点忙，你可以稍后再试试。",
        history=new_history,
        slots=slots,
        trip_plan=trip_plan_result,
        resources=resources_out,
    )
