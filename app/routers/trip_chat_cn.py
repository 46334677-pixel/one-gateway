import json
import os
import re
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.llm_cn import (
    ChatMessage,
    TripChatLLMInput,
    TripChatLLMDecision,
    call_qwen_for_trip_chat,
)
from app.routers.trip import TripPlanRequest, TripPlanResponse, trip_plan
from app.routers import kb  # 新增：知识库

router = APIRouter(prefix="/cn/v1", tags=["trip_chat"])

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
        "opentime": (poi.get("opentime") or poi.get("open_time") or poi.get("business_time") or "") if poi else "",
        "url": f"https://www.amap.com/place/{poi.get('id')}" if poi and poi.get("id") else "",
        "source": "gaode",
        "type": "景点",
    }

# ---------- 请求/响应模型 ----------
class TripChatRequest(BaseModel):
    user_id: Optional[str] = None
    history: List[ChatMessage] = []
    input_text: str
    current_slots: Optional[TripPlanRequest] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None


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
    m = re.search(r"(?:去|到|去往|想去|想去到|去一趟)([\u4e00-\u9fa5A-Za-z]{2,10})", text)
    if m:
        return m.group(1)
    m2 = re.search(r"([\u4e00-\u9fa5A-Za-z]{2,10})\s*(?:\d+天|日)游", text)
    if m2:
        return m2.group(1)
    return None

def _guess_days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:天|日)", text)
    if m:
        return max(1, min(int(m.group(1)), 10))
    return None

def _guess_origin(text: str) -> Optional[str]:
    m = re.search(r"从([\u4e00-\u9fa5A-Za-z]{2,10})出发", text)
    return m.group(1) if m else None

def _guess_preferences(text: str) -> List[str]:
    prefs = []
    for kw in ["亲子", "美食", "自然", "海岛", "海", "酒店", "休闲", "博物馆", "夜景", "购物"]:
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

def _missing_fields(slots: TripPlanRequest) -> List[str]:
    missing = []
    if not slots.origin:
        missing.append("出发地")
    if not slots.destination:
        missing.append("目的地")
    if not slots.date_range:
        missing.append("出行日期")
    if slots.adults is None and slots.people_count is None:
        missing.append("成人人数")
    return missing

# ---------- LLM 调用及解析 ----------
def _build_llm_input(prompt: str, req: TripChatRequest) -> TripChatLLMInput:
    return TripChatLLMInput(
        user_id=req.user_id or "",
        history=req.history,
        new_user_message=prompt,
        location_lat=req.location_lat,
        location_lng=req.location_lng,
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
无法生成时输出: {{"summary": "无法生成", "spots": []}}
"""

def _retry_json_prompt(user_text: str) -> str:
    return f"""{user_text}
再次提醒：只输出 JSON（无 Markdown），格式:
{{"summary":"...","spots":[{{"name":"...","category":"...","brief_desc":"..."}}]}}
无法生成时输出 {{"summary":"无法生成","spots":[]}}
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

def _call_llm_for_json(req: TripChatRequest):
    decision = call_qwen_for_trip_chat(_build_llm_input(_force_json_prompt(req.input_text), req))
    reply_text = decision.reply or ""
    parsed = _try_parse_json(reply_text)
    if parsed is not None:
        return reply_text, parsed, decision
    decision2 = call_qwen_for_trip_chat(_build_llm_input(_retry_json_prompt(req.input_text), req))
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
def trip_chat(req: TripChatRequest) -> TripChatResponse:
    # 1) 知识库优先
    kb_res = kb.kb_search(kb.KbSearch(query=req.input_text, destination=""))
    kb_items = kb_res.get("items") if kb_res else []
    resources_from_kb = _kb_to_resources(kb_items) if kb_items else None

    # 2) 调 LLM（强约束 JSON，失败重试一次），保留原始回复
    reply_text, parsed_json, decision_used = _call_llm_for_json(req)

    new_history = list(req.history) + [
        ChatMessage(role="user", content=req.input_text),
        ChatMessage(role="assistant", content=reply_text),
    ]
    user_turns = _count_user_turns(new_history)

    # 3) 解析 slots_json，或用 current_slots
    slots_from_llm: Optional[TripPlanRequest] = None
    try:
        if getattr(decision_used, "slots_json", None):
            raw_slots = json.loads(decision_used.slots_json)
            if req.location_lat is not None:
                raw_slots.setdefault("user_lat", req.location_lat)
            if req.location_lng is not None:
                raw_slots.setdefault("user_lng", req.location_lng)
            slots_from_llm = TripPlanRequest.parse_obj(raw_slots)
    except Exception:
        slots_from_llm = None
    slots: Optional[TripPlanRequest] = slots_from_llm or req.current_slots

    # 4) 猜槽位（目的地/天数/出发地/偏好）
    guess_slots = TripPlanRequest()
    text_merge = req.input_text + "\n" + "\n".join([m.content for m in req.history if m.role == "user"])
    guess_slots.destination = _guess_destination(text_merge)
    days = _guess_days(text_merge)
    if days:
        guess_slots.date_range = []
    guess_slots.origin = _guess_origin(text_merge)
    guess_slots.preferences = _guess_preferences(text_merge)

    if slots is None and guess_slots.destination:
        slots = guess_slots

    # 5) 如果 KB 命中，直接返回 resources
    if resources_from_kb:
        return TripChatResponse(
            reply=reply_text,
            history=new_history,
            slots=slots,
            trip_plan=None,
            resources=resources_from_kb,
        )

    # 6) 决定是否调 TripPlan：有目的地，或用户已说满 3 轮
    should_call_plan = False
    if slots and slots.destination:
        should_call_plan = True
    elif user_turns >= 3 and guess_slots.destination:
        slots = guess_slots
        should_call_plan = True

    trip_plan_result: Optional[TripPlanResponse] = None
    if should_call_plan and slots:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        try:
            trip_plan_result = trip_plan(safe_slots)
            if missing:
                reply_text = f"我先按目前信息出了一版草稿（缺少：{'、'.join(missing)}），请补充后我再优化。\n" + reply_text
        except Exception:
            trip_plan_result = None

    return TripChatResponse(
        reply=reply_text or "这边现在有点忙，你可以稍后再试试。",
        history=new_history,
        slots=slots,
        trip_plan=trip_plan_result,
        resources=None,
    )
