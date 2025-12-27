# -*- coding: utf-8 -*-
import json
import os
import re
import urllib.parse
import urllib.request
import logging
import time
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
from app.schemas.trip_dialog import (
    DialogState,
    TripProfile,
    PendingQuestion,
    SlotCompleteness,
    NextAction,
)
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
    mode: str = "EXPLORE"
    dialog_state: DialogState = DialogState.DISCOVERY
    slot_completeness: SlotCompleteness = Field(default_factory=SlotCompleteness)
    pending_questions: List[PendingQuestion] = Field(default_factory=list)
    next_action: NextAction = Field(default_factory=NextAction)
    trip_profile: Optional[TripProfile] = None


class RouteOption(BaseModel):
    title: str
    days: Optional[int] = None
    highlights: List[str] = Field(default_factory=list)
    season_hint: Optional[str] = None


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


def _parse_user_slots(text: str) -> Dict[str, Any]:
    t = text or ""
    result: Dict[str, Any] = {}

    if re.search(r"(舒适|舒服点|品质点|别太省|不要太省)", t):
        result["budget_level"] = "舒适"

    m = re.search(r"(\d+)\s*(?:人|位|个)", t)
    if m:
        try:
            result["traveler_count"] = max(1, min(int(m.group(1)), 20))
        except Exception:
            pass
    else:
        cn_map = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        m_cn = re.search(r"([一二两三四五六七八九十])\s*(?:人|位|个)", t)
        if m_cn:
            result["traveler_count"] = cn_map.get(m_cn.group(1))

    tags = []
    tag_keywords = ["购物", "美食", "主题公园", "城市观光", "文化", "自然", "亲子"]
    for kw in tag_keywords:
        if kw in t and kw not in tags:
            tags.append(kw)
    if tags:
        result["interest_tags"] = tags

    return result


def _merge_slots(prev_slots: Optional[TripPlanRequest], new_slots: Dict[str, Any]) -> TripPlanRequest:
    slots = prev_slots or TripPlanRequest()
    if not new_slots:
        return slots

    meta = getattr(slots, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    slot_sources = meta.get("slot_sources")
    if not isinstance(slot_sources, dict):
        slot_sources = {}

    def _mark_source(key: str, value: Any):
        slot_sources[key] = {
            "value": value,
            "source": "text",
            "ts": int(time.time()),
        }

    budget_level = new_slots.get("budget_level")
    if budget_level:
        old_value = getattr(slots, "budget_level", None)
        if old_value != budget_level:
            _mark_source("budget_level", budget_level)
        slots.budget_level = budget_level

    traveler_count = new_slots.get("traveler_count")
    if traveler_count:
        old_value = getattr(slots, "people_count", None)
        if old_value != traveler_count:
            _mark_source("traveler_count", traveler_count)
        slots.people_count = traveler_count

    interest_tags = new_slots.get("interest_tags") or []
    if interest_tags:
        old_value = list(getattr(slots, "preferences", None) or [])
        merged = []
        for item in old_value + interest_tags:
            if item and item not in merged:
                merged.append(item)
        if merged != old_value:
            _mark_source("interest_tags", merged)
        slots.preferences = merged

    meta["slot_sources"] = slot_sources
    setattr(slots, "meta", meta)
    return slots


def _estimate_required_done(slots: Optional[TripPlanRequest]) -> int:
    if slots is None:
        return 0
    done = 0
    if getattr(slots, "origin", None):
        done += 1
    if getattr(slots, "destination", None):
        done += 1
    if _valid_date_range(getattr(slots, "date_range", None)):
        done += 1
    if getattr(slots, "people_count", None) is not None or getattr(slots, "adults", None) is not None:
        done += 1
    return done


def _infer_profile_from_slots(slots: Optional[TripPlanRequest]) -> Optional[TripPlanRequest]:
    if slots is None:
        return None
    tags = list(getattr(slots, "preferences", None) or []) or list(getattr(slots, "interests", None) or [])
    tag_set = set(tags)
    relax_tags = {"购物", "美食", "主题公园", "亲子"}

    if tag_set.intersection(relax_tags):
        if not getattr(slots, "style", None):
            setattr(slots, "style", "休闲度假")
        if not getattr(slots, "pace_level", None):
            setattr(slots, "pace_level", "均衡偏轻松")

    if getattr(slots, "budget_level", None) == "舒适" and not getattr(slots, "pace_level", None):
        setattr(slots, "pace_level", "均衡")

    return slots


def _set_origin_with_source(slots: TripPlanRequest, origin: str, source: str) -> None:
    if not origin:
        return
    slots.origin = origin
    meta = getattr(slots, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    meta["origin_source"] = source
    setattr(slots, "meta", meta)


def _get_asked_slots(slots: Optional[TripPlanRequest]) -> List[Dict[str, Any]]:
    if slots is None:
        return []
    meta = getattr(slots, "meta", None)
    if not isinstance(meta, dict):
        return []
    asked = meta.get("asked_slots")
    return asked if isinstance(asked, list) else []


def _record_asked_slot(slots: Optional[TripPlanRequest], slot_key: str, qid: str, turn: int) -> None:
    if slots is None:
        return
    meta = getattr(slots, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    asked = meta.get("asked_slots")
    if not isinstance(asked, list):
        asked = []
    for it in asked:
        if it.get("slot_key") == slot_key and turn - int(it.get("turn", 0)) <= 10:
            logger.warning("repeat_question_detected slot_key=%s turn=%s", slot_key, turn)
            break
    asked.append({"slot_key": slot_key, "qid": qid, "turn": turn})
    meta["asked_slots"] = asked[-20:]
    setattr(slots, "meta", meta)


def _slot_filled(slot_key: str, slots: Optional[TripPlanRequest], days_guess: Optional[int]) -> bool:
    if slots is None:
        return False
    if slot_key == "traveler_count":
        return getattr(slots, "people_count", None) is not None or getattr(slots, "adults", None) is not None
    if slot_key == "days_or_date_range":
        return _valid_date_range(getattr(slots, "date_range", None)) or bool(days_guess)
    if slot_key == "destination":
        return bool(getattr(slots, "destination", None))
    if slot_key == "origin":
        return bool(getattr(slots, "origin", None))
    if slot_key == "budget_level":
        return bool(getattr(slots, "budget_level", None))
    if slot_key == "style":
        return bool(getattr(slots, "style", None))
    return False


def _should_ask(
    slot_key: str,
    slots: Optional[TripPlanRequest],
    history: List[ChatMessage],
    current_turn: int,
    days_guess: Optional[int],
    max_turns: int = 10,
) -> bool:
    if _slot_filled(slot_key, slots, days_guess):
        return False
    for it in _get_asked_slots(slots):
        if it.get("slot_key") == slot_key and current_turn - int(it.get("turn", 0)) <= max_turns:
            return False

    if history:
        recent_assistant = [m for m in history if m.role == "assistant"][-max_turns:]
        prompt_map = {
            "traveler_count": "几位出行",
            "days_or_date_range": "计划玩几天",
            "destination": "目的地",
            "origin": "出发",
            "budget_level": "预算",
            "style": "休闲度假",
        }
        marker = prompt_map.get(slot_key)
        if marker and any(marker in (m.content or "") for m in recent_assistant):
            return False

    return True


def _question_id_for_slot(slot_key: str) -> str:
    return f"q_{slot_key}_v1"


def _detect_mode(text: str, has_trip_plan: bool) -> str:
    t = text or ""
    if has_trip_plan and re.search(r"(更舒适|舒服点|别太赶|加|删|换酒店)", t):
        return "REFINE"
    if re.search(r"(介绍|推荐|特色|经典路线|怎么玩|有什么)", t):
        return "EXPLORE"
    if re.search(r"(行程|规划|安排|每天|路线|攻略)", t):
        return "PLAN"
    return "EXPLORE"


def _detect_refine_intent(text: str) -> bool:
    t = text or ""
    return bool(
        re.search(
            r"(更舒适|舒服点|别太赶|太赶|慢一点|节奏慢|加购物|加美食|加主题公园|删景点|删掉|减少景点|调整|改一下|优化|修改|微调)",
            t,
        )
    )


def _evaluate_missing_required(slots: Optional[TripPlanRequest], days_guess: Optional[int], has_user_origin: bool) -> List[str]:
    missing: List[str] = []
    traveler_count = getattr(slots, "people_count", None) if slots is not None else None
    adults = getattr(slots, "adults", None) if slots is not None else None
    has_traveler = traveler_count is not None or adults is not None
    has_dates = False
    if slots is not None and _valid_date_range(getattr(slots, "date_range", None)):
        has_dates = True
    if days_guess:
        has_dates = True

    destination_value = getattr(slots, "destination", None) if slots is not None else None

    if not has_traveler:
        missing.append("traveler_count")
    if not has_dates:
        missing.append("days_or_date_range")
    if not destination_value:
        missing.append("destination")
    if not has_user_origin:
        missing.append("origin")
    return missing


def _set_meta_value(slots: Optional[TripPlanRequest], key: str, value: Any) -> None:
    if slots is None:
        return
    meta = getattr(slots, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    meta[key] = value
    setattr(slots, "meta", meta)


def _decide_next_action(missing_required: List[str]) -> NextAction:
    if not missing_required:
        return NextAction(type="CALL_TRIP_PLAN", reason="required_complete")
    return NextAction(type="ASK", reason=f"missing_{missing_required[0]}")


def _build_pending_question(missing_key: str) -> PendingQuestion:
    question_id = _question_id_for_slot(missing_key)
    prompts = {
        "traveler_count": "几位出行？",
        "days_or_date_range": "计划玩几天，或具体日期是哪几天？",
        "destination": "这次想去哪个目的地？",
        "origin": "从哪个城市出发？",
    }
    return PendingQuestion(
        id=question_id,
        question_id=question_id,
        slot_key=missing_key,
        prompt=prompts.get(missing_key, "请补充关键信息。"),
        status="open",
        asked_at=str(int(time.time())),
    )


def _build_trip_profile(slots: Optional[TripPlanRequest]) -> TripProfile:
    if slots is None:
        return TripProfile()
    traveler_count = None
    if getattr(slots, "people_count", None) is not None:
        traveler_count = int(getattr(slots, "people_count"))
    else:
        adults = getattr(slots, "adults", None) or 0
        children = getattr(slots, "children", None) or 0
        if adults or children:
            traveler_count = int(adults + children)
    return TripProfile(
        origin_city=getattr(slots, "origin", None),
        destination=getattr(slots, "destination", None),
        date_range=getattr(slots, "date_range", None) or None,
        traveler_count=traveler_count,
        budget_level=getattr(slots, "budget_level", None),
        interest_tags=list(getattr(slots, "preferences", None) or []),
        pace_level=getattr(slots, "pace_level", None) or getattr(slots, "pace", None),
        style=getattr(slots, "style", None),
        slot_sources=(getattr(slots, "meta", None) or {}).get("slot_sources"),
    )


def _summarize_known(
    slots: Optional[TripPlanRequest],
    days_guess: Optional[int],
    has_user_origin: bool,
) -> str:
    parts: List[str] = []
    if slots:
        if getattr(slots, "destination", None):
            parts.append(str(getattr(slots, "destination")))
        if has_user_origin and getattr(slots, "origin", None):
            parts.append(f"{getattr(slots, 'origin')}出发")
        if _valid_date_range(getattr(slots, "date_range", None)):
            parts.append("已定日期")
        elif days_guess:
            parts.append(f"{days_guess}天")
        if getattr(slots, "people_count", None) is not None:
            parts.append(f"{getattr(slots, 'people_count')}人")
        elif getattr(slots, "adults", None) is not None:
            parts.append("已有人数")
        if getattr(slots, "pace_level", None):
            parts.append("轻松节奏" if "轻松" in str(getattr(slots, "pace_level")) else "均衡节奏")
        elif getattr(slots, "style", None) == "休闲度假":
            parts.append("休闲度假")

    summary = "已知：" + ("，".join(parts) if parts else "先了解需求")
    return summary[:25]


def _options_for_missing(missing_key: Optional[str]) -> str:
    options_map = {
        "traveler_count": "(1-2人/3-4人/5人以上/不确定)",
        "days_or_date_range": "(2天/3天/4-5天/不确定)",
        "destination": "(热门城市/周边/自然/不确定)",
        "origin": "(本地出发/周边城市/不确定)",
    }
    return options_map.get(missing_key or "", "(可补充任意偏好)")


def _render_reply(
    slots: Optional[TripPlanRequest],
    next_action: NextAction,
    pending_questions: List[PendingQuestion],
    days_guess: Optional[int],
    has_user_origin: bool,
    origin_conflict_note: Optional[str],
) -> str:
    summary = _summarize_known(slots, days_guess, has_user_origin)
    missing_key = pending_questions[0].slot_key if pending_questions else None
    options = _options_for_missing(missing_key)
    prefix_lines = [summary]
    if origin_conflict_note:
        prefix_lines.append(origin_conflict_note)

    if next_action.type == "ASK" and pending_questions:
        question = pending_questions[0].prompt
        return "\n".join(prefix_lines + [question, options])
    if next_action.type == "REFINE_PLAN":
        return "\n".join(prefix_lines + ["我先按你的调整优化行程。", options])
    if next_action.type == "CALL_TRIP_PLAN":
        return "\n".join(prefix_lines + ["我先出一版行程。", options])
    return "\n".join(prefix_lines + ["如果有其他偏好，也可以继续补充。", options])


def _build_route_menu(destination: str, days: Optional[int], season_hint: Optional[str]) -> List[RouteOption]:
    if destination != "东北":
        return []
    season = season_hint or ""
    winter = "冬" in season or "雪" in season or "冰" in season
    day_count = days or 5
    options = [
        RouteOption(
            title="A 哈尔滨冰雪线",
            days=day_count,
            highlights=["中央大街", "索菲亚教堂", "冰雪大世界", "松花江"],
            season_hint="冬季" if winter else None,
        ),
        RouteOption(
            title="B 长白山温泉线",
            days=day_count,
            highlights=["长白山天池", "北坡/西坡", "温泉酒店", "雪景体验"],
            season_hint="冬季" if winter else None,
        ),
        RouteOption(
            title="C 哈尔滨+长白山混合线",
            days=day_count,
            highlights=["哈尔滨城市夜景", "长白山雪景", "温泉放松", "美食打卡"],
            season_hint="冬季" if winter else None,
        ),
    ]
    return options


def _render_route_menu_text(options: List[RouteOption]) -> str:
    if not options:
        return ""
    lines = ["先给你 3 条经典路线："]
    for opt in options:
        highlights = " / ".join(opt.highlights[:4])
        days_txt = f"{opt.days}日" if opt.days else ""
        lines.append(f"{opt.title}（{days_txt}）")
        lines.append(f"亮点：{highlights}")
    return "\n".join(lines)


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
        return ["哈尔滨", "长白山", "大连", "沈阳", "我还不确定"]
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


def _classify_intent(text: str) -> str:
    t = text or ""
    is_plan = bool(re.search(r"(规划|行程|路线|安排|按天|几日游|生成行程|出行计划|行程表)", t))
    is_explore = bool(re.search(r"(介绍|推荐|特色|怎么玩|必去|有哪些城市|景点|概览|对比)", t))
    if is_explore:
        if re.search(r"(生成|按天|安排|路线怎么走)", t):
            return "plan"
        return "explore"
    if is_plan:
        return "plan"
    return "chat"

def _is_region_query(text: str) -> bool:
    t = text or ""
    has_region = bool(re.search(r"(东北|华北|华东|华中|华南|西北|西南|云南|新疆|海南)", t))
    has_query = bool(re.search(r"(介绍|推荐|怎么玩|必去|有哪些|景点|概览|对比|路线|行程|几日游)", t))
    return has_region and has_query

def _pick_region(text: str) -> Optional[str]:
    t = text or ""
    if "东北" in t:
        return "东北"
    if "华北" in t:
        return "华北"
    if "华东" in t:
        return "华东"
    if "华中" in t:
        return "华中"
    if "华南" in t:
        return "华南"
    if "西北" in t:
        return "西北"
    if "西南" in t:
        return "西南"
    if "云南" in t:
        return "云南"
    if "新疆" in t:
        return "新疆"
    if "海南" in t:
        return "海南"
    return None

def _build_region_overview(region: str) -> str:
    if not region:
        return ""
    if region == "东北":
        return "\n".join(
            [
                "东北适合做冰雪、山林、滨海与城市历史的组合旅行，路程跨度大，通常需要在城市之间取舍。",
                "方向参考：哈尔滨冰雪｜长白山雪景｜大连海滨｜沈阳历史｜延边美食。",
                "你更偏好冰雪/自然，还是城市休闲？",
            ]
        )
    return "\n".join(
        [
            f"{region}适合做多城市组合旅行，建议先确定偏好与出行时长再细化。",
            "方向参考：城市文化｜自然山水｜海滨度假｜特色美食｜亲子休闲。",
            "你更偏好哪一类？",
        ]
    )

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


def _pick_followup_question(
    slots: Optional[TripPlanRequest],
    history: List[ChatMessage],
    current_turn: int,
    days_guess: Optional[int],
) -> Optional[Dict[str, str]]:
    candidates = [
        ("style", "你更偏户外运动还是休闲度假？"),
        ("budget_level", "预算大概什么档位（经济/舒适/高端）？"),
    ]
    for key, prompt in candidates:
        if _should_ask(key, slots, history, current_turn, days_guess):
            return {"slot_key": key, "prompt": prompt}
    return None


def _build_followup_questions(has_dest: bool) -> str:
    if has_dest:
        return "如果有其他偏好，也可以继续补充。"
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
    text_origin = _guess_origin(req.input_text)

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
        if text_origin:
            _set_origin_with_source(slots, text_origin, "text")
        elif not getattr(slots, "origin", None) and getattr(guess_slots, "origin", None):
            _set_origin_with_source(slots, guess_slots.origin, "text")
        if not getattr(slots, "preferences", None) and getattr(guess_slots, "preferences", None):
            slots.preferences = guess_slots.preferences

        # 若仍无 origin，尝试用显式 origin 或 default_origin 补上
        if not getattr(slots, "origin", None):
            if origin_value:
                _set_origin_with_source(slots, origin_value, "location")
            elif req.default_origin:
                _set_origin_with_source(slots, req.default_origin, "location")

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

    # ----- 4.15) 解析用户显式槽位并合并 -----
    slots = _merge_slots(slots, _parse_user_slots(req.input_text))
    slots = _infer_profile_from_slots(slots)

    # ----- 4.2) empty/generic reply fallback -----
    destination_value = slots.destination if slots is not None else None
    intent = _classify_intent(req.input_text)
    region = _pick_region(text_merge) if _is_region_query(text_merge) else None
    should_plan = intent == "plan" and not (region and not destination_value)
    origin_conflict_note = ""
    if slots is not None and req.origin and getattr(slots, "origin", None) and req.origin != slots.origin:
        origin_conflict_note = (
            f"我检测到你当前定位在{req.origin}，但你说从{slots.origin}出发。"
            f"行程以{slots.origin}出发为准对吗？"
        )

    generic_explore_reply = "\n".join(
        [
            "先给你一个探索方向的概览：",
            "1）核心城市打卡：地标 + 城市气质体验。",
            "2）自然山水线：山林/湖海/国家公园。",
            "3）美食与夜景线：本地必吃 + 夜市/夜景。",
            "你更偏好哪一类，或计划玩几天？",
        ]
    )

    def _build_plan_followup_text() -> str:
        skeleton_lines = [
            "我可以帮你做行程规划，先补两点关键信息：",
            "A. 目的地与出发地（若已确定其一可跳过）。",
            "B. 出行日期与人数。",
        ]
        questions = []
        priorities = [
            ("目的地", not destination_value),
            ("出行日期", slots is None or not _valid_date_range(getattr(slots, "date_range", None))),
            (
                "同行人数",
                slots is None
                or (getattr(slots, "adults", None) is None and getattr(slots, "people_count", None) is None),
            ),
            ("出发地", slots is None or not getattr(slots, "origin", None)),
        ]
        for label, needed in priorities:
            if needed:
                questions.append(label)
            if len(questions) >= 1:
                break
        if questions:
            skeleton_lines.append(f"先确认：{questions[0]}。")
        else:
            skeleton_lines.append("信息齐了，我可以开始生成行程。")
        return "\n".join(skeleton_lines)

    if _needs_recommendations(req.input_text):
        if destination_value:
            if _is_generic_ack(reply_text) or _count_actionable_items(reply_text) < 5:
                followup = _pick_followup_question(slots, history_with_new_user, user_turns, _guess_days(text_merge))
                if followup:
                    qid = _question_id_for_slot(followup["slot_key"])
                    _record_asked_slot(slots, followup["slot_key"], qid, user_turns)
                    reply_text = _build_reco_list(destination_value) + "\n\n" + followup["prompt"]
                else:
                    reply_text = _build_reco_list(destination_value) + "\n\n" + _build_followup_questions(True)
        else:
            if intent == "explore" or (intent == "plan" and region):
                if region == "东北":
                    route_menu = _build_route_menu("东北", _guess_days(req.input_text), req.input_text)
                    menu_text = _render_route_menu_text(route_menu)
                    if menu_text:
                        reply_text = menu_text + "\n\n" + "你更想选哪条路线？"
                    else:
                        reply_text = _build_region_overview(region)
                else:
                    reply_text = _build_region_overview(region) if region else generic_explore_reply
            elif intent == "plan":
                reply_text = _build_plan_followup_text()
            else:
                reply_text = generic_explore_reply
    elif _is_generic_ack(reply_text):
        if (intent == "explore" or (intent == "plan" and region)) and not destination_value:
            if region == "东北":
                route_menu = _build_route_menu("东北", _guess_days(req.input_text), req.input_text)
                menu_text = _render_route_menu_text(route_menu)
                reply_text = menu_text + "\n\n" + "你更想选哪条路线？" if menu_text else _build_region_overview(region)
            else:
                reply_text = _build_region_overview(region) if region else generic_explore_reply
        elif intent == "plan":
            reply_text = _build_plan_followup_text()
        elif not destination_value:
            reply_text = generic_explore_reply
        else:
            followup = _pick_followup_question(slots, history_with_new_user, user_turns, _guess_days(text_merge))
            if followup:
                qid = _question_id_for_slot(followup["slot_key"])
                _record_asked_slot(slots, followup["slot_key"], qid, user_turns)
                reply_text = followup["prompt"]
            else:
                reply_text = _build_followup_questions(True)

    # ----- 5) TripPlan 触发条件判断 -----
    trip_plan_result: Optional[TripPlanResponse] = None
    missing: List[str] = []

    if should_plan and slots is not None:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        if not _valid_date_range(getattr(safe_slots, "date_range", None)) and "出行日期" not in missing:
            missing.append("出行日期")
        if missing:
            reply_text = _build_plan_followup_text()
            trip_plan_result = None
        else:
            try:
                logger.info("TripChat: calling trip_plan with slots=%s", safe_slots.dict())
                trip_plan_result = trip_plan(safe_slots, request)

                if trip_plan_result is not None and date_hint_needed:
                    msg = "你填写的是节假日/周末范围，如需更精确可补充具体日期"
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
    elif should_plan:
        missing = ["目的地", "出行日期"]
        reply_text = _build_plan_followup_text()

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

    resources_out: Dict[str, Any] = resources_from_kb if isinstance(resources_from_kb, dict) else {}
    if quick_replies:
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in quick_replies:
            if item not in existing:
                existing.append(item)
    if region == "东北" and intent == "explore":
        menu_replies = ["选A", "选B", "选C", "我不确定"]
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in menu_replies:
            if item not in existing:
                existing.append(item)
    if origin_conflict_note and getattr(slots, "origin", None) and req.origin:
        conflict_replies = [f"以{slots.origin}出发", f"改为{req.origin}出发"]
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in conflict_replies:
            if item not in existing:
                existing.append(item)

    # ----- 8) 组装响应：resources_from_kb 和 trip_plan 一起返回 -----
    slot_completeness = SlotCompleteness(
        required_done=_estimate_required_done(slots),
        required_total=4,
    )
    days_guess = _guess_days(text_merge)
    has_user_origin = bool(req.origin) or (
        bool(getattr(slots, "origin", None)) and getattr(slots, "origin", None) != req.default_origin
    )
    missing_required = _evaluate_missing_required(slots, days_guess, has_user_origin)
    has_trip_plan = trip_plan_result is not None
    refine_intent = _detect_refine_intent(req.input_text)
    if refine_intent and has_trip_plan:
        logger.info("refine_triggered trace_id=%s turn=%s", trace_id, user_turns)
    if refine_intent and has_trip_plan:
        dialog_state = DialogState.REFINEMENT
    elif has_trip_plan:
        dialog_state = DialogState.PLAN_PRESENTED
    elif missing_required:
        dialog_state = DialogState.DISCOVERY
    else:
        dialog_state = DialogState.PLAN_DRAFTING

    next_action = NextAction(type="NONE", reason="not_plan")
    pending_questions: List[PendingQuestion] = []
    if refine_intent and has_trip_plan:
        next_action = NextAction(type="REFINE_PLAN", reason="user_refine")
    elif should_plan:
        ask_key = None
        for key in missing_required:
            if _should_ask(key, slots, history_with_new_user, user_turns, days_guess):
                ask_key = key
                break
        if ask_key is None:
            if not missing_required:
                next_action = NextAction(type="CALL_TRIP_PLAN", reason="required_complete")
            else:
                next_action = NextAction(type="NONE", reason="asked_recently")
                logger.warning(
                    "repeat_question_detected trace_id=%s turn=%s missing_required=%s",
                    trace_id,
                    user_turns,
                    ",".join(missing_required),
                )
        else:
            next_action = NextAction(type="ASK", reason=f"missing_{ask_key}")
            pending_questions = [_build_pending_question(ask_key)]
            _record_asked_slot(slots, ask_key, pending_questions[0].question_id, user_turns)
    trip_profile = _build_trip_profile(slots)
    if has_trip_plan:
        meta = getattr(slots, "meta", None) if slots is not None else None
        if not isinstance(meta, dict) or meta.get("first_plan_turn") is None:
            _set_meta_value(slots, "first_plan_turn", user_turns)
            logger.info("turns_to_first_plan trace_id=%s turn=%s", trace_id, user_turns)
    if should_plan and next_action.type in {"ASK", "CALL_TRIP_PLAN", "REFINE_PLAN"}:
        reply_text = _render_reply(
            slots,
            next_action,
            pending_questions,
            days_guess,
            has_user_origin,
            origin_conflict_note,
        )
    elif origin_conflict_note:
        reply_text = origin_conflict_note + ("\n" + reply_text if reply_text else "")
    mode = _detect_mode(req.input_text, has_trip_plan)
    return TripChatResponse(
        reply=reply_text or "这边现在有点忙，你可以稍后再试试。",
        history=new_history,
        slots=slots,
        trip_plan=trip_plan_result,
        resources=resources_out,
        mode=mode,
        dialog_state=dialog_state,
        slot_completeness=slot_completeness,
        pending_questions=pending_questions,
        next_action=next_action,
        trip_profile=trip_profile,
    )
