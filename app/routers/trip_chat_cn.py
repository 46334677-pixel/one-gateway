# -*- coding: utf-8 -*-
import json
import os
import re
import urllib.parse
import urllib.request
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

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
    DialogDraft,
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


def _strip_admin_suffix(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"(市|区|县|省)$", "", text)


def _normalize_dest_key(destination: Optional[str], text_merge: str) -> Optional[str]:
    text = text_merge or ""
    if "东北" in text:
        return "东北"
    if any(k in text for k in ("港澳", "澳门", "香港")):
        return "港澳"
    if any(k in text for k in ("海南", "三亚", "海口", "万宁", "陵水", "文昌", "琼海")):
        return "海南"
    return destination or None


def _route_archetype(dest_key: Optional[str], text: str, region: Optional[str]) -> str:
    if dest_key == "东北":
        return "NORTHEAST_SNOW"
    if dest_key == "港澳":
        return "HK_MO"
    if dest_key == "海南":
        return "HAINAN_ISLAND"

    t = text or ""
    if re.search(r"(美食|夜宵|小吃|吃什么|必吃|探店)", t):
        return "FOOD_CITY"
    if dest_key in {"成都", "重庆", "长沙", "广州", "顺德", "潮汕", "武汉"}:
        return "FOOD_CITY"

    if re.search(r"(博物馆|历史|古城|人文|遗址|古都)", t):
        return "HISTORY_CITY"
    if dest_key in {"北京", "西安", "南京", "洛阳", "开封"}:
        return "HISTORY_CITY"

    if dest_key:
        return "CITY_BREAK"
    return "GENERIC"


def _render_actions_starter(
    archetype: str,
    dest_key: Optional[str],
    days: Optional[int],
    origin_hint: Optional[str],
    text: Optional[str] = None,
) -> Tuple[str, List[str]]:
    hook = "先给你一个更清晰的选择框架，方便快速定方向。"
    dest_text = dest_key or "目的地"
    days_text = f"{days}天" if days else "几天"
    origin_text = f"从{origin_hint}出发" if origin_hint else "出发地未定"
    confirm = f"先按{dest_text}{days_text}的思路给你三种经典走法（{origin_text}）。"

    options: List[str]
    advice: str
    questions: List[str]
    quick_replies: List[str]

    text_hint = text or ""
    if archetype == "GENERIC":
        options = [
            "经典打卡线｜适合：第一次来 | 亮点：地标/老街/夜景 | 节奏：适中 | 避坑：错峰出行",
            "轻松度假线｜适合：想放松 | 亮点：酒店/咖啡/温泉 | 节奏：轻松 | 避坑：减少换酒店",
            "亲子友好线｜适合：带娃 | 亮点：项目集中/午休友好 | 节奏：轻松 | 避坑：中午避晒",
        ]
        if days and days >= 3:
            advice = "Advice：如果有 3 天游玩，更推荐轻松度假线。"
        else:
            advice = "Advice：如果只有 1-2 天，更推荐经典打卡线。"
        questions = [
            "Next：出行时间更接近哪类？（本周末/节假日/自定日期）",
            "Next：同行结构更像哪种？（仅大人/亲子/含老人）",
            "Next：希望整体节奏？（轻松/适中/偏赶）",
        ]
        quick_replies = [
            "选经典",
            "选度假",
            "选亲子",
            "本周末",
            "节假日",
            "自定日期",
            "仅大人",
            "亲子",
            "含老人",
            "轻松",
            "适中",
            "偏赶",
            "我不确定",
        ]
    elif archetype == "NORTHEAST_SNOW":
        options = [
            "A 哈尔滨–亚布力–雪乡｜适合：第一次东北冰雪 | 亮点：冰雕/滑雪/雪景 | 节奏：适中 | 避坑：提前订住宿和车",
            "B 长白山–二道白河–延吉｜适合：温泉+雪景 | 亮点：温泉/天池/延吉美食 | 节奏：适中 | 避坑：防寒与路况",
            "C 沈阳–大连｜适合：城市人文 | 亮点：人文/海景/轻雪 | 节奏：轻松 | 避坑：行程别太赶",
            "混合建议：哈尔滨(2)→亚布力(1)→长白山温泉(1)→返程(1)",
        ]
        advice = "Advice：首去东北更推荐 A 线，想温泉放松则选 B。"
        questions = [
            "Next：出行时间更接近哪类？（元旦/寒假/自定日期）",
            "Next：同行结构是？（仅大人/亲子/含老人）",
            "Next：行程重心选哪种？（滑雪为主/温泉为主/各一半）",
        ]
        quick_replies = [
            "选A",
            "选B",
            "选C",
            "选混合",
            "我不确定",
            "元旦",
            "寒假",
            "自定日期",
            "滑雪为主",
            "温泉为主",
            "各一半",
        ]
    elif archetype == "HK_MO":
        options = [
            "澳门休闲线｜适合：想放松 | 亮点：老城+美食+酒店度假 | 节奏：轻松 | 避坑：错峰订酒店",
            "香港城市线｜适合：城市探索 | 亮点：维港夜景+街区+观景点 | 节奏：适中 | 避坑：地铁通勤规划",
            "港澳混合线｜适合：一次打卡 | 亮点：2+2 或 3+2 少折腾 | 节奏：适中 | 避坑：减少跨境次数",
        ]
        advice = "Advice：首次去港澳，默认建议港澳混合线。"
        questions = [
            "Next：是否购物/免税？（是/一般/不考虑）",
            "Next：是否亲子或带老人？（是/否）",
            "Next：节奏偏好？（轻松/适中）",
        ]
        quick_replies = [
            "选澳门",
            "选香港",
            "选混合",
            "购物是",
            "购物一般",
            "购物不",
            "亲子",
            "不亲子",
            "轻松",
            "适中",
        ]
    elif archetype == "HAINAN_ISLAND":
        options = [
            "三亚躺平度假｜适合：想放松 | 亮点：海景酒店+沙滩日落 | 节奏：轻松 | 避坑：少折腾",
            "轻环线｜适合：想多看一点 | 亮点：三亚+万宁 或 海口+文昌 | 节奏：适中 | 避坑：减少搬家次数",
            "亲子玩水｜适合：带娃 | 亮点：海边+室内备选+轻体力 | 节奏：轻松 | 避坑：防晒与休息",
        ]
        advice = "Advice：首次去海南，默认建议三亚躺平度假。"
        questions = [
            "Next：是否下水？（要/可能/不下水）",
            "Next：酒店偏好？（海景/亲子设施/性价比）",
            "Next：交通选择？（少折腾/更省）",
        ]
        quick_replies = [
            "要",
            "可能",
            "不下水",
            "海景",
            "亲子设施",
            "性价比",
            "少折腾",
            "更省",
        ]
    elif archetype == "CITY_BREAK":
        options = [
            f"{dest_text}名片线｜适合：第一次来 | 亮点：地标+老城+夜景 | 节奏：适中 | 避坑：错峰出行",
            f"{dest_text}美食慢逛线｜适合：爱吃 | 亮点：小吃街+夜景街区+1个代表性景点 | 节奏：轻松 | 避坑：少换住宿",
            f"{dest_text}亲子轻松线｜适合：带娃 | 亮点：公园/动物园/博物馆/室内备选 | 节奏：轻松 | 避坑：午后避晒",
        ]
        if re.search(r"吃", text_hint):
            advice = "Advice：如果你更关注吃的，优先美食慢逛线。"
        elif days and days <= 2:
            advice = "Advice：如果只有 1-2 天，更推荐城市名片线。"
        else:
            advice = "Advice：默认先走城市名片线，信息补齐后再细化。"
        questions = [
            "Next：更偏住哪里？（市中心/景区附近/交通枢纽）",
            "Next：同行结构是？（仅大人/亲子/含老人）",
            "Next：更偏哪类？（拍照打卡/吃喝逛/文化深度）",
        ]
        quick_replies = [
            "住市中心",
            "住景区",
            "住交通枢纽",
            "仅大人",
            "亲子",
            "含老人",
            "拍照",
            "吃喝",
            "文化",
            "我不确定",
        ]
    else:
        options = [
            "方案A｜适合：第一次来 | 亮点：经典地标 | 节奏：适中 | 避坑：错峰出行",
            "方案B｜适合：想放松 | 亮点：慢游体验 | 节奏：轻松 | 避坑：减少换酒店",
            "方案C｜适合：想深度 | 亮点：主题小众 | 节奏：紧凑 | 避坑：预留交通",
        ]
        advice = "Advice：如果是第一次来，更推荐方案A。"
        questions = [
            "Next：你更偏好的节奏？（轻松/适中/紧凑）",
            "Next：出行时间更接近哪类？（元旦/寒假/自定日期）",
            "Next：是否有特殊偏好？（美食/亲子/人文/自然/我不确定）",
        ]
        quick_replies = [
            "我不确定",
            "轻松",
            "适中",
            "紧凑",
            "元旦",
            "寒假",
            "自定日期",
            "美食",
            "亲子",
            "人文",
            "自然",
        ]

    options_block = "\n".join(["Options："] + [f"- {line}" for line in options])
    reply_text = "\n".join([f"Hook：{hook}", f"Confirm：{confirm}", options_block, advice] + questions[:3])
    return reply_text, quick_replies


def _render_discovery_7qs(origin_hint: Optional[str], region_hint: Optional[str]) -> str:
    origin_show = origin_hint or "（你也可以直接告诉我从哪里出发）"
    region_show = region_hint or "这趟行程"
    return "\n".join(
        [
            f"好呀，{region_show}我可以帮你做成一份“好抄作业”的自由行行程。",
            "我先用 7 个关键点把需求对齐（你只回 1–2 条也行，其余我先按常见值默认）：",
            f"1. 出发地：我这边看到你可能从「{origin_show}」出发（可改：北京/上海/广州等）",
            "2. 出行时间：元旦 / 寒假 / 周末 / 自定日期 / 我不确定",
            "3. 天数：2天 / 3天 / 4-5天 / 6-7天 / 我不确定",
            "4. 同行人：1人 / 情侣夫妻 / 2大1小 / 3大2小 / 带老人 / 我不确定",
            "5. 预算：经济 / 中等 / 舒适 / 高端（或直接说总预算）",
            "6. 玩法偏好：滑雪 / 温泉 / 冰雪 / 美食 / 人文 / 亲子 / 混合",
            "7. 约束/在意点：怕冷少走路 / 需要午睡&推车 / 想住温泉酒店 / 想轻松不赶路",
            "你也可以直接回一句整合信息，例如：",
            "“香港出发，元旦，4-5天，3大2小，中等预算，想滑雪+温泉，节奏休闲”",
        ]
    )


def _render_region_starter_card(region_value: str, origin_hint: Optional[str]) -> Tuple[str, List[str]]:
    region_text = region_value or "目的地"
    origin_text = f"从{origin_hint}出发" if origin_hint else "出发地未定"
    lines = [
        f"{region_text}范围较大，我先给你一个首轮选择卡（{origin_text}）：",
        "【A】经典打卡线：城市地标 + 代表景点，节奏适中",
        "【B】轻松度假线：酒店/温泉/慢游，节奏轻松",
        "【C】主题玩法线：按季节与兴趣选（如滑雪/美食/亲子）",
        "你更偏 A/B/C？也可以直接告诉我大概几天、几位同行。",
    ]
    quick_replies = ["选A", "选B", "选C", "我不确定", "元旦", "寒假", "未定"]
    return "\n".join(lines), quick_replies


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

    m_family = re.search(r"(\d+)\s*大\s*(\d+)\s*(?:小|娃|儿童)?", t)
    if m_family:
        try:
            adults = int(m_family.group(1))
            children = int(m_family.group(2))
            if adults > 0 or children > 0:
                result["adults"] = adults
                result["children"] = children
                result["people_count"] = adults + children
        except Exception:
            pass

    if "people_count" not in result:
        m_adult_child = re.search(r"(\d+)\s*(?:成人|大)\s*(\d+)\s*(?:小|娃|儿童)", t)
        if m_adult_child:
            try:
                adults = int(m_adult_child.group(1))
                children = int(m_adult_child.group(2))
                if adults > 0 or children > 0:
                    result["adults"] = adults
                    result["children"] = children
                    result["people_count"] = adults + children
            except Exception:
                pass

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

    m_route = re.search(
        r"(?:路线|线路)?\s*[\(（]?\s*([A-D])\s*[\)）]?\s*(?:线|路线|线路)?",
        t,
        re.IGNORECASE,
    )
    if m_route:
        result["route_choice"] = m_route.group(1).upper()
    elif re.search(r"(选A|A线|走A)", t, re.IGNORECASE):
        result["route_choice"] = "A"
    elif re.search(r"(选B|B线|走B)", t, re.IGNORECASE):
        result["route_choice"] = "B"
    elif re.search(r"(选C|C线|走C)", t, re.IGNORECASE):
        result["route_choice"] = "C"
    elif re.search(r"(选D|D线|走D)", t, re.IGNORECASE):
        result["route_choice"] = "D"
    elif re.search(r"(混合|都想要|A和B)", t, re.IGNORECASE):
        result["route_choice"] = "MIX"

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

    adults = new_slots.get("adults")
    children = new_slots.get("children")
    people_count = new_slots.get("people_count")
    traveler_count = new_slots.get("traveler_count")

    if adults is not None:
        old_value = getattr(slots, "adults", None)
        if old_value != adults:
            _mark_source("adults", adults)
        slots.adults = adults
    if children is not None:
        old_value = getattr(slots, "children", None)
        if old_value != children:
            _mark_source("children", children)
        slots.children = children

    if people_count is not None:
        old_value = getattr(slots, "people_count", None)
        if old_value is None:
            _mark_source("people_count", people_count)
            slots.people_count = people_count
    elif traveler_count:
        has_detail = getattr(slots, "adults", None) is not None or getattr(slots, "children", None) is not None
        if not has_detail:
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

    route_choice = new_slots.get("route_choice")
    if route_choice:
        old_choice = meta.get("route_choice")
        if old_choice != route_choice:
            _mark_source("route_choice", route_choice)
        meta["route_choice"] = route_choice

    meta["slot_sources"] = slot_sources
    setattr(slots, "meta", meta)
    return slots


def _rebuild_slots_from_history(
    origin: Optional[str],
    destination: Optional[str],
    history_with_new_user: List[ChatMessage],
    days_guess: Optional[int],
) -> TripPlanRequest:
    slots = TripPlanRequest()
    if destination:
        slots.destination = destination
        meta = getattr(slots, "meta", None)
        if not isinstance(meta, dict):
            meta = {}
        meta["destination_source"] = "client"
        setattr(slots, "meta", meta)
    if origin:
        _set_origin_with_source(slots, origin, "location")

    user_msgs = [m for m in history_with_new_user if m and m.role == "user"]
    for msg in user_msgs[-12:]:
        parsed = _parse_user_slots(msg.content or "")
        slots = _merge_slots(slots, parsed)

    return slots


def _merge_slots_from_existing(base: TripPlanRequest, incoming: Optional[TripPlanRequest]) -> TripPlanRequest:
    if incoming is None:
        return base
    if not getattr(base, "destination", None) and getattr(incoming, "destination", None):
        base.destination = incoming.destination
    if not getattr(base, "origin", None) and getattr(incoming, "origin", None):
        _set_origin_with_source(base, incoming.origin, "client")
    if not _valid_date_range(getattr(base, "date_range", None)) and _valid_date_range(
        getattr(incoming, "date_range", None)
    ):
        base.date_range = incoming.date_range
    if getattr(base, "adults", None) is None and getattr(incoming, "adults", None) is not None:
        base.adults = incoming.adults
    if getattr(base, "children", None) is None and getattr(incoming, "children", None) is not None:
        base.children = incoming.children
    if getattr(base, "people_count", None) is None and getattr(incoming, "people_count", None) is not None:
        base.people_count = incoming.people_count
    if getattr(base, "budget_level", None) is None and getattr(incoming, "budget_level", None) is not None:
        base.budget_level = incoming.budget_level
    if not getattr(base, "preferences", None) and getattr(incoming, "preferences", None):
        base.preferences = incoming.preferences
    if getattr(base, "pace_level", None) is None and getattr(incoming, "pace_level", None) is not None:
        base.pace_level = incoming.pace_level
    if getattr(base, "style", None) is None and getattr(incoming, "style", None) is not None:
        base.style = incoming.style
    incoming_meta = getattr(incoming, "meta", None)
    if isinstance(incoming_meta, dict):
        base_meta = getattr(base, "meta", None)
        if not isinstance(base_meta, dict):
            base_meta = {}
        if incoming_meta.get("route_choice") and not base_meta.get("route_choice"):
            base_meta["route_choice"] = incoming_meta["route_choice"]
        setattr(base, "meta", base_meta)
    return base


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


def _ensure_meta(slots: TripPlanRequest) -> Dict[str, Any]:
    if slots.meta is None:
        slots.meta = {}
    return slots.meta


def _mark_declined(slots: TripPlanRequest, slot_key: str, reason: str = "user_declined") -> None:
    meta = _ensure_meta(slots)
    declined = meta.get("declined_slots") or {}
    declined[slot_key] = {"reason": reason, "at": int(time.time())}
    meta["declined_slots"] = declined


def _last_asked_slot(slots: TripPlanRequest) -> Optional[str]:
    meta = _ensure_meta(slots)
    asked = meta.get("asked_slots") or []
    if not asked:
        return None
    last = asked[-1] if isinstance(asked, list) else None
    return last.get("slot_key") if isinstance(last, dict) else None


def _is_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"(不告诉你|不想说|不方便说|不说|不想聊|先不说|不回答|不了|随便|你猜)",
            t,
        )
    )


def _slot_filled(slot_key: str, slots: Optional[TripPlanRequest], days_guess: Optional[int]) -> bool:
    if slots is None:
        return False
    if slot_key == "traveler_count":
        return getattr(slots, "people_count", None) is not None or getattr(slots, "adults", None) is not None
    if slot_key == "days_or_date_range":
        return _valid_date_range(getattr(slots, "date_range", None)) or bool(days_guess)
    if slot_key == "destination":
        return bool(getattr(slots, "destination", None))
    if slot_key == "destination_city":
        dest = getattr(slots, "destination", None)
        return bool(dest) and not _is_region_destination_value(dest)
    if slot_key == "origin":
        return bool(getattr(slots, "origin", None))
    if slot_key == "budget_level":
        return bool(getattr(slots, "budget_level", None))
    if slot_key == "preferences":
        prefs = list(getattr(slots, "preferences", None) or []) or list(getattr(slots, "interests", None) or [])
        return bool(prefs)
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
    if slots is not None:
        meta = getattr(slots, "meta", None)
        if isinstance(meta, dict):
            declined = meta.get("declined_slots") or {}
            if declined.get(slot_key):
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
            "destination_city": "哪个城市",
            "origin": "出发",
            "budget_level": "预算",
            "preferences": "偏好",
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
        "destination_city": "你更想去哪个城市？",
        "origin": "从哪个城市出发？",
        "budget_level": "预算大概什么档位？",
        "preferences": "你更偏好哪类玩法？",
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
        "destination_city": "(哈尔滨/长白山/大连/沈阳/不确定)",
        "origin": "(本地出发/周边城市/不确定)",
        "budget_level": "(经济/舒适/高端/不确定)",
        "preferences": "(滑雪/温泉/美食/自然/不确定)",
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


REGION_KEYWORDS = [
    "东北",
    "华北",
    "华东",
    "华南",
    "西北",
    "西南",
    "川西",
    "江浙沪",
    "大湾区",
    "长三角",
    "珠三角",
    "华中",
    "新疆",
    "内蒙",
    "青甘",
    "西北地区",
    "西南地区",
]

REGION_CITY_CANDIDATES = {
    "东北": ["哈尔滨", "长春", "沈阳", "大连", "长白山"],
    "华东": ["上海", "杭州", "苏州", "南京", "厦门"],
    "西南": ["成都", "重庆", "昆明", "丽江", "西双版纳"],
    "西北": ["西安", "兰州", "西宁", "张掖", "敦煌"],
    "青甘": ["西宁", "张掖", "敦煌", "青海湖", "茶卡盐湖"],
    "大湾区": ["深圳", "广州", "珠海", "中山", "佛山"],
    "江浙沪": ["上海", "杭州", "苏州", "南京", "无锡"],
    "长三角": ["上海", "杭州", "苏州", "南京", "无锡"],
    "珠三角": ["深圳", "广州", "珠海", "中山", "佛山"],
    "川西": ["成都", "稻城亚丁", "康定", "四姑娘山", "色达"],
    "新疆": ["乌鲁木齐", "喀什", "伊犁", "吐鲁番", "阿勒泰"],
    "内蒙": ["呼和浩特", "包头", "鄂尔多斯", "阿拉善", "满洲里"],
}


def _extract_region_from_text(text: str) -> Optional[str]:
    t = text or ""
    for key in REGION_KEYWORDS:
        if key and key in t:
            return key
    return None


def _is_region_destination_value(dest: Optional[str]) -> bool:
    if not dest:
        return False
    return any(key in str(dest) for key in REGION_KEYWORDS)


def _region_city_candidates(region: Optional[str]) -> List[str]:
    if not region:
        return []
    for key, candidates in REGION_CITY_CANDIDATES.items():
        if key in region:
            return list(candidates)
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


def _enforce_v1_actions_style(reply_text: str) -> str:
    text = (reply_text or "").strip()
    if not text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    numbered = [ln for ln in lines if re.match(r"^\s*\d+[.、)]\s*", ln)]

    # v2: allow up to 7 numbered questions; only trim if too many.
    if len(numbered) >= 8:
        kept = []
        count = 0
        for ln in lines:
            if re.match(r"^\s*\d+[.、)]\s*", ln):
                count += 1
                if count <= 7:
                    kept.append(ln)
                continue
            kept.append(ln)
        return "\n".join(kept).strip()

    return text


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
    has_region = bool(
        re.search(
            r"(东北|华北|华东|华中|华南|西北|西南|川西|江浙沪|大湾区|长三角|珠三角|内蒙|青甘|西北地区|西南地区|云南|新疆|海南)",
            t,
        )
    )
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
    if "川西" in t:
        return "川西"
    if "江浙沪" in t:
        return "江浙沪"
    if "大湾区" in t:
        return "大湾区"
    if "长三角" in t:
        return "长三角"
    if "珠三角" in t:
        return "珠三角"
    if "内蒙" in t:
        return "内蒙"
    if "青甘" in t:
        return "青甘"
    if "西北地区" in t:
        return "西北地区"
    if "西南地区" in t:
        return "西南地区"
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


def _select_actions_question(
    slots: Optional[TripPlanRequest],
    text_merge: str,
    days_guess: Optional[int],
    history: List[ChatMessage],
    current_turn: int,
) -> Tuple[Optional[str], Optional[str], List[str], Optional[TripPlanRequest]]:
    declined = {}
    starter_card_shown = False
    if slots is not None:
        meta = getattr(slots, "meta", None)
        if isinstance(meta, dict):
            declined = meta.get("declined_slots") or {}
            starter_card_shown = bool(meta.get("starter_card_shown"))
    destination_value = getattr(slots, "destination", None) if slots is not None else None
    region_value = destination_value if _is_region_destination_value(destination_value) else None

    if not destination_value:
        region_hit = _extract_region_from_text(text_merge)
        if region_hit:
            region_value = region_hit
            if slots is None:
                slots = TripPlanRequest()
            slots.destination = region_hit
            destination_value = region_hit

    candidates = [
        ("destination_city", region_value is not None),
        ("destination", not destination_value),
        ("days_or_date_range", not _valid_date_range(getattr(slots, "date_range", None)) and not days_guess),
        (
            "traveler_count",
            getattr(slots, "people_count", None) is None and getattr(slots, "adults", None) is None,
        ),
        ("budget_level", not getattr(slots, "budget_level", None)),
        (
            "preferences",
            not list(getattr(slots, "preferences", None) or []) and not list(getattr(slots, "interests", None) or []),
        ),
    ]

    for slot_key, needed in candidates:
        if not needed:
            continue
        if declined.get(slot_key):
            continue
        if slot_key == "destination_city" and starter_card_shown:
            continue
        if not _should_ask(slot_key, slots, history, current_turn, days_guess):
            continue
        if slot_key == "destination_city":
            origin_hint = getattr(slots, "origin", None) if slots is not None else None
            prompt = _render_discovery_7qs(origin_hint=origin_hint, region_hint=region_value)
            quick_replies = _region_city_candidates(region_value)
            quick_replies = (quick_replies or []) + ["我还不确定"]
            return slot_key, prompt, quick_replies, slots
        if slot_key == "destination":
            prompt = "这次想去哪个城市？"
            quick_replies = _dest_quick_replies(text_merge) or ["北京", "上海", "成都", "广州", "我还不确定"]
            return slot_key, prompt, quick_replies, slots
        if slot_key == "days_or_date_range":
            prompt = "计划玩几天，或具体日期是哪几天？"
            quick_replies = ["本周末", "下周末", "元旦", "春节", "2天", "3天", "4-5天", "我还不确定"]
            return slot_key, prompt, quick_replies, slots
        if slot_key == "traveler_count":
            prompt = "几位出行？"
            quick_replies = ["1人", "2人", "3人", "4人以上", "我还不确定"]
            return slot_key, prompt, quick_replies, slots
        if slot_key == "budget_level":
            prompt = "预算大概什么档位？"
            quick_replies = ["经济", "舒适", "高端", "我还不确定"]
            return slot_key, prompt, quick_replies, slots
        if slot_key == "preferences":
            prompt = "你更偏好哪类玩法？"
            quick_replies = ["滑雪", "温泉", "冰雕", "美食", "自然风光", "城市打卡", "我还不确定"]
            return slot_key, prompt, quick_replies, slots

    return None, None, [], slots


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
    declined = {}
    if slots is not None:
        meta = getattr(slots, "meta", None)
        if isinstance(meta, dict):
            declined = meta.get("declined_slots") or {}
    candidates = [
        ("style", "你更偏户外运动还是休闲度假？"),
        ("budget_level", "预算大概什么档位（经济/舒适/高端）？"),
    ]
    for key, prompt in candidates:
        if declined.get(key):
            continue
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


def _season_hint(text: str) -> str:
    if re.search(r"(冬|雪|冰|元旦|寒假|12月|1月|2月)", text or ""):
        return "winter"
    return ""


def _build_dialog_context(
    req: TripChatRequest,
    slots: Optional[TripPlanRequest],
    intent: str,
    region: Optional[str],
    days_guess: Optional[int],
) -> Dict[str, Any]:
    destination = getattr(slots, "destination", None) if slots is not None else None
    dest_key = _normalize_dest_key(destination, req.input_text)
    origin = None
    if slots is not None and getattr(slots, "origin", None):
        origin = slots.origin
    else:
        origin = req.origin or req.default_origin

    traveler = {
        "people_count": getattr(slots, "people_count", None) if slots is not None else None,
        "adults": getattr(slots, "adults", None) if slots is not None else None,
        "children": getattr(slots, "children", None) if slots is not None else None,
        "elders": getattr(slots, "elders", None) if slots is not None else None,
    }
    preferences = []
    if slots is not None:
        preferences = list(getattr(slots, "preferences", None) or []) or list(
            getattr(slots, "interests", None) or []
        )

    return {
        "dest_key": dest_key or region,
        "origin": origin,
        "days_guess": days_guess,
        "date_range": getattr(slots, "date_range", None) if slots is not None else None,
        "traveler": traveler,
        "preferences": preferences,
        "season_hint": _season_hint(req.input_text),
        "intent": intent,
        "note": "不要问超过3个问题，先给路线/方向菜单",
    }


def _force_dialog_draft_prompt(user_text: str, ctx: Dict[str, Any]) -> str:
    ctx_json = json.dumps(ctx or {}, ensure_ascii=False)
    return (
        f"{user_text}\n"
        "你是旅行助手，请严格输出 JSON-only（不要 Markdown、不要多余文字），结构如下：\n"
        "{\n"
        '  "hook": "...",\n'
        '  "confirm": "...",\n'
        '  "options": [\n'
        '    {"key":"A","title":"...","fit_for":"...","highlights":["..."],"pace":["..."],"pitfalls":["..."] }\n'
        "  ],\n"
        '  "recommend_key":"A",\n'
        '  "recommend_reason":"...",\n'
        '  "questions":[\n'
        '    {"key":"time","prompt":"...","options":["元旦","寒假","自定日期","我不确定"]}\n'
        "  ],\n"
        '  "next_step":"...",\n'
        '  "quick_replies":["选A","选B","我不确定"]\n'
        "}\n"
        "规则：\n"
        "- 先 options 再 questions\n"
        "- options 2-4 个；每个 highlights 3-6，pace 2-4，pitfalls 0-3\n"
        "- questions 0-3 个；每个 options 3-6 且包含“我不确定”\n"
        "- quick_replies 需要包含：选A/选B/选C/选混合（当 dest_key 是区域/大范围如“东北/海南/港澳”时）\n"
        "上下文（仅参考）：\n"
        f"{ctx_json}\n"
    )


def _call_llm_for_dialog_draft(
    req: TripChatRequest, system_prompt: str, ctx: Dict[str, Any]
) -> Tuple[Optional[DialogDraft], Optional[str]]:
    def _validate_draft(parsed: Dict[str, Any]) -> Optional[DialogDraft]:
        if parsed is None:
            return None
        try:
            return DialogDraft.model_validate(parsed)
        except Exception:
            try:
                return DialogDraft.parse_obj(parsed)
            except Exception:
                return None

    prompt = _force_dialog_draft_prompt(req.input_text, ctx)
    decision = call_qwen_for_trip_chat(_build_llm_input(prompt, req, system_prompt))
    reply1 = decision.reply or ""
    parsed = _try_parse_json(reply1)
    draft = _validate_draft(parsed)
    if draft is not None:
        return draft, reply1

    retry_prompt = _force_dialog_draft_prompt(req.input_text, ctx) + "再次提醒：只输出 JSON。"
    decision2 = call_qwen_for_trip_chat(_build_llm_input(retry_prompt, req, system_prompt))
    reply2 = decision2.reply or ""
    parsed2 = _try_parse_json(reply2)
    draft2 = _validate_draft(parsed2)
    if draft2 is not None:
        return draft2, reply2
    return None, reply2 or reply1


def _render_dialog_draft(draft: DialogDraft, slots: TripPlanRequest) -> str:
    lines: List[str] = []
    meta = getattr(slots, "meta", None)
    route_choice = meta.get("route_choice") if isinstance(meta, dict) else None

    def _should_keep_question(qkey: str) -> bool:
        if qkey in {"days_or_date_range", "date_range"}:
            return not _valid_date_range(getattr(slots, "date_range", None))
        if qkey in {"traveler_count", "people_count"}:
            return getattr(slots, "people_count", None) is None and getattr(slots, "adults", None) is None
        if qkey == "budget_level":
            return not getattr(slots, "budget_level", None)
        if qkey in {"pace_level", "pace"}:
            return not getattr(slots, "pace_level", None)
        return True

    if route_choice and draft.options:
        picked = None
        for card in draft.options:
            if str(card.key or "").upper() == str(route_choice or "").upper():
                picked = card
                break
        if picked:
            lines.append(f"你选了【{picked.key}】{picked.title}，我会按这个路线为你细化。")
            lines.append(f"适合：{picked.fit_for}")
            lines.append("亮点：" + " / ".join(picked.highlights))
            lines.append("节奏：" + " / ".join(picked.pace))
            if picked.pitfalls:
                lines.append("避坑：" + " / ".join(picked.pitfalls))
    else:
        lines.append("【路线选项】")
        lines.append(draft.hook)
        lines.append(draft.confirm)
        lines.append("我先给你几个方向，你选完我再细化：")
        for card in draft.options:
            lines.append(f"【{card.key}】{card.title}")
            lines.append(f"适合：{card.fit_for}")
            lines.append("亮点：" + " / ".join(card.highlights))
            lines.append("节奏：" + " / ".join(card.pace))
            if card.pitfalls:
                lines.append("避坑：" + " / ".join(card.pitfalls))
        lines.append(f"推荐：我更建议选【{draft.recommend_key}】：{draft.recommend_reason}")
        lines.extend(
            [
                "我先用 7 个关键点把需求对齐（你只回 1–2 条也行，其余我先按常见值默认）：",
                "1. 出发地：你可以直接告诉我从哪里出发",
                "2. 出行时间：元旦 / 寒假 / 周末 / 自定日期 / 我不确定",
                "3. 天数：2天 / 3天 / 4-5天 / 6-7天 / 我不确定",
                "4. 同行人：1人 / 情侣夫妻 / 2大1小 / 3大2小 / 带老人 / 我不确定",
                "5. 预算：经济 / 中等 / 舒适 / 高端（或直接说总预算）",
                "6. 玩法偏好：滑雪 / 温泉 / 冰雪 / 美食 / 人文 / 亲子 / 混合",
                "7. 约束/在意点：怕冷少走路 / 需要午睡&推车 / 想住温泉酒店 / 想轻松不赶路",
            ]
        )

    if draft.questions:
        remaining = [q for q in draft.questions if _should_keep_question(q.key)]
        if remaining:
            lines.append("再确认 1-2 个小问题，我就能给你出草稿：")
            for idx, q in enumerate(remaining[:3], 1):
                opts = " / ".join(q.options)
                lines.append(f"{idx}) {q.prompt}（{opts}）")
    lines.append(draft.next_step)
    return "\n".join([ln for ln in lines if (ln or "").strip()])


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


def _looks_like_spots_json(s: str) -> bool:
    t = (s or "").strip()
    if not t.startswith("{"):
        return False
    return ('"summary"' in t or "summary" in t) and ('"spots"' in t or "spots" in t)


def _render_spots_json_to_user(obj: dict, destination: str) -> str:
    summary = (obj.get("summary") or "").strip()
    spots = obj.get("spots") or []
    if summary == "无法生成" or not isinstance(spots, list) or len(spots) == 0:
        return (
            f"我先接住：你现在想玩【{destination}】。\n"
            "我可以先给你两种方向，你选一个，我再按你家带娃的节奏细化：\n"
            "A）滑雪 + 温泉（轻松度假）\n"
            "B）天池 + 雾凇（经典打卡）\n\n"
            "你更偏 A 还是 B？顺便告诉我：大概几天、几位同行（有几个小朋友）？"
        )

    lines = [f"先给你一份【{destination}】必玩清单（我再按你的天数去排程）："]
    for i, it in enumerate(spots[:8], 1):
        name = (it.get("name") or "").strip()
        cat = (it.get("category") or "").strip()
        brief = (it.get("brief_desc") or "").strip()
        if not name:
            continue
        tail = "｜" + cat if cat else ""
        lines.append(f"{i}. {name}{tail}：{brief}")
    lines.append("\n你更想偏“滑雪/温泉/天池/亲子轻松”？我按偏好给你 2-3 条路线选项。")
    return "\n".join(lines)


def _is_interrupt(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"(讲个故事|来个故事|睡前故事|讲个笑话|来个笑话|心情不好|难受|焦虑|安慰我|鼓励我|先不聊旅行|暂停一下)",
            t,
        )
    )


def _render_interrupt_reply(text: str) -> str:
    t = (text or "").strip()
    if re.search(r"(笑话|逗我开心)", t):
        return (
            "给你来个短的：\n有个人去买伞，老板问：要大的还是小的？\n他说：要能遮住我最近的坏心情的。\n"
            "老板想了想：那得买把‘想开点’。\n\n"
            "如果你愿意，跟我说一句你今天最烦的事，我帮你把它拆小一点。"
        )
    if re.search(r"(心情不好|难受|焦虑|安慰|鼓励)", t):
        return (
            "我听到了：你现在有点扛不住，但你并没有放弃把事情变好。\n\n"
            "先做一个很小的动作：喝口水、把肩膀放松、深呼吸三次。\n"
            "然后告诉我：你更像是“累”还是“烦”还是“委屈”？我按这个给你一个更贴合的缓解方案。"
        )
    return (
        "给你讲个很短的故事：\n有个旅人背着一袋石头走路，越走越累。路边的老人问：你为什么不放下一块？\n"
        "旅人说：我怕少了一块就不完整。\n老人笑了：完整不是把所有都背着，而是知道什么时候该放下。\n\n"
        "你也一样。先把最重的那一块告诉我是哪一块。"
    )


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

    # ----- 0) 中断优先：讲故事/笑话/情绪安抚，先接住再说 -----
    if _is_interrupt(req.input_text):
        reply_text = _render_interrupt_reply(req.input_text)

        history_with_new_user = list(req.history or [])
        if not history_with_new_user:
            history_with_new_user = [ChatMessage(role="user", content=req.input_text)]
        else:
            last = history_with_new_user[-1]
            if not (
                last.role == "user"
                and (last.content or "").strip() == (req.input_text or "").strip()
            ):
                history_with_new_user.append(ChatMessage(role="user", content=req.input_text))

        new_history = list(history_with_new_user) + [
            ChatMessage(role="assistant", content=reply_text)
        ]

        resources_out = {
            "quick_replies": ["继续规划", "我想看路线选项", "换个目的地", "改预算档位", "改出行日期"]
        }

        return TripChatResponse(
            reply=reply_text,
            history=new_history,
            slots=None,
            trip_plan=None,
            resources=resources_out,
            mode="CHAT",
            dialog_state=DialogState.PAUSED,
            pending_questions=[],
            next_action=NextAction(reason="interrupt"),
        )

    # ----- 1) 知识库：不再早退，只先记录 resources -----
    kb_res = kb.kb_search(kb.KbSearch(query=req.input_text, destination=""))
    kb_items = kb_res.get("items") if kb_res else []
    resources_from_kb = _kb_to_resources(kb_items) if kb_items else None

    # ----- 2) 调 LLM（强约束 JSON，仅推荐场景使用） -----
    reply_text = ""
    parsed_json = None
    decision_used = None

    if _needs_recommendations(req.input_text):
        _raw_json_reply, parsed_json, decision_used = _call_llm_for_json(req, system_content)

    # 只把“本轮用户消息”先加进 history，用于统计 user_turns
    history_with_new_user = list(req.history)
    if not history_with_new_user:
        history_with_new_user = [ChatMessage(role="user", content=req.input_text)]
    else:
        last = history_with_new_user[-1]
        same_user = (
            last.role == "user"
            and (last.content or "").strip() == (req.input_text or "").strip()
        )
        if not same_user:
            history_with_new_user.append(ChatMessage(role="user", content=req.input_text))
    user_turns = _count_user_turns(history_with_new_user)
    _ = user_turns  # 保留变量以避免未来逻辑改动时被误删

    text_merge = req.input_text + "\n" + "\n".join(
        [m.content for m in history_with_new_user if m.role == "user"]
    )
    days_guess = _guess_days(text_merge)

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

    slots = _rebuild_slots_from_history(
        req.origin, req.destination, history_with_new_user, days_guess
    )
    slots = _merge_slots_from_existing(slots, slots_from_llm)
    slots = _merge_slots_from_existing(slots, req.current_slots)
    if req.slots is not None:
        try:
            raw_slots = req.slots.dict(exclude_none=True)
            dr = raw_slots.get("date_range")
            if isinstance(dr, str):
                raw_slots["date_range"] = []
            elif isinstance(dr, dict):
                start = dr.get("start_date") or dr.get("start")
                end = dr.get("end_date") or dr.get("end")
                raw_slots["date_range"] = [start, end] if (start or end) else []
            slots = _merge_slots_from_existing(slots, TripPlanRequest.parse_obj(raw_slots))
        except Exception:
            logger.exception("TripChat: failed to parse req.slots")

    # ----- 4) 猜槽位（目的地/天数/出发地/偏好），并与 slots 融合 -----
    guess_slots = TripPlanRequest()
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
    if slots is not None and _is_refusal(req.input_text):
        last = _last_asked_slot(slots)
        if last:
            _mark_declined(slots, last)
        if last in {"destination_city", "traveler_count", "budget_level", "preferences"}:
            dest_key = _normalize_dest_key(getattr(slots, "destination", None) or req.destination, text_merge)
            region_value = _pick_region(text_merge) if _is_region_query(text_merge) else None
            archetype = _route_archetype(dest_key, text_merge, region_value)
            reply_text, actions_quick_replies = _render_actions_starter(
                archetype,
                dest_key,
                _guess_days(text_merge),
                origin_value or req.default_origin or req.origin,
                text=req.input_text,
            )
            reply_text = (reply_text or "") + "\n\n你只要回 1 条即可。"
            resources_out = resources_from_kb if isinstance(resources_from_kb, dict) else {}
            if actions_quick_replies:
                resources_out["quick_replies"] = actions_quick_replies[:12]
            new_history = list(history_with_new_user) + [ChatMessage(role="assistant", content=reply_text)]
            slot_completeness = SlotCompleteness(
                required_done=_estimate_required_done(slots),
                required_total=4,
            )
            return TripChatResponse(
                reply=reply_text,
                history=new_history,
                slots=slots,
                trip_plan=None,
                resources=resources_out,
                mode="EXPLORE",
                dialog_state=DialogState.DISCOVERY,
                slot_completeness=slot_completeness,
                pending_questions=[],
                next_action=NextAction(type="NONE", reason="user_declined"),
                trip_profile=_build_trip_profile(slots),
            )

    # ---- DialogDraft takeover (V1) ----
    intent = _classify_intent(req.input_text)
    # ----- 意图修正：在旅行上下文里，纯地名输入也应视为 explore -----
    if getattr(slots, "destination", None) and len((req.input_text or "").strip()) <= 8:
        intent = "explore"
    region = _pick_region(text_merge) if _is_region_query(text_merge) else None
    days_guess = _guess_days(text_merge)
    has_user_origin = bool(req.origin) or (
        bool(getattr(slots, "origin", None)) and getattr(slots, "origin", None) != req.default_origin
    )
    missing_required = _evaluate_missing_required(slots, days_guess, has_user_origin)
    actions_intent = (intent in {"plan", "explore"} or _needs_recommendations(req.input_text))
    region_value = None
    if slots is not None and _is_region_destination_value(getattr(slots, "destination", None)):
        region_value = slots.destination
    if not region_value:
        region_hit = _extract_region_from_text(text_merge)
        if region_hit:
            region_value = region_hit
            if slots is None:
                slots = TripPlanRequest()
            slots.destination = region_hit
    if region_value and user_turns <= 2:
        meta = _ensure_meta(slots)
        if not meta.get("starter_card_shown"):
            meta["starter_card_shown"] = True
            reply_text, starter_quick_replies = _render_region_starter_card(
                region_value,
                origin_value or req.default_origin or req.origin,
            )
            resources_out = resources_from_kb if isinstance(resources_from_kb, dict) else {}
            if starter_quick_replies:
                resources_out["quick_replies"] = starter_quick_replies[:12]
            new_history = list(history_with_new_user) + [ChatMessage(role="assistant", content=reply_text)]
            slot_completeness = SlotCompleteness(
                required_done=_estimate_required_done(slots),
                required_total=4,
            )
            return TripChatResponse(
                reply=reply_text,
                history=new_history,
                slots=slots,
                trip_plan=None,
                resources=resources_out,
                mode="EXPLORE",
                dialog_state=DialogState.DISCOVERY,
                slot_completeness=slot_completeness,
                pending_questions=[],
                next_action=NextAction(type="ASK", reason="starter_card"),
                trip_profile=_build_trip_profile(slots),
            )
    suggested_quick_replies: List[str] = []
    actions_slot_key, actions_prompt, actions_quick_replies, slots = _select_actions_question(
        slots,
        text_merge,
        days_guess,
        history_with_new_user,
        user_turns,
    )
    if actions_intent and actions_prompt and not _detect_refine_intent(req.input_text):
        suggested_quick_replies = actions_quick_replies or []
    use_dialog = (
        (intent in {"plan", "explore"} or _needs_recommendations(req.input_text))
        and (missing_required or region)
        and not _detect_refine_intent(req.input_text)
    )
    draft = None
    draft_reply = None
    if use_dialog:
        ctx = _build_dialog_context(req, slots, intent, region, days_guess)
        draft, draft_reply = _call_llm_for_dialog_draft(req, system_content, ctx)
    logger.info(
        "dialog_draft_takeover use_dialog=%s ok=%s trace_id=%s",
        use_dialog,
        draft is not None,
        trace_id,
    )
    if use_dialog and draft is not None:
        reply_text = _render_dialog_draft(draft, slots)

        resources_out = resources_from_kb if isinstance(resources_from_kb, dict) else {}
        qrs = list(getattr(draft, "quick_replies", None) or [])
        meta = getattr(slots, "meta", None)
        if isinstance(meta, dict) and meta.get("route_choice"):
            qrs = ["元旦", "寒假", "未定", "3大2小", "2大1小", "1人", "轻松", "均衡", "紧凑"]
        if qrs:
            existing = resources_out.setdefault("quick_replies", [])
            if not isinstance(existing, list):
                existing = []
                resources_out["quick_replies"] = existing
            for x in qrs:
                if x and x not in existing:
                    existing.append(x)
            resources_out["quick_replies"] = existing[:12]

        new_history = list(history_with_new_user) + [ChatMessage(role="assistant", content=reply_text)]
        slot_completeness = SlotCompleteness(
            required_done=_estimate_required_done(slots),
            required_total=4,
        )
        return TripChatResponse(
            reply=reply_text,
            history=new_history,
            slots=slots,
            trip_plan=None,
            resources=resources_out,
            mode="EXPLORE",
            dialog_state=DialogState.DISCOVERY,
            slot_completeness=slot_completeness,
            pending_questions=[],
            next_action=NextAction(type="ASK", reason="dialog_draft"),
            trip_profile=_build_trip_profile(slots),
        )
    if use_dialog and draft is None:
        reply_preview = (draft_reply or "")[:200]
        logger.warning(
            "dialog_draft_failed trace_id=%s reply_preview=%s",
            trace_id,
            reply_preview,
        )
        actions_slot_key, actions_prompt, actions_quick_replies, slots = _select_actions_question(
            slots,
            text_merge,
            days_guess,
            history_with_new_user,
            user_turns,
        )
        if actions_prompt:
            suggested_quick_replies = actions_quick_replies or []

    # ----- 4.2) empty/generic reply fallback -----
    destination_value = slots.destination if slots is not None else None
    intent = _classify_intent(req.input_text)
    # ----- 意图修正：在旅行上下文里，纯地名输入也应视为 explore -----
    if getattr(slots, "destination", None) and len((req.input_text or "").strip()) <= 8:
        intent = "explore"
    region = _pick_region(text_merge) if _is_region_query(text_merge) else None
    should_plan = intent == "plan" and not (region and not destination_value)
    days_guess = _guess_days(text_merge)
    has_user_origin = bool(req.origin) or (
        bool(getattr(slots, "origin", None)) and getattr(slots, "origin", None) != req.default_origin
    )
    missing_required = _evaluate_missing_required(slots, days_guess, has_user_origin)
    # ----- 强制推进：有目的地但关键槽位缺失时，一定要问问题 -----
    destination_value = getattr(slots, "destination", None) if slots is not None else None
    if destination_value and missing_required:
        followup = _pick_followup_question(slots, history_with_new_user, user_turns, _guess_days(text_merge))
        if followup:
            reply_text = followup["prompt"]
            resources_out = resources_from_kb if isinstance(resources_from_kb, dict) else {}
            qrs = followup.get("quick_replies") or []
            if qrs:
                resources_out["quick_replies"] = qrs[:12]

            new_history = list(history_with_new_user) + [ChatMessage(role="assistant", content=reply_text)]
            return TripChatResponse(
                reply=reply_text,
                history=new_history,
                slots=slots,
                trip_plan=None,
                resources=resources_out,
                mode="EXPLORE",
                dialog_state=DialogState.DISCOVERY,
                slot_completeness=SlotCompleteness(
                    required_done=_estimate_required_done(slots), required_total=4
                ),
                pending_questions=[],
                next_action=NextAction(type="ASK", reason="force_followup"),
                trip_profile=_build_trip_profile(slots),
            )
    refine_intent = _detect_refine_intent(req.input_text)
    dest_key = _normalize_dest_key(destination_value or req.destination, text_merge)
    use_dialog_draft = False
    use_dialog_draft_success = False
    dialog_quick_replies: List[str] = []
    if (intent in {"plan", "explore"} or _needs_recommendations(req.input_text)) and not refine_intent:
        if missing_required or region or dest_key in {"东北"}:
            use_dialog_draft = True
    if use_dialog_draft:
        ctx = _build_dialog_context(req, slots, intent, region, days_guess)
        draft, _draft_reply = _call_llm_for_dialog_draft(req, system_content, ctx)
        if draft is not None:
            reply_text = _render_dialog_draft(draft, slots)
            dialog_quick_replies = list(draft.quick_replies or [])
            meta = getattr(slots, "meta", None)
            if isinstance(meta, dict) and meta.get("route_choice"):
                dialog_quick_replies = ["元旦", "寒假", "未定", "3大2小", "2大1小", "1人", "轻松", "均衡", "紧凑"]
            use_dialog_draft_success = True
    logger.info(
        "dialog_draft_result trace_id=%s success=%s",
        trace_id,
        use_dialog_draft_success,
    )
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

    actions_routed = False
    actions_quick_replies: List[str] = []
    actions_trigger = re.search(r"(几日游|几天|行程|路线|怎么?玩|推荐|攻略|特色|项目|安排)", req.input_text)
    if actions_trigger and missing_required and not use_dialog_draft_success:
        archetype = _route_archetype(dest_key, text_merge, region)
        reply_text, actions_quick_replies = _render_actions_starter(
            archetype,
            dest_key,
            days_guess,
            origin_value or req.default_origin or req.origin,
            text=req.input_text,
        )
        actions_routed = True

    if not actions_routed and not use_dialog_draft_success and _needs_recommendations(req.input_text):
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
    elif not actions_routed and not use_dialog_draft_success and _is_generic_ack(reply_text):
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

    if should_plan and slots is not None and not use_dialog_draft_success:
        safe_slots = _fill_defaults(slots)
        missing = _missing_fields(safe_slots)
        if not _valid_date_range(getattr(safe_slots, "date_range", None)) and "出行日期" not in missing:
            missing.append("出行日期")
        if missing:
            if not actions_routed:
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
    elif should_plan and not use_dialog_draft_success:
        missing = ["目的地", "出行日期"]
        if not actions_routed:
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
    if suggested_quick_replies:
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in suggested_quick_replies:
            if item not in existing:
                existing.append(item)
    if actions_quick_replies:
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in actions_quick_replies:
            if item not in existing:
                existing.append(item)
    if use_dialog_draft_success and dialog_quick_replies:
        existing = resources_out.setdefault("quick_replies", [])
        if not isinstance(existing, list):
            existing = []
            resources_out["quick_replies"] = existing
        for item in dialog_quick_replies:
            if item not in existing:
                existing.append(item)
        resources_out["quick_replies"] = existing[:12]
    if region == "东北" and intent == "explore" and not use_dialog_draft_success:
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
    if use_dialog_draft_success:
        existing = resources_out.get("quick_replies")
        if isinstance(existing, list):
            resources_out["quick_replies"] = existing[:12]

    # ----- 8) 组装响应：resources_from_kb 和 trip_plan 一起返回 -----
    slot_completeness = SlotCompleteness(
        required_done=_estimate_required_done(slots),
        required_total=4,
    )
    has_trip_plan = trip_plan_result is not None
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
    elif should_plan and not actions_routed and not use_dialog_draft_success:
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
    if actions_routed or use_dialog_draft_success:
        next_action = NextAction(type="NONE", reason="actions_template")
        pending_questions = []
    if (
        not actions_routed
        and not use_dialog_draft_success
        and should_plan
        and next_action.type in {"ASK", "CALL_TRIP_PLAN", "REFINE_PLAN"}
    ):
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

    # ----- 防止内部 spots-json 泄漏到前端 -----
    if _looks_like_spots_json(reply_text):
        try:
            obj = json.loads(reply_text)
        except Exception:
            obj = None
        dest = getattr(slots, "destination", None) if slots is not None else (req.destination or "")
        if isinstance(obj, dict) and dest:
            reply_text = _render_spots_json_to_user(obj, dest)
        else:
            reply_text = ""

    reply_text = _enforce_v1_actions_style(reply_text)
    final_reply = reply_text or "这边现在有点忙，你可以稍后再试试。"
    final_history = history_with_new_user + [ChatMessage(role="assistant", content=final_reply)]
    if final_history and final_history[-1].role == "assistant" and final_history[-1].content != final_reply:
        logger.error("history_reply_mismatch trace_id=%s", trace_id)
    return TripChatResponse(
        reply=final_reply,
        history=final_history,
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
