import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.security import require_api_key

router = APIRouter(prefix="/v1", tags=["trip"], dependencies=[Depends(require_api_key)])


class TripPlanRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="用户 ID，用于简单偏好记忆")
    origin: Optional[str] = Field(default=None, description="出发地")
    destination: Optional[str] = Field(default=None, description="目的地（可为空，表示本地游）")
    date_range: Optional[List[str]] = Field(default=None, description="日期范围 [开始, 结束]，YYYY-MM-DD")

    people_count: Optional[int] = Field(default=None, description="总人数")
    adults: Optional[int] = Field(default=None, description="成人人数")
    children: Optional[int] = Field(default=None, description="儿童人数")
    elders: Optional[int] = Field(default=None, description="老人/长者人数")

    interests: Optional[List[str]] = Field(default=None, description="兴趣偏好（旧字段）")
    preferences: Optional[List[str]] = Field(default=None, description="兴趣偏好（新字段）")
    pace: Optional[str] = Field(default=None, description="行程节奏：relaxed/normal/tight 等")

    budget_cny: Optional[int] = Field(default=None, description="预算（元）")
    budget_level: Optional[str] = Field(default=None, description="预算档位：low/medium/high")
    notes: Optional[str] = Field(default=None, description="其他备注需求")

    user_lat: Optional[float] = Field(default=None, description="用户当前纬度")
    user_lng: Optional[float] = Field(default=None, description="用户当前经度")

    dry_run: bool = Field(default=True, description="草案开关")
    confirm: bool = Field(default=False, description="确认开关")


class ResourceItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    opentime: Optional[str] = None
    url: Optional[str] = None
    photo: Optional[str] = None
    source: Optional[str] = None


class TripResources(BaseModel):
    tickets: Optional[List[ResourceItem]] = None
    foods: Optional[List[ResourceItem]] = None
    hotels: Optional[List[ResourceItem]] = None
    search: Optional[List[ResourceItem]] = None


# 新增可选模型
class Segment(BaseModel):
    day: int
    from_poi: str
    to_poi: str
    distance_km: Optional[float] = None
    duration_min: Optional[float] = None
    mode: Optional[str] = None  # drive/walk/public


class PoiInfo(BaseModel):
    name: str
    desc: Optional[str] = None
    ticket_tips: Optional[str] = None
    url: Optional[str] = None
    photo: Optional[str] = None


class Packages(BaseModel):
    flight_hotel: Optional[str] = None
    free_trip: Optional[str] = None
    tips: Optional[str] = None


class ReverseGeocodeRequest(BaseModel):
    lat: float = Field(..., description="Latitude (GCJ-02)")
    lng: float = Field(..., description="Longitude (GCJ-02)")


class ReverseGeocodeResponse(BaseModel):
    city_name: Optional[str] = Field(None, description="城市名称，例如 武汉市")
    province: Optional[str] = None
    district: Optional[str] = None
    adcode: Optional[str] = None
    formatted_address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None  # 原始响应，可选


class TripPlanResponse(BaseModel):
    mode: str
    summary: str
    inputs: TripPlanRequest
    itinerary: List[str]
    resources: Optional[TripResources] = None
    opening_list: Optional[List[Dict[str, Any]]] = None
    weather_daily: Optional[List[Dict[str, Any]]] = None
    debug: Optional[Dict[str, Any]] = None
    # 新增可选字段
    route_map_url: Optional[str] = None
    segments: Optional[List[Segment]] = None
    poi_info: Optional[List[PoiInfo]] = None
    packages: Optional[Packages] = None


class UserMemory(BaseModel):
    last_destinations: List[str] = []
    preferences: List[str] = []
    last_budget: Optional[int] = None


USER_MEMORY: Dict[str, UserMemory] = {}


def _http_get_json(base_url: str, params: Dict[str, str], timeout: float = 5.0):
    try:
        qs = urllib.parse.urlencode(params)
        url = f"{base_url}?{qs}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


def _reverse_geocode_gaode(lat: float, lng: float) -> Optional[ReverseGeocodeResponse]:
    """
    使用高德逆地理编码接口将经纬度解析为城市/省份等信息。
    文档示例：https://restapi.amap.com/v3/geocode/regeo
    """
    key = os.getenv("GAODE_KEY") or os.getenv("AMAP_KEY")
    if not key:
        return None

    data = _http_get_json(
        "https://restapi.amap.com/v3/geocode/regeo",
        {
            "key": key,
            "location": f"{lng},{lat}",  # 高德要求：lng,lat
            "radius": "1000",
            "extensions": "base",
        },
        timeout=5.0,
    )
    if not data or data.get("status") != "1":
        return None

    regeocode = data.get("regeocode") or {}
    comp = regeocode.get("addressComponent") or {}

    city = comp.get("city")
    # 直辖市等 city 可能是空字符串或列表
    if isinstance(city, list):
        city = city[0] if city else None
    if not city:
        city = comp.get("province")

    province = comp.get("province")
    # TODO: 如果需要，可在这里补充 country 等字段映射
    district = comp.get("district")
    adcode = comp.get("adcode")
    formatted_address = regeocode.get("formatted_address")

    return ReverseGeocodeResponse(
        city_name=city,
        province=province,
        district=district,
        adcode=adcode,
        formatted_address=formatted_address,
        lat=lat,
        lng=lng,
        raw=data,
    )


def _need_more_info(req: TripPlanRequest) -> List[str]:
    missing: List[str] = []
    if not req.destination:
        missing.append("目的地")
    if not req.date_range:
        missing.append("出行日期")
    if req.adults is None and req.people_count is None:
        missing.append("出行人数")
    if not req.origin:
        missing.append("出发地")
    return missing


def _update_user_memory(req: TripPlanRequest) -> Optional[UserMemory]:
    if not req.user_id:
        return None
    memory = USER_MEMORY.get(req.user_id) or UserMemory()

    if req.destination and req.destination not in memory.last_destinations:
        memory.last_destinations.append(req.destination)

    prefs = (req.preferences or []) + (req.interests or [])
    for p in prefs:
        if p and p not in memory.preferences:
            memory.preferences.append(p)

    if req.budget_cny:
        memory.last_budget = req.budget_cny

    USER_MEMORY[req.user_id] = memory
    return memory


def _infer_days(date_range: Optional[List[str]]) -> int:
    if not date_range:
        return 2
    try:
        if len(date_range) == 1:
            return 1
        start = dt.date.fromisoformat(date_range[0])
        end = dt.date.fromisoformat(date_range[-1])
        days = (end - start).days + 1
        return max(1, min(days, 10))
    except Exception:
        return 2


def _fetch_weather_daily(req: TripPlanRequest) -> Optional[List[Dict[str, Any]]]:
    key = os.getenv("WEATHER_KEY")
    destination = req.destination
    if not key or not destination:
        return None

    geo = _http_get_json(
        "https://geoapi.qweather.com/v2/city/lookup",
        {"location": destination, "key": key},
    )
    if not geo or geo.get("code") != "200":
        return None

    locations = geo.get("location") or []
    if not locations:
        return None
    location_id = locations[0].get("id")
    if not location_id:
        return None

    weather = _http_get_json(
        "https://devapi.qweather.com/v7/weather/3d",
        {"location": location_id, "key": key},
    )
    if not weather or weather.get("code") != "200":
        return None

    daily = weather.get("daily") or []
    if not daily:
        return None

    results: List[Dict[str, Any]] = []
    for item in daily[:3]:
        results.append(
            {
                "date": item.get("fxDate"),
                "text_day": item.get("textDay"),
                "text_night": item.get("textNight"),
                "temp_min": item.get("tempMin"),
                "temp_max": item.get("tempMax"),
                "wind_dir_day": item.get("windDirDay"),
                "wind_scale_day": item.get("windScaleDay"),
            }
        )
    return results or None


def _fetch_poi_list(destination: Optional[str], types: str, keywords: str) -> Optional[List[Dict[str, Any]]]:
    key = os.getenv("GAODE_KEY") or os.getenv("AMAP_KEY")
    if not key or not destination:
        return None

    data = _http_get_json(
        "https://restapi.amap.com/v3/place/text",
        {
            "key": key,
            "keywords": keywords,
            "types": types,
            "city": destination,
            "page": 1,
            "offset": 5,
            "extensions": "all",
        },
    )
    if not data or data.get("status") != "1":
        return None
    pois = data.get("pois") or []
    return pois or None


def _fetch_opening_list(destination: Optional[str]) -> Optional[List[Dict[str, str]]]:
    pois = _fetch_poi_list(destination, types="110000", keywords="景点|景区")
    if not pois:
        return None

    results: List[Dict[str, str]] = []
    for poi in pois[:5]:
        name = poi.get("name") or ""
        if not name:
            continue
        opentime = (
            poi.get("opentime")
            or poi.get("open_time")
            or poi.get("business_time")
            or ""
        )
        address = poi.get("address") or poi.get("adname") or ""
        poi_id = poi.get("id") or poi.get("parent")
        url = f"https://www.amap.com/place/{poi_id}" if poi_id else None
        photo = None
        photos = poi.get("photos") or []
        if photos and photos[0].get("url"):
            photo = photos[0].get("url")
        results.append(
            {
                "name": name,
                "address": address,
                "opentime": opentime or "营业时间以实际公示信息为准（请在高德/携程再次确认）",
                "url": url,
                "photo": photo,
                "source": "gaode",
            }
        )

    return results or None


def _fetch_food_list(destination: Optional[str]) -> Optional[List[ResourceItem]]:
    pois = _fetch_poi_list(destination, types="050000", keywords="美食|小吃")
    if not pois:
        return None

    items: List[ResourceItem] = []
    for poi in pois[:5]:
        name = poi.get("name") or ""
        if not name:
            continue
        opentime = (
            poi.get("opentime")
            or poi.get("open_time")
            or poi.get("business_time")
            or ""
        )
        address = poi.get("address") or poi.get("adname") or ""
        poi_id = poi.get("id") or poi.get("parent")
        url = f"https://www.amap.com/place/{poi_id}" if poi_id else None
        photo = None
        photos = poi.get("photos") or []
        if photos and photos[0].get("url"):
            photo = photos[0].get("url")
        items.append(
            ResourceItem(
                name=name,
                address=address,
                opentime=opentime or "营业时间以实际公示信息为准（请在高德/携程再次确认）",
                url=url,
                photo=photo,
                source="gaode",
            )
        )
    return items or None


def _build_resources(destination: str, opening_list: Optional[List[Dict[str, str]]], foods: Optional[List[ResourceItem]]):
    affiliate_base = (
        os.getenv("CTRIP_AFFILIATE_URL") or "https://www.trip.com/t/jigG65FsJS2"
    )
    quoted_dest = urllib.parse.quote(destination)

    tickets = []
    if opening_list:
        tickets = [
            ResourceItem(
                name=item.get("name"),
                address=item.get("address"),
                opentime=item.get("opentime"),
                url=item.get("url"),
                photo=item.get("photo"),
                source=item.get("source"),
            )
            for item in opening_list
        ]

    hotels = [
        ResourceItem(
            name=f"{destination} 酒店参考搜索",
            url=f"{affiliate_base}#/hotels?city={quoted_dest}",
            source="ctrip",
        )
    ]
    search = [
        ResourceItem(
            name=f"{destination} 门票/景点搜索",
            url=f"{affiliate_base}#/tickets?city={quoted_dest}",
            source="ctrip",
        ),
        ResourceItem(
            name=f"{destination} 综合攻略搜索",
            url=f"{affiliate_base}?keyword={quoted_dest}",
            source="ctrip",
        ),
    ]

    return TripResources(tickets=tickets or None, foods=foods, hotels=hotels, search=search)


def _build_itinerary(
    req: TripPlanRequest,
    memory: Optional[UserMemory],
    opening_list: Optional[List[Dict[str, str]]],
    weather_daily: Optional[List[Dict[str, Any]]],
) -> List[str]:
    destination = req.destination or "本地周边"
    days = _infer_days(req.date_range)

    adults = req.adults or (req.people_count or 2)
    children = req.children or 0

    prefs = req.preferences or req.interests or []
    is_family = any(p in ("kids", "亲子", "family") for p in prefs)
    is_food = any(p in ("food", "美食") for p in prefs)

    itinerary: List[str] = []
    poi_names = [p["name"] for p in (opening_list or []) if p.get("name")]

    for i in range(days):
        day_label = f"Day {i + 1}：{destination}第 {i + 1} 天"
        itinerary.append(day_label)

        poi_hint = ""
        if poi_names:
            poi_hint = f"（可选：{poi_names[i % len(poi_names)]} 等）"

        if is_family:
            morning = (
                "上午：安排亲子友好的主题乐园 / 动物园 / 亲子博物馆，控制在 2-3 小时，预留午睡时间"
                f"{poi_hint}。"
            )
        else:
            morning = (
                "上午：安排城市地标景点或自然风光路线，步行与交通时间控制在 3 小时内"
                f"{poi_hint}。"
            )

        if is_food:
            noon = "中午：选择当地口碑较好的特色餐厅，就近就餐，注意提前错峰或预约。"
        else:
            noon = "中午：就近简餐或商场餐饮，保证休息与补水。"

        if is_family:
            afternoon = (
                "下午：选择节奏更轻松的项目，如沿河/海边散步、公园遛娃或轻量级室内乐园，"
                "尽量避免午后最晒的时段长时间户外暴晒。"
            )
        else:
            afternoon = (
                "下午：结合天气安排室外/室内穿插，如遇高温/下雨，可改为博物馆、购物中心等室内行程。"
            )

        evening = (
            "晚上：视体力安排轻量夜景/夜市，如携带儿童建议 21 点前回到酒店休息。"
        )

        itinerary.append(f"- {morning}")
        itinerary.append(f"- {noon}")
        itinerary.append(f"- {afternoon}")
        itinerary.append(f"- {evening}")

    # 营业时间 & 天气提醒（仅有日期时详细提示）
    if req.date_range:
        if opening_list:
            itinerary.append("【开放 & 营业时间提醒】")
            for item in opening_list[:5]:
                itinerary.append(
                    f"- {item.get('name','')}: {item.get('opentime','营业时间以实际公示为准')}（参考地址：{item.get('address','未知')}）"
                )
        else:
            itinerary.append(
                "- 未拿到具体营业时间，请在出发前查看高德/携程或官方公众号，再次确认是否临时闭园/调休。"
            )

        if weather_daily:
            first = weather_daily[0]
            desc = (
                f"{first.get('date','')} 白天{first.get('text_day','未知')}，夜间{first.get('text_night','未知')}，"
                f"气温约 {first.get('temp_min','?')}~{first.get('temp_max','?')}℃。"
            )
            itinerary.append("【天气提醒】")
            itinerary.append(f"- {desc} 请根据体感增减衣物，预留室内备选。")
        else:
            itinerary.append(
                "- 未拿到准确天气，出发前 1-2 天请再查和风天气/手机天气 App，遇高温/暴雨/台风时调整为室内行程。"
            )
    else:
        itinerary.append(
            "- 暂未确认具体出行日期，建议出发前再查看高德/携程营业时间与天气预报。"
        )

    # 海边 / 台风安全提醒
    itinerary.append(
        "- 如行程包含海边/海岛，务必关注台风路径和海况预警，避免在强风浪或禁泳时段下水。"
    )

    if memory and memory.last_destinations:
        visited = "、".join(memory.last_destinations[-3:])
        itinerary.append(
            f"- 根据你最近的出行记录（例如：{visited}），本次行程整体节奏会偏向 "
            f"{req.pace or '适中'}，如希望更放松或更紧凑，可以在对话中直接说。"
        )

    return itinerary


@router.post(
    "/trip_plan", response_model=TripPlanResponse, summary="出行规划（含营业时间/天气提示）"
)
def trip_plan(req: TripPlanRequest):
    mode = "dry-run" if req.dry_run or not req.confirm else "confirmed"

    # 必要信息不全时直接提示
    missing = _need_more_info(req)
    if missing:
        summary = "需要更多信息后再生成行程：" + "、".join(missing)
        return TripPlanResponse(
            mode="need-info",
            summary=summary,
            inputs=req,
            itinerary=[
                "请补充：出发地、目的地、出行日期（范围）、出行人数和大致偏好/节奏；如有儿童请说明年龄段。"
            ],
            resources=None,
            opening_list=None,
            weather_daily=None,
            debug={"missing": missing},
        )

    # 更新并获取用户记忆（仅内存级，进程重启后会丢失）
    memory = _update_user_memory(req)

    # 获取天气和营业时间信息（失败时自动降级为文案提醒）
    weather_daily = _fetch_weather_daily(req) if req.date_range else None
    opening_list = _fetch_opening_list(req.destination)
    foods = _fetch_food_list(req.destination)
    resources = _build_resources(req.destination or "本地周边", opening_list, foods)

    itinerary = _build_itinerary(req, memory, opening_list, weather_daily)

    # 生成摘要
    days = _infer_days(req.date_range)
    destination = req.destination or "本地周边"
    adults = req.adults or (req.people_count or 2)
    children = req.children or 0

    summary_parts = [
        f"{destination}{days} 天行程草案",
        f"出行人：{adults} 大 {children} 小",
    ]
    if req.budget_level:
        summary_parts.append(f"预算档位：{req.budget_level}")
    elif req.budget_cny:
        summary_parts.append(f"预算约 {req.budget_cny} 元")
    if req.pace:
        summary_parts.append(f"行程节奏：{req.pace}")
    if memory and memory.preferences:
        summary_parts.append("偏好记忆：" + "、".join(memory.preferences[-5:]))

    summary = "；".join(summary_parts) + "。"

    debug = {
        "gaode_key": bool(os.getenv("GAODE_KEY") or os.getenv("AMAP_KEY")),
        "weather_key": bool(os.getenv("WEATHER_KEY")),
        "opening_count": len(opening_list or []),
        "weather_days": len(weather_daily or []),
        "food_count": len(foods or []),
        "has_photos_food": any((f.photo for f in (foods or []))),
        "has_photos_spot": any((p.get("photo") for p in (opening_list or []))),
    }

    # 占位构造：路线图/分段距离/门票信息/携程深链
    segments = []
    if opening_list and len(opening_list) >= 2:
        for i in range(len(opening_list) - 1):
            segments.append(
                Segment(
                    day=1,
                    from_poi=opening_list[i].get("name", ""),
                    to_poi=opening_list[i + 1].get("name", ""),
                    mode="drive",
                )
            )
    poi_info = []
    if opening_list:
        for item in opening_list[:5]:
            poi_info.append(
                PoiInfo(
                    name=item.get("name", ""),
                    desc="适合半天，行前关注营业时间/预约",
                    ticket_tips="可在高德/携程搜索，关注夜场票/学生票等优惠",
                    url=item.get("url"),
                    photo=item.get("photo"),
                )
            )
    affiliate_base = os.getenv("CTRIP_AFFILIATE_URL") or "https://www.trip.com/t/jigG65FsJS2"
    dep = urllib.parse.quote(req.origin or "出发地")
    arr = urllib.parse.quote(req.destination or "目的地")
    packages = Packages(
        flight_hotel=f"{affiliate_base}#/packageland?dep={dep}&arr={arr}",
        free_trip=f"{affiliate_base}#/packageland?dep={dep}&arr={arr}&type=free",
        tips="出发地填你的城市 → 目的地选机票+酒店或自由行 → 选3-4晚，看系统推荐酒店；航班尽量中午后到、次日午后返。",
    )

    return TripPlanResponse(
        mode=mode,
        summary=summary,
        inputs=req,
        itinerary=itinerary,
        resources=resources,
        opening_list=opening_list,
        weather_daily=weather_daily,
        route_map_url="https://via.placeholder.com/600x400.png?text=Route+Map",  # 可替换为高德静态地图
        segments=segments or None,
        poi_info=poi_info or None,
        packages=packages,
        debug=debug,
    )


@router.post("/reverse_geocode", response_model=ReverseGeocodeResponse)
def reverse_geocode(req: ReverseGeocodeRequest) -> ReverseGeocodeResponse:
    """
    小程序调用的逆地理编码接口。
    输入 GCJ-02 坐标，返回城市名称等基础信息。
    """
    result = _reverse_geocode_gaode(req.lat, req.lng)
    if result is None:
        # 调用失败时也返回 200，只是字段为空，避免前端崩溃
        return ReverseGeocodeResponse(
            city_name=None,
            province=None,
            district=None,
            adcode=None,
            formatted_address=None,
            lat=req.lat,
            lng=req.lng,
            raw=None,
        )
    return result


@router.get("/reverse_geocode", response_model=ReverseGeocodeResponse)
def reverse_geocode_get(
    lat: float = Query(..., description="Latitude (GCJ-02)"),
    lng: float = Query(..., description="Longitude (GCJ-02)"),
) -> ReverseGeocodeResponse:
    """
    兼容 GET 方式的调用（通过 query 传 lat/lng），内部复用 POST 核心逻辑。
    """
    req = ReverseGeocodeRequest(lat=lat, lng=lng)
    return reverse_geocode(req)
