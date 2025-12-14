import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from app.routers.trip import (  # noqa: E402
    ResourceItem,
    TripPlanRequest,
    TripPlanResponse,
    TripResources,
    postProcessTripPlan,
)


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def test_fill_evening():
    req = TripPlanRequest(origin="上海", destination="杭州", date_range=["2025-01-01", "2025-01-01"], preferences=[])
    plan = TripPlanResponse(
        mode="dry-run",
        summary="test",
        inputs=req,
        itinerary=["Day1 上午：西湖", "Day1 下午：灵隐寺"],
        resources=TripResources(tickets=[ResourceItem(name="西湖"), ResourceItem(name="灵隐寺")]),
    )
    out = postProcessTripPlan(plan, req)
    _assert(any("Day1 晚上" in x for x in out.itinerary), "should fill evening segment")
    _assert(out.meta and out.meta.fixed and "补齐三段" in out.meta.fixed, "meta.fixed should include 补齐三段")


def test_trim_too_many_items():
    req = TripPlanRequest(origin="上海", destination="上海", date_range=["2025-01-01", "2025-01-01"], preferences=[])
    plan = TripPlanResponse(
        mode="dry-run",
        summary="test",
        inputs=req,
        itinerary=["Day1 上午：A、B、C、D、E、F", "Day1 下午：G", "Day1 晚上：H"],
        resources=TripResources(),
    )
    out = postProcessTripPlan(plan, req)
    morning = next((x for x in out.itinerary if "Day1 上午：" in x), "")
    content = morning.split("：", 1)[1] if "：" in morning else ""
    items = [x for x in content.split("、") if x]
    _assert(len(items) <= 3, "morning should be trimmed to <=3 items")
    _assert(out.meta and out.meta.warnings and any("已精简" in w for w in out.meta.warnings), "should have trim warning")


def test_preference_family_in_summary_or_meta():
    req = TripPlanRequest(
        origin="上海",
        destination="上海",
        date_range=["2025-01-01", "2025-01-01"],
        preferences=["亲子"],
    )
    plan = TripPlanResponse(
        mode="dry-run",
        summary="test",
        inputs=req,
        itinerary=["Day1 上午：外滩", "Day1 下午：豫园"],
        resources=TripResources(),
    )
    out = postProcessTripPlan(plan, req)
    _assert("亲子" in out.summary or (out.meta and out.meta.fixed and any("偏好" in x for x in out.meta.fixed)), "should mention family preference")


if __name__ == "__main__":
    test_fill_evening()
    test_trim_too_many_items()
    test_preference_family_in_summary_or_meta()
    print("postprocess_selftest OK")

