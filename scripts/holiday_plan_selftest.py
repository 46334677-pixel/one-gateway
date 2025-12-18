import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from app.routers.trip import TripPlanRequest, trip_plan  # noqa: E402


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main():
    req = TripPlanRequest(
        destination="大阪",
        date_range=[],
        adults=2,
        children=1,
        preferences=["亲子"],
        confirm=False,
    )
    plan = trip_plan(req)
    _assert(plan is not None, "trip_plan should return response")
    _assert(plan.mode != "need-info", "mode should not be need-info when date_range missing")
    _assert(plan.itinerary and len(plan.itinerary) >= 3, "itinerary should be generated")

    warnings = (plan.meta.warnings or []) if plan.meta else []
    _assert(any("未确认出行日期" in w for w in warnings), "meta.warnings should include date hint")

    joined = "\n".join(plan.itinerary or [])
    _assert("Day1" in joined and "Day2" in joined, "should contain Day1/Day2 lines")

    print("holiday_plan_selftest OK", {"mode": plan.mode, "warnings": warnings[:3]})


if __name__ == "__main__":
    main()
