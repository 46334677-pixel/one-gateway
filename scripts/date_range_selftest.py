import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from app.utils.date_range import extract_cn_date_range  # noqa: E402
from app.routers.trip import TripPlanRequest  # noqa: E402


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main():
    text = "2026年1月1日-1月2日出发，上海2天亲子美食"
    dr = extract_cn_date_range(text)
    _assert(dr is not None, "extract_cn_date_range should return non-empty dict")
    _assert(dr["start_date"] == "2026-01-01", "start_date mismatch")
    _assert(dr["end_date"] == "2026-01-02", "end_date mismatch")

    req = TripPlanRequest(origin="上海", destination="上海", date_range=[dr["start_date"], dr["end_date"]])
    _assert(req.date_range and len(req.date_range) == 2, "TripPlanRequest date_range should be set")

    print("date_range_selftest OK", dr)


if __name__ == "__main__":
    main()
