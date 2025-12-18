import datetime as dt
import re
from typing import Dict, Optional, Tuple


def extract_cn_date_range(text: str) -> Optional[Dict[str, str]]:
    """
    Deterministic CN date range extractor (priority over LLM).
    Supports:
      - 2026年1月1日-1月2日
      - 2026年1月1日-2026年1月2日
      - 2026-01-01 到 2026-01-02
    Separators: - — – ~ ～ 到 至
    Returns: {"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}
    """
    if not text:
        return None

    s = str(text)
    sep = r"(?:\s*(?:-|—|–|~|～|到|至)\s*)"

    # 2026年1月1日-1月2日 (end year optional)
    m = re.search(
        rf"(\d{{4}})\s*年\s*(\d{{1,2}})\s*月\s*(\d{{1,2}})\s*(?:日|号)?{sep}(?:(\d{{4}})\s*年\s*)?(\d{{1,2}})\s*月\s*(\d{{1,2}})\s*(?:日|号)?",
        s,
    )
    if m:
        y1, mo1, d1, y2, mo2, d2 = m.groups()
        y2 = y2 or y1
        try:
            start = dt.date(int(y1), int(mo1), int(d1))
            end = dt.date(int(y2), int(mo2), int(d2))
        except Exception:
            return None
        if end < start:
            start, end = end, start
        return {"start_date": start.isoformat(), "end_date": end.isoformat()}

    # 2026-01-01 到 2026-01-02 (also allow / . as separators)
    m2 = re.search(
        rf"(\d{{4}})[-/\.](\d{{1,2}})[-/\.](\d{{1,2}}){sep}(\d{{4}})[-/\.](\d{{1,2}})[-/\.](\d{{1,2}})",
        s,
    )
    if m2:
        y1, mo1, d1, y2, mo2, d2 = m2.groups()
        try:
            start = dt.date(int(y1), int(mo1), int(d1))
            end = dt.date(int(y2), int(mo2), int(d2))
        except Exception:
            return None
        if end < start:
            start, end = end, start
        return {"start_date": start.isoformat(), "end_date": end.isoformat()}

    return None


_CN_NUM = {
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


def _infer_duration_days(text: str) -> Optional[int]:
    if not text:
        return None
    s = str(text)
    m = re.search(r"(\d+)\s*(?:天|日)", s)
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(n, 10))
        except Exception:
            return None
    m2 = re.search(r"([一二两三四五六七八九十])\s*(?:天|日)", s)
    if m2:
        return _CN_NUM.get(m2.group(1))
    return None


def _next_fixed_date(month: int, day: int, now: dt.datetime) -> dt.date:
    y = now.year
    d0 = dt.date(y, month, day)
    if d0 < now.date():
        d0 = dt.date(y + 1, month, day)
    return d0


def _next_weekend(now: dt.datetime, next_week: bool) -> Tuple[dt.date, dt.date]:
    # Saturday=5, Sunday=6
    wd = now.weekday()
    days_until_sat = (5 - wd) % 7
    if next_week:
        days_until_sat += 7 if days_until_sat == 0 else 7
    start = now.date() + dt.timedelta(days=days_until_sat)
    end = start + dt.timedelta(days=1)
    return start, end


def infer_cn_date_range_from_text(
    text: str, now: Optional[dt.datetime] = None
) -> Optional[Dict[str, str]]:
    """
    Natural-language date inference fallback (low confidence).

    Returns {"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"} or None.

    Supported keywords:
      - 元旦 -> next 01-01
      - 五一/劳动节 -> next 05-01
      - 国庆 -> next 10-01
      - 周末/本周末/下周末 -> infer Saturday-Sunday

    If "X天" exists, uses it to extend end_date (end = start + X-1 days).
    """
    if not text:
        return None
    now = now or dt.datetime.now()
    t = str(text).strip()
    if not t:
        return None

    duration = _infer_duration_days(t) or 1

    start: Optional[dt.date] = None
    end: Optional[dt.date] = None

    if "元旦" in t:
        start = _next_fixed_date(1, 1, now)
        end = start + dt.timedelta(days=duration - 1)
    elif ("五一" in t) or ("劳动节" in t):
        start = _next_fixed_date(5, 1, now)
        end = start + dt.timedelta(days=duration - 1)
    elif "国庆" in t:
        start = _next_fixed_date(10, 1, now)
        end = start + dt.timedelta(days=duration - 1)
    elif ("下周末" in t) or ("周末" in t) or ("本周末" in t):
        start, end = _next_weekend(now, next_week=("下周末" in t))
        if duration > 2:
            end = start + dt.timedelta(days=duration - 1)

    if not start or not end:
        return None

    if end < start:
        start, end = end, start
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}
