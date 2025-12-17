import datetime as dt
import re
from typing import Dict, Optional


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

