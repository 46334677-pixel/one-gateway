# scripts/import_kb.py
import csv
import json
import os
import sys
from typing import List

import requests

API_URL = os.environ.get("KB_API_URL", "http://127.0.0.1:8000/v1/kb_ingest")
API_KEY = os.environ.get("KB_API_KEY", "change_811202ts")  # 填你的 X-Api-Key

def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_xlsx(path: str) -> List[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("需要 pandas 才能读取 xlsx，请 pip install pandas 或改用 CSV")
        sys.exit(1)
    df = pd.read_excel(path)
    return df.to_dict(orient="records")

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/import_kb.py data.csv|data.xlsx")
        sys.exit(1)
    path = sys.argv[1]
    if path.lower().endswith(".csv"):
        rows = load_csv(path)
    elif path.lower().endswith((".xls", ".xlsx")):
        rows = load_xlsx(path)
    else:
        print("只支持 csv/xlsx")
        sys.exit(1)

    headers = {"X-Api-Key": API_KEY}
    for r in rows:
        payload = {
            "url": r.get("url") or "",
            "destination": r.get("destination") or "",
            "title": r.get("title") or r.get("name") or "",
            "platform": r.get("platform") or "kb",
            "tags": [t.strip() for t in (r.get("tags") or "").split(",") if t.strip()],
            "photos": [p.strip() for p in (r.get("photos") or "").split(",") if p.strip()],
            "raw_text": r.get("raw_text") or "",
        }
        resp = requests.post(API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"导入失败 {payload['url']}: {resp.status_code} {resp.text}")
        else:
            print(f"导入成功 {payload['url']}")
    print("完成")

if __name__ == "__main__":
    main()
