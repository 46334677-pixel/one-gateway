# app/routers/kb.py
import os, json, math, sqlite3, hashlib
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.security import require_api_key

DB_PATH = os.getenv("KB_DB_PATH") or "/root/one-gateway/kb.sqlite"
router = APIRouter(prefix="/v1", tags=["kb"], dependencies=[Depends(require_api_key)])

def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS kb (
        id TEXT PRIMARY KEY,
        destination TEXT,
        title TEXT,
        source_url TEXT,
        platform TEXT,
        summary TEXT,
        tags TEXT,
        photos TEXT,
        embedding TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    return conn

def _embedding(text: str) -> List[float]:
    h = hashlib.md5(text.encode("utf-8")).digest()
    return [(b / 255.0) for b in h]

def _sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-9)

class KbIngest(BaseModel):
    url: str
    destination: Optional[str] = ""
    title: Optional[str] = ""
    platform: Optional[str] = "kb"
    tags: Optional[List[str]] = []
    photos: Optional[List[str]] = []
    raw_text: Optional[str] = ""

class KbSearch(BaseModel):
    query: str
    destination: Optional[str] = ""
    top_k: int = 5

@router.post("/kb_ingest")
def kb_ingest(item: KbIngest):
    conn = _get_conn()
    text = item.raw_text or item.title or item.url
    emb = _embedding(text)
    conn.execute(
        "INSERT OR REPLACE INTO kb (id,destination,title,source_url,platform,summary,tags,photos,embedding) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            hashlib.md5(item.url.encode("utf-8")).hexdigest(),
            item.destination or "",
            item.title or item.url,
            item.url,
            item.platform,
            item.raw_text[:500] if item.raw_text else "",
            json.dumps(item.tags or []),
            json.dumps(item.photos or []),
            json.dumps(emb),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@router.post("/kb_search")
def kb_search(req: KbSearch):
    conn = _get_conn()
    cur = conn.execute("SELECT id,destination,title,source_url,platform,summary,tags,photos,embedding FROM kb")
    rows = cur.fetchall()
    conn.close()
    q_emb = _embedding(req.query + (req.destination or ""))
    scored = []
    for r in rows:
        emb = json.loads(r[8]) if r[8] else []
        s = _sim(q_emb, emb)
        if req.destination and req.destination in (r[1] or ""):
            s += 0.1
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for s, r in scored[: req.top_k]:
        results.append({
            "id": r[0],
            "destination": r[1],
            "title": r[2],
            "url": r[3],
            "platform": r[4],
            "summary": r[5],
            "tags": json.loads(r[6]) if r[6] else [],
            "photos": json.loads(r[7]) if r[7] else [],
            "score": s,
        })
    return {"items": results}
