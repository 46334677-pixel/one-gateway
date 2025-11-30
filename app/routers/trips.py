# app/routers/trips.py
import os
import sqlite3
import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.security import require_api_key

DB_PATH = os.getenv("TRIPS_DB_PATH") or "/opt/one-gateway/trips.sqlite"
router = APIRouter(prefix="/v1", tags=["trips"], dependencies=[Depends(require_api_key)])


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
      user_id TEXT PRIMARY KEY,
      platform TEXT,
      openid TEXT,
      email TEXT,
      phone TEXT,
      extra TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS trips (
      trip_id TEXT PRIMARY KEY,
      user_id TEXT,
      destination TEXT,
      date_range TEXT,
      adults INTEGER,
      children INTEGER,
      budget_level TEXT,
      preferences TEXT,
      plan_json TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_id TEXT,
      user_id TEXT,
      rating INTEGER,
      comment TEXT,
      lat REAL,
      lng REAL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS checkins (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      poi_name TEXT,
      poi_id TEXT,
      lat REAL,
      lng REAL,
      comment TEXT,
      photos TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    return conn


# --------- models ----------
class UserUpsert(BaseModel):
    user_id: str
    platform: Optional[str] = "wechat"
    openid: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    extra: Optional[dict] = None


class TripCreate(BaseModel):
    trip_id: str
    user_id: str
    destination: str
    date_range: List[str] = []
    adults: int = 2
    children: int = 0
    budget_level: Optional[str] = None
    preferences: List[str] = []
    plan_json: dict


class TripQuery(BaseModel):
    user_id: str
    limit: int = 20
    offset: int = 0


class TripDetail(BaseModel):
    trip_id: str
    user_id: str


class FeedbackCreate(BaseModel):
    trip_id: str
    user_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class CheckinCreate(BaseModel):
    user_id: str
    poi_name: str
    poi_id: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    comment: Optional[str] = None
    photos: Optional[List[str]] = []


# --------- endpoints ----------
@router.post("/users/upsert")
def users_upsert(u: UserUpsert):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO users (user_id,platform,openid,email,phone,extra) VALUES (?,?,?,?,?,?)",
        (u.user_id, u.platform, u.openid, u.email, u.phone, json.dumps(u.extra or {})),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@router.post("/trips/create")
def trips_create(t: TripCreate):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO trips (trip_id,user_id,destination,date_range,adults,children,budget_level,preferences,plan_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            t.trip_id,
            t.user_id,
            t.destination,
            json.dumps(t.date_range),
            t.adults,
            t.children,
            t.budget_level,
            json.dumps(t.preferences),
            json.dumps(t.plan_json),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@router.post("/trips/list")
def trips_list(q: TripQuery):
    conn = _conn()
    cur = conn.execute(
        "SELECT trip_id,destination,date_range,adults,children,budget_level,preferences,plan_json,created_at "
        "FROM trips WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (q.user_id, q.limit, q.offset),
    )
    rows = cur.fetchall()
    conn.close()
    trips = []
    for r in rows:
        trips.append(
            {
                "trip_id": r[0],
                "destination": r[1],
                "date_range": json.loads(r[2] or "[]"),
                "adults": r[3],
                "children": r[4],
                "budget_level": r[5],
                "preferences": json.loads(r[6] or "[]"),
                "plan_json": json.loads(r[7] or "{}"),
                "created_at": r[8],
            }
        )
    return {"items": trips}


@router.post("/trips/detail")
def trips_detail(q: TripDetail):
    conn = _conn()
    cur = conn.execute(
        "SELECT trip_id,destination,date_range,adults,children,budget_level,preferences,plan_json,created_at "
        "FROM trips WHERE user_id=? AND trip_id=?",
        (q.user_id, q.trip_id),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return {
        "trip_id": row[0],
        "destination": row[1],
        "date_range": json.loads(row[2] or "[]"),
        "adults": row[3],
        "children": row[4],
        "budget_level": row[5],
        "preferences": json.loads(row[6] or "[]"),
        "plan_json": json.loads(row[7] or "{}"),
        "created_at": row[8],
    }


@router.post("/trips/feedback")
def trips_feedback(fb: FeedbackCreate):
    if fb.rating < 1 or fb.rating > 5:
        raise HTTPException(status_code=400, detail="rating 1-5")
    conn = _conn()
    conn.execute(
        "INSERT INTO feedback (trip_id,user_id,rating,comment,lat,lng) VALUES (?,?,?,?,?,?)",
        (fb.trip_id, fb.user_id, fb.rating, fb.comment, fb.lat, fb.lng),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@router.post("/trips/checkin")
def trips_checkin(ci: CheckinCreate):
    conn = _conn()
    conn.execute(
        "INSERT INTO checkins (user_id,poi_name,poi_id,lat,lng,comment,photos) VALUES (?,?,?,?,?,?,?)",
        (
            ci.user_id,
            ci.poi_name,
            ci.poi_id,
            ci.lat,
            ci.lng,
            ci.comment,
            json.dumps(ci.photos or []),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}
