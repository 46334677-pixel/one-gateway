from fastapi import FastAPI

from app.settings import settings
from app.routers import trip_chat_cn
from app.routers import asr_stream
from app.routers import trip as trip_router
from app.routers import compare as compare_router
from app.routers import biz as biz_router
from app.routers import trips as trips_router  # 新增

app = FastAPI(
    title="OneGateway",
    version="0.1.0",
    description="Trip/Compare/Biz aggregation gateway",
)

app.include_router(trip_chat_cn.router)
app.include_router(asr_stream.router)
# 新增行程/用户/反馈/打卡接口
app.include_router(trips_router.router)

if settings.FEATURE_TRIP:
    app.include_router(trip_router.router)
if settings.FEATURE_COMPARE:
    app.include_router(compare_router.router)
if settings.FEATURE_BIZ:
    app.include_router(biz_router.router)


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"ok": True}
