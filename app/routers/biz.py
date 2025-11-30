from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from app.security import require_api_key


router = APIRouter(prefix="/v1", tags=["biz"], dependencies=[Depends(require_api_key)])


class BizInsightRequest(BaseModel):
    city: str = Field(description="城市，如 '武汉'")
    industry: Optional[str] = Field(default="retail", description="行业，如 retail/food/fitness 等")
    area: Optional[str] = Field(default=None, description="区/商圈名称，可选")
    metrics: Optional[List[str]] = Field(default=None, description="关注指标，如 flow, heat, competition")
    dry_run: bool = Field(default=True, description="仅返回洞察草案，不调用真实三方")
    confirm: bool = Field(default=False, description="确认后才调用真实三方")


class BizInsightResponse(BaseModel):
    mode: str
    summary: str
    inputs: BizInsightRequest
    insights: List[str]
    actions: List[str]


@router.post("/biz_insight", response_model=BizInsightResponse, summary="本地商业策略洞察（dry-run 默认）")
def biz_insight(req: BizInsightRequest):
    mode = "dry-run" if req.dry_run or not req.confirm else "confirmed"
    insights = [
        "核心商圈周末亲子客流较高，上午 10-12 点达到峰值（示例）",
        "同类亲子活动集中于购物中心 A/B，存在差异化空间（示例）",
    ]
    actions = [
        "优先选择靠近地铁口的室内场地，便于家庭出行（示例）",
        "设置互动打卡点+小礼品，提高停留与分享率（示例）",
    ]
    return BizInsightResponse(
        mode=mode,
        summary="示例洞察（未接真实数据源）。确认后可接 POI/热度/竞品等数据。",
        inputs=req,
        insights=insights,
        actions=actions,
    )

