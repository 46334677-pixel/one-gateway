from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from app.security import require_api_key


router = APIRouter(prefix="/v1", tags=["compare"], dependencies=[Depends(require_api_key)])


class CompareRequest(BaseModel):
    query: str = Field(description="商品搜索关键词，如 '婴童推车 可折叠 7kg 以下'")
    max_results: int = Field(default=5, ge=1, le=20, description="最多返回条数")
    dry_run: bool = Field(default=True, description="仅返回比价草案，不调用真实三方")
    confirm: bool = Field(default=False, description="确认后才调用真实三方")


class CompareItem(BaseModel):
    title: str
    price_cny: float
    source: str
    url: str


class CompareResponse(BaseModel):
    mode: str
    summary: str
    inputs: CompareRequest
    items: List[CompareItem]


@router.post("/compare", response_model=CompareResponse, summary="购物比价（dry-run 默认）")
def compare(req: CompareRequest):
    mode = "dry-run" if req.dry_run or not req.confirm else "confirmed"
    sample_items = [
        CompareItem(title="示例款 轻便婴童推车 A", price_cny=499, source="JD", url="https://example.com/jd/a"),
        CompareItem(title="示例款 轻便婴童推车 B", price_cny=469, source="Taobao", url="https://example.com/tb/b"),
        CompareItem(title="示例款 轻便婴童推车 C", price_cny=459, source="Pinduoduo", url="https://example.com/pdd/c"),
    ][: req.max_results]
    return CompareResponse(
        mode=mode,
        summary="示例比价结果（未接三方）。确认后可接 JD/淘宝/拼多多 等数据源。",
        inputs=req,
        items=sample_items,
    )

