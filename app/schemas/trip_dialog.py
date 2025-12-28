# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field, ConfigDict, field_validator


class DialogState(str, Enum):
    DISCOVERY = "DISCOVERY"
    PLAN_DRAFTING = "PLAN_DRAFTING"
    PLAN_PRESENTED = "PLAN_PRESENTED"
    REFINEMENT = "REFINEMENT"
    CLOSING = "CLOSING"


class PendingQuestionStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class NextActionType(str, Enum):
    ASK = "ASK"
    CALL_TRIP_PLAN = "CALL_TRIP_PLAN"
    REFINE_PLAN = "REFINE_PLAN"
    NONE = "NONE"


class TripProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    origin_city: Optional[str] = None
    destination: Optional[str] = None
    date_range: Optional[List[str]] = None
    days: Optional[int] = None
    traveler_count: Optional[int] = None
    budget_level: Optional[str] = None
    interest_tags: List[str] = Field(default_factory=list)
    pace_level: Optional[str] = None
    style: Optional[str] = None
    slot_sources: Optional[Dict[str, str]] = None


class PendingQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    question_id: str
    slot_key: str
    prompt: str
    status: PendingQuestionStatus = PendingQuestionStatus.OPEN
    asked_at: Optional[str] = None


class SlotCompleteness(BaseModel):
    required_done: int = 0
    required_total: int = 4


class NextAction(BaseModel):
    type: NextActionType = NextActionType.NONE
    reason: str = "init"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items or []:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class DialogOptionCard(BaseModel):
    model_config = ConfigDict(extra="allow")
    key: str
    title: str
    fit_for: str
    highlights: List[str]
    pace: List[str]
    pitfalls: List[str] = Field(default_factory=list)

    @field_validator("highlights")
    @classmethod
    def _validate_highlights(cls, v: List[str]) -> List[str]:
        items = _dedupe_keep_order(v)
        if len(items) < 3:
            raise ValueError("highlights must have at least 3 items")
        return items[:6]

    @field_validator("pace")
    @classmethod
    def _validate_pace(cls, v: List[str]) -> List[str]:
        items = _dedupe_keep_order(v)
        if len(items) < 2:
            raise ValueError("pace must have at least 2 items")
        return items[:4]

    @field_validator("pitfalls")
    @classmethod
    def _validate_pitfalls(cls, v: List[str]) -> List[str]:
        items = _dedupe_keep_order(v)
        return items[:3]


class DialogQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")
    key: str
    prompt: str
    options: List[str]

    @field_validator("options")
    @classmethod
    def _validate_options(cls, v: List[str]) -> List[str]:
        items = _dedupe_keep_order(v)
        if "我不确定" not in items:
            items.append("我不确定")
        if len(items) < 3:
            raise ValueError("question options must have at least 3 items")
        if len(items) > 6:
            head = [i for i in items if i != "我不确定"][:5]
            items = head + ["我不确定"]
        return items


class DialogDraft(BaseModel):
    model_config = ConfigDict(extra="allow")
    hook: str
    confirm: str
    options: List[DialogOptionCard]
    recommend_key: str
    recommend_reason: str
    questions: List[DialogQuestion]
    next_step: str
    quick_replies: List[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _validate_options(cls, v: List[DialogOptionCard]) -> List[DialogOptionCard]:
        if len(v) < 2 or len(v) > 4:
            raise ValueError("options must have 2-4 items")
        return v

    @field_validator("questions")
    @classmethod
    def _validate_questions(cls, v: List[DialogQuestion]) -> List[DialogQuestion]:
        if len(v) > 3:
            raise ValueError("questions must have at most 3 items")
        return v

    @field_validator("quick_replies")
    @classmethod
    def _validate_quick_replies(cls, v: List[str]) -> List[str]:
        items = _dedupe_keep_order(v)
        if "我不确定" not in items:
            items.append("我不确定")
        if len(items) > 12:
            head = [i for i in items if i != "我不确定"][:11]
            items = head + ["我不确定"]
        return items
