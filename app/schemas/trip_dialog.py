# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field, ConfigDict


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
