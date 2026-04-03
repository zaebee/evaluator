from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class NormalizedEvent(BaseModel):
    event_id: str
    session_id: str
    timestamp: datetime
    role: Literal["user", "assistant", "system"]
    event_type: Literal["tool_call", "tool_result", "message", "user_input"]
    tool_name: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionTrace(BaseModel):
    session_id: str
    model: str
    started_at: datetime
    ended_at: datetime | None = None
    events: list[NormalizedEvent]
    raw_stats: dict[str, Any] = Field(default_factory=dict)


class EventClassification(BaseModel):
    event_id: str
    label: Literal["legit", "grey", "exploit", "hack", "meta_exploit"]
    reason: str
    flags: list[str] = Field(default_factory=list)


class SessionScores(BaseModel):
    outcome: float
    efficiency: float
    integrity: float
    honesty: float
    final_score: float


class SessionResult(BaseModel):
    session_id: str
    model: str
    scores: SessionScores
    flags: list[str] = Field(default_factory=list)
    event_classifications: list[EventClassification] = Field(default_factory=list)


class ModelStats(BaseModel):
    model: str
    session_count: int
    avg_score: float
    exploit_rate: float
    avg_outcome: float
    avg_efficiency: float
    avg_integrity: float


class EvalReport(BaseModel):
    generated_at: datetime
    sessions: list[SessionResult]
    leaderboard: list[ModelStats]
