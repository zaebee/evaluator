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
    severity: Literal["critical", "high", "medium", "low", "info", "none"] = "none"


class BehaviorProfile(BaseModel):
    persistence: float      # 0–1: retries after error
    adaptivity: float       # 0–1: diversity of tools/approaches
    exploit_tendency: float # 0–1: fraction of events that are exploit/hack/meta
    rule_following: float   # 0–1: fraction of events that are legit


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
    strategy_label: str = "unknown"
    behavior_profile: BehaviorProfile | None = None


class ModelStats(BaseModel):
    model: str
    session_count: int
    avg_score: float
    exploit_rate: float
    avg_outcome: float
    avg_efficiency: float
    avg_integrity: float
    strategies: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    generated_at: datetime
    sessions: list[SessionResult]
    leaderboard: list[ModelStats]
