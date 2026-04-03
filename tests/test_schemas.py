from datetime import datetime, timezone
import pytest
from schemas import (
    BehaviorProfile,
    EventClassification,
    ModelStats,
    SessionResult,
    SessionScores,
)


def test_behavior_profile_fields():
    bp = BehaviorProfile(
        persistence=0.5,
        adaptivity=0.3,
        exploit_tendency=0.1,
        rule_following=0.9,
    )
    assert bp.persistence == 0.5
    assert bp.adaptivity == 0.3
    assert bp.exploit_tendency == 0.1
    assert bp.rule_following == 0.9


def test_event_classification_has_severity_default():
    c = EventClassification(event_id="e1", label="legit", reason="ok", flags=[])
    assert c.severity == "none"


def test_event_classification_accepts_severity():
    c = EventClassification(
        event_id="e1", label="exploit", reason="test", flags=["force_push_detected"],
        severity="medium",
    )
    assert c.severity == "medium"


def test_session_result_has_strategy_label():
    scores = SessionScores(outcome=3.0, efficiency=0.8, integrity=0.0, honesty=0.5, final_score=14.5)
    r = SessionResult(session_id="s1", model="claude", scores=scores)
    assert r.strategy_label == "unknown"


def test_session_result_accepts_behavior_profile():
    scores = SessionScores(outcome=3.0, efficiency=0.8, integrity=0.0, honesty=0.5, final_score=14.5)
    bp = BehaviorProfile(persistence=0.1, adaptivity=0.6, exploit_tendency=0.0, rule_following=1.0)
    r = SessionResult(session_id="s1", model="claude", scores=scores, behavior_profile=bp, strategy_label="legit_solver")
    assert r.behavior_profile.adaptivity == 0.6
    assert r.strategy_label == "legit_solver"


def test_model_stats_has_strategies():
    stats = ModelStats(
        model="claude",
        session_count=1,
        avg_score=10.0,
        exploit_rate=0.0,
        avg_outcome=3.0,
        avg_efficiency=0.8,
        avg_integrity=0.0,
    )
    assert stats.strategies == {}


def test_model_stats_accepts_strategies():
    stats = ModelStats(
        model="claude",
        session_count=2,
        avg_score=8.0,
        exploit_rate=0.5,
        avg_outcome=3.0,
        avg_efficiency=0.5,
        avg_integrity=-1.0,
        strategies={"direct_exploiter": 0.5, "legit_solver": 0.5},
    )
    assert stats.strategies["direct_exploiter"] == 0.5
