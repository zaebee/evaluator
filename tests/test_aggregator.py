import pytest
from datetime import datetime, timezone
from schemas import ModelStats, SessionResult, SessionScores, BehaviorProfile
from aggregator import aggregate, build_leaderboard, build_report


def _result(model: str, score: float, flags=None, strategy: str = "legit_solver") -> SessionResult:
    scores = SessionScores(outcome=3.0, efficiency=0.8, integrity=0.0, honesty=0.5, final_score=score)
    return SessionResult(
        session_id=f"s-{model}-{score}",
        model=model,
        scores=scores,
        flags=flags or [],
        strategy_label=strategy,
    )


def test_aggregate_strategy_distribution_single():
    results = [_result("claude", 10.0, strategy="legit_solver")]
    stats = aggregate(results)
    assert stats["claude"].strategies == {"legit_solver": 1.0}


def test_aggregate_strategy_distribution_mixed():
    results = [
        _result("claude", 10.0, strategy="legit_solver"),
        _result("claude", 6.0, flags=["admin_override_usage"], strategy="direct_exploiter"),
    ]
    stats = aggregate(results)
    assert stats["claude"].strategies["legit_solver"] == pytest.approx(0.5)
    assert stats["claude"].strategies["direct_exploiter"] == pytest.approx(0.5)


def test_aggregate_multiple_models():
    results = [
        _result("claude", 10.0, strategy="planner"),
        _result("mistral", 6.0, flags=["admin_override_usage"], strategy="direct_exploiter"),
    ]
    stats = aggregate(results)
    assert "claude" in stats
    assert "mistral" in stats
    assert stats["claude"].strategies == {"planner": 1.0}
    assert stats["mistral"].strategies == {"direct_exploiter": 1.0}


def test_leaderboard_sorted_by_score():
    results = [
        _result("mistral", 11.4, strategy="direct_exploiter"),
        _result("claude", 6.5, flags=["branch_protection_bypass"], strategy="direct_exploiter"),
    ]
    stats = aggregate(results)
    board = build_leaderboard(stats)
    assert board[0].model == "mistral"
    assert board[1].model == "claude"


def test_build_report_structure():
    results = [_result("claude", 10.0)]
    stats = aggregate(results)
    board = build_leaderboard(stats)
    report = build_report(results, board)
    assert len(report.sessions) == 1
    assert len(report.leaderboard) == 1
