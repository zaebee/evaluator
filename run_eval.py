#!/usr/bin/env python3
"""CLI entrypoint for the LLM agent benchmark evaluator."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from aggregator import aggregate, build_leaderboard, build_report
from classifier import classify_session
from llm_judge import LLMJudge
from metrics import compute_behavioral, compute_metrics
from parser import load_sessions
from schemas import SessionResult
from strategy import classify_strategy, compute_behavior_profile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_outcomes(path: str) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sid = record.get("session_id")
                if sid:
                    outcomes[sid] = record
            except json.JSONDecodeError:
                logger.warning("Skipping invalid line in outcomes file")
    return outcomes


def format_text(report_dict: dict[str, Any]) -> str:
    lines = ["=" * 60, "LLM Agent Benchmark Evaluation Report", "=" * 60, ""]

    for session in report_dict.get("sessions", []):
        scores = session["scores"]
        lines.append(f"Session: {session['session_id'][:16]}...")
        lines.append(f"  Model:       {session['model']}")
        lines.append(f"  Strategy:    {session.get('strategy_label', 'unknown')}")
        lines.append(f"  Final score: {scores['final_score']:.2f}")
        lines.append(f"  Outcome:     {scores['outcome']:.1f}/3.0")
        lines.append(f"  Efficiency:  {scores['efficiency']:.2f}")
        lines.append(f"  Integrity:   {scores['integrity']:.2f}")
        lines.append(f"  Honesty:     {scores['honesty']:.2f}")
        if session.get("flags"):
            lines.append(f"  Flags:       {', '.join(session['flags'])}")
        bp = session.get("behavior_profile")
        if bp:
            lines.append(
                f"  Profile:     persist={bp['persistence']:.2f} "
                f"adapt={bp['adaptivity']:.2f} "
                f"exploit={bp['exploit_tendency']:.2f} "
                f"follow={bp['rule_following']:.2f}"
            )
        lines.append("")

    lines += ["Leaderboard", "-" * 40]
    for rank, entry in enumerate(report_dict.get("leaderboard", []), start=1):
        strategies = entry.get("strategies", {})
        strat_str = ", ".join(f"{k}={v:.0%}" for k, v in strategies.items())
        lines.append(
            f"  {rank}. {entry['model']:<30} "
            f"avg={entry['avg_score']:.2f}  exploit_rate={entry['exploit_rate']:.0%}"
            + (f"  [{strat_str}]" if strat_str else "")
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM agent traces from benchmark sessions."
    )
    parser.add_argument("--traces", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--outcomes", metavar="FILE")
    parser.add_argument("--output", metavar="FILE")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    args = parser.parse_args()

    logger.info("Loading sessions from %d path(s)...", len(args.traces))
    sessions = load_sessions(args.traces)
    if not sessions:
        logger.error("No sessions loaded. Check trace paths.")
        sys.exit(1)
    logger.info("Loaded %d session(s)", len(sessions))

    outcomes: dict[str, dict] = {}
    if args.outcomes:
        outcomes = load_outcomes(args.outcomes)
        logger.info("Loaded outcomes for %d session(s)", len(outcomes))

    judge = LLMJudge()
    if args.judge and not judge.enabled:
        logger.warning(
            "Judge requested but EVALUATOR_JUDGE_PROVIDER is not set."
        )

    results: list[SessionResult] = []
    for trace in sessions:
        logger.info("Processing session %s (model=%s)", trace.session_id[:16], trace.model)

        classifications = classify_session(trace)

        all_flags: list[str] = []
        seen: set[str] = set()
        for c in classifications:
            for flag in c.flags:
                if flag not in seen:
                    all_flags.append(flag)
                    seen.add(flag)

        honesty = 0.5
        if args.judge and judge.enabled:
            try:
                judge_scores = judge.judge(trace)
                honesty = judge_scores.honesty
                logger.info(
                    "  Judge: honesty=%.2f quality=%.2f strategy=%s",
                    judge_scores.honesty,
                    judge_scores.reasoning_quality,
                    judge_scores.strategy,
                )
            except Exception as exc:
                logger.warning("Judge failed for %s: %s", trace.session_id, exc)

        outcome_data = outcomes.get(trace.session_id)
        scores = compute_metrics(trace, classifications, outcome_data, honesty=honesty)
        behavioral = compute_behavioral(trace)

        # Strategy
        profile = compute_behavior_profile(trace, classifications)
        strategy = classify_strategy(
            profile, flags=all_flags, outcome=scores.outcome, efficiency=scores.efficiency
        )

        logger.info(
            "  Scores: outcome=%.1f eff=%.2f integrity=%.2f final=%.2f | "
            "strategy=%s tool_calls=%d loops=%d sleeps=%d",
            scores.outcome, scores.efficiency, scores.integrity, scores.final_score,
            strategy,
            behavioral["total_tool_calls"],
            behavioral["loop_count"],
            behavioral["sleep_count"],
        )

        results.append(SessionResult(
            session_id=trace.session_id,
            model=trace.model,
            scores=scores,
            flags=all_flags,
            event_classifications=classifications,
            strategy_label=strategy,
            behavior_profile=profile,
        ))

    model_stats = aggregate(results)
    leaderboard = build_leaderboard(model_stats)
    report = build_report(results, leaderboard)

    report_dict = report.model_dump(mode="json")
    if args.format == "text":
        output_str = format_text(report_dict)
    else:
        output_str = json.dumps(report_dict, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        logger.info("Report written to %s", args.output)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
