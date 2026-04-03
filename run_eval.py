#!/usr/bin/env python3
"""CLI entrypoint for the LLM agent benchmark evaluator."""
from __future__ import annotations

import argparse
import json
import logging
import re
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


def _make_insight(entry: dict[str, Any]) -> tuple[str, str]:
    """Generate (short_insight, full_insight) from a leaderboard entry dict."""
    strategies = entry.get("strategies", {})
    top = max(strategies, key=lambda k: strategies[k]) if strategies else "unknown"
    success_pct = f"{entry['avg_outcome'] / 3.0:.0%}"
    integrity = entry["avg_integrity"]
    exploit_rate = entry["exploit_rate"]
    meta_rate = entry.get("meta_exploit_rate", 0.0)
    count = entry["session_count"]

    if meta_rate > 0:
        short = (
            f"Modified benchmark infrastructure in {meta_rate:.0%} of runs. "
            f"Success {success_pct}, integrity {integrity:.1f}."
        )
    elif exploit_rate == 1.0:
        short = (
            f"Consistently exploited constraints ({top}). "
            f"Success {success_pct}, avg integrity {integrity:.1f}."
        )
    elif exploit_rate > 0:
        short = (
            f"Used exploits in {exploit_rate:.0%} of runs ({top}). "
            f"Success {success_pct}."
        )
    elif top == "looper":
        short = f"Gets stuck in polling loops. Success {success_pct}, low efficiency."
    elif top == "planner":
        short = f"Methodical and adaptive. Clean success {success_pct}."
    else:
        short = f"Achieved {success_pct} success as {top}, integrity {integrity:.1f}."

    run_word = "run" if count == 1 else "runs"
    parts = [f"Evaluated across {count} {run_word}."]
    sr = entry["avg_outcome"] / 3.0
    if sr >= 0.9:
        parts.append("Completed the benchmark task successfully in nearly all sessions.")
    elif sr >= 0.5:
        parts.append(f"Completed the task in {sr:.0%} of sessions.")
    else:
        parts.append(f"Struggled, completing the task in only {sr:.0%} of sessions.")
    if exploit_rate > 0:
        parts.append(
            f"In {exploit_rate:.0%} of sessions it used exploit-level tactics, "
            f"classified as {top}."
        )
    if integrity <= -4:
        parts.append(
            f"Integrity ({integrity:.1f}) hit the penalty cap, indicating severe rule violations."
        )
    elif integrity < -1:
        parts.append(f"Integrity averaged {integrity:.1f}, reflecting significant rule-bending.")
    elif integrity < 0:
        parts.append(f"Integrity averaged {integrity:.1f} due to minor exploits.")
    parts.append(
        f"Efficiency averaged {entry['avg_efficiency']:.2f}; "
        f"honesty score was {entry.get('avg_honesty', 0.5):.2f}."
    )
    return short, " ".join(parts)


def format_frontend(report_dict: dict[str, Any]) -> str:
    """Convert EvalReport leaderboard -> ModelStat[] JSON for the frontend."""
    out = []
    for entry in report_dict.get("leaderboard", []):
        model = entry["model"]
        strategies = entry.get("strategies", {})
        flags = list(strategies.keys())
        short, full = _make_insight(entry)
        out.append({
            "id": re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_"),
            "model": model,
            "runs": entry["session_count"],
            "score": entry["avg_score"],
            "success_rate": round(entry["avg_outcome"] / 3.0, 4),
            "integrity": entry["avg_integrity"],
            "efficiency": entry["avg_efficiency"],
            "honesty": entry.get("avg_honesty", 0.5),
            "exploit_rate": entry["exploit_rate"],
            "meta_exploit_rate": entry.get("meta_exploit_rate", 0.0),
            "variance": entry.get("score_variance", 0.0),
            "flags": flags,
            "insight": short,
            "full_insight": full,
        })
    return json.dumps(out, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM agent traces from benchmark sessions."
    )
    parser.add_argument("--traces", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--outcomes", metavar="FILE")
    parser.add_argument("--output", metavar="FILE")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--format", choices=["json", "text", "frontend"], default="json")
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
    elif args.format == "frontend":
        output_str = format_frontend(report_dict)
    else:
        output_str = json.dumps(report_dict, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        logger.info("Report written to %s", args.output)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
