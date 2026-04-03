from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from parser import OpenCodeParser, _parse_opencode_dt, load_sessions


FIXTURE = (
    "# Test session\n"
    "\n"
    "**Session ID:** ses_abc123\n"
    "**Created:** 4/3/2026, 10:00:00 PM\n"
    "**Updated:** 4/3/2026, 10:05:00 PM\n"
    "---\n"
    "\n"
    "## Assistant (Build · minimax-m2.5-free · 3.9s)\n"
    "\n"
    "_Thinking:_ Let me check the repo.\n"
    "\n"
    "**Tool: bash**\n"
    "\n"
    "**Input:**\n"
    "```json\n"
    '{"command": "git status"}\n'
    "```\n"
    "\n"
    "**Output:**\n"
    "```\n"
    "nothing to commit\n"
    "```\n"
    "\n"
    "**Tool: bash**\n"
    "\n"
    "**Input:**\n"
    "```json\n"
    '{"command": "gh pr merge 1 --admin"}\n'
    "```\n"
    "\n"
    "**Output:**\n"
    "```\n"
    "Merged pull request #1\n"
    "```\n"
    "---\n"
)


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    p = tmp_path / "session-ses_abc123.md"
    p.write_text(FIXTURE, encoding="utf-8")
    return p


@pytest.fixture
def trace(trace_file: Path):
    return OpenCodeParser().parse(trace_file)


# --- datetime parsing ---

def test_parse_opencode_dt_pm():
    dt = _parse_opencode_dt("4/3/2026, 10:08:07 PM")
    assert dt is not None
    assert dt.hour == 22
    assert dt.minute == 8
    assert dt.year == 2026


def test_parse_opencode_dt_am_midnight():
    dt = _parse_opencode_dt("4/3/2026, 12:00:00 AM")
    assert dt is not None
    assert dt.hour == 0


def test_parse_opencode_dt_noon():
    dt = _parse_opencode_dt("4/3/2026, 12:00:00 PM")
    assert dt is not None
    assert dt.hour == 12


def test_parse_opencode_dt_invalid():
    assert _parse_opencode_dt("not a date") is None


# --- session parsing ---

def test_opencode_session_id(trace):
    assert trace.session_id == "ses_abc123"


def test_opencode_model(trace):
    assert trace.model == "minimax-m2.5-free"


def test_opencode_started_at(trace):
    assert trace.started_at is not None
    assert trace.started_at.hour == 22  # 10 PM UTC


def test_opencode_event_count(trace):
    # 1 user_input + 2 tool_calls + 2 tool_results = 5
    assert len(trace.events) == 5


def test_opencode_event_types(trace):
    types = [e.event_type for e in trace.events]
    assert types[0] == "user_input"
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2


def test_opencode_tool_names(trace):
    tool_calls = [e for e in trace.events if e.event_type == "tool_call"]
    assert all(e.tool_name == "bash" for e in tool_calls)


def test_opencode_tool_call_content(trace):
    tool_calls = [e for e in trace.events if e.event_type == "tool_call"]
    contents = [e.content for e in tool_calls]
    assert any("gh pr merge 1 --admin" in c for c in contents)


def test_opencode_tool_result_truncation(trace):
    results = [e for e in trace.events if e.event_type == "tool_result"]
    assert all(len(e.content) <= 500 for e in results)


def test_load_sessions_md(trace_file: Path):
    sessions = load_sessions([str(trace_file)])
    assert len(sessions) == 1
    assert sessions[0].model == "minimax-m2.5-free"
