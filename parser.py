from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas import NormalizedEvent, SessionTrace

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


_OPENCODE_DT_PAT = re.compile(
    r'(\d+)/(\d+)/(\d{4}),\s*(\d+):(\d+):(\d+)\s*(AM|PM)', re.IGNORECASE
)


def _parse_opencode_dt(value: str) -> datetime | None:
    """Parse US-format datetime like '4/3/2026, 10:08:07 PM' into UTC datetime."""
    m = _OPENCODE_DT_PAT.search(value)
    if not m:
        return None
    month, day, year, hour, minute, second, ampm = m.groups()
    h = int(hour)
    if ampm.upper() == "PM" and h != 12:
        h += 12
    elif ampm.upper() == "AM" and h == 12:
        h = 0
    try:
        return datetime(int(year), int(month), int(day), h, int(minute), int(second), tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _make_id(session_id: str, index: int) -> str:
    return f"{session_id}-{index:04d}"


class ClaudeParser:
    def parse(self, path: str | Path) -> SessionTrace:
        path = Path(path)
        records: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in %s", path)

        session_id = path.stem
        model = "claude"
        started_at: datetime | None = None
        ended_at: datetime | None = None
        events: list[NormalizedEvent] = []
        event_index = 0
        first_user_done = False

        for rec in records:
            # Extract session_id from first record that has it
            if "sessionId" in rec and session_id == path.stem:
                session_id = rec["sessionId"]

            rec_type = rec.get("type")

            if rec_type == "assistant":
                msg = rec.get("message", {})
                if not model or model == "claude":
                    model = msg.get("model", model)
                ts = _parse_dt(rec.get("timestamp")) or started_at or _utcnow()
                content_list = msg.get("content", [])
                if not isinstance(content_list, list):
                    continue
                for item in content_list:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "tool_use":
                        events.append(NormalizedEvent(
                            event_id=_make_id(session_id, event_index),
                            session_id=session_id,
                            timestamp=ts,
                            role="assistant",
                            event_type="tool_call",
                            tool_name=item.get("name"),
                            content=json.dumps(item.get("input", {})),
                            metadata={"tool_use_id": item.get("id", "")},
                        ))
                        event_index += 1
                    elif item_type == "text":
                        text = item.get("text", "").strip()
                        if text:
                            events.append(NormalizedEvent(
                                event_id=_make_id(session_id, event_index),
                                session_id=session_id,
                                timestamp=ts,
                                role="assistant",
                                event_type="message",
                                tool_name=None,
                                content=text,
                            ))
                            event_index += 1

            elif rec_type == "user":
                msg = rec.get("message", {})
                ts = _parse_dt(rec.get("timestamp")) or started_at or _utcnow()
                if started_at is None:
                    started_at = ts
                raw_content = msg.get("content", "")

                # Plain string content = user input (first one only)
                if isinstance(raw_content, str) and not first_user_done:
                    first_user_done = True
                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=ts,
                        role="user",
                        event_type="user_input",
                        tool_name=None,
                        content=raw_content[:2000],  # truncate very long prompts
                    ))
                    event_index += 1

                # List content = tool results
                elif isinstance(raw_content, list):
                    for item in raw_content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "tool_result":
                            text = _extract_text(item.get("content", ""))
                            events.append(NormalizedEvent(
                                event_id=_make_id(session_id, event_index),
                                session_id=session_id,
                                timestamp=ts,
                                role="user",
                                event_type="tool_result",
                                tool_name=None,
                                content=text[:500],  # truncate large tool outputs
                                metadata={"tool_use_id": item.get("tool_use_id", "")},
                            ))
                            event_index += 1

        if started_at is None:
            started_at = _utcnow()

        raw_stats: dict[str, Any] = {}
        # Collect usage from last assistant message
        for rec in reversed(records):
            if rec.get("type") == "assistant":
                usage = rec.get("message", {}).get("usage", {})
                if usage:
                    raw_stats["usage"] = usage
                break

        return SessionTrace(
            session_id=session_id,
            model=model,
            started_at=started_at,
            ended_at=ended_at,
            events=events,
            raw_stats=raw_stats,
        )


class MistralParser:
    def parse(self, dir_path: str | Path) -> SessionTrace:
        dir_path = Path(dir_path)
        meta_path = dir_path / "meta.json"
        messages_path = dir_path / "messages.jsonl"

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        session_id = meta.get("session_id", str(uuid.uuid4()))
        model = meta.get("model", "mistral")
        started_at = _parse_dt(meta.get("start_time")) or _utcnow()
        ended_at = _parse_dt(meta.get("end_time"))
        raw_stats = meta.get("stats", {})

        events: list[NormalizedEvent] = []
        event_index = 0
        first_user_done = False

        with messages_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in %s", messages_path)
                    continue

                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls") or []

                if role == "assistant" and tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        events.append(NormalizedEvent(
                            event_id=_make_id(session_id, event_index),
                            session_id=session_id,
                            timestamp=started_at,
                            role="assistant",
                            event_type="tool_call",
                            tool_name=fn.get("name"),
                            content=fn.get("arguments", ""),
                            metadata={"tool_call_id": tc.get("id", "")},
                        ))
                        event_index += 1

                elif role == "tool":
                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=started_at,
                        role="user",
                        event_type="tool_result",
                        tool_name=msg.get("name"),
                        content=str(content)[:500],
                        metadata={"tool_call_id": msg.get("tool_call_id", "")},
                    ))
                    event_index += 1

                elif role == "user" and not first_user_done:
                    first_user_done = True
                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=started_at,
                        role="user",
                        event_type="user_input",
                        tool_name=None,
                        content=str(content)[:2000],
                    ))
                    event_index += 1

                elif role == "assistant" and content and not tool_calls:
                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=started_at,
                        role="assistant",
                        event_type="message",
                        tool_name=None,
                        content=str(content)[:500],
                    ))
                    event_index += 1

        return SessionTrace(
            session_id=session_id,
            model=model,
            started_at=started_at,
            ended_at=ended_at,
            events=events,
            raw_stats=raw_stats,
        )


class OpenCodeParser:
    """Parser for OpenCode markdown session traces.

    Format:
        **Session ID:** <id>
        **Created:** <US datetime>
        **Updated:** <US datetime>

        ## Assistant (Family · model-name · Xs)

        **Tool: bash**

        **Input:**
        ```json
        {"command": "...", ...}
        ```

        **Output:**
        ```
        ...output...
        ```
    """

    _SESSION_ID_PAT = re.compile(r'\*\*Session ID:\*\*\s+(\S+)')
    _CREATED_PAT = re.compile(r'\*\*Created:\*\*\s+(.+)')
    _UPDATED_PAT = re.compile(r'\*\*Updated:\*\*\s+(.+)')
    _TURN_PAT = re.compile(r'^## Assistant \(([^)]+)\)', re.MULTILINE)
    _TOOL_NAME_PAT = re.compile(r'^\*\*Tool:\s+(\w[\w-]*)\*\*\s*$')

    def parse(self, path: str | Path) -> SessionTrace:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # --- Metadata ---
        sid_m = self._SESSION_ID_PAT.search(text)
        session_id = sid_m.group(1) if sid_m else path.stem

        created_m = self._CREATED_PAT.search(text)
        started_at = _parse_opencode_dt(created_m.group(1)) if created_m else _utcnow()

        updated_m = self._UPDATED_PAT.search(text)
        ended_at = _parse_opencode_dt(updated_m.group(1)) if updated_m else None

        # Model from first Assistant heading: "Build · minimax-m2.5-free · 3.9s"
        model = "opencode"
        turn_m = self._TURN_PAT.search(text)
        if turn_m:
            parts = [p.strip() for p in turn_m.group(1).split("·")]
            if len(parts) >= 2:
                model = parts[1]

        events: list[NormalizedEvent] = []
        event_index = 0

        # Preamble (before first ## Assistant) → user_input
        first_turn = self._TURN_PAT.search(text)
        if first_turn:
            preamble = text[: first_turn.start()].strip()
            if preamble:
                events.append(NormalizedEvent(
                    event_id=_make_id(session_id, event_index),
                    session_id=session_id,
                    timestamp=started_at or _utcnow(),
                    role="user",
                    event_type="user_input",
                    tool_name=None,
                    content=preamble[:2000],
                ))
                event_index += 1

        # State machine: parse tool call/result pairs
        # States: IDLE, WANT_INPUT, INPUT_BLOCK_START, IN_INPUT,
        #         WANT_OUTPUT, OUTPUT_BLOCK_START, IN_OUTPUT
        state = "IDLE"
        current_tool: str | None = None
        input_lines: list[str] = []
        output_lines: list[str] = []
        ts = started_at or _utcnow()

        for line in lines:
            stripped = line.rstrip()

            if state == "IDLE":
                m = self._TOOL_NAME_PAT.match(stripped)
                if m:
                    current_tool = m.group(1)
                    state = "WANT_INPUT"

            elif state == "WANT_INPUT":
                if stripped == "**Input:**":
                    state = "INPUT_BLOCK_START"

            elif state == "INPUT_BLOCK_START":
                if stripped.startswith("```"):
                    input_lines = []
                    state = "IN_INPUT"

            elif state == "IN_INPUT":
                if stripped == "```":
                    state = "WANT_OUTPUT"
                else:
                    input_lines.append(stripped)

            elif state == "WANT_OUTPUT":
                if stripped == "**Output:**":
                    state = "OUTPUT_BLOCK_START"

            elif state == "OUTPUT_BLOCK_START":
                if stripped.startswith("```"):
                    output_lines = []
                    state = "IN_OUTPUT"

            elif state == "IN_OUTPUT":
                if stripped == "```":
                    input_content = "\n".join(input_lines)
                    output_content = "\n".join(output_lines)

                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=ts,
                        role="assistant",
                        event_type="tool_call",
                        tool_name=current_tool,
                        content=input_content[:2000],
                    ))
                    event_index += 1

                    events.append(NormalizedEvent(
                        event_id=_make_id(session_id, event_index),
                        session_id=session_id,
                        timestamp=ts,
                        role="user",
                        event_type="tool_result",
                        tool_name=current_tool,
                        content=output_content[:500],
                    ))
                    event_index += 1

                    current_tool = None
                    input_lines = []
                    output_lines = []
                    state = "IDLE"
                else:
                    output_lines.append(stripped)

        return SessionTrace(
            session_id=session_id,
            model=model,
            started_at=started_at or _utcnow(),
            ended_at=ended_at,
            events=events,
            raw_stats={},
        )


def load_sessions(paths: list[str]) -> list[SessionTrace]:
    claude = ClaudeParser()
    mistral = MistralParser()
    opencode = OpenCodeParser()
    sessions: list[SessionTrace] = []

    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            logger.warning("Path does not exist: %s", p)
            continue

        if p.is_file() and p.suffix == ".jsonl":
            try:
                sessions.append(claude.parse(p))
            except Exception as exc:
                logger.warning("Failed to parse Claude trace %s: %s", p, exc)

        elif p.is_dir():
            if (p / "messages.jsonl").exists():
                try:
                    sessions.append(mistral.parse(p))
                except Exception as exc:
                    logger.warning("Failed to parse Mistral trace %s: %s", p, exc)
            else:
                # Directory of Claude JSONL files
                jsonl_files = sorted(p.glob("*.jsonl"))
                if not jsonl_files:
                    logger.warning("No .jsonl files found in directory: %s", p)
                for jf in jsonl_files:
                    try:
                        sessions.append(claude.parse(jf))
                    except Exception as exc:
                        logger.warning("Failed to parse %s: %s", jf, exc)
        elif p.is_file() and p.suffix == ".md":
            try:
                sessions.append(opencode.parse(p))
            except Exception as exc:
                logger.warning("Failed to parse OpenCode trace %s: %s", p, exc)

        else:
            logger.warning("Cannot detect format for path: %s", p)

    return sessions
