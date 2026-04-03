from __future__ import annotations

import json
import logging
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


def load_sessions(paths: list[str]) -> list[SessionTrace]:
    claude = ClaudeParser()
    mistral = MistralParser()
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
        else:
            logger.warning("Cannot detect format for path: %s", p)

    return sessions
