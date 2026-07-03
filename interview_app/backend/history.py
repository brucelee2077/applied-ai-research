"""
history.py — persist completed interview sessions to coach/interview_history/.

Each finished session is written as <session_id>.json (full record: transcript +
per-question grades + scorecard). A lightweight summary is appended to index.json
for the history list. Writes are atomic (temp-then-rename), the same idiom
coach.core.save_state uses. This directory is gitignored (per-user data).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import coach.core as cc

HISTORY_DIR = cc.COACH_DIR / "interview_history"
INDEX_PATH = HISTORY_DIR / "index.json"


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_session(record: dict[str, Any]) -> None:
    """Write the full session record and append a summary to the index."""
    session_id = record["session_id"]
    _atomic_write(HISTORY_DIR / f"{session_id}.json", record)

    index = load_index()
    summary = {
        "session_id": session_id,
        "track": record.get("track"),
        "topic": record.get("topic"),
        "mode": record.get("mode"),
        "overall_band": record.get("overall_band"),
        "overall_score_pct": record.get("overall_score_pct"),
        "passed": record.get("passed"),
        "ended_at": record.get("ended_at"),
        "num_questions": len(record.get("per_question", [])),
    }
    # Newest first; replace any existing entry with the same id.
    index = [e for e in index if e.get("session_id") != session_id]
    index.insert(0, summary)
    _atomic_write(INDEX_PATH, index)


def load_index() -> list[dict[str, Any]]:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def load_session(session_id: str) -> dict[str, Any] | None:
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
