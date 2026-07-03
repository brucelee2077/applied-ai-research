"""
sessions.py — in-memory store for live interview sessions.

A single local user means an in-process dict is enough. A session holds the
selected questions, the running transcript, and per-question grades. Completed
sessions are persisted to disk by history.py on /end; an in-progress session is
lost on server restart (acceptable).
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    session_id: str
    track: str
    mode: str
    module_id: str
    topic: str
    questions: list[dict[str, Any]]
    live: bool
    started_at: datetime.datetime
    current_index: int = 0
    turns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    grades: dict[str, dict[str, Any]] = field(default_factory=dict)
    probe_counts: dict[str, int] = field(default_factory=dict)

    # ── current question helpers ─────────────────────────────────────────────
    def current(self) -> Optional[dict[str, Any]]:
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def total(self) -> int:
        return len(self.questions)

    def has_next(self) -> bool:
        return self.current_index + 1 < len(self.questions)

    def advance(self) -> bool:
        if self.has_next():
            self.current_index += 1
            return True
        return False

    def is_complete(self) -> bool:
        return len(self.grades) >= len(self.questions) and len(self.questions) > 0

    # ── transcript / grading ─────────────────────────────────────────────────
    def add_turn(self, question_id: str, role: str, text: str) -> None:
        self.turns.setdefault(question_id, []).append(
            {"role": role, "text": text, "ts": datetime.datetime.now().isoformat()}
        )

    def transcript_for(self, question_id: str) -> list[dict[str, Any]]:
        return self.turns.get(question_id, [])

    def set_grade(self, question_id: str, grade: dict[str, Any]) -> None:
        self.grades[question_id] = grade

    def bump_probe(self, question_id: str) -> int:
        self.probe_counts[question_id] = self.probe_counts.get(question_id, 0) + 1
        return self.probe_counts[question_id]

    def elapsed_minutes(self) -> float:
        delta = datetime.datetime.now() - self.started_at
        return round(delta.total_seconds() / 60.0, 2)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(
        self,
        track: str,
        mode: str,
        module_id: str,
        topic: str,
        questions: list[dict[str, Any]],
        live: bool,
    ) -> SessionState:
        session_id = uuid.uuid4().hex[:12]
        state = SessionState(
            session_id=session_id,
            track=track,
            mode=mode,
            module_id=module_id,
            topic=topic,
            questions=questions,
            live=live,
            started_at=datetime.datetime.now(),
        )
        self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Process-wide singleton.
STORE = SessionStore()
