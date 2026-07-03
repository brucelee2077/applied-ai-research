"""
question_bank.py — load the built question bank (data/questions.json) and serve
projections: track listing, question lookup, and session question selection.

Loaded once at import; call reload() after re-running build_questions.py.
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional

from .config import QUESTIONS_PATH

# Human-readable labels for the tracks.
TRACK_LABELS = {
    "ml_system_design": "ML System Design",
    "fundamentals": "ML / AI Fundamentals",
    "frontier_lab": "Frontier Lab (24-Week Track)",
    "frontier_research": "Frontier Research (RL / LLM / Alignment)",
    "behavioral": "Behavioral & Leadership",
}

_BANK: dict[str, Any] = {"version": 0, "questions": []}
_BY_ID: dict[str, dict[str, Any]] = {}

# A full mock walks at most this many questions (ML Design case studies have ~8
# sections; fundamentals modules can have 100+, so cap to keep sessions sane).
MAX_FULL_MOCK = 8


def reload() -> None:
    global _BANK, _BY_ID
    if QUESTIONS_PATH.exists():
        _BANK = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    else:
        _BANK = {"version": 0, "questions": []}
    _BY_ID = {q["id"]: q for q in _BANK.get("questions", [])}


def all_questions() -> list[dict[str, Any]]:
    return _BANK.get("questions", [])


def get(question_id: str) -> Optional[dict[str, Any]]:
    return _BY_ID.get(question_id)


def _label(track: str) -> str:
    return TRACK_LABELS.get(track, track.replace("_", " ").title())


def track_listing() -> list[dict[str, Any]]:
    """Nested tracks -> modules -> question summaries for the picker UI."""
    tracks: dict[str, dict[str, Any]] = {}
    for q in all_questions():
        t = tracks.setdefault(
            q["track"], {"track": q["track"], "label": _label(q["track"]), "modules": {}}
        )
        m = t["modules"].setdefault(
            q["module_id"], {"module_id": q["module_id"], "topic": q.get("topic", q["module_id"]), "questions": []}
        )
        m["questions"].append(
            {
                "id": q["id"],
                "topic": q.get("topic", ""),
                "section_title": q.get("section_title", ""),
                "order": q.get("order", 0),
                "difficulty": q.get("difficulty", 3),
                "prompt_preview": _preview(q["prompt"]),
            }
        )

    out: list[dict[str, Any]] = []
    for t in tracks.values():
        modules = []
        for m in t["modules"].values():
            m["questions"].sort(key=lambda x: (x["order"], x["id"]))
            modules.append(
                {
                    "module_id": m["module_id"],
                    "topic": m["topic"],
                    "question_count": len(m["questions"]),
                    "questions": m["questions"],
                }
            )
        modules.sort(key=lambda x: x["module_id"])
        out.append(
            {
                "track": t["track"],
                "label": t["label"],
                "question_count": sum(mm["question_count"] for mm in modules),
                "modules": modules,
            }
        )
    out.sort(key=lambda x: x["track"])
    return out


def select_questions(
    track: str,
    mode: str,
    module_id: Optional[str] = None,
    question_id: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> list[dict[str, Any]]:
    """Pick the question(s) for a session.

    - explicit question_ids win
    - drill: a single question (the given id, else first/random of the scope)
    - full_mock: all questions of a module in order (a full case-study walkthrough),
      or a sample of the track if no module is given
    """
    rng = rng or random.Random()

    if question_ids:
        picked = [get(qid) for qid in question_ids]
        return [q for q in picked if q]

    scope = [q for q in all_questions() if q["track"] == track]
    if module_id:
        scope = [q for q in scope if q["module_id"] == module_id]

    if mode == "drill":
        if question_id and get(question_id):
            return [get(question_id)]
        if not scope:
            return []
        scope_sorted = sorted(scope, key=lambda x: (x.get("order", 0), x["id"]))
        return [scope_sorted[0]]

    # full_mock
    if module_id:
        ordered = sorted(scope, key=lambda x: (x.get("order", 0), x["id"]))
        return ordered[:MAX_FULL_MOCK]
    # No module -> a small sampled mock across the track.
    sample = scope[:]
    rng.shuffle(sample)
    return sample[: min(MAX_FULL_MOCK, len(sample))]


def _preview(prompt: str, n: int = 140) -> str:
    prompt = prompt.strip().replace("\n", " ")
    return prompt if len(prompt) <= n else prompt[: n - 1].rstrip() + "…"


# Load at import.
reload()
