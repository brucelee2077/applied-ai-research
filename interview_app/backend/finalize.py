"""
finalize.py — aggregate a finished session into a scorecard, commit XP to the
COACH engine, and build the full record persisted to interview history.
"""

from __future__ import annotations

import datetime
from typing import Any

from . import coach_bridge, history
from .grading import level_label, score_to_band
from .sessions import SessionState


def _criteria_coverage(state: SessionState) -> dict[str, float]:
    """Aggregate per-criterion coverage across all graded questions.

    coverage[c] = (# times c appeared in criteria_met) / (# times c appeared at
    all). Empty when the rubric carried no explicit criteria (the offline
    staff_guide path) — the frontend then shows only the per-question bars.
    """
    met: dict[str, int] = {}
    total: dict[str, int] = {}
    for grade in state.grades.values():
        for c in grade.get("criteria_met", []):
            met[c] = met.get(c, 0) + 1
            total[c] = total.get(c, 0) + 1
        for c in grade.get("criteria_missing", []):
            total[c] = total.get(c, 0) + 1
    return {c: round(met.get(c, 0) / n, 3) for c, n in total.items() if n > 0}


def finalize(state: SessionState) -> dict[str, Any]:
    """Build the scorecard, commit XP, persist history. Returns a Scorecard dict."""
    elapsed = state.elapsed_minutes()
    ended_at = datetime.datetime.now().isoformat()

    per_question: list[dict[str, Any]] = []
    scores: list[float] = []
    for q in state.questions:
        grade = state.grades.get(q["id"])
        if not grade:
            continue
        scores.append(float(grade["score_pct"]))
        per_question.append(
            {
                "question_id": q["id"],
                "section_title": q.get("section_title") or q.get("topic", ""),
                "level": grade["level"],
                "score_pct": round(float(grade["score_pct"]), 3),
            }
        )

    overall_raw = sum(scores) / len(scores) if scores else 0.0
    # Band from the UNROUNDED aggregate so display rounding can't cross a boundary.
    overall_level, overall_band = score_to_band(overall_raw)
    overall = round(overall_raw, 3)
    passed = overall_raw >= 0.5
    coverage = _criteria_coverage(state)

    # Commit XP to COACH (single record per finished session).
    reason = f"interview_{state.track}_{state.module_id}"
    xp = coach_bridge.commit_result(state.module_id, overall_raw, elapsed, reason)

    scorecard = {
        "session_id": state.session_id,
        "track": state.track,
        "mode": state.mode,
        "topic": state.topic,
        "module_id": state.module_id,
        "overall_score_pct": overall,
        "overall_level": overall_level,
        "overall_band": overall_band,
        "passed": passed,
        "elapsed_minutes": elapsed,
        "per_question": per_question,
        "criteria_coverage": coverage,
        "xp": xp,
    }

    # Full record for the history browser (includes transcript + prompts).
    record = dict(scorecard)
    record.update(
        {
            "live": state.live,
            "started_at": state.started_at.isoformat(),
            "ended_at": ended_at,
            "questions": [
                {
                    "question_id": q["id"],
                    "section_title": q.get("section_title", ""),
                    "prompt": q["prompt"],
                    "transcript": state.transcript_for(q["id"]),
                    "grade": state.grades.get(q["id"]),
                }
                for q in state.questions
            ],
        }
    )
    # Surface the full per-question record (transcript + grade) on the scorecard
    # too, so "Review transcript" works immediately after /end (same shape as
    # the history detail), not only when replayed from history.
    scorecard["questions"] = record["questions"]
    history.save_session(record)
    return scorecard


__all__ = ["finalize", "level_label"]
