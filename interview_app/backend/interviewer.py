"""
interviewer.py — the live AI interviewer: adaptive probing and structured grading.

Two AI call types, both on the configured model:

1. next_action(state, question)  — looks at the running transcript and decides to
   either ask ONE focused follow-up probe, or signal that it's time to grade
   (the model outputs the sentinel [READY_TO_GRADE]). Probe budget is enforced
   in code (drill <=1, full_mock <=2) regardless of what the model wants.

2. grade(state, question)        — forced tool-use call returning a structured
   Grade (level, score, criteria, strengths, gaps, what-would-elevate). The
   backend then anchors score_pct to the level band so XP math stays consistent.

Callers wrap both in try/except over llm_client.CONNECTION_ERRORS to fall
back to the offline reveal path per question.
"""

from __future__ import annotations

import json
from typing import Any

from . import llm_client as llm
from .grading import anchor_score, clamp01
from .schemas import Grade
from .sessions import SessionState

from pydantic import ValidationError

READY = "[READY_TO_GRADE]"

PROBE_BUDGET = {"drill": 1, "full_mock": 2}

_TRACK_PERSONA = {
    "ml_system_design": "a senior staff engineer running an ML system design interview",
    "fundamentals": "a senior research engineer running an ML/AI fundamentals interview",
    "frontier_research": "a research scientist at a frontier AI lab running a research-depth interview",
    "behavioral": "an engineering director running a behavioral and leadership interview",
}

# Grade tool schema (OpenAI function-calling format) — mirrors schemas.Grade.
# Forced tool choice guarantees a structured JSON payload.
_GRADE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_grade",
        "description": "Submit the structured grade for the candidate's answer to this question.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["no_hire", "weak_hire", "hire", "strong_hire"]},
                "score_pct": {"type": "number", "description": "0.0-1.0, consistent with level"},
                "criteria_met": {"type": "array", "items": {"type": "string"}},
                "criteria_missing": {"type": "array", "items": {"type": "string"}},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "gaps": {"type": "array", "items": {"type": "string"}},
                "what_would_elevate": {"type": "string"},
                "model_answer_pointer": {"type": "string"},
            },
            "required": [
                "level", "score_pct", "criteria_met", "criteria_missing",
                "strengths", "gaps", "what_would_elevate", "model_answer_pointer",
            ],
        },
    },
}


def _persona(track: str) -> str:
    return _TRACK_PERSONA.get(track, "a senior interviewer at a frontier AI lab")


def _transcript_text(state: SessionState, question_id: str) -> str:
    lines = []
    for turn in state.transcript_for(question_id):
        who = "Candidate" if turn["role"] == "candidate" else "Interviewer"
        lines.append(f"{who}: {turn['text']}")
    return "\n\n".join(lines)


def _criteria_block(question: dict[str, Any]) -> str:
    crit = question.get("criteria") or []
    if not crit:
        return "(no explicit checklist — judge against the question and your own bar)"
    return "\n".join(f"- {c}" for c in crit)


def next_action(state: SessionState, question: dict[str, Any]) -> dict[str, Any]:
    """Decide probe-vs-grade for the current question. Returns
    {"action": "probe", "probe": str} or {"action": "grade_ready"}.

    Raises on connection errors (caller degrades to offline)."""
    qid = question["id"]
    budget = PROBE_BUDGET.get(state.mode, 1)
    if state.probe_counts.get(qid, 0) >= budget:
        return {"action": "grade_ready"}

    client = llm.get_client()
    if client is None:
        return {"action": "grade_ready"}

    system = (
        f"You are {_persona(state.track)} at a frontier AI lab (OpenAI/Anthropic/DeepMind style). "
        "You are mid-interview on ONE question. Read the transcript so far. "
        "Decide: does the candidate need one more focused follow-up to reveal depth, "
        "or have they been tested enough on this question?\n"
        f"- If a probe helps, ask exactly ONE short, pointed follow-up that pushes on a GAP "
        "(not something already answered well). Output only the probe text, nothing else.\n"
        f"- If they've been adequately tested, output exactly {READY} and nothing else.\n"
        "Never reveal the grading rubric or model answers. Stay in character; no meta-commentary."
    )
    user = (
        f"Question being interviewed:\n{question['prompt']}\n\n"
        f"Coverage checklist for this question:\n{_criteria_block(question)}\n\n"
        f"Transcript so far:\n{_transcript_text(state, qid)}\n\n"
        f"You have asked {state.probe_counts.get(qid, 0)} of {budget} allowed probes. "
        f"Either ask one more probe, or output {READY}."
    )
    resp = client.chat.completions.create(
        model=llm.model_id(),
        max_tokens=400,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    text = _first_text(resp).strip()
    if READY in text or not text:
        return {"action": "grade_ready"}
    state.bump_probe(qid)
    return {"action": "probe", "probe": text}


def grade(state: SessionState, question: dict[str, Any]) -> dict[str, Any]:
    """Grade the candidate's answer to one question. Returns a Grade dict with
    score_pct anchored to the chosen level. Raises on connection errors."""
    client = llm.get_client()
    if client is None:
        raise RuntimeError("live grading unavailable")

    qid = question["id"]
    rubric_text = _rubric_text(question)
    system = (
        f"You are {_persona(state.track)} at a frontier AI lab. Grade the candidate's "
        "answer to ONE interview question against the calibrated rubric provided. "
        "Be honest and calibrated to the staff/principal bar. Choose exactly one level "
        "(no_hire, weak_hire, hire, strong_hire). Set score_pct consistent with the level "
        "(no_hire~0.25, weak_hire~0.5, hire~0.75, strong_hire~1.0). Cite concrete strengths "
        "and gaps from what the candidate actually said. Submit via the submit_grade tool."
    )
    user = (
        f"Question:\n{question['prompt']}\n\n"
        f"Coverage checklist:\n{_criteria_block(question)}\n\n"
        f"Calibrated rubric (model answers by level):\n{rubric_text}\n\n"
        f"Candidate transcript:\n{_transcript_text(state, qid)}"
    )
    resp = client.chat.completions.create(
        model=llm.model_id(),
        max_tokens=1500,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=[_GRADE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_grade"}},
    )
    raw = _first_tool_input(resp)
    return _normalize_grade(raw)


# ── parsing helpers ──────────────────────────────────────────────────────────
def _first_text(resp: Any) -> str:
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _first_tool_input(resp: Any) -> dict[str, Any]:
    try:
        calls = resp.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        calls = []
    if not calls:
        raise RuntimeError("grader returned no tool call")
    try:
        return json.loads(calls[0].function.arguments)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise RuntimeError(f"grader returned unparseable arguments: {exc}")


def _normalize_grade(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate with the Grade model, then anchor score_pct to the level band.

    A model that returns an out-of-enum level or a non-numeric score would make
    pydantic/float raise ValidationError/ValueError — errors NOT in the caller's
    degrade set. Re-raise them as RuntimeError so grade_question degrades to the
    offline reveal path instead of returning a 500."""
    try:
        g = Grade(
            level=raw.get("level", "weak_hire"),
            score_pct=clamp01(float(raw.get("score_pct", 0.5))),
            criteria_met=raw.get("criteria_met", []) or [],
            criteria_missing=raw.get("criteria_missing", []) or [],
            strengths=raw.get("strengths", []) or [],
            gaps=raw.get("gaps", []) or [],
            what_would_elevate=raw.get("what_would_elevate", "") or "",
            model_answer_pointer=raw.get("model_answer_pointer", "") or "",
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise RuntimeError(f"grader returned an invalid grade: {exc}")
    out = g.model_dump()
    out["score_pct"] = anchor_score(out["level"], out["score_pct"])
    return out


def _rubric_text(question: dict[str, Any]) -> str:
    parts = []
    labels = {"no_hire": "No Hire", "weak_hire": "Weak Hire", "hire": "Hire", "strong_hire": "Strong Hire"}
    for key in ("no_hire", "weak_hire", "hire", "strong_hire"):
        lv = question.get("rubric", {}).get(key)
        if lv:
            parts.append(f"### {labels[key]}\n{lv['answer']}")
    return "\n\n".join(parts)
