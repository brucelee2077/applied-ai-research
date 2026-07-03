"""
Tests for the staff_interview_guide.md parser and the built question bank.

Run from the repo root:
    python -m pytest interview_app/tests/test_parser.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interview_app.backend.config import QUESTIONS_PATH, REPO_ROOT
from interview_app.backend.parser.staff_guide_md import parse_staff_guide
from interview_app.backend.parser.interview_md import parse_interview_md

ML_DESIGN = REPO_ROOT / "ML Design"


def _section(questions: list[dict], n: int) -> dict:
    return next(q for q in questions if q["order"] == n)


def test_visual_search_subheading_form():
    """Visual search uses '#### <Level> Answer' subheadings + a clarify table."""
    qs = parse_staff_guide(ML_DESIGN / "02-visual-search" / "staff_interview_guide.md", "02-visual-search")
    s1 = _section(qs, 1)
    assert s1["id"] == "02-visual-search__staff-guide__s1"
    assert s1["track"] == "ml_system_design"
    assert s1["module_id"] == "02-visual-search"
    assert "visual search" in s1["prompt"].lower()
    assert set(s1["rubric"]) == {"no_hire", "weak_hire", "hire", "strong_hire"}
    # Clarify table's Question column populated probes; Dimension column -> criteria.
    assert len(s1["probes"]) >= 1
    assert any("objective" in c.lower() for c in s1["criteria"])


def test_street_view_inline_form_and_lean_variants():
    """Street view uses inline '**Level:**' with Lean No Hire / Lean Hire variants."""
    qs = parse_staff_guide(ML_DESIGN / "03-google-street-view" / "staff_interview_guide.md", "03-google-street-view")
    s1 = _section(qs, 1)
    assert "street view" in s1["prompt"].lower()
    # Lean No Hire -> weak_hire, Lean Hire -> hire.
    assert set(s1["rubric"]) == {"no_hire", "weak_hire", "hire", "strong_hire"}
    # Follow-up Probes bullets become probes.
    assert any("latency" in p.lower() for p in s1["probes"])


def test_ml_design_prep_synthesized_prompt():
    """01-ml-design-prep sections lack an Interviewer Prompt -> synthesize one."""
    qs = parse_staff_guide(ML_DESIGN / "01-ml-design-prep" / "staff_interview_guide.md", "01-ml-design-prep")
    assert qs, "expected gradeable sections"
    s2 = _section(qs, 2)
    assert s2["prompt"].startswith("Let's go deep on")
    # Acronyms preserved (no "mL").
    assert "mL" not in s2["prompt"]
    assert len(s2["rubric"]) >= 1


def test_ids_unique_within_file():
    qs = parse_staff_guide(ML_DESIGN / "06-video-recommendation" / "staff_interview_guide.md", "06-video-recommendation")
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))


def test_interview_md_heading_and_bold_forms():
    """interview_md parses both '**Q1: ...**' (transformers) and '### Q1:' (RL)."""
    tx = parse_interview_md(REPO_ROOT / "01-transformers/architecture/attention-mechanisms-interview.md", "transformers")
    assert tx, "expected transformer questions"
    q1 = next(q for q in tx if q["order"] == 1)
    assert "d_k" in q1["prompt"] or "√" in q1["prompt"]
    assert set(q1["rubric"]) == {"no_hire", "weak_hire", "hire", "strong_hire"}
    assert q1["rubric"]["no_hire"]["answer"]
    assert q1["criteria"], "criteria_met should populate the checklist"

    rl = parse_interview_md(
        REPO_ROOT / "10-reinforcement-learning/fundamentals/what-is-reinforcement-learning-interview.md",
        "rl-fundamentals",
    )
    assert rl
    r1 = next(q for q in rl if q["order"] == 1)
    assert "supervised" in r1["prompt"].lower()
    assert r1["rubric"]["no_hire"]["interviewer_note"]
    assert any("non-stationarity" in c.lower() for c in r1["criteria"])


@pytest.mark.skipif(not QUESTIONS_PATH.exists(), reason="run build_questions.py first")
def test_built_bank_invariants():
    bank = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = bank["questions"]
    assert qs, "question bank is empty"
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in qs:
        assert q["prompt"].strip(), f"empty prompt: {q['id']}"
        assert q["rubric"], f"no rubric levels: {q['id']}"
        assert set(q["rubric"]).issubset({"no_hire", "weak_hire", "hire", "strong_hire"})
        for level in q["rubric"].values():
            assert level["answer"].strip(), f"empty answer in {q['id']}"
