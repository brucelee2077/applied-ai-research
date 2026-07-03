"""
Unit tests for question_bank.select_questions — the session question picker.

Covers the full-mock cap (MAX_FULL_MOCK), ordering, and single-question drills.
Pure — reads the committed data/questions.json, no client / network.

Run from repo root:
    python -m pytest interview_app/tests/test_question_bank.py -q
"""

from __future__ import annotations

import random

from interview_app.backend import question_bank as qb


def test_full_mock_module_is_ordered_and_capped():
    qs = qb.select_questions(track="ml_system_design", mode="full_mock", module_id="02-visual-search")
    assert 1 <= len(qs) <= qb.MAX_FULL_MOCK
    orders = [q.get("order", 0) for q in qs]
    assert orders == sorted(orders), "full mock must walk a module in order"


def test_full_mock_track_sample_is_capped():
    # fundamentals has ~183 questions; a module-less full mock samples the track
    # and must never exceed MAX_FULL_MOCK. Seeded RNG keeps this deterministic.
    qs = qb.select_questions(
        track="fundamentals", mode="full_mock", module_id=None, rng=random.Random(0)
    )
    assert len(qs) == qb.MAX_FULL_MOCK


def test_drill_returns_single_question():
    qs = qb.select_questions(track="ml_system_design", mode="drill", module_id="02-visual-search")
    assert len(qs) == 1


def test_explicit_question_ids_win():
    listing = qb.track_listing()
    msd = next(t for t in listing if t["track"] == "ml_system_design")
    ids = [q["id"] for q in msd["modules"][0]["questions"][:2]]
    qs = qb.select_questions(track="ml_system_design", mode="drill", question_ids=ids)
    assert [q["id"] for q in qs] == ids
