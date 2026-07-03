"""
Unit tests for grading.py — pure functions, no fixtures / network.

Locks the two invariants XP consistency depends on:
  * score_to_band boundaries match the SCORE_BANDS thresholds exactly, and
  * anchor_score keeps every level strictly inside its own band, so
    score_to_band(anchor_score(level, x)) == level for all levels and inputs.

Run from repo root:
    python -m pytest interview_app/tests/test_grading.py -q
"""

from __future__ import annotations

import pytest

from interview_app.backend.grading import (
    _LEVEL_BANDS,
    anchor_score,
    clamp01,
    level_to_score,
    score_to_band,
)

LEVELS = ["no_hire", "weak_hire", "hire", "strong_hire"]


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, "strong_hire"),
        (0.85, "strong_hire"),   # exact boundary lands in the higher band
        (0.8499, "hire"),        # just below
        (0.65, "hire"),
        (0.6499, "weak_hire"),
        (0.40, "weak_hire"),
        (0.3999, "no_hire"),
        (0.0, "no_hire"),
    ],
)
def test_score_to_band_boundaries(score, expected):
    assert score_to_band(score)[0] == expected


def test_level_to_score_centers():
    assert level_to_score("no_hire") == 0.25
    assert level_to_score("weak_hire") == 0.5
    assert level_to_score("hire") == 0.75
    assert level_to_score("strong_hire") == 1.0


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("model_score", [None, 0.0, 0.5, 1.0, -3.0, 4.2])
def test_anchor_score_never_crosses_its_band(level, model_score):
    s = anchor_score(level, model_score)
    lo, hi = _LEVEL_BANDS[level]
    assert lo <= s <= hi, (level, model_score, s)
    # The anchored score must map back to the SAME level via score_to_band.
    assert score_to_band(s)[0] == level, (level, model_score, s)


def test_clamp01():
    assert clamp01(-1.0) == 0.0
    assert clamp01(2.0) == 1.0
    assert clamp01(0.3) == 0.3
