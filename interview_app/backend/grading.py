"""
grading.py — turn rubric levels into scores and back, consistently with the
existing COACH boss-battle widget so XP math lines up across the whole app.
"""

from __future__ import annotations

from .config import LEVEL_SCORE, SCORE_BANDS

_LEVEL_LABELS = {
    "no_hire": "No Hire",
    "weak_hire": "Weak Hire",
    "hire": "Hire",
    "strong_hire": "Strong Hire",
}

# Each level's score band, matching the SCORE_BANDS display thresholds
# (no_hire [0,0.40), weak [0.40,0.65), hire [0.65,0.85), strong [0.85,1.0]).
# anchor_score keeps a grade strictly inside its band so a level always maps
# back to the same band via score_to_band.
_LEVEL_BANDS = {
    "no_hire": (0.0, 0.40),
    "weak_hire": (0.40, 0.65),
    "hire": (0.65, 0.85),
    "strong_hire": (0.85, 1.0),
}
_EPS = 1e-4


def level_to_score(level: str) -> float:
    """Band-center score for a level (no_hire 0.25 ... strong_hire 1.0)."""
    return LEVEL_SCORE.get(level, 0.25)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def anchor_score(level: str, model_score: float | None) -> float:
    """Anchor a score to its level's band center, allowing a small (+/-0.1)
    nudge from the model's number — but never letting it cross into a neighboring
    band. Guarantees score_to_band(anchor_score(level, x)) == level, so the
    per-question level and its displayed score always agree."""
    center = level_to_score(level)
    lo, hi = _LEVEL_BANDS.get(level, (0.0, 1.0))
    if model_score is None:
        return center
    nudged = center + max(-0.1, min(0.1, model_score - center))
    low_bound = lo if lo == 0.0 else lo + _EPS
    high_bound = hi if hi == 1.0 else hi - _EPS
    return clamp01(max(low_bound, min(high_bound, nudged)))


def score_to_band(score_pct: float) -> tuple[str, str]:
    """Map an aggregate score back to a (level_key, display_label) band."""
    for threshold, level, label in SCORE_BANDS:
        if score_pct >= threshold:
            return level, label
    return "no_hire", "No Hire"


def level_label(level: str) -> str:
    return _LEVEL_LABELS.get(level, level)
