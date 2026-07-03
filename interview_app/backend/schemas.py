"""
schemas.py — Pydantic models for the API request/response bodies and the
structured grading output.

These are the contract between the frontend and backend. The `Grade` model is
also the structured-output schema the live AI grader is forced to return.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["no_hire", "weak_hire", "hire", "strong_hire"]
Mode = Literal["drill", "full_mock"]


# ── Question bank projections ────────────────────────────────────────────────
class QuestionSummary(BaseModel):
    id: str
    topic: str
    section_title: str = ""
    order: int = 0
    difficulty: int = 3
    prompt_preview: str


class ModuleSummary(BaseModel):
    module_id: str
    topic: str
    question_count: int
    questions: list[QuestionSummary]


class TrackSummary(BaseModel):
    track: str
    label: str
    question_count: int
    modules: list[ModuleSummary]


# ── Progress / gamification projection ───────────────────────────────────────
class BadgeView(BaseModel):
    id: str
    name: str
    desc: str
    rarity: str
    emoji: str


class ProgressView(BaseModel):
    xp: int
    level: int
    level_name: str
    xp_into_level: int
    xp_needed_for_next: int
    tokens: int
    percentile: int
    badges: list[BadgeView]


# ── Session lifecycle ────────────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    track: str
    module_id: Optional[str] = None
    mode: Mode = "drill"
    question_ids: Optional[list[str]] = None
    # Drill: optionally target a single question by id.
    question_id: Optional[str] = None
    # Force offline reveal even when a key is present (None -> auto by key).
    live: Optional[bool] = None


class QuestionView(BaseModel):
    """A question as shown to the candidate — rubric is intentionally withheld."""
    id: str
    topic: str
    section_title: str = ""
    difficulty: int = 3
    prompt: str
    index: int          # 0-based position in this session
    total: int          # number of questions in this session


class StartSessionResponse(BaseModel):
    session_id: str
    track: str
    mode: Mode
    live: bool
    live_available: bool
    total_questions: int
    question: QuestionView


class AnswerRequest(BaseModel):
    text: str


class AnswerResponse(BaseModel):
    # Offline mode returns "reveal_ready"; live mode returns "probe"/"grade_ready".
    action: Literal["reveal_ready", "probe", "grade_ready"]
    probe: Optional[str] = None


class RubricLevelView(BaseModel):
    answer: str
    interviewer_note: str = ""
    criteria_met: list[str] = Field(default_factory=list)
    criteria_missing: list[str] = Field(default_factory=list)


class RevealResponse(BaseModel):
    question_id: str
    prompt: str
    criteria: list[str]
    rubric: dict[str, RubricLevelView]


class Grade(BaseModel):
    """Structured grade for one question. Also the live grader's output schema."""
    level: Level
    score_pct: float = Field(ge=0.0, le=1.0)
    criteria_met: list[str] = Field(default_factory=list)
    criteria_missing: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    what_would_elevate: str = ""
    model_answer_pointer: str = ""


class SelfGradeRequest(BaseModel):
    level: Level


class GradeResponse(BaseModel):
    grade: Optional[Grade] = None
    degraded: bool = False          # live grading unreachable -> fall back to reveal
    has_next: bool = False
    next_question: Optional[QuestionView] = None


# ── End-of-session scorecard ─────────────────────────────────────────────────
class PerQuestionResult(BaseModel):
    question_id: str
    section_title: str
    level: Level
    score_pct: float


class XpDelta(BaseModel):
    xp_awarded: int
    tokens_awarded: int
    level_up: bool
    new_level: int
    new_level_name: str
    new_badges: list[BadgeView] = Field(default_factory=list)
    newly_unlocked_modules: list[str] = Field(default_factory=list)


class Scorecard(BaseModel):
    session_id: str
    track: str
    mode: Mode
    topic: str
    module_id: str
    overall_score_pct: float
    overall_level: Level
    overall_band: str
    passed: bool
    elapsed_minutes: float
    per_question: list[PerQuestionResult]
    criteria_coverage: dict[str, float] = Field(default_factory=dict)
    # Full per-question record (prompt + transcript turns + grade) so the client
    # can render "Review transcript" immediately, without a second history fetch.
    questions: list[dict[str, Any]] = Field(default_factory=list)
    xp: XpDelta


class HealthResponse(BaseModel):
    live_available: bool
    model: str
    base_url: str
