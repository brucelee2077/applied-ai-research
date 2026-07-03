"""
main.py — FastAPI app for the mock interview system.

Serves the vanilla-JS frontend as static files (same origin -> no CORS, SSE works
cleanly) and exposes the /api surface. Phase 1 implements the offline path fully
(question -> answer -> reveal model answers -> self-grade -> scorecard + XP). The
live AI interviewer (SSE streaming + AI grading) is layered on in Phase 3.

Run from the repo root so `coach` is importable:
    uvicorn interview_app.backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import llm_client as llm
from . import coach_bridge, finalize, interviewer, question_bank
from .config import FRONTEND_DIR, get_base_url, get_model_id
from .grading import level_to_score
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    Grade,
    GradeResponse,
    HealthResponse,
    ProgressView,
    QuestionView,
    RevealResponse,
    Scorecard,
    SelfGradeRequest,
    StartSessionRequest,
    StartSessionResponse,
    TrackSummary,
)
from .sessions import STORE, SessionState

app = FastAPI(title="Mock Interview System", version="1.0")

# Live AI calls degrade to offline reveal on these (network/SDK) errors.
_DEGRADE_ERRORS = tuple(llm.CONNECTION_ERRORS) + (RuntimeError,)


# ── helpers ──────────────────────────────────────────────────────────────────
def _question_view(state: SessionState) -> QuestionView:
    q = state.current()
    if q is None:
        raise HTTPException(409, "No current question (session complete).")
    return QuestionView(
        id=q["id"],
        topic=q.get("topic", ""),
        section_title=q.get("section_title", ""),
        difficulty=q.get("difficulty", 3),
        prompt=q["prompt"],
        index=state.current_index,
        total=state.total(),
    )


def _get_session(session_id: str) -> SessionState:
    state = STORE.get(session_id)
    if state is None:
        raise HTTPException(404, "Session not found (it may have expired on restart).")
    return state


# ── health / tracks / progress ───────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(live_available=llm.is_live_available(), model=get_model_id(), base_url=get_base_url())


@app.get("/api/tracks", response_model=list[TrackSummary])
def tracks() -> list:
    return question_bank.track_listing()


@app.get("/api/progress", response_model=ProgressView)
def progress() -> dict:
    return coach_bridge.get_progress()


# ── history ──────────────────────────────────────────────────────────────────
@app.get("/api/history")
def history_index() -> list:
    return finalize.history.load_index()


@app.get("/api/history/{session_id}")
def history_detail(session_id: str) -> dict:
    record = finalize.history.load_session(session_id)
    if record is None:
        raise HTTPException(404, "No such interview in history.")
    return record


# ── session lifecycle ────────────────────────────────────────────────────────
@app.post("/api/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    questions = question_bank.select_questions(
        track=req.track,
        mode=req.mode,
        module_id=req.module_id,
        question_id=req.question_id,
        question_ids=req.question_ids,
    )
    if not questions:
        raise HTTPException(400, "No questions match that track/module selection.")

    topic = questions[0].get("topic", req.module_id or req.track)
    module_id = req.module_id or questions[0].get("module_id", "")
    # Live unless the interviewer is disabled, or the client explicitly forced offline.
    live_avail = llm.is_live_available()
    live = live_avail if req.live is None else (req.live and live_avail)
    state = STORE.create(
        track=req.track, mode=req.mode, module_id=module_id, topic=topic, questions=questions, live=live
    )
    return StartSessionResponse(
        session_id=state.session_id,
        track=state.track,
        mode=state.mode,
        live=live,
        live_available=live_avail,
        total_questions=state.total(),
        question=_question_view(state),
    )


@app.post("/api/session/{session_id}/answer", response_model=AnswerResponse)
def submit_answer(session_id: str, req: AnswerRequest) -> AnswerResponse:
    state = _get_session(session_id)
    q = state.current()
    if q is None:
        raise HTTPException(409, "Session has no current question.")
    state.add_turn(q["id"], "candidate", req.text)

    if state.live:
        # Live path: the interviewer either probes or signals it's time to grade.
        try:
            action = interviewer.next_action(state, q)
        except _DEGRADE_ERRORS:
            state.live = False  # degrade just this session to offline reveal
            return AnswerResponse(action="reveal_ready")
        if action["action"] == "probe":
            state.add_turn(q["id"], "interviewer", action["probe"])
            return AnswerResponse(action="probe", probe=action["probe"])
        return AnswerResponse(action="grade_ready")

    # Offline path: reveal the model answers for self-grading.
    return AnswerResponse(action="reveal_ready")


@app.post("/api/session/{session_id}/grade", response_model=GradeResponse)
def grade_question(session_id: str) -> GradeResponse:
    """Live AI grading of the current question. Degrades to offline reveal if the
    API is unreachable (frontend then shows the self-grade card)."""
    state = _get_session(session_id)
    q = state.current()
    if q is None:
        raise HTTPException(409, "Session has no current question.")
    if not state.live:
        return GradeResponse(grade=None, degraded=True, has_next=state.has_next())
    try:
        grade = interviewer.grade(state, q)
    except _DEGRADE_ERRORS:
        state.live = False
        return GradeResponse(grade=None, degraded=True, has_next=state.has_next())

    state.set_grade(q["id"], grade)
    has_next = state.advance()
    next_view = _question_view(state) if has_next else None
    return GradeResponse(grade=Grade(**grade), degraded=False, has_next=has_next, next_question=next_view)


@app.get("/api/session/{session_id}/reveal", response_model=RevealResponse)
def reveal(session_id: str) -> RevealResponse:
    state = _get_session(session_id)
    q = state.current()
    if q is None:
        raise HTTPException(409, "Session has no current question.")
    return RevealResponse(
        question_id=q["id"],
        prompt=q["prompt"],
        criteria=q.get("criteria", []),
        rubric=q.get("rubric", {}),
    )


@app.post("/api/session/{session_id}/self_grade", response_model=GradeResponse)
def self_grade(session_id: str, req: SelfGradeRequest) -> GradeResponse:
    state = _get_session(session_id)
    q = state.current()
    if q is None:
        raise HTTPException(409, "Session has no current question.")

    rubric_level = q.get("rubric", {}).get(req.level, {})
    grade = {
        "level": req.level,
        "score_pct": level_to_score(req.level),
        "criteria_met": rubric_level.get("criteria_met", []),
        "criteria_missing": rubric_level.get("criteria_missing", []),
        "strengths": [],
        "gaps": [],
        "what_would_elevate": "",
        "model_answer_pointer": req.level,
    }
    state.set_grade(q["id"], grade)

    has_next = state.advance()
    next_view = _question_view(state) if has_next else None
    return GradeResponse(grade=grade, has_next=has_next, next_question=next_view)


@app.post("/api/session/{session_id}/end", response_model=Scorecard)
def end_session(session_id: str) -> Scorecard:
    state = _get_session(session_id)
    if not state.grades:
        raise HTTPException(400, "Cannot end a session with no graded questions.")
    scorecard = finalize.finalize(state)
    STORE.drop(session_id)
    return Scorecard(**scorecard)


# ── static frontend (registered LAST so /api/* always wins) ──────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
