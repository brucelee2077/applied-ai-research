"""
Integration tests for the offline interview flow.

Uses a FastAPI TestClient and monkeypatches the COACH state path + history dir
to a temp location so the real coach/state.json and history are never touched.

Run from the repo root:
    python -m pytest interview_app/tests/test_backend_offline.py -q
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import coach.core as cc
from interview_app.backend import history
from interview_app.backend import llm_client as llm
from interview_app.backend.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Redirect all COACH state + history writes into the temp dir.
    state_path = tmp_path / "state.json"
    hist_dir = tmp_path / "interview_history"
    monkeypatch.setattr(cc, "STATE_PATH", state_path)
    monkeypatch.setattr(history, "HISTORY_DIR", hist_dir)
    monkeypatch.setattr(history, "INDEX_PATH", hist_dir / "index.json")
    # Force offline so these tests exercise the reveal/self-grade path
    # deterministically (never touch the real genAI bridge).
    monkeypatch.setattr(llm, "is_live_available", lambda: False)
    return TestClient(app)


def test_health_offline(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "live_available" in body and "model" in body


def test_tracks_lists_ml_system_design(client):
    r = client.get("/api/tracks")
    assert r.status_code == 200
    tracks = {t["track"]: t for t in r.json()}
    assert "ml_system_design" in tracks
    msd = tracks["ml_system_design"]
    assert msd["question_count"] > 0
    assert any(m["module_id"] == "02-visual-search" for m in msd["modules"])


def test_full_mock_offline_flow_awards_xp(client):
    # Baseline XP from the temp state.
    xp_before = client.get("/api/progress").json()["xp"]

    # Start a full mock on a real ML Design module.
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "02-visual-search", "mode": "full_mock"},
    ).json()
    session_id = start["session_id"]
    total = start["total_questions"]
    assert total >= 2
    assert start["question"]["index"] == 0

    # Walk every question: answer -> reveal -> self_grade as "hire".
    graded = 0
    while True:
        ans = client.post(f"/api/session/{session_id}/answer", json={"text": "My answer."}).json()
        assert ans["action"] == "reveal_ready"

        reveal = client.get(f"/api/session/{session_id}/reveal").json()
        assert reveal["rubric"], "reveal should expose rubric levels"
        # Pick a level that actually exists in this question's rubric.
        level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]

        sg = client.post(f"/api/session/{session_id}/self_grade", json={"level": level}).json()
        graded += 1
        assert 0.0 <= sg["grade"]["score_pct"] <= 1.0
        if not sg["has_next"]:
            break
    assert graded == total

    # End -> scorecard.
    sc = client.post(f"/api/session/{session_id}/end").json()
    assert sc["module_id"] == "02-visual-search"
    assert len(sc["per_question"]) == total
    assert sc["xp"]["xp_awarded"] == cc.XP_MOCK_INTERVIEW
    assert sc["overall_band"] in {"No Hire", "Weak Hire", "Hire", "Strong Hire"}

    # XP landed in the (temp) state.json.
    state = json.loads(cc.STATE_PATH.read_text())
    assert state["player"]["xp"] == xp_before + cc.XP_MOCK_INTERVIEW
    mod = state["modules"]["02-visual-search"]
    assert mod["boss_attempted"] is True
    assert mod["boss_passed"] == (sc["overall_score_pct"] >= 0.5)
    assert any(b["module"] == "02-visual-search" for b in state["boss_battles"]["completed"])

    # History was written.
    idx = client.get("/api/history").json()
    assert any(e["session_id"] == session_id for e in idx)
    detail = client.get(f"/api/history/{session_id}").json()
    assert len(detail["questions"]) == total
    assert detail["questions"][0]["transcript"], "transcript should be persisted"


def test_drill_offline_single_question(client):
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "06-video-recommendation", "mode": "drill"},
    ).json()
    assert start["total_questions"] == 1
    sid = start["session_id"]
    client.post(f"/api/session/{sid}/answer", json={"text": "Answer."})
    reveal = client.get(f"/api/session/{sid}/reveal").json()
    level = "strong_hire" if "strong_hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
    sg = client.post(f"/api/session/{sid}/self_grade", json={"level": level}).json()
    assert sg["has_next"] is False
    sc = client.post(f"/api/session/{sid}/end").json()
    assert sc["overall_score_pct"] == sg["grade"]["score_pct"]


def test_net_new_track_commits_xp_natively(client):
    """Behavioral (a net-new track) is registered in ALL_MODULES, so XP lands in
    modules['behavioral'] via record_boss_result, not the fallback path."""
    tracks = {t["track"] for t in client.get("/api/tracks").json()}
    assert {"frontier_research", "behavioral"}.issubset(tracks)

    start = client.post(
        "/api/session/start",
        json={"track": "behavioral", "module_id": "behavioral", "mode": "drill"},
    ).json()
    sid = start["session_id"]
    client.post(f"/api/session/{sid}/answer", json={"text": "A STAR-structured story."})
    reveal = client.get(f"/api/session/{sid}/reveal").json()
    level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
    client.post(f"/api/session/{sid}/self_grade", json={"level": level})
    client.post(f"/api/session/{sid}/end")

    state = json.loads(cc.STATE_PATH.read_text())
    assert state["modules"]["behavioral"]["boss_attempted"] is True
    assert any(b["module"] == "behavioral" for b in state["boss_battles"]["completed"])


# ── read-endpoint contracts ──────────────────────────────────────────────────
def test_progress_shape(client):
    p = client.get("/api/progress").json()
    for key in ("xp", "level", "level_name", "xp_into_level", "xp_needed_for_next",
                "tokens", "percentile", "badges"):
        assert key in p, f"progress missing {key}"
    assert isinstance(p["badges"], list)


def test_tracks_shape_all_four_and_counts_add_up(client):
    tracks = {t["track"]: t for t in client.get("/api/tracks").json()}
    assert set(tracks) == {"ml_system_design", "fundamentals", "frontier_lab", "frontier_research", "behavioral"}
    for t in tracks.values():
        assert t["question_count"] == sum(m["question_count"] for m in t["modules"])
        for m in t["modules"]:
            assert m["question_count"] == len(m["questions"])


def test_history_empty_index_and_404(client):
    assert client.get("/api/history").json() == []
    assert client.get("/api/history/does-not-exist").status_code == 404


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/api/session/nope/reveal", None),
        ("post", "/api/session/nope/answer", {"text": "x"}),
        ("post", "/api/session/nope/grade", {}),
        ("post", "/api/session/nope/self_grade", {"level": "hire"}),
        ("post", "/api/session/nope/end", {}),
    ],
)
def test_unknown_session_returns_404(client, method, path, body):
    r = client.get(path) if method == "get" else client.post(path, json=body)
    assert r.status_code == 404


def test_end_with_no_grades_returns_400(client):
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "02-visual-search", "mode": "drill"},
    ).json()
    r = client.post(f"/api/session/{start['session_id']}/end")
    assert r.status_code == 400


def test_self_grade_advances_through_a_multi_question_session(client):
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "02-visual-search", "mode": "full_mock"},
    ).json()
    total = start["total_questions"]
    assert total >= 2
    sid = start["session_id"]

    idx = 0
    while True:
        client.post(f"/api/session/{sid}/answer", json={"text": "a"})
        reveal = client.get(f"/api/session/{sid}/reveal").json()
        level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
        sg = client.post(f"/api/session/{sid}/self_grade", json={"level": level}).json()
        if sg["has_next"]:
            assert sg["next_question"] is not None
            assert sg["next_question"]["index"] == idx + 1
            assert sg["next_question"]["total"] == total
            idx += 1
        else:
            assert sg["next_question"] is None
            break
    assert idx == total - 1  # walked every question exactly once


def test_criteria_coverage_aggregation_math():
    import datetime

    from interview_app.backend.finalize import _criteria_coverage
    from interview_app.backend.sessions import SessionState

    st = SessionState(
        session_id="t", track="x", mode="drill", module_id="m", topic="t",
        questions=[{"id": "q1"}, {"id": "q2"}], live=False,
        started_at=datetime.datetime.now(),
    )
    st.set_grade("q1", {"criteria_met": ["a", "b"], "criteria_missing": ["c"]})
    st.set_grade("q2", {"criteria_met": ["a"], "criteria_missing": ["b"]})
    cov = _criteria_coverage(st)
    assert cov["a"] == 1.0    # met 2 / seen 2
    assert cov["b"] == 0.5    # met 1 / seen 2
    assert cov["c"] == 0.0    # met 0 / seen 1


def test_scorecard_carries_full_transcript_record(client):
    """The /end scorecard now includes the same transcript-bearing `questions`
    the history detail has, so "Review transcript" works immediately without a
    second fetch (regression guard for the empty-review bug)."""
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "06-video-recommendation", "mode": "drill"},
    ).json()
    sid = start["session_id"]
    client.post(f"/api/session/{sid}/answer", json={"text": "A structured answer with trade-offs."})
    reveal = client.get(f"/api/session/{sid}/reveal").json()
    level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
    client.post(f"/api/session/{sid}/self_grade", json={"level": level})

    sc = client.post(f"/api/session/{sid}/end").json()
    assert "questions" in sc and len(sc["questions"]) == start["total_questions"]
    q0 = sc["questions"][0]
    assert q0.get("prompt"), "scorecard question must carry its prompt"
    assert q0.get("transcript"), "scorecard question must carry the transcript turns"
    assert q0.get("grade"), "scorecard question must carry the grade"


def test_end_survives_corrupt_state_json(client):
    """A corrupt coach/state.json must not 500 /end or drop the finished session:
    XP degrades to zero but the scorecard + history are still produced."""
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "06-video-recommendation", "mode": "drill"},
    ).json()
    sid = start["session_id"]
    client.post(f"/api/session/{sid}/answer", json={"text": "a"})
    reveal = client.get(f"/api/session/{sid}/reveal").json()
    level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
    client.post(f"/api/session/{sid}/self_grade", json={"level": level})

    cc.STATE_PATH.write_text("{ this is not valid json ")  # corrupt right before commit

    r = client.post(f"/api/session/{sid}/end")
    assert r.status_code == 200, "corrupt state must degrade, not 500"
    sc = r.json()
    assert sc["xp"]["xp_awarded"] == 0
    assert any(e["session_id"] == sid for e in client.get("/api/history").json())
