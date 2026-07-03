"""
Tests for the live AI interviewer path using a MOCKED OpenAI client (the genAI
bridge speaks the OpenAI Chat Completions protocol). Never hits the network.
Proves: probe loop, forced function-call grading, score anchoring to level
bands, and graceful degradation on connection errors.

Run from repo root:
    python -m pytest interview_app/tests/test_live_interviewer.py -q
"""

from __future__ import annotations

import json
import types

import httpx
import pytest
from fastapi.testclient import TestClient

import coach.core as cc
from interview_app.backend import llm_client as llm
from interview_app.backend import history, interviewer
from interview_app.backend.main import app


# ── fakes mimicking the OpenAI chat.completions response shape ───────────────
def _msg(content=None, tool_args=None):
    tool_calls = None
    if tool_args is not None:
        fn = types.SimpleNamespace(name="submit_grade", arguments=json.dumps(tool_args))
        tool_calls = [types.SimpleNamespace(function=fn)]
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class FakeCompletions:
    """Returns a probe on plain calls (from a script) and a grade on tool calls."""

    def __init__(self, script, grade_payload):
        self._script = list(script)
        self._grade = grade_payload

    def create(self, **kw):
        if kw.get("tools"):  # grading call (forced function tool)
            return _resp(_msg(tool_args=dict(self._grade)))
        nxt = self._script.pop(0) if self._script else interviewer.READY
        return _resp(_msg(content=nxt))


class FakeClient:
    def __init__(self, script, grade_payload):
        self.chat = types.SimpleNamespace(completions=FakeCompletions(script, grade_payload))


@pytest.fixture()
def live_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "hist")
    monkeypatch.setattr(history, "INDEX_PATH", tmp_path / "hist" / "index.json")
    monkeypatch.setattr(llm, "is_live_available", lambda: True)
    return TestClient(app)


def _install_client(monkeypatch, script, grade_payload):
    fake = FakeClient(script, grade_payload)
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    return fake


def test_drill_probe_then_grade(live_env, monkeypatch):
    grade_payload = {
        "level": "hire",
        "score_pct": 0.9,  # model overshoots; backend anchors into the hire band
        "criteria_met": ["scale", "latency"],
        "criteria_missing": ["feedback loop"],
        "strengths": ["clear framing"],
        "gaps": ["no IPS"],
        "what_would_elevate": "Discuss positional bias correction.",
        "model_answer_pointer": "strong_hire",
    }
    _install_client(monkeypatch, script=["Can you quantify the index memory at 200B vectors?"], grade_payload=grade_payload)
    client = live_env

    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "02-visual-search", "mode": "drill"},
    ).json()
    assert start["live"] is True
    sid = start["session_id"]

    r1 = client.post(f"/api/session/{sid}/answer", json={"text": "Use IVF-PQ over 200B vectors."}).json()
    assert r1["action"] == "probe"
    assert "memory" in r1["probe"].lower()

    r2 = client.post(f"/api/session/{sid}/answer", json={"text": "~6.4TB at 32 bytes/vector."}).json()
    assert r2["action"] == "grade_ready"

    g = client.post(f"/api/session/{sid}/grade").json()
    assert g["grade"]["level"] == "hire"
    assert abs(g["grade"]["score_pct"] - 0.75) <= 0.1
    assert g["has_next"] is False

    sc = client.post(f"/api/session/{sid}/end").json()
    assert sc["overall_level"] == "hire"
    state = json.loads(cc.STATE_PATH.read_text())
    assert state["player"]["xp"] == cc.XP_MOCK_INTERVIEW


def test_grade_degrades_to_reveal_on_connection_error(live_env, monkeypatch):
    class BoomCompletions:
        def create(self, **kw):
            if kw.get("tools"):
                raise llm.openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:11211"))
            return _resp(_msg(content=interviewer.READY))

    class BoomClient:
        chat = types.SimpleNamespace(completions=BoomCompletions())

    monkeypatch.setattr(llm, "get_client", lambda: BoomClient())
    client = live_env

    sid = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "06-video-recommendation", "mode": "drill"},
    ).json()["session_id"]

    a = client.post(f"/api/session/{sid}/answer", json={"text": "An answer."}).json()
    assert a["action"] == "grade_ready"

    g = client.post(f"/api/session/{sid}/grade").json()
    assert g["degraded"] is True
    assert g["grade"] is None

    reveal = client.get(f"/api/session/{sid}/reveal").json()
    assert reveal["rubric"]
    level = "hire" if "hire" in reveal["rubric"] else sorted(reveal["rubric"])[0]
    sg = client.post(f"/api/session/{sid}/self_grade", json={"level": level}).json()
    assert sg["grade"]["level"] == level


def test_force_offline_even_when_live(live_env, monkeypatch):
    _install_client(monkeypatch, script=[], grade_payload={})
    client = live_env
    start = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "08-ad-click-prediction", "mode": "drill", "live": False},
    ).json()
    assert start["live"] is False
    a = client.post(f"/api/session/{start['session_id']}/answer", json={"text": "x"}).json()
    assert a["action"] == "reveal_ready"


def test_grade_degrades_on_malformed_grade(live_env, monkeypatch):
    """A grader that returns an out-of-enum level / non-numeric score makes
    _normalize_grade raise (ValidationError/ValueError) -> re-raised as
    RuntimeError -> grade_question degrades to reveal instead of 500ing."""
    bad = {
        "level": "excellent",        # not in the Level enum
        "score_pct": "very high",    # not a float
        "criteria_met": [], "criteria_missing": [],
        "strengths": [], "gaps": [], "what_would_elevate": "", "model_answer_pointer": "",
    }
    _install_client(monkeypatch, script=[interviewer.READY], grade_payload=bad)
    client = live_env

    sid = client.post(
        "/api/session/start",
        json={"track": "ml_system_design", "module_id": "06-video-recommendation", "mode": "drill"},
    ).json()["session_id"]

    a = client.post(f"/api/session/{sid}/answer", json={"text": "x"}).json()
    assert a["action"] == "grade_ready"

    g = client.post(f"/api/session/{sid}/grade")
    assert g.status_code == 200, "malformed grade must degrade, not 500"
    body = g.json()
    assert body["degraded"] is True and body["grade"] is None
