import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_judge as cj


# ---------------------------------------------------------------------------
# judge_interest_absolute — the notebook-FREE interest floor. Runs on EVERY
# lesson (unlike judge_interest, which needs a notebook). The bridge call is
# isolated behind cj._chat so it can be mocked (no network in tests).
# ---------------------------------------------------------------------------
def test_absolute_runs_without_notebook(monkeypatch):
    monkeypatch.setattr(cj, "_chat", lambda *a, **k:
        '{"dimensions":[{"name":"aspiration_hook","verdict":"GOOD"}],'
        '"overall":"FLOOR_MET","top_fixes":[],"summary":"ok"}')
    out = cj.judge_interest_absolute("some lesson text")   # NO notebook arg
    assert out["status"] == "OK"
    assert out["overall"] == "FLOOR_MET"
    assert out["dimensions"][0]["name"] == "aspiration_hook"


def test_absolute_below_floor(monkeypatch):
    monkeypatch.setattr(cj, "_chat", lambda *a, **k:
        '{"overall":"BELOW_FLOOR","dimensions":[],"top_fixes":["add a hook"],"summary":"dry"}')
    out = cj.judge_interest_absolute("t")
    assert out["overall"] == "BELOW_FLOOR"
    assert out["top_fixes"] == ["add a hook"]


def test_absolute_defaults_to_below_when_ambiguous(monkeypatch):
    # A response that does not clearly say FLOOR_MET must fail safe to BELOW_FLOOR
    # (flag for review) rather than silently pass.
    monkeypatch.setattr(cj, "_chat", lambda *a, **k:
        '{"overall":"maybe ok","dimensions":[],"top_fixes":[],"summary":"?"}')
    out = cj.judge_interest_absolute("t")
    assert out["overall"] == "BELOW_FLOOR"


def test_absolute_bridge_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no bridge")
    monkeypatch.setattr(cj, "_chat", boom)
    out = cj.judge_interest_absolute("t")
    assert out["status"] == "BRIDGE_UNAVAILABLE"
    assert out["overall"] == "N/A"   # graceful, never raises
