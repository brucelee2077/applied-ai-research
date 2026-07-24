import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_judge as cj


# ---------------------------------------------------------------------------
# judge_body_engagement — the per-concept BODY-voice floor. Grades whether each
# concept's BUILD-UP prose stays engaging (spine analogy alive, re-hook, narrated
# vs cold Step 1/2/3 dump, predict/discovery) — NOT structure. Own _chat-seam
# judge (not a 5th structure axis) so it is mockable + has its own token budget.
# ---------------------------------------------------------------------------
def test_body_engagement_parses_per_concept(monkeypatch):
    monkeypatch.setattr(cj, "_chat", lambda *a, **k:
        '{"concepts":[{"concept":"c1","body_engagement":"MISSING","note":"cold dump","fix":"add voice"},'
        '{"concept":"c2","body_engagement":"GOOD","note":"narrated","fix":""}],'
        '"overall":"WEAK","summary":"mixed"}')
    out = cj.judge_body_engagement("lesson text", ["c1", "c2"])
    assert out["status"] == "OK"
    assert out["concepts"][0]["body_engagement"] == "MISSING"
    assert out["concepts"][1]["body_engagement"] == "GOOD"
    assert out["overall"] == "WEAK"


def test_body_engagement_na_without_concepts():
    out = cj.judge_body_engagement("text", [])
    assert out["status"] == "N/A"
    assert out["concepts"] == []


def test_body_engagement_bridge_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no bridge")
    monkeypatch.setattr(cj, "_chat", boom)
    out = cj.judge_body_engagement("t", ["c1"])
    assert out["status"] == "BRIDGE_UNAVAILABLE"   # graceful, never raises
    assert out["concepts"] == []


def test_body_engagement_prompt_names_axis_and_na_triggers():
    # The judge must instruct on the body_engagement axis, spell out the NA escape
    # (recap unit / definitional one-liner), and reserve MISSING for a genuinely COLD body.
    sysmsg = cj._BODY_SYS
    prompt = cj._body_prompt("some lesson text", ["c1", "c2"])
    for token in ("body_engagement", "cold"):
        assert token in (sysmsg + prompt).lower()
    assert "recap" in (sysmsg + prompt).lower()      # NA trigger spelled out
    assert "c1" in prompt and "c2" in prompt          # concept names passed in
