import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import evidence_judge as ej


# ---------------------------------------------------------------------------
# _evidence_prompt — includes blog + code + output, the 5 axes, numbers_match
# (deterministic, no network)
# ---------------------------------------------------------------------------
def test_evidence_prompt_contains_all_three_artifacts_and_axes():
    p = ej._evidence_prompt('the blog prose body',
                            'print("experiment code here")',
                            'stdout: experiment output here')
    # all three artifacts present
    assert 'the blog prose body' in p
    assert 'print("experiment code here")' in p
    assert 'stdout: experiment output here' in p
    # numbers_match key literally present
    assert 'numbers_match' in p
    # all five axis names literally present
    for axis in ('technical_soundness', 'non_triviality',
                 'reproducibility', 'communication', 'numbers_match'):
        assert axis in p, 'prompt must name axis %r' % axis
    # asks for the structured JSON keys
    assert 'verdict' in p and 'findings' in p and 'summary' in p


def test_evidence_prompt_truncates_each_artifact():
    long = 'z' * (ej._EVID_MAX + 5000)
    p = ej._evidence_prompt(long, long, long)
    # each artifact slice is capped at _EVID_MAX; the full untruncated body
    # (which would appear 3x) is never emitted whole
    assert long[:ej._EVID_MAX] in p
    assert long not in p


# ---------------------------------------------------------------------------
# judge_evidence never raises and degrades gracefully when the SDK is absent
# ---------------------------------------------------------------------------
def test_judge_evidence_graceful_when_sdk_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'openai':
            raise ImportError('simulated missing sdk')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    res = ej.judge_evidence('blog', 'code', 'out')
    assert res['status'] == 'BRIDGE_UNAVAILABLE'
    assert res['verdict'] == 'WEAK'
    assert res['numbers_match'] is False
    assert res['findings'] == []
    assert res['summary'] == ''


def test_judge_evidence_never_raises_on_bad_client(monkeypatch):
    # SDK imports fine but the client call blows up -> still the graceful stub.
    import builtins
    real_import = builtins.__import__

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(*a, **k):
                    raise RuntimeError('bridge down')

    import types
    fake_openai = types.SimpleNamespace(OpenAI=_BoomClient)

    def fake_import(name, *a, **k):
        if name == 'openai':
            return fake_openai
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    res = ej.judge_evidence('blog', 'code', 'out')
    assert res['status'] == 'BRIDGE_UNAVAILABLE'
    assert res['findings'] == []


# ---------------------------------------------------------------------------
# _extract_json is reused from coverage_judge (single source), not re-defined
# ---------------------------------------------------------------------------
def test_extract_json_reused_from_coverage_judge():
    assert ej._extract_json('{"verdict": "OK"}') == {'verdict': 'OK'}
    assert ej._extract_json('garbage') is None
