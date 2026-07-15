import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_judge as cj


# ---------------------------------------------------------------------------
# _extract_json — tolerant JSON parsing (deterministic, no network)
# ---------------------------------------------------------------------------
def test_extract_plain_json():
    d = cj._extract_json('{"execution": [], "summary": "ok"}')
    assert d == {'execution': [], 'summary': 'ok'}

def test_extract_fenced_json():
    d = cj._extract_json('```json\n{"a": 1}\n```')
    assert d == {'a': 1}

def test_extract_json_amid_prose():
    d = cj._extract_json('Sure, here is the verdict:\n{"skill_gaps": [{"concept":"x"}]}\nDone.')
    assert d == {'skill_gaps': [{'concept': 'x'}]}

def test_extract_returns_none_on_garbage():
    assert cj._extract_json('no json here at all') is None
    assert cj._extract_json('') is None


# ---------------------------------------------------------------------------
# _prompt — includes spec, curation, notebook concepts, lesson text (no network)
# ---------------------------------------------------------------------------
def test_prompt_contains_all_sections():
    spec = [('relu', ['relu']), ('leaky relu', ['leaky relu'])]
    nb = ['Step Function', 'Leaky ReLU']
    curation = {'deferred': {'softmax': 'day-04'}, 'out_of_scope': ['prelu']}
    p = cj._prompt('the lesson body about relu', spec, nb, curation)
    assert 'relu' in p and 'leaky relu' in p
    assert 'Step Function' in p and 'Leaky ReLU' in p
    assert 'softmax -> day-04' in p
    assert 'prelu' in p
    assert 'the lesson body about relu' in p
    # asks for strict JSON with the three axes
    assert 'execution' in p and 'skill_gaps' in p and 'intuition' in p


def test_prompt_truncates_long_lesson():
    long_text = 'x' * (cj._MAX_LESSON_CHARS + 5000)
    p = cj._prompt(long_text, [('relu', ['relu'])], [], None)
    # the lesson slice must not exceed the cap (plus the surrounding template)
    assert long_text[:cj._MAX_LESSON_CHARS] in p
    assert long_text not in p  # full untruncated body is NOT included


# ---------------------------------------------------------------------------
# judge() never raises and degrades gracefully when the SDK/bridge is absent
# ---------------------------------------------------------------------------
def test_judge_graceful_when_sdk_missing(monkeypatch):
    # Force the openai import inside judge() to fail -> BRIDGE_UNAVAILABLE, never raises.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'openai':
            raise ImportError('simulated missing sdk')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    res = cj.judge('lesson text', [('relu', ['relu'])], ['ReLU'], None)
    assert res['status'] == 'BRIDGE_UNAVAILABLE'
    assert res['execution'] == [] and res['skill_gaps'] == [] and res['intuition'] == []


# --- concept-structure judge ------------------------------------------------
def test_struct_prompt_contains_concepts_and_axes():
    p = cj._struct_prompt('lesson body about relu and sigmoid', ['ReLU', 'Sigmoid'])
    assert 'ReLU' in p and 'Sigmoid' in p
    assert 'intuition_first' in p and 'analogy' in p and 'buildup' in p
    assert 'lesson body about relu' in p

def test_struct_prompt_truncates_long_lesson():
    long_text = 'y' * (cj._STRUCT_MAX + 5000)
    p = cj._struct_prompt(long_text, ['ReLU'])
    assert long_text[:cj._STRUCT_MAX] in p
    assert long_text not in p

def test_judge_structure_graceful_when_sdk_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == 'openai':
            raise ImportError('simulated missing sdk')
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    res = cj.judge_concept_structure('lesson text', ['ReLU'])
    assert res['status'] == 'BRIDGE_UNAVAILABLE'
    assert res['concepts'] == [] and res['overall'] == 'N/A'

def test_judge_structure_empty_concepts_is_na():
    res = cj.judge_concept_structure('lesson', [])
    assert res['status'] == 'N/A'
