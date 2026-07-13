import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_gate


# ---------------------------------------------------------------------------
# 1. N/A when no notebook yardstick.
# ---------------------------------------------------------------------------
def test_na_when_no_notebook_yardstick():
    status, msgs = coverage_gate.run('<p>relu sigmoid</p>', {'notebook_yardstick': None})
    assert status == 'N/A'
    assert any('N/A' in m for m in msgs)

def test_na_when_yardstick_literal_null_string():
    status, _ = coverage_gate.run('<p>x</p>', {'notebook_yardstick': 'null'})
    assert status == 'N/A'


# ---------------------------------------------------------------------------
# 2. Deterministic diff: coverage_topics, one covered one missing, no curation.
# ---------------------------------------------------------------------------
def test_deterministic_diff_flags_uncovered_topic():
    meta = {'notebook_yardstick': 'some/nb.ipynb',
            'coverage_topics': ['relu', 'softmax']}
    html = '<h2>ReLU</h2><p>the relu activation keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'ADVISORY'
    # softmax is a GAP; relu is covered
    assert any(m.startswith('GAP') and 'softmax' in m for m in msgs)
    assert any(m == 'pass covered: relu' for m in msgs)
    assert not any(m.startswith('GAP') and 'relu' in m.lower() and 'softmax' not in m.lower()
                   for m in msgs)


# ---------------------------------------------------------------------------
# 3. Curation suppresses a gap (injected curation dict).
# ---------------------------------------------------------------------------
def test_curation_deferred_suppresses_gap():
    meta = {'notebook_yardstick': 'some/nb.ipynb',
            'coverage_topics': ['relu', 'softmax']}
    html = '<p>the relu activation keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta,
                                     curation={'deferred': {'softmax': 'day-04'}})
    assert status == 'PASS'
    assert any(m.startswith('defer softmax') and 'day-04' in m for m in msgs)
    assert not any(m.startswith('GAP') for m in msgs)

def test_curation_out_of_scope_suppresses_gap():
    meta = {'notebook_yardstick': 'some/nb.ipynb',
            'coverage_topics': ['relu', 'step function']}
    html = '<p>relu keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta,
                                     curation={'out_of_scope': ['step function']})
    assert status == 'PASS'
    assert any(m.startswith('oos step function') for m in msgs)
    assert not any(m.startswith('GAP') for m in msgs)


# ---------------------------------------------------------------------------
# 4. Determinism: two runs on identical input give identical (status, msgs).
# ---------------------------------------------------------------------------
def test_determinism_identical_runs():
    meta = {'notebook_yardstick': 'some/nb.ipynb',
            'coverage_topics': ['relu', 'softmax', 'tanh', 'sigmoid']}
    html = '<p>relu and tanh and sigmoid, three activations</p>'
    r1 = coverage_gate.run(html, meta)
    r2 = coverage_gate.run(html, meta)
    assert r1 == r2
    assert r1[0] == 'ADVISORY'  # softmax uncovered


# ---------------------------------------------------------------------------
# Extra: keyword form of coverage_topics; script text does not fake coverage.
# ---------------------------------------------------------------------------
def test_coverage_topics_keyword_dict_form():
    meta = {'notebook_yardstick': 'nb.ipynb',
            'coverage_topics': [{'topic': 'vanishing gradient', 'keywords': ['gradient shrinks']}]}
    html = '<p>as you go deeper the gradient shrinks toward zero</p>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'PASS'
    assert any(m == 'pass covered: vanishing gradient' for m in msgs)

def test_script_body_does_not_create_phantom_coverage():
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['softmax']}
    html = '<p>relu</p><script>function softmax(){return 1}</script>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'ADVISORY'
    assert any(m.startswith('GAP') and 'softmax' in m for m in msgs)


# ---------------------------------------------------------------------------
# Extra: never raises regardless of status; sidecar written when source_dir set.
# ---------------------------------------------------------------------------
def test_writes_sidecar_and_never_touches_html(tmp_path):
    d = tmp_path / 'day-x'
    d.mkdir()
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu', 'softmax']}
    status, _ = coverage_gate.run('<p>relu</p>', meta, source_dir=str(d))
    sidecar = d / '_coverage.md'
    assert sidecar.exists()
    body = sidecar.read_text()
    assert 'GAP' in body and 'COVERED' in body
    # no lesson.html was created by the gate
    assert not (d / 'lesson.html').exists()


# ---------------------------------------------------------------------------
# Extra: real manifest read — softmax curated in m02 manifest is NOT a gap.
# ---------------------------------------------------------------------------
def test_real_m02_manifest_curation_clears_softmax():
    repo = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
    day_dir = os.path.join(repo, 'sessions', 'm02-the-neuron', 'day-02-activations')
    if not os.path.isdir(day_dir):
        import pytest
        pytest.skip('m02 day-02 not present')
    meta = {'notebook_yardstick': '00-neural-networks/fundamentals/03_activation_functions.ipynb',
            'coverage_topics': ['relu', 'softmax', 'weight initialization']}
    html = '<p>relu keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta, source_dir=day_dir)
    # softmax -> day-04-loss and weight initialization -> day-06 in the manifest
    assert any(m.startswith('defer softmax') for m in msgs), '\n'.join(msgs)
    assert any(m.startswith('defer weight initialization') for m in msgs), '\n'.join(msgs)
    assert status == 'PASS'
    # clean up the advisory sidecar the run wrote into the real day dir
    sc = os.path.join(day_dir, '_coverage.md')
    if os.path.exists(sc):
        os.remove(sc)
