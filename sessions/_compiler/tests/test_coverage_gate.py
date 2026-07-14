import os, sys, json
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_gate


# ---------------------------------------------------------------------------
# helper: write a tiny notebook (the TEST ORACLE) with the given markdown headings
# ---------------------------------------------------------------------------
def _nb(tmp_path, headings, name='nb.ipynb'):
    cells = [{'cell_type': 'markdown', 'source': ['## ' + h + '\n']} for h in headings]
    p = tmp_path / name
    p.write_text(json.dumps({'cells': cells}))
    return str(p)


# ===========================================================================
# N/A cases — nothing to check
# ===========================================================================
def test_na_when_no_spec_and_no_notebook():
    status, msgs = coverage_gate.run('<p>relu sigmoid</p>', {'notebook_yardstick': None})
    assert status == 'N/A'
    assert any('N/A' in m for m in msgs)

def test_na_when_yardstick_literal_null_and_no_spec():
    status, _ = coverage_gate.run('<p>x</p>', {'notebook_yardstick': 'null'})
    assert status == 'N/A'


# ===========================================================================
# CHECK A — execution: does the lesson realize the SKILL-DRAFTED spec?
# ===========================================================================
def test_check_a_flags_exec_gap():
    meta = {'notebook_yardstick': None, 'coverage_topics': ['relu', 'softmax']}
    html = '<h2>ReLU</h2><p>the relu activation keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'ADVISORY'
    assert any(m.startswith('A/EXEC-GAP') and 'softmax' in m for m in msgs)
    assert any(m == 'A/covered: relu' for m in msgs)

def test_check_a_deferred_is_not_a_gap():
    meta = {'notebook_yardstick': None, 'coverage_topics': ['relu', 'softmax']}
    html = '<p>the relu activation keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta, curation={'deferred': {'softmax': 'day-04'}})
    assert status == 'PASS'
    assert any(m.startswith('A/defer: softmax') and 'day-04' in m for m in msgs)
    assert not any('EXEC-GAP' in m for m in msgs)

def test_check_a_keyword_dict_form():
    meta = {'notebook_yardstick': None,
            'coverage_topics': [{'topic': 'vanishing gradient', 'keywords': ['gradient shrinks']}]}
    html = '<p>as you go deeper the gradient shrinks toward zero</p>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'PASS'
    assert any(m == 'A/covered: vanishing gradient' for m in msgs)

def test_check_a_script_body_no_phantom_coverage():
    meta = {'notebook_yardstick': None, 'coverage_topics': ['softmax']}
    html = '<p>relu</p><script>function softmax(){return 1}</script>'
    status, msgs = coverage_gate.run(html, meta)
    assert status == 'ADVISORY'
    assert any(m.startswith('A/EXEC-GAP') and 'softmax' in m for m in msgs)


# ===========================================================================
# CHECK B — skill eval: does the notebook teach a concept the SPEC missed?
# This is the inversion: notebook = held-out TEST, not the coverage source.
# ===========================================================================
def test_check_b_flags_notebook_concept_missing_from_spec(tmp_path):
    """The core failure this redesign fixes: step + leaky are in the notebook,
    absent from the spec and lesson, and NOT curated -> SKILL-GAP."""
    nb = _nb(tmp_path, ['Activation Function #1: Step Function',
                        'Activation Function #4: ReLU',
                        'Activation Function #5: Leaky ReLU'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu']}
    html = '<p>relu keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta, root=str(tmp_path))
    assert status == 'ADVISORY'
    assert any(m.startswith('B/SKILL-GAP') and 'Step Function' in m for m in msgs), '\n'.join(msgs)
    assert any(m.startswith('B/SKILL-GAP') and 'Leaky ReLU' in m for m in msgs), '\n'.join(msgs)

def test_check_b_relu_in_spec_does_NOT_account_for_leaky_relu(tmp_path):
    """Whole-concept guard: a spec entry 'relu' must not silently cover 'leaky relu'."""
    nb = _nb(tmp_path, ['Activation Function #5: Leaky ReLU'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu']}
    status, msgs = coverage_gate.run('<p>relu</p>', meta, root=str(tmp_path))
    assert any(m.startswith('B/SKILL-GAP') and 'Leaky ReLU' in m for m in msgs), '\n'.join(msgs)

def test_check_b_oos_elu_does_NOT_account_for_relu_or_leaky(tmp_path):
    """Short-token guard: out-of-scope 'elu' must not swallow 'relu'/'leaky relu'
    (the 'elu' ⊂ 'relu' substring trap)."""
    nb = _nb(tmp_path, ['Activation Function #4: ReLU', 'Activation Function #5: Leaky ReLU'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['sigmoid']}
    status, msgs = coverage_gate.run('<p>sigmoid</p>', meta, root=str(tmp_path),
                                     curation={'out_of_scope': ['elu']})
    assert any(m.startswith('B/SKILL-GAP') and m.strip().endswith('re-eval') and 'ReLU' in m for m in msgs), '\n'.join(msgs)
    # both ReLU and Leaky ReLU should be skill-gaps, not accounted by 'elu'
    gaps = [m for m in msgs if m.startswith('B/SKILL-GAP')]
    assert any('Leaky ReLU' in g for g in gaps) and any(g for g in gaps if 'ReLU' in g and 'Leaky' not in g), '\n'.join(msgs)

def test_check_b_spec_accounts_when_concept_listed(tmp_path):
    nb = _nb(tmp_path, ['Activation Function #1: Step Function',
                        'Activation Function #5: Leaky ReLU'])
    meta = {'notebook_yardstick': 'nb.ipynb',
            'coverage_topics': ['step function', 'leaky relu']}
    html = '<p>step function is the on/off ancestor; leaky relu leaks a little</p>'
    status, msgs = coverage_gate.run(html, meta, root=str(tmp_path))
    assert status == 'PASS', '\n'.join(msgs)
    assert not any(m.startswith('B/SKILL-GAP') for m in msgs)

def test_check_b_deferred_and_oos_account_for_notebook_concepts(tmp_path):
    nb = _nb(tmp_path, ['Activation Function #6: Softmax', 'Activation Function #1: Step Function'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu']}
    status, msgs = coverage_gate.run('<p>relu</p>', meta, root=str(tmp_path),
                                     curation={'deferred': {'softmax': 'day-04'},
                                               'out_of_scope': ['step function']})
    assert status == 'PASS', '\n'.join(msgs)
    assert not any(m.startswith('B/SKILL-GAP') for m in msgs)

def test_check_b_defer_temperature_covers_temperature_scaling(tmp_path):
    """A general deferred key ('temperature') should cover a more specific
    notebook concept ('Temperature Scaling in Softmax')."""
    nb = _nb(tmp_path, ['Temperature Scaling in Softmax'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu']}
    status, msgs = coverage_gate.run('<p>relu</p>', meta, root=str(tmp_path),
                                     curation={'deferred': {'temperature': 'm05a'}})
    assert status == 'PASS', '\n'.join(msgs)

def test_check_b_chrome_headings_filtered(tmp_path):
    nb = _nb(tmp_path, ['When to Use This', 'Key Takeaways', "What's Next",
                        'Visualizing Sigmoid', 'Activation Function #2: Sigmoid'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['sigmoid']}
    status, msgs = coverage_gate.run('<p>sigmoid</p>', meta, root=str(tmp_path))
    assert status == 'PASS', '\n'.join(msgs)
    assert not any(m.startswith('B/SKILL-GAP') for m in msgs)


# ===========================================================================
# Determinism + sidecar + real manifest
# ===========================================================================
def test_determinism_identical_runs(tmp_path):
    nb = _nb(tmp_path, ['Activation Function #5: Leaky ReLU'])
    meta = {'notebook_yardstick': 'nb.ipynb', 'coverage_topics': ['relu', 'sigmoid']}
    html = '<p>relu and sigmoid</p>'
    r1 = coverage_gate.run(html, meta, root=str(tmp_path))
    r2 = coverage_gate.run(html, meta, root=str(tmp_path))
    assert r1 == r2

def test_writes_sidecar_with_both_checks_and_never_touches_html(tmp_path):
    d = tmp_path / 'day-x'
    d.mkdir()
    meta = {'notebook_yardstick': None, 'coverage_topics': ['relu', 'softmax']}
    coverage_gate.run('<p>relu</p>', meta, source_dir=str(d))
    sidecar = d / '_coverage.md'
    assert sidecar.exists()
    body = sidecar.read_text()
    assert 'Check A' in body and 'Check B' in body and 'EXEC-GAP' in body
    assert not (d / 'lesson.html').exists()

def test_real_m02_manifest_curation_defers_softmax_and_weight_init():
    repo = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
    day_dir = os.path.join(repo, 'sessions', 'm02-the-neuron', 'day-02-activations')
    if not os.path.isdir(day_dir):
        import pytest
        pytest.skip('m02 day-02 not present')
    # front-matter spec overrides manifest covers; softmax + weight-init are deferred in the manifest.
    meta = {'notebook_yardstick': None,
            'coverage_topics': ['relu', 'softmax', 'weight initialization']}
    html = '<p>relu keeps positives</p>'
    status, msgs = coverage_gate.run(html, meta, source_dir=day_dir)
    assert any(m.startswith('A/defer: softmax') for m in msgs), '\n'.join(msgs)
    assert any(m.startswith('A/defer: weight initialization') for m in msgs), '\n'.join(msgs)
    sc = os.path.join(day_dir, '_coverage.md')
    if os.path.exists(sc):
        os.remove(sc)
