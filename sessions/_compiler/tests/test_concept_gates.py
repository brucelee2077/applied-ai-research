import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import v8lib
def _compile_mini():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(HERE, '..', 'shells', 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta


# ---------------------------------------------------------------------------
# TASK 9 — concept_shell_gate (runs on compiled HTML)
# ---------------------------------------------------------------------------
import concept_shell_gate

def test_concept_shell_gate_passes_valid_lesson():
    html, meta = _compile_mini()
    ok, msgs = concept_shell_gate.run(html, meta)
    assert ok, '\n'.join(msgs)

def test_concept_shell_gate_fails_when_a_concept_has_no_visual():
    html, meta = _compile_mini()
    import re
    broken = re.sub(r'(id="c1".*?)<div class="build-viz">.*?</div>(.*?</section>)', r'\1\2', html, count=1, flags=re.DOTALL)
    ok, msgs = concept_shell_gate.run(broken, meta)
    assert not ok
    assert any('visual' in m.lower() for m in msgs)


# --- FAIL-path coverage for the remaining concept_shell_gate invariants ---
def _mut_leaked_marker(html):
    # inject a stray region marker into the content
    return html.replace('<main id="content">', '<main id="content">@@@ stray')

def _mut_wrong_quiz_count(html):
    # drop the last quiz question so the count is 3, not 4
    return html.replace('<div class="q" data-correct="1"><div class="q-ask">All ReLUs output 0?</div>',
                        '<div class="q" data-correct="1"><div class="q-ask">All ReLUs output 0?</div>'.replace('class="q"', 'class="qx"'))

def _mut_break_nav_parity(html):
    # remove the c2 sidebar nav button -> targets no longer match section ids
    return html.replace('<button class="nav-link" data-target="c2"><span class="nl-dot"></span>2 · The bend</button>', '')

def _mut_missing_produce_artifact(html):
    # drop the experiment.py reference from the produce section
    return html.replace('experiment.py', 'notebook')

@pytest.mark.parametrize('mutate, needle', [
    (_mut_leaked_marker, 'marker'),
    (_mut_wrong_quiz_count, 'question'),
    (_mut_break_nav_parity, 'parity'),
    (_mut_missing_produce_artifact, 'artifact'),
])
def test_concept_shell_gate_fails_on_mutation(mutate, needle):
    html, meta = _compile_mini()
    ok, msgs = concept_shell_gate.run(mutate(html), meta)
    assert not ok, 'gate should reject: ' + needle
    assert any(needle in m.lower() and m.startswith('FAIL') for m in msgs), \
        'expected a FAIL msg mentioning %r; got:\n%s' % (needle, '\n'.join(msgs))


def test_concept_shell_gate_anchored_slice_tolerates_svg_id_collision():
    """FIX A: an author SVG in c1 that uses a literal id="c2" must not corrupt the
    per-concept slice — anchoring on the <section> open tag keeps c1 and c2 distinct,
    so BOTH still register their own visual and the gate PASSES."""
    html, meta = _compile_mini()
    import re
    # inject a literal id="c2" INSIDE concept c1's svg (SVG gradients/clipPaths do this)
    c1 = re.search(r'<section class="module-section" id="c1".*?</section>', html, re.DOTALL).group(0)
    c1_poisoned = c1.replace('<svg ', '<svg id="c2" ', 1)
    assert c1_poisoned != c1
    html2 = html.replace(c1, c1_poisoned, 1)
    assert html2.count('id="c2"') >= 2  # the poison + the real c2 section
    ok, msgs = concept_shell_gate.run(html2, meta)
    assert ok, 'anchored slice should still pass despite id="c2" inside c1:\n' + '\n'.join(msgs)
    # both concepts still report a visual (no slice cross-contamination)
    assert sum(1 for m in msgs if m == 'pass concept c1 has a visual') == 1
    assert sum(1 for m in msgs if m == 'pass concept c2 has a visual') == 1


# ---------------------------------------------------------------------------
# TASK 10 — reader_flow_gate concept-mode branch (runs on SOURCE blocks)
# ---------------------------------------------------------------------------
import reader_flow_gate
def _blocks_and_meta():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    return meta, v8lib.parse_blocks(body)

def test_reader_flow_concept_mode_passes():
    meta, blocks = _blocks_and_meta()
    ok, msgs = reader_flow_gate.run(meta, blocks)
    assert ok, '\n'.join(msgs)
    assert not any('s1' in m and 'FAIL' in m for m in msgs)

def test_reader_flow_concept_fails_concept_without_visual():
    meta, blocks = _blocks_and_meta()
    for b in blocks:
        if b['type']=='concept' and b['args'].get('id')=='c1':
            b['lines'] = ['Intro prose only, no visual.', 'Build-up prose only.']
    ok, msgs = reader_flow_gate.run(meta, blocks)
    assert not ok
    assert any('visual' in m.lower() and 'c1' in m for m in msgs)
