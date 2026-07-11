import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
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
