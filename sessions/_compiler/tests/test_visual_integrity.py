import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import glob
import pytest
import visual_integrity_gate as vg

REAL = os.path.abspath(os.path.join(HERE, '..', '..', 'm02-the-neuron', 'day-02-activations', 'source.md'))
_REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

CONCEPT_HDR = """---
quest_id: t
mode: concept
donor: v9-base.donor
module_label: "T"
title: "T"
subtitle: "s"
brand_sub: "b"
spine: "bend"
notebook_yardstick: null
fin_title: "x"
fin_body: "y"
---
@@@ hero
@lede imagine a bend, you
@goal g
"""


def _write(tmp, name, text):
    p = os.path.join(str(tmp), name)
    open(p, 'w', encoding='utf-8').write(text)
    return p


# --- unit: degeneracy + protocol extraction (deterministic, no files) --------
def test_svg_degeneracy_unit():
    assert vg._svg_is_degenerate('<svg viewBox="0 0 10 10"><path d="M0 0"/></svg>')       # bare move
    assert vg._svg_is_degenerate('<svg><path d="M0 10 L5 0"/></svg>')                     # no viewBox
    assert vg._svg_is_degenerate('<svg viewBox="0 0 10 10"></svg>')                       # nothing drawn
    assert not vg._svg_is_degenerate('<svg viewBox="0 0 10 10"><path d="M0 10 L5 0"/></svg>')
    assert not vg._svg_is_degenerate('<svg viewBox="0 0 10 10"><rect x="1" y="1" width="3" height="3"/></svg>')


def test_expected_type_read_from_donor():
    # the shipped v9-base.donor receiver filters on 'viz-height'
    assert vg._expected_msg_type() == 'viz-height'


# --- the real lesson must pass (structural integrity holds) ------------------
def test_real_m02_lesson_passes():
    ok, msgs = vg.run(REAL)
    assert ok, '\n'.join(msgs)


# --- each silent-blank failure mode must be CAUGHT ---------------------------
def _concept_with_viz(rel):
    return (CONCEPT_HDR + '@@@ concept id=c1 tag="A" title="A" gotit="g"\nintro\n'
            + '%%% viz src=' + rel + ' title="x" caption="c"\n%%%\n')


def test_missing_viz_file_fails(tmp_path):
    p = _write(tmp_path, 'source.md', _concept_with_viz('./nope.html'))
    ok, msgs = vg.run(p)
    assert not ok and any('MISSING file' in m for m in msgs), msgs


def test_missing_local_dep_fails(tmp_path):
    _write(tmp_path, 'v.html', '<html><script src="./missing.js"></script>'
           '<script>window.parent.postMessage({type:"viz-height",px:1},"*")</script></html>')
    p = _write(tmp_path, 'source.md', _concept_with_viz('./v.html'))
    ok, msgs = vg.run(p)
    assert not ok and any('MISSING' in m and 'missing.js' in m for m in msgs), msgs


def test_no_height_sender_fails(tmp_path):
    _write(tmp_path, 'v.html', '<html><body>no sender here</body></html>')
    p = _write(tmp_path, 'source.md', _concept_with_viz('./v.html'))
    ok, msgs = vg.run(p)
    assert not ok and any('NO height-sender' in m for m in msgs), msgs


def test_protocol_mismatch_fails(tmp_path):
    _write(tmp_path, 'v.html', '<html><script>window.parent.postMessage({type:"WRONG",px:1},"*")</script></html>')
    p = _write(tmp_path, 'source.md', _concept_with_viz('./v.html'))
    ok, msgs = vg.run(p)
    assert not ok and any('protocol MISMATCH' in m for m in msgs), msgs


def test_degenerate_inline_svg_fails(tmp_path):
    src = CONCEPT_HDR + '@@@ concept id=c1 tag="A" title="A" gotit="g"\nintro\n%%% svg\n<svg viewBox="0 0 10 10"><path d="M0 0"/></svg>\n%%%\n'
    p = _write(tmp_path, 'source.md', src)
    ok, msgs = vg.run(p)
    assert not ok and any('degenerate' in m.lower() for m in msgs), msgs


def test_healthy_viz_passes(tmp_path):
    _write(tmp_path, 'ok.js', '// vendored lib')
    _write(tmp_path, 'v.html', '<html><script src="./ok.js"></script>'
           '<script>window.parent.postMessage({type:"viz-height",px:400},"*")</script></html>')
    p = _write(tmp_path, 'source.md', _concept_with_viz('./v.html'))
    ok, msgs = vg.run(p)
    assert ok, msgs


def test_non_concept_source_is_na(tmp_path):
    p = _write(tmp_path, 'source.md', '---\nmode: section\n---\n@@@ hero\n@lede x\n')
    ok, msgs = vg.run(p)
    assert ok and any('skip' in m for m in msgs)


# --- curriculum guard: every SHIPPED concept lesson must have sound visuals ---
def _shipped_concept_lessons():
    out = []
    for s in glob.glob(os.path.join(_REPO, 'sessions', 'm*', '*', 'source.md')):
        if 'mode: concept' in open(s, encoding='utf-8').read(400):
            out.append(s)
    return sorted(out)


@pytest.mark.parametrize('src', _shipped_concept_lessons())
def test_shipped_concept_lesson_visual_integrity(src):
    ok, msgs = vg.run(src)
    assert ok, os.path.relpath(src, _REPO) + '\n' + '\n'.join(msgs)
