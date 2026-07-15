import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import concept_structure_gate as g

_INTRO = "This is a plain-words intro that explains the idea before any picture appears, in enough words to be real."
_BUILD = "Now the build-up: a worked example that walks step by step through the mechanism in concrete detail here."
_SVG = "%%% svg\n<svg viewBox=\"0 0 10 10\"><path d=\"M0 0 L10 10\"/></svg>\n%%%"

def _concept(cid, intro=_INTRO, svg=_SVG, build=_BUILD):
    return f'@@@ concept id={cid} tag="t" title="T" gotit="ok"\n{intro}\n{svg}\n{build}\n'

def _doc(*concepts):
    return "---\nmode: concept\n---\n@@@ hero\n@lede hook\n@goal g\n" + "".join(concepts) + "@@@ fin\n"

def test_three_well_formed_concepts_pass():
    src = _doc(_concept('c1'), _concept('c2'), _concept('c3'))
    ok, msgs = g.run(src)
    assert ok, msgs

def test_fewer_than_three_concepts_fail():
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2')))
    assert not ok
    assert any('>=3 concept units' in m and m.startswith('FAIL') for m in msgs)

def test_concept_without_visual_fails():
    bad = '@@@ concept id=c3 tag="t" title="T" gotit="ok"\n' + _INTRO + '\n' + _BUILD + '\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), bad))
    assert not ok
    assert any('c3 has a visual' in m and m.startswith('FAIL') for m in msgs)

def test_concept_with_visual_but_no_buildup_fails():
    nobuild = f'@@@ concept id=c3 tag="t" title="T" gotit="ok"\n{_INTRO}\n{_SVG}\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), nobuild))
    assert not ok
    assert any('c3 has build-up' in m and m.startswith('FAIL') for m in msgs)

def test_concept_with_visual_but_no_intro_fails():
    nointro = f'@@@ concept id=c3 tag="t" title="T" gotit="ok"\n{_SVG}\n{_BUILD}\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), nointro))
    assert not ok
    assert any('c3 has intro prose' in m and m.startswith('FAIL') for m in msgs)
