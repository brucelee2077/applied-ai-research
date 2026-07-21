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


# --- "visualize the build-up" ADVISORY warn (never flips exit code) ----------
_OPT = ('!!! c-info \U0001F50E\n<b>Optional (skippable) — the formula.</b> '
        'Some heavy math a beginner can skip, several narrated symbols.\n!!!')
_DEMO = '%%% demo id=d label="run it"\ncode: f(2)\nout: 4\ntake: watch it land.\n%%%'
_MLAD = '%%% mathladder\nwords | formula | numbers | sanity check\n%%%'
_FORMULA = '%%% formula\ndef forward(x): return x @ W + b\n%%%'


def _heavy(cid, extra):
    """anchor svg + build prose + `extra` (an Optional box / mathladder / demo / formula)."""
    return f'@@@ concept id={cid} tag="t" title="T" gotit="ok"\n{_INTRO}\n{_SVG}\n{_BUILD}\n{extra}\n'


def test_heavy_optional_box_without_buildup_visual_warns():
    # Optional-box (heavy) + only the opening anchor svg -> WARN, but ok stays True (advisory).
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), _heavy('c3', _OPT)))
    assert ok, msgs  # advisory: exit code / ok are driven ONLY by the triad checks
    assert any(m.startswith('warn') and 'c3' in m and 'has no build-up' in m for m in msgs), msgs


def test_heavy_optional_box_with_demo_in_buildup_no_warn():
    # Same Optional box but a %%% demo now sits in the build-up region -> satisfied, NO warn.
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), _heavy('c3', _DEMO + '\n' + _OPT)))
    assert ok, msgs
    assert not any(m.startswith('warn') and 'c3' in m for m in msgs), msgs


def test_mathladder_in_buildup_satisfies_no_warn():
    # A Math Ladder IS the build-up depiction (clause i) -> NO warn even with no 2nd svg.
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), _heavy('c3', _MLAD)))
    assert ok, msgs
    assert not any(m.startswith('warn') and 'c3' in m for m in msgs), msgs


def test_formula_widget_alone_does_not_trigger():
    # %%% formula (m04 uses it for code/one-line rules) is NOT a heavy-build-up trigger -> NO warn.
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), _heavy('c3', _FORMULA)))
    assert ok, msgs
    assert not any(m.startswith('warn') and 'c3' in m for m in msgs), msgs


def test_light_concept_never_warns():
    # No Optional box, no mathladder — a plain intro/svg/build concept never warns.
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), _concept('c3')))
    assert ok, msgs
    assert not any(m.startswith('warn') for m in msgs), msgs
