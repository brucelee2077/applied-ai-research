import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import beginner_language_gate as blg


# ---------------------------------------------------------------------------
# A lesson body that is CLEAN on all four beginner axes. Used as the base that
# each test below perturbs by exactly ONE thing, so a failure names one cause.
# ---------------------------------------------------------------------------
CLEAN = '''---
title: "T"
notebook_yardstick: null
---
@@@ hero
@lede
Picture a light switch on your bedroom wall. You flip it and the room changes.
@@@ concept id=c1 title="The switch"
A neuron makes one small choice. It says yes or no.
%%% viz src="sessions/viz/thing.html" cap="drag the slider"
%%%
%%% demo id=d1 label="predict the total, then reveal"
code: 2 + 2
out: 4
%%%
So the switch flips. That is the whole idea.
'''


def _msgs(txt, **kw):
    ok, msgs = blg.run(txt, **kw)
    return ok, '\n'.join(msgs)


def test_clean_body_passes():
    ok, out = _msgs(CLEAN)
    assert ok, out


# --- axis 1: simple language ----------------------------------------------
def test_banned_dismissive_phrase_fails():
    # CLAUDE.md section 5 bans these outright: they tell a beginner their
    # confusion is illegitimate.
    ok, out = _msgs(CLEAN.replace('A neuron makes one small choice.',
                                  'Obviously a neuron makes one small choice.'))
    assert not ok
    assert 'obviously' in out.lower()


def test_idiom_fails():
    ok, out = _msgs(CLEAN.replace('It says yes or no.',
                                  'Under the hood it says yes or no.'))
    assert not ok
    assert 'under the hood' in out.lower()


def test_long_sentence_is_flagged_but_advisory():
    # Run-ons WARN rather than block: a colon-introduced list reads as one long
    # "sentence" to a splitter while reading as chunks on the page. Banned
    # phrases (above) are unambiguous and DO block.
    run_on = ('A neuron takes every incoming number and multiplies each one by its '
              'own weight and then adds them all together and adds a bias on top '
              'and then pushes the total through a bend so that the answer can '
              'curve instead of staying straight forever and ever.')
    ok, out = _msgs(CLEAN.replace('A neuron makes one small choice.', run_on))
    assert ok, 'run-on must not block'
    assert 'warn plain language' in out
    assert 'sentence' in out.lower()


def test_plain_language_ignores_code_and_output_lines():
    # `code:` / `out:` payloads are program text, not prose the author is writing
    # for the reader, so a banned word appearing there must NOT trip the gate.
    # (SVG <text> LABELS are different — the reader reads those, and they ARE
    # checked; see test_dismissive_phrase_inside_an_svg_label_is_caught.)
    ok, out = _msgs(CLEAN.replace('code: 2 + 2',
                                  'code: x = 2 + 2  # obviously, under the hood'))
    assert ok, out


# --- axis 2/3: real play, honestly labelled --------------------------------
def test_zero_interactive_widgets_fails():
    ok, out = _msgs(CLEAN.replace('%%% viz src="sessions/viz/thing.html" cap="drag the slider"\n%%%\n', ''))
    assert not ok
    assert 'interactive' in out.lower()


def test_inline_range_slider_counts_as_play():
    # An in-body slider is real interactivity even with no %%% viz iframe —
    # m02 ships play this way, and the gate must not demand one specific form.
    swapped = CLEAN.replace('%%% viz src="sessions/viz/thing.html" cap="drag the slider"',
                            '%%% svg\n<svg><foreignObject><input type="range" min="0" max="2"></foreignObject></svg>')
    ok, out = _msgs(swapped)
    assert ok, out


def test_demo_label_promising_a_run_fails():
    # A %%% demo only un-hides a pre-baked answer. Saying "run" oversells it.
    ok, out = _msgs(CLEAN.replace('label="predict the total, then reveal"',
                                  'label="predict the total, then run it"'))
    assert not ok
    assert 'demo' in out.lower()


def test_demo_label_without_label_arg_is_fine():
    # No explicit label => v8lib's default ("reveal") is used, which is honest.
    ok, out = _msgs(CLEAN.replace(' label="predict the total, then reveal"', ''))
    assert ok, out


# --- axis 4: digestibility (the density measure that nothing ran) ----------
def test_main_line_wall_fails():
    wall = 'This is a long unbroken run of main-line prose. ' * 16   # >600 chars
    ok, out = _msgs(CLEAN.replace('So the switch flips. That is the whole idea.', wall))
    assert not ok
    assert 'wall' in out.lower()


def test_wall_inside_optional_box_is_not_a_main_line_wall():
    # An "Optional (skippable)" aside is ALREADY correctly placed by the math-
    # restraint rule; only its inside needs breaking up. Promoting it to main
    # prose would be the wrong fix, so it must not be reported as a main wall.
    aside = '!!! c-info 🧮\nOptional (skippable) — ' + ('detail on the algebra. ' * 30) + '\n!!!'
    ok, out = _msgs(CLEAN.replace('So the switch flips. That is the whole idea.', aside))
    assert ok, out


# --- the yardstick escape hatch -------------------------------------------
def test_nulling_yardstick_when_notebook_exists_fails():
    # Writing `notebook_yardstick: null` silently disables the beginner-
    # friendliness + interest-ceiling judges. Doing that when a real notebook
    # exists is how m04 day-06 shipped with tone ungraded.
    ok, out = _msgs(CLEAN, notebook_exists=True)
    assert not ok
    assert 'yardstick' in out.lower()


def test_declared_yardstick_passes():
    body = CLEAN.replace('notebook_yardstick: null',
                         'notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb')
    ok, out = _msgs(body, notebook_exists=True)
    assert ok, out


# --- vocabulary is checked on EVERYTHING the reader sees --------------------
# Regression: the first version checked only _prose_only, which masks step:/why:
# rungs, SVG <text> labels and gloss bodies. On a real authored unit those masked
# regions were 398 of 833 reader-visible words (48%) and held three idioms, so the
# check inspected barely half the page while reporting a clean pass.
def test_idiom_inside_a_steps_rung_is_caught():
    body = CLEAN.replace('So the switch flips. That is the whole idea.',
                         '%%% steps\nstep: add them up\nwhy: under the hood this is a sum\n%%%')
    ok, out = _msgs(body)
    assert not ok
    assert 'under the hood' in out.lower()


def test_dismissive_phrase_inside_an_svg_label_is_caught():
    body = CLEAN.replace('So the switch flips. That is the whole idea.',
                         '%%% svg\n<svg><text>obviously the same</text></svg>\n%%%')
    ok, out = _msgs(body)
    assert not ok
    assert 'obviously' in out.lower()


def test_idiom_inside_a_gloss_body_is_caught():
    body = CLEAN.replace('A neuron makes one small choice.',
                         'A [[neuron||a tiny decider; in a nutshell it just votes]] decides.')
    ok, out = _msgs(body)
    assert not ok
    assert 'in a nutshell' in out.lower()


def test_svg_shape_markup_still_does_not_trip_it():
    # Attribute values and shape markup are NOT reader-visible words.
    body = CLEAN.replace('So the switch flips.',
                         '%%% svg\n<svg viewBox="0 0 10 10"><rect x="1" class="of course, bad"/></svg>\n%%%\nSo the switch flips.')
    ok, out = _msgs(body)
    assert ok, out
