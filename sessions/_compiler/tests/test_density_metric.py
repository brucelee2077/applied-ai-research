"""Tests for the build-up DENSITY metric (sessions/_density_scan.py).

This metric had no tests, and shipped three counting bugs of the same family:
it measured MARKUP as if it were prose. Twice caught (inline <svg>, then a
`~~~html` block), once not: an `!!! c-info` callout body.

Why the callout case matters. AUTHORING.md §"Concept-mode rules" rule 4 (Math
restraint) REQUIRES an author to demote heavy math into an `!!! c-info` box
marked "Optional (skippable)". The compiler treats `!!! ` as a block opener
(`v8lib.is_special`) and renders it as a set-off `<div class="callout">`. So a
correctly-demoted aside was being scored as a main-line wall of text — the
metric penalised authors for obeying the rule.

The body of such a box can still be one long unbroken run, which is a real
digestibility problem — just a *skippable* one with a different, cheaper fix
(break it internally). So it is measured as an ASIDE wall, not erased.
"""
import os, sys
import importlib.util

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
_spec = importlib.util.spec_from_file_location(
    '_density_scan', os.path.join(ROOT, 'sessions', '_density_scan.py'))
density = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(density)


LONG_A = 'The gradient reaches this weight only through the slope above it. ' * 12   # ~780 chars


def test_main_line_prose_run_is_a_wall():
    """Regression guard: an ordinary long paragraph still counts."""
    assert density.longest_wall(LONG_A) > 600


def test_widget_body_is_not_a_wall():
    """Regression guard (bug 1): an inline <svg> is a picture, not prose."""
    text = '%%% svg\n<svg viewBox="0 0 10 10">' + LONG_A + '</svg>\n%%%\n'
    assert density.longest_wall(text) == 0


def test_raw_html_block_is_not_a_wall():
    """Regression guard (bug 2): a ~~~html escape is markup, not prose."""
    text = '~~~html\n<p>%s</p>\n~~~\n' % LONG_A
    assert density.longest_wall(text) == 0


def test_callout_body_is_not_a_main_line_wall():
    """Bug 3: an `!!! c-info` box is a set-off, often skippable aside."""
    text = '!!! c-info 🔬\n<b>Optional (skippable) — the algebra.</b> %s\n!!!\n' % LONG_A
    assert density.longest_wall(text) == 0


def test_callout_body_is_reported_as_an_aside_wall():
    """It is not erased: a long aside is still measured, in its own class."""
    text = '!!! c-info 🔬\n<b>Optional (skippable) — the algebra.</b> %s\n!!!\n' % LONG_A
    assert density.aside_wall(text) > 600


def test_main_line_prose_is_not_counted_as_an_aside():
    assert density.aside_wall(LONG_A) == 0


def test_a_callout_does_not_merge_with_the_prose_around_it():
    """The box is a boundary: two short paragraphs either side stay short."""
    short = 'A short beat before the box. '
    text = '%s\n\n!!! c-info 🔬\n%s\n!!!\n\n%s' % (short, LONG_A, short)
    assert density.longest_wall(text) < 200


def test_scan_reports_main_and_aside_walls_separately():
    """scan_day reports the two wall classes as separate numbers.

    Built on a synthetic day, deliberately NOT on a real lesson: an earlier
    version of this test asserted that day-03-attention-scores carried three
    long asides, which pinned the very defect the next pass was about to fix.
    A test that fails when the content improves is testing the content, not the
    metric.
    """
    import tempfile
    day = ('---\nquest_id: t\n---\n\n'
           '@@@ concept id=c1 title="main-line wall"\n'
           'Opening intuition line.\n'
           '%%% svg\n<svg><text x="1" y="1">anchor</text></svg>\n%%%\n'
           + LONG_A + '\n\n'
           '@@@ concept id=c2 title="boxed aside"\n'
           'Opening intuition line.\n'
           '%%% svg\n<svg><text x="1" y="1">anchor</text></svg>\n%%%\n'
           'A short main-line beat.\n\n'
           '!!! c-info 🔬\n<b>Optional (skippable) — the algebra.</b> ' + LONG_A + '\n!!!\n')
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'source.md'), 'w', encoding='utf-8') as fh:
            fh.write(day)
        snap = density.scan_day(d)

    assert snap['concepts'] == 2
    # c1's long paragraph is main-line; c2's identical text is inside a box
    assert snap['walls_over_600'] == 1
    assert snap['asides_over_600'] == 1
    assert snap['max_wall'] > 600
    assert snap['max_aside_wall'] > 600
    assert snap['per_concept']['boxed aside']['wall'] < 200
