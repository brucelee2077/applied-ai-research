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


def test_scan_reports_aside_walls_per_day():
    """A day carrying a long optional aside reports it separately."""
    snap = density.scan_day(os.path.join(
        ROOT, 'sessions', 'm03-attention', 'day-03-attention-scores'))
    assert snap is not None
    # this day's three long runs are all inside "Optional (skippable)" boxes
    assert snap['walls_over_600'] == 0
    assert snap['asides_over_600'] >= 3
    assert snap['max_aside_wall'] > 900
