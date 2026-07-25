"""Repeated `out:` lines in a %%% demo must all reach the rendered <pre>.

Regression test for a shipped defect the correctness judge caught during the
m02 day-07 rebuild (2026-07-24): `_kv()` treats every `word:` line as a NEW
field, so a demo written as

    out: step 1: velocity = 1.000
    out: step 2: velocity = 1.900
    out: step 3: velocity = 2.710

kept only `step 3` — the reader saw the punchline of a build-up with the build
removed. 43 demo blocks across m02/m04/m05a/m07 lose content this way.

`_kv` already folds INDENTED continuation lines. This adds the repeated-key
form: a key that repeats ACCUMULATES instead of overwriting. Fields that
legitimately appear once (code:, take:, predict:) are unaffected because they
are not repeated.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v8lib


def test_repeated_out_lines_accumulate():
    d = v8lib._kv(['code: run()', 'out: line1', 'out: line2', 'out: line3'])
    assert d['out'] == 'line1\nline2\nline3', d
    assert d['code'] == 'run()'


def test_repeated_out_survives_into_rendered_demo():
    html = v8lib.render_demo({'id': 'm'}, [
        'code: momentum()',
        'out: step 1: velocity = 1.000',
        'out: step 2: velocity = 1.900',
        'out: step 3: velocity = 2.710',
        'take: speed builds up',
    ])
    for line in ('step 1: velocity = 1.000',
                 'step 2: velocity = 1.900',
                 'step 3: velocity = 2.710'):
        assert line in html, 'lost %r from the rendered demo' % line


def test_indented_continuation_still_works():
    """The existing continuation form must keep behaving exactly as before."""
    d = v8lib._kv(['code: a', 'out: line1', '  line2', '  line3'])
    assert d['out'] == 'line1\nline2\nline3', d


def test_mixed_repeated_key_and_continuation():
    d = v8lib._kv(['out: a1', '  a2', 'out: b1', '  b2'])
    assert d['out'] == 'a1\na2\nb1\nb2', d


def test_single_valued_fields_unaffected():
    d = v8lib._kv(['predict: guess?', 'code: f()', 'out: 42', 'take: done'])
    assert d == {'predict': 'guess?', 'code': 'f()', 'out': '42', 'take': 'done'}, d


def test_repeated_key_in_mathladder_accumulates():
    """_kv is shared — the same rule applies to every widget that uses it."""
    d = v8lib._kv(['words: first', 'words: second'])
    assert d['words'] == 'first\nsecond', d
