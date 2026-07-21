import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import v8lib


def test_render_svg_wraps_in_build_viz():
    lines = ['<svg viewBox="0 0 10 10"><path d="M0 0"/></svg>']
    out = v8lib.render_widget('svg', {}, lines)
    assert 'class="build-viz"' in out
    assert '<svg viewBox="0 0 10 10">' in out
    assert out.count('<svg') == 1


def test_render_demo_emits_runnable_console():
    lines = ['code: relu(np.array([-3,-1,0,2,5]))', 'out: array([0, 0, 0, 2, 5])', 'take: ReLU zeros negatives, passes positives.']
    out = v8lib.render_widget('demo', {'id': 'relu', 'label': 'run it'}, lines)
    assert 'class="demo"' in out
    assert 'data-demo="relu"' in out
    assert 'class="demo-run"' in out
    assert 'relu(np.array([-3,-1,0,2,5]))' in out
    assert 'array([0, 0, 0, 2, 5])' in out
    assert 'ReLU zeros negatives' in out


def test_render_demo_keeps_every_line_of_a_multiline_out():
    # Regression: a multi-line `out:` used to be truncated to only its first
    # line (continuation lines were silently dropped by _kv), which made the
    # rendered demo contradict its take-away. Every out line must survive.
    lines = [
        'code: head_1 must answer using BOTH describe->tired AND place->mat',
        'out: describe-only would give:  [tired 1.00, mat 0.00]   (crisp)',
        '     place-only would give:     [tired 0.00, mat 1.00]   (crisp)',
        '     ONE head must do both  ->  [tired 0.50, mat 0.50]   (muddy)',
        'take: <b>One head splits the difference.</b>',
    ]
    out = v8lib.render_widget('demo', {'id': 'average', 'label': 'run'}, lines)
    import re
    pre = re.search(r'<pre class="demo-out" hidden>(.*?)</pre>', out, re.S).group(1)
    assert 'describe-only would give' in pre
    assert 'place-only would give' in pre
    assert 'ONE head must do both' in pre
    assert '[tired 0.50, mat 0.50]' in pre   # the punchline line the take-away cites
    assert pre.count('\n') == 2               # three lines total
    # continuation lines are dedented but keep internal column alignment
    assert '\nplace-only would give' in pre


def test_kv_single_line_values_are_unchanged():
    # The multi-line fix must not alter simple single-line key: value parsing.
    d = v8lib._kv(['code: relu(x)', 'out: array([0, 0, 2])', 'take: keeps positives.'])
    assert d == {'code': 'relu(x)', 'out': 'array([0, 0, 2])', 'take': 'keeps positives.'}


def test_render_demo_keeps_aligned_continuation_with_word_colon():
    # Regression: a continuation line whose first token is a single word followed
    # by an ALIGNED colon (e.g. "join  :  …") must stay part of `out`, not be
    # misread as a new field. A field only opens when the word directly touches
    # the colon ("out:"), never when spaces separate them ("join  :").
    lines = [
        'code: d_model = 6, h = 3; split -> join -> W_O',
        'out: split :  d_k = 6/3 = 2      -> 3 answers of width 2',
        '     join  :  stitch side by side -> one vector of width 6',
        '     W_O   :  mix the width-6 vector -> final width 6',
        'take: <b>Width 6 in, width 6 out.</b>',
    ]
    d = v8lib._kv(lines)
    assert set(d) == {'code', 'out', 'take'}   # no stray 'join' / 'W_O' keys
    assert 'split :' in d['out']
    assert 'join  :' in d['out']
    assert 'W_O   :' in d['out']
    assert 'final width 6' in d['out']
    assert d['out'].count('\n') == 2           # all three out lines present


def test_kv_value_keeps_an_internal_colon():
    # A value may itself contain a colon after the first one; only the FIRST
    # colon separates key from value.
    d = v8lib._kv(['code: answer using BOTH: a AND b'])
    assert d['code'] == 'answer using BOTH: a AND b'


def test_render_quiz_emits_four_q_blocks_with_answer_marker():
    lines = [
        'q: What does an activation add? | a:1 | More params | Non-linearity | Faster matmul | A bias | fb: The bend.',
        'q: No activation, ten layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.',
        'q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.',
        'q: Loss plateaus, ReLUs all 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.',
    ]
    out = v8lib.render_widget('quiz', {}, lines)
    assert out.count('class="q"') == 4
    assert out.count('class="q-opt"') == 16
    assert out.count('data-correct="1"') == 4
    assert 'The bend.' in out


def test_render_quiz_does_not_crash_on_option_starting_with_a_colon():
    # An option like "a:b mapping" must NOT be misread as an answer marker and crash.
    lines = ['q: Which is a mapping? | a:1 | a:b mapping | plain option | fb: careful.']
    out = v8lib.render_widget('quiz', {}, lines)
    assert out.count('class="q"') == 1
    # the "a:b mapping" part degrades into a normal option, not an answer marker
    assert 'a:b mapping' in out
    assert 'data-correct="1"' in out


def test_render_quiz_missing_answer_marker_defaults_safely():
    # No 'a:N' marker present — must not crash; data-correct defaults to 0.
    lines = ['q: No marker here | first | second | fb: no answer given.']
    out = v8lib.render_widget('quiz', {}, lines)
    assert out.count('class="q"') == 1
    assert 'data-correct="0"' in out


def test_render_concept_is_tracked_section_with_one_gotit():
    block = {'type': 'concept',
             'args': {'id': 'c3', 'num': '3', 'tag': 'Meet ReLU', 'title': 'ReLU — a one-way valve', 'gotit': 'Met ReLU'},
             'lines': ['Intro prose about ReLU.', '%%% svg', '<svg viewBox="0 0 10 10"></svg>', '%%%', 'Build-up prose.']}
    out = v8lib.render_concept(block)
    assert 'class="module-section"' in out
    assert 'id="c3"' in out and 'data-sec="c3"' in out
    assert out.count('class="gotit"') == 1
    assert 'Met ReLU' in out
    assert 'ReLU — a one-way valve' in out
    assert '<svg viewBox="0 0 10 10">' in out
    assert out.count('<svg') == 1


def test_concept_nav_items_number_in_order():
    blocks = [
        {'type': 'hero', 'args': {}, 'lines': []},
        {'type': 'concept', 'args': {'id': 'c1', 'title': 'The collapse'}, 'lines': []},
        {'type': 'concept', 'args': {'id': 'c2', 'title': 'The bend'}, 'lines': []},
        {'type': 'quiz', 'args': {'id': 'quiz', 'title': 'Check'}, 'lines': []},
        {'type': 'produce', 'args': {'id': 'produce', 'title': 'Produce'}, 'lines': []},
    ]
    items = v8lib.concept_nav_items(blocks)
    targets = [it['target'] for it in items]
    assert targets == ['home', 'c1', 'c2', 'quiz', 'produce']
    labels = {it['target']: it['label'] for it in items}
    assert labels['c1'].startswith('1') and 'collapse' in labels['c1'].lower()
    assert labels['c2'].startswith('2')
