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
