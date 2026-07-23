import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib


def test_warmup_renders_recall_quiz_with_concept():
    html = v8lib.render_widget('warmup', {}, [
        'q: Yesterday, what did a neuron do? | a:1 | it forgot | weigh, add, decide | it slept'
        ' | concept: m02-neuron-basics | fb: weigh then add then decide.'])
    assert 'class="warmup"' in html          # not .quiz -> the SR-recording engine handles it
    assert 'data-concept="m02-neuron-basics"' in html
    assert 'data-correct="1"' in html
    assert 'weigh, add, decide' in html
    assert 'concept:' not in html            # the concept: field is consumed, not shown as an option


def test_warmup_multiple_questions():
    html = v8lib.render_widget('warmup', {}, [
        'q: Q1 | a:0 | right | wrong | x | y | concept: c1',
        'q: Q2 | a:2 | a | b | right | d | concept: c2 | fb: yep',
    ])
    assert html.count('class="q"') == 2
    assert 'data-concept="c1"' in html and 'data-concept="c2"' in html
