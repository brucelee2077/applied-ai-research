import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib


def test_steps_renders_numbered_build_steps():
    html = v8lib.render_widget('steps', {}, [
        'step: (x·W1)·W2',
        'why: two matmuls **back to back**',
        'step: = x·(W1·W2)',
        'why: they collapse into ONE — no [[bend||non-linearity]] between them'])
    assert 'class="build"' in html                 # the scroll-reveal wrapper
    assert html.count('class="build-step"') == 2    # one per step/why pair
    assert html.count('class="build-num"') == 2      # numbered badges
    assert '>1<' in html and '>2<' in html           # the numbers
    assert 'class="build-note"' in html
    assert '<strong>back to back</strong>' in html   # inline() on the why gloss
    assert 'class="term"' in html                    # [[bend||...]] glossary in a step
    assert '(x·W1)·W2' in html                        # the work of step 1


def test_steps_tolerates_step_without_why():
    html = v8lib.render_widget('steps', {}, ['step: start here'])
    assert html.count('class="build-step"') == 1
    assert 'start here' in html
