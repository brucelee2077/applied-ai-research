import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib

# The EXACT output render_demo produces today for this input, with no `predict:` field.
# Pinning it guarantees the predict-then-reveal change is byte-identical when predict is absent
# (protects recompile-idempotence + any shipped-output snapshot).
_NO_PREDICT_INPUT = ['code: relu([-3,2])', 'out: [0,2]', 'take: zeros negatives']
_NO_PREDICT_EXPECTED = (
    '<div class="demo" data-demo="relu">'
    '<div class="demo-code"><code>relu([-3,2])</code>'
    '<button class="demo-run" type="button">run it ▶</button></div>'
    '<pre class="demo-out" hidden>[0,2]</pre>'
    '<div class="demo-take" hidden>zeros negatives</div></div>')


def test_demo_without_predict_is_byte_identical():
    html = v8lib.render_demo({'id': 'relu', 'label': 'run it'}, _NO_PREDICT_INPUT)
    assert html == _NO_PREDICT_EXPECTED
    assert 'demo-predict' not in html


def test_demo_with_predict_renders_prompt():
    html = v8lib.render_demo({'id': 'relu'}, [
        'predict: what does relu([-3,2]) **give**?',
        'code: relu([-3,2])', 'out: [0,2]', 'take: zeros negatives'])
    assert 'demo-predict' in html                 # the predict prompt element
    assert '🤔' in html                            # predict glyph
    assert 'what does relu' in html
    assert '<strong>give</strong>' in html         # inline() ran on the predict text
    # predict prompt appears BEFORE the code row (predict FIRST, then reveal)
    assert html.index('demo-predict') < html.index('demo-code')
    # the rest of the demo is intact
    assert 'class="demo-out" hidden' in html and 'class="demo-take" hidden' in html
