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
