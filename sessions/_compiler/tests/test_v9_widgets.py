import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import v8lib


def test_render_svg_wraps_in_build_viz():
    lines = ['<svg viewBox="0 0 10 10"><path d="M0 0"/></svg>']
    out = v8lib.render_widget('svg', {}, lines)
    assert 'class="build-viz"' in out
    assert '<svg viewBox="0 0 10 10">' in out
    assert out.count('<svg') == 1
