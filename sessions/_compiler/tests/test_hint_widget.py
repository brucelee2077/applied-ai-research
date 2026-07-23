import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib
import pytest


# %%% hint: a tiered, offline, progressive-disclosure hint ladder for stuck learners.
def test_hint_renders_progressive_tiers():
    html = v8lib.render_widget('hint', {}, [
        't1: a gentle nudge',
        't2: a worked micro-step',
        't3: the idea in one sentence',
    ])
    assert 'data-hint-tier="1"' in html
    assert 'data-hint-tier="3"' in html
    assert 'hint-reveal' in html            # the "Stuck? reveal a hint" control
    # tiers ship hidden (revealed one-by-one by donor JS)
    assert 'hidden' in html


def test_hint_no_network():
    html = v8lib.render_widget('hint', {}, ['t1: nudge', 't2: step'])
    assert 'fetch(' not in html and 'http' not in html


def test_unknown_widget_still_raises():
    with pytest.raises(ValueError):
        v8lib.render_widget('nonsense', {}, ['x'])
