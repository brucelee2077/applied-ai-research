import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib


# A %%% demo only un-hides a pre-baked answer (it does not compute), so the DEFAULT
# button label must not imply execution. Author-supplied labels are preserved verbatim
# (so existing lessons recompile byte-identical).
def test_demo_default_label_is_reveal_not_run():
    html = v8lib.render_demo({'id': 'd'}, ['code: 2+2', 'out: 4', 'take: the sum'])
    assert 'reveal' in html.lower()
    assert 'run it ▶' not in html      # the misleading default is gone


def test_demo_author_label_preserved():
    html = v8lib.render_demo(
        {'id': 'd', 'label': 'run it — one hop with lr = 0.1'},
        ['code: 2.0 - 0.1*4.0', 'out: 1.6', 'take: only lr scales the hop'])
    assert 'run it — one hop with lr = 0.1 ▶' in html
