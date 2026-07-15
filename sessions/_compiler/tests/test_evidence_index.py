import os, sys, json
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import evidence_index as ei


def _write_meta(root, module, day, **over):
    pdir = os.path.join(root, 'portfolio', module, day)
    os.makedirs(pdir, exist_ok=True)
    meta = {'module': module, 'day': day, 'title': '%s %s title' % (module, day),
            'has_experiment': True, 'has_plot': False, 'viz': []}
    meta.update(over)
    json.dump(meta, open(os.path.join(pdir, 'meta.json'), 'w'))
    # a stub index.html so the link target exists
    open(os.path.join(pdir, 'index.html'), 'w').write('<html></html>')


def test_build_index_two_days(tmp_path):
    root = str(tmp_path)
    _write_meta(root, 'M', 'D1', title='First Day', has_plot=True, viz=['a.html'])
    _write_meta(root, 'M', 'D2', title='Second Day')
    n = ei.build_index(root)
    assert n == 2
    idx = os.path.join(root, 'portfolio', 'index.html')
    assert os.path.exists(idx)
    html = open(idx, encoding='utf-8').read()
    # both titles rendered
    assert 'First Day' in html and 'Second Day' in html
    # correct RELATIVE links to each day's page
    assert 'M/D1/index.html' in html
    assert 'M/D2/index.html' in html
    # never escapes portfolio
    assert '../sessions' not in html
    assert 'href="/' not in html


def test_build_index_badges(tmp_path):
    root = str(tmp_path)
    _write_meta(root, 'M', 'D1', has_experiment=True, has_plot=True, viz=['a.html', 'b.html'])
    ei.build_index(root)
    html = open(os.path.join(root, 'portfolio', 'index.html'), encoding='utf-8').read()
    # a "2 viz" badge appears (len(viz))
    assert '2' in html


def test_build_index_idempotent(tmp_path):
    root = str(tmp_path)
    _write_meta(root, 'M', 'D1')
    _write_meta(root, 'M', 'D2')
    ei.build_index(root)
    first = open(os.path.join(root, 'portfolio', 'index.html'), encoding='utf-8').read()
    ei.build_index(root)
    second = open(os.path.join(root, 'portfolio', 'index.html'), encoding='utf-8').read()
    assert first == second


def test_build_index_empty(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, 'portfolio'))
    n = ei.build_index(root)
    assert n == 0
    assert os.path.exists(os.path.join(root, 'portfolio', 'index.html'))
