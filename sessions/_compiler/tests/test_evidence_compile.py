import os, sys, json
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import evidence_compile as ec


# ---------------------------------------------------------------------------
# viz_refs — pull src paths out of `%%% viz ... src=...` lines
# ---------------------------------------------------------------------------
def test_viz_refs_quoted_and_bare():
    md = (
        'intro\n'
        '%%% viz src="../../viz/foo.html" title=t\n'
        'more\n'
        '%%% viz title=b src=../../viz/bar.html\n'
        'end\n'
    )
    refs = ec.viz_refs(md)
    assert '../../viz/foo.html' in refs
    assert '../../viz/bar.html' in refs


def test_viz_refs_none():
    assert ec.viz_refs('no viz blocks here\njust text') == []


# ---------------------------------------------------------------------------
# _md_to_html — tiny hand-rolled renderer
# ---------------------------------------------------------------------------
def test_md_to_html_basics():
    html = ec._md_to_html('# Title\n\nbody **x** and `code`.')
    assert '<h1>' in html and 'Title' in html
    assert '<strong>x</strong>' in html
    assert '<code>code</code>' in html
    assert '<p>' in html


def test_md_to_html_escapes_and_lists_and_fences():
    html = ec._md_to_html('- one\n- two\n\n```\na < b\n```')
    assert '<ul>' in html and '<li>one</li>' in html
    assert '<pre><code>' in html
    assert '&lt;' in html  # the < inside the fence is escaped


# ---------------------------------------------------------------------------
# assemble — the deterministic portfolio-day builder (fake root via tmp_path)
# ---------------------------------------------------------------------------
def _make_fixture(root):
    module, day = 'M', 'D'
    pdir = os.path.join(root, 'portfolio', module, day)
    os.makedirs(pdir)
    open(os.path.join(pdir, 'blog.md'), 'w').write('# Title\n\nbody **x** and `code`.')
    open(os.path.join(pdir, 'experiment.py'), 'w').write('print(1)')
    open(os.path.join(pdir, 'experiment_out.txt'), 'w').write('1')
    # lesson source referencing a viz two dirs up
    sdir = os.path.join(root, 'sessions', module, day)
    os.makedirs(sdir)
    open(os.path.join(sdir, 'source.md'), 'w').write('%%% viz src=../../viz/foo.html title=t\n')
    vizdir = os.path.join(root, 'sessions', 'viz')
    os.makedirs(vizdir)
    open(os.path.join(vizdir, 'foo.html'), 'w').write('<html>viz</html>')
    return module, day


def test_assemble_builds_self_contained_index(tmp_path):
    root = str(tmp_path)
    module, day = _make_fixture(root)
    idx = ec.assemble(module, day, root)
    assert os.path.exists(idx)
    html = open(idx, encoding='utf-8').read()
    # title, experiment code, and output are embedded
    assert 'Title' in html
    assert 'print(1)' in html
    assert '>1<' in html or '1' in html
    # the viz was copied and embedded as an iframe with a relative link
    assert '<iframe' in html
    assert 'assets/foo.html' in html
    # NEVER escapes the portfolio
    assert '../sessions' not in html
    assert 'src="/' not in html and "src='/" not in html
    # asset was actually copied
    assert os.path.exists(os.path.join(root, 'portfolio', module, day, 'assets', 'foo.html'))


def test_assemble_meta_json(tmp_path):
    root = str(tmp_path)
    module, day = _make_fixture(root)
    ec.assemble(module, day, root)
    meta = json.load(open(os.path.join(root, 'portfolio', module, day, 'meta.json')))
    assert meta['module'] == module and meta['day'] == day
    assert meta['title'] == 'Title'
    assert meta['has_experiment'] is True
    assert meta['has_plot'] is False
    assert meta['viz'] == ['foo.html']


def test_assemble_missing_files_ok(tmp_path):
    root = str(tmp_path)
    pdir = os.path.join(root, 'portfolio', 'M2', 'D2')
    os.makedirs(pdir)
    # no blog / experiment / source at all -> still builds, no crash
    idx = ec.assemble('M2', 'D2', root)
    assert os.path.exists(idx)
    meta = json.load(open(os.path.join(pdir, 'meta.json')))
    assert meta['has_experiment'] is False
    assert meta['viz'] == []
    # title falls back to the day slug
    assert meta['title'] == 'D2'
