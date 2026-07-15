import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import publish_portfolio as pp


def _write_index(root, module, day, body):
    pdir = os.path.join(root, 'portfolio', module, day)
    os.makedirs(pdir, exist_ok=True)
    open(os.path.join(pdir, 'index.html'), 'w').write(body)


def test_validate_clean(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<html><img src="assets/plot.png">'
                 '<iframe src="assets/foo.html"></iframe></html>')
    assert pp.validate_self_contained(root) == []


def test_validate_flags_sessions_escape(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<html><iframe src="../sessions/x.html"></iframe></html>')
    viol = pp.validate_self_contained(root)
    assert viol  # non-empty
    # the offending file and the offending link are both reported
    joined = ' '.join('%s|%s' % (f, l) for f, l in viol)
    assert 'index.html' in joined
    assert '../sessions/x.html' in joined


def test_validate_flags_absolute_link(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<html><img src="/etc/passwd"></html>')
    viol = pp.validate_self_contained(root)
    assert viol
    assert any('/etc/passwd' in l for _, l in viol)


def test_validate_flags_absolute_href(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<a href="/somewhere/else.html">x</a>')
    viol = pp.validate_self_contained(root)
    assert any('/somewhere/else.html' in l for _, l in viol)


def test_validate_empty_when_no_portfolio(tmp_path):
    # no portfolio dir at all -> clean (nothing to violate)
    assert pp.validate_self_contained(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Copied viz (assets/*.html) must ALSO be scanned, not just index.html
# ---------------------------------------------------------------------------
def _write_asset(root, module, day, name, body):
    adir = os.path.join(root, 'portfolio', module, day, 'assets')
    os.makedirs(adir, exist_ok=True)
    open(os.path.join(adir, name), 'w').write(body)


def test_validate_scans_assets_html_and_flags_parent_escape(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<html></html>')  # clean index
    _write_asset(root, 'M', 'D', 'viz.html', '<a href="../index.html">back</a>')
    viol = pp.validate_self_contained(root)
    assert viol  # the copied viz IS scanned now
    joined = ' '.join('%s|%s' % (f, l) for f, l in viol)
    assert 'viz.html' in joined
    assert '../index.html' in joined


def test_validate_assets_html_external_and_hash_pass(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<html></html>')
    _write_asset(root, 'M', 'D', 'viz.html',
                 '<link href="https://fonts.example/x.css">'
                 '<a href="#">jump</a>'
                 '<img src="assets/x.png">'
                 '<script src="./d3.js"></script>')
    assert pp.validate_self_contained(root) == []


def test_validate_flags_parent_escape_anywhere(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D', '<img src="../../sessions/viz/x.png">')
    viol = pp.validate_self_contained(root)
    assert any('../../sessions/viz/x.png' in l for _, l in viol)


def test_validate_allows_protocol_relative_mailto_data(tmp_path):
    root = str(tmp_path)
    _write_index(root, 'M', 'D',
                 '<link href="//cdn.example.com/a.css">'
                 '<a href="mailto:x@y.com">mail</a>'
                 '<img src="data:image/png;base64,AAAA">'
                 '<a href="https://x.com">ext</a>')
    assert pp.validate_self_contained(root) == []


def test_validate_flags_unquoted_parent(tmp_path):
    root = str(tmp_path)
    # unquoted href value must also be caught
    _write_index(root, 'M', 'D', '<a href=../secrets.html>x</a>')
    viol = pp.validate_self_contained(root)
    assert any('../secrets.html' in l for _, l in viol)
