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
