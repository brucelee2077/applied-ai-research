"""Tests for the experiment.py acceptance check (sessions/_experiment_check.py).

`gates/experiment_contract.py` is a pure string/AST check — it never executes the
file. That is deliberate and cheap, but it means an artifact can satisfy every
structural rule and still crash, hang, hit the network, or print ❌ when the
learner runs it. This check closes that gap: contract THEN execution.
"""
import os, sys, importlib.util

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
_spec = importlib.util.spec_from_file_location(
    '_experiment_check', os.path.join(ROOT, 'sessions', '_experiment_check.py'))
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)

GOOD = '''import numpy as np

def build():
    return np.array([1, 2, 3])

if __name__ == "__main__":
    x = build()
    assert x.shape == (3,), "expected (3,), got %s" % (x.shape,)
    print("shape:", x.shape)
    print("✅ all checks passed")
'''

CRASHES = '''import numpy as np

if __name__ == "__main__":
    assert np.array([1, 2]).shape == (3,), "boom"
    print("✅ all checks passed")
'''

HANGS = '''import time

if __name__ == "__main__":
    assert True
    time.sleep(30)
    print("✅ done")
'''

NETWORK = '''import urllib.request

if __name__ == "__main__":
    urllib.request.urlopen("http://example.com")
    assert True
    print("✅ done")
'''

SILENT = '''import numpy as np

if __name__ == "__main__":
    assert np.array([1]).shape == (1,)
'''


def _w(tmp_path, text, name='experiment.py'):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def test_a_real_self_checking_artifact_passes(tmp_path):
    r = ec.check(_w(tmp_path, GOOD))
    assert r.ok, r.reasons
    assert '✅' in r.stdout


def test_a_failing_assert_is_caught(tmp_path):
    r = ec.check(_w(tmp_path, CRASHES))
    assert not r.ok
    assert any('exit' in x or 'AssertionError' in x for x in r.reasons), r.reasons


def test_a_hanging_script_is_caught_not_waited_on(tmp_path):
    r = ec.check(_w(tmp_path, HANGS), timeout=3)
    assert not r.ok
    assert any('timed out' in x for x in r.reasons), r.reasons


def test_a_script_that_reaches_the_network_is_caught(tmp_path):
    """These run in a sandbox and must not depend on downloads."""
    r = ec.check(_w(tmp_path, NETWORK), timeout=20)
    assert not r.ok, 'network access must not be allowed to pass'


def test_a_script_that_prints_no_success_marker_fails(tmp_path):
    r = ec.check(_w(tmp_path, SILENT))
    assert not r.ok
    assert any('pass-fail' in x or '✅' in x for x in r.reasons), r.reasons


def test_the_placeholder_stub_fails_the_contract(tmp_path):
    stub = "# Placeholder. Fill this from the lesson's PRODUCE step\n"
    r = ec.check(_w(tmp_path, stub))
    assert not r.ok
    assert any('placeholder' in x.lower() for x in r.reasons), r.reasons


def test_result_records_what_it_ran(tmp_path):
    r = ec.check(_w(tmp_path, GOOD))
    assert r.path.endswith('experiment.py')
    assert r.seconds is not None and r.seconds >= 0
