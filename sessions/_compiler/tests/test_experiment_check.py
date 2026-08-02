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


# =============================================================================
# Golden output (expected_output.txt)
# =============================================================================
# The checks above all live INSIDE the artifact's own view of itself: an assert
# can only see the value it is handed. That leaves a whole class open —
#
#     print("x:", shown_x)   ->   print("x:", shown_x * 2)
#
# `shown_x` is still correctly bound and still correctly asserted; the
# corruption happens at the print CALL, after the assertion's view of the world
# ends. Measured on a real plant engine: 498 plants of this shape, 4 caught
# (0.8%), and not one catch was an assertion. Pinning rendered text in-file
# defends line N only and is brittle across a field-width or seed change.
# One reference file per day closes the class with zero in-file pins.
#
# Opt-in BY FILE EXISTENCE: 69 of 115 days are still placeholder stubs and many
# real days have no reference yet — none of them may change behaviour.

# Binds a value, asserts it, prints it. This is the honest version.
SHOWS_A_VALUE = '''import numpy as np

def scaled(v):
    return v * np.array([1.0, 2.0, 3.0])

if __name__ == "__main__":
    out = scaled(np.array([2.0, 2.0, 2.0]))
    assert out.shape == (3,), "expected (3,), got %s" % (out.shape,)
    assert abs(float(out[1]) - 4.0) < 1e-9, "expected 4.0, got %s" % out[1]
    print("middle value:", float(out[1]))
    print("✅ all checks passed")
'''

# Byte-identical EXCEPT the print site. Every assert above still holds, the
# script still exits 0, and it still prints ✅. No self-check can see this.
SHOWS_A_VALUE_CORRUPTED = SHOWS_A_VALUE.replace(
    'print("middle value:", float(out[1]))',
    'print("middle value:", float(out[1]) * 2)')


def _run_clean(tmp_path, src=SHOWS_A_VALUE):
    """Run an artifact with no reference and return its stdout."""
    r = ec.check(_w(tmp_path, src))
    assert r.ok, r.reasons
    return r.stdout


def _pin(tmp_path, text):
    p = tmp_path / 'expected_output.txt'
    p.write_text(text, encoding='utf-8')
    return p


# --- backward compatibility: the one that must never break ------------------

def test_a_day_with_no_expected_output_behaves_exactly_as_before(tmp_path):
    """69 stubs and every unpinned real day must be untouched by this feature."""
    r = ec.check(_w(tmp_path, GOOD))
    assert r.ok, r.reasons
    assert r.reasons == []
    assert r.golden == 'absent'
    assert not os.path.exists(str(tmp_path / 'expected_output.txt')), \
        'checking must never create a reference as a side effect'


def test_no_reference_still_fails_the_same_way_for_a_broken_artifact(tmp_path):
    r = ec.check(_w(tmp_path, CRASHES))
    assert not r.ok
    assert any('exit' in x for x in r.reasons), r.reasons


# --- the comparison ---------------------------------------------------------

def test_a_matching_reference_passes(tmp_path):
    out = _run_clean(tmp_path)
    _pin(tmp_path, out)
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert r.ok, r.reasons
    assert r.golden == 'match'


def test_a_one_character_difference_fails_and_says_where(tmp_path):
    out = _run_clean(tmp_path)
    _pin(tmp_path, out.replace('middle value: 4.0', 'middle value: 4.1'))
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert not r.ok, 'a one-character drift must not pass'
    assert r.golden == 'mismatch'
    blob = '\n'.join(r.reasons)
    assert 'expected_output.txt' in blob
    assert '-middle value: 4.1' in blob and '+middle value: 4.0' in blob, blob
    assert '1 line' in blob or '1 differing' in blob, blob
    assert '--write-expected' in blob, 'must say how to re-pin deliberately'


def test_the_print_site_corruption_no_assertion_can_see_is_caught(tmp_path):
    """THE motivating case. Assertions are blind here; the reference is not."""
    # 1. the corrupted artifact passes every existing check on its own
    bad_dir = tmp_path / 'unpinned'
    bad_dir.mkdir()
    r_unpinned = ec.check(_w(bad_dir, SHOWS_A_VALUE_CORRUPTED))
    assert r_unpinned.ok, (
        'fixture is wrong: the corruption must survive contract + run + ✅ '
        'so the test proves the golden compare adds the catch')
    assert 'middle value: 8.0' in r_unpinned.stdout

    # 2. pin the reference from the HONEST version, then plant the corruption
    good_dir = tmp_path / 'pinned'
    good_dir.mkdir()
    honest = _run_clean(good_dir)
    assert 'middle value: 4.0' in honest
    _w(good_dir, SHOWS_A_VALUE_CORRUPTED)
    (good_dir / 'expected_output.txt').write_text(honest, encoding='utf-8')

    r = ec.check(str(good_dir / 'experiment.py'))
    assert not r.ok, 'print-site corruption must be caught by the reference'
    assert r.golden == 'mismatch'
    blob = '\n'.join(r.reasons)
    assert 'middle value: 8.0' in blob and 'middle value: 4.0' in blob, blob


def test_a_long_mismatch_diff_is_truncated_not_dumped(tmp_path):
    src = '''import numpy as np

if __name__ == "__main__":
    xs = np.arange(400)
    assert xs.shape == (400,)
    for i in xs:
        print("row", int(i))
    print("✅ all checks passed")
'''
    r0 = ec.check(_w(tmp_path, src))
    assert r0.ok, r0.reasons
    _pin(tmp_path, r0.stdout.replace('row ', 'ROW '))
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert not r.ok
    blob = '\n'.join(r.reasons)
    assert len(blob.splitlines()) < 80, 'a 400-line diff must not be dumped whole'
    assert 'more diff line' in blob, blob
    assert '400 ' in blob or '400 differing' in blob or '800' in blob, blob


# --- edge cases -------------------------------------------------------------

def test_a_missing_trailing_newline_is_not_a_mismatch(tmp_path):
    out = _run_clean(tmp_path)
    _pin(tmp_path, out.rstrip('\n'))
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert r.ok, r.reasons


def test_extra_trailing_blank_lines_are_not_a_mismatch(tmp_path):
    out = _run_clean(tmp_path)
    _pin(tmp_path, out + '\n\n\n')
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert r.ok, r.reasons


def test_a_crlf_reference_matches_lf_stdout(tmp_path):
    out = _run_clean(tmp_path)
    _pin(tmp_path, out.replace('\n', '\r\n'))
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert r.ok, r.reasons


def test_an_empty_reference_is_a_failure_not_a_free_pass(tmp_path):
    _run_clean(tmp_path)
    _pin(tmp_path, '')
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert not r.ok, 'an empty reference is never a valid pin'
    assert r.golden == 'empty'
    assert any('empty' in x for x in r.reasons), r.reasons


def test_a_whitespace_only_reference_is_also_a_failure(tmp_path):
    _run_clean(tmp_path)
    _pin(tmp_path, '\n\n  \n')
    r = ec.check(str(tmp_path / 'experiment.py'))
    assert not r.ok
    assert r.golden == 'empty'


def test_a_broken_artifact_reports_the_crash_not_a_diff(tmp_path):
    """A crash is the story; do not bury it under a whole-output diff."""
    _pin(tmp_path, 'middle value: 4.0\n✅ all checks passed\n')
    r = ec.check(_w(tmp_path, CRASHES))
    assert not r.ok
    assert any('exit' in x for x in r.reasons), r.reasons
    assert not any('@@' in x for x in r.reasons), r.reasons
    assert r.golden == 'skipped'


# --- --write-expected -------------------------------------------------------

def test_write_expected_creates_the_reference(tmp_path):
    path = _w(tmp_path, SHOWS_A_VALUE)
    rc = ec.main_argv([path, '--write-expected'])
    assert rc == 0
    ref = tmp_path / 'expected_output.txt'
    assert ref.exists()
    body = ref.read_text(encoding='utf-8')
    assert 'middle value: 4.0' in body and body.endswith('\n')
    # and the freshly written reference makes the day pass
    assert ec.check(path).golden == 'match'


def test_write_expected_refuses_to_overwrite_and_shows_what_changed(tmp_path, capsys):
    path = _w(tmp_path, SHOWS_A_VALUE)
    assert ec.main_argv([path, '--write-expected']) == 0
    before = (tmp_path / 'expected_output.txt').read_text(encoding='utf-8')

    _w(tmp_path, SHOWS_A_VALUE_CORRUPTED)          # output legitimately changed
    rc = ec.main_argv([path, '--write-expected'])
    out = capsys.readouterr().out
    assert rc != 0, 'refusing must be visible to a script, not a silent no-op'
    assert (tmp_path / 'expected_output.txt').read_text(encoding='utf-8') == before, \
        'a gate that repairs its own oracle is worthless'
    assert '--force' in out
    assert 'middle value: 8.0' in out and 'middle value: 4.0' in out, out


def test_write_expected_with_force_rewrites(tmp_path):
    path = _w(tmp_path, SHOWS_A_VALUE)
    assert ec.main_argv([path, '--write-expected']) == 0
    _w(tmp_path, SHOWS_A_VALUE_CORRUPTED)
    assert ec.main_argv([path, '--write-expected', '--force']) == 0
    body = (tmp_path / 'expected_output.txt').read_text(encoding='utf-8')
    assert 'middle value: 8.0' in body
    assert ec.check(path).golden == 'match'


def test_write_expected_is_a_no_op_when_the_output_is_unchanged(tmp_path):
    path = _w(tmp_path, SHOWS_A_VALUE)
    assert ec.main_argv([path, '--write-expected']) == 0
    assert ec.main_argv([path, '--write-expected']) == 0, \
        're-pinning identical output is not a conflict'


def test_write_expected_refuses_to_pin_a_broken_artifact(tmp_path):
    path = _w(tmp_path, CRASHES)
    rc = ec.main_argv([path, '--write-expected'])
    assert rc != 0
    assert not (tmp_path / 'expected_output.txt').exists(), \
        'never freeze the output of something that does not work'


def test_writing_never_happens_as_a_side_effect_of_a_plain_check(tmp_path):
    path = _w(tmp_path, SHOWS_A_VALUE)
    assert ec.main_argv([path]) == 0
    assert not (tmp_path / 'expected_output.txt').exists()


# --- aggregation ------------------------------------------------------------

def _fake_module(tmp_path, monkeypatch, days):
    """Build sessions/<mod>/day-XX/experiment.py under a fake ROOT."""
    mod = tmp_path / 'sessions' / 'm99-fake'
    for i, (src, expected) in enumerate(days, start=1):
        d = mod / ('day-%02d' % i)
        d.mkdir(parents=True)
        (d / 'experiment.py').write_text(src, encoding='utf-8')
        if expected is not None:
            (d / 'expected_output.txt').write_text(expected, encoding='utf-8')
    monkeypatch.setattr(ec, 'ROOT', str(tmp_path))
    return mod


def test_module_aggregates_a_mix_of_pinned_and_unpinned_days(tmp_path, monkeypatch, capsys):
    good_out = 'middle value: 4.0\n✅ all checks passed\n'
    _fake_module(tmp_path, monkeypatch, [
        (GOOD, None),                              # unpinned, passes
        (SHOWS_A_VALUE, good_out),                 # pinned, matches
        (SHOWS_A_VALUE_CORRUPTED, good_out),       # pinned, print-site drift
    ])
    rc = ec.main_argv(['--module', 'm99-fake'])
    out = capsys.readouterr().out
    assert rc == 1
    assert '3 checked, 2 passed, 1 failed' in out, out
    assert 'day-03' in out and 'FAIL' in out


def test_all_aggregates_across_modules(tmp_path, monkeypatch, capsys):
    good_out = 'middle value: 4.0\n✅ all checks passed\n'
    _fake_module(tmp_path, monkeypatch, [
        (SHOWS_A_VALUE, good_out),
        (SHOWS_A_VALUE_CORRUPTED, good_out),
    ])
    rc = ec.main_argv(['--all'])
    out = capsys.readouterr().out
    assert rc == 1
    assert '2 checked, 1 passed, 1 failed' in out, out


def test_write_expected_works_over_a_whole_module(tmp_path, monkeypatch):
    mod = _fake_module(tmp_path, monkeypatch,
                       [(SHOWS_A_VALUE, None), (GOOD, None)])
    assert ec.main_argv(['--module', 'm99-fake', '--write-expected']) == 0
    for i in (1, 2):
        assert (mod / ('day-%02d' % i) / 'expected_output.txt').exists()
    assert ec.main_argv(['--module', 'm99-fake']) == 0


def test_json_report_records_the_golden_state(tmp_path, monkeypatch):
    good_out = 'middle value: 4.0\n✅ all checks passed\n'
    _fake_module(tmp_path, monkeypatch, [
        (SHOWS_A_VALUE, good_out),
        (SHOWS_A_VALUE_CORRUPTED, good_out),
        (GOOD, None),
    ])
    report = str(tmp_path / 'r.json')
    ec.main_argv(['--module', 'm99-fake', '--json', report])
    import json as _json
    rows = _json.load(open(report, encoding='utf-8'))
    assert [x['golden'] for x in rows] == ['match', 'mismatch', 'absent']
