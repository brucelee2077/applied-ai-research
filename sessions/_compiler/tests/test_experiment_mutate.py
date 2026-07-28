"""Tests for the self-check strength meter (sessions/_experiment_mutate.py).

`_experiment_check.py` proves an artifact RUNS and prints ✅. It cannot tell
whether that ✅ means anything. The m01 pilot showed why that gap matters: two
days printed "✅ you got it" on provably broken code, because their asserts
re-derived the expected value from the same code path they were meant to test.

    kaplan_ok = abs(recovered_b - kaplan_b) < 1e-12   # recovered_b came FROM kaplan_b

That passes for every value of kaplan_b. A reviewer caught it by hand. With 95
more days to backfill, it has to be caught mechanically.

The meter perturbs one numeric literal at a time and re-runs. A self-check worth
having FAILS (non-zero exit, or a printed ❌) on a changed computation. A mutant
that survives is a claim nothing is actually checking.
"""
import os, sys, importlib.util

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
_spec = importlib.util.spec_from_file_location(
    '_experiment_mutate', os.path.join(ROOT, 'sessions', '_experiment_mutate.py'))
mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mu)


# A genuine self-check: the expected values are written down independently of
# the computation, so changing the computation breaks it.
STRONG = '''import numpy as np

def scale(x):
    return x * 3

if __name__ == "__main__":
    got = scale(np.array([1, 2, 4]))
    print("got:", got)
    ok = got.tolist() == [3, 6, 12]
    print("\\u2705 you got it" if ok else "\\u274c not yet")
    assert ok, "scale(x) should multiply by 3"
'''

# A circular self-check: the "expected" value is recomputed the same way, so it
# agrees with any multiplier. This is the m01 day-05 defect in miniature.
CIRCULAR = '''import numpy as np

FACTOR = 3

def scale(x):
    return x * FACTOR

if __name__ == "__main__":
    got = scale(np.array([1, 2, 4]))
    expected = np.array([1, 2, 4]) * FACTOR
    print("got:", got)
    ok = np.array_equal(got, expected)
    print("\\u2705 you got it" if ok else "\\u274c not yet")
    assert ok, "scale should multiply by FACTOR"
'''


def _w(tmp_path, text):
    p = tmp_path / 'experiment.py'
    p.write_text(text.replace('\\u2705', '✅').replace('\\u274c', '❌'),
                 encoding='utf-8')
    return str(p)


def test_numeric_literals_are_found_as_mutation_sites():
    sites = mu.mutation_sites(STRONG)
    assert sites, 'expected at least one numeric literal to mutate'
    assert any(s.value == 3 for s in sites), [s.value for s in sites]


def test_a_strong_self_check_kills_its_mutants(tmp_path):
    r = mu.score(_w(tmp_path, STRONG), max_mutants=6, timeout=30)
    assert r.killed > 0
    assert r.survivors == [], r.survivors
    assert r.kill_rate == 1.0


def test_a_circular_self_check_leaves_survivors(tmp_path):
    """The whole point: this file passes the acceptance gate but proves nothing."""
    r = mu.score(_w(tmp_path, CIRCULAR), max_mutants=6, timeout=30)
    assert r.survivors, 'a circular assert must leave at least one live mutant'
    assert r.kill_rate < 1.0


def test_a_survivor_reports_where_it_was(tmp_path):
    r = mu.score(_w(tmp_path, CIRCULAR), max_mutants=6, timeout=30)
    s = r.survivors[0]
    assert 'line' in s and 'was' in s and 'became' in s, s


def test_the_original_file_is_restored_afterwards(tmp_path):
    path = _w(tmp_path, CIRCULAR)
    before = open(path, encoding='utf-8').read()
    mu.score(path, max_mutants=4, timeout=30)
    assert open(path, encoding='utf-8').read() == before, 'mutation must not persist'


def test_a_file_that_does_not_run_is_reported_not_scored(tmp_path):
    r = mu.score(_w(tmp_path, 'import numpy\nthis is not python\n'), max_mutants=3, timeout=30)
    assert r.error is not None
    assert r.kill_rate == 0.0


# --- survivor classification ------------------------------------------------
# The raw meter flagged all six pilot days WEAK, because a literal that only sets
# a print width can never be "killed" — nothing asserts it. Those are noise. A
# survivor that sits in a computation is the real signal (m01 day-05's circular
# `fit_ok = abs(fitted_b - kaplan_b) < 0.01`). Classify, do not discard.

DISPLAY_ONLY = '''import numpy as np

if __name__ == "__main__":
    got = np.array([1.23456, 2.34567]) * 2
    print("got:", np.round(got, 4))
    print("width", 6)
    ok = got.tolist() == [2.46912, 4.69134]
    print("\\u2705 you got it" if ok else "\\u274c not yet")
    assert ok
'''


def test_display_only_literals_are_classified_as_cosmetic(tmp_path):
    r = mu.score(_w(tmp_path, DISPLAY_ONLY), max_mutants=8, timeout=30)
    # the rounding width and the print argument cannot be killed by any assert
    assert r.survivors, 'expected the display literals to survive'
    assert all(s.get('kind') == 'display' for s in r.survivors), r.survivors
    assert r.suspicious == [], r.suspicious


def test_a_circular_check_survivor_is_classified_suspicious(tmp_path):
    r = mu.score(_w(tmp_path, CIRCULAR), max_mutants=8, timeout=30)
    assert r.suspicious, 'a circular assert survivor must be flagged suspicious'
    assert all(s['kind'] == 'computation' for s in r.suspicious), r.suspicious


def test_survivor_records_line_and_values(tmp_path):
    r = mu.score(_w(tmp_path, CIRCULAR), max_mutants=8, timeout=30)
    s = r.suspicious[0]
    for key in ('line', 'was', 'became', 'snippet', 'kind'):
        assert key in s, (key, s)
