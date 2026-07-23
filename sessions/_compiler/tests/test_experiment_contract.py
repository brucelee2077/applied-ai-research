import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
from experiment_contract import check_experiment

STUB = ("# day-01-single-neuron - experiment\n#\n"
        "# Placeholder. Fill this from the lesson's PRODUCE step (open lesson.html):\n"
        "#   Option A: write it yourself, or  Option B: paste the frontier-experiment-lab prompt.\n")

GOOD = ('import numpy as np\n'
        'def forward(x, w, b):\n'
        '    # TODO: return the weighted sum + bias\n'
        '    return float(x @ w + b)\n'
        'if __name__ == "__main__":\n'
        '    got = forward(np.array([2.0, 3.0]), np.array([0.5, -1.0]), 1.0)\n'
        '    expected = -1.0\n'
        '    ok = abs(got - expected) < 1e-9\n'
        '    print("✅ you got it" if ok else "❌ not yet -- expected %s" % expected)\n'
        '    assert ok, got\n')


def test_stub_fails():
    r = check_experiment_text(STUB)
    assert r.ok is False
    assert any('placeholder' in x.lower() for x in r.reasons)


def test_good_passes():
    r = check_experiment_text(GOOD)
    assert r.ok is True, r.reasons


def test_missing_selfcheck_fails():
    no_main = 'import numpy as np\ndef f(x):\n    # TODO\n    return x\n'
    r = check_experiment_text(no_main)
    assert r.ok is False


# check_experiment reads a path; give tests a text shim via a tmp file helper.
def check_experiment_text(text):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(text); p = fh.name
    try:
        return check_experiment(p)
    finally:
        os.unlink(p)
