#!/usr/bin/env python3
# =============================================================================
# Experiment Contract Gate (v9) — deterministic. Real-doing plan Task 1.
# =============================================================================
# A lesson's produce artifact (experiment.py) must be a REAL, self-checking,
# fill-the-TODO scaffold — never the 5-line "Placeholder. Fill this…" stub. This
# gate is a pure string/AST check (no execution): it verifies the artifact is a
# scaffold with a runnable self-check that prints a pass/fail signal.
#
# Reusable:  from experiment_contract import check_experiment
#            r = check_experiment(path); r.ok / r.reasons
# CLI:       python3 gates/experiment_contract.py <experiment.py>   (exit 0/3)
# =============================================================================
import sys, ast
from dataclasses import dataclass, field


@dataclass
class Result:
    ok: bool
    reasons: list = field(default_factory=list)


_PLACEHOLDER = 'placeholder. fill this'


def check_experiment(path):
    """Return Result(ok, reasons). ok iff the file is a real self-checking scaffold."""
    try:
        src = open(path, encoding='utf-8').read()
    except Exception as e:
        return Result(False, ['unreadable: %s' % e])

    reasons = []
    low = src.lower()

    # 1. not the placeholder stub
    if _PLACEHOLDER in low:
        reasons.append('is the placeholder stub (contains "Placeholder. Fill this")')
    # 2. parses as Python
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return Result(False, reasons + ['does not parse: %s' % e])
    # 3. >=1 import
    if not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
        reasons.append('no import (a real scaffold uses numpy/etc.)')
    # 4. a self-check block: `if __name__ == "__main__":`
    has_main = any(
        isinstance(n, ast.If) and _is_main_guard(n.test)
        for n in tree.body
    )
    if not has_main:
        reasons.append('no `if __name__ == "__main__":` self-check block')
    # 5. the self-check asserts something
    if not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
        reasons.append('no assert (the self-check must verify the expected value)')
    # 6. prints a pass/fail signal
    if '✅' not in src and '❌' not in src:
        reasons.append('no ✅/❌ pass-fail print (so running it does not tell the learner if they got it)')

    return Result(not reasons, reasons)


def _is_main_guard(test):
    # match: __name__ == "__main__"
    return (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == '__name__'
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == '__main__')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    a = ap.parse_args()
    r = check_experiment(a.path)
    if r.ok:
        print('PASS', a.path)
    else:
        print('FAIL', a.path)
        for why in r.reasons:
            print('  -', why)
    sys.exit(0 if r.ok else 3)


if __name__ == '__main__':
    main()
