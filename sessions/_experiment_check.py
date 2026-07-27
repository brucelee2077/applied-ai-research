#!/usr/bin/env python3
"""Acceptance check for a lesson's experiment.py: the contract AND a real run.

`gates/experiment_contract.py` is a pure string/AST check. That is deliberate —
it is cheap and runs everywhere — but it never executes the file, so an artifact
can satisfy every structural rule and still crash, hang, or quietly print ❌ the
first time a learner runs it. Structure is not evidence that the thing works.

This adds the missing half. It runs the contract first (fast reject), then
actually executes the script in a subprocess and requires:
  - exit code 0
  - a ✅ in stdout, and no ❌
  - completion inside a timeout (a learner will not wait, and neither will CI)
  - no network access (these run in a sandbox; an artifact that downloads a
    dataset is not reproducible for the reader)

Network is blocked by running with a sitecustomize shim that makes socket
connections raise, rather than by trusting the author not to reach out.

Usage:
  python3 sessions/_experiment_check.py <experiment.py> [...]     exit 0 = all pass
  python3 sessions/_experiment_check.py --all                     every stub-free day
  python3 sessions/_experiment_check.py --module m01-shape-of-data
"""
import os, sys, glob, time, json, tempfile, subprocess
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler', 'gates'))
import experiment_contract as contract  # noqa: E402

DEFAULT_TIMEOUT = 180

# Made importable ahead of the artifact so any socket use raises instead of
# silently depending on a network the reader may not have.
_NO_NET = '''
import socket as _s
class _Blocked(OSError):
    pass
def _deny(*a, **k):
    raise _Blocked("network access is not allowed in a lesson artifact")
_s.socket.connect = _deny
_s.create_connection = _deny
_s.socket.connect_ex = _deny
'''


@dataclass
class Result:
    path: str
    ok: bool
    reasons: list = field(default_factory=list)
    stdout: str = ''
    stderr: str = ''
    seconds: float = None


def check(path, timeout=DEFAULT_TIMEOUT, run=True):
    """Contract, then execution. Returns Result."""
    rel = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path
    c = contract.check_experiment(path)
    if not c.ok:
        return Result(rel, False, list(c.reasons), seconds=0.0)
    if not run:
        return Result(rel, True, [], seconds=0.0)

    with tempfile.TemporaryDirectory() as shim:
        with open(os.path.join(shim, 'sitecustomize.py'), 'w', encoding='utf-8') as fh:
            fh.write(_NO_NET)
        env = dict(os.environ)
        env['PYTHONPATH'] = shim + os.pathsep + env.get('PYTHONPATH', '')
        env['MPLBACKEND'] = 'Agg'          # never try to open a window
        env['PYTHONWARNINGS'] = 'ignore'
        started = time.time()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(path)],
                               cwd=os.path.dirname(os.path.abspath(path)),
                               capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return Result(rel, False, ['timed out after %ss' % timeout],
                          seconds=time.time() - started)
        elapsed = time.time() - started

    reasons = []
    out, err = p.stdout or '', p.stderr or ''
    if p.returncode != 0:
        tail = (err.strip() or out.strip())[-400:]
        reasons.append('exit %d: %s' % (p.returncode, tail))
    if '❌' in out:
        reasons.append('printed ❌ — its own self-check reports failure')
    elif '✅' not in out:
        reasons.append('no ✅ pass-fail marker in stdout — running it tells the learner nothing')
    return Result(rel, not reasons, reasons, out, err, elapsed)


def _targets(argv):
    if '--all' in argv:
        return sorted(glob.glob(os.path.join(ROOT, 'sessions', '**', 'experiment.py'),
                                recursive=True))
    if '--module' in argv:
        mod = argv[argv.index('--module') + 1]
        return sorted(glob.glob(os.path.join(ROOT, 'sessions', mod, '**', 'experiment.py'),
                                recursive=True))
    return [a for a in argv if a.endswith('.py')]


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    timeout = int(argv[argv.index('--timeout') + 1]) if '--timeout' in argv else DEFAULT_TIMEOUT
    out_json = argv[argv.index('--json') + 1] if '--json' in argv else None

    targets = _targets(argv)
    if not targets:
        print('no experiment.py matched')
        return 2

    results, failed = [], 0
    for t in targets:
        r = check(t, timeout=timeout)
        results.append(r)
        if r.ok:
            print('  ok   %-72s %5.1fs' % (r.path, r.seconds or 0))
        else:
            failed += 1
            print('  FAIL %-72s %5.1fs' % (r.path, r.seconds or 0))
            for why in r.reasons:
                print('         %s' % why.replace('\n', ' ')[:300])
    print('\n%d checked, %d passed, %d failed' % (len(results), len(results) - failed, failed))
    if out_json:
        json.dump([{'path': r.path, 'ok': r.ok, 'reasons': r.reasons,
                    'seconds': r.seconds} for r in results],
                  open(out_json, 'w', encoding='utf-8'), indent=1)
        print('-> %s' % out_json)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
