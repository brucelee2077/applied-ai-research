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

GOLDEN OUTPUT (expected_output.txt)
-----------------------------------
Everything above is still inside the artifact's own view of itself, and an
assert can only see the value it is handed. That leaves a whole class open:

    print("x:", shown_x)   ->   print("x:", shown_x * 2)

`shown_x` is still correctly bound and still correctly asserted — the
corruption happens at the print CALL, after the assertion's view of the world
ends. Measured on a real plant engine: 498 plants of this shape, 4 caught
(0.8%), and not one catch was an assertion. Pinning rendered text inside the
file defends line N only, and these are teaching artifacts meant to be read and
edited, so hundreds of in-file pins are actively harmful (a field-width or seed
change breaks dozens at once).

So: if a day directory contains `expected_output.txt`, the artifact's full
stdout must match it exactly. One artifact per day, zero in-file pins.

  - OPT-IN BY FILE EXISTENCE. No reference -> behaviour is exactly as before.
    (69 of 115 days are still placeholder stubs; none of them may start
    failing because this exists.)
  - SAME RUN. The comparison reuses the stdout this check already captured;
    the artifact is never executed twice.
  - FAIL, NEVER HEAL. A mismatch is a failure. Nothing here ever rewrites a
    reference — a gate that repairs its own oracle is worthless. Re-pinning is
    a separate, explicit act: `--write-expected` (plus `--force` to replace an
    existing reference).

Deliberate edge-case choices:
  - trailing newline at EOF: ignored (trailing blank lines are stripped from
    both sides before comparing), so an editor adding or removing one is not a
    failure;
  - CRLF: normalised to LF on both sides — a checkout on Windows must not
    fail every day;
  - a reference that exists but is empty/whitespace-only: HARD FAIL. It is
    never a valid pin, and treating it as "matches nothing" would silently
    disarm the day;
  - empty artifact stdout: already impossible to pass — the run check requires
    a ✅ — so it is reported as the missing-marker failure, not as a diff;
  - a run-level failure (crash, timeout, ❌, no ✅): the crash is the story, so
    the diff is suppressed rather than dumped on top of it.

Usage:
  python3 sessions/_experiment_check.py <experiment.py> [...]     exit 0 = all pass
  python3 sessions/_experiment_check.py --all                     every stub-free day
  python3 sessions/_experiment_check.py --module m01-shape-of-data
  python3 sessions/_experiment_check.py <target...> --write-expected [--force]
"""
import os, sys, glob, time, json, difflib, tempfile, subprocess
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler', 'gates'))
import experiment_contract as contract  # noqa: E402

DEFAULT_TIMEOUT = 180
EXPECTED_NAME = 'expected_output.txt'
MAX_DIFF_LINES = 40

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
    # 'absent'  no expected_output.txt (behaves exactly as before)
    # 'match' | 'mismatch' | 'empty' | 'skipped' (run already failed / not asked)
    golden: str = 'absent'


def expected_path(experiment_path):
    """The day's golden reference sits next to its artifact."""
    return os.path.join(os.path.dirname(os.path.abspath(experiment_path)),
                        EXPECTED_NAME)


def _norm_lines(text):
    """Compare on content, not on how an editor happened to end the file.

    CRLF -> LF, and trailing blank lines dropped from both sides so a missing
    or extra newline at EOF is not a failure.
    """
    text = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _canonical(text):
    """What we write to disk: LF endings, exactly one newline at EOF."""
    body = '\n'.join(_norm_lines(text))
    return body + '\n' if body else ''


def _differing_line_count(expected, actual):
    """Lines that changed — a one-line edit counts 1, not 2 (- plus +)."""
    n = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, expected, actual, autojunk=False).get_opcodes():
        if tag == 'replace':
            n += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            n += i2 - i1
        elif tag == 'insert':
            n += j2 - j1
    return n


def _regen_command(rel):
    return 'python3 sessions/_experiment_check.py %s --write-expected --force' % rel


def golden_diff(rel, expected_raw, actual_raw, max_lines=MAX_DIFF_LINES):
    """None if the run matches the reference, else a readable failure message."""
    exp, act = _norm_lines(expected_raw), _norm_lines(actual_raw)
    if exp == act:
        return None
    diff = list(difflib.unified_diff(exp, act, fromfile=EXPECTED_NAME,
                                     tofile='actual stdout', lineterm='', n=2))
    shown, hidden = diff[:max_lines], max(0, len(diff) - max_lines)
    msg = ['stdout does not match %s — %d differing line(s) '
           '(%d expected / %d actual)'
           % (EXPECTED_NAME, _differing_line_count(exp, act), len(exp), len(act))]
    msg += ['  ' + ln for ln in shown]
    if hidden:
        msg.append('  ... %d more diff line(s) suppressed' % hidden)
    msg.append('  If the output changed on purpose, re-pin it deliberately:')
    msg.append('    %s' % _regen_command(rel))
    return '\n'.join(msg)


def check(path, timeout=DEFAULT_TIMEOUT, run=True, golden=True):
    """Contract, then execution, then the golden comparison. Returns Result."""
    rel = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path
    c = contract.check_experiment(path)
    if not c.ok:
        return Result(rel, False, list(c.reasons), seconds=0.0, golden='skipped')
    if not run:
        return Result(rel, True, [], seconds=0.0, golden='skipped')

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
                          seconds=time.time() - started, golden='skipped')
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

    # Golden comparison last, on the run we already did. Opt-in by file
    # existence, and suppressed when the run itself failed (the crash is the
    # story — do not bury it under a whole-output diff).
    ref = expected_path(path)
    state = 'absent'
    if golden and os.path.exists(ref):
        if reasons:
            state = 'skipped'
        else:
            try:
                raw = open(ref, encoding='utf-8', errors='replace').read()
            except Exception as e:                       # unreadable pin
                state = 'mismatch'
                reasons.append('%s is unreadable: %s' % (EXPECTED_NAME, e))
                raw = None
            if raw is not None:
                if not raw.strip():
                    state = 'empty'
                    reasons.append(
                        '%s exists but is empty — an empty reference is never a valid '
                        'pin, so it cannot be compared against. Write a real one:\n'
                        '    %s' % (EXPECTED_NAME, _regen_command(rel)))
                else:
                    why = golden_diff(rel, raw, out)
                    state = 'match' if why is None else 'mismatch'
                    if why:
                        reasons.append(why)
    return Result(rel, not reasons, reasons, out, err, elapsed, golden=state)


def _targets(argv):
    if '--all' in argv:
        return sorted(glob.glob(os.path.join(ROOT, 'sessions', '**', 'experiment.py'),
                                recursive=True))
    if '--module' in argv:
        mod = argv[argv.index('--module') + 1]
        return sorted(glob.glob(os.path.join(ROOT, 'sessions', mod, '**', 'experiment.py'),
                                recursive=True))
    return [a for a in argv if a.endswith('.py')]


def _print_reasons(reasons):
    """Single-line reasons print exactly as they always have; a multi-line
    reason (the golden diff) keeps its lines instead of being flattened."""
    for why in reasons:
        lines = why.split('\n')
        if len(lines) == 1:
            print('         %s' % why[:300])
        else:
            for ln in lines:
                print('         %s' % ln)


def write_expected(targets, force=False, timeout=DEFAULT_TIMEOUT):
    """Explicitly (re)generate each target's expected_output.txt.

    Never called by a check. Refuses to overwrite an existing reference without
    --force, and refuses to pin an artifact that does not pass its own check.
    """
    wrote = unchanged = refused = broken = 0
    for t in targets:
        r = check(t, timeout=timeout, golden=False)
        ref = expected_path(t)
        if not r.ok:
            broken += 1
            print('  SKIP  %s — does not pass its own check; not pinning a broken artifact'
                  % r.path)
            _print_reasons(r.reasons)
            continue
        body = _canonical(r.stdout)
        if os.path.exists(ref):
            old = open(ref, encoding='utf-8', errors='replace').read()
            if _norm_lines(old) == _norm_lines(body):
                unchanged += 1
                print('  same  %s (%s already up to date)' % (r.path, EXPECTED_NAME))
                continue
            if not force:
                refused += 1
                print('  REFUSED %s — %s already exists and the output has CHANGED.'
                      % (r.path, EXPECTED_NAME))
                print('         Not overwriting a reference implicitly. Here is what would change:')
                for ln in list(difflib.unified_diff(
                        _norm_lines(old), _norm_lines(body),
                        fromfile='%s (on disk)' % EXPECTED_NAME,
                        tofile='new output', lineterm='', n=2))[:MAX_DIFF_LINES]:
                    print('         %s' % ln)
                print('         If the new output is correct, re-run with --force:')
                print('           %s' % _regen_command(r.path))
                continue
        with open(ref, 'w', encoding='utf-8') as fh:
            fh.write(body)
        wrote += 1
        shown = os.path.relpath(ref, ROOT) if ref.startswith(ROOT) else ref
        print('  wrote %s' % shown)
    print('\n%d written, %d unchanged, %d refused, %d skipped (artifact failing)'
          % (wrote, unchanged, refused, broken))
    return 1 if (refused or broken) else 0


def main_argv(argv):
    if not argv:
        print(__doc__)
        return 2
    timeout = int(argv[argv.index('--timeout') + 1]) if '--timeout' in argv else DEFAULT_TIMEOUT
    out_json = argv[argv.index('--json') + 1] if '--json' in argv else None

    targets = _targets(argv)
    if not targets:
        print('no experiment.py matched')
        return 2

    if '--write-expected' in argv:
        return write_expected(targets, force='--force' in argv, timeout=timeout)

    results, failed = [], 0
    for t in targets:
        r = check(t, timeout=timeout)
        results.append(r)
        if r.ok:
            print('  ok   %-72s %5.1fs' % (r.path, r.seconds or 0))
        else:
            failed += 1
            print('  FAIL %-72s %5.1fs' % (r.path, r.seconds or 0))
            _print_reasons(r.reasons)
    print('\n%d checked, %d passed, %d failed' % (len(results), len(results) - failed, failed))
    if out_json:
        json.dump([{'path': r.path, 'ok': r.ok, 'reasons': r.reasons,
                    'seconds': r.seconds, 'golden': r.golden} for r in results],
                  open(out_json, 'w', encoding='utf-8'), indent=1)
        print('-> %s' % out_json)
    return 1 if failed else 0


def main():
    return main_argv(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
