#!/usr/bin/env python3
"""Measure whether an experiment.py's self-check is worth anything.

`_experiment_check.py` proves an artifact RUNS and prints ✅. It cannot tell
whether that ✅ means anything, and the m01 pilot showed the gap is not
theoretical: two of six days printed "✅ you got it" on provably broken code.
Both had the same shape of defect — the "expected" value was re-derived from the
same code path it was meant to test:

    loss = power_law(compute, 2.0, kaplan_b)      # built FROM kaplan_b
    ...
    kaplan_ok = abs(recovered_b - kaplan_b) < 1e-12

`log(a·x^b) = log a + b·log x` makes that exact for every `b`. A human reviewer
caught it by planting values. With 95 more days to backfill, it has to be
mechanical.

So: perturb ONE numeric literal at a time, re-run, and see whether the file's own
self-check notices. A self-check worth having FAILS on a changed computation
(non-zero exit, or a printed ❌). A mutant that survives is a claim nothing
checks.

This is a REVIEW aid, not a gate. A surviving mutant is not automatically a bug —
a literal can be cosmetic (a print width, a plot limit) or a day can legitimately
be about randomness. It tells a reviewer where to look.

Usage:
  python3 sessions/_experiment_mutate.py <experiment.py> [--max N] [--timeout S]
  python3 sessions/_experiment_mutate.py --module m01-shape-of-data
"""
import os, re, sys, ast, glob, json, subprocess, tempfile
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TIMEOUT = 90
MAX_MUTANTS = 12

# Literals that are structure rather than claims: perturbing them changes what
# the script IS, not whether its answer is right, so they only add noise.
_SKIP_VALUES = {0, 1, -1}


@dataclass
class Site:
    lineno: int
    col: int
    value: object
    kind: str = 'computation'      # 'display' when the literal only formats output


@dataclass
class Score:
    path: str
    killed: int = 0
    survivors: list = field(default_factory=list)
    error: str = None

    @property
    def total(self):
        return self.killed + len(self.survivors)

    @property
    def kill_rate(self):
        return (self.killed / self.total) if self.total else 0.0

    @property
    def suspicious(self):
        """Survivors that sit in a COMPUTATION — the ones worth a human's time.

        A display-only survivor (a `round(x, 4)` width, a literal inside a
        print) can never be killed, because nothing asserts it. Reporting those
        as weakness buried the real signal: on the m01 pilot every one of the six
        days looked WEAK, while only two had a genuinely circular self-check.
        """
        return [s for s in self.survivors if s.get('kind') == 'computation']


def _display_literals(tree):
    """Positions of literals that only shape OUTPUT, never a checked value.

    Two cases cover almost all of it: anything lexically inside a `print(...)`
    call, and the digits argument of `round`/`np.round`.
    """
    marked = set()

    def mark(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                marked.add((sub.lineno, sub.col_offset))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else '')
        if name == 'print':
            mark(node)
        elif name == 'round' and len(node.args) > 1:
            mark(node.args[1])
    return marked


def mutation_sites(src):
    """Numeric literals worth perturbing, in source order, tagged by role."""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    display = _display_literals(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
           and not isinstance(node.value, bool):
            if node.value in _SKIP_VALUES:
                continue
            kind = 'display' if (node.lineno, node.col_offset) in display else 'computation'
            out.append(Site(node.lineno, node.col_offset, node.value, kind))
    return out


def _perturb(value):
    """A change big enough to matter, small enough to stay plausible."""
    if isinstance(value, int):
        return value + 1 if abs(value) < 1000 else value * 2
    return round(value * 1.7 + 0.37, 9) if value else 0.5


def _mutate_source(src, site):
    """Replace exactly the literal at (lineno, col) — never a lookalike elsewhere."""
    lines = src.splitlines(keepends=True)
    if site.lineno > len(lines):
        return None
    line = lines[site.lineno - 1]
    m = re.compile(r'\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?').match(line, site.col)
    if not m:
        return None
    new = repr(_perturb(site.value))
    lines[site.lineno - 1] = line[:m.start()] + new + line[m.end():]
    return ''.join(lines), line.strip()[:90], new


def _run(path, timeout):
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONWARNINGS'] = 'ignore'
    try:
        p = subprocess.run([sys.executable, os.path.abspath(path)],
                           cwd=os.path.dirname(os.path.abspath(path)),
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, '', 'timeout'
    return p.returncode, p.stdout or '', p.stderr or ''


def score(path, max_mutants=MAX_MUTANTS, timeout=DEFAULT_TIMEOUT):
    """Perturb literals one at a time; report which mutants the self-check missed."""
    path = os.path.abspath(path)
    rel = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path
    src = open(path, encoding='utf-8').read()

    code, out, err = _run(path, timeout)
    if code != 0 or '✅' not in out:
        return Score(rel, error='baseline does not pass (exit %s): %s'
                     % (code, (err or out).strip()[-200:]))

    sites = mutation_sites(src)
    if not sites:
        return Score(rel, error='no numeric literals to mutate')
    # Spend the budget on computation literals first — those are where a circular
    # self-check hides. Display literals are sampled only with what is left over.
    compute = [s for s in sites if s.kind == 'computation']
    display = [s for s in sites if s.kind == 'display']

    def spread(items, n):
        if not items or n <= 0:
            return []
        step = max(1, len(items) // n)
        return items[::step][:n]

    chosen = spread(compute, max_mutants)
    chosen += spread(display, max_mutants - len(chosen))
    chosen.sort(key=lambda s: (s.lineno, s.col))

    result = Score(rel)
    backup = src
    try:
        for site in chosen:
            made = _mutate_source(src, site)
            if not made:
                continue
            mutated, snippet, became = made
            if mutated == src:
                continue
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(mutated)
            code, out, _err = _run(path, timeout)
            # killed = the file noticed: crashed, asserted, or printed ❌
            if code != 0 or '❌' in out or '✅' not in out:
                result.killed += 1
            else:
                result.survivors.append({
                    'line': site.lineno, 'was': site.value, 'became': became,
                    'snippet': snippet, 'kind': site.kind})
    finally:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(backup)
    return result


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    timeout = int(argv[argv.index('--timeout') + 1]) if '--timeout' in argv else DEFAULT_TIMEOUT
    max_m = int(argv[argv.index('--max') + 1]) if '--max' in argv else MAX_MUTANTS
    if '--module' in argv:
        targets = sorted(glob.glob(os.path.join(
            ROOT, 'sessions', argv[argv.index('--module') + 1], '**', 'experiment.py'),
            recursive=True))
    else:
        targets = [a for a in argv if a.endswith('.py')]

    weak = 0
    for t in targets:
        r = score(t, max_mutants=max_m, timeout=timeout)
        if r.error:
            print('  --   %-64s %s' % (r.path, r.error))
            continue
        sus = r.suspicious
        flag = 'WEAK' if sus else 'ok  '
        print('  %s %-64s killed %d/%d%s' % (
            flag, r.path, r.killed, r.total,
            '' if not r.survivors else '  (%d display-only survivor(s) ignored)'
            % (len(r.survivors) - len(sus))))
        for s in sus:
            print('         SUSPICIOUS line %d: %r -> %s  |  %s'
                  % (s['line'], s['was'], s['became'], s['snippet']))
        weak += bool(sus)
    print('\n%d file(s) checked, %d with a SUSPICIOUS surviving mutant' % (len(targets), weak))
    return 0


if __name__ == '__main__':
    sys.exit(main())
