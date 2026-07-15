#!/usr/bin/env python3
# =============================================================================
# Concept Structure Gate (v9) — deterministic per-concept-unit TRIAD check.
# =============================================================================
# Runs on the SOURCE (mode:concept). For every @@@ concept block asserts the
# three beats IN ORDER: (1) intro prose BEFORE its first visual, (2) a real
# visual (%%% svg / %%% viz / a closed <svg>...</svg>), (3) build-up prose AFTER
# the visual. Complements concept_shell_gate (which checks "a visual exists" on
# compiled HTML) by enforcing intuition-first ordering. Semantic quality
# (is the analogy good? intuition-first *in spirit*?) is the LLM judge's job
# (coverage_judge.judge_concept_structure) — this gate is the cheap structural
# floor.
#
# Reusable:  from concept_structure_gate import run ; ok, msgs = run(source_text)
# CLI:       python3 gates/concept_structure_gate.py <source.md>   (exit 0/3)
# =============================================================================
import sys, re

_MIN_PROSE = 40  # chars of real prose required on each side of the visual (tunable)
_VIS_OPEN = re.compile(r'^%%%\s+(svg|viz)\b', re.MULTILINE)
_SVG_CLOSED = re.compile(r'<svg[\s>].*?</svg>', re.DOTALL)
_WIDGET = re.compile(r'%%%.*?%%%', re.DOTALL)  # strip any widget when measuring prose


def _concept_blocks(body):
    """Yield (args_line, block_body) for each '@@@ concept ...' up to the next '@@@'."""
    for part in re.split(r'(?m)^@@@\s+', body):
        if part.startswith('concept'):
            line, _, rest = part.partition('\n')
            yield line, rest


def run(source_text):
    """Return (ok: bool, msgs: [str]). msgs are 'pass '/'FAIL ' prefixed labels."""
    msgs, ok = [], [True]

    def chk(cond, label):
        msgs.append(('pass ' if cond else 'FAIL ') + label)
        ok[0] = ok[0] and bool(cond)

    body = re.sub(r'^---.*?\n---\s*', '', source_text, count=1, flags=re.DOTALL)
    blocks = list(_concept_blocks(body))
    chk(len(blocks) >= 3, '>=3 concept units (got %d)' % len(blocks))

    for args, text in blocks:
        m = re.search(r'id=(?:"([^"]+)"|(\S+))', args)
        cid = (m.group(1) or m.group(2)) if m else '?'

        vis = _VIS_OPEN.search(text)
        svg = _SVG_CLOSED.search(text)
        # first visual is whichever appears earliest
        first = min([x for x in (vis, svg) if x], key=lambda mm: mm.start(), default=None)
        chk(bool(first), 'concept %s has a visual' % cid)
        if not first:
            continue

        intro = _WIDGET.sub('', text[:first.start()]).strip()
        chk(len(intro) >= _MIN_PROSE, 'concept %s has intro prose before its visual' % cid)

        # find where the first visual ends, then measure build-up after it
        if first is vis:
            close = re.search(r'(?m)^%%%\s*$', text[first.end():])
            after = text[first.end():][close.end():] if close else ''
        else:
            after = text[first.end():]
        buildup = _WIDGET.sub('', after).strip()
        chk(len(buildup) >= _MIN_PROSE, 'concept %s has build-up after its visual' % cid)

    return ok[0], msgs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    a = ap.parse_args()
    ok, msgs = run(open(a.source, encoding='utf-8').read())
    for m in msgs:
        print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
