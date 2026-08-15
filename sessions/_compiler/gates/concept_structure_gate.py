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
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

# English-character EQUIVALENTS of real prose required on each side of the visual.
# Measured with v8lib.text_weight, not len(): 40 Han characters is roughly 92
# English characters' worth of text, so a raw len() check is ~2.3x stricter on a
# Chinese lesson than the number it was calibrated as.
_MIN_PROSE = 40
_VIS_OPEN = re.compile(r'^%%%\s+(svg|viz)\b', re.MULTILINE)
_SVG_CLOSED = re.compile(r'<svg[\s>].*?</svg>', re.DOTALL)
_WIDGET = re.compile(r'%%%.*?%%%', re.DOTALL)  # strip any widget when measuring prose
# --- "visualize the build-up" ADVISORY (warn-only) markers ------------------
# A build-up is HEAVY (should be DRAWN, not just written) iff it carries a Math
# Ladder or math demoted into an "Optional (skippable)" callout box. Keyed off
# these EXPLICIT author markers so the warn ~never false-positives. A build-up
# visual is any svg/viz/demo/mathladder that sits AFTER the opening anchor visual.
_MATHLADDER = re.compile(r'(?m)^%%%\s+mathladder\b')
_BUILDUP_VIS = re.compile(r'(?m)^%%%\s+(svg|viz|demo|mathladder)\b')
# Build-up CONTENT widget (satisfies the "has build-up after its visual" prose FLOOR — a
# %%% steps narrated worked-example, or any build-up widget, IS substantial build-up even
# with no surrounding prose). Distinct from _BUILDUP_VIS: `steps` is narration, NOT a
# visual, so it counts toward the build-up floor but never toward "visualize the build-up".
_BUILDUP_CONTENT = re.compile(r'(?m)^%%%\s+(steps|demo|mathladder|svg|viz)\b')
# an "Optional…" demoted-math box: a `!!! c-… <emoji>` callout whose body's first
# line (after any leading HTML tags) begins with the word "Optional". `.` excludes
# newlines, so it matches ONLY when "Optional" is the box's immediate first line.
_OPT_BOX = re.compile(r'(?im)^!!!\s+\S.*\n\s*(?:<[^>]+>\s*)*Optional\b')


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
        chk(v8lib.text_weight(intro) >= _MIN_PROSE, 'concept %s has intro prose before its visual' % cid)

        # find where the first visual ends, then measure build-up after it
        if first is vis:
            close = re.search(r'(?m)^%%%\s*$', text[first.end():])
            after = text[first.end():][close.end():] if close else ''
        else:
            after = text[first.end():]
        buildup = _WIDGET.sub('', after).strip()
        # The build-up floor is met by ≥_MIN_PROSE chars of narration OR a build-up
        # content widget (%%% steps / demo / mathladder / a 2nd svg-viz) — a narrated
        # %%% steps worked-example IS substantial build-up even with no surrounding prose.
        has_bw = bool(_BUILDUP_CONTENT.search(after))
        chk(v8lib.text_weight(buildup) >= _MIN_PROSE or has_bw, 'concept %s has build-up after its visual' % cid)

        # -- ADVISORY (warn-only; NEVER flips ok[0] / exit code): visualize the build-up --
        # If this concept's build-up is HEAVY (a Math Ladder, or math demoted into an
        # "Optional (skippable)" box), the build-up should itself be SHOWN in the build-up
        # region (a 2nd svg/demo/viz after the anchor, or the Math Ladder itself). The LLM
        # buildup_visualized judge axis is the real enforcer; this is the cheap offline floor.
        after_start = len(text) - len(after)
        ml = _MATHLADDER.search(text)
        opt = _OPT_BOX.search(text)
        if ml or opt:
            why = 'Math Ladder' if ml else "'Optional (skippable)' box"
            ladder_in_buildup = bool(ml) and ml.start() >= after_start
            vis_in_buildup = any(mm.start() >= after_start for mm in _BUILDUP_VIS.finditer(text))
            if not (ladder_in_buildup or vis_in_buildup):
                msgs.append('warn concept ' + cid + ': heavy build-up (' + why + ') has no build-up '
                            'visual in the build-up region — add a %%% svg/demo/viz that draws the '
                            'mechanism/worked example (advisory)')

    # -- ADVISORY (warn-only; NEVER flips ok[0] / exit code): failure-mode momentum cluster --
    # >=3 CONSECUTIVE concept units whose title/tag reads as a failure/limit AND that carry no
    # play/payoff widget (%%% demo / %%% viz) => a late-lesson "trap wall" where a beginner's
    # momentum dies (observed on 7/9 m02 days). Remedy is to INTERLEAVE a win / live widget
    # between the traps, NOT to cut or defer coverage. The interest judge's `momentum` lever is
    # the real enforcer; this is the cheap offline floor. 'problem' is deliberately NOT a token
    # (too generic); the >=3-consecutive threshold guards against incidental single matches.
    _FAIL = re.compile(r"(?i)\b(dead|vanish\w*|saturat\w*|explod\w*|diverg\w*|collapse|overfit\w*|"
                       r"underfit\w*|nan|trap|puzzle|pitfall|wall|cannot|can['’]?t|fails?|"
                       r"failure|limits?|breaks?)\b")
    _PLAY = re.compile(r'(?m)^%%%\s+(demo|viz)\b')
    run_len = 0
    for args, text in blocks:
        if _FAIL.search(args) and not _PLAY.search(text):
            run_len += 1
            if run_len == 3:
                msgs.append('warn failure-mode cluster: >=3 consecutive failure/limit concept units '
                            'with no play/payoff widget between them — interleave a win or a live '
                            '%%% demo/viz to keep momentum (advisory; do not cut coverage)')
        else:
            run_len = 0

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
