#!/usr/bin/env python3
# =============================================================================
# Visual Integrity Gate  — catches "compiles green but renders BLANK" visuals.
# =============================================================================
# concept_shell_gate proves a visual MARKER is present (a closed <svg> OR a
# build-embed iframe). It does NOT prove the visual will actually RENDER. The
# recurring failure mode: a `%%% viz src=…` iframe whose file is missing, whose
# vendored JS (e.g. d3.v7.min.js) is missing, or whose height-sender protocol
# drifted from the lesson receiver — all COMPILE CLEAN yet show an empty box at
# runtime (AUTHORING.md §4 admits this). Likewise an inline `<svg>` with only a
# trivial `M0 0` path is a marker with nothing drawn.
#
# This gate proves the DETERMINISTIC, checkable half of "will it render":
#   1. every `%%% viz src=PATH` file EXISTS (path resolved from the lesson dir);
#   2. each viz file's local `<script src=…>` deps EXIST (the missing-d3 case);
#   3. each viz file carries a postMessage height-sender whose message TYPE
#      matches what the lesson's donor receiver filters on (protocol match) —
#      else the iframe sticks at its default height or never resizes;
#   4. every inline `%%% svg` is NON-DEGENERATE (has a viewBox AND a real shape,
#      not just an empty/near-empty path).
#
# It CANNOT verify actual pixels — there is no browser in this environment, and
# a nested `file://` iframe may be blocked by the browser regardless of content
# (serve over http to view). Those residuals need an http-served headless run or
# a human screenshot. This gate closes the part a test CAN own.
#
# Reusable:  from visual_integrity_gate import run ; ok, msgs = run(source_path)
# CLI:       python3 gates/visual_integrity_gate.py <source.md>   (exit 0 / 4)
# =============================================================================
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

_GATES_DIR = os.path.dirname(os.path.abspath(__file__))
DONOR = os.path.join(os.path.dirname(_GATES_DIR), 'shells', 'v9-base.donor')

# real drawing elements (an <svg> with none of these draws nothing visible)
_SHAPE = re.compile(r'<(path|rect|circle|ellipse|line|polyline|polygon|text|image|use)\b', re.I)
# a path is "real" only if it has a draw command beyond a lone move/close
_PATH_DRAW = re.compile(r'[LCQAHVSTlcqahvst]')


def _expected_msg_type(donor_path=DONOR):
    """The message `type` the lesson's resize receiver filters on, e.g. 'viz-height'.
    Read it from the donor so a rename on either side is caught, not hard-coded."""
    try:
        d = open(donor_path, encoding='utf-8').read()
    except Exception:
        return 'viz-height'
    m = re.search(r"""d\.type\s*!==\s*['"]([^'"]+)['"]""", d)
    return m.group(1) if m else 'viz-height'


def _svg_is_degenerate(svg):
    """True if the <svg> has no viewBox, or draws nothing real (only empty/near-empty
    paths). A rect/circle/line/text/etc. counts as real content; a path counts only if
    it has a draw command beyond a bare move (so `M0 0` alone is degenerate)."""
    if 'viewbox' not in svg.lower():
        return True
    shapes = [s.lower() for s in _SHAPE.findall(svg)]
    if not shapes:
        return True
    if any(s != 'path' for s in shapes):   # has a non-path shape -> real content
        return False
    paths = re.findall(r'<path\b[^>]*\bd="([^"]*)"', svg, re.I)
    return not any(_PATH_DRAW.search(d) for d in paths)


def run(source_path, donor_path=DONOR):
    """Return (ok, msgs). N/A (ok=True) for non-concept sources."""
    msgs, ok = [], [True]
    def fail(m): ok[0] = False; msgs.append('FAIL ' + m)
    def pas(m): msgs.append('pass ' + m)

    src = open(source_path, encoding='utf-8').read()
    meta, _body = v8lib.split_frontmatter(src)
    if meta.get('mode') != 'concept':
        return True, ['skip (not concept mode — visual integrity gate is V9-only)']

    sdir = os.path.dirname(os.path.abspath(source_path))
    expected = _expected_msg_type(donor_path)

    # 1/2/3 — interactive viz embeds
    vizzes = re.findall(r'%%%\s*viz\s+src=(\S+)', src)
    for rel in vizzes:
        path = os.path.normpath(os.path.join(sdir, rel))
        if not os.path.exists(path):
            fail('viz embed points at a MISSING file: %s (renders an empty iframe)' % rel)
            continue
        pas('viz file exists: %s' % rel)
        vb = open(path, encoding='utf-8').read()
        for dep in re.findall(r'<script[^>]*src="([^"]+)"', vb):
            if dep.startswith('http'):
                continue
            dpath = os.path.normpath(os.path.join(os.path.dirname(path), dep))
            (pas('  local dep exists: %s' % dep) if os.path.exists(dpath)
             else fail('  viz %s needs local script %s which is MISSING (JS crashes -> blank)' % (rel, dep)))
        if 'postMessage' not in vb:
            fail('  viz %s has NO height-sender (postMessage) -> iframe stuck at default height' % rel)
        elif expected not in vb:
            fail("  viz %s sender protocol MISMATCH: receiver expects type '%s', not found in viz" % (rel, expected))
        else:
            pas("  viz %s has a matching height-sender ('%s')" % (rel, expected))

    # 4 — inline SVGs must draw something real
    degens = [m.group(0) for m in re.finditer(r'<svg\b.*?</svg>', src, re.DOTALL | re.I)
              if _svg_is_degenerate(m.group(0))]
    for svg in degens:
        snippet = re.sub(r'\s+', ' ', svg[:80])
        fail('degenerate inline SVG (no viewBox or nothing drawn): %s…' % snippet)

    total_svg = len(re.findall(r'<svg\b', src, re.I))
    pas('checked %d inline SVG(s) + %d viz embed(s)' % (total_svg, len(vizzes)))
    return ok[0], msgs


def main():
    if len(sys.argv) < 2:
        print('usage: visual_integrity_gate.py <source.md>'); sys.exit(1)
    ok, msgs = run(sys.argv[1])
    print('== Visual Integrity Gate:', os.path.relpath(sys.argv[1]), '==')
    for m in msgs:
        print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 4)


if __name__ == '__main__':
    main()
