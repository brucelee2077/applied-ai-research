#!/usr/bin/env python3
# =============================================================================
# Shell Invariant Gate  (v8 Phase C)  — runs on the COMPILED lesson.html.
# =============================================================================
# Asserts the compiler preserved every frozen invariant:
#   quest-id, 7 sections, 8 sidebar data-targets, DEMOS/BUILD/QS, playground>=3,
#   quiz q:4 o:16, localStorage keys, .fin, nav hrefs, no unresolved markers.
# Optionally (with --donor) proves CSS + JS-engine SHELL byte-identity, i.e. the
# compiler only rewrote content regions.
#
# Reusable:
#   from shell_invariant_gate import run ; ok, msgs = run(html, meta, donor=None)
# CLI:
#   python3 gates/shell_invariant_gate.py <lesson.html> --source <source.md> [--donor <donor>]
#   (exit 0 pass / 3 fail)
# =============================================================================
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib


def _shell_regions(html):
    """Return the (css, [scripts]) that must stay byte-identical to the donor."""
    css = re.search(r'<style>.*?</style>', html, re.DOTALL)
    scripts = re.findall(r'<script>.*?</script>', html, re.DOTALL)
    return (css.group(0) if css else None), scripts


def _mask_data(s):
    """Blank out the DEMOS/BUILD/QS data blocks so the JS *engine* can be compared to
    the donor while the authored playground/build/quiz DATA is allowed to change."""
    for name in ('DEMOS', 'BUILD', 'QS'):
        s = re.sub(v8lib.REGION_PATTERNS[name], '__%s__' % name, s, flags=re.DOTALL)
    return s


def run(html, meta, donor=None):
    msgs, ok = [], [True]
    def chk(cond, label):
        msgs.append(('pass ' if cond else 'FAIL ') + label); ok[0] = ok[0] and bool(cond)

    qid = meta.get('quest_id')
    if qid:
        chk(('data-quest-id="%s"' % qid) in html, 'quest-id frozen (%s)' % qid)
    chk(html.count('class="module-section"') == 7,
        '7 module-sections (got %d)' % html.count('class="module-section"'))
    for t in ['home', 's1', 's2', 's3', 's4', 's5', 's6', 's7']:
        chk(('data-target="%s"' % t) in html, 'sidebar data-target=%s' % t)
    chk('var DEMOS = {' in html, 'DEMOS present')
    chk('var BUILD=[' in html, 'BUILD present')
    chk('var QS=[' in html, 'QS present')
    chk(html.count('data-demo=') >= 3, 'playground >=3 demo buttons (got %d)' % html.count('data-demo='))
    nq = len(re.findall(r'\bans:\s*\d', html))
    chk(nq == 4, 'quiz has 4 questions (got %d)' % nq)
    chk(len(re.findall(r"opts:\[", html)) == 4, 'quiz has 4 opts arrays')
    chk('frontier-lesson:' in html, 'localStorage key frontier-lesson:')
    chk('frontier-theme' in html, 'localStorage key frontier-theme')
    chk('class="fin" id="fin"' in html, '.fin completion banner')
    chk('.gotit' in html or html.count('class="gotit"') == 7, 'gotit buttons present')
    chk(html.count('class="gotit"') == 7, '7 gotit buttons (got %d)' % html.count('class="gotit"'))
    if meta.get('nav_prev_href'):
        chk(('href="%s"' % meta['nav_prev_href']) in html, 'prev nav href')
    if meta.get('nav_next_href'):
        chk(('href="%s"' % meta['nav_next_href']) in html, 'next nav href')
    # One check per marker, so the message names the leak instead of just saying no.
    # `~~~` joins the list because it is a block fence: render_md consumes `~~~html`
    # and `~~~zh`, but an unterminated one falls through to the paragraph branch and
    # ships as literal text the reader sees.
    _leaked = [mk for mk in ('{{', '@@@', '%%%', '~~~') if mk in html]
    chk(not _leaked, 'no unresolved markers%s' % ('' if not _leaked else ' (leaked: %s)' % ', '.join(map(repr, _leaked))))
    chk('experiment.py' in html, 'artifact (experiment.py) referenced')

    if donor is not None:
        dc, ds = _shell_regions(donor)
        hc, hs = _shell_regions(html)
        chk(dc == hc, 'CSS block byte-identical to donor')
        chk(len(ds) == len(hs), 'same number of <script> blocks (%d)' % len(hs))
        for i, (a, b) in enumerate(zip(ds, hs)):
            # compare the JS engine, masking DEMOS/BUILD/QS (editable data lives inside script #1)
            chk(_mask_data(a) == _mask_data(b), 'script #%d engine byte-identical to donor (data masked)' % i)
    return ok[0], msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson')
    ap.add_argument('--source')
    ap.add_argument('--donor')
    args = ap.parse_args()
    html = open(args.lesson, encoding='utf-8').read()
    meta = {}
    if args.source:
        meta, _ = v8lib.split_frontmatter(open(args.source, encoding='utf-8').read())
    else:
        m = re.search(r'data-quest-id="([^"]+)"', html)
        if m: meta['quest_id'] = m.group(1)
    donor = open(args.donor, encoding='utf-8').read() if args.donor else None
    ok, msgs = run(html, meta, donor=donor)
    print('== Shell Invariant Gate:', os.path.relpath(args.lesson), '==')
    for m in msgs: print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
