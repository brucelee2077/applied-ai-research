#!/usr/bin/env python3
# =============================================================================
# _lang_shell_sweep.py — put the reading-language toggle on every shell page
# =============================================================================
# The 47 lessons compiled from a source.md get the toggle by recompiling against
# v9-base.donor. The other ~246 pages carrying the same sidebar shell are hand-
# written or migrated by _shell_migrate.py and have no compiler, so they need a
# sweep. This is that sweep.
#
# Every inserted byte is SLICED OUT OF THE DONOR, never written here. Two copies
# of the same JavaScript in two files is how shells/js/sr.js and reveal.js became
# unverified mirrors of inlined donor code ("mirror of shells/js/reveal.js" in the
# donor, with nothing checking they still match). Reading the donor at run time
# makes divergence impossible by construction.
#
# FAILS CLOSED. Every anchor must match exactly once on every page; a page whose
# anchor is missing or duplicated is reported and the run exits 1. Nothing is ever
# skipped silently, because a silently skipped page is a dead button nobody
# notices. Measured before this was written, over the 246 pages:
#   * the theme-CSS / theme-row / setTheme-init anchors match exactly once on 246
#   * the checklist builder block is BYTE-IDENTICAL across the 244 that have one,
#     so the donor's own version of it is exactly what each page needs
#   * index.html and roadmap.html carry the shell but have no #checklist and no
#     refresh(), which is why the donor's language controller guards both calls
#     with typeof — one shared text, no per-page variants
#
# Usage:
#   python3 sessions/_lang_shell_sweep.py            # check only, no writes
#   python3 sessions/_lang_shell_sweep.py --apply    # rewrite the pages
#   exit 0 = all good ; 1 = an anchor did not match exactly once
# =============================================================================
import sys, os, re, glob, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DONOR = os.path.join(HERE, '_compiler', 'shells', 'v9-base.donor')


def _slice(donor, start, end, label):
    a = donor.find(start)
    if a < 0:
        raise SystemExit('donor marker missing (%s start): %s' % (label, start[:70]))
    b = donor.find(end, a)
    if b < 0:
        raise SystemExit('donor marker missing (%s end): %s' % (label, end[:70]))
    return donor[a:b + len(end)]


SENTINELS = {
    # name        (open sentinel, close sentinel)
    'prepaint':   ('<!-- frontier-lang:prepaint -->', '<!-- /frontier-lang:prepaint -->'),
    'css':        ('/* frontier-lang:css */', '/* /frontier-lang:css */'),
    'markup':     ('    <!-- frontier-lang:markup -->', '    <!-- /frontier-lang:markup -->'),
    'checklist':  ('/* frontier-lang:checklist */', '/* /frontier-lang:checklist */'),
    'controller': ('/* frontier-lang:controller */', '/* /frontier-lang:controller */'),
}


def donor_parts(donor_text=None):
    """The five blocks the toggle is made of, verbatim from the donor.

    Delimited by SENTINEL COMMENTS, not by their own content. The first version of
    this keyed the boundaries off the first and last line of each block, which
    works for inserting and is useless for refreshing: locating an OLD block needs
    a marker that exists in the old text, and the natural end marker is exactly
    what a content edit changes. Adding one CSS rule made 293 pages unrefreshable
    with "end marker missing". Sentinels never change, so anything between them
    can.
    """
    d = donor_text if donor_text is not None else open(DONOR, encoding='utf-8').read()
    parts = {}
    for name, (a_tok, b_tok) in SENTINELS.items():
        a = d.find(a_tok)
        b = d.find(b_tok, a) if a >= 0 else -1
        if a < 0 or b < 0:
            raise SystemExit('donor is missing the %s sentinel pair (%s ... %s)'
                             % (name, a_tok, b_tok))
        parts[name] = d[a:b + len(b_tok)]
    return parts


# --- anchors on the target pages ---------------------------------------------
# A_PREPAINT stops at the comment's opening words on purpose: index.html and
# roadmap.html append their own note to it ("— hub defaults to dark", "— defaults
# to dark, shares the hub/lesson key") because those two default to dark while
# lessons default to dim. Matching the full lesson comment missed exactly those
# two pages, and the sweep reported it instead of skipping them.
A_PREPAINT = re.compile(r"<script>/\* set appearance before paint.*?</script>", re.S)
A_CSS = re.compile(r"\.theme-btn\.active\{background:var\(--accent\);color:#fff\}")
A_MARKUP = re.compile(r'<div class="theme-row"[^>]*>.*?</div>\s*\n\s*</div>', re.S)
A_SETTHEME = re.compile(r"setTheme\(document\.documentElement\.getAttribute\('data-theme'\)\s*\|\|\s*'\w+'\);")
A_CHECKLIST_VAR = re.compile(r"var checklist = document\.getElementById\('checklist'\), checkItems = \{\};")
A_CHECKLIST_BLOCK = re.compile(
    r"secs\.forEach\(function\(sec\)\{\n"
    r"  var key = sec\.getAttribute\('data-sec'\);\n"
    r"  var title = sec\.querySelector\('\.sec-h'\) \? sec\.querySelector\('\.sec-h'\)\.textContent : key;\n"
    r".*?\n  checkItems\[key\] = li;\n\}\);", re.S)

CORE_ANCHORS = ((A_PREPAINT, 'theme-prepaint'), (A_CSS, 'theme-css'),
                (A_MARKUP, 'theme-row-markup'), (A_SETTHEME, 'setTheme-init'))
CHECKLIST_ANCHORS = ((A_CHECKLIST_VAR, 'checklist-var'), (A_CHECKLIST_BLOCK, 'checklist-block'))


def shell_pages(root):
    """Pages carrying the sidebar shell, excluding the A/B scratch dirs that
    publish_pages skips and never serves."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, 'sessions', '**', '*.html'), recursive=True)):
        rel = os.path.relpath(p, os.path.join(root, 'sessions'))
        if rel.split(os.sep)[0] in ('_coldgen', '_compare'):
            continue
        h = open(p, encoding='utf-8').read()
        if 'class="theme-row"' in h:
            out.append((rel, p, h))
    return out


def plan_page(rel, html, parts, problems):
    """The rewritten page, or None if it needs no change or cannot be done.

    Two jobs. On a page without the toggle: insert all five blocks. On a page that
    already has it but whose blocks no longer match the donor: REFRESH those
    blocks. Without the refresh half, every future donor edit would silently leave
    293 pages behind, which is the exact drift this script exists to prevent —
    and test_page_toggle_text_is_byte_identical_to_the_donor would go red with no
    tool to make it green again.

    Appends a message to `problems` for every anchor that is not unique — the
    caller turns a non-empty `problems` into exit 1.
    """
    if 'class="lang-row"' in html:
        return refresh_page(rel, html, parts, problems)

    has_checklist = 'id="checklist"' in html
    ok = True
    for rx, name in CORE_ANCHORS + (CHECKLIST_ANCHORS if has_checklist else ()):
        n = len(rx.findall(html))
        if n != 1:
            problems.append('%s: anchor %s matched %d times (expected 1)' % (rel, name, n))
            ok = False
    if not ok:
        return None

    out = html
    # 1. the language pre-paint IIFE, right after the appearance one
    m = A_PREPAINT.search(out); out = out[:m.end()] + '\n' + parts['prepaint'] + out[m.end():]
    # 2. the CSS, right after the last .theme-btn rule
    m = A_CSS.search(out);      out = out[:m.end()] + '\n' + parts['css'] + out[m.end():]
    # 3. the sidebar row, right after the Appearance row
    m = A_MARKUP.search(out);   out = out[:m.end()] + '\n\n' + parts['markup'] + out[m.end():]
    # 4. the language-aware checklist builder — the donor's own text, because the
    #    block it replaces is byte-identical on every page that has one
    if has_checklist:
        # the sentinel block carries the var line, secLabel, buildChecklist and its
        # call, so the old var line becomes the whole block and the old forEach
        # goes away. Doing it in that order handles both shipped layouts: 215 pages
        # declare shorten() after the forEach, 29 review pages before it.
        out = A_CHECKLIST_VAR.sub(lambda _m: parts['checklist'], out, count=1)
        out = A_CHECKLIST_BLOCK.sub('', out, count=1)
    # 5. the controller, right after setTheme's init call
    m = A_SETTHEME.search(out); out = out[:m.end()] + '\n\n' + parts['controller'] + out[m.end():]
    return out


def refresh_page(rel, html, parts, problems):
    """Bring an already-swept page's blocks back to the donor's text.

    Located by sentinel, so a donor edit INSIDE a block propagates to all 293
    pages and an edit to a sentinel itself is reported rather than half-applied.
    """
    out = html
    for name, (a_tok, b_tok) in SENTINELS.items():
        if name == 'checklist' and 'id="checklist"' not in html:
            continue
        if parts[name] in out:
            continue                                   # already current
        if out.count(a_tok) != 1 or out.count(b_tok) != 1:
            problems.append('%s: cannot refresh %s — sentinel %s appears %dx and %s %dx (want 1 each)'
                            % (rel, name, a_tok, out.count(a_tok), b_tok, out.count(b_tok)))
            continue
        a = out.find(a_tok)
        b = out.find(b_tok, a)
        if b < 0:
            problems.append('%s: cannot refresh %s — close sentinel precedes the open one' % (rel, name))
            continue
        out = out[:a] + parts[name] + out[b + len(b_tok):]
    return out if out != html else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='rewrite the pages (default: report only)')
    ap.add_argument('--root', default=os.path.dirname(HERE))
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()

    parts = donor_parts()
    pages = shell_pages(a.root)
    problems, changed, already = [], [], []
    for rel, path, html in pages:
        new = plan_page(rel, html, parts, problems)
        if new is None:
            already.append(rel)
            continue
        changed.append(rel)
        if a.apply:
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(new)
            os.replace(tmp, path)

    print('== language shell sweep — %s ==' % ('APPLY' if a.apply else 'CHECK, no writes'))
    print('   shell pages         : %d' % len(pages))
    print('   already current     : %d' % len(already))
    print('   %-19s : %d' % ('rewritten' if a.apply else 'would rewrite', len(changed)))
    if a.verbose:
        for r in changed:
            print('      +', r)
    if problems:
        print('\n   ANCHOR PROBLEMS (%d) — those pages were NOT touched:' % len(problems))
        for p in problems:
            print('      !', p)
        print('\nFAIL — an anchor did not match exactly once. Nothing was skipped silently.')
        sys.exit(1)
    print('\nOK')


if __name__ == '__main__':
    main()
