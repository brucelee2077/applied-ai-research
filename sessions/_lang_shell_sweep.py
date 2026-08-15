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
    'ui':         ('/* frontier-lang:ui */', '/* /frontier-lang:ui */'),
    'controller': ('/* frontier-lang:controller */', '/* /frontier-lang:controller */'),
}

# Strings the code REPLACES at runtime, or renders once as chrome. A CSS toggle
# cannot reach a textContent assignment, so in Chinese mode the page would flip
# back to English the moment the reader pressed anything. Unlike the sentinel
# blocks these are edits IN PLACE, so the contract is different: measured across
# the 293 shell pages, every one of these appears 0 or 1 times and never more —
# the dynamic widget strings only exist on the 47 compiled pages, the chrome on
# 261-293. So: replace when present, skip when absent, FAIL when duplicated.
# Idempotent, because after the replacement the old text is gone.
STRING_SUBS = [
    ("run.textContent = 'ran ✓';", "run.textContent = ui('reveal_done');"),
    ("g.textContent='All answered — check ✓';", "g.textContent=ui('all_answered');"),
    ('btn.textContent = "— that\'s all the hints —";', "btn.textContent = ui('hints_end');"),
    ("btn.textContent = '💡 still stuck? another hint ('+shown+'/'+tiers.length+')';",
     "btn.textContent = ui('hint_more')+' ('+shown+'/'+tiers.length+')';"),
    ("btn.textContent=ok?'✓ copied':'select & copy manually';",
     "btn.textContent=ok?ui('copied'):ui('copy_manual');"),
    ("if(!confirm('Reset today\\'s progress?')) return;",
     "if(!confirm(ui('reset_confirm'))) return;"),
    ("+total+' sections done';", "+total+ui('sections_done');"),
    ('<span class="nav-group-label" style="padding:0">Appearance</span>',
     '<span class="nav-group-label" style="padding:0"><span class="lang-en">Appearance</span>'
     '<span class="lang-zh">外观</span></span>'),
    ('<div class="nav-group-label">Progress checklist</div>',
     '<div class="nav-group-label"><span class="lang-en">Progress checklist</span>'
     '<span class="lang-zh">进度清单</span></div>'),
    ('type="button">↺ Reset progress</button>',
     'type="button"><span class="lang-en">↺ Reset progress</span>'
     '<span class="lang-zh">↺ 清空进度</span></button>'),
    ('<span class="d">← Prev</span>',
     '<span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span>'),
    ('<span class="d">Next →</span>',
     '<span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span>'),
    ('<span class="d">▦ Map</span>',
     '<span class="d"><span class="lang-en">▦ Map</span><span class="lang-zh">▦ 地图</span></span>'),
    ('<span class="t">Back to curriculum</span>',
     '<span class="t"><span class="lang-en">Back to curriculum</span>'
     '<span class="lang-zh">回到课程地图</span></span>'),
]


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


# Where a MISSING block gets inserted. One table serves both jobs — first sweep and
# later refresh — because they are the same job: make the page match the donor. The
# first version could only REPLACE an existing block, so adding the `ui` block left
# 246 pages permanently reporting "cannot refresh ui" with no way to fix them.
# 'controller' anchors on the ui CLOSE sentinel rather than on setTheme, so the two
# cannot land in the wrong order when both are missing.
INSERT_AT = {
    'prepaint':   (lambda h: A_PREPAINT.search(h), '\n', ''),
    'css':        (lambda h: A_CSS.search(h), '\n', ''),
    'markup':     (lambda h: A_MARKUP.search(h), '\n\n', ''),
    'ui':         (lambda h: A_SETTHEME.search(h), '\n\n', ''),
    'controller': (lambda h: re.search(re.escape(SENTINELS['ui'][1]), h), '\n', ''),
}


def _sync_block(rel, out, name, parts, problems):
    """Make `out` carry the donor's current text for one sentinel block."""
    a_tok, b_tok = SENTINELS[name]
    if parts[name] in out:
        return out                                     # already current
    na, nb = out.count(a_tok), out.count(b_tok)
    if na == 1 and nb == 1:                            # present but stale -> replace
        a = out.find(a_tok)
        b = out.find(b_tok, a)
        if b < 0:
            problems.append('%s: %s close sentinel precedes the open one' % (rel, name))
            return out
        return out[:a] + parts[name] + out[b + len(b_tok):]
    if na == 0 and nb == 0:                            # absent -> insert at its anchor
        finder, pre, post = INSERT_AT[name]
        m = finder(out)
        if not m:
            problems.append('%s: cannot insert %s — its anchor is missing' % (rel, name))
            return out
        return out[:m.end()] + pre + parts[name] + post + out[m.end():]
    problems.append('%s: %s sentinels are unbalanced (%dx open, %dx close)'
                    % (rel, name, na, nb))
    return out


def _localize_strings(rel, html, problems):
    """Runtime strings that no CSS toggle can reach, because the code REPLACES
    textContent. Measured across the 293 shell pages: each appears 0 or 1 times,
    never more — the dynamic widget strings only exist on the 47 compiled pages,
    the chrome on 261-293. So replace when present, skip when absent, report when
    duplicated. Idempotent: after the replacement the old text is gone."""
    out = html
    for old, new in STRING_SUBS:
        n = out.count(old)
        if n == 0:
            continue
        if n > 1:
            problems.append('%s: runtime string %r appears %d times (expected 0 or 1)'
                            % (rel, old[:40], n))
            continue
        out = out.replace(old, new, 1)
    return out


def plan_page(rel, html, parts, problems):
    """The rewritten page, or None if it already matches the donor.

    Insert, replace and localize are all the same job: make this page carry what the
    donor says. The checklist is the one special case — on a page that has never
    been swept it must REPLACE the old builder rather than be inserted beside it.
    """
    out = html
    has_checklist = 'id="checklist"' in html

    # the never-swept case: the old checklist builder has to go
    if has_checklist and SENTINELS['checklist'][0] not in out:
        ok = True
        for rx, name in CHECKLIST_ANCHORS:
            n = len(rx.findall(out))
            if n != 1:
                problems.append('%s: anchor %s matched %d times (expected 1)' % (rel, name, n))
                ok = False
        if ok:
            out = A_CHECKLIST_VAR.sub(lambda _m: parts['checklist'], out, count=1)
            out = A_CHECKLIST_BLOCK.sub('', out, count=1)

    for name in ('prepaint', 'css', 'markup', 'ui', 'controller'):
        out = _sync_block(rel, out, name, parts, problems)
    if has_checklist:
        out = _sync_block(rel, out, 'checklist', parts, problems)
    out = _localize_strings(rel, out, problems)
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
