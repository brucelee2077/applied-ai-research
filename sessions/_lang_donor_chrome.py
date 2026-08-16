#!/usr/bin/env python3
# =============================================================================
# Pair the SHARED chrome strings in m01's six per-lesson donors.
# =============================================================================
# Why this exists, and why it is one script rather than six agents:
#
# m01's days are region-mode. Their reader-facing prose lives in `@@@ region`
# blocks and is already bilingual, but sections 3/5/6 (play / build / quiz) and
# the shell chrome are DONOR-owned — they never pass through a region, so they
# reached a Chinese reader in English. Measured on the compiled pages with a
# stack-based HTML parse: 9,397 characters of untwinned reader-visible prose
# across the six days (code and identifiers excluded and counted separately).
#
# 23 of those strings are BYTE-IDENTICAL in all six days, totalling 572
# characters. Handing those to six parallel agents would produce six slightly
# different Chinese renderings of "Playground" — the exact tone drift that
# CLAUDE.md section 4 exists to prevent. So they are translated once, here, and
# applied mechanically. Only the day-SPECIFIC play/build prose needs authoring.
#
# The pairing markup is deliberately the same shape the compiler emits for
# concept-mode days: <span class="lang-en">…</span><span class="lang-zh">…</span>.
# A node with NEITHER class shows under both languages, so anything this script
# misses degrades to English rather than going blank.
#
# Usage:  python3 sessions/_lang_donor_chrome.py [--check]
# Exit 0 = all six donors carry the pairs; 1 = a replacement did not apply.
# =============================================================================
import glob
import os
import re
import sys

DONORS = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '_compiler', 'shells', 'm01-day-0*.donor')))

# -- the shared chrome table --------------------------------------------------
# Each entry is (english, chinese, expected_occurrences_per_donor).
# Counts were measured per donor before writing this; a mismatch fails closed
# rather than silently patching a different number of sites on different days.
SHARED = [
    ('☰ Sections',                                   '☰ 目录',                    1),
    ('/7 sections done',                             '/7 节完成',                  1),
    ('Playground',                                   '试一试',                     2),
    ('Build it up',                                  '一步步搭起来',                2),
    ('Quiz',                                         '小测',                      2),
    ('Click all three',                              '三个都点一遍',                1),
    ('to finish this section.',                      '就算完成这一节。',             1),
    ('try all three first',                          '先把三个都点一遍',             1),
    ('Each click shows a one-line "what this means" here.',
     '每点一次，这里会出现一行「这是什么意思」。',                                    1),
    ('# click a button above to start',              '# 点上面的按钮开始',           1),
    ('↓ scroll to reveal each piece ↓',              '↓ 往下滚，一块一块出现 ↓',     1),
    ('scroll through the build first',               '先把搭建过程滚完',             1),
    ('Click an answer — instant feedback on each',   '点一个答案 —— 每题都会立刻给你反馈', 1),
    ('4 questions, instant feedback.',               '四道题，每题立刻给反馈。',       1),
    ('Answer all four',                              '四题都答完',                  1),
    ('to complete today.',                           '就算今天完成。',               1),
    ('answer all four first',                        '先把四题都答完',               1),
    ('📋 copy',                                       '📋 复制',                   1),
]

# Deliberately NOT paired, with the reason recorded so nobody "fixes" it later:
#   'Frontier'            — the site's brand name, a proper noun.
#   'Light' / 'Dark' / 'Midnight' — theme names; they are labels on a control
#                           whose values are also written in English in
#                           localStorage, and the existing zh UI table leaves
#                           them alone for consistency.
#   'Language · 语言'      — already bilingual by construction.
#   the <title> and the footer 'Frontier Lab · Foundations — …' blurb — the
#                           title is not body prose and the footer is a
#                           technical note about localStorage, not a lesson.

PAIR = '<span class="lang-en">%s</span><span class="lang-zh">%s</span>'

# -- strings JS WRITES at runtime -------------------------------------------
# CSS cannot reach a `textContent =` assignment, so these four cannot be paired
# with spans — the assignment would just overwrite the pair. They have to go
# through the donor's existing UI/ui() table, which reads <html data-lang> at
# call time. Measured: all four appear exactly once in each of the six donors.
#
# NOT propagated to v9-base.donor on purpose. These four keys only exist because
# m01's donors own an s3/s5/s6 gotit button and a quiz feedback prefix; the 47
# concept-mode lessons have no such markup. Adding dead keys to v9-base would
# change its JS and force a recompile of all 47 for no reader-visible gain. If
# _lang_shell_sweep.py is ever re-run over m01 it would refresh this block from
# v9-base and drop these keys — that is caught loudly, not silently, by
# concept_shell_gate's donor byte-identity check.
RUNTIME = [
    # (key, english, chinese, the exact JS expression being replaced)
    ('saw_all_three', 'Saw all three — got it ✓', '三个都看了 —— 明白了 ✓',
     "g.textContent='Saw all three — got it ✓'"),
    ('saw_build', 'Saw the whole build — got it ✓', '整个搭建过程都看了 —— 明白了 ✓',
     "g.textContent='Saw the whole build — got it ✓'"),
    ('answered_four', 'All four answered — check ✓', '四题都答完了 —— 看结果 ✓',
     "g.textContent='All four answered — check ✓'"),
    ('wrong_prefix', 'The correct answer is the green one. ', '正确答案是绿色那个。',
     "fb.innerHTML='The correct answer is the green one. '+item.fb"),
]

# Where to splice the new keys into each language's object literal in the table.
_EN_ANCHOR = "reset_confirm:\"Reset today's progress?\", sections_done:' sections done'}"
_ZH_ANCHOR = ("reset_confirm:'\\u8981\\u6e05\\u7a7a\\u4eca\\u5929\\u7684\\u8fdb\\u5ea6\\u5417\\uff1f', "
              "sections_done:' \\u5c0f\\u8282\\u5b8c\\u6210'}")


def _patch_runtime(text, path):
    """Add the 4 keys to UI.en / UI.zh and route the JS through ui()."""
    if "ui('saw_all_three')" in text:
        return text, 0                      # idempotent
    en_add = ', '.join("%s:%r" % (k, en) for k, en, _zh, _js in RUNTIME)
    zh_add = ', '.join("%s:%r" % (k, zh) for k, _en, zh, _js in RUNTIME)
    for anchor, add in ((_EN_ANCHOR, en_add), (_ZH_ANCHOR, zh_add)):
        if text.count(anchor) != 1:
            raise SystemExit('FAIL %s: UI table anchor matched %d time(s), expected 1. '
                             'The table changed — re-measure.'
                             % (os.path.basename(path), text.count(anchor)))
        text = text.replace(anchor, anchor[:-1] + ', ' + add + '}', 1)
    for key, _en, _zh, js in RUNTIME:
        if text.count(js) != 1:
            raise SystemExit('FAIL %s: %r appears %d time(s), expected 1.'
                             % (os.path.basename(path), js[:44], text.count(js)))
        if js.endswith("+item.fb"):
            new = "fb.innerHTML=ui('wrong_prefix')+item.fb"
        else:
            new = js.split('=', 1)[0] + "=ui('%s')" % key
        text = text.replace(js, new, 1)
    return text, len(RUNTIME)



def patch(text, path):
    """Return (new_text, n_applied). Fails closed on any count mismatch."""
    applied = 0
    for en, zh, want in SHARED:
        # Skip a string already paired — makes the script idempotent, so a
        # re-run after a donor edit does not double-wrap.
        if PAIR % (en, zh) in text:
            applied += 1
            continue
        n = text.count(en)
        if n != want:
            raise SystemExit(
                'FAIL %s: %r appears %d time(s), expected %d. The donor changed '
                '— re-measure before patching.' % (os.path.basename(path), en, n, want))
        text = text.replace(en, PAIR % (en, zh), want)
        applied += 1
    return text, applied


def main():
    check = '--check' in sys.argv
    if not DONORS:
        raise SystemExit('FAIL no m01 donors found')
    bad = 0
    for d in DONORS:
        src = open(d, encoding='utf-8').read()
        if check:
            missing = [en for en, zh, _ in SHARED if PAIR % (en, zh) not in src]
            if "ui('saw_all_three')" not in src:
                missing.append('<runtime ui() keys>')
            if missing:
                bad += 1
                print('  FAIL %-38s %d unpaired: %r…'
                      % (os.path.basename(d), len(missing), missing[0]))
            else:
                print('  ok   %-38s all %d shared + %d runtime strings paired'
                      % (os.path.basename(d), len(SHARED), len(RUNTIME)))
            continue
        out, n = patch(src, d)
        out, r = _patch_runtime(out, d)
        if out != src:
            open(d, 'w', encoding='utf-8').write(out)
        print('  %-38s %d shared + %d runtime paired%s'
              % (os.path.basename(d), n, r, '' if out != src else ' (already current)'))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
