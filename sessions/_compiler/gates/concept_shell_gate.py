#!/usr/bin/env python3
# =============================================================================
# Concept Shell Gate (v9) — asserts concept-lesson invariants on compiled HTML.
# =============================================================================
# The v9 analogue of shell_invariant_gate for mode:concept lessons. Runs on the
# COMPILED lesson.html and proves the concept-driven shell was assembled intact:
#   quest-id frozen, >=3 concept sections each with a visual + one gotit, exactly
#   one 4-question quiz section, one produce section (artifact-referenced),
#   sidebar nav parity, localStorage keys, .fin banner, no leaked markers.
#
# Reusable:  from concept_shell_gate import run ; ok, msgs = run(html, meta, donor=None)
# CLI:       python3 gates/concept_shell_gate.py <lesson.html> --source <source.md>
#            (exit 0 pass / 3 fail)
# =============================================================================
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

def run(html, meta, donor=None):
    msgs, ok = [], [True]
    def chk(cond, label):
        msgs.append(('pass ' if cond else 'FAIL ') + label); ok[0] = ok[0] and bool(cond)

    qid = meta.get('quest_id')
    if qid:
        chk(('data-quest-id="%s"' % qid) in html, 'quest-id frozen (%s)' % qid)

    ids = re.findall(r'<section class="module-section" id="(c\w+)"', html)
    chk(len(ids) >= 3, '>=3 concept sections (got %d)' % len(ids))
    for cid in ids:
        sec = re.search(r'id="%s".*?</section>' % re.escape(cid), html, re.DOTALL).group(0)
        has_visual = ('<svg' in sec) or ('build-embed' in sec)
        chk(has_visual, 'concept %s has a visual' % cid)
        chk(sec.count('class="gotit"') == 1, 'concept %s has exactly one gotit' % cid)

    chk(html.count('data-sec="quiz"') == 1, 'exactly one quiz section')
    chk(html.count('class="q"') == 4, 'quiz has 4 questions (got %d)' % html.count('class="q"'))
    chk(html.count('data-sec="produce"') == 1, 'exactly one produce section')
    if meta.get('require_artifact', True):
        prod = re.search(r'data-sec="produce".*?</section>', html, re.DOTALL)
        chk(bool(prod) and 'experiment.py' in prod.group(0), 'produce references an experiment.py artifact')

    targets = set(re.findall(r'data-target="([^"]+)"', html)) - {'home'}
    sec_ids = set(re.findall(r'<section class="module-section" id="([^"]+)"', html))
    chk(targets == sec_ids, 'sidebar nav parity (targets=%s sections=%s)' % (sorted(targets), sorted(sec_ids)))

    chk('frontier-lesson:' in html, 'localStorage frontier-lesson:')
    chk('frontier-theme' in html, 'localStorage frontier-theme')
    chk('class="fin" id="fin"' in html, '.fin banner')
    for marker in ('<!--V9_CONTENT-->', '<!--V9_NAV-->', '__QUEST_ID__', '@@@', '%%%'):
        chk(marker not in html, 'no leaked marker %r' % marker)
    return ok[0], msgs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson'); ap.add_argument('--source', required=True)
    a = ap.parse_args()
    meta, _ = v8lib.split_frontmatter(open(a.source, encoding='utf-8').read())
    ok, msgs = run(open(a.lesson, encoding='utf-8').read(), meta)
    for m in msgs: print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL')); sys.exit(0 if ok else 3)

if __name__ == '__main__':
    main()
