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
        sec = re.search(r'<section class="module-section" id="%s".*?</section>' % re.escape(cid), html, re.DOTALL).group(0)
        # a real figure = a genuinely closed <svg> element OR a build-embed wrapper that
        # actually holds an iframe. Bare substrings 'build-embed'/'<svg' in prose don't count.
        has_svg = bool(re.search(r'<svg[\s>].*?</svg>', sec, re.DOTALL))
        has_iframe = bool(re.search(r'class="build-embed"[^>]*>\s*<iframe', sec, re.DOTALL))
        has_visual = has_svg or has_iframe
        chk(has_visual, 'concept %s has a visual' % cid)
        chk(sec.count('class="gotit"') == 1, 'concept %s has exactly one gotit' % cid)

    chk(html.count('data-sec="quiz"') == 1, 'exactly one quiz section')
    # Count questions in the QUIZ section only. A %%% warmup recall block also emits
    # class="q" blocks (inside a .warmup wrapper), so a whole-page count would over-count;
    # scope to the quiz section to keep the "exactly 4 quiz questions" invariant honest.
    qsec = re.search(r'data-sec="quiz".*?</section>', html, re.DOTALL)
    nq = qsec.group(0).count('class="q"') if qsec else html.count('class="q"')
    chk(nq == 4, 'quiz has 4 questions (got %d)' % nq)
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
    # `~~~` is in this list because it is a BLOCK FENCE the reader must never see.
    # render_md consumes `~~~html` and `~~~zh`, but a fence whose terminator the
    # author forgot (or misspelled) falls through to the paragraph branch and ships
    # as literal text: `render_md("English.\n\n~~~zh\n中文。\n~~~")` emits
    # `<p>~~~zh 中文。</p><p>~~~</p>`. Verified zero of the 47 shipped lessons
    # contain `~~~`, so this can only catch a real leak.
    for marker in ('<!--V9_CONTENT-->', '<!--V9_NAV-->', '__QUEST_ID__', '@@@', '%%%', '~~~'):
        chk(marker not in html, 'no leaked marker %r' % marker)
    # A glossary tooltip is PLAIN TEXT. An `<` or `>` inside data-tip means an inline
    # rule leaked a tag into the attribute (v8lib.inline used to substitute glosses
    # before the ** / * rules ran, so emphasis inside a gloss injected a literal <em>,
    # and two glosses on one line could cross-pair an <em>…</em> across two attributes).
    # Cost when unchecked: the reader hovering "gradient" saw `steepest <em>increase</em>`,
    # and coverage_judge._readable_text desynced on the stray `>` so every LLM judge
    # graded mangled prose and passed it. 3 shipped m04 lessons, 4 tooltips.
    bad_tips = [t for t in re.findall(r'data-tip="([^"]*)"', html) if '<' in t or '>' in t]
    chk(not bad_tips, 'no tag leaked into a data-tip tooltip%s'
        % ('' if not bad_tips else ' (%d bad, e.g. %r)' % (len(bad_tips), bad_tips[0][:60])))
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
