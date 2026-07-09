#!/usr/bin/env python3
# =============================================================================
# Notebook Smoothness Gate  (v8 Phase D)  — PILOT / EXEMPLAR ONLY.
# =============================================================================
# Compares a compiled lesson's FIRST SCREEN (hero) to its notebook yardstick on
# the learning-barrier axis (v8 plan §6): human-first opening, no formula wall,
# a mental picture/curiosity hook — the lesson must read at least as smoothly as
# the notebook's opening.
#
# Notebook is a YARDSTICK, not a dependency: a day whose front-matter has
# notebook_yardstick: null is recorded **N/A (skipped, never failed)** — this is
# the No-Notebook rule applied within a module.
#
# Reusable:  from notebook_smoothness_gate import run ; status, msgs = run(html, meta, root)
# CLI:
#   python3 gates/notebook_smoothness_gate.py <module-dir>     # batch over day-*/
#   python3 gates/notebook_smoothness_gate.py <lesson.html> --source <source.md>
# Exit 0 unless a day with a yardstick FAILS the barrier comparison.
# =============================================================================
import sys, os, re, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CURIOSITY = ['you', 'your', 'imagine', 'picture', 'brain', '?', 'ever ', 'what if',
             'feels like', 'story', 'familiar', 'think about', 'yesterday', 'here is']
# a "formula wall" in the opening = an equation shown before any intuition
FORMULA_SIGNS = ['=', '∑', 'Σ', '∂', 'ŷ', 'w·', '·x', '@ w', 'argmax', 'σ(', '\\sum', 'log(']


def _first_sentence(t):
    return re.split(r'(?<=[.!?])\s', t.strip(), 1)[0] if t.strip() else ''


def _hero_lede(html):
    m = re.search(r'<p class="lede">(.*?)</p>', html, re.DOTALL)
    return re.sub(r'<[^>]+>', ' ', m.group(1)) if m else ''


def _notebook_opening(path):
    """First 1-2 markdown cells' text of a Jupyter notebook (the reader's first screen)."""
    try:
        nb = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        return None, 'could not read notebook: %s' % e
    md = []
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'markdown':
            md.append(''.join(c.get('source', [])))
            if len(md) >= 2:
                break
    return '\n'.join(md), None


def run(html, meta, root=ROOT):
    yard = meta.get('notebook_yardstick')
    msgs = []
    if not yard or str(yard).lower() in ('null', 'none', ''):
        return 'N/A', ['N/A — no notebook yardstick (No-Notebook rule: skipped, never failed)']

    lede = _hero_lede(html)
    low = lede.lower()
    first = _first_sentence(lede).lower()

    frontier = [x for x in v8lib.FRONTIER_TOKENS if x.lower() in first]
    human = [c for c in CURIOSITY if c in low]
    wall = [s for s in FORMULA_SIGNS if s in lede]

    nb_text, err = _notebook_opening(os.path.join(root, yard))
    if err:
        msgs.append('warn ' + err)
        nb_human = None
    else:
        nb_low = nb_text.lower()
        nb_human = [c for c in CURIOSITY if c in nb_low]
        msgs.append('note notebook opens human-first: %s (%s)'
                    % (bool(nb_human), os.path.basename(yard)))

    ok = True
    if frontier:
        ok = False; msgs.append('FAIL hero opens frontier-first (%s) — barrier higher than the notebook' % frontier)
    else:
        msgs.append('pass first sentence not frontier-pressure')
    if human:
        msgs.append('pass hero human/curiosity cues: %s' % human[:4])
    else:
        ok = False; msgs.append('FAIL hero has no human/curiosity cue (notebook opens with one)')
    if wall:
        ok = False; msgs.append('FAIL hero opens on a formula wall (%s) before intuition' % wall)
    else:
        msgs.append('pass no formula wall in the hero')

    verdict = 'PASS' if ok else 'FAIL'
    if nb_human is not None and ok:
        msgs.append('=> compiled first screen is at least as smooth as the notebook')
    return verdict, msgs


def _run_day(day_dir):
    src = os.path.join(day_dir, 'source.md')
    lesson = os.path.join(day_dir, 'lesson.html')
    if not (os.path.exists(src) and os.path.exists(lesson)):
        return None
    meta, _ = v8lib.split_frontmatter(open(src, encoding='utf-8').read())
    html = open(lesson, encoding='utf-8').read()
    return run(html, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target')          # module dir (batch) or a lesson.html
    ap.add_argument('--source')
    args = ap.parse_args()

    if os.path.isdir(args.target):
        days = sorted(d for d in
                      (os.path.join(args.target, x) for x in os.listdir(args.target))
                      if os.path.basename(d).startswith('day-') and os.path.isdir(d))
        print('== Notebook Smoothness Gate (batch):', os.path.relpath(args.target), '==\n')
        n_pass = n_na = n_fail = 0
        any_fail = False
        for d in days:
            res = _run_day(d)
            if res is None:
                print('  --  %-32s (no source.md/lesson.html)' % os.path.basename(d)); continue
            verdict, msgs = res
            if verdict == 'PASS': n_pass += 1
            elif verdict == 'N/A': n_na += 1
            else: n_fail += 1; any_fail = True
            print('  %-4s %s' % (verdict, os.path.basename(d)))
            for m in msgs:
                if m.startswith(('FAIL', 'note', '=>')):
                    print('         ' + m)
        print('\nSUMMARY: PASS=%d  N/A=%d  FAIL=%d  (of %d days)' % (n_pass, n_na, n_fail, len(days)))
        print('Notebook Smoothness Gate: ' + ('FAIL' if any_fail else 'PASS'))
        sys.exit(1 if any_fail else 0)
    else:
        meta = {}
        if args.source:
            meta, _ = v8lib.split_frontmatter(open(args.source, encoding='utf-8').read())
        html = open(args.target, encoding='utf-8').read()
        verdict, msgs = run(html, meta)
        print('== Notebook Smoothness Gate:', os.path.relpath(args.target), '==')
        for m in msgs: print('  ', m)
        print('\n' + verdict)
        sys.exit(1 if verdict == 'FAIL' else 0)


if __name__ == '__main__':
    main()
