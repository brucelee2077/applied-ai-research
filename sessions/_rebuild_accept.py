#!/usr/bin/env python3
"""Per-day ACCEPT GATE for the concept-body rebuild (Task 10, step 3).

A full blind re-author regenerates a whole lesson. The existing 14 days already
clear the body + interest floors and carry hard-won visual density (up to 23
SVGs on one day), so a rebuild is only worth keeping if it improved
DIGESTIBILITY without losing anything else.

Compares the WORKING TREE against the day's committed version at a git ref:

  hard checks (any failure => REVERT)
    - compile_lesson.py exits 0
    - concept_structure_gate.py exits 0
    - recompile is idempotent (lesson.html byte-identical on a second compile)
    - front-matter is byte-identical to the ref (frozen invariants held)
    - body_engagement: no MISSING, and GOOD count >= ref GOOD count
    - interest floor: FLOOR_MET
    - no coverage loss: concept count and coverage_topics preserved
    - no visual loss: svg + viz + demo count >= ref
  purpose check (failure => REVERT: the rebuild did not do its job)
    - chunking widgets actually used (steps + insight + predict > 0)
    - build-up prose density not worse than ref

Usage:
  python3 sessions/_rebuild_accept.py <module>/<day> [--ref HEAD] [--json out.json]

Exit 0 = KEEP, 1 = REVERT, 2 = harness error.
"""
import os, re, sys, json, glob, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler'))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler', 'gates'))
import coverage_judge as cj  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, 'sessions'))
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location('_density_scan', os.path.join(ROOT, 'sessions', '_density_scan.py'))
density = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(density)


def sh(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def git_show(ref, relpath):
    """File contents at a git ref, or None when absent there."""
    code, out = sh(['git', 'show', '%s:%s' % (ref, relpath)])
    return out if code == 0 else None


def front_matter(src):
    if not src.startswith('---\n'):
        return None
    end = src.find('\n---\n', 3)
    return src[4:end + 1] if end != -1 else None


# The FROZEN invariants are the WIRING keys: progress tracking, the prev/next
# chain, the sidebar group, the narrative-spine gate, the notebook oracle. Prose
# keys (fin_title/fin_body/title/subtitle) are NOT frozen — the author owns the
# voice, and a rebuild must be allowed to correct a factual error that is baked
# into that prose (day-05's fin_body claimed the gradient points DOWNhill).
FROZEN_KEYS = ('quest_id', 'mode', 'donor', 'nav_prev_href', 'nav_prev_label',
               'nav_next_href', 'nav_next_label', 'module_label', 'page_title',
               'brand_sub', 'spine', 'notebook_yardstick', 'require_artifact')


def fm_keys(src):
    """Top-level `key: value` pairs of the front-matter, as a dict."""
    fm = front_matter(src) or ''
    out = {}
    for line in fm.splitlines():
        m = re.match(r'([a-z_]+):(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def counts(src):
    """Structural inventory of a source.md — what a rebuild must not lose."""
    return {
        'concepts': len(re.findall(r'(?m)^@@@\s+concept\b', src)),
        'svg': len(re.findall(r'(?m)^%%%\s+svg\b', src)),
        'viz': len(re.findall(r'(?m)^%%%\s+viz\b', src)),
        'demo': len(re.findall(r'(?m)^%%%\s+demo\b', src)),
        'steps': len(re.findall(r'(?m)^%%%\s+steps\b', src)),
        'insight': len(re.findall(r'(?m)^%%%\s+insight\b', src)),
        'predict': len(re.findall(r'(?m)^predict:', src)),
        'coverage_topics': len(re.findall(r'(?m)^\s+-\s', front_matter(src) or '')),
    }


def grade(lesson_html_text, src):
    titles = re.findall(r'@@@\s+concept\b[^\n]*\btitle="([^"]+)"', src)
    text = cj._readable_text(lesson_html_text)
    body = cj.judge_body_engagement(text, titles)
    interest = cj.judge_interest_absolute(text)
    per = {}
    for c in body.get('concepts', []):
        per[str(c.get('concept', '?'))] = str(c.get('body_engagement', '?')).upper()
    tally = {g: sum(1 for v in per.values() if v == g)
             for g in ('GOOD', 'WEAK', 'MISSING', 'NA')}
    return {
        'body_status': body.get('status'), 'body_overall': body.get('overall'),
        'per_concept': per, 'tally': tally,
        'interest_status': interest.get('status'),
        'interest_overall': interest.get('overall'),
    }


def density_of(src):
    walls = []
    prose = []
    for _title, body in density.concept_blocks(src):
        buildup = density.buildup_of(body)
        walls.append(density.longest_wall(buildup))
        prose.append(len(density.WIDGET_BLOCK_RE.sub('\n\n', buildup).strip()))
    n = max(1, len(walls))
    return {
        'max_wall': max(walls) if walls else 0,
        'mean_wall': round(sum(walls) / n),
        'walls_over_600': sum(1 for w in walls if w > 600),
        'mean_prose': round(sum(prose) / n),
    }


def check(day_rel, ref='HEAD'):
    day_dir = os.path.join(ROOT, 'sessions', day_rel)
    src_rel = 'sessions/%s/source.md' % day_rel
    lesson_rel = 'sessions/%s/lesson.html' % day_rel
    src_path = os.path.join(ROOT, src_rel)
    lesson_path = os.path.join(ROOT, lesson_rel)

    if not os.path.exists(src_path):
        return {'day': day_rel, 'verdict': 'ERROR', 'fail': ['source.md missing']}

    new_src = open(src_path, encoding='utf-8').read()
    old_src = git_show(ref, src_rel)
    if old_src is None:
        return {'day': day_rel, 'verdict': 'ERROR',
                'fail': ['no committed version at %s' % ref]}
    old_lesson = git_show(ref, lesson_rel)

    fails, warns = [], []

    # --- hard gates -----------------------------------------------------
    code_c, out_c = sh(['python3', 'sessions/_compiler/compile_lesson.py', src_rel])
    if code_c != 0:
        fails.append('compile_lesson.py exit %d: %s' % (code_c, out_c.strip()[-400:]))
    code_g, out_g = sh(['python3', 'sessions/_compiler/gates/concept_structure_gate.py', src_rel])
    if code_g != 0:
        fails.append('concept_structure_gate exit %d: %s' % (code_g, out_g.strip()[-400:]))

    # idempotent recompile: compiling twice must produce identical bytes
    after_first = open(lesson_path, encoding='utf-8').read() if os.path.exists(lesson_path) else ''
    sh(['python3', 'sessions/_compiler/compile_lesson.py', src_rel])
    after_second = open(lesson_path, encoding='utf-8').read() if os.path.exists(lesson_path) else ''
    if after_first != after_second:
        fails.append('recompile is not idempotent')

    # frozen WIRING keys must survive verbatim; prose keys may legitimately change
    new_fm, old_fm = fm_keys(new_src), fm_keys(old_src)
    for key in FROZEN_KEYS:
        if old_fm.get(key) != new_fm.get(key):
            fails.append('frozen front-matter key %r changed: %r -> %r'
                         % (key, old_fm.get(key), new_fm.get(key)))
    for key in ('fin_title', 'fin_body', 'title', 'subtitle'):
        if key in old_fm and old_fm.get(key) != new_fm.get(key):
            warns.append('prose front-matter key %r rewritten (allowed — review it)' % key)

    new_counts, old_counts = counts(new_src), counts(old_src)
    for key, label in (('concepts', 'concept'), ('coverage_topics', 'coverage-topic')):
        if new_counts[key] < old_counts[key]:
            fails.append('%s count dropped %d -> %d' % (label, old_counts[key], new_counts[key]))
    new_vis = new_counts['svg'] + new_counts['viz'] + new_counts['demo']
    old_vis = old_counts['svg'] + old_counts['viz'] + old_counts['demo']
    if new_vis < old_vis:
        fails.append('visual count dropped %d -> %d (svg+viz+demo)' % (old_vis, new_vis))

    # --- purpose: did the rebuild actually CHUNK? ------------------------
    chunking = new_counts['steps'] + new_counts['insight'] + new_counts['predict']
    if chunking == 0:
        fails.append('no chunking widgets used (steps/insight/predict all 0) — '
                     'the rebuild did not do its job')

    new_den, old_den = density_of(new_src), density_of(old_src)
    if new_den['mean_prose'] > old_den['mean_prose'] * 1.15:
        warns.append('build-up prose grew %d -> %d chars/concept'
                     % (old_den['mean_prose'], new_den['mean_prose']))
    if new_den['walls_over_600'] > old_den['walls_over_600']:
        warns.append('walls>600 grew %d -> %d'
                     % (old_den['walls_over_600'], new_den['walls_over_600']))

    # --- LLM floors ------------------------------------------------------
    new_grade = grade(after_second, new_src) if after_second else None
    old_grade = grade(old_lesson, old_src) if old_lesson else None
    if new_grade is None:
        fails.append('no compiled lesson to grade')
    else:
        if new_grade['tally']['MISSING'] > 0:
            fails.append('body_engagement MISSING on %d concept(s): %s'
                         % (new_grade['tally']['MISSING'],
                            [k for k, v in new_grade['per_concept'].items() if v == 'MISSING']))
        if new_grade['interest_overall'] != 'FLOOR_MET':
            fails.append('interest floor: %s' % new_grade['interest_overall'])
        if old_grade and new_grade['tally']['GOOD'] < old_grade['tally']['GOOD']:
            fails.append('body GOOD count dropped %d -> %d'
                         % (old_grade['tally']['GOOD'], new_grade['tally']['GOOD']))

    return {
        'day': day_rel, 'ref': ref,
        'verdict': 'KEEP' if not fails else 'REVERT',
        'fail': fails, 'warn': warns,
        'before': {'counts': old_counts, 'density': old_den,
                   'grade': old_grade},
        'after': {'counts': new_counts, 'density': new_den,
                  'grade': new_grade},
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    day_rel = sys.argv[1].strip('/')
    ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else 'HEAD'
    out_json = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None

    res = check(day_rel, ref)
    print('%s: %s' % (res['day'], res['verdict']))
    for f in res['fail']:
        print('  FAIL  %s' % f)
    for w in res['warn']:
        print('  warn  %s' % w)
    if res.get('after') and res['after'].get('grade'):
        a, b = res['after'], res['before']
        print('  body  %s -> %s' % (b['grade']['tally'] if b.get('grade') else '?', a['grade']['tally']))
        print('  chunk steps=%d insight=%d predict=%d | prose %d -> %d chars/concept' % (
            a['counts']['steps'], a['counts']['insight'], a['counts']['predict'],
            b['density']['mean_prose'], a['density']['mean_prose']))
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)
    return 0 if res['verdict'] == 'KEEP' else (2 if res['verdict'] == 'ERROR' else 1)


if __name__ == '__main__':
    sys.exit(main())
