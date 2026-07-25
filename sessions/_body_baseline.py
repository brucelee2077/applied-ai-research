#!/usr/bin/env python3
"""Per-day body_engagement + interest-floor snapshot for the 14 m02+m03 days.

The aggregate before-state lives in commit 4758589, but the rebuild needs a
PER-DAY record to enforce the plan's "keep the rebuild only if it is >= the
committed baseline" rule. Writes JSON so before/after are diffable.

Usage: python3 sessions/_body_baseline.py <out.json> [--days d1,d2]
"""
import os, re, sys, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler'))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler', 'gates'))
import coverage_judge as cj

DAYS = sorted(glob.glob(os.path.join(ROOT, 'sessions', 'm02-the-neuron', 'day-*'))) + \
       sorted(glob.glob(os.path.join(ROOT, 'sessions', 'm03-attention', 'day-*')))


def snapshot(day_dir):
    """Grade one day. Returns None when the day has no compiled lesson."""
    lesson_html = os.path.join(day_dir, 'lesson.html')
    source_md = os.path.join(day_dir, 'source.md')
    if not (os.path.exists(lesson_html) and os.path.exists(source_md)):
        return None
    html = open(lesson_html, encoding='utf-8').read()
    src = open(source_md, encoding='utf-8').read()
    titles = re.findall(r'@@@\s+concept\b[^\n]*\btitle="([^"]+)"', src)
    text = cj._readable_text(html)

    body = cj.judge_body_engagement(text, titles)
    interest = cj.judge_interest_absolute(text)

    counts = {'GOOD': 0, 'WEAK': 0, 'MISSING': 0, 'NA': 0}
    per_concept = {}
    for c in body.get('concepts', []):
        grade = str(c.get('body_engagement', '?')).upper()
        counts[grade] = counts.get(grade, 0) + 1
        per_concept[str(c.get('concept', '?'))] = grade

    return {
        'body_status': body.get('status'),
        'body_overall': body.get('overall'),
        'counts': counts,
        'per_concept': per_concept,
        'cold': [k for k, v in per_concept.items() if v in ('MISSING', 'WEAK')],
        'interest_status': interest.get('status'),
        'interest_overall': interest.get('overall'),
        'concept_count': len(titles),
        'source_bytes': os.path.getsize(source_md),
        # count the three chunking widgets the rebuild is supposed to introduce
        'widgets': {
            'steps': len(re.findall(r'(?m)^%%%\s+steps\b', src)),
            'insight': len(re.findall(r'(?m)^%%%\s+insight\b', src)),
            'predict': len(re.findall(r'(?m)^predict:', src)),
        },
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/body_baseline.json'
    only = None
    if '--days' in sys.argv:
        only = set(sys.argv[sys.argv.index('--days') + 1].split(','))

    result = {}
    for day_dir in DAYS:
        name = os.path.basename(day_dir)
        if only and name not in only:
            continue
        snap = snapshot(day_dir)
        if snap is None:
            continue
        snap['module'] = os.path.basename(os.path.dirname(day_dir))
        result[name] = snap
        c = snap['counts']
        print('%-34s body=%-8s G%d W%d M%d NA%d  interest=%-12s steps=%d insight=%d predict=%d %s' % (
            name, snap['body_overall'], c['GOOD'], c['WEAK'], c['MISSING'], c['NA'],
            snap['interest_overall'], snap['widgets']['steps'], snap['widgets']['insight'],
            snap['widgets']['predict'],
            ('| cold: ' + ', '.join(snap['cold'])) if snap['cold'] else ''))

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=1, sort_keys=True)

    tot = {'GOOD': 0, 'WEAK': 0, 'MISSING': 0, 'NA': 0}
    for snap in result.values():
        for k, v in snap['counts'].items():
            tot[k] = tot.get(k, 0) + v
    print('\nTOTAL %d days: %s -> %s' % (len(result), tot, out_path))


if __name__ == '__main__':
    main()
