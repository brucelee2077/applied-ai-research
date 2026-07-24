#!/usr/bin/env python3
"""Scan body_engagement across the 14 m02+m03 target days (before/after the rebuild).
Prints per-day: overall + counts of GOOD/WEAK/MISSING/NA and the cold (MISSING/WEAK) concepts."""
import os, re, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script is in sessions/)
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler'))
sys.path.insert(0, os.path.join(ROOT, 'sessions', '_compiler', 'gates'))
import coverage_judge as cj

DAYS = sorted(glob.glob(os.path.join(ROOT, 'sessions', 'm02-the-neuron', 'day-*')) +
              glob.glob(os.path.join(ROOT, 'sessions', 'm03-attention', 'day-*')))

tot = {'GOOD': 0, 'WEAK': 0, 'MISSING': 0, 'NA': 0}
p0_days = []
for d in DAYS:
    lh, sm = os.path.join(d, 'lesson.html'), os.path.join(d, 'source.md')
    if not (os.path.exists(lh) and os.path.exists(sm)):
        continue
    html = open(lh, encoding='utf-8').read()
    src = open(sm, encoding='utf-8').read()
    titles = re.findall(r'@@@\s+concept\b[^\n]*\btitle="([^"]+)"', src)
    res = cj.judge_body_engagement(cj._readable_text(html), titles)
    cs = res.get('concepts', [])
    counts = {k: 0 for k in tot}
    cold = []
    for c in cs:
        v = str(c.get('body_engagement', '?')).upper()
        if v in counts:
            counts[v] += 1; tot[v] += 1
        if v in ('MISSING', 'WEAK'):
            cold.append('%s[%s]' % (c.get('concept', '?')[:30], v))
    name = os.path.basename(d)
    flag = 'P0' if counts['MISSING'] else ('P1' if counts['WEAK'] else 'ok')
    if counts['MISSING']:
        p0_days.append(name)
    print('%-40s %-8s overall=%-8s G%d W%d M%d NA%d %s' % (
        name, flag, res.get('overall', '?'),
        counts['GOOD'], counts['WEAK'], counts['MISSING'], counts['NA'],
        ('| cold: ' + ', '.join(cold)) if cold else ''))
print('\nTOTAL concepts:', tot, '| P0 days (any MISSING):', p0_days or 'none')
