#!/usr/bin/env python3
"""Assemble the per-day args for the concept-body rebuild (Task 10).

Each day's author gets: its FROZEN front-matter, the visual inventory it must
not lose, and the specific concepts whose build-ups are densest — so the
rebuild is a targeted CHUNKING pass, not a blind re-author.
Writes /tmp/rebuild_args.json.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSITY = os.path.join(ROOT, 'sessions', '_refactor', 'density_before.json')
FROZEN = '/tmp/frozen_fm.json'
OUT = '/tmp/rebuild_args.json'


def widget_count(src, typ):
    return len(re.findall(r'(?m)^%%%\s+' + typ + r'\b', src))


def main():
    density = json.load(open(DENSITY))
    frozen = json.load(open(FROZEN))
    out = {}
    for day, d in density.items():
        src_path = os.path.join(ROOT, 'sessions', d['module'], day, 'source.md')
        src = open(src_path, encoding='utf-8').read()
        dense = sorted(((c['prose_chars'], c['wall'], t)
                        for t, c in d['per_concept'].items()), reverse=True)[:4]
        out[day] = {
            'module': d['module'],
            'frozen': frozen[day]['frozen'],
            'inventory': {k: widget_count(src, k) for k in ('svg', 'viz', 'demo', 'formula', 'table')},
            'concepts': d['concepts'],
            'source_bytes': d['source_bytes'],
            'mean_prose': round(sum(c['prose_chars'] for c in d['per_concept'].values())
                                / max(1, d['concepts'])),
            'densest': [{'concept': t, 'prose_chars': p, 'longest_wall': w} for p, w, t in dense],
        }
    json.dump(out, open(OUT, 'w'), indent=1)
    print('days: %d -> %s' % (len(out), OUT))
    for day in sorted(out):
        o = out[day]
        print('%-30s concepts=%-3d prose/concept=%-5d inv=%s' % (
            day, o['concepts'], o['mean_prose'], o['inventory']))


if __name__ == '__main__':
    main()
