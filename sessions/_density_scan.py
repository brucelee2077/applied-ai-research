#!/usr/bin/env python3
"""Deterministic DENSITY scan of concept build-ups (no LLM, no bridge).

The body_engagement judge grades build-up VOICE. It is blind to DENSITY — and
"hard to digest" (user, 2026-07-24) is a density complaint, not a voice one.
This measures the thing the judge cannot see, so a rebuild can be shown to have
actually CHUNKED the build-ups rather than just re-worded them.

Per concept build-up (everything after the concept's first visual widget):
  wall      — longest unbroken prose run in characters (the "wall of text")
  chunks    — number of chunk boundaries (####, %%% steps rungs, callouts, lists)
  widgets   — chunking widgets used (steps / insight / predict)

Usage: python3 sessions/_density_scan.py <out.json>
"""
import os, re, sys, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAYS = sorted(glob.glob(os.path.join(ROOT, 'sessions', 'm02-the-neuron', 'day-*'))) + \
       sorted(glob.glob(os.path.join(ROOT, 'sessions', 'm03-attention', 'day-*')))

# a widget block opens with "%%% <type>" at line start and closes with a line that
# is exactly "%%%" (AUTHORING.md §4).
WIDGET_RE = re.compile(r'(?m)^%%%\s+([a-z]+)\b')
# a whole widget block, opening line through its closing "%%%" fence
WIDGET_BLOCK_RE = re.compile(r'(?m)^%%%\s+[a-z]+\b.*?(?:\n%%%\s*$|\Z)', re.DOTALL)
# a raw-HTML escape block (AUTHORING.md §"Raw HTML escape"): ~~~html … ~~~.
# Markup, not prose — strip it for the same reason widget bodies are stripped.
RAW_HTML_RE = re.compile(r'(?m)^~~~\w*\b.*?(?:\n~~~\s*$|\Z)', re.DOTALL)
# anything that gives the eye a place to rest / breaks a wall into one-idea chunks
BREAK_RE = re.compile(r'(?m)^(?:####+\s|%%%\s+\w|step:|why:|[-*]\s|\d+\.\s|>\s)')


def concept_blocks(src):
    """Split source.md into (title, body) per @@@ concept block."""
    parts = re.split(r'(?m)^@@@\s+', src)
    out = []
    for part in parts:
        if not part.startswith('concept'):
            continue
        head, _, body = part.partition('\n')
        title = re.search(r'title="([^"]+)"', head)
        out.append((title.group(1) if title else '?', body))
    return out


def buildup_of(body):
    """The build-up = everything after the concept's FIRST visual widget BLOCK.

    That first widget is the opening anchor (AUTHORING.md rule: intro -> own
    inline visual -> build-up), so the text after it is the build-up region.
    Slice past the widget's CLOSING "%%%" fence, not just its opening line, or
    the widget's body (e.g. raw SVG) leaks into the prose measurement.
    """
    first = WIDGET_BLOCK_RE.search(body)
    return body[first.end():] if first else body


def longest_wall(text):
    """Longest run of PROSE (chars) with no chunk boundary and no blank line.

    Widget bodies (raw SVG, demo key:value lines, steps rungs) are stripped first —
    a 4k-char inline <svg> is not a wall of text for the reader, it is a picture.
    """
    prose = RAW_HTML_RE.sub('\n\n', WIDGET_BLOCK_RE.sub('\n\n', text))
    worst = 0
    for para in re.split(r'\n\s*\n', prose):
        stripped = para.strip()
        if not stripped or BREAK_RE.match(stripped):
            continue
        # a paragraph that is one long block of prose IS the wall
        worst = max(worst, len(stripped))
    return worst


def scan_day(day_dir):
    source_md = os.path.join(day_dir, 'source.md')
    if not os.path.exists(source_md):
        return None
    src = open(source_md, encoding='utf-8').read()
    concepts = concept_blocks(src)

    walls, per_concept = [], {}
    for title, body in concepts:
        buildup = buildup_of(body)
        wall = longest_wall(buildup)
        breaks = len(BREAK_RE.findall(buildup))
        prose_chars = len(RAW_HTML_RE.sub('\n\n', WIDGET_BLOCK_RE.sub('\n\n', buildup)).strip())
        walls.append(wall)
        per_concept[title] = {'wall': wall, 'breaks': breaks, 'prose_chars': prose_chars}

    return {
        'module': os.path.basename(os.path.dirname(day_dir.rstrip('/'))),
        'concepts': len(concepts),
        'max_wall': max(walls) if walls else 0,
        'mean_wall': round(sum(walls) / len(walls)) if walls else 0,
        'walls_over_600': sum(1 for w in walls if w > 600),
        'widgets': {
            'steps': len(re.findall(r'(?m)^%%%\s+steps\b', src)),
            'insight': len(re.findall(r'(?m)^%%%\s+insight\b', src)),
            'predict': len(re.findall(r'(?m)^predict:', src)),
        },
        'source_bytes': os.path.getsize(source_md),
        'per_concept': per_concept,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/density.json'
    result = {}
    for day_dir in DAYS:
        snap = scan_day(day_dir)
        if snap is None:
            continue
        name = os.path.basename(day_dir)
        result[name] = snap
        w = snap['widgets']
        print('%-34s concepts=%-3d max_wall=%-5d mean_wall=%-4d walls>600=%-2d  steps=%d insight=%d predict=%d' % (
            name, snap['concepts'], snap['max_wall'], snap['mean_wall'],
            snap['walls_over_600'], w['steps'], w['insight'], w['predict']))

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=1, sort_keys=True)

    tot_over = sum(s['walls_over_600'] for s in result.values())
    worst = max((s['max_wall'] for s in result.values()), default=0)
    print('\n%d days | walls>600 total=%d | worst wall=%d -> %s' % (
        len(result), tot_over, worst, out_path))


if __name__ == '__main__':
    main()
