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
# a CALLOUT block (AUTHORING.md §5): "!!! <class> <emoji>" … "!!!". The compiler
# treats "!!! " as a block opener (v8lib.is_special) and renders a set-off
# <div class="callout">, and concept rule 4 REQUIRES heavy math to be demoted
# into one marked "Optional (skippable)". So its body is not main-line prose —
# it is an ASIDE, measured separately by aside_wall() rather than erased.
# NOTE "[ \t]+" not "\s+": \s would match the newline after a CLOSING "!!!" and
# swallow the following line as if it opened a box.
CALLOUT_BLOCK_RE = re.compile(r'(?m)^!!![ \t]+\S+.*?(?:\n!!!\s*$|\Z)', re.DOTALL)
# anything that gives the eye a place to rest / breaks a wall into one-idea chunks
BREAK_RE = re.compile(r'(?m)^(?:####+\s|%%%\s+\w|!!!\s|step:|why:|[-*]\s|\d+\.\s|>\s)')


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


def _strip_markup(text):
    """Drop everything that is not main-line prose: widget bodies, raw-HTML
    escapes, and callout boxes. Each is replaced by a blank line so the prose
    on either side does not fuse into one apparent wall."""
    out = WIDGET_BLOCK_RE.sub('\n\n', text)
    out = RAW_HTML_RE.sub('\n\n', out)
    return CALLOUT_BLOCK_RE.sub('\n\n', out)


def _worst_run(prose):
    """Longest paragraph with no chunk boundary and no blank line."""
    worst = 0
    for para in re.split(r'\n\s*\n', prose):
        stripped = para.strip()
        if not stripped or BREAK_RE.match(stripped):
            continue
        # a paragraph that is one long block of prose IS the wall
        worst = max(worst, len(stripped))
    return worst


def longest_wall(text):
    """Longest run of MAIN-LINE prose (chars) with no chunk boundary.

    Widget bodies (raw SVG, demo key:value lines, steps rungs) are stripped
    first — a 4k-char inline <svg> is not a wall of text for the reader, it is
    a picture. Callout boxes are stripped too and measured by aside_wall().
    """
    return _worst_run(_strip_markup(text))


def aside_wall(text):
    """Longest unbroken run INSIDE callout boxes ("Optional (skippable)" asides).

    A boxed aside off the critical path is a milder problem than a main-line
    wall, and it has a cheaper fix (break it internally with <br> and bold
    sub-lead-ins) than re-authoring a concept. Kept as its own number so long
    asides stay visible instead of being hidden by the strip above.
    """
    worst = 0
    for block in CALLOUT_BLOCK_RE.findall(text):
        body = re.sub(r'(?m)^!!!.*$', '', block)              # drop both fences
        body = RAW_HTML_RE.sub('\n\n', WIDGET_BLOCK_RE.sub('\n\n', body))
        for run in re.split(r'<br\s*/?>|\n\s*\n', body):      # <br> is a real line break
            worst = max(worst, len(run.strip()))
    return worst


def scan_day(day_dir):
    source_md = os.path.join(day_dir, 'source.md')
    if not os.path.exists(source_md):
        return None
    src = open(source_md, encoding='utf-8').read()
    concepts = concept_blocks(src)

    walls, asides, per_concept = [], [], {}
    for title, body in concepts:
        buildup = buildup_of(body)
        wall = longest_wall(buildup)
        aside = aside_wall(buildup)
        breaks = len(BREAK_RE.findall(buildup))
        prose_chars = len(RAW_HTML_RE.sub('\n\n', WIDGET_BLOCK_RE.sub('\n\n', buildup)).strip())
        walls.append(wall)
        asides.append(aside)
        per_concept[title] = {'wall': wall, 'aside_wall': aside,
                              'breaks': breaks, 'prose_chars': prose_chars}

    return {
        'module': os.path.basename(os.path.dirname(day_dir.rstrip('/'))),
        'concepts': len(concepts),
        'max_wall': max(walls) if walls else 0,
        'mean_wall': round(sum(walls) / len(walls)) if walls else 0,
        'walls_over_600': sum(1 for w in walls if w > 600),
        'max_aside_wall': max(asides) if asides else 0,
        'asides_over_600': sum(1 for a in asides if a > 600),
        'widgets': {
            'steps': len(re.findall(r'(?m)^%%%\s+steps\b', src)),
            'insight': len(re.findall(r'(?m)^%%%\s+insight\b', src)),
            'predict': len(re.findall(r'(?m)^predict:', src)),
        },
        'source_bytes': os.path.getsize(source_md),
        'per_concept': per_concept,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/density.json'
    result = {}
    for day_dir in DAYS:
        snap = scan_day(day_dir)
        if snap is None:
            continue
        name = os.path.basename(day_dir)
        result[name] = snap
        w = snap['widgets']
        print('%-34s concepts=%-3d max_wall=%-5d mean_wall=%-4d walls>600=%-2d aside>600=%-2d max_aside=%-5d steps=%d insight=%d predict=%d' % (
            name, snap['concepts'], snap['max_wall'], snap['mean_wall'],
            snap['walls_over_600'], snap['asides_over_600'], snap['max_aside_wall'],
            w['steps'], w['insight'], w['predict']))

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=1, sort_keys=True)

    tot_over = sum(s['walls_over_600'] for s in result.values())
    tot_aside = sum(s['asides_over_600'] for s in result.values())
    worst = max((s['max_wall'] for s in result.values()), default=0)
    worst_aside = max((s['max_aside_wall'] for s in result.values()), default=0)
    print('\n%d days | MAIN walls>600=%d (worst %d) | ASIDE walls>600=%d (worst %d) -> %s' % (
        len(result), tot_over, worst, tot_aside, worst_aside, out_path))


if __name__ == '__main__':
    main()
