#!/usr/bin/env python3
# =============================================================================
# extract_source.py  (v8 Phase D)
# =============================================================================
# Turn an already-shipped lesson.html into a canonical source.md in VERBATIM
# (migration) mode: every reader-flow region is captured byte-for-byte using the
# SAME regexes the compiler uses (v8lib.REGION_PATTERNS), so recompiling the
# emitted source.md reproduces the donor byte-identically (zero regression).
#
# This "sources" an existing good lesson onto the v8 pipeline with no content
# risk. Clean typed-block conversion (like Day 1) is later, per-day work.
#
# Usage:
#   python3 sessions/_compiler/extract_source.py <lesson.html> \
#       --donor <donor-filename> [--out <source.md>] [--spine "..."] [--yardstick "..."]
# =============================================================================
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8lib


def extract(html):
    qm = re.search(r'data-quest-id="([^"]+)"', html)
    if not qm:
        raise RuntimeError("no data-quest-id in lesson")
    tm = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
    regions = {}
    missing = []
    for name, pat in v8lib.REGION_PATTERNS.items():
        m = re.search(pat, html, re.DOTALL)
        if not m:
            missing.append(name)
        else:
            regions[name] = m.group(0)
    if missing:
        raise RuntimeError("regions not found (donor not v8-shaped?): %s" % missing)
    return qm.group(1), (tm.group(1) if tm else ''), regions


def build_source(qid, page_title, regions, donor, spine, yardstick):
    fm = ['---',
          'quest_id: %s' % qid,
          'donor: %s' % donor,
          'mode: exemplar',
          'source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson',
          'page_title: %s' % _yaml_str(page_title)]
    if spine:
        fm.append('spine: %s' % _yaml_str(spine))
    if yardstick:
        fm.append('notebook_yardstick: %s' % yardstick)
    else:
        fm.append('notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)')
    fm.append('---')
    body = ['']
    # order regions for readability (reader-flow order first, then shell/data)
    order = ['title', 'brand_sub', 'sidebar_nav', 'nav_prev', 'nav_next',
             'hero', 's1', 's2', 's4', 's7', 'fin', 'DEMOS', 'BUILD', 'QS']
    for name in order:
        body.append('@@@ region name=%s' % name)
        body.append(regions[name])
    return '\n'.join(fm) + '\n' + '\n'.join(body) + '\n'


def _yaml_str(s):
    # quote to be safe (titles contain '·', ':' etc.)
    return '"%s"' % s.replace('\\', '\\\\').replace('"', '\\"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson')
    ap.add_argument('--donor', required=True)
    ap.add_argument('--out')
    ap.add_argument('--spine', default='')
    ap.add_argument('--yardstick', default='')
    args = ap.parse_args()
    html = open(args.lesson, encoding='utf-8').read()
    qid, page_title, regions = extract(html)
    src = build_source(qid, page_title, regions, args.donor, args.spine, args.yardstick)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.lesson)), 'source.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(src)
    print('extracted', qid, '->', os.path.relpath(out), '(%d regions, %d chars)' % (len(regions), len(src)))


if __name__ == '__main__':
    main()
