#!/usr/bin/env python3
# =============================================================================
# Evidence Index (Plan 2)  — deterministic portfolio landing page.
# =============================================================================
# Scans every portfolio/<module>/<day>/meta.json written by evidence_compile and
# renders portfolio/index.html: one card per day (title, module, a relative link
# to that day's index.html, and small badges for has_experiment / has_plot /
# "<N> viz"). Deterministic + idempotent: same meta.json set -> same index bytes.
#
# CLI:
#   python3 evidence_index.py
# =============================================================================
import sys, os, glob, json, html as _htmlmod, argparse

# repo root = two dirs up from this file (sessions/_compiler/evidence_index.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_metas(root):
    """Return metas sorted deterministically by (module, day)."""
    metas = []
    pattern = os.path.join(root, 'portfolio', '*', '*', 'meta.json')
    for mpath in glob.glob(pattern):
        try:
            m = json.load(open(mpath, encoding='utf-8'))
        except Exception:
            continue
        # derive module/day from the path if absent, so links are always right
        day = os.path.basename(os.path.dirname(mpath))
        module = os.path.basename(os.path.dirname(os.path.dirname(mpath)))
        m.setdefault('module', module)
        m.setdefault('day', day)
        m['_module'], m['_day'] = module, day
        metas.append(m)
    metas.sort(key=lambda m: (m['_module'], m['_day']))
    return metas


def _card(m):
    module, day = m['_module'], m['_day']
    title = m.get('title') or day
    link = '%s/%s/index.html' % (module, day)                    # RELATIVE
    badges = []
    if m.get('has_experiment'):
        badges.append('<span class="badge exp">experiment</span>')
    if m.get('has_plot'):
        badges.append('<span class="badge plot">plot</span>')
    nviz = len(m.get('viz') or [])
    if nviz:
        badges.append('<span class="badge viz">%d viz</span>' % nviz)
    return (
        '<a class="card" href="%s">' % _htmlmod.escape(link)
        + '<div class="mod">%s</div>' % _htmlmod.escape(module)
        + '<div class="title">%s</div>' % _htmlmod.escape(title)
        + '<div class="badges">%s</div>' % ''.join(badges)
        + '</a>'
    )


def build_index(root):
    """Render portfolio/index.html from all meta.json files. Returns the count
    of days indexed. Deterministic + idempotent."""
    pdir = os.path.join(root, 'portfolio')
    os.makedirs(pdir, exist_ok=True)
    metas = _load_metas(root)
    cards = '\n'.join(_card(m) for m in metas) or '<p><em>No portfolio days yet.</em></p>'
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Portfolio</title>\n'
        '<style>body{max-width:960px;margin:2rem auto;padding:0 1rem;'
        'font-family:system-ui,-apple-system,sans-serif;color:#1c2530}'
        'h1{margin-bottom:1.5rem}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}'
        '.card{display:block;text-decoration:none;color:inherit;border:1px solid #d9e0e8;'
        'border-radius:10px;padding:1rem;background:#fff;transition:box-shadow .15s}'
        '.card:hover{box-shadow:0 4px 16px rgba(0,0,0,.08)}'
        '.mod{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#64748b}'
        '.title{font-weight:600;margin:.35rem 0 .6rem}'
        '.badges{display:flex;flex-wrap:wrap;gap:.35rem}'
        '.badge{font-size:.7rem;padding:.15em .5em;border-radius:999px;background:#eef2f7;color:#334155}'
        '.badge.exp{background:#dcfce7;color:#166534}'
        '.badge.plot{background:#dbeafe;color:#1e40af}'
        '.badge.viz{background:#fef3c7;color:#92400e}'
        '</style></head><body>\n'
        '<h1>Portfolio &mdash; %d %s</h1>\n' % (len(metas), 'day' if len(metas) == 1 else 'days')
        + '<div class="grid">\n' + cards + '\n</div>\n'
        + '</body></html>\n'
    )
    open(os.path.join(pdir, 'index.html'), 'w', encoding='utf-8').write(html)
    return len(metas)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    argparse.ArgumentParser().parse_args()
    n = build_index(ROOT)
    print('indexed %d portfolio day%s -> %s'
          % (n, '' if n == 1 else 's', os.path.relpath(os.path.join(ROOT, 'portfolio', 'index.html'), ROOT)))
    sys.exit(0)


if __name__ == '__main__':
    main()
