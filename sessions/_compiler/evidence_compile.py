#!/usr/bin/env python3
# =============================================================================
# Evidence Compile (Plan 2)  — deterministic portfolio-day assembler.
# =============================================================================
# Turns a portfolio "day" (blog.md + experiment.py + experiment_out.txt, plus any
# plots the experiment wrote and any viz the lesson referenced) into ONE self-
# contained, publishable page:
#
#   portfolio/<module>/<day>/
#     blog.md              (input, authored)
#     experiment.py        (input, runnable backing)
#     experiment_out.txt   (input, captured stdout)
#     assets/*.png         (input, written by experiment.py) + copied viz
#     index.html           (OUTPUT — self-contained: blog + experiment + assets)
#     meta.json            (OUTPUT — index metadata for the portfolio index)
#
# EVERY asset link in index.html is relative ("assets/..."). It NEVER points at
# "../sessions/" or an absolute path, so the whole portfolio/ tree can be copied
# and served anywhere. Viz pages referenced by the lesson source are COPIED into
# assets/ (not linked across the tree).
#
# Deterministic + idempotent: same inputs -> same index.html/meta.json bytes.
#
# CLI:
#   python3 evidence_compile.py <module> <day>
# =============================================================================
import sys, os, re, json, html as _htmlmod, argparse

# repo root = two dirs up from this file (sessions/_compiler/evidence_compile.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VIZ_RE = re.compile(r'^%%%\s+viz\b.*?\bsrc=(?:"([^"]+)"|(\S+))', re.MULTILINE)


def viz_refs(source_md_text):
    """Return the list of `src` paths from `%%% viz ... src=...` lines in a lesson
    source. Handles both quoted (src="x") and bare (src=x) forms. [] if none."""
    if not source_md_text:
        return []
    refs = []
    for m in _VIZ_RE.finditer(source_md_text):
        refs.append(m.group(1) if m.group(1) is not None else m.group(2))
    return refs


# ---------------------------------------------------------------------------
# Tiny hand-rolled markdown -> HTML (no new dependency). Supports:
#   # .. ######  headings,  ``` fenced code,  blank-line paragraphs,
#   - bullet lists,  **bold**,  `code`.  Text nodes are HTML-escaped.
# ---------------------------------------------------------------------------
def _inline(text):
    """Escape a text run, then apply **bold** and `code` inline spans."""
    text = _htmlmod.escape(text)
    # `code` first so ** inside code is left literal
    text = re.sub(r'`([^`]+)`', lambda m: '<code>%s</code>' % m.group(1), text)
    text = re.sub(r'\*\*([^*]+)\*\*', lambda m: '<strong>%s</strong>' % m.group(1), text)
    return text


def _md_to_html(md):
    lines = (md or '').split('\n')
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):                       # fenced code block
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith('```'):
                buf.append(lines[i])
                i += 1
            i += 1                                        # skip closing fence
            out.append('<pre><code>%s</code></pre>' % _htmlmod.escape('\n'.join(buf)))
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', line)          # heading
        if m:
            lvl = len(m.group(1))
            out.append('<h%d>%s</h%d>' % (lvl, _inline(m.group(2).strip()), lvl))
            i += 1
            continue
        if re.match(r'^\s*-\s+', line):                   # bullet list
            items = []
            while i < len(lines) and re.match(r'^\s*-\s+', lines[i]):
                items.append('<li>%s</li>' % _inline(re.sub(r'^\s*-\s+', '', lines[i])))
                i += 1
            out.append('<ul>%s</ul>' % ''.join(items))
            continue
        if line.strip() == '':                            # blank -> paragraph break
            i += 1
            continue
        para = []                                         # paragraph: gather until blank/block
        while i < len(lines) and lines[i].strip() != '' \
                and not lines[i].startswith('```') \
                and not re.match(r'^#{1,6}\s', lines[i]) \
                and not re.match(r'^\s*-\s+', lines[i]):
            para.append(lines[i])
            i += 1
        out.append('<p>%s</p>' % _inline(' '.join(s.strip() for s in para)))
    return '\n'.join(out)


def _read(path):
    try:
        return open(path, encoding='utf-8').read()
    except Exception:
        return ''


# A quoted href whose value starts with `../` — a parent-escaping NAV back-link
# (e.g. href="../index.html"). Copied viz carry these back to their old lesson
# location; once relocated into portfolio/<m>/<d>/assets/ they are dead. Only
# NAV (href=) links that escape the parent are neutralized. External links
# (http/https//), same-dir links, and ALL src= attributes are left untouched
# (viz srcs are inline/same-dir; rewriting them would break the visual).
_PARENT_HREF_RE = re.compile(r'''href\s*=\s*(["'])\.\.\/[^"']*\1''', re.IGNORECASE)


def _sanitize_copied_viz(html):
    """Neutralize parent-escaping NAV back-links in a copied viz's HTML: any
    quoted href whose value starts with `../` becomes href="#". Leaves external
    (http/https/protocol-relative) links, same-dir links, and every src=
    attribute untouched. Returns the sanitized HTML."""
    return _PARENT_HREF_RE.sub('href="#"', html or '')


def _title_from_blog(blog_md, day_slug):
    """Blog's first `# ` heading, else the day slug."""
    for line in (blog_md or '').split('\n'):
        m = re.match(r'^#\s+(.*)$', line)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return day_slug


def assemble(module, day, root):
    """Assemble portfolio/<module>/<day>/index.html + meta.json from the day's
    inputs. Copies referenced viz + any experiment plots into assets/ with
    RELATIVE links only. Returns the index.html path."""
    pdir = os.path.join(root, 'portfolio', module, day)
    os.makedirs(pdir, exist_ok=True)
    blog_md = _read(os.path.join(pdir, 'blog.md'))
    exp_code = _read(os.path.join(pdir, 'experiment.py'))
    exp_out = _read(os.path.join(pdir, 'experiment_out.txt'))

    # lesson source (if present) supplies viz references
    src_dir = os.path.join(root, 'sessions', module, day)
    source_md = _read(os.path.join(src_dir, 'source.md'))
    refs = viz_refs(source_md)

    assets_dir = os.path.join(pdir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # Copy referenced viz files (resolved relative to the lesson source dir) into
    # assets/. Sanitize each first: read -> neutralize parent-escaping NAV links ->
    # write. A raw shutil.copy would carry dead href="../index.html" back-links
    # (and hide them from the self-containment gate, which now scans copied viz too).
    copied_viz = []
    for ref in refs:
        resolved = os.path.normpath(os.path.join(src_dir, ref))
        if os.path.isfile(resolved):
            base = os.path.basename(resolved)
            sanitized = _sanitize_copied_viz(_read(resolved))
            open(os.path.join(assets_dir, base), 'w', encoding='utf-8').write(sanitized)
            if base not in copied_viz:
                copied_viz.append(base)

    # Any pre-existing plots (written by experiment.py) live in assets/ already.
    pngs = sorted(f for f in os.listdir(assets_dir)
                  if f.lower().endswith('.png') and os.path.isfile(os.path.join(assets_dir, f)))

    title = _title_from_blog(blog_md, day)
    has_experiment = bool(exp_code.strip())
    has_plot = bool(pngs)

    # ---- render self-contained index.html (all asset links relative) ----
    parts = [
        '<!DOCTYPE html>',
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>%s</title>' % _htmlmod.escape(title),
        '<style>body{max-width:820px;margin:2rem auto;padding:0 1rem;'
        'font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1c2530}'
        'pre{background:#0f172a;color:#e2e8f0;padding:1rem;border-radius:8px;overflow:auto}'
        'code{font-family:ui-monospace,Menlo,monospace}'
        'p code,li code{background:#eef2f7;padding:.1em .3em;border-radius:4px}'
        'img,iframe{max-width:100%;border:1px solid #d9e0e8;border-radius:8px}'
        'iframe{width:100%;height:520px}'
        'h2.repro{margin-top:2.5rem;border-top:1px solid #d9e0e8;padding-top:1.5rem}'
        '</style></head><body>',
        '<article>',
        _md_to_html(blog_md) if blog_md.strip() else '<p><em>(no blog authored yet)</em></p>',
        '</article>',
    ]
    # plots
    for png in pngs:
        parts.append('<figure><img src="assets/%s" alt="%s"></figure>'
                     % (png, _htmlmod.escape(png)))
    # embedded viz
    for base in copied_viz:
        parts.append('<iframe src="assets/%s" title="%s" loading="lazy"></iframe>'
                     % (base, _htmlmod.escape(base)))
    # reproducible experiment section
    parts.append('<h2 class="repro">Reproducible experiment</h2>')
    if has_experiment:
        parts.append('<h3>experiment.py</h3>')
        parts.append('<pre><code>%s</code></pre>' % _htmlmod.escape(exp_code))
    if exp_out.strip():
        parts.append('<h3>Output</h3>')
        parts.append('<pre>%s</pre>' % _htmlmod.escape(exp_out))
    parts.append('</body></html>')

    index_path = os.path.join(pdir, 'index.html')
    open(index_path, 'w', encoding='utf-8').write('\n'.join(parts) + '\n')

    meta = {'module': module, 'day': day, 'title': title,
            'has_experiment': has_experiment, 'has_plot': has_plot,
            'viz': copied_viz}
    with open(os.path.join(pdir, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return index_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('module')
    ap.add_argument('day')
    a = ap.parse_args()
    path = assemble(a.module, a.day, ROOT)
    print('wrote', os.path.relpath(path, ROOT))
    sys.exit(0)


if __name__ == '__main__':
    main()
