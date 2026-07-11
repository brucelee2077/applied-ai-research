#!/usr/bin/env python3
# =============================================================================
# v8lib — shared core for the v8 source-first lesson compiler (Phase C)
# =============================================================================
# Reusable, side-effect-free building blocks used by:
#   * compile_lesson.py          (source.md -> lesson.html)
#   * gates/reader_flow_gate.py  (Reader Flow Gate on source)
#   * gates/shell_invariant_gate.py (Shell Invariant Gate on output)
#
# Everything here is a pure function so compilation stays deterministic.
# =============================================================================
import re

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

FRONTIER_TOKENS = ["GPT-3", "49,152", "49152", "billions of", "frontier model", "frontier lab"]
HUMAN_TOKENS    = ["your brain", "brain", "you "]

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def split_frontmatter(text):
    if not text.startswith('---'):
        raise ValueError("source.md must start with a --- YAML front-matter block")
    end = text.find('\n---', 3)
    if end < 0:
        raise ValueError("unterminated front-matter")
    if yaml is None:
        raise RuntimeError("PyYAML required")
    meta = yaml.safe_load(text[3:end].strip('\n')) or {}
    return meta, text[end + 4:]

def parse_args_kv(s):
    args = {}
    for m in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', s):
        args[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return args

def parse_blocks(body):
    """Flat top-level blocks delimited by lines beginning '@@@ '."""
    blocks, cur = [], None
    for line in body.split('\n'):
        if line.startswith('@@@ '):
            if cur:
                blocks.append(cur)
            parts = line[4:].split(None, 1)
            cur = {'type': parts[0], 'args': parse_args_kv(parts[1] if len(parts) > 1 else ''),
                   'lines': []}
        elif cur is not None:
            cur['lines'].append(line)
    if cur:
        blocks.append(cur)
    return blocks

# ---------------------------------------------------------------------------
# inline markdown-lite
# ---------------------------------------------------------------------------
def attr_esc(s):
    return s.replace('&', '&amp;').replace('"', '&quot;')

def inline(t):
    # [[phrase||tooltip]] -> glossary term span  (before other inline rules)
    t = re.sub(r'\[\[(.+?)\|\|(.+?)\]\]',
               lambda m: '<span class="term" data-tip="%s">%s</span>' % (attr_esc(m.group(2)), m.group(1)),
               t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', t)
    return t

# ---------------------------------------------------------------------------
# typed widget renderers  (replace the old raw-HTML escapes; reusable for D2-D9)
# ---------------------------------------------------------------------------
def _kv(lines):
    """Parse 'key: value' body lines into an ordered dict (values keep colons)."""
    d = {}
    for ln in lines:
        if ':' in ln and not ln.strip().startswith(('#', '-')):
            k, v = ln.split(':', 1)
            if re.fullmatch(r'\w+', k.strip()):
                d[k.strip()] = v.strip()
    return d

def _kv_multiline(lines):
    d = {}
    for ln in lines:
        if ':' in ln:
            k, v = ln.split(':', 1)
            if re.fullmatch(r'\w+', k.strip()):
                d[k.strip()] = v.strip()
    return d

def attr_esc_text(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def render_jargon(lines):
    rows = [tuple(x.strip() for x in ln.split('|', 1)) for ln in lines if '|' in ln]
    head = ('<table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:.86rem">'
            '<thead><tr style="text-align:left;border-bottom:2px solid var(--line2)">'
            '<th style="padding:.4rem .6rem;color:var(--k)">Word</th>'
            '<th style="padding:.4rem .6rem;color:var(--muted)">Plain-English meaning — before we use it</th>'
            '</tr></thead><tbody>')
    body = ''.join(
        '<tr style="border-bottom:1px solid var(--line)">'
        '<td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">%s</td>'
        '<td style="padding:.4rem .6rem;color:var(--ink2)">%s</td></tr>' % (inline(t), inline(g))
        for t, g in rows)
    return head + body + '</tbody></table>'

def render_table(lines):
    """3-col comparison table (:: separated). First row = headers."""
    rows = [[c.strip() for c in ln.split('::')] for ln in lines if ln.strip()]
    if not rows:
        return ''
    hdr_colors = ['var(--k)', 'var(--q)', 'var(--muted)']
    ths = ''.join('<th style="padding:.5rem .6rem;color:%s">%s</th>'
                  % (hdr_colors[min(i, 2)], inline(c)) for i, c in enumerate(rows[0]))
    out = ['<table style="width:100%;border-collapse:collapse;margin:16px 0;font-size:.88rem">'
           '<thead><tr style="text-align:left;border-bottom:2px solid var(--line2)">' + ths
           + '</tr></thead><tbody>']
    for ri, r in enumerate(rows[1:]):
        last = (ri == len(rows) - 2)
        tr = '<tr>' if last else '<tr style="border-bottom:1px solid var(--line)">'
        tds = ''
        for ci, c in enumerate(r):
            style = ('color:var(--ink);font-weight:600' if ci == 0 else 'color:var(--ink2)')
            tds += '<td style="padding:.5rem .6rem;%s">%s</td>' % (style, inline(c))
        out.append(tr + tds + '</tr>')
    out.append('</tbody></table>')
    return ''.join(out)

def render_cards(lines):
    """Everyday-analogy cards.  Each line: emoji | title | body"""
    cards = []
    for ln in lines:
        if ln.count('|') >= 2:
            emoji, title, body = [x.strip() for x in ln.split('|', 2)]
            cards.append('<div class="card"><span class="big">%s</span><h5>%s</h5><p>%s</p></div>'
                         % (emoji, inline(title), inline(body)))
    return '<div class="relate">' + ''.join(cards) + '</div>'

def render_formula(lines):
    d = _kv(lines)
    return ('<div class="callout c-ok"><span class="ic">🧮</span><div><code>%s</code>'
            '<br><span style="font-size:.85rem;color:var(--text2)">%s</span></div></div>'
            % (d.get('expr', ''), inline(d.get('note', ''))))

def render_mathladder(lines):
    d = _kv(lines)
    return ('<div class="callout c-info"><span class="ic">🪜</span><div>'
            '<b>Math Ladder — %s</b>'
            '<br><b>1 · In words:</b> %s'
            '<br><b>2 · The formula:</b> %s'
            '<br><b>3 · Tiny numbers:</b> %s'
            '<br><b>4 · Sanity check:</b> %s</div></div>'
            % (inline(d.get('title', '')), inline(d.get('words', '')), inline(d.get('formula', '')),
               inline(d.get('numbers', '')), inline(d.get('sanity', ''))))

def render_prompt(args, lines):
    """Produce Option-B prompt box.  args: id, label.  body = verbatim prompt text."""
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    text = '\n'.join(lines)
    pid = args.get('id', 'pp')
    return ('<div class="prompt">\n'
            '        <div class="prompt-h"><span class="prompt-l">%s</span>'
            '<button class="copy" type="button" data-copy="#%s">📋 copy</button></div>\n'
            '        <pre class="prompt-t" id="%s">%s</pre>\n'
            '      </div>' % (args.get('label', ''), pid, pid, text))

def render_viz(args):
    """Live-viz embed. Emits the .build-embed iframe the shared auto-resize script already drives."""
    src = args.get('src', '')
    title = args.get('title', 'interactive visualization')
    cap = args.get('caption', 'interactive — try the controls.')
    return ('<div class="build-embed"><iframe src="%s" title="%s" loading="lazy"></iframe>'
            '<div class="cap">%s <a href="%s" target="_blank" rel="noopener">open full screen</a></div></div>'
            % (src, title, cap, src))

def render_svg(lines):
    """Inline static visual: pass raw SVG through, wrapped for consistent styling."""
    svg = '\n'.join(lines).strip()
    return '<div class="build-viz">%s</div>' % svg

def render_demo(args, lines):
    """Inline run-demo: code line, hidden output + takeaway revealed on click."""
    d = _kv_multiline(lines)
    did = args.get('id', 'demo'); label = args.get('label', 'run it')
    code = attr_esc_text(d.get('code', '')); out = attr_esc_text(d.get('out', '')); take = inline(d.get('take', ''))
    return ('<div class="demo" data-demo="%s">'
            '<div class="demo-code"><code>%s</code>'
            '<button class="demo-run" type="button">%s ▶</button></div>'
            '<pre class="demo-out" hidden>%s</pre>'
            '<div class="demo-take" hidden>%s</div></div>'
            % (did, code, label, out, take))

def render_quiz(lines):
    """Authored quiz: one question per line, '|'-separated. q: ask | a:N | opt | ... | fb: text"""
    blocks = []
    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split('|')]
        q = parts[0][2:].strip() if parts[0].lower().startswith('q:') else parts[0]
        ans, opts, fb = 0, [], ''
        for p in parts[1:]:
            if re.match(r'a\s*:', p, re.I):
                ans = int(re.split(r':', p, 1)[1].strip())
            elif p.lower().startswith('fb:'):
                fb = p[3:].strip()
            else:
                opts.append(p)
        optshtml = ''.join('<button class="q-opt" type="button" data-opt="%d"><span class="mark"></span><span>%s</span></button>' % (i, inline(o)) for i, o in enumerate(opts))
        blocks.append('<div class="q" data-correct="%d"><div class="q-ask">%s</div><div class="q-opts">%s</div><div class="q-fb" data-fb="%s"></div></div>' % (ans, inline(q), optshtml, attr_esc(fb)))
    return '<div class="quiz">' + ''.join(blocks) + '</div>'

def render_widget(typ, args, lines):
    if typ == 'jargon':     return render_jargon(lines)
    if typ == 'table':      return render_table(lines)
    if typ == 'cards':      return render_cards(lines)
    if typ == 'formula':    return render_formula(lines)
    if typ == 'mathladder': return render_mathladder(lines)
    if typ == 'prompt':     return render_prompt(args, lines)
    if typ == 'viz':        return render_viz(args)
    if typ == 'svg':        return render_svg(lines)
    if typ == 'demo':       return render_demo(args, lines)
    if typ == 'quiz':       return render_quiz(lines)
    raise ValueError("unknown %%%% widget type: %s" % typ)

# ---------------------------------------------------------------------------
# block-level markdown-lite renderer
# ---------------------------------------------------------------------------
def is_special(line):
    s = line.strip()
    return (s.startswith('!!! ') or s.startswith('~~~') or s.startswith('%%%')
            or s.startswith('#### ') or s.startswith('- ') or bool(re.match(r'^\d+\.\s', s)))

def render_md(text):
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('~~~html'):                      # raw-HTML escape (kept as a fallback)
            i += 1; buf = []
            while i < len(lines) and lines[i].strip() != '~~~':
                buf.append(lines[i]); i += 1
            i += 1
            out.append('\n'.join(buf).strip())
            continue
        if s.startswith('!!! '):                         # callout
            hdr = s[4:].strip().split(None, 1)
            cls = hdr[0]; icon = hdr[1] if len(hdr) > 1 else ''
            i += 1; buf = []
            while i < len(lines) and lines[i].strip() != '!!!':
                buf.append(lines[i]); i += 1
            i += 1
            out.append('<div class="callout %s"><span class="ic">%s</span><div>%s</div></div>'
                       % (cls, icon, inline('\n'.join(buf).strip())))
            continue
        if s.startswith('%%%') and s.strip() != '%%%':   # typed widget
            hdr = s[3:].strip().split(None, 1)
            typ = hdr[0]; wargs = parse_args_kv(hdr[1] if len(hdr) > 1 else '')
            i += 1; buf = []
            while i < len(lines) and lines[i].strip() != '%%%':
                buf.append(lines[i]); i += 1
            i += 1
            out.append(render_widget(typ, wargs, buf))
            continue
        if s.startswith('#### '):
            out.append('<h4>' + inline(s[5:].strip()) + '</h4>'); i += 1; continue
        if s.startswith('- ') or re.match(r'^\d+\.\s', s):
            ordered = bool(re.match(r'^\d+\.\s', s)); items = []
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith('- '): items.append(t[2:]); i += 1
                elif re.match(r'^\d+\.\s', t): items.append(re.sub(r'^\d+\.\s', '', t)); i += 1
                else: break
            tag = 'ol' if ordered else 'ul'
            out.append('<%s>%s</%s>' % (tag, ''.join('<li>%s</li>' % inline(x) for x in items), tag))
            continue
        if s == '':
            i += 1; continue
        buf = [s]; i += 1                                # paragraph
        while i < len(lines) and lines[i].strip() and not is_special(lines[i]):
            buf.append(lines[i].strip()); i += 1
        out.append('<p>' + inline(' '.join(buf)) + '</p>')
    return '\n      '.join(out)

# ---------------------------------------------------------------------------
# region renderers
# ---------------------------------------------------------------------------
def render_hero(meta, block):
    txt = '\n'.join(block['lines'])
    lede = goal = ''
    if '@lede' in txt:
        after = txt.split('@lede', 1)[1]
        lede, _, goal = after.partition('@goal')
    lede = inline(' '.join(l.strip() for l in lede.strip().split('\n') if l.strip()))
    goal = inline(' '.join(l.strip() for l in goal.strip().split('\n') if l.strip()))
    return ('<section id="home" class="hero">\n'
            '      <span class="kicker">%s</span>\n'
            '      <h1>%s<span class="sub">%s</span></h1>\n'
            '      <p class="lede">%s</p>\n'
            '      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>%s</div></div>\n'
            '    </section>' % (meta['module_label'], meta['title'], meta['subtitle'], lede, goal))

def render_section(block):
    a = block['args']
    disabled = a.get('gotit_disabled', 'false') == 'true'
    body = render_md('\n'.join(block['lines']))
    btn = ('<button class="gotit" type="button"%s>%s</button>'
           % (' disabled' if disabled else '', a.get('gotit', 'Done')))
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num %s">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['data_sec'], a['numclass'], a['num'], a['title'], a['tag'], body, btn))

def render_sidebar_nav(meta):
    rows = ['      <div class="nav-group-label">Module 02 · Train</div>']
    for it in meta['sidebar']:
        rows.append('      <button class="nav-link" data-target="%s"><span class="nl-dot"></span>%s</button>'
                    % (it['target'], it['label']))
    return '<nav aria-label="Sections">\n' + '\n'.join(rows) + '\n    </nav>'

def render_fin(meta):
    return ('<div class="fin" id="fin" role="status" aria-hidden="true">\n'
            '      <span class="em" aria-hidden="true">🎉</span>\n'
            '      <h3>%s</h3>\n'
            '      <p>%s</p>\n'
            '    </div>' % (meta['fin_title'], meta['fin_body']))

# ---------------------------------------------------------------------------
# compile: donor + source -> lesson.html   (marker-based region replacement)
# ---------------------------------------------------------------------------
def sub_once(pattern, repl, text, label, flags=re.DOTALL):
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError("region replace matched %d times (expected 1): %s" % (n, label))
    return new

CONTENT_SECTIONS = ['s1', 's2', 's4', 's7']   # reader-flow prose regions authored in source

# Region name -> locating regex (DOTALL). Shared by the compiler AND the
# extractor so a verbatim round-trip is guaranteed byte-identical.
REGION_PATTERNS = {
    'title':       r'<title>.*?</title>',
    'brand_sub':   r'<div class="brand-sub">.*?</div>',
    'sidebar_nav': r'<nav aria-label="Sections">.*?</nav>',
    'hero':        r'<section id="home" class="hero">.*?</section>',
    's1':          r'<section class="module-section" id="s1".*?</section>',
    's2':          r'<section class="module-section" id="s2".*?</section>',
    's4':          r'<section class="module-section" id="s4".*?</section>',
    's7':          r'<section class="module-section" id="s7".*?</section>',
    'fin':         r'<div class="fin" id="fin".*?</div>',
    'nav_prev':    r'<a class="lnav prev" href="[^"]*">.*?</a>',
    'nav_next':    r'<a class="lnav next" href="[^"]*">.*?</a>',
    'DEMOS':       r'var DEMOS = \{.*?\n\};',
    'BUILD':       r'var BUILD=\[.*?\n\];',
    'QS':          r'var QS=\[.*?\n\];',
}

def compile_html(meta, blocks, donor):
    bt = {b['type']: b for b in blocks}
    secs = {b['args'].get('id'): b for b in blocks if b['type'] == 'section'}
    js = {b['args'].get('name'): '\n'.join(b['lines']).strip() for b in blocks if b['type'] == 'js'}
    # verbatim regions (extractor / migration mode): name -> exact HTML/JS
    # rstrip('\n') drops the trailing blank line the final source newline adds to the
    # last @@@ region block, so a verbatim round-trip is exactly byte-identical.
    regions = {b['args'].get('name'): '\n'.join(b['lines']).rstrip('\n')
               for b in blocks if b['type'] == 'region'}

    qid = meta['quest_id']
    m = re.search(r'data-quest-id="([^"]+)"', donor)
    if not m:
        raise RuntimeError("donor missing data-quest-id")
    if m.group(1) != qid:
        raise RuntimeError("quest-id mismatch: donor=%s source=%s (FROZEN)" % (m.group(1), qid))

    # A region is replaced from the verbatim @@@ region block if present,
    # else from the clean rendered/meta path (Day-1 exemplar mode).
    def repl_for(name, rendered):
        return regions[name] if name in regions else rendered

    H = donor
    H = sub_once(REGION_PATTERNS['title'],
                 repl_for('title', '<title>%s</title>' % meta.get('page_title', '')), H, 'title')
    H = sub_once(REGION_PATTERNS['brand_sub'],
                 repl_for('brand_sub', '<div class="brand-sub">%s</div>' % meta.get('brand_sub', '')),
                 H, 'brand-sub')
    H = sub_once(REGION_PATTERNS['sidebar_nav'],
                 repl_for('sidebar_nav', render_sidebar_nav(meta) if 'sidebar_nav' not in regions else ''),
                 H, 'sidebar-nav')
    H = sub_once(REGION_PATTERNS['hero'],
                 repl_for('hero', render_hero(meta, bt['hero']) if 'hero' not in regions else ''),
                 H, 'hero')
    for sid in CONTENT_SECTIONS:
        if sid in regions:
            H = sub_once(REGION_PATTERNS[sid], regions[sid], H, sid)
        elif sid in secs:
            H = sub_once(REGION_PATTERNS[sid], render_section(secs[sid]), H, sid)
    H = sub_once(REGION_PATTERNS['fin'],
                 repl_for('fin', render_fin(meta) if 'fin' not in regions else ''), H, 'fin')
    H = sub_once(REGION_PATTERNS['nav_prev'],
                 repl_for('nav_prev',
                          '<a class="lnav prev" href="%s"><span class="d">← Prev</span><span class="t">%s</span></a>'
                          % (meta.get('nav_prev_href', ''), meta.get('nav_prev_label', ''))), H, 'nav-prev')
    H = sub_once(REGION_PATTERNS['nav_next'],
                 repl_for('nav_next',
                          '<a class="lnav next" href="%s"><span class="d">Next →</span><span class="t">%s</span></a>'
                          % (meta.get('nav_next_href', ''), meta.get('nav_next_label', ''))), H, 'nav-next')
    for name in ('DEMOS', 'BUILD', 'QS'):
        if name in regions:
            H = sub_once(REGION_PATTERNS[name], regions[name], H, name)
        elif name in js:
            H = sub_once(REGION_PATTERNS[name], js[name], H, name)
    return H

