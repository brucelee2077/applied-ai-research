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
    # `<` and `>` MUST be escaped, not just `&` and `"`. Without them a gloss body
    # containing markdown emphasis got an <em> injected INSIDE the data-tip attribute
    # by the rules below, which terminated the attribute early. See inline().
    return (s.replace('&', '&amp;').replace('"', '&quot;')
             .replace('<', '&lt;').replace('>', '&gt;'))

def inline(t):
    # [[phrase||tooltip]] -> glossary term span.
    #
    # The rendered span is STASHED behind a placeholder (exactly as the `code` guard
    # below does) instead of being emitted inline. Reason, measured on the shipped
    # corpus: this substitution used to run FIRST, so the ** / * rules further down
    # rewrote the text INSIDE the finished data-tip="..." attribute. Two real bugs
    # came out of that on m04 (4 corrupted tooltips across 3 lessons):
    #   (a) one gloss holding *emphasis* got a literal <em> injected into its tooltip,
    #       so the reader hovering "gradient" saw `steepest <em>increase</em>`;
    #   (b) two glosses on ONE line, each holding a single `*`, cross-paired into an
    #       <em>…</em> spanning TWO DIFFERENT attributes.
    # It also desynced coverage_judge._readable_text (a naive tag strip) on the stray
    # `>`, so every LLM judge graded mangled prose and passed it.
    _tips = []
    def _stash_tip(m):
        _tips.append('<span class="term" data-tip="%s">%s</span>'
                     % (attr_esc(m.group(2)), m.group(1)))
        return '\x01%d\x01' % (len(_tips) - 1)
    t = re.sub(r'\[\[(.+?)\|\|(.+?)\]\]', _stash_tip, t)
    # Protect `code` spans: stash them behind placeholders so later inline rules
    # (** bold, * em) can never reach INTO code — e.g. `(x-y)**2` must not start a
    # <strong> that runs to the next ** elsewhere in the paragraph.
    _code = []
    def _stash(m):
        _code.append('<code>%s</code>' % m.group(1))
        return '\x00%d\x00' % (len(_code) - 1)
    t = re.sub(r'`([^`]+)`', _stash, t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'\x00(\d+)\x00', lambda m: _code[int(m.group(1))], t)
    t = re.sub(r'\x01(\d+)\x01', lambda m: _tips[int(m.group(1))], t)
    return t

# ---------------------------------------------------------------------------
# typed widget renderers  (replace the old raw-HTML escapes; reusable for D2-D9)
# ---------------------------------------------------------------------------
def _kv(lines):
    """Parse 'key: value' body lines into an ordered dict (values keep colons).

    A line opens a NEW field only when it is `key:` — a `\\w+` word IMMEDIATELY
    followed by a colon (no space between), which is the authoring convention
    for every field (`code:`, `out:`, `take:`, `expr:`, `note:`, `words:`, …).
    Any other line (including an aligned continuation like ``join  :  …`` whose
    word has spaces before the colon) is a CONTINUATION of the previous field's
    value and is appended with a newline. This lets a `demo` `out:` (or a
    `formula`/`mathladder` field) carry several lines that all reach the
    rendered <pre> — without it, only the first line survived and every
    following line was silently dropped.
    """
    d = {}
    last = None
    cont_raw = {}  # key -> list of raw continuation lines (for dedented alignment)
    for ln in lines:
        stripped = ln.strip()
        # new field: leading word directly touching a colon, e.g. "out:" / "take:".
        m = re.match(r'\s*(\w+):', ln)
        if m and not stripped.startswith(('#', '-')):
            last = m.group(1)
            value = ln.split(':', 1)[1].strip()
            # A REPEATED key accumulates instead of overwriting. Authors write a
            # multi-line result as several `out:` lines; overwriting kept only the
            # last one, so a build-up shipped with its build removed. Flush any
            # continuation lines gathered so far first, to preserve author order.
            if last in d:
                d[last] = _join_cont(d[last], cont_raw.pop(last, None))
                d[last] += '\n' + value
            else:
                d[last] = value
            cont_raw[last] = []
            continue
        # continuation line: fold it into the current field's value so multi-line
        # values survive. Keep raw (rstrip only) so the author's column
        # alignment is preserved for watchable numbers; a common leading indent
        # is removed below.
        if last is not None and stripped:
            cont_raw[last].append(ln.rstrip())
    # append continuation lines, dedented by their common leading indent, so
    # aligned columns stay aligned in the rendered <pre> without a big left gap.
    for k, raws in cont_raw.items():
        d[k] = _join_cont(d[k], raws)
    return d


def _join_cont(value, raws):
    """Append continuation lines to a field value, dedented by their common indent.

    Keeping the author's relative column alignment matters for watchable
    numbers; the shared leading indent is stripped so the block does not render
    with a big left gap.
    """
    if not raws:
        return value
    indent = min(len(r) - len(r.lstrip(' ')) for r in raws)
    return value + '\n' + '\n'.join(r[indent:] for r in raws)

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
    """Inline run-demo: code line, hidden output + takeaway revealed on click.
    NOTE: this widget only UN-HIDES a pre-baked output; it does not compute. So the
    default button label is a reveal verb, not 'run it' (which oversells execution).
    Author-supplied `label:` is used verbatim (existing lessons stay byte-identical).
    Optional `predict:` field (Build-Up Register) prepends a "predict first" prompt so a
    worked example becomes a discovery — the reader forms a guess BEFORE revealing. When
    `predict:` is absent the output is byte-identical to before."""
    d = _kv(lines)
    did = args.get('id', 'demo'); label = args.get('label', 'reveal')
    code = attr_esc_text(d.get('code', '')); out = attr_esc_text(d.get('out', '')); take = inline(d.get('take', ''))
    pred = d.get('predict')
    pred_html = ('<div class="demo-predict" style="padding:.6rem .9rem;font-size:.85rem;'
                 'color:var(--ink2);background:var(--panel);border-bottom:1px solid var(--line)">'
                 '🤔 <b>Predict first:</b> %s</div>' % inline(pred)) if pred else ''
    return ('<div class="demo" data-demo="%s">%s'
            '<div class="demo-code"><code>%s</code>'
            '<button class="demo-run" type="button">%s ▶</button></div>'
            '<pre class="demo-out" hidden>%s</pre>'
            '<div class="demo-take" hidden>%s</div></div>'
            % (did, pred_html, code, label, out, take))

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
            # answer marker: only when the tail after 'a:' is all digits (else it is an option)
            m = re.match(r'a\s*:\s*(.*)$', p, re.I)
            if m and re.fullmatch(r'\d+', m.group(1).strip()):
                ans = int(m.group(1).strip())
            elif p.lower().startswith('fb:'):
                fb = p.split(':', 1)[1].strip()
            else:
                opts.append(p)
        optshtml = ''.join('<button class="q-opt" type="button" data-opt="%d"><span class="mark"></span><span>%s</span></button>' % (i, inline(o)) for i, o in enumerate(opts))
        blocks.append('<div class="q" data-correct="%d"><div class="q-ask">%s</div><div class="q-opts">%s</div><div class="q-fb" data-fb="%s"></div></div>' % (ans, inline(q), optshtml, attr_esc(fb)))
    return '<div class="quiz">' + ''.join(blocks) + '</div>'

def render_hint(lines):
    """Tiered hint ladder (%%% hint): 't1:/t2:/t3:' lines -> progressive-disclosure hints
    for a stuck learner. Tiers ship hidden; donor JS un-hides ONE per click. Offline —
    no network. t1 = a gentle nudge, t2 = a worked micro-step, t3 = the idea in one sentence."""
    d = _kv(lines)
    tiers = [d[k] for k in ('t1', 't2', 't3', 't4') if d.get(k)]
    if not tiers:
        return ''
    items = ''.join('<div class="hint-tier" data-hint-tier="%d" hidden>%s</div>' % (i + 1, inline(t))
                    for i, t in enumerate(tiers))
    return ('<div class="hint" data-hint-total="%d">'
            '<button class="hint-reveal" type="button">💡 Stuck? reveal a hint</button>'
            '%s</div>' % (len(tiers), items))

def render_insight(lines):
    """%%% insight — an inline "why this matters / notice this" RE-HOOK callout for the
    build-up. The Build-Up Register's stakes beat: a one-line pull that keeps the reader
    engaged mid-mechanism instead of letting the body go cold. Reuses the styled .takeaway
    block (no donor CSS change); the 💡 is inline text. Body prose runs through inline() so
    [[term||gloss]] / **bold** / `code` all work inside it."""
    text = inline(' '.join(l.strip() for l in lines if l.strip()))
    return '<div class="takeaway">💡 %s</div>' % text

def render_steps(lines):
    """%%% steps — a narrated stepped worked-example (Build-Up Register). Body is repeated
    'step:' (the work) + 'why:' (plain-English gloss) pairs. Renders into the .build /
    .build-step / .build-num / .build-note scaffold the donor's __revealBuild() reveals on
    scroll — turning a cold "Step 1/2/3" dump into a narrated build a beginner can follow.
    A 'step:' with no following 'why:' is tolerated; a stray continuation line folds into
    the last field. Both work and gloss run through inline()."""
    steps, last = [], None   # last in {'work','why'}
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r'(step|why)\s*:(.*)$', s, re.I)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == 'step':
                steps.append({'work': val, 'why': ''}); last = 'work'
            else:
                if not steps:
                    steps.append({'work': '', 'why': val})
                else:
                    steps[-1]['why'] = (steps[-1]['why'] + ' ' + val).strip()
                last = 'why'
        elif steps and last:
            steps[-1][last] = (steps[-1][last] + ' ' + s).strip()
    if not steps:
        return ''
    rows = []
    for i, st in enumerate(steps, 1):
        work = ('<b>%s</b>' % inline(st['work'])) if st['work'] else ''
        why = (' — %s' % inline(st['why'])) if st['why'] else ''
        rows.append('<div class="build-step"><div class="build-note">'
                    '<span class="build-num">%d</span>%s%s</div></div>' % (i, work, why))
    return '<div class="build">' + ''.join(rows) + '</div>'

def render_warmup(lines):
    """%%% warmup — a top-of-lesson RECALL quiz on PRIOR-day concepts (retention / spaced
    retrieval). Same line format as %%% quiz plus an optional 'concept: <id>' per line;
    answering records SM-2 in the donor (srReview). Rendered in a .warmup wrapper (NOT
    .quiz) so the generic quiz engine ignores it and the warm-up engine (which records SR)
    handles it. Answered BEFORE new content = effortful recall, the strongest retention lever."""
    blocks = []
    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split('|')]
        q = parts[0][2:].strip() if parts[0].lower().startswith('q:') else parts[0]
        ans, opts, fb, cid = 0, [], '', ''
        for p in parts[1:]:
            m = re.match(r'a\s*:\s*(.*)$', p, re.I)
            mc = re.match(r'concept\s*:\s*(.*)$', p, re.I)
            if m and re.fullmatch(r'\d+', m.group(1).strip()):
                ans = int(m.group(1).strip())
            elif p.lower().startswith('fb:'):
                fb = p.split(':', 1)[1].strip()
            elif mc:
                cid = mc.group(1).strip()
            else:
                opts.append(p)
        optshtml = ''.join('<button class="q-opt" type="button" data-opt="%d"><span class="mark"></span><span>%s</span></button>' % (i, inline(o)) for i, o in enumerate(opts))
        blocks.append('<div class="q" data-correct="%d" data-concept="%s"><div class="q-ask">%s</div><div class="q-opts">%s</div><div class="q-fb" data-fb="%s"></div></div>'
                      % (ans, attr_esc(cid), inline(q), optshtml, attr_esc(fb)))
    return ('<div class="warmup"><div class="warmup-h">🔁 Warm-up — do you still remember? (from earlier days)</div>'
            + ''.join(blocks) + '</div>')

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
    if typ == 'hint':       return render_hint(lines)
    if typ == 'warmup':     return render_warmup(lines)
    if typ == 'insight':    return render_insight(lines)
    if typ == 'steps':      return render_steps(lines)
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
    # Optional %%% warmup ... %%% recall block (retention): extract BEFORE the lede/goal
    # split so it isn't swallowed into the goal text; render it after the goal.
    warm_html = ''
    mw = re.search(r'(?ms)^%%%\s+warmup\s*\n(.*?)^%%%\s*$', txt)
    if mw:
        warm_html = render_warmup(mw.group(1).split('\n'))
        txt = txt[:mw.start()] + txt[mw.end():]
    lede = goal = ''
    if '@lede' in txt:
        after = txt.split('@lede', 1)[1]
        lede, _, goal = after.partition('@goal')
    lede = inline(' '.join(l.strip() for l in lede.strip().split('\n') if l.strip()))
    goal = inline(' '.join(l.strip() for l in goal.strip().split('\n') if l.strip()))
    warm_line = ('\n      ' + warm_html) if warm_html else ''
    return ('<section id="home" class="hero">\n'
            '      <span class="kicker">%s</span>\n'
            '      <h1>%s<span class="sub">%s</span></h1>\n'
            '      <p class="lede">%s</p>\n'
            '      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>%s</div></div>%s\n'
            '    </section>' % (meta['module_label'], meta['title'], meta['subtitle'], lede, goal, warm_line))

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

def render_concept(block):
    """V9 concept unit: a tracked .module-section (intro -> inline visual -> build-up) with one gotit."""
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    num = a.get('num', ''); numclass = a.get('numclass', 's-study')
    btn = '<button class="gotit" type="button">%s</button>' % a.get('gotit', 'Got it')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num %s">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], numclass, num, a.get('title', ''), a.get('tag', ''), body, btn))

def concept_nav_items(blocks):
    """V9 sidebar nav: auto-number concept units in source order; keep quiz/produce."""
    items = [{'target': 'home', 'label': 'Start here'}]
    n = 0
    for b in blocks:
        if b['type'] == 'concept':
            n += 1
            items.append({'target': b['args']['id'], 'label': '%d · %s' % (n, b['args'].get('tag') or b['args'].get('title', ''))})
        elif b['type'] in ('quiz', 'produce'):
            items.append({'target': b['args']['id'], 'label': b['args'].get('tag') or b['args'].get('title', b['type'].title())})
    return items

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
    if meta.get('mode') == 'concept':
        return _compile_concept(meta, blocks, donor)
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


# ---------------------------------------------------------------------------
# V9 concept-mode compile: marker-based assembly into a neutral donor
# ---------------------------------------------------------------------------
def _compile_concept(meta, blocks, donor):
    # quest-id: v9 donor is a NEUTRAL template carrying data-quest-id="__QUEST_ID__".
    # Substitute the source's quest_id, then verify it landed. NO donor-vs-source mismatch compare.
    qid = meta['quest_id']
    donor = donor.replace('__QUEST_ID__', qid)
    if ('data-quest-id="%s"' % qid) not in donor:
        raise RuntimeError('donor missing data-quest-id="__QUEST_ID__" template token')

    bt = {b['type']: b for b in blocks}
    parts = []
    if 'hero' in bt:
        parts.append(render_hero(meta, bt['hero']))
    n = 0
    for b in blocks:
        if b['type'] == 'concept':
            n += 1
            b['args']['num'] = str(n)
            parts.append(render_concept(b))
        elif b['type'] == 'quiz':
            parts.append(render_quiz_section(b))
        elif b['type'] == 'produce':
            parts.append(render_produce_section(b))
    fin_html = render_fin(meta)
    content = '\n\n    '.join(parts)
    nav = render_sidebar_nav_items(meta, concept_nav_items(blocks))

    H = donor
    H = sub_once(r'<title>.*?</title>', '<title>%s</title>' % meta.get('page_title', ''), H, 'title')
    H = sub_once(r'<div class="brand-sub">.*?</div>', '<div class="brand-sub">%s</div>' % meta.get('brand_sub', ''), H, 'brand-sub')
    H = H.replace('<!--V9_NAV-->', nav, 1)
    H = H.replace('<!--V9_CONTENT-->', content + '\n\n    ' + fin_html, 1)
    H = sub_once(REGION_PATTERNS['nav_prev'], '<a class="lnav prev" href="%s"><span class="d">← Prev</span><span class="t">%s</span></a>' % (meta.get('nav_prev_href', ''), meta.get('nav_prev_label', '')), H, 'nav-prev')
    H = sub_once(REGION_PATTERNS['nav_next'], '<a class="lnav next" href="%s"><span class="d">Next →</span><span class="t">%s</span></a>' % (meta.get('nav_next_href', ''), meta.get('nav_next_label', '')), H, 'nav-next')
    return H


def render_quiz_section(block):
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    btn = '<button class="gotit" type="button" disabled>%s</button>' % a.get('gotit', 'answer all first')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num s-quiz">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], a.get('num', ''), a.get('title', ''), a.get('tag', 'Quiz'), body, btn))


def render_produce_section(block):
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    btn = '<button class="gotit" type="button">%s</button>' % a.get('gotit', 'Done')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num s-produce">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], a.get('num', ''), a.get('title', ''), a.get('tag', 'Produce'), body, btn))


def render_sidebar_nav_items(meta, items):
    rows = ['      <div class="nav-group-label">%s</div>' % meta.get('module_label', '')]
    for it in items:
        rows.append('      <button class="nav-link" data-target="%s"><span class="nl-dot"></span>%s</button>' % (it['target'], it['label']))
    return '\n'.join(rows)

