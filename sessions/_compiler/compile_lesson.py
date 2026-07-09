#!/usr/bin/env python3
# =============================================================================
# v8 Source-First Lesson Compiler  (Phase B — minimal viable pilot)
# =============================================================================
# source.md  ->  lesson.html
#
# Design (see sessions/_refactor/v8_source_first_authoring_plan.md):
#   * The author writes reader-flow content in source.md.
#   * The compiler REUSES the proven shell verbatim from a pristine donor
#     snapshot and marker-replaces ONLY the reader-flow regions + the authored
#     DEMOS/BUILD/QS data. The CSS and JS engine are never touched.
#   * Deterministic + idempotent: compile(source, donor) is a pure function of
#     its two file inputs, so running it twice yields byte-identical output.
#
# Gates:
#   * Reader Flow Gate runs on the PARSED SOURCE *before* compiling. If it
#     fails, nothing is written.
#   * Shell Invariant Gate runs on the COMPILED OUTPUT.
#
# Usage:
#   python3 sessions/_compiler/compile_lesson.py <source.md>
#       [--donor <html>] [--out <html>] [--check-only] [--quiet]
#   exit 0 = compiled + all gates pass ; 2 = reader-flow gate failed (no write)
#           ; 3 = shell-invariant gate failed ; 1 = usage / parse error
# =============================================================================
import sys, os, re, argparse

try:
    import yaml
except Exception:
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
    fm = text[3:end].strip('\n')
    body = text[end+4:]
    if yaml is None:
        raise RuntimeError("PyYAML required")
    meta = yaml.safe_load(fm) or {}
    return meta, body

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
# markdown-lite renderer  (only what the pilot lesson needs)
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

def is_special(line):
    s = line.strip()
    return (s.startswith('!!! ') or s.startswith('~~~') or s.startswith('%%%')
            or s.startswith('#### ') or s.startswith('- ') or bool(re.match(r'^\d+\.\s', s)))

def render_jargon(rows):
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

def render_md(text):
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('~~~html'):
            i += 1; buf = []
            while i < len(lines) and lines[i].strip() != '~~~':
                buf.append(lines[i]); i += 1
            i += 1
            out.append('\n'.join(buf).strip())
            continue
        if s.startswith('!!! '):
            hdr = s[4:].strip().split(None, 1)
            cls = hdr[0]; icon = hdr[1] if len(hdr) > 1 else ''
            i += 1; buf = []
            while i < len(lines) and lines[i].strip() != '!!!':
                buf.append(lines[i]); i += 1
            i += 1
            out.append('<div class="callout %s"><span class="ic">%s</span><div>%s</div></div>'
                       % (cls, icon, inline('\n'.join(buf).strip())))
            continue
        if s.startswith('%%% jargon'):
            i += 1; rows = []
            while i < len(lines) and lines[i].strip() != '%%%':
                if '|' in lines[i]:
                    a, b = lines[i].split('|', 1); rows.append((a.strip(), b.strip()))
                i += 1
            i += 1
            out.append(render_jargon(rows))
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
        buf = [s]; i += 1
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
        lede, _, rest = after.partition('@goal')
        goal = rest
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
    gotit_disabled = a.get('gotit_disabled', 'false') == 'true'
    body = render_md('\n'.join(block['lines']))
    btn = ('<button class="gotit" type="button"%s>%s</button>'
           % (' disabled' if gotit_disabled else '', a.get('gotit', 'Done')))
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
# reader-flow gate  (on parsed source, BEFORE compile)
# ---------------------------------------------------------------------------
def reader_flow_gate(meta, blocks):
    msgs, ok = [], True
    def fail(m):
        nonlocal ok; ok = False; msgs.append('FAIL ' + m)
    def passed(m):
        msgs.append('pass ' + m)

    bt = {b['type']: b for b in blocks}
    secs = {b['args'].get('id'): b for b in blocks if b['type'] == 'section'}

    hero = bt.get('hero')
    if not hero:
        fail('no hero block')
    else:
        htxt = '\n'.join(hero['lines'])
        lede = htxt.split('@lede', 1)[1].split('@goal', 1)[0] if '@lede' in htxt else htxt
        low = lede.lower()
        hit = [t for t in FRONTIER_TOKENS if t.lower() in low]
        if hit:
            fail('hero opens frontier-first (found %s) — must open on human intuition' % hit)
        else:
            passed('hero has no frontier-pressure opening')
        if any(t in low for t in [h.strip().lower() for h in HUMAN_TOKENS]):
            passed('hero opens on human intuition/curiosity')
        else:
            fail('hero missing a human/curiosity anchor')

    s1 = secs.get('s1')
    if not s1:
        fail('no s1 section')
    else:
        s1txt = '\n'.join(s1['lines'])
        # jargon ladder present
        if '%%% jargon' in s1txt:
            passed('s1 has a front-loaded Jargon Ladder')
        else:
            fail('s1 missing Jargon Ladder (%%% jargon)')
        # picture before first term/definition: the "picture" heading/paragraph must
        # appear before the jargon table AND before the three-step list
        idx_pic = s1txt.lower().find('picture')
        idx_jar = s1txt.find('%%% jargon')
        idx_term = s1txt.find('[[')
        if idx_pic >= 0 and (idx_jar < 0 or idx_pic < idx_jar) and (idx_term < 0 or idx_pic < idx_term):
            passed('mental picture precedes vocabulary in s1')
        else:
            fail('s1 mental picture does not clearly precede the vocabulary')
        if any(t in s1txt for t in ['GPT-3', '49,152', '49152']):
            fail('s1 leaks frontier payoff (belongs in s4, after mechanism)')
        else:
            passed('s1 defers frontier payoff')

    s4 = secs.get('s4')
    if s4:
        s4txt = '\n'.join(s4['lines'])
        if any(t in s4txt for t in ['GPT-3', '49,152', '49152']):
            passed('frontier payoff lands in s4 (after mechanism)')
        else:
            msgs.append('warn s4 has no explicit frontier payoff')
        if 'interview' in s4txt.lower():
            passed('s4 carries a staff/interview grounding')

    # narrative spine present across hero + s1 + s2 + s4
    spine_word = 'brain'
    present = [name for name, b in [('hero', bt.get('hero')), ('s1', secs.get('s1')),
                                    ('s2', secs.get('s2')), ('s4', secs.get('s4'))]
               if b and spine_word in '\n'.join(b['lines']).lower()]
    if len(present) >= 3:
        passed("spine ('%s') runs through %s" % (spine_word, '+'.join(present)))
    else:
        fail("spine ('%s') appears in too few blocks: %s" % (spine_word, present))

    s7 = secs.get('s7')
    if s7:
        s7txt = '\n'.join(s7['lines']).lower()
        if 'predict' in s7txt and ('what you should see' in s7txt or 'observe' in s7txt):
            passed('produce is discovery-framed (predict + observe)')
        else:
            fail('produce is not discovery-framed (need predict + "what you should see")')
    return ok, msgs

# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
def sub_once(pattern, repl, text, label, flags=re.DOTALL):
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError("region replace matched %d times (expected 1): %s" % (n, label))
    return new

def compile_lesson(meta, blocks, donor):
    bt = {b['type']: b for b in blocks}
    secs = {b['args'].get('id'): b for b in blocks if b['type'] == 'section'}
    js = {b['args'].get('name'): '\n'.join(b['lines']).strip() for b in blocks if b['type'] == 'js'}

    # quest-id invariant
    qid = meta['quest_id']
    m = re.search(r'data-quest-id="([^"]+)"', donor)
    if not m:
        raise RuntimeError("donor missing data-quest-id")
    if m.group(1) != qid:
        raise RuntimeError("quest-id mismatch: donor=%s source=%s (FROZEN)" % (m.group(1), qid))

    H = donor
    H = sub_once(r'<title>.*?</title>', '<title>%s</title>' % meta['page_title'], H, 'title')
    H = sub_once(r'<div class="brand-sub">.*?</div>',
                 '<div class="brand-sub">%s</div>' % meta['brand_sub'], H, 'brand-sub')
    H = sub_once(r'<nav aria-label="Sections">.*?</nav>', render_sidebar_nav(meta), H, 'sidebar-nav')
    H = sub_once(r'<section id="home" class="hero">.*?</section>', render_hero(meta, bt['hero']), H, 'hero')
    for sid in ['s1', 's2', 's4', 's7']:
        if sid in secs:
            H = sub_once(r'<section class="module-section" id="%s".*?</section>' % sid,
                         render_section(secs[sid]), H, sid)
    H = sub_once(r'<div class="fin" id="fin".*?</div>', render_fin(meta), H, 'fin')
    H = sub_once(r'<a class="lnav prev" href="[^"]*">.*?</a>',
                 '<a class="lnav prev" href="%s"><span class="d">← Prev</span><span class="t">%s</span></a>'
                 % (meta['nav_prev_href'], meta['nav_prev_label']), H, 'nav-prev')
    H = sub_once(r'<a class="lnav next" href="[^"]*">.*?</a>',
                 '<a class="lnav next" href="%s"><span class="d">Next →</span><span class="t">%s</span></a>'
                 % (meta['nav_next_href'], meta['nav_next_label']), H, 'nav-next')
    if 'DEMOS' in js:
        H = sub_once(r'var DEMOS = \{.*?\n\};', js['DEMOS'], H, 'DEMOS')
    if 'BUILD' in js:
        H = sub_once(r'var BUILD=\[.*?\n\];', js['BUILD'], H, 'BUILD')
    if 'QS' in js:
        H = sub_once(r'var QS=\[.*?\n\];', js['QS'], H, 'QS')
    return H

# ---------------------------------------------------------------------------
# shell-invariant gate  (on compiled output)
# ---------------------------------------------------------------------------
def shell_invariant_gate(html, meta):
    msgs, ok = [], True
    def check(cond, label):
        nonlocal ok
        msgs.append(('pass ' if cond else 'FAIL ') + label); ok = ok and cond
    check('data-quest-id="%s"' % meta['quest_id'] in html, 'quest-id frozen (%s)' % meta['quest_id'])
    check(html.count('class="module-section"') == 7, '7 module-sections (got %d)' % html.count('class="module-section"'))
    for t in ['home', 's1', 's2', 's3', 's4', 's5', 's6', 's7']:
        check(('data-target="%s"' % t) in html, 'sidebar data-target=%s' % t)
    check('var DEMOS = {' in html, 'DEMOS present')
    check('var BUILD=[' in html, 'BUILD present')
    check('var QS=[' in html, 'QS present')
    check(html.count('data-demo=') >= 3, 'playground >=3 demo buttons (got %d)' % html.count('data-demo='))
    nq = len(re.findall(r'\bans:\s*\d', html))
    check(nq == 4, 'quiz has 4 questions (got %d)' % nq)
    nopts = len(re.findall(r"opts:\[", html))
    check(nopts == 4, 'quiz has 4 opts arrays (got %d)' % nopts)
    check('frontier-lesson:' in html, "localStorage key frontier-lesson:")
    check('frontier-theme' in html, "localStorage key frontier-theme")
    check('class="fin" id="fin"' in html, '.fin completion banner')
    check(('href="%s"' % meta['nav_prev_href']) in html, 'prev nav href')
    check(('href="%s"' % meta['nav_next_href']) in html, 'next nav href')
    check('{{' not in html and '@@@' not in html, 'no unresolved markers')
    return ok, msgs

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--donor')
    ap.add_argument('--out')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    src_dir = os.path.dirname(os.path.abspath(args.source))
    text = open(args.source, encoding='utf-8').read()
    meta, body = split_frontmatter(text)
    blocks = parse_blocks(body)

    donor_path = args.donor or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'shells', meta['donor'])
    out_path = args.out or os.path.join(src_dir, 'lesson.html')

    def log(*a):
        if not args.quiet:
            print(*a)

    log('== v8 compile:', os.path.relpath(args.source), '->', os.path.relpath(out_path))
    log('   donor:', os.path.relpath(donor_path), '| mode:', meta.get('mode'))

    ok, msgs = reader_flow_gate(meta, blocks)
    log('\n-- Reader Flow Gate (source) --')
    for m in msgs: log('  ', m)
    if not ok:
        log('\nReader Flow Gate FAILED — nothing written.')
        sys.exit(2)

    donor = open(donor_path, encoding='utf-8').read()
    html = compile_lesson(meta, blocks, donor)

    sok, smsgs = shell_invariant_gate(html, meta)
    log('\n-- Shell Invariant Gate (output) --')
    for m in smsgs: log('  ', m)

    if not args.check_only:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        log('\nwrote', os.path.relpath(out_path), '(%d bytes)' % len(html))
    else:
        log('\n--check-only: not written')

    if not sok:
        log('\nShell Invariant Gate FAILED.')
        sys.exit(3)
    log('\nOK — compiled and all gates pass.')

if __name__ == '__main__':
    main()
