# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Worksheet + applier for authoring a Chinese twin of a lesson source.

  python3 sessions/_compiler/zh_worksheet.py <source.md>     # what still needs a twin
  # then, from a script:
  #   import zh_worksheet as wk
  #   wk.front_matter(P, {...}); wk.block_args(P, {...}); wk.apply(P, T, LAB)


worksheet(path)  -> prints numbered items needing translation, with SVG geometry
                    stripped so a big day is readable.
apply(path, T)   -> inserts a ~~~zh fence after each translated prose SPAN and
                    pairs each translated SVG label. Fails loudly on any mismatch.

A "span" is the run of consecutive non-drawing blocks inside one @@@ block, exactly
what a ~~~zh fence pairs. The applier walks the file line by line and rebuilds it,
so no anchor strings are needed anywhere.
"""
import re, sys

FENCE = re.compile(r'^\s*~~~')
WIDGET = re.compile(r'^%%%(\s+(\w+))?')
CALLOUT = re.compile(r'^!!!')
DRAW = ('svg', 'viz')


def parse(path):
    """[(block_header, [items])] where each item is a dict."""
    lines = open(path, encoding='utf-8').read().split('\n')
    # find front-matter end
    fm_end = 0
    if lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                fm_end = i; break
    blocks, cur = [], None
    i = fm_end + 1
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('@@@ '):
            cur = {'hdr': ln, 'hdr_line': i, 'items': []}
            blocks.append(cur); i += 1; continue
        if cur is None:
            i += 1; continue
        s = ln.strip()
        if FENCE.match(s):                      # already translated -> skip whole fence
            j = i + 1
            while j < len(lines) and lines[j].strip() != '~~~':
                j += 1
            cur['items'].append({'kind': 'fence', 'a': i, 'b': j})
            i = j + 1; continue
        m = WIDGET.match(s)
        if m and s != '%%%':
            typ = m.group(2) or ''
            j = i + 1
            while j < len(lines) and lines[j].strip() != '%%%':
                j += 1
            cur['items'].append({'kind': 'draw' if typ in DRAW else 'widget',
                                 'typ': typ, 'a': i, 'b': j,
                                 'body': lines[i + 1:j]})
            i = j + 1; continue
        if CALLOUT.match(s) and s != '!!!':
            j = i + 1
            while j < len(lines) and lines[j].strip() != '!!!':
                j += 1
            cur['items'].append({'kind': 'callout', 'a': i, 'b': j, 'body': lines[i + 1:j]})
            i = j + 1; continue
        if s:
            j = i
            while j < len(lines) and lines[j].strip() and not (
                    lines[j].startswith('@@@ ') or WIDGET.match(lines[j].strip())
                    or CALLOUT.match(lines[j].strip()) or FENCE.match(lines[j].strip())):
                j += 1
            cur['items'].append({'kind': 'prose', 'a': i, 'b': j - 1,
                                 'body': lines[i:j]})
            i = j; continue
        i += 1
    return lines, blocks


def spans(blk):
    """Runs of non-drawing items that still need a Chinese twin.

    A ~~~zh fence marks the span BEFORE it as already translated, so that span is
    discarded rather than reported. Without this the tool reported every finished
    span as outstanding (120 of them on the already-translated pilot day).
    """
    out, cur = [], []
    for it in blk['items']:
        if it['kind'] == 'fence':
            cur = []                      # the span it closes is DONE
            continue
        if it['kind'] == 'draw':
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def worksheet(path):
    lines, blocks = parse(path)
    print("### %s" % path)
    for bi, blk in enumerate(blocks):
        hdr = blk['hdr']
        if hdr.startswith('@@@ fin'):
            continue
        print("\n#### B%d  %s" % (bi, hdr[:110]))
        for si, sp in enumerate(spans(blk)):
            print("  [S%d.%d]" % (bi, si))
            for it in sp:
                tag = it['kind'] if it['kind'] != 'widget' else 'w:' + it['typ']
                txt = '\n'.join(it['body'])
                txt = re.sub(r'<svg.*?</svg>', '«svg»', txt, flags=re.S)
                print("     (%s) %s" % (tag, txt.replace('\n', '\n         ')))
        labs = []
        for it in blk['items']:
            if it['kind'] == 'draw':
                for m in re.finditer(r'<text\b(?![^>]*class="lang-)[^>]*>(.*?)</text>',
                                     '\n'.join(it['body']), re.S):
                    if re.search(r'[A-Za-z]{3,}', m.group(1)):
                        labs.append(m.group(1))
        if labs:
            print("  [LABELS B%d] %d" % (bi, len(labs)))
            for k, t in enumerate(labs):
                print("     L%d.%d :: %s" % (bi, k, t))


def apply(path, T, LAB):
    """T: {'S<b>.<s>': zh_text}   LAB: {'L<b>.<k>': zh_text}"""
    lines, blocks = parse(path)
    ins = {}                                    # line index -> fence text
    used_s = set()
    for bi, blk in enumerate(blocks):
        for si, sp in enumerate(spans(blk)):
            key = 'S%d.%d' % (bi, si)
            if key not in T:
                continue
            used_s.add(key)
            end = sp[-1]['b']
            # The HERO is not fence-parsed: render_hero splits its body on
            # @lede/@goal markers and never calls render_md, so a ~~~zh fence there
            # ships as literal text. Its translation goes in RAW — it supplies
            # @zh_lede / @zh_goal itself, and wraps only its %%% warmup in a fence,
            # which render_hero lifts out explicitly.
            raw = blk['hdr'].startswith('@@@ hero')
            body = T[key].strip('\n')
            if not raw:
                # A span translation must NOT carry its own fence — the applier adds
                # the outer one, so an inner ~~~zh nests and the compiler refuses it
                # ("~~~zh cannot nest"). Only the HERO's translation supplies its own,
                # because it is inserted raw and fences only its %%% warmup. Caught
                # 18 times on day-08, where the hero pattern was copied by mistake.
                assert '~~~zh' not in body, (
                    '%s: span translation contains its own ~~~zh fence; the applier '
                    'adds it. Only the hero supplies its own.' % key)
            ins.setdefault(end, []).append(body if raw else '~~~zh\n%s\n~~~' % body)
    missing = set(T) - used_s
    assert not missing, 'translations with no matching span: %s' % sorted(missing)

    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if i in ins:
            for f in ins[i]:
                out.append(f)
    text = '\n'.join(out)

    # labels
    used_l = set()
    for bi, blk in enumerate(blocks):
        k = 0
        for it in blk['items']:
            if it['kind'] != 'draw':
                continue
            for m in re.finditer(r'<text\b(?![^>]*class="lang-)[^>]*>(.*?)</text>',
                                 '\n'.join(it['body']), re.S):
                if not re.search(r'[A-Za-z]{3,}', m.group(1)):
                    continue
                key = 'L%d.%d' % (bi, k); k += 1
                if key not in LAB:
                    continue
                used_l.add(key)
                whole = m.group(0)
                assert text.count(whole) == 1, '%s: label appears %d times: %r' % (
                    key, text.count(whole), whole[:70])
                attrs = re.match(r'<text\s+([^>]*)>', whole).group(1)
                text = text.replace(
                    whole,
                    whole.replace('<text ', '<text class="lang-en" ', 1)
                    + '<text class="lang-zh" %s>%s</text>' % (attrs, LAB[key]), 1)
    missing_l = set(LAB) - used_l
    assert not missing_l, 'labels with no match: %s' % sorted(missing_l)
    open(path, 'w', encoding='utf-8').write(text)
    print("  %s: +%d fences, +%d labels" % (path.split('/')[-2], len(used_s), len(used_l)))


if __name__ == '__main__':
    worksheet(sys.argv[1])


# ---------------------------------------------------------------------------
# generic front-matter + block-arg patchers
# ---------------------------------------------------------------------------
# Both locate by KEY / by id=, never by the full English string. Matching a long
# title verbatim failed on day-07 twice — once on a typographic apostrophe in
# "ball's", once because the worksheet had truncated the title and the tail was
# guessed ("can climb" vs "climbs").

def front_matter(path, pairs):
    """pairs: {'title': '译文', ...} -> inserts `zh_<key>: "译文"` after each key."""
    s = open(path, encoding='utf-8').read()
    for k, zh in pairs.items():
        assert not re.search(r'(?m)^zh_%s\s*:' % k, s), 'zh_%s already present' % k
        m = re.search(r'(?m)^%s\s*:.*$' % k, s)
        assert m, 'front-matter key %r not found' % k
        s = s[:m.end()] + '\nzh_%s: "%s"' % (k, zh.replace('"', '\\"')) + s[m.end():]
    open(path, 'w', encoding='utf-8').write(s)
    print("  front-matter: +%d zh_ keys" % len(pairs))


def block_args(path, spec):
    """spec: {'c1': (zh_tag, zh_title, zh_gotit)} -> rewrite that block's header."""
    s = open(path, encoding='utf-8').read()
    n = 0
    for bid, (tag, title, gotit) in spec.items():
        m = re.search(r'(?m)^@@@ (\w+) id=%s\b.*$' % re.escape(bid), s)
        assert m, 'no block with id=%s' % bid
        line = m.group(0)
        assert 'zh_tag=' not in line, '%s already has zh_ args' % bid
        out = line
        for key, val in (('tag', tag), ('title', title), ('gotit', gotit)):
            assert ('%s="' % key) in out, '%s has no %s=' % (bid, key)
            out = re.sub(r'(\b%s=")' % key, 'zh_%s="%s" \\1' % (key, val), out, count=1)
        s = s[:m.start()] + out + s[m.end():]
        n += 1
    open(path, 'w', encoding='utf-8').write(s)
    print("  block args: %d blocks" % n)
