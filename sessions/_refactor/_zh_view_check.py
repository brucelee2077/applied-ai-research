#!/usr/bin/env python3
"""Render the ZH view of a compiled lesson and prove nothing vanished.

Two checks that a character count cannot make:
  1. ADJACENCY — every element carrying class `lang-en` has a `lang-zh` sibling
     inside the same parent. An unpaired lang-en is display:none for a Chinese
     reader, so it silently VANISHES; that is the one way this task can regress.
  2. ZH VIEW — drop every lang-en subtree, keep every lang-zh subtree, and print
     the surviving visible text per data-sec. A section that went blank shows up
     here immediately.
"""
import re
import sys
from html.parser import HTMLParser

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
        'meta', 'param', 'source', 'track', 'wbr'}
SKIP = {'script', 'style'}


class ZhView(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []           # (tag, lang_class_or_None, data_sec)
        self.out = {}             # data_sec -> [text]
        self.parents = []         # per-open-element: list of child lang classes
        self.orphans = []

    def _sec(self):
        for _t, _l, sec in reversed(self.stack):
            if sec:
                return sec
        return '(chrome)'

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get('class') or ''
        lang = 'en' if 'lang-en' in cls else ('zh' if 'lang-zh' in cls else None)
        if lang and self.parents:
            self.parents[-1].append(lang)
        if tag in VOID:
            return
        self.stack.append((tag, lang, a.get('data-sec')))
        self.parents.append([])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                # every element closing here: check its direct lang children pair up
                for j in range(i, len(self.stack)):
                    kids = self.parents[j] if j < len(self.parents) else []
                    if kids.count('en') != kids.count('zh'):
                        self.orphans.append((self.stack[j][0], self._sec(),
                                             kids.count('en'), kids.count('zh')))
                del self.stack[i:]
                del self.parents[i:]
                return

    def handle_data(self, data):
        if any(t in SKIP for t, _l, _s in self.stack):
            return
        if any(l == 'en' for _t, l, _s in self.stack):   # hidden in ZH mode
            return
        t = data.strip()
        if t:
            self.out.setdefault(self._sec(), []).append(t)


LATIN = re.compile(r'[A-Za-z]{4,}')

if __name__ == '__main__':
    path = sys.argv[1]
    p = ZhView()
    p.feed(open(path, encoding='utf-8').read())
    print('== ZH view of %s ==' % path.split('sessions/')[-1])
    if p.orphans:
        print('  FAIL %d unbalanced lang container(s):' % len(p.orphans))
        for tag, sec, en, zh in p.orphans[:20]:
            print('       <%s> in %-9s en=%d zh=%d' % (tag, sec, en, zh))
    else:
        print('  ok   every lang-en has a lang-zh sibling in the same parent')
    for sec in ('what', 'intuition', 'play', 'why', 'build', 'quiz', 'produce'):
        texts = p.out.get(sec, [])
        joined = ' '.join(texts)
        han = sum(1 for c in joined if '一' <= c <= '鿿')
        print('  %-9s %4d visible nodes · %4d 汉字 · %s'
              % (sec, len(texts), han, 'EMPTY!' if not texts else 'ok'))
    sys.exit(1 if p.orphans else 0)
