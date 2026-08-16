#!/usr/bin/env python3
# =============================================================================
# Measure untwinned reader-visible prose in a COMPILED lesson.html.
# =============================================================================
# A node with NEITHER lang class shows under BOTH languages, so any text node
# that is not inside an element whose class contains `lang-` is prose a Chinese
# reader sees in English. This walks the compiled page with html.parser, keeps
# an explicit element stack, and sums len() of every text node that:
#   (a) contains a run of 4+ latin letters   -> it is prose, not a number/symbol
#   (b) is not inside any element with a `lang-` class
#   (c) is not inside script/style/code/pre/kbd  -> code is never translated
#
# Usage:  python3 _untwinned_scan.py <lesson.html> [more.html ...] [--detail]
# =============================================================================
import re
import sys
from html.parser import HTMLParser

SKIP_TAGS = {'script', 'style', 'code', 'pre', 'kbd'}
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
        'meta', 'param', 'source', 'track', 'wbr'}
WORD = re.compile(r'[A-Za-z]{4,}')


class Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []          # list of (tag, has_lang_class)
        self.total = 0
        self.hits = []           # (data_sec, text)

    # -- helpers --------------------------------------------------------------
    def _sec(self):
        for tag, _lang, sec in reversed(self.stack):
            if sec:
                return sec
        return '(chrome)'

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        a = dict(attrs)
        lang = 'lang-' in (a.get('class') or '')
        self.stack.append((tag, lang, a.get('data-sec')))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if any(t in SKIP_TAGS for t, _l, _s in self.stack):
            return
        if any(lang for _t, lang, _s in self.stack):
            return
        text = data.strip()
        if not text or not WORD.search(text):
            return
        self.total += len(text)
        self.hits.append((self._sec(), text))


def scan(path):
    p = Scan()
    p.feed(open(path, encoding='utf-8').read())
    return p


if __name__ == '__main__':
    detail = '--detail' in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    grand = 0
    for path in paths:
        p = scan(path)
        grand += p.total
        by_sec = {}
        for sec, text in p.hits:
            by_sec[sec] = by_sec.get(sec, 0) + len(text)
        print('%-72s %5d chars untwinned' % (path.split('sessions/')[-1], p.total))
        for sec in sorted(by_sec, key=lambda s: -by_sec[s]):
            print('      %-14s %5d' % (sec, by_sec[sec]))
        if detail:
            for sec, text in p.hits:
                print('      [%-8s] %4d  %s' % (sec, len(text), text[:150]))
    if len(paths) > 1:
        print('TOTAL %d chars across %d pages' % (grand, len(paths)))
