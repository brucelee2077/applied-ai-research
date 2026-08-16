#!/usr/bin/env python3
# =============================================================================
# Language Parity Gate — is the Chinese version actually COMPLETE?
# =============================================================================
# The reader's second principle: a module's depth and breadth are fixed up front
# and gated at the end — in BOTH languages. Every other gate measures one
# language's quality. This one measures whether the two languages say the same
# amount, because the failure mode of a bilingual page is not bad Chinese, it is
# HALF Chinese: a concept whose build-up was never translated, a drawing whose
# labels are English-only, a quiz whose Chinese twin points at the wrong answer.
# Each of those renders a page that looks finished and teaches less.
#
# WHEN IT APPLIES. A day with no Chinese at all is not "failing" — it has not
# started, and the CSS fallback shows English. So the gate is inert until the
# source DECLARES Chinese by containing a `~~~zh` fence or any `zh_*` key. From
# that moment the day must be complete: partial is the state worth failing.
#
# Six checks, all deterministic and offline:
#   1. every @@@ concept carries Chinese, and no prose span is left untwinned
#  1r. the same question for `mode: exemplar` days, which have `@@@ region` blocks of
#      verbatim HTML and no concepts at all: paired lang-en/lang-zh node counts must
#      balance, no region may hide a ~~~zh fence (a region is never rendered, so the
#      fence would ship as literal text), and no prose region may be wholly English.
#      Added because check 1 iterated concepts and therefore reported
#      "pass all 0 concept units carry Chinese" on every one of m01's six days.
#   2. every reader-visible SVG label has a paired <text class="lang-zh">
#      (symbol/number-only labels are auto-exempt — "N = 7 000 000 000" needs no
#       translation, and demanding one would be noise)
#   3. every reader-visible front-matter key has its zh_ twin
#   4. every quiz question has a Chinese twin WITH THE SAME ANSWER INDEX
#      (this one is a correctness red line, not a formatting nit)
#   5. no untranslated English sentence hides inside the Chinese — Latin word runs
#      in Chinese prose must be whitelisted technical terms
#   6. every manifest `covers` topic is reachable from the Chinese reading path
#
# Reusable:  from lang_parity_gate import run ; ok, msgs = run(source_text)
# CLI:       python3 gates/lang_parity_gate.py <source.md> [--audit]
#            exit 0 pass / 6 fail   (6 was free: 2,3,4,5 are taken)
# =============================================================================
import sys, os, re, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

# Front-matter keys the reader actually SEES on the page. page_title is absent on
# purpose: it is the browser tab, and HTML has no way to show two of them.
_VISIBLE_FM = ['title', 'subtitle', 'module_label', 'fin_title', 'fin_body']

# A label needing no translation: no Latin word of 3+ letters. Numbers, operators,
# ticks, single letters and bare identifiers are the same in both languages.
_NEEDS_WORDS = re.compile(r'[A-Za-z]{3,}')

# Counting the language classes in a REGION (raw HTML written by hand). Deliberately
# tolerant, because every form below renders identically to the reader and a count
# that only recognised the canonical one could be dodged by accident:
#   class="lang-en"      the canonical form (686 of 686 occurrences in m01 today)
#   class='lang-en'      single quotes — invisible to a double-quote-only regex
#   class=\"lang-en\"    escaped, which is what you get inside a double-quoted JS
#                        string in the DEMOS / BUILD / QS regions
#   class="lede lang-en" a second class alongside it
def _lang_class_re(lang):
    return re.compile(r'class\s*=\s*\\?["\'][^"\']*\blang-%s\b' % lang)


_LANG_EN_RE = _lang_class_re('en')
_LANG_ZH_RE = _lang_class_re('zh')


def _visible_text(region_html):
    """Roughly how much reader-facing text a region carries, tags removed.

    Only used against a threshold, so entity decoding and JS-string quoting do not
    need to be exact — it just has to tell a <title> line (57 chars) apart from a
    translated section body (2000+).
    """
    t = re.sub(r'(?s)<!--.*?-->', ' ', region_html)
    t = re.sub(r'<[^>]+>', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# sessions/_compiler/gates -> sessions/_compiler -> sessions -> _refactor/zh_terms.yaml.
# The first version stopped one dirname short and pointed at
# sessions/_compiler/_refactor/, which does not exist. Combined with the silent
# except below, that produced an EMPTY whitelist and check 5 flagged every
# deliberate technical term — weight, neuron, bias — as untranslated English.
_SESSIONS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TERMS_FILE = os.path.join(_SESSIONS, '_refactor', 'zh_terms.yaml')


class WhitelistError(RuntimeError):
    pass


def _load_whitelist(path=None):
    """Technical terms allowed to stay in Latin script inside Chinese prose.

    Kept in sessions/_refactor/zh_terms.yaml so it is a curriculum decision, not a
    gate implementation detail, and so a module can extend it.

    RAISES rather than returning an empty set. An empty whitelist does not make
    check 5 strict, it makes it LIE: every deliberate English term reads as
    untranslated prose, the warning fills with noise, and the one thing the check
    exists to find — a paragraph nobody translated — is invisible in the middle of
    it. Found exactly that way: a path one dirname short plus a bare `except:
    pass`.
    """
    p = path or _TERMS_FILE
    if not os.path.exists(p):
        raise WhitelistError('term whitelist not found: %s' % p)
    import yaml
    data = yaml.safe_load(open(p, encoding='utf-8')) or {}
    terms = set()
    for group in (data.get('terms') or {}).values():
        for t in (group or []):
            terms.add(str(t).lower())
    for t in (data.get('extra') or []):
        terms.add(str(t).lower())
    if len(terms) < 20:
        raise WhitelistError('term whitelist looks empty or malformed (%d terms in %s)'
                             % (len(terms), p))
    return terms


# --- source slicing ----------------------------------------------------------
def _strip_frontmatter(src):
    if src.startswith('---'):
        end = src.find('\n---', 3)
        if end > 0:
            return src[end + 4:], src[:end]
    return src, ''


def _blocks(body):
    """[(kind, args_text, block_text)] for each @@@ block."""
    out, cur = [], None
    for line in body.split('\n'):
        if line.startswith('@@@ '):
            if cur:
                out.append(cur)
            parts = line[4:].split(None, 1)
            cur = [parts[0], parts[1] if len(parts) > 1 else '', []]
        elif cur is not None:
            cur[2].append(line)
    if cur:
        out.append(cur)
    return [(k, a, '\n'.join(ls)) for k, a, ls in out]


def _fences(text):
    """[(start_line, end_line, body)] for each ~~~zh fence."""
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip().startswith('~~~zh'):
            j = i + 1
            buf = []
            while j < len(lines) and lines[j].strip() != '~~~':
                buf.append(lines[j]); j += 1
            out.append((i, j, '\n'.join(buf)))
            i = j + 1
        else:
            i += 1
    return out


def _declares_chinese(src):
    body, fm = _strip_frontmatter(src)
    # `_LANG_ZH_RE` rather than a literal `'class="lang-zh"' in body`: a day whose only
    # Chinese used single quotes or lived escaped inside a JS string would otherwise
    # declare nothing, and the whole gate would go inert on a half-translated page.
    # Widening is additive here — the corpus contains 0 non-canonical forms today.
    return bool(_fences(body)) or bool(re.search(r'(?m)^zh_\w+\s*:', fm)) \
        or 'zh_tag=' in body or 'zh_title=' in body or bool(_LANG_ZH_RE.search(body))


def _quiz_rows(text):
    """Quiz/warmup rows keyed by their answer index, in order."""
    rows = []
    for ln in text.split('\n'):
        s = ln.strip()
        m = re.search(r'\|\s*a\s*:\s*(\d+)\s*\|', s)
        if s.lower().startswith('q:') and m:
            rows.append((int(m.group(1)), s))
    return rows


class ManifestError(Exception):
    """A declared bilingual requirement could not be read. Never swallowed."""


def _load_requirement(source_path):
    """Return (required: bool|None, why: str) for the day at `source_path`.

    None means "no module-level declaration" — the day keeps the historical inert
    behaviour. True means the module's manifest names this day as one that must be
    bilingual, so having no Chinese is a FAILURE rather than a pass.

    Reads `zh.require` from `<module>/_refactor/manifest.yaml`: either the string
    `all`, or a list of day-directory names. `zh.scope` is deliberately NOT read —
    it is free prose in every existing manifest and nothing machine-readable could
    be derived from it, which is precisely how a module came to declare itself
    bilingual while shipping entirely English with a green board.
    """
    if not source_path:
        return None, 'no source path supplied, so no manifest could be located'
    day = os.path.basename(os.path.dirname(os.path.abspath(source_path)))
    mod = os.path.dirname(os.path.dirname(os.path.abspath(source_path)))
    mf = os.path.join(mod, '_refactor', 'manifest.yaml')
    if not os.path.exists(mf):
        return None, 'module has no _refactor/manifest.yaml, so nothing is declared'
    try:
        import yaml
        data = yaml.safe_load(open(mf, encoding='utf-8').read()) or {}
    except Exception as e:
        # Raise, never return None. A parse error that silently downgraded to
        # "nothing declared" would turn this whole check into the inert pass it
        # exists to remove — the same failure shape as the whitelist loader,
        # which used to `except: pass` into an empty set and made check 5 lie.
        raise ManifestError('%s: %s' % (mf, e))
    zh = data.get('zh')
    if not isinstance(zh, dict):
        return None, 'manifest has no zh: block, so bilingual output is not declared'
    langs = zh.get('langs') or []
    if 'zh' not in langs:
        return None, "manifest zh.langs does not include 'zh'"
    req = zh.get('require')
    if req is None:
        return None, ("manifest declares zh.langs %s but has no machine-readable "
                      "zh.require, so nothing can be enforced" % (langs,))
    if req == 'all':
        return True, "manifest zh.require is 'all'"
    if isinstance(req, (list, tuple)):
        if day in req:
            return True, 'manifest zh.require names this day'
        return False, ('manifest zh.require covers %d other day(s) but not this one'
                       % len(req))
    raise ManifestError('%s: zh.require must be "all" or a list of day-dir names, '
                        'got %r' % (mf, req))


def run(source_text, whitelist=None, manifest_covers=None, source_path=None,
        enforce_declaration=True):
    msgs, ok = [], [True]
    def fail(m): ok[0] = False; msgs.append('FAIL ' + m)
    def pas(m): msgs.append('pass ' + m)
    def note(m): msgs.append('warn ' + m)

    body, fm = _strip_frontmatter(source_text)

    # ---- 0. no U+FFFD, checked on EVERY day, Chinese or not -------------------
    # A literal replacement character means a byte sequence was mangled on its way
    # into the file, and it ships straight to the reader as a black diamond. Found
    # 22 of them in one freshly translated day and 21 in an already-PUBLISHED one,
    # plus 3 in English-only prose (an em-dash in m02/day-04) — so this check must
    # run before the Chinese early-return, not inside it. The file still decodes as
    # valid UTF-8, which is why nothing else caught it: the corruption is a
    # perfectly well-formed encoding OF the replacement character.
    if '�' in source_text:
        sites = [source_text[max(0, m.start() - 16):m.end() + 12].replace('\n', ' ')
                 for m in re.finditer('�+', source_text)]
        fail('%d U+FFFD replacement character(s) — text was corrupted on the way in '
             'and will render as a black diamond. First: %r'
             % (source_text.count('�'), sites[0]))
    else:
        pas('no U+FFFD corruption')

    if not _declares_chinese(source_text):
        # ---- 0b. a DECLARED bilingual day with no Chinese is a failure ---------
        # Until this existed, the branch below was an unconditional pass, so a
        # module could set zh.langs: [en, zh] and ship every day in English with a
        # green board — silence was indistinguishable from success. The inertness
        # itself is load-bearing (it let the toggle reach 293 pages without
        # touching content), so it is kept for undeclared modules and REMOVED only
        # where a manifest has explicitly opted in.
        required, why = _load_requirement(source_path)
        if required and not enforce_declaration:
            # ADVISORY here by caller's request. An untranslated day is a
            # ROLLOUT-COMPLETENESS problem, not a page-validity one: the page is
            # valid and renders correctly in English. compile_lesson.py passes
            # enforce_declaration=False because failing the compile deadlocks
            # authoring — the English author cannot satisfy a Chinese finding, so
            # every fix round burns and the lesson never converges, which in turn
            # prevents the translate phase that would have fixed it. Enforcement
            # lives at the CLI (exit 6) and in the published-corpus test.
            note('this day carries no Chinese and its module declares it must (%s). '
                 'ADVISORY at compile time; it is a hard failure at the CLI and in '
                 'the publish gate, so it cannot ship untranslated.' % why)
        elif required:
            fail('this day carries NO Chinese, but its module declares it must '
                 '(%s). An English-only day inside a declared-bilingual module is '
                 'the silence this check exists to remove: every other check below '
                 'is inert without a twin, so the board would go green on an '
                 'untranslated page. Either translate the day, or narrow zh.require '
                 'in the module manifest to a list that excludes it.' % why)
        elif required is False:
            msgs.append('n/a  this day declares no Chinese and its module '
                        'deliberately excludes it (%s).' % why)
        else:
            msgs.append('n/a  this day declares no Chinese yet — nothing to check. '
                        'The CSS fallback shows English, and the 中文 button is '
                        'disabled. NOT ENFORCED: %s.' % why)
        # ok[0], not a hardcoded True: check 0 runs above this return and applies to
        # English-only days too, so returning True here would discard its verdict.
        return ok[0], msgs
    required, why = _load_requirement(source_path)
    if required:
        pas('module declares this day bilingual (%s)' % why)
    elif required is None:
        note('this day IS bilingual, but no module manifest requires it (%s) — so '
             'nothing would have caught it being dropped. Add zh.require to the '
             'module manifest.' % why)
    terms = whitelist if whitelist is not None else _load_whitelist()

    # ---- 1. every concept carries Chinese, and no span is left untwinned ----
    concepts = [(a, t) for k, a, t in _blocks(body) if k == 'concept']
    regions = [(a, t) for k, a, t in _blocks(body) if k == 'region']

    # THIS CHECK USED TO FAIL OPEN ON A WHOLE MODULE. It iterates `@@@ concept`
    # blocks, so on m01 — six `mode: exemplar` days built from 14 `@@@ region` blocks
    # and ZERO concepts — the loop never ran and the gate printed "pass all 0 concept
    # units carry Chinese". Six fully translated days had their prose parity certified
    # by a message that had examined nothing. Check 1r below is the region-mode half.
    # Neither half is allowed to print a pass when it had no input to look at.
    if concepts:
        bare = []
        for args, text in concepts:
            m = re.search(r'id=(\S+)', args)
            cid = m.group(1) if m else '?'
            if not _fences(text):
                bare.append(cid)
        if bare:
            fail('%d concept unit(s) have no Chinese at all: %s. A page that is Chinese in '
                 'places and English in others is worse than either — finish the unit or '
                 'remove its Chinese.' % (len(bare), ', '.join(bare)))
        else:
            pas('all %d concept units carry Chinese' % len(concepts))

        # a trailing prose span with no fence after it shows in BOTH languages
        unpaired = []
        for args, text in concepts:
            cid = re.search(r'id=(\S+)', args).group(1) if re.search(r'id=(\S+)', args) else '?'
            fences = _fences(text)
            if not fences:
                continue
            tail = '\n'.join(text.split('\n')[fences[-1][1] + 1:])
            tail = re.sub(r'(?ms)^%%%.*?^%%%\s*$', '', tail)      # widgets are not prose
            if v8lib.text_weight(tail.strip()) > 80:
                unpaired.append((cid, v8lib.text_weight(tail.strip())))
        if unpaired:
            fail('%d concept unit(s) end with prose that has no Chinese twin, so it shows in '
                 'BOTH languages: %s (weight in English-char equivalents). Close the span '
                 'with a ~~~zh fence.'
                 % (len(unpaired), ', '.join('%s:%d' % u for u in unpaired)))
        else:
            pas('no concept ends with an untwinned prose span')
    elif not regions:
        # Chinese is declared but there is no unit of ANY kind to hang it on. Never
        # silently pass: say out loud that the check had nothing to read.
        note('this day declares Chinese but has neither @@@ concept nor @@@ region '
             'blocks — prose parity UNCHECKED this run (no unit to measure).')

    # ---- 1r. region-mode days: the same question, asked of raw HTML -----------
    # A region is pasted into the page BYTE-FOR-BYTE (v8lib.compile_html) and never
    # goes through render_md, so its bilingual markup is hand-written as paired
    # `class="lang-en"` / `class="lang-zh"` nodes. Three ways that goes wrong, none of
    # them visible to checks 1-6:
    #   a. an UNPAIRED node. A node with NEITHER class shows under both languages, so
    #      an untouched page degrades to English safely — but a node explicitly marked
    #      `lang-en` with no `lang-zh` twin is display:none for a Chinese reader. It
    #      does not fall back. It VANISHES. The mirror case shows Chinese to an
    #      English reader.
    #   b. a `~~~zh` FENCE inside a region. Because the region is verbatim, the fence
    #      is not a fence: the literal characters `~~~zh` and the Chinese after them
    #      ship to the reader as visible text.
    #   c. a region of real prose with no Chinese in it at all — an untranslated
    #      section that reads as finished, because English shows through.
    if regions:
        tot_en = tot_zh = 0
        skew, fenced, untranslated = [], [], []
        for args, text in regions:
            m = re.search(r'name=(\S+)', args)
            name = m.group(1) if m else '?'
            en = len(_LANG_EN_RE.findall(text))
            zh = len(_LANG_ZH_RE.findall(text))
            tot_en += en
            tot_zh += zh
            if en != zh:
                skew.append('%s: %d en vs %d zh' % (name, en, zh))
            if _fences(text):
                fenced.append(name)
            # A region long enough to be reader-facing prose must carry SOME Chinese.
            # Threshold measured on the real corpus: across m01's 84 regions the
            # shortest one that is genuinely prose is `fin` at 166 characters of text,
            # and the longest that CANNOT be bilingual is `title` (a <title> element,
            # and a browser tab cannot show two) at 57. 200 sits in that gap with room
            # on both sides.
            if not zh and len(_visible_text(text)) > 200:
                untranslated.append('%s (%d chars)' % (name, len(_visible_text(text))))

        # (a) balance. Both numbers are reported whatever the verdict — the point of
        # the check is the pair of counts, not just the boolean. Localised per region
        # as well, because equal TOTALS can still hide two skewed regions that cancel
        # each other out (+1 here, -1 there), which is exactly a fail-open path.
        if tot_en != tot_zh or skew:
            fail('region language classes are unbalanced: %d class="lang-en" vs %d '
                 'class="lang-zh" across %d region(s)%s. An unpaired lang-en node is '
                 'display:none for a Chinese reader — it VANISHES, it does not fall back '
                 'to English; an unpaired lang-zh node shows Chinese to an English reader.'
                 % (tot_en, tot_zh, len(regions),
                    ('; skewed: ' + ', '.join(skew[:6])) if skew else ''))
        else:
            pas('region language classes balance: %d class="lang-en" vs %d '
                'class="lang-zh" across %d region(s)' % (tot_en, tot_zh, len(regions)))

        # (b) a fence cannot work inside a verbatim region
        if fenced:
            fail('%d region(s) contain a ~~~zh fence: %s. A region is pasted into the page '
                 'byte-for-byte and never rendered, so the fence is not a fence — the '
                 'literal text "~~~zh" ships to the reader. Use paired '
                 '<p class="lang-en"> / <p class="lang-zh"> nodes instead.'
                 % (len(fenced), ', '.join(fenced)))
        else:
            pas('no region hides a ~~~zh fence that would ship as literal text')

        # (c) a whole region nobody translated
        if untranslated:
            fail('%d region(s) hold reader-visible prose with no class="lang-zh" node at '
                 'all: %s. English shows through, so the page looks finished and teaches a '
                 'Chinese reader nothing in that section.'
                 % (len(untranslated), ', '.join(untranslated)))
        else:
            pas('every region over 200 characters of prose carries Chinese')

    # ---- 2. SVG labels ------------------------------------------------------
    en_lab = re.findall(r'<text class="lang-en"[^>]*>(.*?)</text>', body, re.S)
    zh_lab = re.findall(r'<text class="lang-zh"[^>]*>(.*?)</text>', body, re.S)
    # A label needs a twin only if it says something a Chinese reader could not read.
    # Two exemptions, both measured against real drawings:
    #   * no Latin word of 3+ letters — "N = 7 000 000 000", "5 x 3", a tick
    #   * every word in it is a WHITELISTED TERM — a label that is just "ReLU" or
    #     "softmax" must STAY English by the term policy, so demanding
    #     <text class="lang-zh">ReLU</text> would add a duplicate node that teaches
    #     nothing and then require it forever.
    def _needs_twin(t):
        words = _NEEDS_WORDS.findall(t)
        return bool(words) and not all(w.lower() in terms for w in words)
    plain = [t for t in re.findall(r'<text(?![^>]*class="lang-)[^>]*>(.*?)</text>', body, re.S)
             if _needs_twin(t)]
    if len(en_lab) != len(zh_lab):
        fail('SVG labels are unbalanced: %d <text class="lang-en"> vs %d "lang-zh". Every '
             'paired label needs both halves or one language loses a caption.'
             % (len(en_lab), len(zh_lab)))
    elif plain:
        fail('%d SVG label(s) with real words have no Chinese twin, e.g. %r. A drawing is '
             'shared between the languages, so each worded label needs a paired '
             '<text class="lang-en"> / <text class="lang-zh">. Symbol- and number-only '
             'labels are exempt automatically.' % (len(plain), plain[0].strip()[:60]))
    else:
        pas('%d paired SVG label(s); %d symbol-only label(s) exempt'
            % (len(en_lab), len(re.findall(r'<text(?![^>]*class="lang-)', body)) ))

    # ---- 3. front-matter twins ---------------------------------------------
    missing_fm = [k for k in _VISIBLE_FM
                  if re.search(r'(?m)^%s\s*:' % k, fm) and not re.search(r'(?m)^zh_%s\s*:' % k, fm)]
    if missing_fm:
        fail('front-matter keys the reader SEES have no Chinese twin: %s (add zh_%s). '
             'page_title is exempt — a browser tab cannot show two titles.'
             % (', '.join(missing_fm), missing_fm[0]))
    else:
        pas('every reader-visible front-matter key has its zh_ twin')

    # ---- 4. quiz answer indices --------------------------------------------
    bad_quiz = []
    for kind, args, text in _blocks(body):
        for f_start, f_end, f_body in _fences(text):
            before = '\n'.join(text.split('\n')[:f_start])
            en_rows = _quiz_rows(before)
            zh_rows = _quiz_rows(f_body)
            if not en_rows and not zh_rows:
                continue
            if len(en_rows) != len(zh_rows):
                bad_quiz.append('%s: %d English question(s) vs %d Chinese'
                                % (kind, len(en_rows), len(zh_rows)))
                continue
            for n, (e, z) in enumerate(zip(en_rows, zh_rows), 1):
                if e[0] != z[0]:
                    bad_quiz.append('%s q%d: English answer a:%d but Chinese a:%d'
                                    % (kind, n, e[0], z[0]))
    if bad_quiz:
        fail('quiz twins disagree — %s. This is a CORRECTNESS defect, not a formatting '
             'one: a Chinese reader would be told the wrong option is right.'
             % '; '.join(bad_quiz[:4]))
    else:
        pas('quiz twins agree on every answer index')

    # ---- 5. untranslated English hiding inside the Chinese ------------------
    leaks = {}
    for _s, _e, f_body in _fences(body):
        t = re.sub(r'`[^`]*`', ' ', f_body)                 # code spans stay English
        t = re.sub(r'<[^>]+>', ' ', t)                      # tags/attrs
        t = re.sub(r'\[\[([^\|\]]+)\|\|[^\]]*\]\]', r'\1', t)   # keep the term, drop the gloss
        t = re.sub(r'(?m)^\s*(code|out|src|expr):.*$', ' ', t)
        # AUTHORING MARKERS are not prose, and flagging them buries the one thing
        # this check exists to find — a paragraph nobody translated. Masked:
        #   `%%% steps` / `!!! c-warn`  widget and callout fence lines
        #   `step:` / `why:` / `take:`  the ASCII field openers _kv recognises
        #   `concept: xor-limit`        a spaced-repetition id
        t = re.sub(r'(?m)^\s*%%%.*$', ' ', t)
        t = re.sub(r'(?m)^\s*!!!.*$', ' ', t)
        t = re.sub(r'(?m)^\s*[A-Za-z_]\w*:', ' ', t)
        t = re.sub(r'\bconcept:\s*\S+', ' ', t)
        for w in re.findall(r'[A-Za-z][A-Za-z\-\'/]{2,}', t):
            if w.lower() not in terms:
                leaks[w] = leaks.get(w, 0) + 1
    if leaks:
        worst = sorted(leaks.items(), key=lambda kv: -kv[1])[:6]
        note('%d Latin word(s) in the Chinese are not in the term whitelist: %s. Either '
             'translate them or add real technical terms to '
             'sessions/_refactor/zh_terms.yaml.'
             % (len(leaks), ', '.join('%s x%d' % w for w in worst)))
    else:
        pas('no untranslated English left inside the Chinese')

    # ---- 6. manifest coverage reachable in Chinese --------------------------
    if manifest_covers:
        zh_text = ' '.join(f for _a, _b, f in _fences(body))
        shared = ' '.join(re.findall(r'<text[^>]*>(.*?)</text>', body, re.S))
        reach = (zh_text + ' ' + shared).lower()
        gaps = [t for t in manifest_covers
                if not any(k.lower() in reach for k in ([t] if isinstance(t, str) else t))]
        if gaps:
            fail('%d manifest topic(s) are unreachable from the Chinese reading path: %s. '
                 'Breadth must hold in both languages, not just English.'
                 % (len(gaps), ', '.join(map(str, gaps[:5]))))
        else:
            pas('all %d manifest topic(s) reachable in Chinese' % len(manifest_covers))
    else:
        msgs.append('warn no manifest coverage list supplied — breadth parity UNCHECKED '
                    'this run (never a silent pass)')

    return ok[0], msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--audit', action='store_true',
                    help='report only; always exit 0')
    ap.add_argument('--terms', help='path to zh_terms.yaml (default: sessions/_refactor/zh_terms.yaml)')
    a = ap.parse_args()
    ok, msgs = run(open(a.source, encoding='utf-8').read(),
                   whitelist=_load_whitelist(a.terms) if a.terms else None,
                   source_path=a.source)
    print('== Language Parity Gate:', os.path.relpath(a.source), '==')
    for m in msgs:
        print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if (ok or a.audit) else 6)


if __name__ == '__main__':
    main()
