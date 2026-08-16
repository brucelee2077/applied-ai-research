#!/usr/bin/env python3
# =============================================================================
# Beginner Language Gate (v9) — deterministic floor for the FOUR beginner axes.
# =============================================================================
# Why this exists (user directive 2026-08-05): the four things that matter most
# for a first-time learner are SIMPLE LANGUAGE, CURIOSITY, VISUALIZATION, and
# 12-YEAR-OLD ANALOGIES. Three of those already had an always-on enforcer
# (interest floor, concept_structure/shell gates, the concept-structure judge's
# analogy axis). SIMPLE LANGUAGE had NONE: `plain_language` lives only in
# coverage_judge.judge_tone, which early-returns N/A when notebook_yardstick is
# null — true for 9 of the 20 shipped m02-m04 days and for most future modules.
# lesson_build.js's tone lens then emitted neither a finding NOR an "unenforced"
# note, so it read as a clean pass. This gate closes that hole, plus the three
# other stated-but-unchecked rules found in the same audit.
#
# It is DETERMINISTIC and offline on purpose: no bridge, no notebook, cannot
# drift, and cannot fail open. The LLM judges remain the semantic enforcers —
# this is the cheap floor under them.
#
# Four checks:
#   1. plain language  — banned dismissive phrases + idioms (repo CLAUDE.md s5),
#                        and run-on sentences. Markup/code/SVG text is EXCLUDED,
#                        because penalising a word inside a drawing punishes the
#                        author for obeying the visual rules.
#   2. real play       — >=1 genuinely interactive widget (a %%% viz embed, or an
#                        in-body slider/checkbox). Accepts EITHER form: m02 ships
#                        play as inline sliders, m03 as viz iframes.
#   3. demo honesty    — a %%% demo only un-hides a pre-baked answer, so an
#                        author-supplied label must not promise execution.
#                        (v8lib's DEFAULT label is already "reveal"; 48 of 103
#                        author labels in m02-m04 override it with "run".)
#   4. digestibility   — no unbroken MAIN-LINE prose wall. A long "Optional
#                        (skippable)" aside is already correctly placed by the
#                        math-restraint rule and is NOT reported here; promoting
#                        it into main prose would be the wrong fix.
#
# Plus one escape-hatch guard: `notebook_yardstick: null` while a real notebook
# exists silently disables the tone + interest-ceiling judges.
#
# Reusable:  from beginner_language_gate import run ; ok, msgs = run(source_text)
# CLI:       python3 gates/beginner_language_gate.py <source.md>   (exit 0/5)
# =============================================================================
import sys, os, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from v8lib import CJK_RE as _CJK_RE
except Exception:                                    # pragma: no cover
    _CJK_RE = re.compile(r'[一-鿿]')

# --- axis 1 lexicons (repo CLAUDE.md s5) ------------------------------------
# Dismissive: tells the reader their confusion is illegitimate.
_BANNED = [
    'as you can see', 'trivially', 'obviously', 'recall that',
    'it is left as an exercise', 'this is just', 'clearly,', 'of course,',
    'needless to say', 'it should be obvious',
]
# Idioms: opaque to a reader whose first language is not English.
_IDIOMS = [
    'under the hood', 'out of the box', 'rule of thumb', 'ballpark',
    'a piece of cake', 'in a nutshell', 'the elephant in the room',
    'bread and butter', 'low-hanging fruit',
]
# --- axis 1, Chinese ---------------------------------------------------------
# Chinese was measured to fail open on two of the four axes: the English lists hit
# nothing, `.lower()` is a no-op on Han, and `[^.!?\n]+` + `.split()` scored a
# 51-character Chinese sentence as ONE word in ONE sentence, so the run-on check
# was provably dead. These are the Chinese equivalents of the same two rules.
_ZH_BANNED = [
    '显然', '很显然', '不难看出', '不难发现', '众所周知', '不言而喻', '如你所见',
    '一目了然', '稍加思考', '留给读者', '读者自证', '自不必说', '无需多言', '这只是',
]
# 成语 and set literary phrases. For a 12-year-old these are the Chinese analogue of
# an English idiom: four characters that assume a shared cultural reference the
# reader may not have, and that cannot be worked out from the parts.
_ZH_IDIOMS = [
    '一举两得', '举一反三', '水到渠成', '事半功倍', '牵一发而动全身', '一石二鸟',
    '提纲挈领', '化繁为简', '不胜枚举', '大同小异', '一蹴而就', '立竿见影',
    '顺理成章', '相辅相成', '一劳永逸', '万变不离其宗',
]
# Chinese verbs that oversell a %%% demo, which only un-hides a pre-baked answer.
# Verified: the English `\b(run|execute|compute)\b` cannot match 运行一下 — no word
# boundaries exist in Chinese — so demo honesty passed on any Chinese label.
_ZH_RUNNY = ['运行', '执行', '跑一下', '跑一遍', '计算出', '算出来', '真的算']

_MAX_SENTENCE_WORDS = 45   # generous: flags genuine run-ons, not normal prose
# Chinese sentences are measured in CHARACTERS, split on 。！？；— not on . ! ?
# A further rule is needed that English does not need: Chinese writers legitimately
# chain clauses with 逗号, so a "sentence" can be one idea or six. Measured on a real
# 58-character sample, splitting on 。！？； alone still yielded ONE segment. So a
# comma-chain is counted too.
_MAX_ZH_SENTENCE_CHARS = 60
_MAX_ZH_COMMAS = 4

# A run of SHORT list items joined by `、` — numbers, identifiers, or one-to-three
# character words: （1.0、0.1、0.01、0.001） or 「太大胆、太胆小」. Such a run is one
# idea, so its separators must not count toward the clause-chain limit. Anchored on
# item shape rather than on the surrounding brackets, because the same enumeration
# appears bare in prose as often as it appears parenthesised.
_ENUM_RE = re.compile(
    r'(?:[0-9A-Za-z_.·%\-]{1,10}|[一-鿿]{1,3})'
    r'(?:、(?:[0-9A-Za-z_.·%\-]{1,10}|[一-鿿]{1,3})){2,}')
_MAX_MAIN_WALL = 600       # matches _density_scan.py's walls_over_600 threshold
                           # (both are ENGLISH-CHARACTER EQUIVALENTS — see
                           #  v8lib.text_weight, which _density_scan now applies)

_INTERACTIVE_SRC = re.compile(r'(?m)^%%%\s+viz\b')
_INTERACTIVE_INLINE = re.compile(r'<input[^>]+type="(range|checkbox)"', re.I)
_DEMO_HDR = re.compile(r'(?m)^%%%\s+demo\b([^\n]*)')
_LABEL = re.compile(r'label="([^"]*)"')
_RUNNY = re.compile(r'\b(run|runs|running|execute|executes|compute|computes)\b', re.I)


def _strip_frontmatter(src):
    if src.startswith('---'):
        end = src.find('\n---', 3)
        if end > 0:
            return src[end + 4:]
    return src


def _mask(s, pattern, flags=0):
    """Blank out a region, PRESERVING LENGTH so offsets stay meaningful.

    flags defaults to 0 — NEVER re.S by default. A line-anchored pattern like
    `^@@@.*$` combined with DOTALL matches from the first marker to the END of
    the file and silently blanks the whole lesson (which is exactly what it did
    on the first run of this gate: every check passed vacuously).
    """
    return re.sub(pattern, lambda m: ' ' * len(m.group(0)), s, flags=flags)


def _reader_visible_words(body):
    """EVERYTHING a human reads on the page — for the banned-phrase / idiom check.

    Distinct from _prose_only on purpose. _prose_only masks `step:`/`why:` payloads,
    SVG `<text>` labels and `[[gloss||bodies]]` because those are chunked or non-prose
    and must not be measured as WALLS or as run-on SENTENCES. But the reader still READS
    every one of them, so an idiom hiding in a `why:` rung or an SVG label excludes a
    second-language reader exactly as much as one in a paragraph. Measured on a real
    authored unit: the masked regions were 398 of 833 reader-visible words (48%), and
    three idioms sat inside them — so checking vocabulary on _prose_only alone inspects
    barely half the page.

    Keeps: paragraph prose, step:/why: rungs, take:/cap:/predict: lines, SVG text
    content, gloss bodies. Drops: code, tags/attributes, widget fences, quiz rows.
    """
    t = _mask(body, r'(?m)^\s*(code|out|src):.*$', re.M)     # code + its output
    t = _mask(t, r'`[^`]*`')                                  # inline code spans
    t = _mask(t, r'(?m)^.*\|\s*a:\d+\s*\|.*$', re.M)          # quiz / recall rows
    # SVG: keep the TEXT the reader sees, drop the markup around it.
    t = re.sub(r'<text\b[^>]*>', ' ', t)
    t = re.sub(r'</text>', ' ', t)
    t = _mask(t, r'<[^>]+>', re.S)                            # all remaining tags/attrs
    t = re.sub(r'\[\[([^\|\]]+)\|\|([^\]]*)\]\]', r'\1 \2', t)  # term AND its gloss
    t = _mask(t, r'(?m)^%%%.*$', re.M)
    t = _mask(t, r'(?m)^@@@.*$', re.M)
    t = re.sub(r'(?m)^\s*(why|step|take|cap|predict|q|concept|label):', ' ', t)
    return t


def _prose_only(body):
    """Reader-facing WORDS: no markup, no code, no drawings, no quiz options.

    Quiz/recall rows are masked wherever they appear, keyed off the unambiguous
    ` | a:<digit> | ` answer syntax rather than the enclosing block type — a
    lesson carries several quiz-shaped blocks (the day quiz plus spaced-repetition
    recall), and masking only `%%% quiz` left the recall rows behind, which then
    read as a single 56-word "run-on sentence".

    `step:` / `why:` / `q:` payload lines are masked too: they are already
    one-idea chunks by construction (a %%% steps rung), so measuring them as
    free prose punishes the author for using the prescribed chunking widget.
    """
    t = _mask(body, r'<svg[\s>].*?</svg>', re.S)     # drawings (multi-line: needs re.S)
    t = _mask(t, r'(?ms)^%%%\s+prompt\b.*?^%%%\s*$')   # produce/artifact instructions
    t = _mask(t, r'(?m)^.*\|\s*a:\d+\s*\|.*$', re.M)  # any quiz / recall row
    t = _mask(t, r'(?m)^\s*(code|out|src|cap|predict|take|label|step|why|q|concept):.*$', re.M)
    t = _mask(t, r'(?m)^\s*\|.*$', re.M)             # table rows
    t = _mask(t, r'`[^`]*`')                         # code spans
    t = _mask(t, r'<[^>]+>', re.S)                   # any other html tag
    # A gloss body is a TOOLTIP — it is hover-only, so it must not count toward the
    # main-line sentence length. `[^\]]*` could not cross a `]` INSIDE the gloss, so
    # a gloss like `[[epsilon clipping||clamp to [ε, 1−ε] before the log]]` failed to
    # match at all and its whole body was measured as main-line prose. That produced
    # phantom "sentence over 60 汉字" reports on m02/day-04 that authors then
    # "fixed" by splitting sentences that were never long. Allow a single `]`, stop
    # only at `]]`.
    t = _mask(t, r'\[\[[^\|\]]+\|\|(?:[^\]]|\](?!\]))*\]\]')   # gloss bodies (tooltips)
    t = _mask(t, r'(?m)^%%%.*$', re.M)               # widget fences/headers
    t = _mask(t, r'(?m)^@@@.*$', re.M)               # block headers
    t = _mask(t, r'(?m)^@\w+.*$', re.M)              # @lede / @goal markers
    return t


def _aside_spans(body):
    """Character spans of `!!! ... !!!` callout boxes (incl. Optional-math boxes)."""
    return [(m.start(), m.end()) for m in re.finditer(r'(?ms)^!!!.*?(?:\n!!!|\Z)', body)]


def _longest_main_wall(body):
    """Longest MAIN-LINE prose wall, via _density_scan's own tested functions.

    Uses _density_scan's `concept_blocks` + `buildup_of` + `longest_wall` rather
    than re-slicing here, so this gate reports the SAME number the established
    metric does. Two earlier attempts disagreed with it — first by 4x (a private
    wall regex), then by measuring intro prose the established metric excludes
    (14 days "failed" that _density_scan scores as 0). Matching the calibrated
    tool is the point: two disagreeing measures of one thing sends authors
    chasing walls that do not exist.

    Returns None if the module cannot be imported, so the caller reports the wall
    as UNMEASURED rather than silently passing.
    """
    try:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, '..', '..', '_density_scan.py')
        spec = importlib.util.spec_from_file_location('_density_scan_ro', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)      # module level only; main() is __main__-guarded
        walls = [mod.longest_wall(mod.buildup_of(b)) for _a, b in mod.concept_blocks(body)]
        return max(walls, default=0)
    except Exception:
        return None


def _find_companion_notebook(source_path):
    """Best-effort: is there a notebook on disk that this day could use as a yardstick?

    The gate used to take `notebook_exists` as a caller-supplied boolean defaulting
    to False, and the CLI never looked at the filesystem — so the documented command
    printed "pass … no companion notebook" on m04 day-06, which has a real 263 KB
    notebook sitting unused. That is a FALSE PASS on the exact case the check exists
    for. Now the CLI resolves it itself.

    Conservative on purpose: a sibling day in the same module declaring a yardstick
    is strong evidence the module HAS notebooks, which is the actual signal (m04
    day-05 declares 10_pytorch_equivalent.ipynb; day-06 nulls it).
    Returns a path string, or None.
    """
    if not source_path:
        return None
    try:
        day_dir = os.path.dirname(os.path.abspath(source_path))
        module_dir = os.path.dirname(day_dir)
        for sib in sorted(glob.glob(os.path.join(module_dir, 'day-*', 'source.md'))):
            if os.path.abspath(sib) == os.path.abspath(source_path):
                continue
            head = open(sib, encoding='utf-8').read(4000)
            m = re.search(r'(?m)^notebook_yardstick:\s*(?!null|none)(\S+)\s*$', head)
            if m:
                root = os.path.dirname(os.path.dirname(module_dir))  # repo root
                cand = os.path.join(root, m.group(1).strip().strip('"\''))
                if os.path.exists(cand):
                    return cand
    except Exception:
        return None
    return None


def run(source_text, notebook_exists=False, source_path=None):
    msgs, ok = [], [True]
    def fail(m): ok[0] = False; msgs.append('FAIL ' + m)
    def pas(m): msgs.append('pass ' + m)

    body = _strip_frontmatter(source_text)
    prose = _prose_only(body)
    asides = _aside_spans(body)

    # MAIN-LINE prose = reader prose with `!!!` callout boxes blanked out too.
    # Blanking the SPAN (rather than testing each match's start offset) is what
    # makes the aside exclusion correct: a wall/sentence match can BEGIN in main
    # prose and run into an aside, and an offset test on `.start()` alone lets
    # the whole flattened file through as one 700-char "main" wall.
    main_chars = list(prose)
    for a, b in asides:
        for i in range(a, min(b, len(main_chars))):
            main_chars[i] = ' '
    main_prose = ''.join(main_chars)

    # ---- 1. plain language ------------------------------------------------
    # Vocabulary is checked on EVERY word the reader sees (incl. step:/why: rungs and
    # SVG labels); walls + run-ons are measured on main-line prose only. Two views,
    # because they answer two different questions.
    hits = []
    visible = _reader_visible_words(body)
    low = visible.lower()
    for phrase in _BANNED + _IDIOMS:
        for m in re.finditer(re.escape(phrase), low):
            hits.append((phrase, m.start()))
    # Chinese: match on the UNLOWERED text — .lower() does nothing to Han, and
    # lowering is meaningless for these literals.
    for phrase in _ZH_BANNED + _ZH_IDIOMS:
        for m in re.finditer(re.escape(phrase), visible):
            hits.append((phrase, m.start()))
    if hits:
        shown = ', '.join(sorted({h[0] for h in hits}))
        fail('plain language: %d banned/idiomatic phrase(s) in reader-visible text '
             '(paragraphs, %%%% steps rungs, SVG labels, glosses) — %s '
             '(repo CLAUDE.md s5: say what you mean; do not tell the reader their '
             'confusion is illegitimate)' % (len(hits), shown))
    else:
        pas('no banned dismissive phrases or idioms in reader prose')

    # run-on sentences (main-line prose only; an Optional box may carry denser math)
    long_sents = []
    for m in re.finditer(r'[^.!?\n]+', main_prose):
        seg = m.group(0)
        words = [w for w in seg.split() if any(c.isalnum() for c in w)]
        if len(words) > _MAX_SENTENCE_WORDS:
            long_sents.append((len(words), ' '.join(words[:12])))
    # Chinese sentences: split on 。！？； and measure CHARACTERS, plus a comma-chain
    # rule. Both are needed — see the constants above for the measurements.
    zh_long = []
    for m in re.finditer(r'[^。！？；\n]+', main_prose):
        seg = m.group(0).strip()
        han = len(_CJK_RE.findall(seg))
        if han < 8:                       # not a Chinese sentence; the English rule owns it
            continue
        # `、` is the Chinese ENUMERATION comma, not a clause joint. Counting it as
        # one over-reported a 29-汉字 sentence on m02/day-08 as a 5-comma chain,
        # because 3 of the 5 were separators in the numeric list （1.0、0.1、0.01、
        # 0.001）— a list is ONE idea, however many items it has. The rule exists to
        # catch clause chaining, so only count a `、` whose neighbours look like
        # prose rather than list items.
        enum = _ENUM_RE.findall(seg)
        commas = seg.count('，') + seg.count('、') - sum(s.count('、') for s in enum)

        if han > _MAX_ZH_SENTENCE_CHARS:
            zh_long.append(('%d 汉字' % han, seg[:16]))
        elif commas > _MAX_ZH_COMMAS:
            zh_long.append(('%d 个逗号' % commas, seg[:16]))
    if zh_long:
        msgs.append('warn plain language (Chinese): %d main-line sentence(s) over %d 汉字 '
                    'or with more than %d commas (worst %s: "%s…") — one idea per sentence'
                    % (len(zh_long), _MAX_ZH_SENTENCE_CHARS, _MAX_ZH_COMMAS,
                       zh_long[0][0], zh_long[0][1]))
    if long_sents:
        worst = max(long_sents)
        # ADVISORY, not blocking: a colon-introduced list legitimately reads as one
        # long "sentence" to a splitter while reading as chunks on the page. Banned
        # phrases above ARE unambiguous and do block. Keeping this warn-only avoids
        # the failure mode that made `momentum` a permanently-ignored warning: a
        # check that fires on nearly every day teaches everyone to skip the output.
        msgs.append('warn plain language: %d main-line sentence(s) over %d words '
                    '(worst %d: "%s…") — prefer one idea per sentence'
                    % (len(long_sents), _MAX_SENTENCE_WORDS, worst[0], worst[1]))
    else:
        pas('no main-line run-on sentences')

    # ---- 2. real play ----------------------------------------------------
    n_viz = len(_INTERACTIVE_SRC.findall(body))
    n_inline = len(_INTERACTIVE_INLINE.findall(body))
    if n_viz + n_inline == 0:
        fail('real play: 0 genuinely interactive widgets — ship at least one '
             '"change something -> see it change" (a %%% viz embed or an in-body '
             'slider). A %%% demo only reveals a pre-baked answer and does not count')
    else:
        pas('real play: %d interactive widget(s) (%d viz + %d inline control)'
            % (n_viz + n_inline, n_viz, n_inline))

    # ---- 3. demo honesty -------------------------------------------------
    dishonest = []
    for m in _DEMO_HDR.finditer(body):
        lm = _LABEL.search(m.group(1))
        if lm and (_RUNNY.search(lm.group(1))
                   or any(v in lm.group(1) for v in _ZH_RUNNY)):
            dishonest.append(lm.group(1))
    if dishonest:
        fail('demo honesty: %d %%%% demo label(s) promise execution on a widget that '
             'only un-hides a pre-baked answer — e.g. "%s". Use a reveal verb'
             % (len(dishonest), dishonest[0]))
    else:
        pas('demo labels do not oversell what the widget does')

    # ---- 4. digestibility -------------------------------------------------
    # DELEGATE to _density_scan.longest_wall — the already-tested metric (+8 tests)
    # that knows a 4k-char inline <svg> is a picture, not a wall, and measures `!!!`
    # asides separately because they have a different cure. A second private
    # implementation here disagreed with it by 4x on m02 day-01 (1885 vs 472) and
    # would have sent authors chasing walls that do not exist.
    wall = _longest_main_wall(body)
    if wall is None:
        msgs.append('warn digestibility: _density_scan unavailable — wall UNMEASURED '
                    'this run (never a silent pass)')
    elif wall > _MAX_MAIN_WALL:
        fail('digestibility: main-line prose wall of %d chars (limit %d) — break it '
             'into one-idea chunks with %%%% steps, %%%% insight, or #### sub-beats. '
             'Chunk the density; do NOT cut coverage' % (wall, _MAX_MAIN_WALL))
    else:
        pas('no main-line prose wall over %d chars (worst %d)' % (_MAX_MAIN_WALL, wall))

    # ---- yardstick escape hatch ------------------------------------------
    fm = source_text[:source_text.find('\n---', 3)] if source_text.startswith('---') else ''
    nulled = re.search(r'(?m)^notebook_yardstick:\s*(null|none)\s*$', fm, re.I)
    found = _find_companion_notebook(source_path) if (nulled and not notebook_exists) else None
    if nulled and (notebook_exists or found):
        fail('yardstick: notebook_yardstick is null but a companion notebook EXISTS — '
             'that silently turns OFF the beginner-friendliness (tone) and interest-'
             'ceiling judges. Declare the notebook, or say why in the manifest%s'
             % (' — found: %s' % found if found else ''))
    elif nulled:
        # Do NOT claim "no companion notebook exists": this is a best-effort lookup,
        # so state what was actually checked.
        msgs.append('warn notebook_yardstick is null — no sibling day in this module '
                    'declares a notebook, so tone/plain_language + the interest CEILING '
                    'are N/A this run. Confirm no notebook exists before trusting that')
    else:
        pas('notebook_yardstick declared')

    return ok[0], msgs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--notebook-exists', action='store_true',
                    help='a companion notebook exists for this day (enables the yardstick guard)')
    a = ap.parse_args()
    ok, msgs = run(open(a.source, encoding='utf-8').read(),
                   notebook_exists=a.notebook_exists, source_path=a.source)
    for m in msgs:
        print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 5)


if __name__ == '__main__':
    main()
