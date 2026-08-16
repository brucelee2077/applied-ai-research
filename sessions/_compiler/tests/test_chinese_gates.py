import os, sys, glob, re, importlib.util
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import v8lib
import lang_parity_gate as parity
import beginner_language_gate as blg
import coverage_gate

REPO = os.path.join(HERE, '..', '..', '..')

FM = '---\nquest_id: t\ntitle: T\nnotebook_yardstick: null\n---\n\n'
SVG = '%%% svg\n<svg viewBox="0 0 9 9"><text x="1" y="1">L</text></svg>\n%%%'


# =============================================================================
# v8lib.text_weight — the one calibrated constant the length rules share
# =============================================================================

def test_english_text_weighs_its_own_length():
    assert v8lib.text_weight('a' * 300) == 300
    assert v8lib.text_weight('') == 0
    assert v8lib.text_weight(None) == 0


def test_han_weighs_about_2_3_english_characters():
    # Measured on the first authored EN/ZH twin: 240 EN chars vs 128 ZH, and
    # 363 vs 157 — solving ascii + w*cjk = len(EN) gives 2.03 and 2.58.
    assert v8lib.text_weight('数' * 100) == 230


def test_mixed_text_weighs_each_script_separately():
    # Every Chinese lesson is mixed: technical terms stay English by design, and
    # full-width punctuation counts as CJK because it takes a full column.
    assert v8lib.text_weight('abc数数') == 3 + round(2 * 2.3)
    # 'weight（权重）' = 6 ascii + 4 full-width/Han
    assert v8lib.text_weight('weight（权重）') == round(6 + 4 * 2.3)


def test_a_300_han_wall_now_exceeds_the_600_limit():
    # The concrete miscalibration this fixes: a 300-character Chinese wall used to
    # score 300 and pass a limit written for English.
    assert v8lib.text_weight('数' * 300) > blg._MAX_MAIN_WALL
    assert len('数' * 300) < blg._MAX_MAIN_WALL


# =============================================================================
# coverage_gate._norm — stop deleting Han
# =============================================================================

def test_norm_keeps_han():
    assert coverage_gate._norm('注意力 attention 头') == '注意力 attention 头'


def test_norm_is_unchanged_on_english():
    # Widening a keep-set can only ADD matches, so no English verdict can move.
    for s in ['Attention Is All You Need', 'kv-cache & MFU (0.4)', 'N = 7,000,000,000']:
        assert coverage_gate._norm(s) == re.sub(
            r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', s.lower())).strip()


# =============================================================================
# beginner_language_gate — the two axes that used to fail open on Chinese
# =============================================================================

def _run(body):
    return blg.run(FM + '@@@ concept id=c1 tag=t title=t\n' + body)


PLAY = '\n\n%%% viz src="x.html"\n%%%\n'


@pytest.mark.parametrize('phrase', ['很显然', '众所周知', '不言而喻', '一举两得', '事半功倍'])
def test_chinese_dismissive_phrases_and_idioms_are_caught(phrase):
    ok, msgs = _run('%s，这一步不用解释。' % phrase + PLAY)
    assert not ok
    assert any(m.startswith('FAIL') and 'banned' in m for m in msgs), msgs


def test_clean_chinese_prose_is_not_flagged():
    ok, msgs = _run('我们先数一数按了几次。这一步很简单。' + PLAY)
    assert any('pass no banned' in m for m in msgs), msgs


def test_a_long_chinese_sentence_is_reported():
    # Verified before this existed: a 51-character Chinese sentence measured as ONE
    # word in ONE sentence, so the English run-on check was provably dead.
    long_zh = '在给任何东西定价之前我们需要一个可以数的东西这就是我们要数的而且它比你想象的要小得多一次乘法或者一次加法就是全部了不是一整个求和也不是一行代码。'
    ok, msgs = _run(long_zh + PLAY)
    assert any('Chinese' in m and '汉字' in m for m in msgs), msgs


def test_a_chinese_comma_chain_is_reported():
    # Splitting on 。！？； alone is not enough — Chinese writers legitimately chain
    # clauses with 逗号, so one "sentence" can hold six ideas.
    chain = '我们先看这个，然后看那个，接着再看第三个，最后还要看第四个，这样才算完整，你说是不是。'
    ok, msgs = _run(chain + PLAY)
    assert any('Chinese' in m and '逗号' in m for m in msgs), msgs


def test_short_chinese_sentences_are_not_reported():
    ok, msgs = _run('先数一次。再数一次。就这样。' + PLAY)
    assert not any('Chinese' in m for m in msgs), msgs


def test_a_chinese_demo_label_cannot_oversell():
    # Verified: the English \b(run|execute|compute)\b cannot match 运行一下, because
    # Chinese has no word boundaries — so demo honesty passed on any Chinese label.
    body = '文字。' + PLAY + '\n%%% demo id=d label="运行一下看结果"\ncode: x\nout: y\n%%%\n'
    ok, msgs = blg.run(FM + '@@@ concept id=c1 tag=t title=t\n' + body)
    assert not ok
    assert any(m.startswith('FAIL') and 'demo honesty' in m for m in msgs), msgs


def test_an_honest_chinese_demo_label_passes():
    body = '文字。' + PLAY + '\n%%% demo id=d label="点开看答案"\ncode: x\nout: y\n%%%\n'
    ok, msgs = blg.run(FM + '@@@ concept id=c1 tag=t title=t\n' + body)
    assert any('pass demo labels' in m for m in msgs), msgs


def test_english_verdicts_are_unchanged_on_every_shipped_day():
    # The batch is additive by construction; this pins it across the corpus so a
    # future Chinese rule cannot quietly start failing English days.
    #
    # Scoped to the V9-AUTHORED cohort. m01's six days were later put on the pipeline
    # by verbatim extraction from their shipped HTML — they are legacy pages that were
    # never written to the v9 beginner bar, so all six fail this gate for pre-existing
    # English reasons and would only add noise to the pin.
    fails = v9 = 0
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        if 'm01-shape-of-data' in p:
            continue
        v9 += 1
        ok, _ = blg.run(open(p, encoding='utf-8').read(), source_path=p)
        fails += (not ok)
    assert v9 >= 47, 'expected >=47 v9-authored days, saw %d' % v9
    # 41 of 47 fail today for pre-existing English reasons; pinned so a regression
    # shows up as a change rather than as noise.
    assert fails == 41, ('English failure count moved to %d — a Chinese rule is firing '
                         'on English days' % fails)


# =============================================================================
# lang_parity_gate
# =============================================================================

WL = {'weight', 'neuron', 'bias', 'flop', 'token'}


def test_a_day_with_no_chinese_is_inert():
    ok, msgs = parity.run(FM + '@@@ concept id=c1 tag=t title=t\nEnglish only.\n', whitelist=WL)
    assert ok
    assert any(m.startswith('n/a') for m in msgs), msgs


def test_a_concept_with_no_chinese_fails_once_the_day_declares_chinese():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n\n'
           '@@@ concept id=c2 tag=t title=t\nB, never translated.\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('c2' in m and m.startswith('FAIL') for m in msgs), msgs


def test_an_untwinned_trailing_span_is_reported():
    tail = 'This trailing paragraph was never translated. ' * 3
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n\n' + tail + '\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('untwinned' in m or 'no Chinese twin, so it shows in' in m for m in msgs), msgs


def test_an_english_only_svg_label_with_real_words_fails():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n\n'
           '%%% svg\n<svg><text x="1" y="1">stack them</text></svg>\n%%%\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('SVG label' in m and m.startswith('FAIL') for m in msgs), msgs


def test_a_symbol_only_svg_label_needs_no_twin():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n\n'
           '%%% svg\n<svg><text x="1" y="1">N = 7 000 000 000</text>'
           '<text x="2" y="2">5 × 3</text><text x="3" y="3">✔</text></svg>\n%%%\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert any('exempt' in m for m in msgs), msgs
    assert not any('SVG label' in m and m.startswith('FAIL') for m in msgs), msgs


def test_unbalanced_paired_labels_fail():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n\n'
           '%%% svg\n<svg><text class="lang-en" x="1" y="1">a word</text></svg>\n%%%\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('unbalanced' in m for m in msgs), msgs


def test_a_missing_front_matter_twin_fails():
    src = ('---\nquest_id: t\ntitle: T\nsubtitle: S\n---\n\n'
           '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('front-matter' in m and 'title' in m for m in msgs), msgs


def test_page_title_is_exempt_from_the_twin_requirement():
    src = ('---\nquest_id: t\npage_title: P\nzh_title: 中\ntitle: T\n---\n\n'
           '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not any('page_title' in m and m.startswith('FAIL') for m in msgs), msgs


def test_a_quiz_twin_with_a_different_answer_index_fails():
    # The correctness red line: a Chinese reader told the wrong option is right.
    src = (FM + '@@@ quiz id=quiz tag=Q title=Q\n'
           '%%% quiz\nq: What? | a:1 | wrong | right | c | d | fb: x\n%%%\n'
           '~~~zh\n%%% quiz\nq: 什么？ | a:2 | 错 | 对 | 丙 | 丁 | fb: x\n%%%\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('answer index' in m or 'a:1' in m for m in msgs), msgs


def test_matching_quiz_twins_pass():
    src = (FM + '@@@ quiz id=quiz tag=Q title=Q\n'
           '%%% quiz\nq: What? | a:1 | wrong | right | c | d | fb: x\n%%%\n'
           '~~~zh\n%%% quiz\nq: 什么？ | a:1 | 错 | 对 | 丙 | 丁 | fb: x\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert any('quiz twins agree' in m for m in msgs), msgs


def test_untranslated_english_inside_the_chinese_is_reported():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n'
           '~~~zh\n这里有一段 completely untranslated sentence 留在里面。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert any('Latin word' in m for m in msgs), msgs


def test_whitelisted_terms_inside_the_chinese_are_fine():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n'
           '~~~zh\n一个 neuron（神经元）会把每个输入乘上 weight（权重）。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert any('no untranslated English' in m for m in msgs), msgs


def test_manifest_breadth_parity_is_checked_when_supplied():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n讲的是 weight（权重）。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL, manifest_covers=['weight', 'kv-cache'])
    assert not ok
    assert any('unreachable from the Chinese' in m for m in msgs), msgs


def test_missing_breadth_list_is_a_warning_not_a_silent_pass():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert any('UNCHECKED' in m for m in msgs), msgs


# --- the whitelist itself must not be able to fail open ----------------------

def test_the_real_whitelist_loads_and_is_substantial():
    terms = parity._load_whitelist()
    assert len(terms) > 60, len(terms)
    for t in ('neuron', 'weight', 'gradient', 'mfu', 'seq_len'):
        assert t in terms, t


def test_a_missing_whitelist_raises_instead_of_returning_an_empty_set():
    # An empty whitelist does not make check 5 strict, it makes it LIE: every
    # deliberate term reads as untranslated prose and buries the real finding.
    with pytest.raises(parity.WhitelistError):
        parity._load_whitelist('/nonexistent/zh_terms.yaml')


def test_a_malformed_whitelist_raises():
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write('terms: {core: [only, three, words]}\n')
        p = f.name
    with pytest.raises(parity.WhitelistError):
        parity._load_whitelist(p)


# =============================================================================
# lang_parity_gate check 1r — region-mode days, where check 1 FAILED OPEN
# =============================================================================
# THE MEASURED DEFECT. Check 1 iterates `@@@ concept` blocks. m01's six days are
# `mode: exemplar`: 14 `@@@ region` blocks of verbatim HTML each, and ZERO concepts.
# So on all six the loop ran zero times and the gate printed
#     pass all 0 concept units carry Chinese
# Six freshly translated days — 686 hand-written lang-en / lang-zh pairs across
# them — had their prose parity certified by a message that examined nothing. These
# tests pin the replacement, and the last two pin that the vacuous pass is gone.

REGION_FM = '---\nquest_id: t\nmode: exemplar\ntitle: T\nzh_title: 题\n---\n\n'


def _region_src(*regions):
    """regions = (name, html) pairs -> a minimal `mode: exemplar` source."""
    return REGION_FM + ''.join('@@@ region name=%s\n%s\n' % (n, h) for n, h in regions)


def _region_msgs(src):
    ok, msgs = parity.run(src, whitelist=WL)
    return ok, [m for m in msgs if 'region' in m]


# --- assertion (a): the lang-en / lang-zh counts must balance -----------------

def test_balanced_region_pairs_pass_and_the_gate_reports_both_counts():
    ok, msgs = _region_msgs(_region_src(
        ('brand_sub', '<span class="lang-en">M1 Day 1</span><span class="lang-zh">M1 第 1 天</span>'),
        ('hero', '<p class="lang-en">A shape is a list.</p><p class="lang-zh">shape 就是一串数。</p>')))
    assert ok
    # The numbers are the deliverable, not just the boolean — a reviewer has to be
    # able to read the two counts out of the pass line.
    assert any(m.startswith('pass') and '2 class="lang-en" vs 2 class="lang-zh"' in m
               for m in msgs), msgs


def test_an_unpaired_lang_en_node_fails_because_it_VANISHES_for_a_chinese_reader():
    # Not a cosmetic gap. A node with NEITHER class shows under both languages, so an
    # untouched page degrades to English safely. A node explicitly marked lang-en with
    # no twin is display:none in Chinese mode: the sentence is simply not on the page.
    ok, msgs = _region_msgs(_region_src(
        ('hero', '<p class="lang-en">Twinned.</p><p class="lang-zh">有孪生。</p>'
                 '<p class="lang-en">This one has no twin and disappears.</p>')))
    assert not ok
    assert any(m.startswith('FAIL') and '2 class="lang-en" vs 1 class="lang-zh"' in m
               for m in msgs), msgs
    assert any('hero: 2 en vs 1 zh' in m for m in msgs), msgs


def test_an_unpaired_lang_zh_node_fails_because_it_shows_to_an_english_reader():
    ok, msgs = _region_msgs(_region_src(
        ('hero', '<p class="lang-en">Twinned.</p><p class="lang-zh">有孪生。</p>'
                 '<p class="lang-zh">这句英文读者也会看到。</p>')))
    assert not ok
    assert any('1 class="lang-en" vs 2 class="lang-zh"' in m for m in msgs), msgs


def test_two_regions_whose_skews_CANCEL_still_fail():
    # The fail-open path in a totals-only check, and the reason 1r localises per
    # region: region a is short one Chinese twin, region b is short one English twin,
    # so the day-level totals read a clean 3 vs 3 while TWO nodes are broken.
    src = _region_src(
        ('a', '<p class="lang-en">one</p><p class="lang-zh">一</p><p class="lang-en">two</p>'),
        ('b', '<p class="lang-en">three</p><p class="lang-zh">三</p><p class="lang-zh">四</p>'))
    ok, msgs = _region_msgs(src)
    assert not ok
    assert any('3 class="lang-en" vs 3 class="lang-zh"' in m and m.startswith('FAIL')
               for m in msgs), msgs
    assert any('a: 2 en vs 1 zh' in m and 'b: 1 en vs 2 zh' in m for m in msgs), msgs


@pytest.mark.parametrize('form,why', [
    ("class='lang-en'", 'single quotes — invisible to a double-quote-only regex'),
    ('class=\\"lang-en\\"', 'escaped, which is what a double-quoted JS string in '
                            'DEMOS/BUILD/QS produces'),
    ('class="lede lang-en"', 'a second class alongside it'),
])
def test_the_count_cannot_be_dodged_by_a_different_spelling_of_the_class(form, why):
    # All 686 occurrences in m01 are the canonical double-quoted form today, so this
    # is prevention: a regex that only saw one spelling would let an unpaired node
    # through as a silent pass, which is the exact failure being closed here.
    ok, msgs = _region_msgs(_region_src(('hero', '<p %s>x</p>' % form)))
    assert not ok, why
    assert any('1 class="lang-en" vs 0 class="lang-zh"' in m for m in msgs), (why, msgs)


# --- assertion (b): a ~~~zh fence inside a region ships as literal text -------

def test_a_zh_fence_inside_a_region_fails():
    # A REAL TRAP nothing caught before. v8lib.compile_html pastes a region into the
    # page BYTE-FOR-BYTE and never calls render_md, so `~~~zh` is not a fence — the
    # four literal characters and the Chinese after them render as visible text on the
    # page. It is also the one mistake an author who knows the concept-mode grammar
    # will make first.
    ok, msgs = _region_msgs(_region_src(
        ('hero', '<p class="lang-en">A shape is a list.</p>\n~~~zh\nshape 就是一串数。\n~~~\n'
                 '<p class="lang-zh">shape 就是一串数。</p>')))
    assert not ok
    assert any(m.startswith('FAIL') and '~~~zh fence' in m and 'hero' in m
               for m in msgs), msgs


def test_paired_nodes_with_no_fence_pass_the_fence_check():
    ok, msgs = _region_msgs(_region_src(
        ('hero', '<p class="lang-en">A shape is a list.</p><p class="lang-zh">shape 就是一串数。</p>')))
    assert ok
    assert any(m.startswith('pass') and 'no region hides a ~~~zh fence' in m for m in msgs), msgs


# --- assertion (c): a prose region with no Chinese at all ---------------------

def test_a_wholly_untranslated_prose_region_is_reported_not_passed():
    # The failure this catches is the quiet one: a class-less English region renders
    # fine for both readers, so the page LOOKS finished. Balance and fence checks both
    # pass on it — there is nothing unpaired and no fence — and only a "this region
    # has no Chinese in it" check can see it.
    #
    # RETROACTIVELY VERIFIED against the real pre-translation state. Run the new check
    # on m01/day-01-arrays as it stood before this batch and it reports:
    #     FAIL 8 region(s) ... no class="lang-zh" node at all: hero (461 chars),
    #     s1 (1725), s2 (1277), s4 (2198), s7 (1990), DEMOS (1380), BUILD (1765),
    #     QS (1191)
    # — 12 027 characters of untranslated prose, correctly named region by region. The
    # old check 1 printed "pass all 0 concept units carry Chinese" on that same file.
    prose = '<p>' + 'A row is one line of the grid. ' * 8 + '</p>'   # 248 chars of text
    ok, msgs = _region_msgs(_region_src(
        ('hero', '<p class="lang-en">Hi.</p><p class="lang-zh">你好。</p>'),
        ('s4', prose)))
    assert not ok
    assert any(m.startswith('FAIL') and 'no class="lang-zh" node at all' in m and 's4' in m
               for m in msgs), msgs


def test_a_short_chrome_region_is_exempt_from_the_prose_requirement():
    # Threshold calibration, measured over m01's 84 regions: the longest region that
    # CANNOT be bilingual is `title` — a <title> element, and a browser tab cannot
    # show two — at 57 characters of text; the shortest that genuinely is prose is
    # `fin` at 166. The 200-character line sits in that gap.
    ok, msgs = _region_msgs(_region_src(
        ('title', '<title>Module 1 · Day 1 — Numbers, Arrays, and the Shape of Data</title>'),
        ('hero', '<p class="lang-en">Hi.</p><p class="lang-zh">你好。</p>')))
    assert ok, msgs
    assert any(m.startswith('pass') and 'over 200 characters of prose carries Chinese' in m
               for m in msgs), msgs


def test_a_long_region_that_is_almost_all_markup_is_not_a_false_positive():
    # `_visible_text` strips tags and comments before measuring, so a region that is
    # 900 bytes of SVG attributes and 40 characters of label text is not accused of
    # being an untranslated wall of prose.
    svg = ('<svg viewBox="0 0 520 200">'
           + ''.join('<rect x="%d" y="20" width="30" height="30" fill="#89b4fa" '
                     'stroke="#1e1e2e" stroke-width="2" rx="4"></rect>' % (20 + 40 * i)
                     for i in range(10))
           + '</svg>')
    assert len(svg) > 700
    ok, msgs = _region_msgs(_region_src(
        ('BUILD', svg), ('hero', '<p class="lang-en">Hi.</p><p class="lang-zh">你好。</p>')))
    assert ok, msgs


# --- inertness: the 40-odd English-only days must not start failing -----------

def test_a_region_day_with_no_chinese_at_all_stays_inert():
    # Same contract as the concept-mode checks. A day with no Chinese has not started;
    # the CSS fallback shows English and there is nothing to be half-done about.
    src = _region_src(('s1', '<p>' + 'English prose only. ' * 30 + '</p>'))
    src = src.replace('zh_title: 题\n', '')
    ok, msgs = parity.run(src, whitelist=WL)
    assert ok
    assert any(m.startswith('n/a') for m in msgs), msgs
    assert not any('region' in m for m in msgs), 'check 1r must not run: %s' % msgs


def test_a_day_that_declares_chinese_only_with_single_quotes_is_not_inert():
    # `_declares_chinese` used to test the literal string 'class="lang-zh"'. A page
    # whose only Chinese was single-quoted would have declared nothing, and the ENTIRE
    # gate — U+FFFD aside — would have gone inert on a half-translated day.
    src = _region_src(('hero', "<p class='lang-en'>x</p><p class='lang-zh'>甲</p>"))
    ok, msgs = parity.run(src, whitelist=WL)
    assert not any(m.startswith('n/a') for m in msgs), msgs
    assert any('1 class="lang-en" vs 1 class="lang-zh"' in m for m in msgs), msgs


# --- the vacuous pass itself, and the corpus pin -----------------------------

def test_a_region_day_never_reports_the_vacuous_all_zero_concepts_pass():
    # The exact string that certified six days while reading nothing.
    _ok, msgs = parity.run(_region_src(
        ('hero', '<p class="lang-en">Hi.</p><p class="lang-zh">你好。</p>')), whitelist=WL)
    assert not any('all 0 concept units carry Chinese' in m for m in msgs), msgs


def test_a_day_with_neither_concepts_nor_regions_says_UNCHECKED():
    # The third case, so the two branches cannot BOTH be skipped into a silent pass:
    # Chinese is declared but there is no unit of any kind to hang it on.
    ok, msgs = parity.run(REGION_FM + 'Loose prose.\n~~~zh\n散落的中文。\n~~~\n', whitelist=WL)
    assert any('UNCHECKED' in m and 'prose parity' in m for m in msgs), msgs


def _region_sources():
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        src = open(p, encoding='utf-8').read()
        body, _fm = parity._strip_frontmatter(src)
        if any(k == 'region' for k, _a, _t in parity._blocks(body)):
            out.append((p, src, body))
    return out


def test_every_shipped_region_mode_day_balances_and_is_actually_measured():
    # THE CORPUS PIN, and it guards against a loop over an empty set as much as
    # against an imbalance: assert the cohort was found before asserting anything
    # about it. m01's six days measured 122 / 114 / 112 / 105 / 109 / 124 pairs across
    # 14 regions each when this was written — all balanced, all now on evidence rather
    # than on a vacuous pass.
    days = _region_sources()
    assert len(days) >= 6, 'expected >=6 region-mode days, saw %d' % len(days)
    broken, unmeasured = {}, []
    for path, src, body in days:
        rel = os.path.relpath(path, os.path.join(REPO, 'sessions'))
        en = len(parity._LANG_EN_RE.findall(body))
        zh = len(parity._LANG_ZH_RE.findall(body))
        if en != zh:
            broken[rel] = (en, zh)
        ok, msgs = parity.run(src)
        if not any(m.startswith('pass') and 'region language classes balance' in m
                   for m in msgs):
            unmeasured.append((rel, [m for m in msgs if 'region' in m or m.startswith('n/a')]))
    assert not broken, 'unbalanced lang classes (en, zh): %s' % broken
    assert not unmeasured, 'check 1r did not produce a real verdict: %s' % unmeasured


def test_no_shipped_region_hides_a_zh_fence():
    # Pins the trap corpus-wide: today 0 of 84 m01 regions contain one, and a fence
    # written into a region tomorrow would ship the characters `~~~zh` to the reader.
    bad = {}
    for path, _src, body in _region_sources():
        for _k, args, text in parity._blocks(body):
            if _k == 'region' and parity._fences(text):
                bad.setdefault(os.path.relpath(path, os.path.join(REPO, 'sessions')),
                               []).append(args.strip())
    assert not bad, 'literal ~~~zh fences inside verbatim regions: %s' % bad


# =============================================================================
# reader_flow_gate — the Chinese side of the curiosity and discovery rules
# =============================================================================
# The English lists keep working on a bilingual day because the English text is
# still there. What was missing is the Chinese side: '?' cannot match '？', and an
# English spine word can never appear in Chinese prose.
import reader_flow_gate as rf


def _bilingual_source(hero_zh, prod_zh, zh_spine=None, concept_zh='这里讲 bend 的中文，很清楚。'):
    # Built with .format and a literal '%%% svg' rather than %-formatting: an
    # earlier version concatenated a variable into the middle of the template, which
    # silently bound `% i` to only the trailing literal, so `%%%%` never collapsed
    # to `%%` and the widget fence was never a fence.
    concept = ('@@@ concept id=c{n} tag="Recap" title="t"\n'
               'bend prose\n\n'
               '%%% svg\n<svg><text>x</text></svg>\n%%%\n\n'
               'more bend prose\n~~~zh\n{zh}\n~~~\n\n')
    return ('---\nquest_id: t\nmode: concept\nspine: bend\n'
            + ('zh_spine: %s\n' % zh_spine if zh_spine else '') + '---\n\n'
            '@@@ hero\n@lede Have you ever wondered? bend\n@goal g\n~~~zh\n' + hero_zh + '\n~~~\n\n'
            + ''.join(concept.format(n=i, zh=concept_zh) for i in (1, 2, 3))
            + '@@@ produce id=produce tag=P title=P\npredict what you should see\n'
              '~~~zh\n' + prod_zh + '\n~~~\n')


def _rf(src):
    meta, body = v8lib.split_frontmatter(src)
    return rf.run(meta, v8lib.parse_blocks(body))


def test_a_chinese_hero_that_asks_the_reader_something_passes():
    ok, msgs = _rf(_bilingual_source('你有没有想过？这是一个弯。', '先预测一下会看到什么。'))
    assert any('Chinese hero has a human' in m and m.startswith('pass') for m in msgs), msgs


def test_a_flat_chinese_hero_fails():
    ok, msgs = _rf(_bilingual_source('这是一段平铺直叙的说明文字。', '先预测一下会看到什么。'))
    assert not ok
    assert any('Chinese hero has no human' in m for m in msgs), msgs


def test_a_chinese_produce_that_is_not_discovery_framed_fails():
    ok, msgs = _rf(_bilingual_source('你有没有想过？', '把脚本跑完就行。'))
    assert not ok
    assert any('not discovery-framed' in m and '预测' in m for m in msgs), msgs


def test_zh_spine_must_reach_three_blocks():
    ok, msgs = _rf(_bilingual_source('你有没有想过？弯', '先预测一下。',
                                     zh_spine='尺子'))
    assert not ok
    assert any("zh_spine ('尺子')" in m for m in msgs), msgs


def test_zh_spine_carried_through_the_concepts_passes():
    ok, msgs = _rf(_bilingual_source('你有没有想过？这个弯。', '先预测一下这个弯。',
                                     zh_spine='弯', concept_zh='这里讲这个弯的中文，很清楚。'))
    assert any("zh_spine ('弯')" in m and m.startswith('pass') for m in msgs), msgs


def test_a_missing_zh_spine_is_an_explicit_unchecked_warning():
    ok, msgs = _rf(_bilingual_source('你有没有想过？', '先预测一下。'))
    assert any('UNCHECKED' in m for m in msgs), msgs


def test_the_chinese_checks_are_silent_on_an_english_only_day():
    src = ('---\nquest_id: t\nmode: concept\nspine: bend\n---\n\n'
           '@@@ hero\n@lede Have you ever wondered? bend\n@goal g\n\n'
           + ''.join('@@@ concept id=c%d tag="Recap" title="t"\nbend prose\n\n'
                     '%%%% svg\n<svg><text>x</text></svg>\n%%%%\n\nmore bend prose\n\n' % i
                     for i in (1, 2, 3))
           + '@@@ produce id=produce tag=P title=P\npredict what you should see\n')
    ok, msgs = _rf(src)
    assert not any('Chinese' in m or 'zh_spine' in m for m in msgs), msgs


# =============================================================================
# check 0 — U+FFFD corruption, on every day
# =============================================================================
# Real defect: 22 replacement characters in a freshly translated day, 21 in one
# already published, and 3 in English-only prose. The files decode as valid UTF-8
# (they contain a well-formed encoding of U+FFFD), so nothing else could see them.

def test_a_replacement_character_fails_even_with_no_chinese():
    ok, msgs = parity.run(FM + '@@@ concept id=c1 tag=t title=t\nsame rule as yes/no � only x.\n',
                          whitelist=WL)
    assert not ok
    assert any('U+FFFD' in m and m.startswith('FAIL') for m in msgs), msgs


def test_a_replacement_character_inside_the_chinese_fails():
    src = (FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n我们又该往哪边走才能学�东西。\n~~~\n')
    ok, msgs = parity.run(src, whitelist=WL)
    assert not ok
    assert any('U+FFFD' in m for m in msgs), msgs


def test_clean_text_passes_check_zero():
    ok, msgs = parity.run(FM + '@@@ concept id=c1 tag=t title=t\nEnglish only.\n', whitelist=WL)
    assert ok
    assert any('no U+FFFD' in m and m.startswith('pass') for m in msgs), msgs


def test_no_shipped_source_carries_a_replacement_character():
    # The corpus-wide pin. All six affected files were repaired by hand; this stops
    # the next authoring pass from re-introducing one silently.
    bad = {}
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        n = open(p, encoding='utf-8').read().count('�')
        if n:
            bad[os.path.relpath(p, os.path.join(REPO, 'sessions'))] = n
    assert not bad, 'U+FFFD in shipped sources: %s' % bad


def test_no_compiled_lesson_carries_a_replacement_character():
    bad = {}
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'lesson.html'))):
        n = open(p, encoding='utf-8').read().count('�')
        if n:
            bad[os.path.relpath(p, os.path.join(REPO, 'sessions'))] = n
    assert not bad, 'U+FFFD in compiled lessons: %s' % bad


# =============================================================================
# gloss-body masking — a tooltip must not count as main-line prose
# =============================================================================
# Real defect: the mask regex was `\[\[[^\|\]]+\|\|[^\]]*\]\]`, whose `[^\]]*`
# cannot cross a `]` inside the gloss. A gloss body containing brackets — e.g.
# `clamp to [ε, 1−ε] before the log` — made the pattern fail to match AT ALL, so
# the whole hover-only body was measured as main-line prose. That produced phantom
# "Chinese sentence over 60 汉字" findings on m02/day-04 which an author then
# "fixed" by splitting sentences that were never long.

def test_a_gloss_body_is_masked_out_of_the_measured_prose():
    body = ('把每个概率挤进 [[epsilon clipping||把概率夹到 [ε, 1−ε] 这个小区间里，'
            '这样 −log 永远不会炸到无穷，也不会污染后面每一步]] 就行了。' + PLAY)
    ok, msgs = _run(body)
    assert not any('Chinese' in m and '汉字' in m for m in msgs), \
        'a bracketed gloss body leaked into the main-line sentence measurement: %s' % msgs


def test_a_gloss_without_brackets_is_still_masked():
    body = '这是一个 [[gradient||loss 上升最快的方向]] 的例子。' + PLAY
    ok, msgs = _run(body)
    assert not any('Chinese' in m and '汉字' in m for m in msgs), msgs


def test_a_genuinely_long_chinese_sentence_is_still_caught_outside_a_gloss():
    # The fix must not become a way to hide real run-ons.
    long_zh = ('在给任何东西定价之前我们需要一个可以数的东西这就是我们要数的而且它比你'
               '想象的要小得多一次乘法或者一次加法就是全部了不是一整个求和也不是一行代码。')
    ok, msgs = _run(long_zh + PLAY)
    assert any('Chinese' in m and '汉字' in m for m in msgs), msgs


# =============================================================================
# `、` is the enumeration comma, not a clause joint
# =============================================================================
# Real defect: counting 、 as a clause comma reported a 29-汉字 sentence on
# m02/day-08 as a 5-comma chain, because 3 of the 5 were separators inside the
# numeric list （1.0、0.1、0.01、0.001）. A list is ONE idea however many items it
# has, and the rule exists to catch clause chaining.

def test_a_numeric_enumeration_does_not_count_as_a_comma_chain():
    body = '我们扫四个速率（1.0、0.1、0.01、0.001），然后读每一条曲线。' + PLAY
    ok, msgs = _run(body)
    assert not any('逗号' in m for m in msgs), \
        'enumeration separators were counted as clause commas: %s' % msgs


def test_a_short_word_enumeration_does_not_count_either():
    body = '六个陷阱是太大胆、太胆小、没清零、顺序固定、停得太早、练得太久。' + PLAY
    ok, msgs = _run(body)
    assert not any('逗号' in m for m in msgs), msgs


def test_a_real_clause_chain_is_still_caught():
    # The fix must not become a way to hide clause chaining.
    body = '我们先看这个，然后看那个，接着再看第三个，最后还要看第四个，这样才算完整，你说是不是。' + PLAY
    ok, msgs = _run(body)
    assert any('逗号' in m for m in msgs), msgs


def test_a_clause_chain_that_also_holds_a_list_is_still_caught():
    # An enumeration must not launder the clauses around it.
    body = ('我们先扫四个速率（1.0、0.1、0.01、0.001），然后读曲线，接着挑一个，'
            '再训一遍，最后比一比，你说是不是。' + PLAY)
    ok, msgs = _run(body)
    assert any('逗号' in m for m in msgs), msgs


# =============================================================================
# check 0b — a DECLARED bilingual day with no Chinese must FAIL
# =============================================================================
# The gap this closes: every Chinese check is inert until a day declares Chinese,
# so a module could set `zh.langs: [en, zh]` and ship every day in English with a
# green board — silence was indistinguishable from success. `zh.scope` could not
# help: it is free prose in every manifest that has it. So enforcement reads a
# machine-readable `zh.require`, which is either "all" or a list of day-dir names.
#
# The inertness itself is load-bearing (it let the toggle reach 293 pages without
# touching content), so it is preserved for modules that never opted in.

import tempfile, textwrap


def _day_tree(manifest_yaml, day='day-01-thing', source=None):
    """Build <root>/mod/<day>/source.md plus <root>/mod/_refactor/manifest.yaml."""
    root = tempfile.mkdtemp()
    mod = os.path.join(root, 'mod')
    os.makedirs(os.path.join(mod, day))
    os.makedirs(os.path.join(mod, '_refactor'))
    if manifest_yaml is not None:
        open(os.path.join(mod, '_refactor', 'manifest.yaml'), 'w',
             encoding='utf-8').write(textwrap.dedent(manifest_yaml))
    sp = os.path.join(mod, day, 'source.md')
    open(sp, 'w', encoding='utf-8').write(
        source if source is not None
        else FM + '@@@ concept id=c1 tag=t title=t\nEnglish only.\n')
    return sp


def test_a_declared_bilingual_day_with_no_chinese_fails():
    sp = _day_tree('zh:\n  langs: [en, zh]\n  require: all\n')
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert not ok
    assert any(m.startswith('FAIL') and 'declares it must' in m for m in msgs), msgs


def test_a_day_excluded_by_a_require_list_stays_inert():
    sp = _day_tree('zh:\n  langs: [en, zh]\n  require: [day-99-other]\n')
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert ok
    assert any('deliberately excludes it' in m for m in msgs), msgs


def test_a_day_named_in_a_require_list_fails():
    sp = _day_tree('zh:\n  langs: [en, zh]\n  require: [day-01-thing]\n')
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert not ok, msgs


def test_a_module_with_no_manifest_stays_inert_and_says_so():
    # m03-m08 are in this state on purpose; they must not start failing.
    sp = _day_tree(None)
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert ok
    assert any('NOT ENFORCED' in m and 'no _refactor/manifest.yaml' in m for m in msgs), msgs


def test_a_manifest_without_a_zh_block_stays_inert():
    sp = _day_tree('coverage_topics: []\n')
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert ok
    assert any('NOT ENFORCED' in m and 'no zh: block' in m for m in msgs), msgs


def test_declaring_langs_without_require_is_reported_as_unenforceable():
    # The exact prior state of m02: zh.langs said [en, zh] and nothing could act.
    sp = _day_tree('zh:\n  langs: [en, zh]\n  scope: "prose a gate cannot read"\n')
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert ok
    assert any('no machine-readable' in m for m in msgs), msgs


def test_an_unreadable_require_raises_rather_than_degrading():
    # Degrading to "nothing declared" would restore the exact silent pass this
    # check removes — the same shape as the whitelist loader's old `except: pass`.
    sp = _day_tree('zh:\n  langs: [en, zh]\n  require: {oops: 1}\n')
    with pytest.raises(parity.ManifestError):
        parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)


def test_malformed_manifest_yaml_raises():
    sp = _day_tree('zh:\n  langs: [en, zh\n  require: all\n')     # unclosed bracket
    with pytest.raises(parity.ManifestError):
        parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)


def test_run_without_a_source_path_keeps_the_old_behaviour():
    # Callers that predate source_path must not change verdict.
    ok, msgs = parity.run(FM + '@@@ concept id=c1 tag=t title=t\nEnglish.\n', whitelist=WL)
    assert ok
    assert any('NOT ENFORCED' in m for m in msgs), msgs


def test_a_bilingual_day_in_an_unrequiring_module_is_warned_not_failed():
    sp = _day_tree(None, source=(
        FM + '@@@ concept id=c1 tag=t title=t\nA.\n~~~zh\n甲。\n~~~\n'))
    ok, msgs = parity.run(open(sp, encoding='utf-8').read(), whitelist=WL, source_path=sp)
    assert any(m.startswith('warn') and 'no module manifest requires it' in m
               for m in msgs), msgs


def test_the_real_m01_and_m02_days_are_all_declared_and_required():
    # The point of the whole change: these 15 days can no longer be silently
    # dropped back to English.
    for mod in ('m01-shape-of-data', 'm02-the-neuron'):
        for sp in sorted(glob.glob(os.path.join(REPO, 'sessions', mod, 'day-*', 'source.md'))):
            req, why = parity._load_requirement(sp)
            assert req is True, '%s not required: %s' % (sp, why)


def test_every_day_its_module_requires_is_actually_bilingual():
    """The real enforcement point for zh.require.

    check 0b is ADVISORY inside compile_lesson.py, because failing a compile on an
    untranslated day deadlocks the build: the English author cannot satisfy a
    Chinese finding, so its fix rounds burn, the lesson never converges, and the
    translate phase that would have fixed it never runs. An untranslated day is a
    rollout-completeness problem, not a page-validity one.

    So completeness is enforced HERE instead — this test runs in the `compiler-tests`
    publish gate, which means a declared-bilingual day cannot reach the site in
    English. Verified both directions: the gate CLI still exits 6 on such a day.
    """
    missing = []
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        req, _why = parity._load_requirement(p)
        if req is not True:
            continue
        if not parity._declares_chinese(open(p, encoding='utf-8').read()):
            missing.append(os.path.relpath(p, os.path.join(REPO, 'sessions')))
    assert not missing, (
        '%d day(s) whose module declares zh.require carry NO Chinese: %s. Either '
        'translate them or narrow zh.require in the module manifest.'
        % (len(missing), missing))
