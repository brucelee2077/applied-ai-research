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
