import os, sys, re
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import v8lib
import concept_shell_gate

SHELLS = os.path.join(HERE, '..', 'shells')
REPO = os.path.join(HERE, '..', '..', '..')
FIXTURE = os.path.join(HERE, 'fixtures', 'mini_concept.md')


def _compile(src_text):
    meta, body = v8lib.split_frontmatter(src_text)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(SHELLS, 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta


# =============================================================================
# bilingual() — the fallback rule everything else depends on
# =============================================================================
# A node with NEITHER language class shows under both, because the donor only
# hides .lang-zh under data-lang="en" and .lang-en under data-lang="zh".

def test_no_twin_returns_the_english_unwrapped():
    assert v8lib.bilingual('Arrays', None) == 'Arrays'
    assert v8lib.bilingual('Arrays', '') == 'Arrays'
    assert v8lib.bilingual('Arrays', '   ') == 'Arrays'


def test_a_twin_produces_a_paired_span():
    out = v8lib.bilingual('The bend', '这个弯')
    assert out == '<span class="lang-en">The bend</span><span class="lang-zh">这个弯</span>'


def test_the_tag_is_configurable():
    assert v8lib.bilingual('a', '甲', tag='div').startswith('<div class="lang-en">')


# =============================================================================
# the ~~~zh fence
# =============================================================================

def test_a_fence_pairs_the_span_above_it():
    out = v8lib.render_md('English one.\n\nEnglish two.\n\n~~~zh\n中文一。\n\n中文二。\n~~~')
    assert '<div class="lang-en"><p>English one.</p>' in out
    assert 'English two.</p></div>' in out
    assert '<div class="lang-zh"><p>中文一。</p>' in out
    # the whole span is ONE wrapper, not one per paragraph
    assert out.count('class="lang-en"') == 1
    assert out.count('class="lang-zh"') == 1


def test_blocks_after_the_last_fence_stay_unwrapped():
    out = v8lib.render_md('Para A.\n\n~~~zh\n甲。\n~~~\n\nPara B.')
    assert out.count('class="lang-en"') == 1
    tail = out[out.rindex('</div>'):]
    assert 'Para B' in tail and 'lang-' not in tail, tail


def test_two_fences_make_two_independent_spans():
    out = v8lib.render_md('A.\n\n~~~zh\n甲。\n~~~\n\nB.\n\n~~~zh\n乙。\n~~~')
    assert out.count('class="lang-en"') == 2
    assert out.count('class="lang-zh"') == 2
    # the second English span must not swallow the first
    en = re.findall(r'<div class="lang-en">(.*?)</div>', out, re.S)
    assert 'A.' in en[0] and 'B.' not in en[0]
    assert 'B.' in en[1] and 'A.' not in en[1]


def test_the_full_widget_grammar_works_inside_a_fence():
    out = v8lib.render_md(
        'Count the presses.\n\n'
        '%%% steps\nstep: 5 × 3\nwhy: one multiply\n%%%\n\n'
        '~~~zh\n'
        '数一数按了几次。\n\n'
        '%%% steps\nstep: 5 × 3\nwhy: 一次乘法\n%%%\n'
        '~~~')
    zh = re.search(r'<div class="lang-zh">(.*?)</div>\s*$', out, re.S).group(1)
    assert 'class="build-step"' in zh, 'the %%% steps widget did not render inside the fence'
    assert '一次乘法' in zh


def test_a_gloss_inside_a_fence_gets_its_own_tooltip():
    # The Chinese prose carries its own [[term||tip]], so inline() needs no change
    # and there is no data-tip-zh to keep in sync.
    out = v8lib.render_md('A [[FLOP||one multiply or one add]].\n\n'
                          '~~~zh\n一次 [[FLOP||一次乘法，或者一次加法]]。\n~~~')
    tips = re.findall(r'data-tip="([^"]*)"', out)
    assert tips == ['one multiply or one add', '一次乘法，或者一次加法']


def test_a_callout_inside_a_fence_renders():
    out = v8lib.render_md('!!! c-warn 😕\nEnglish warning\n!!!\n\n'
                          '~~~zh\n!!! c-warn 😕\n中文提示\n!!!\n~~~')
    assert out.count('class="callout c-warn"') == 2
    assert '中文提示' in out


# --- the error cases, which must be loud rather than silently wrong ----------

def test_an_unterminated_fence_raises_instead_of_leaking():
    # Without this the marker ships as literal text: render_md would emit
    # <p>~~~zh 中文。</p>. concept_shell_gate now also catches a leaked ~~~, but
    # failing at compile time names the cause.
    with pytest.raises(ValueError, match='unterminated'):
        v8lib.render_md('English.\n\n~~~zh\n中文。')


def test_a_fence_with_no_english_above_raises():
    # Chinese-only content is invisible to an English reader — a silent hole.
    with pytest.raises(ValueError, match='no English blocks'):
        v8lib.render_md('~~~zh\n中文。\n~~~')


def test_an_empty_fence_raises():
    with pytest.raises(ValueError, match='empty'):
        v8lib.render_md('English.\n\n~~~zh\n\n~~~')


def test_a_nested_fence_raises():
    with pytest.raises(ValueError, match='nest'):
        v8lib.render_md('English.\n\n~~~zh\n中文。\n\n~~~zh\n更多。\n~~~\n~~~')


def test_the_html_escape_hatch_still_works():
    out = v8lib.render_md('~~~html\n<p>raw</p>\n~~~')
    assert out == '<p>raw</p>'


# =============================================================================
# front-matter and block-argument twins
# =============================================================================

def test_hero_markers_split_in_any_order():
    f = v8lib._hero_fields('@lede English lede @zh_lede 中文引子 @goal English goal @zh_goal 中文目标')
    assert f['lede'].strip() == 'English lede'
    assert f['zh_lede'].strip() == '中文引子'
    assert f['goal'].strip() == 'English goal'
    assert f['zh_goal'].strip() == '中文目标'


def test_zh_lede_is_not_read_as_lede():
    # The `@` anchor is what separates them: after the @, the text is 'zh_lede',
    # which does not begin with 'lede'. (Reordering the alternation is harmless for
    # the same reason — an earlier version of this file claimed otherwise.)
    f = v8lib._hero_fields('@zh_lede 中文 @lede English')
    assert f['zh_lede'].strip() == '中文'
    assert f['lede'].strip() == 'English'


def test_the_word_lede_in_prose_is_not_a_marker():
    # The `\b` and the `@` are load-bearing: prose mentioning a marker name, or a
    # longer @-word, must not split the hero.
    f = v8lib._hero_fields('@lede The lede goal of this ledex is @goals nothing. @goal G')
    assert set(f) == {'lede', 'goal'}, f
    assert 'ledex' in f['lede'] and '@goals' in f['lede']
    assert f['goal'].strip() == 'G'


def test_a_hero_with_only_english_markers_yields_no_twins():
    f = v8lib._hero_fields('@lede E @goal G')
    assert 'zh_lede' not in f and 'zh_goal' not in f


def test_missing_front_matter_keys_do_not_crash_the_compiler():
    # render_hero and render_fin used bracket access, which exits with a bare
    # KeyError BEFORE any gate runs — indistinguishable from a real parse error.
    assert v8lib.render_fin({}) .count('<h3></h3>') == 1
    blk = {'lines': ['@lede hi', '@goal go'], 'args': {}}
    html = v8lib.render_hero({}, blk)
    assert 'class="kicker"' in html and 'lang-' not in html


@pytest.mark.parametrize('en_key,zh_key', [
    ('title', 'zh_title'), ('subtitle', 'zh_subtitle'), ('module_label', 'zh_module_label'),
])
def test_hero_front_matter_twins_pair(en_key, zh_key):
    blk = {'lines': ['@lede hi', '@goal go'], 'args': {}}
    meta = {'title': 'T', 'subtitle': 'S', 'module_label': 'M', zh_key: '中'}
    html = v8lib.render_hero(meta, blk)
    assert '<span class="lang-zh">中</span>' in html


def test_fin_twins_pair():
    html = v8lib.render_fin({'fin_title': 'Done!', 'fin_body': 'Nice.',
                             'zh_fin_title': '完成！', 'zh_fin_body': '很好。'})
    assert '<span class="lang-zh">完成！</span>' in html
    assert '<span class="lang-zh">很好。</span>' in html


def test_concept_args_pair_and_data_sec_stays_english():
    blk = {'type': 'concept', 'args': {'id': 'c1', 'tag': 'The bend', 'title': 'A bend',
                                       'gotit': 'Got it', 'zh_tag': '这个弯',
                                       'zh_title': '一个弯', 'zh_gotit': '懂了'},
           'lines': ['Body.']}
    html = v8lib.render_concept(blk)
    assert 'data-sec="c1"' in html, 'progress key must stay the id, not the label'
    for zh in ('这个弯', '一个弯', '懂了'):
        assert '<span class="lang-zh">%s</span>' % zh in html


def test_sidebar_nav_labels_pair_and_omit_when_absent():
    blocks = [
        {'type': 'concept', 'args': {'id': 'c1', 'tag': 'One', 'zh_tag': '一'}, 'lines': []},
        {'type': 'concept', 'args': {'id': 'c2', 'tag': 'Two'}, 'lines': []},
    ]
    items = v8lib.concept_nav_items(blocks)
    assert items[1]['zh_label'] == '1 · 一'
    assert items[2]['zh_label'] is None, 'a block with no Chinese tag must not get an invented one'
    nav = v8lib.render_sidebar_nav_items({'module_label': 'M'}, items)
    assert '<span class="lang-zh">1 · 一</span>' in nav
    assert '>Start here<' in nav, 'the shell string must stay unwrapped'


# =============================================================================
# end to end, through the real donor and the real gate
# =============================================================================

def _bilingual_fixture():
    """The mini fixture with a Chinese twin added to its first concept."""
    text = open(FIXTURE, encoding='utf-8').read()
    text = text.replace('@@@ concept id=c1 tag="The collapse"',
                        '@@@ concept id=c1 zh_tag="坍缩" zh_title="直的加直的还是直的" zh_gotit="懂了" tag="The collapse"')
    # append a fence to the end of c1's body, just before the next block
    i = text.index('@@@ concept id=c2')
    text = text[:i] + '~~~zh\n答案：还是一条直线。两层没有弯的神经元会折回成一层。\n~~~\n\n' + text[i:]
    return text


def test_a_bilingual_lesson_compiles_and_passes_the_shell_gate():
    html, meta = _compile(_bilingual_fixture())
    donor = open(os.path.join(SHELLS, 'v9-base.donor'), encoding='utf-8').read()
    ok, msgs = concept_shell_gate.run(html, meta, donor=donor)
    assert ok, '\n'.join(msgs)


def test_the_compiled_bilingual_page_has_both_languages_and_no_leaked_fence():
    html, _ = _compile(_bilingual_fixture())
    assert 'class="lang-en"' in html and 'class="lang-zh"' in html
    assert '~~~' not in html, 'a fence marker leaked into the page'
    assert '答案：还是一条直线' in html
    assert '坍缩' in html


def test_the_page_still_has_exactly_one_progress_key_per_section():
    # The reader's ticks are stored per data-sec. Pairing labels must not add,
    # rename or duplicate a single one of them.
    plain, _ = _compile(open(FIXTURE, encoding='utf-8').read())
    bi, _ = _compile(_bilingual_fixture())
    assert (re.findall(r'data-sec="([^"]+)"', plain)
            == re.findall(r'data-sec="([^"]+)"', bi))


def test_a_source_with_no_chinese_emits_no_language_wrappers_in_the_content():
    # The property the whole rollout rests on. Scoped to <main>, because the SHELL
    # always carries .lang-en/.lang-zh — in the CSS, the sidebar buttons and the
    # controller. A whole-document check would be vacuously false.
    plain, _ = _compile(open(FIXTURE, encoding='utf-8').read())
    content = re.search(r'<main id="content">(.*?)</main>', plain, re.S).group(1)
    assert 'lang-en' not in content and 'lang-zh' not in content, \
        'the bilingual machinery is not inert on a source with no Chinese'

    bi, _ = _compile(_bilingual_fixture())
    bi_content = re.search(r'<main id="content">(.*?)</main>', bi, re.S).group(1)
    assert 'lang-zh' in bi_content, 'and it must actually fire when Chinese IS authored'


# =============================================================================
# a drawing is SHARED, so it belongs to no span
# =============================================================================
# The normal shape of a concept is: intro prose, picture, build-up prose. The
# picture is one drawing whose labels are paired <text class="lang-en"> /
# <text class="lang-zh">, so it must never be enclosed in a .lang-en wrapper —
# there it would vanish entirely in Chinese mode and the concept would silently
# lose its visual with every gate still green. A drawing therefore closes whatever
# span preceded it and starts a new one.

SVG = '%%% svg\n<svg viewBox="0 0 10 10"><text x="1" y="1">L</text></svg>\n%%%'


def test_a_drawing_between_two_spans_stays_unwrapped():
    out = v8lib.render_md(
        'Intro.\n~~~zh\n引子。\n~~~\n\n' + SVG + '\n\nBuild-up.\n~~~zh\n推导。\n~~~')
    # the drawing sits between the two wrapper pairs, in neither of them
    assert not re.search(r'class="lang-(en|zh)">\s*<div class="build-viz">', out), out[:400]
    assert out.count('<svg') == 1
    assert out.count('class="lang-en"') == 2 and out.count('class="lang-zh"') == 2
    # order preserved: intro, drawing, build-up
    assert out.index('引子') < out.index('<svg') < out.index('推导')


def test_a_fence_right_after_a_drawing_has_nothing_to_pair():
    with pytest.raises(ValueError, match='no English blocks'):
        v8lib.render_md('Intro.\n~~~zh\n引子。\n~~~\n\n' + SVG + '\n\n~~~zh\n孤立中文。\n~~~')


def test_unpaired_prose_before_a_drawing_is_not_an_error():
    # It shows under BOTH languages, which is the fallback. lang_parity_gate is
    # what reports it as incomplete, not the compiler.
    out = v8lib.render_md('Intro with no twin.\n\n' + SVG + '\n\nAfter.\n~~~zh\n之后。\n~~~')
    assert 'Intro with no twin' in out
    assert out.count('class="lang-en"') == 1          # only the paired span
    head = out[:out.index('<svg')]
    assert 'lang-' not in head, 'unpaired intro prose must stay unwrapped'


def test_a_viz_embed_is_shared_too():
    out = v8lib.render_md('Intro.\n~~~zh\n引子。\n~~~\n\n'
                          '%%% viz src="../../viz/x.html" title="t"\n%%%\n\n'
                          'After.\n~~~zh\n之后。\n~~~')
    assert out.count('class="build-embed"') == 1
    assert not re.search(r'class="lang-(en|zh)">\s*<div class="build-embed">', out)


# =============================================================================
# the hero's warm-up, and the bilingual quiz count
# =============================================================================
# Both found by translating a real day, not by inspection.

def test_a_chinese_warmup_twin_is_paired_not_leaked():
    # render_hero does NOT go through render_md — it splits the body on @lede/@goal
    # markers — so a ~~~zh fence there is never parsed as a fence. The Chinese
    # warm-up landed inside @zh_goal and shipped as literal '~~~zh %%% warmup q: …'.
    blk = {'args': {}, 'lines': [
        '@lede Have you ever seen this?', '@goal Do the thing.', '',
        '%%% warmup', 'q: EN? | a:1 | a | b | c | d | fb: x', '%%%', '',
        '~~~zh', '%%% warmup', 'q: 中文？ | a:1 | 甲 | 乙 | 丙 | 丁 | fb: x', '%%%', '~~~']}
    html = v8lib.render_hero({'title': 'T', 'subtitle': 'S', 'module_label': 'M'}, blk)
    assert '~~~' not in html and '%%%' not in html, html[-300:]
    assert html.count('class="warmup"') == 2
    assert '<div class="lang-en"><div class="warmup"' in html
    assert '中文？' in html and 'EN?' in html


def test_a_chinese_warmup_with_no_english_one_raises():
    # An English reader would lose the warm-up entirely.
    blk = {'args': {}, 'lines': [
        '@lede L', '@goal G', '',
        '~~~zh', '%%% warmup', 'q: 中文？ | a:1 | 甲 | 乙 | 丙 | 丁 | fb: x', '%%%', '~~~']}
    with pytest.raises(ValueError, match='no English one'):
        v8lib.render_hero({}, blk)


def test_the_shell_gate_counts_quiz_questions_per_language():
    # Counting NODES reported "got 8" on the first translated day and blocked a
    # correct page: a bilingual quiz carries every question twice.
    html, meta = _compile(_bilingual_fixture())
    ok, msgs = concept_shell_gate.run(html, meta)
    assert ok, '\n'.join(msgs)
    line = [m for m in msgs if 'quiz has 4 questions' in m][0]
    assert line.startswith('pass'), line
    assert 'per language' in line


def test_the_real_pilot_day_passes_every_deterministic_gate():
    # The end-to-end proof: the first fully translated day, through the real gates.
    import subprocess
    day = os.path.join(REPO, 'sessions', 'm02-the-neuron', 'day-02-activations')
    src, lesson = os.path.join(day, 'source.md'), os.path.join(day, 'lesson.html')
    if not os.path.exists(lesson):
        pytest.skip('pilot day not compiled')
    text = open(src, encoding='utf-8').read()
    assert '~~~zh' in text, 'the pilot day lost its Chinese'
    import lang_parity_gate
    ok, msgs = lang_parity_gate.run(text)
    assert ok, '\n'.join(msgs)
    meta, _ = v8lib.split_frontmatter(text)
    donor = open(os.path.join(SHELLS, meta['donor']), encoding='utf-8').read()
    ok2, msgs2 = concept_shell_gate.run(open(lesson, encoding='utf-8').read(), meta, donor=donor)
    assert ok2, '\n'.join(msgs2)


def test_the_pilot_page_has_balanced_language_nodes():
    day = os.path.join(REPO, 'sessions', 'm02-the-neuron', 'day-02-activations', 'lesson.html')
    if not os.path.exists(day):
        pytest.skip('pilot day not compiled')
    h = open(day, encoding='utf-8').read()
    main = re.search(r'<main id="content">(.*?)</main>', h, re.S).group(1)
    assert main.count('class="lang-en"') == main.count('class="lang-zh"')
    assert main.count('<text class="lang-en"') == main.count('<text class="lang-zh"')
    assert not any(m in main for m in ('~~~', '%%%', '@@@'))

