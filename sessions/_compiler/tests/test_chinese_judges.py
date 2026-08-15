import os, sys, json
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import coverage_judge as cj

ZH_SRC = 'English prose.\n~~~zh\n中文段落。\n~~~\n'


@pytest.fixture(autouse=True)
def _restore_chat():
    original = cj._chat
    yield
    cj._chat = original


def _mock(payload):
    cj._chat = lambda *a, **k: payload


# =============================================================================
# the anchor table
# =============================================================================
# The judges are semantic, so Chinese does not CRASH them — they grade the wrong
# thing. An English "no idioms" anchor looks for "under the hood" and waves through
# a page full of 成语, which for a 12-year-old is the harder barrier.

def test_both_languages_define_every_anchor():
    assert set(cj.LANG_ANCHORS) == {'en', 'zh'}
    keys = set(cj.LANG_ANCHORS['en'])
    assert set(cj.LANG_ANCHORS['zh']) == keys, 'the two tables have different keys'
    for lang in ('en', 'zh'):
        for k in keys:
            assert cj.LANG_ANCHORS[lang][k].strip(), '%s/%s is empty' % (lang, k)


def test_an_unknown_language_falls_back_to_english_instead_of_raising():
    assert cj._A('de', 'reader') == cj.LANG_ANCHORS['en']['reader']


def test_an_unknown_anchor_key_returns_empty_rather_than_raising():
    # The module's contract is that a judge degrades, never raises. A typo'd anchor
    # name must not take a whole build round down.
    assert cj._A('zh', 'no_such_anchor') == ''
    assert cj._A('en', 'no_such_anchor') == ''


def test_the_two_languages_actually_differ():
    for k in cj.LANG_ANCHORS['en']:
        assert cj.LANG_ANCHORS['en'][k] != cj.LANG_ANCHORS['zh'][k], \
            'anchor %r is identical in both languages — it was not localized' % k


# =============================================================================
# the anchors reach the prompt text (not just the signature)
# =============================================================================
# A `lang` parameter that is accepted and then ignored is the exact shape of the
# donor= kwarg bug: the call site looks right and the check does not exist.

def test_the_reader_clause_changes_with_lang():
    assert 'ENGLISH IS A SECOND LANGUAGE' in cj._lang_abs_sys('en')
    zh = cj._lang_abs_sys('zh')
    assert '中文母语' in zh
    assert 'ENGLISH IS A SECOND LANGUAGE' not in zh


def test_the_hard_word_and_idiom_anchors_change_with_lang():
    en = cj._lang_abs_prompt('X', 'en')
    zh = cj._lang_abs_prompt('X', 'zh')
    assert 'utilize' in en and 'under the hood' in en
    assert 'utilize' not in zh and 'under the hood' not in zh
    assert '成语' in zh and '一举两得' in zh


def test_the_chinese_sentence_rule_mentions_the_comma_chain():
    # The measured failure: splitting Chinese on 。！？； alone still yields one
    # segment, because clauses are chained with 逗号.
    assert '逗号' in cj._lang_abs_prompt('X', 'zh')


def test_the_gloss_form_anchor_names_the_agreed_shape():
    assert 'attention（注意力）' in cj._lang_abs_prompt('X', 'zh')


def test_the_interest_judge_reader_clause_changes_with_lang():
    assert '中文母语' in cj._interest_abs_sys('zh')
    assert '中文母语' not in cj._interest_abs_sys('en')


def test_the_structure_judge_localizes_its_reader_clause():
    assert '中文母语' in cj._struct_sys('zh')
    assert '中文母语' not in cj._struct_sys('en')
    assert 'ENGLISH IS A SECOND' not in cj._struct_sys('zh')


def test_the_structure_judge_carries_chinese_analogy_exemplars():
    # The English exemplar list (valve, dimmer, see-saw, pizza) is Anglo-Western;
    # a Chinese analogy should be able to be 食堂打饭 or 地铁换乘 without being
    # marked down for not resembling a see-saw.
    assert '食堂打饭' in cj._STRUCT_SYS
    assert 'see-saw' in cj._STRUCT_SYS, 'the English exemplars must not be replaced'


# judge_concept_structure is absent here on purpose: it does NOT use the _chat seam
# (it builds its own client for max_tokens=8000 and reads finish_reason), so it is
# covered by test_the_structure_judge_localizes_its_reader_clause below.
@pytest.mark.parametrize('fn,args', [
    ('judge_plain_language_absolute', ('LESSON',)),
    ('judge_interest_absolute', ('LESSON',)),
])
def test_every_localized_judge_accepts_lang(fn, args):
    seen = {}
    # _chat is called with FOUR POSITIONAL args (system, user, model, timeout). A
    # two-parameter mock raises TypeError, which the judges' never-raises guard
    # swallows into BRIDGE_UNAVAILABLE — so the test saw an empty prompt and
    # "failed" a feature that works.
    cj._chat = lambda *a, **k: seen.update(system=a[0], user=a[1]) or '{}'
    getattr(cj, fn)(*args, lang='zh')
    blob = seen.get('system', '') + seen.get('user', '')
    assert blob, '%s never called _chat — the mock signature does not match' % fn
    assert '中文' in blob or '成语' in blob or '食堂' in blob, \
        '%s took lang="zh" but nothing Chinese reached the prompt' % fn


# =============================================================================
# judge_translation_fidelity
# =============================================================================
# lang_parity_gate can prove the Chinese EXISTS. It cannot read. This is the only
# check that can see the two languages teaching different lessons.

def test_it_is_na_on_an_english_only_source():
    r = cj.judge_translation_fidelity('no chinese at all')
    assert r['overall'] == 'N/A' and r['status'] == 'N/A'


def test_a_dead_bridge_is_reported_not_passed():
    _mock(None)
    r = cj.judge_translation_fidelity(ZH_SRC)
    assert r['status'] == 'BRIDGE_UNAVAILABLE' and r['overall'] == 'N/A'


def test_garbled_output_fails_SAFE():
    # A check whose whole job is to catch a page that LOOKS finished must not read
    # as a pass when the model returns nonsense.
    _mock('not json at all')
    r = cj.judge_translation_fidelity(ZH_SRC)
    assert r['status'] == 'UNPARSEABLE'
    assert r['overall'] == 'FIDELITY_BROKEN'


def test_the_verdict_is_computed_in_code_not_trusted_from_the_model():
    _mock(json.dumps({'dimensions': [{'name': 'same_analogy', 'verdict': 'MISSING'}],
                      'overall': 'FIDELITY_OK'}))
    assert cj.judge_translation_fidelity(ZH_SRC)['overall'] == 'FIDELITY_BROKEN'

    _mock(json.dumps({'dimensions': [{'name': 'a', 'verdict': 'GOOD'},
                                     {'name': 'b', 'verdict': 'WEAK'}],
                      'overall': 'FIDELITY_BROKEN'}))
    assert cj.judge_translation_fidelity(ZH_SRC)['overall'] == 'FIDELITY_OK'


def test_two_weak_levers_break_fidelity():
    _mock(json.dumps({'dimensions': [{'name': 'a', 'verdict': 'WEAK'},
                                     {'name': 'b', 'verdict': 'WEAK'}]}))
    assert cj.judge_translation_fidelity(ZH_SRC)['overall'] == 'FIDELITY_BROKEN'


def test_the_prompt_states_the_shared_drawing_and_term_policy():
    p = cj._fidelity_prompt(ZH_SRC)
    assert 'same_analogy' in p and 'DIFFERENT everyday object' in p
    assert 'attention（注意力）' in p
    assert 'no_untranslated' in p
    assert cj._FIDELITY_SYS.count('SHARED') >= 1


def test_it_reads_the_source_not_the_compiled_page():
    # In source.md each ~~~zh fence sits directly under the English it mirrors,
    # which is the comparison this judge needs; the compiled page interleaves them
    # with markup.
    p = cj._fidelity_prompt('EN\n~~~zh\nZH\n~~~')
    assert '~~~zh' in p
