import os, sys, glob, re
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import v8lib
import concept_shell_gate
import shell_invariant_gate

REPO = os.path.join(HERE, '..', '..', '..')


def _compile_mini():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(HERE, '..', 'shells', 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta


# =============================================================================
# 1. _kv field names must start with an ASCII letter or underscore
# =============================================================================
# `\w+` matches CJK in Python 3, so any Chinese line ending in an ASCII colon
# opened a PHANTOM field inside a widget body and the prose after the colon
# vanished from the rendered page with NO error. This is the single most
# dangerous shape for a bilingual source: silent content loss.

def test_chinese_line_with_ascii_colon_does_not_open_a_field():
    d = v8lib._kv(['take: the answer is 24', '这一步很重要:因为它决定了价格'])
    assert list(d.keys()) == ['take'], 'a Chinese line opened a phantom field: %r' % d
    # the Chinese text is not lost — it folds into the previous field
    assert '这一步很重要' in d['take']


def test_chinese_line_with_full_width_colon_also_folds():
    # A full-width colon never matched the opener even before the fix. Pinning it
    # so the two colon widths cannot diverge later.
    d = v8lib._kv(['take: the answer is 24', '这一步很重要：因为它决定了价格'])
    assert list(d.keys()) == ['take']
    assert '这一步很重要：因为它决定了价格' in d['take']


def test_chinese_line_alone_in_a_body_is_not_swallowed():
    # No preceding field at all: the line has nowhere to fold, but it must not
    # become a field name either.
    d = v8lib._kv(['注意:这里很重要'])
    assert d == {}, 'expected no fields, got %r' % d


@pytest.mark.parametrize('name', [
    'code', 'out', 'take', 'why', 'step', 'predict', 'label', 'expr', 'note',
    'words', 'formula', 'numbers', 'sanity', 'title', 'cap', 'src', 'q',
    'concept', 't1', 't2', 't3', 't4',
])
def test_every_real_field_name_still_opens_a_field(name):
    d = v8lib._kv(['%s: value here' % name])
    assert d == {name: 'value here'}


def test_underscore_prefixed_zh_field_names_open_a_field():
    # Forward-looking: the bilingual grammar adds zh_* twins to widget bodies.
    d = v8lib._kv(['take: english', 'zh_take: 中文'])
    assert d == {'take': 'english', 'zh_take': '中文'}


def test_no_shipped_source_relies_on_a_non_ascii_field_name():
    # Permanent guard: proves the tightened opener changes no shipped parse.
    offenders = []
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        for ln in open(p, encoding='utf-8').read().split('\n'):
            if ln.strip().startswith(('#', '-')):
                continue
            if re.match(r'\s*(\w+):', ln) and not re.match(r'\s*([A-Za-z_]\w*):', ln):
                offenders.append((os.path.relpath(p, REPO), ln.strip()[:70]))
    assert not offenders, 'sources whose parse would change: %r' % offenders[:5]


# =============================================================================
# 2. sub_once must actually count
# =============================================================================
# `re.subn(..., count=1)` caps its returned count at 1, so `if n != 1` could
# only ever catch ZERO matches. Against two or more it silently rewrote the
# first and left the rest — a donor with a second <title> would ship another
# lesson's identity while the gate reported success.

def test_sub_once_raises_on_two_matches():
    text = '<title>A</title> ... <title>B</title>'
    with pytest.raises(RuntimeError) as e:
        v8lib.sub_once(r'<title>.*?</title>', '<title>Z</title>', text, 'title')
    assert 'matched 2 times' in str(e.value)


def test_sub_once_raises_on_zero_matches():
    with pytest.raises(RuntimeError) as e:
        v8lib.sub_once(r'<title>.*?</title>', '<title>Z</title>', 'no title here', 'title')
    assert 'matched 0 times' in str(e.value)


def test_sub_once_replaces_exactly_one():
    out = v8lib.sub_once(r'<title>.*?</title>', '<title>Z</title>', 'x<title>A</title>y', 'title')
    assert out == 'x<title>Z</title>y'


def test_sub_once_counts_matches_with_a_capture_group_pattern():
    # sub_once is generic; a pattern carrying groups must still count MATCHES,
    # not groups (re.findall would return group tuples).
    with pytest.raises(RuntimeError) as e:
        v8lib.sub_once(r'<a href="([^"]*)">(.*?)</a>', 'X',
                       '<a href="1">one</a><a href="2">two</a>', 'link')
    assert 'matched 2 times' in str(e.value)


def test_every_donor_matches_each_substituted_region_exactly_once():
    # The invariant the strict sub_once now enforces, asserted across the corpus
    # so a future donor edit that duplicates a region fails here and not in a
    # half-written lesson.
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        meta, _ = v8lib.split_frontmatter(open(p, encoding='utf-8').read())
        dn = meta.get('donor')
        dp = os.path.join(HERE, '..', 'shells', dn) if dn else None
        if not dp or not os.path.exists(dp):
            continue
        donor = open(dp, encoding='utf-8').read()
        if meta.get('mode') == 'concept':
            names = ('title', 'brand_sub', 'nav_prev', 'nav_next')
        else:
            names = tuple(v8lib.REGION_PATTERNS)
        for name in names:
            n = sum(1 for _ in re.finditer(v8lib.REGION_PATTERNS[name], donor, re.DOTALL))
            assert n == 1, '%s: region %s matches %d times' % (dn, name, n)


# =============================================================================
# 3. `~~~` is a leaked marker
# =============================================================================
# render_md consumes `~~~html` and `~~~zh`, but a fence whose terminator the
# author forgot falls through to the paragraph branch and ships as literal text:
#   render_md("English.\n\n~~~zh\n中文。\n~~~")
#     -> '<p>English.</p><p>~~~zh 中文。</p><p>~~~</p>'
# Nothing caught that before: the marker lists held @@@ and %%% but not ~~~.

def test_concept_shell_gate_catches_a_leaked_tilde_fence():
    html, meta = _compile_mini()
    ok, msgs = concept_shell_gate.run(html, meta)
    assert ok, 'fixture should pass before mutation:\n' + '\n'.join(msgs)
    leaked = html.replace('<main id="content">', '<main id="content"><p>~~~zh</p>')
    ok2, msgs2 = concept_shell_gate.run(leaked, meta)
    assert not ok2
    assert any('~~~' in m and m.startswith('FAIL') for m in msgs2), msgs2


def test_no_shipped_lesson_contains_a_tilde_fence():
    offenders = [os.path.relpath(p, REPO)
                 for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'lesson.html')))
                 if '~~~' in open(p, encoding='utf-8').read()]
    assert not offenders, 'lessons shipping a raw fence: %r' % offenders


def test_shell_invariant_gate_names_the_leaked_marker():
    # The old check ANDed four conditions into one boolean, so a failure said
    # "no unresolved markers" and left you to find which one.
    ok, msgs = shell_invariant_gate.run('<html>@@@ stray and ~~~ too</html>', {})
    assert not ok
    line = [m for m in msgs if 'unresolved markers' in m][0]
    assert "'@@@'" in line and "'~~~'" in line, line
