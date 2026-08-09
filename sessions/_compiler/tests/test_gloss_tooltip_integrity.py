import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import re
import v8lib


def _tips(html):
    return re.findall(r'data-tip="([^"]*)"', html)


# Both shapes below were real defects on shipped m04 lessons (4 corrupted tooltips
# across 3 files). Cause: the [[term||gloss]] substitution ran BEFORE the ** / *
# emphasis rules, so those rewrote text INSIDE the finished data-tip attribute, and
# attr_esc did not escape < or >.
def test_emphasis_inside_a_gloss_does_not_inject_a_tag():
    out = v8lib.inline('the [[gradient||slopes; steepest *increase*, so step opposite]]: go.')
    tip = _tips(out)[0]
    assert '<em>' not in tip and '</em>' not in tip
    assert '*increase*' in tip          # the author's asterisks survive as literal text


def test_two_glosses_on_one_line_do_not_cross_pair():
    # The worse shape: one `*` in each gloss paired into an <em> SPANNING two
    # different attributes, producing invalid markup and a mangled judge extract.
    out = v8lib.inline('a [[step()||replaces W -= lr*dW lines]] and [[SGD||w := w - lr*grad]] done.')
    tips = _tips(out)
    assert len(tips) == 2
    for tip in tips:
        assert '<em>' not in tip and '</em>' not in tip
    assert 'lr*dW' in tips[0] and 'lr*grad' in tips[1]


def test_no_tooltip_may_contain_angle_brackets():
    # The invariant, stated directly: a data-tip value is plain text.
    out = v8lib.inline('[[argmax||returns the *position* of the largest <value>]] here.')
    for tip in _tips(out):
        assert '<' not in tip and '>' not in tip


def test_attr_esc_escapes_all_four():
    got = v8lib.attr_esc('a & b " c < d > e')
    assert got == 'a &amp; b &quot; c &lt; d &gt; e'


# Regressions: the stash must not break the inline rules it was protecting.
def test_bold_em_and_code_still_work_outside_glosses():
    out = v8lib.inline('plain **bold** and *em* and `code**x` stay.')
    assert '<strong>bold</strong>' in out
    assert '<em>em</em>' in out
    assert '<code>code**x</code>' in out


def test_emphasis_around_a_gloss_still_renders():
    out = v8lib.inline('*before* [[t||gloss]] **after**')
    assert '<em>before</em>' in out
    assert '<strong>after</strong>' in out
    assert 'data-tip="gloss"' in out


def test_gloss_term_text_is_still_the_visible_label():
    out = v8lib.inline('[[weight||how much an input matters]]')
    assert '>weight</span>' in out
