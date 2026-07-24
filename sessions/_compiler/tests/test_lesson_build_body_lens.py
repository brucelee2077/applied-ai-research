import os

HERE = os.path.dirname(__file__)
LB = os.path.abspath(os.path.join(HERE, '..', 'workflows', 'lesson_build.js'))


def _src():
    with open(LB, encoding='utf-8') as f:
        return f.read()


def test_body_lens_present():
    s = _src()
    assert "key: 'body'" in s                       # a dedicated body-engagement lens
    assert 'Concept Body Engagement' in s            # it parses the CLI section
    # MISSING -> P0, WEAK -> P1, kind body_engagement
    assert 'body_engagement' in s
    assert 'MISSING' in s and 'P0' in s


def test_kind_enum_includes_body_engagement():
    s = _src()
    # find the JUDGE_SCHEMA kind description and confirm body_engagement is listed
    assert 'body_engagement' in s
    # it should appear in the kind enum description alongside the other kinds
    assert 'skill_gap' in s and 'correctness' in s


def test_author_prompt_keeps_body_alive():
    s = _src()
    assert 'KEEP THE BODY ALIVE' in s
    assert '%%% insight' in s and '%%% steps' in s
    assert 'predict' in s.lower()
