import os, sys, glob, re
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import pytest

REPO = os.path.join(HERE, '..', '..', '..')
DONOR = os.path.join(HERE, '..', 'shells', 'v9-base.donor')

# Every piece the reading-language toggle needs. A page carrying some of these and
# not others is the failure mode of a partial rollout: the row renders but nothing
# happens on click, or the CSS hides Chinese that no button can reveal.
PIECES = {
    'prepaint':      "set reading language before paint",
    'prepaint-lang': "setAttribute('lang'",
    'css-row':       ".lang-row{",
    'css-btn':       ".lang-btn{",
    'css-hide-zh':   'html[data-lang="en"] .lang-zh{display:none}',
    'css-hide-en':   'html[data-lang="zh"] .lang-en{display:none}',
    'markup-row':    'class="lang-row"',
    'markup-en':     'data-lang-set="en"',
    'markup-zh':     'data-lang-set="zh"',
    'js-setlang':    'function setLang(',
    'js-haszh':      'var hasZh =',
    'js-store':      "localStorage.setItem('frontier-lang'",
    'js-checklist':  'function buildChecklist(',
    'js-seclabel':   'function secLabel(',
}


def _shell_pages():
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', '**', '*.html'), recursive=True)):
        rel = os.path.relpath(p, os.path.join(REPO, 'sessions'))
        if rel.startswith(('_coldgen', '_compare')):      # A/B scratch, never published
            continue
        h = open(p, encoding='utf-8').read()
        if 'class="theme-row"' in h:                      # carries the sidebar shell
            out.append((rel, p))
    return out


def _compiled_lessons():
    out = []
    for s in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        L = os.path.join(os.path.dirname(s), 'lesson.html')
        if os.path.exists(L):
            out.append((os.path.relpath(L, os.path.join(REPO, 'sessions')), L))
    return out


SHELL_PAGES = _shell_pages()
COMPILED = _compiled_lessons()


def test_the_globs_matched_something():
    # A glob that silently matches nothing turns every parametrized test below
    # into zero test cases, which reads as a pass.
    assert len(COMPILED) >= 47, 'expected >=47 compiled lessons, got %d' % len(COMPILED)
    assert len(SHELL_PAGES) >= 290, 'expected >=290 shell pages, got %d' % len(SHELL_PAGES)


def test_the_donor_has_every_piece():
    donor = open(DONOR, encoding='utf-8').read()
    missing = [k for k, needle in PIECES.items() if needle not in donor]
    assert not missing, 'v9-base.donor is missing %s' % missing


@pytest.mark.parametrize('rel,path', COMPILED, ids=[c[0] for c in COMPILED])
def test_compiled_lesson_has_every_toggle_piece(rel, path):
    h = open(path, encoding='utf-8').read()
    missing = [k for k, needle in PIECES.items() if needle not in h]
    assert not missing, 'missing %s — recompile from source.md' % missing


@pytest.mark.parametrize('rel,path', SHELL_PAGES, ids=[c[0] for c in SHELL_PAGES])
def test_no_shell_page_has_a_half_applied_toggle(rel, path):
    # Holds before AND after the sweep: a page either has the whole toggle or
    # none of it. Half of it is the state that renders a dead button.
    h = open(path, encoding='utf-8').read()
    present = [k for k, needle in PIECES.items() if needle in h]
    assert present == [] or len(present) == len(PIECES), \
        'partial toggle — has %s, missing %s' % (
            sorted(present), sorted(set(PIECES) - set(present)))


def test_the_theme_switcher_was_not_disturbed():
    # _shell_audit.py asserts count("theme-btn") >= 4; the new class is lang-btn
    # precisely so it cannot inflate that count.
    for rel, path in COMPILED:
        h = open(path, encoding='utf-8').read()
        assert h.count('theme-btn') >= 4, '%s: theme buttons lost' % rel
        assert h.count('data-theme-set=') == 4, '%s: theme options changed' % rel
        assert 'frontier-theme' in h, '%s: theme storage key lost' % rel


def test_the_two_storage_keys_stay_separate():
    donor = open(DONOR, encoding='utf-8').read()
    # frontier-lang must never be read where frontier-theme is expected, and the
    # whitelists must not be shared: theme has 4 valid values, language has 2.
    assert "['en','zh']" in donor.replace(' ', ''), 'language whitelist missing'
    assert "['light','dim','dark','midnight']" in donor.replace(' ', ''), 'theme whitelist changed'


def test_default_language_is_english():
    donor = open(DONOR, encoding='utf-8').read()
    m = re.search(r"getItem\('frontier-lang'\);if\(\['en','zh'\]\.indexOf\(l\)<0\)l='(\w+)'", donor.replace(' ', ''))
    assert m, 'could not find the language fallback in the pre-paint script'
    assert m.group(1) == 'en', "default reading language is %r, expected 'en'" % m.group(1)
