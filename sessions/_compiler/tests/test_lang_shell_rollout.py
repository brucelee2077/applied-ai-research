import os, sys, glob, re
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import pytest

REPO = os.path.join(HERE, '..', '..', '..')
DONOR = os.path.join(HERE, '..', 'shells', 'v9-base.donor')

# Every piece the reading-language toggle needs. A page carrying some of these and
# not others is the failure mode of a partial rollout: the row renders but nothing
# happens on click, or the CSS hides Chinese that no button can reveal.
CORE_PIECES = {
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
}
# Only meaningful on a page that HAS a progress checklist. index.html and
# roadmap.html carry the same sidebar shell but have no #checklist and no
# refresh(), which is why the controller guards both calls with typeof.
CHECKLIST_PIECES = {
    'js-checklist':  'function buildChecklist(',
    'js-seclabel':   'function secLabel(',
}


def _expected(html):
    p = dict(CORE_PIECES)
    if 'id="checklist"' in html:
        p.update(CHECKLIST_PIECES)
    return p


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
    missing = [k for k, needle in _expected(donor).items() if needle not in donor]
    assert not missing, 'v9-base.donor is missing %s' % missing


@pytest.mark.parametrize('rel,path', COMPILED, ids=[c[0] for c in COMPILED])
def test_compiled_lesson_has_every_toggle_piece(rel, path):
    h = open(path, encoding='utf-8').read()
    missing = [k for k, needle in _expected(h).items() if needle not in h]
    assert not missing, 'missing %s — recompile from source.md' % missing


@pytest.mark.parametrize('rel,path', SHELL_PAGES, ids=[c[0] for c in SHELL_PAGES])
def test_no_shell_page_has_a_half_applied_toggle(rel, path):
    # Holds before AND after the sweep: a page either has the whole toggle or
    # none of it. Half of it is the state that renders a dead button.
    h = open(path, encoding='utf-8').read()
    want = _expected(h)
    present = [k for k, needle in want.items() if needle in h]
    assert present == [] or len(present) == len(want), \
        'partial toggle — has %s, missing %s' % (
            sorted(present), sorted(set(want) - set(present)))


@pytest.mark.parametrize('rel,path', SHELL_PAGES, ids=[c[0] for c in SHELL_PAGES])
def test_every_shell_page_carries_the_toggle(rel, path):
    # The rollout invariant. Asserted per page rather than as a count, so the
    # failure names the page that was missed instead of just a number.
    h = open(path, encoding='utf-8').read()
    assert 'class="lang-row"' in h, 'no language row — run sessions/_lang_shell_sweep.py --apply'


# The behavioural test (test_lang_switch.mjs) extracts its code from the DONOR.
# That only proves anything about the 293 shipped pages if what they carry is the
# same text. These two assert exactly that, which is what makes _lang_shell_sweep
# slicing from the donor rather than hardcoding meaningful.
def _sweep_parts():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_lang_shell_sweep', os.path.join(REPO, 'sessions', '_lang_shell_sweep.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.donor_parts()


PARTS = _sweep_parts()


@pytest.mark.parametrize('rel,path', SHELL_PAGES, ids=[c[0] for c in SHELL_PAGES])
def test_page_toggle_text_is_byte_identical_to_the_donor(rel, path):
    h = open(path, encoding='utf-8').read()
    need = ['prepaint', 'css', 'markup', 'controller']
    if 'id="checklist"' in h:
        need.append('checklist')
    drifted = [k for k in need if PARTS[k] not in h]
    assert not drifted, ('%s drifted from the donor — the behavioural test in '
                         'test_lang_switch.mjs no longer describes this page' % drifted)


def test_the_donor_parts_are_not_empty():
    # A slice helper that silently returned '' would make the test above pass on
    # every page while checking nothing.
    for k, v in PARTS.items():
        assert v and len(v) > 20, 'donor part %r is empty or truncated: %r' % (k, v[:40])


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


# =============================================================================
# runtime UI strings
# =============================================================================
# These are the strings no CSS toggle can reach, because the code REPLACES
# textContent instead of showing one of two nodes. Without them, a Chinese reader
# would watch the page flip back to English the moment they pressed anything.

UI_KEYS = ['reveal_done', 'all_answered', 'hints_end', 'hint_more',
           'copied', 'copy_manual', 'reset_confirm', 'sections_done']


def test_the_donor_ui_table_has_both_languages_for_every_key():
    donor = open(DONOR, encoding='utf-8').read()
    block = re.search(r'/\* frontier-lang:ui \*/(.*?)/\* /frontier-lang:ui \*/', donor, re.S)
    assert block, 'no frontier-lang:ui block in the donor'
    en = re.search(r'en:\s*\{(.*?)\},\s*\n?\s*zh:', block.group(1), re.S)
    zh = re.search(r'zh:\s*\{(.*?)\}\s*\n?\};', block.group(1), re.S)
    assert en and zh, block.group(1)[:200]
    for k in UI_KEYS:
        assert k + ':' in en.group(1), 'en table missing %r' % k
        assert k + ':' in zh.group(1), 'zh table missing %r' % k


def test_no_hardcoded_english_runtime_string_survives_on_any_page():
    # Targets the ASSIGNMENTS, not the strings. The English wording still appears on
    # every page — as the `en:` half of the UI table, which is the point. An earlier
    # version of this test searched for the wording itself and flagged the table.
    gone = [
        "run.textContent = 'ran",
        "g.textContent='All answered",
        'btn.textContent = "— that\'s all the hints —"',
        "btn.textContent=ok?'✓ copied'",
        "confirm('Reset today",
        "+' sections done'",
        "btn.textContent = '💡 still stuck? another hint ('",
    ]
    offenders = []
    for rel, path in SHELL_PAGES:
        h = open(path, encoding='utf-8').read()
        for g in gone:
            if g in h:
                offenders.append((rel, g))
    assert not offenders, 'hardcoded runtime assignments left: %r' % offenders[:6]


def test_every_page_that_can_write_those_strings_goes_through_ui():
    # The complement of the test above: prove the replacement actually happened
    # rather than the string merely being absent because the widget is absent.
    # The marker is the demo ENGINE, not `var DEMOS` — v9 concept lessons drive the
    # widget with a generic querySelector('.demo-run') loop and have no DEMOS array.
    n = 0
    for rel, path in SHELL_PAGES:
        h = open(path, encoding='utf-8').read()
        if "querySelector('.demo-run')" in h:
            assert "ui('reveal_done')" in h, '%s runs the demo engine but not ui()' % rel
            n += 1
    assert n >= 40, 'expected ~47 pages to carry the demo engine, saw %d' % n


def test_every_page_with_a_progress_count_goes_through_ui():
    # Keyed on the donor's count-writing CODE, not on the wording: the wording now
    # appears on every page as the UI table's `en:` value, and the hub has its own
    # dashboard that says "lessons complete" from its own render(). Translating the
    # hub's dashboard is a separate job — it also carries module names and
    # capability labels, all English content.
    n = 0
    for rel, path in SHELL_PAGES:
        h = open(path, encoding='utf-8').read()
        if "count.innerHTML = '<b>'+done+'</b>/'+total+" in h:
            assert "ui('sections_done')" in h, '%s writes the count but not via ui()' % rel
            n += 1
    assert n >= 280, 'expected ~291 pages with the donor progress count, saw %d' % n


@pytest.mark.parametrize('rel,path', SHELL_PAGES, ids=[c[0] for c in SHELL_PAGES])
def test_no_page_calls_ui_without_the_table(rel, path):
    # The ordering trap: the sweep inserts the table and the controller at the same
    # anchor, and the string substitutions rewrite call sites elsewhere in the file.
    # A page with calls and no table throws on first interaction.
    h = open(path, encoding='utf-8').read()
    if "ui('" in h:
        assert 'var UI = {' in h, 'calls ui() but has no string table'


def test_the_ui_and_controller_sentinels_do_not_nest():
    # They did: the ui block was inserted inside the controller's sentinel range, so
    # parts['controller'] contained parts['ui']. The table still landed (as part of
    # the controller refresh) but every sweep run reported a spurious
    # "cannot refresh ui" and exited 1.
    assert PARTS['ui'] not in PARTS['controller']
    assert PARTS['controller'] not in PARTS['ui']


def test_the_chrome_is_paired_not_replaced():
    # Chrome is rendered once and never rewritten, so it stays a paired span rather
    # than joining the JS table.
    donor = open(DONOR, encoding='utf-8').read()
    for en, zh in [('Appearance', '外观'), ('Progress checklist', '进度清单'),
                   ('↺ Reset progress', '↺ 清空进度'), ('← Prev', '← 上一天'),
                   ('Next →', '下一天 →'), ('▦ Map', '▦ 地图'),
                   ('Back to curriculum', '回到课程地图')]:
        assert '<span class="lang-en">%s</span><span class="lang-zh">%s</span>' % (en, zh) in donor, en
