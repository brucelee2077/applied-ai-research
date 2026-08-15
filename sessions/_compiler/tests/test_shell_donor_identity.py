import os, sys, glob, re, hashlib
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import pytest
import v8lib
import concept_shell_gate
import shell_invariant_gate as sig

REPO = os.path.join(HERE, '..', '..', '..')
SHELLS = os.path.join(HERE, '..', 'shells')


def _compiled_days():
    """(source.md, lesson.html, donor path) for every day compiled from a source."""
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, 'sessions', 'm*', 'day-*', 'source.md'))):
        meta, _ = v8lib.split_frontmatter(open(p, encoding='utf-8').read())
        dn = meta.get('donor')
        if not dn:
            continue
        dp = os.path.join(SHELLS, dn)
        L = os.path.join(os.path.dirname(p), 'lesson.html')
        if os.path.exists(dp) and os.path.exists(L):
            out.append((p, L, dp, meta))
    return out


COMPILED = _compiled_days()


def test_there_are_compiled_days_to_check():
    # A glob that silently matches nothing turns every test below into a no-op.
    assert len(COMPILED) >= 47, 'expected >=47 compiled days, found %d' % len(COMPILED)


# =============================================================================
# The shell invariant, stated over the equivalence classes that actually exist.
# =============================================================================
# Measured before writing this file: of the 297 pages carrying the sidebar shell,
# only the 47 compiled from a source.md share v9-base.donor's CSS (md5 df4ecc63).
# The other 250 are hand-written or migrated by _shell_migrate.py from a different
# template and legitimately fall into 13 further CSS classes (203 + 29 + 11
# singletons incl. index.html and roadmap.html). So "every shell page's CSS equals
# the donor" is FALSE and asserting it would be a broken gate, not a strict one.
# What IS true, and what the bilingual shell change must preserve:
#   * each compiled lesson's CSS is byte-identical to ITS OWN donor
#   * the donor's <script> blocks appear, data-masked, as an ordered subsequence
#     of the lesson's — 9 of the 47 carry extra inline-lab scripts, interleaved
#     between the donor's own, so a count comparison would be wrong.

@pytest.mark.parametrize('src,lesson,donor,meta', COMPILED,
                         ids=[os.path.relpath(c[1], os.path.join(REPO, 'sessions')) for c in COMPILED])
def test_compiled_lesson_css_is_byte_identical_to_its_donor(src, lesson, donor, meta):
    dc, _ = sig._shell_regions(open(donor, encoding='utf-8').read())
    hc, _ = sig._shell_regions(open(lesson, encoding='utf-8').read())
    assert dc is not None and hc is not None, 'no <style> block found'
    assert dc == hc, 'CSS drifted from %s' % os.path.basename(donor)


@pytest.mark.parametrize('src,lesson,donor,meta', COMPILED,
                         ids=[os.path.relpath(c[1], os.path.join(REPO, 'sessions')) for c in COMPILED])
def test_donor_scripts_are_an_ordered_subsequence_of_the_lesson(src, lesson, donor, meta):
    _, ds = sig._shell_regions(open(donor, encoding='utf-8').read())
    _, hs = sig._shell_regions(open(lesson, encoding='utf-8').read())
    dm = [sig._mask_data(s) for s in ds]
    hm = [sig._mask_data(s) for s in hs]
    remaining = iter(hm)
    missing = [i for i, s in enumerate(dm) if not any(s == cand for cand in remaining)]
    assert not missing, ('donor script index %s missing or out of order '
                         '(donor has %d, lesson has %d)' % (missing, len(ds), len(hs)))


def test_all_47_compiled_lessons_share_one_css_md5():
    # The property that makes a donor edit verifiable in one pass: if this ever
    # splits into two classes, some pages did not get the change.
    md5s = {}
    for _src, lesson, _donor, _meta in COMPILED:
        c, _ = sig._shell_regions(open(lesson, encoding='utf-8').read())
        md5s.setdefault(hashlib.md5(c.encode()).hexdigest()[:8], []).append(
            os.path.relpath(lesson, os.path.join(REPO, 'sessions')))
    assert len(md5s) == 1, 'compiled lessons split into %d CSS classes: %s' % (
        len(md5s), {k: v[:3] for k, v in md5s.items()})


# =============================================================================
# The gate that enforces it must actually be reachable.
# =============================================================================
# concept_shell_gate.run took a `donor=` kwarg and never used it, and its CLI
# never passed one, so mode:concept lessons had no byte-identity check at all.
# Both halves of that no-op are pinned here.

def _mini():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(SHELLS, 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta, donor


def test_gate_passes_when_the_shell_is_intact():
    html, meta, donor = _mini()
    ok, msgs = concept_shell_gate.run(html, meta, donor=donor)
    assert ok, '\n'.join(msgs)
    assert any('CSS block byte-identical' in m for m in msgs), 'the donor check did not run'


def test_gate_catches_a_css_edit_that_missed_the_donor():
    html, meta, donor = _mini()
    tampered = html.replace('.theme-row{padding:10px 8px 0}',
                            '.theme-row{padding:10px 8px 0}.lang-row{padding:10px 8px 0}', 1)
    assert tampered != html, 'the CSS anchor moved — update this test'
    ok, msgs = concept_shell_gate.run(tampered, meta, donor=donor)
    assert not ok
    assert any(m.startswith('FAIL') and 'CSS block' in m for m in msgs), msgs


def test_gate_catches_a_missing_donor_script():
    html, meta, donor = _mini()
    _, hs = sig._shell_regions(html)
    tampered = html.replace(hs[-1], '', 1)
    ok, msgs = concept_shell_gate.run(tampered, meta, donor=donor)
    assert not ok
    assert any(m.startswith('FAIL') and 'donor scripts' in m for m in msgs), msgs


def test_gate_tolerates_an_extra_author_script():
    # 9 of the 47 ship inline-lab scripts interleaved with the donor's.
    html, meta, donor = _mini()
    extra = '<script>(function(){/* inline lab */})();</script>'
    tampered = html.replace('<main id="content">', extra + '<main id="content">', 1)
    ok, msgs = concept_shell_gate.run(tampered, meta, donor=donor)
    assert ok, '\n'.join(msgs)


def test_gate_without_a_donor_skips_the_check_rather_than_passing_it():
    html, meta, _donor = _mini()
    ok, msgs = concept_shell_gate.run(html, meta)
    assert ok
    assert not any('byte-identical' in m for m in msgs), \
        'with no donor the check must be absent, not a silent pass'
