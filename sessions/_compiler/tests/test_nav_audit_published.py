#!/usr/bin/env python3
"""test_nav_audit_published.py — cover sessions/nav_audit.py --published-only.

Lives in sessions/_compiler/tests/ (not next to nav_audit.py) purely because this
is the one pytest root the publish gates already run, so these tests cannot
silently stop being executed. nav_audit is not part of the compiler; the location
is a wiring decision, not a taxonomy claim.

Each test builds a throwaway git repo in tmp_path — no network, no remote, and
nothing touches the real repo. The fixture is the smallest thing nav_audit will
parse: a MODULES-shaped sessions/index.html plus two chained lesson pages.
"""
import os, sys, subprocess, shutil
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', '..'))          # -> sessions/
import nav_audit                                             # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which('git') is None, reason='git required')

INDEX_HTML = """<!DOCTYPE html><html><body><script>
var MODULES = [
  {n:'M1', lessons:[
    ['Day 1','first','m01/day-01/lesson.html'],
    ['Day 2','second','m01/day-02/lesson.html']
  ]}
];
</script></body></html>"""

# day-01 is the FIRST page: its prev points at the hub, next at day-02.
DAY1 = """<html><body>
<a class="lnav prev" href="../../index.html">Start</a>
<a class="lnav next" href="../day-02/lesson.html">Next</a>
<a class="lnav-hub" href="../../index.html">Map</a>
{extra}
</body></html>"""

# day-02 is the LAST page: next returns to the hub.
DAY2 = """<html><body>
<a class="lnav prev" href="../day-01/lesson.html">Prev</a>
<a class="lnav next" href="../../index.html">Next</a>
<a class="lnav-hub" href="../../index.html">Map</a>
</body></html>"""


def _write(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(body)


def _git(root, *args, check=True):
    r = subprocess.run(['git'] + list(args), cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError('git %s failed: %s' % (' '.join(args), r.stderr))
    return r


def make_repo(tmp_path, day1_extra='', gitignore=None):
    """A committed 2-page fixture repo. Returns its root as a str."""
    root = str(tmp_path)
    _write(os.path.join(root, 'sessions', 'index.html'), INDEX_HTML)
    _write(os.path.join(root, 'sessions', 'm01', 'day-01', 'lesson.html'),
           DAY1.format(extra=day1_extra))
    _write(os.path.join(root, 'sessions', 'm01', 'day-02', 'lesson.html'), DAY2)
    if gitignore is not None:
        _write(os.path.join(root, '.gitignore'), gitignore)
    _git(root, 'init', '-q')
    _git(root, 'add', '-A')
    _git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'fixture')
    return root


def run(root, published):
    argv = ['--root', root] + (['--published-only'] if published else [])
    return nav_audit.main(argv)


# ---------------------------------------------------------------------------
# the fixture itself must be clean, or every other test is meaningless
# ---------------------------------------------------------------------------
def test_clean_fixture_passes_in_both_modes(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, published=False) == 0
    assert run(root, published=True) == 0
    out = capsys.readouterr().out
    assert 'PASS' in out


# ---------------------------------------------------------------------------
# the core behaviour: exists-on-disk is NOT good enough for Pages
# ---------------------------------------------------------------------------
def test_default_mode_accepts_untracked_target(tmp_path):
    """Default mode asks the filesystem, so an untracked target looks fine."""
    root = make_repo(tmp_path, day1_extra='<iframe src="../../viz/toy.html"></iframe>')
    _write(os.path.join(root, 'sessions', 'viz', 'toy.html'), '<html>toy</html>')
    assert run(root, published=False) == 0


def test_published_only_flags_untracked_target(tmp_path, capsys):
    """Published mode asks the index: the file exists locally and 404s on Pages."""
    root = make_repo(tmp_path, day1_extra='<iframe src="../../viz/toy.html"></iframe>')
    _write(os.path.join(root, 'sessions', 'viz', 'toy.html'), '<html>toy</html>')
    assert run(root, published=True) == 1
    out = capsys.readouterr().out
    assert '../../viz/toy.html' in out
    assert 'untracked, exists on disk' in out


def test_published_only_ignores_untracked_source_page(tmp_path, capsys):
    """An untracked scratch page's broken links are not a publish problem.

    This is the fix for the 6 false BROKEN failures the real repo reports from
    sessions/_coldgen/ and sessions/_compare/."""
    root = make_repo(tmp_path)
    _write(os.path.join(root, 'sessions', '_scratch', 'lesson.html'),
           '<a href="../nowhere/gone.html">dead</a>')
    assert run(root, published=False) == 1          # default mode still complains
    assert 'gone.html' in capsys.readouterr().out
    assert run(root, published=True) == 0           # publish gate does not
    assert 'gone.html' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# case-exactness — macOS resolves both spellings, GitHub Pages does not
# ---------------------------------------------------------------------------
def test_published_only_is_case_exact(tmp_path, capsys):
    root = make_repo(tmp_path, day1_extra='<iframe src="../../viz/foo.html"></iframe>')
    _write(os.path.join(root, 'sessions', 'viz', 'Foo.html'), '<html>F</html>')
    _git(root, 'add', '-A')
    _git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'add Foo')
    assert run(root, published=True) == 1
    out = capsys.readouterr().out
    assert '../../viz/foo.html' in out
    assert 'CASE mismatch' in out
    assert 'sessions/viz/Foo.html' in out           # names the real index spelling


# ---------------------------------------------------------------------------
# the .gitignore trap: present on disk, invisible to git, unpublishable
# ---------------------------------------------------------------------------
def test_published_only_diagnoses_gitignored_target(tmp_path, capsys):
    root = make_repo(tmp_path, day1_extra='<img src="../../payload.json">',
                     gitignore='*.json\n')
    _write(os.path.join(root, 'sessions', 'payload.json'), '{}')
    assert run(root, published=True) == 1
    out = capsys.readouterr().out
    assert 'ignored by .gitignore' in out


# ---------------------------------------------------------------------------
# a chain page that was never committed
# ---------------------------------------------------------------------------
def test_published_only_flags_unpublished_canon_page(tmp_path, capsys):
    root = make_repo(tmp_path)
    _write(os.path.join(root, 'sessions', 'm01', 'day-03', 'lesson.html'), DAY2)
    assert run(root, published=True) == 1
    out = capsys.readouterr().out
    assert 'UNPUBLISHED' in out
    assert 'm01/day-03/lesson.html' in out


# ---------------------------------------------------------------------------
# soundness precondition is surfaced, not silently assumed
# ---------------------------------------------------------------------------
def test_published_only_warns_when_worktree_is_dirty(tmp_path, capsys):
    root = make_repo(tmp_path)
    p = os.path.join(root, 'sessions', 'm01', 'day-02', 'lesson.html')
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write('<!-- edited, not committed -->')
    run(root, published=True)
    assert 'WARNING' in capsys.readouterr().out


def test_default_mode_does_not_warn_when_dirty(tmp_path, capsys):
    root = make_repo(tmp_path)
    p = os.path.join(root, 'sessions', 'm01', 'day-02', 'lesson.html')
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write('<!-- edited -->')
    run(root, published=False)
    assert 'WARNING' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# tracked_paths() plumbing
# ---------------------------------------------------------------------------
def test_tracked_paths_handles_spaces_in_filenames(tmp_path):
    """core.quotepath quotes such names; splitlines() would mangle them into
    phantom 'untracked' paths. -z is the reason this passes."""
    root = make_repo(tmp_path)
    _write(os.path.join(root, 'sessions', 'a file.html'), '<html></html>')
    _git(root, 'add', '-A')
    _git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'spaces')
    assert 'sessions/a file.html' in nav_audit.tracked_paths(root)


def test_tracked_paths_raises_outside_a_repo(tmp_path):
    """A real directory that is not a git repo: git exits 128, we turn that into
    a SystemExit rather than silently treating the index as empty (which would
    make every single link look untracked)."""
    outside = tmp_path / 'plain-dir'
    outside.mkdir()
    with pytest.raises(SystemExit):
        nav_audit.tracked_paths(str(outside))


# ---------------------------------------------------------------------------
# output shape: deterministic, and default mode gained no new sections
# ---------------------------------------------------------------------------
def test_default_mode_output_is_golden(tmp_path, capsys):
    """Exact stdout. ORPHANS is 0 because both lesson pages link the hub, so
    index.html is reached; the lesson pages are reached via the chain."""
    root = make_repo(tmp_path)
    assert run(root, published=False) == 0
    assert capsys.readouterr().out == (
        '=' * 64 + '\n'
        'nav_audit — 3 pages, chain of 2\n'
        + '=' * 64 + '\n'
        '\n### CHAIN problems: 0\n'
        '\n### BROKEN links: 0\n'
        '\n### CASE mismatches (GitHub Pages): 0\n'
        '\n### ORPHANS (informational): 0\n'
        '\nPASS — all pages wired together.\n'
    )


def test_published_mode_labels_its_header_and_adds_one_section(tmp_path, capsys):
    root = make_repo(tmp_path)
    run(root, published=True)
    out = capsys.readouterr().out
    assert '[published-only]' in out
    assert 'UNPUBLISHED (in chain, not in git index)' in out


def test_broken_link_order_is_deterministic(tmp_path, capsys):
    """set-union iteration order varies with PYTHONHASHSEED; sorted() pins it."""
    root = make_repo(tmp_path, day1_extra='<a href="../../b.html">b</a>'
                                          '<a href="../../a.html">a</a>'
                                          '<a href="../../c.html">c</a>')
    seen = []
    for _ in range(3):
        run(root, published=False)
        seen.append([l for l in capsys.readouterr().out.splitlines() if '->' in l])
    assert seen[0] == seen[1] == seen[2]
    assert seen[0] == sorted(seen[0])
