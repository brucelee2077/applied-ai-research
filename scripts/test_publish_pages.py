#!/usr/bin/env python3
"""test_publish_pages.py — cover scripts/publish_pages.py.

Two kinds of test here:

  * pure-logic tests, which need nothing at all; and
  * injection tests, which drive main() with a FakeGit and a fake gate runner so
    no git command, no network call and no gate subprocess ever happens.

That second point matters more than it looks: `python3 -m pytest scripts -q` IS
one of publish_pages' own gates, so a test that let main() build a real runner
would recursively re-run the entire gate set. Every test that calls main() passes
runner=.
"""
import os
import subprocess
import shutil
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import publish_pages as pubp                                  # noqa: E402

SHA_A = 'a' * 40
SHA_B = 'b' * 40


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_classify_viz_html_is_committed():
    c = pubp.classify_untracked(['sessions/viz/dk-scaling-dial.html'])
    assert c.commit == ['sessions/viz/dk-scaling-dial.html']
    assert c.skip == [] and c.unclassified == []


def test_classify_donor_shell_is_committed():
    c = pubp.classify_untracked(['sessions/_compiler/shells/m04-day-01.donor'])
    assert c.commit == ['sessions/_compiler/shells/m04-day-01.donor']


def test_classify_lesson_and_source_are_committed():
    c = pubp.classify_untracked(['sessions/m09-x/day-03-y/lesson.html',
                                 'sessions/m09-x/day-03-y/source.md',
                                 'sessions/m09-x/day-03-y/experiment.py'])
    assert len(c.commit) == 3 and not c.unclassified


def test_classify_scratch_trees_are_skipped():
    c = pubp.classify_untracked(['sessions/_coldgen/mlp-mnist/lesson.html',
                                 'sessions/_compare/m02-day02-gold/source.md'])
    assert len(c.skip) == 2
    assert c.commit == [] and c.unclassified == []


def test_classify_deny_beats_allow():
    """A coldgen lesson.html matches BOTH lists. Skip must win, or A/B scratch
    gets published — and .nojekyll means underscore dirs really are served."""
    p = 'sessions/_coldgen/m02-day02-activations/lesson.html'
    assert any(pubp.glob_match(p, g) for g in pubp.SKIP_GLOBS)
    c = pubp.classify_untracked([p])
    assert c.skip == [p] and c.commit == []


def test_classify_glob_respects_slash_boundaries():
    """fnmatch's '*' crosses '/', which would drag a deep unrelated file into an
    auto-commit via 'sessions/*/day-*.html'."""
    assert pubp.glob_match('sessions/m01/day-01.html', 'sessions/*/day-*.html')
    assert not pubp.glob_match('sessions/a/b/c/day-1.html', 'sessions/*/day-*.html')
    c = pubp.classify_untracked(['sessions/a/b/c/day-1.html'])
    assert c.unclassified == ['sessions/a/b/c/day-1.html']


def test_classify_doublestar_matches_at_any_depth():
    assert pubp.glob_match('sessions/m01/day-01/_coverage.md',
                           'sessions/**/_coverage.md')
    assert pubp.glob_match('sessions/_coverage.md', 'sessions/**/_coverage.md')
    assert pubp.glob_match('a/b/__pycache__/x.pyc', '**/__pycache__/**')


def test_classify_unknown_path_is_unclassified():
    """Fail closed: the stray 0-byte sessions/m24-review-cmp must block, not be
    guessed at in either direction."""
    c = pubp.classify_untracked(['sessions/m24-review-cmp', 'notes.txt'])
    assert c.unclassified == ['notes.txt', 'sessions/m24-review-cmp']
    assert c.commit == [] and c.skip == []


def test_classify_output_is_sorted_and_deterministic():
    given = ['sessions/viz/z.html', 'sessions/viz/a.html', 'sessions/viz/m.html']
    first = pubp.classify_untracked(given)
    assert first.commit == sorted(given)
    assert pubp.classify_untracked(list(reversed(given))) == first


# ---------------------------------------------------------------------------
# commit message
# ---------------------------------------------------------------------------
def test_build_commit_message_lists_every_path():
    msg = pubp.build_commit_message(['sessions/viz/b.html', 'sessions/viz/a.html'])
    assert 'commit 2 untracked site assets' in msg
    assert msg.index('sessions/viz/a.html') < msg.index('sessions/viz/b.html')
    assert 'Publish-Gates' not in msg


def test_build_commit_message_singular_and_trailer():
    msg = pubp.build_commit_message(['sessions/viz/a.html'], gates_skipped=True)
    assert 'commit 1 untracked site asset for Pages' in msg
    assert msg.rstrip().endswith('Publish-Gates: SKIPPED')


# ---------------------------------------------------------------------------
# reconcile truth table
# ---------------------------------------------------------------------------
def test_reconcile_action_identical():
    assert pubp.reconcile_action(True, True) == 'identical'


def test_reconcile_action_ahead():
    assert pubp.reconcile_action(True, False) == 'ahead'


def test_reconcile_action_behind():
    assert pubp.reconcile_action(False, True) == 'behind'


def test_reconcile_action_diverged_needs_merge():
    assert pubp.reconcile_action(False, False) == 'merge'


# ---------------------------------------------------------------------------
# pages_url
# ---------------------------------------------------------------------------
def test_pages_url_https_dot_git():
    assert pubp.pages_url('https://github.com/brucelee2077/applied-ai-research.git') == \
        'https://brucelee2077.github.io/applied-ai-research/sessions/index.html'


def test_pages_url_https_without_dot_git():
    assert pubp.pages_url('https://github.com/owner/repo') == \
        'https://owner.github.io/repo/sessions/index.html'


def test_pages_url_ssh():
    assert pubp.pages_url('git@github.com:owner/repo.git') == \
        'https://owner.github.io/repo/sessions/index.html'


def test_pages_url_user_site_has_no_path_segment():
    assert pubp.pages_url('https://github.com/owner/owner.github.io') == \
        'https://owner.github.io/sessions/index.html'


def test_pages_url_unrecognised_is_none():
    assert pubp.pages_url('not-a-url') is None
    assert pubp.pages_url('') is None


# ---------------------------------------------------------------------------
# the absolute-link gate
# ---------------------------------------------------------------------------
def test_absolute_link_violations_flags_root_absolute(tmp_path):
    p = tmp_path / 'a.html'
    p.write_text('<img src="/assets/x.png">')
    viol = pubp.absolute_link_violations([str(p)], str(tmp_path))
    assert viol == [('a.html', '/assets/x.png')]


def test_absolute_link_violations_allows_parent_escape(tmp_path):
    """THE lock-in test. sessions/ has ~779 legitimate '../day-NN/lesson.html'
    links — the entire prev/next chain. If someone 'simplifies' this gate to call
    validate_self_contained (which also rejects '../'), the publish gate starts
    reporting 779 false failures and this test fails first."""
    p = tmp_path / 'a.html'
    p.write_text('<a href="../day-01/lesson.html">prev</a>'
                 '<iframe src="../../viz/toy.html"></iframe>')
    assert pubp.absolute_link_violations([str(p)], str(tmp_path)) == []


def test_absolute_link_violations_allows_external_and_anchors(tmp_path):
    p = tmp_path / 'a.html'
    p.write_text('<link href="https://fonts.googleapis.com/x">'
                 '<link href="//cdn/x.css"><a href="#top">t</a>'
                 '<a href="mailto:x@y.z">m</a><img src="data:image/png;base64,AA">')
    assert pubp.absolute_link_violations([str(p)], str(tmp_path)) == []


def test_absolute_link_violations_catches_unquoted(tmp_path):
    p = tmp_path / 'a.html'
    p.write_text('<a href=/oops.html>x</a>')
    assert pubp.absolute_link_violations([str(p)], str(tmp_path)) == \
        [('a.html', '/oops.html')]


# ---------------------------------------------------------------------------
# required files
# ---------------------------------------------------------------------------
def test_missing_required_all_present():
    assert pubp.missing_required(['.nojekyll', 'index.html',
                                  'sessions/index.html', 'other']) == []


def test_missing_required_reports_nojekyll():
    """Without .nojekyll, Jekyll drops every _-prefixed path from the built site."""
    assert pubp.missing_required(['index.html', 'sessions/index.html']) == ['.nojekyll']


# ---------------------------------------------------------------------------
# gate specs
# ---------------------------------------------------------------------------
def _argvs(gates):
    return [' '.join(g.argv) for g in gates if g.argv]


def test_gate_specs_pass_explicit_pytest_path():
    """pytest.ini sets testpaths=interview_app/tests, so a bare `pytest` gate
    would pass while running an unrelated suite."""
    joined = ' | '.join(_argvs(pubp.gate_specs('/repo')))
    assert 'sessions/_compiler/tests' in joined
    assert 'scripts' in joined


def test_gate_specs_exclude_destructive_scripts():
    """Turns the DANGER list into an executable invariant.

    _density_scan.py and _body_baseline.py take an OUTPUT path as argv[1] and have
    destroyed source files. lesson_audit.py with no args writes _recover_set.json.
    _experiment_check.py --all exits 1 by design while stub days remain.
    staff_lens_audit.js exits 2 on every lesson (it looks for the obsolete .sec
    class). coverage_audit.py globs the retired week-* layout."""
    joined = ' | '.join(_argvs(pubp.gate_specs('/repo')))
    for banned in ('_density_scan', '_body_baseline', 'lesson_audit',
                   '_experiment_check', 'staff_lens_audit', 'coverage_audit',
                   '_body_engagement_scan'):
        assert banned not in joined, '%s must never be a publish gate' % banned


def test_gate_specs_include_published_only_flag():
    joined = ' | '.join(_argvs(pubp.gate_specs('/repo')))
    assert 'nav_audit.py --published-only' in joined


def test_gate_specs_are_ordered_cheap_first():
    names = [g.name for g in pubp.gate_specs('/repo')]
    assert names.index('shelf-audit') < names.index('compiler-tests')
    assert names.index('nav-audit-published') < names.index('compiler-tests')


def test_gate_specs_can_be_filtered_by_name():
    names = [g.name for g in pubp.gate_specs('/repo', skip_names=('compiler-tests',))]
    assert 'compiler-tests' not in names


def test_run_gates_fail_fast_stops_early():
    gates = pubp.gate_specs('/repo')
    calls = []

    def runner(g):
        calls.append(g.name)
        return (g.name != 'shelf-audit'), 'boom'

    res = pubp.run_gates(gates, runner, fail_fast=True)
    assert len(res) == 1 and calls == ['shelf-audit']


def test_run_gates_default_runs_all_and_reports_all():
    gates = pubp.gate_specs('/repo')
    res = pubp.run_gates(gates, lambda g: (False, 'x'))
    assert len(res) == len(gates)
    assert all(not r.ok for r in res)


# ---------------------------------------------------------------------------
# FakeGit — records every call, scripts every answer
# ---------------------------------------------------------------------------
class FakeGit:
    def __init__(self, root='/repo', branch='build/capability-spiral',
                 heads=None, untracked=(), tracked=None, codes=None,
                 captures=None, fail=None, has_remote_ref=True, incoming='',
                 outgoing='260', added=''):
        self.root = root
        self.branch = branch
        self.heads = list(heads or [SHA_A])
        self.head_calls = 0
        self.untracked = list(untracked)
        self.tracked = list(tracked if tracked is not None
                            else ['.nojekyll', 'index.html', 'sessions/index.html'])
        self.codes = dict(codes or {})
        self.captures = dict(captures or {})
        self.fail = dict(fail or {})
        self.has_remote_ref = has_remote_ref
        self.incoming, self.outgoing, self.added = incoming, outgoing, added
        self.log = []

    # -- helpers
    def _key(self, args):
        return ' '.join(args)

    def _maybe_fail(self, args):
        key = self._key(args)
        for prefix, err in self.fail.items():
            if key.startswith(prefix):
                raise pubp.GitError(args, 1, err)

    def _next_head(self):
        h = self.heads[min(self.head_calls, len(self.heads) - 1)]
        self.head_calls += 1
        return h

    # -- the Git interface
    def capture(self, *args):
        self.log.append(args)
        self._maybe_fail(args)
        key = self._key(args)
        if key in self.captures:
            return self.captures[key]
        if key == 'rev-parse --abbrev-ref HEAD':
            return self.branch
        if key == 'rev-parse HEAD':
            return self._next_head()
        if key == 'rev-parse --short HEAD':
            return self.heads[min(self.head_calls, len(self.heads) - 1)][:7]
        if key.startswith('rev-parse --short'):
            return 'deadbee'
        if key == 'ls-files --others --exclude-standard -z':
            return '\0'.join(self.untracked)
        if key.startswith('ls-files -z'):
            return '\0'.join(self.tracked)
        if key.startswith('log --oneline'):
            return self.incoming
        if key.startswith('rev-list --count'):
            return self.outgoing
        if key.startswith('diff --name-status'):
            return self.added
        if key == 'diff --stat':
            return ' sessions/x.html | 2 +-'
        if key.startswith('diff --name-only'):
            return 'sessions/conflicted.html'
        if key == 'remote get-url origin':
            return 'https://github.com/owner/repo.git'
        return ''

    def code(self, *args):
        self.log.append(args)
        key = self._key(args)
        if key in self.codes:
            return self.codes[key]
        if key.startswith('rev-parse --verify --quiet refs/remotes'):
            return 0 if self.has_remote_ref else 1
        return 0

    def index_op(self, *args):
        self.log.append(args)
        self._maybe_fail(args)
        return ''

    def mutate(self, *args):
        self.log.append(args)
        self._maybe_fail(args)
        return ''

    # -- assertions
    def subcommands(self):
        return [a[0] for a in self.log]

    def calls_matching(self, prefix):
        return [a for a in self.log if self._key(a).startswith(prefix)]


def ok_runner(gate):
    return True, ''


def bad_runner(gate):
    return (gate.name != 'shelf-audit'), 'exit 1'


# ---------------------------------------------------------------------------
# holds and the quiet window — concurrent-session protection
# ---------------------------------------------------------------------------
def test_apply_holds_removes_matching_paths():
    keep, held = pubp.apply_holds(
        ['sessions/viz/a.html', 'sessions/m08-x/day-01-y/source.md'],
        ['sessions/m08-x/**'])
    assert keep == ['sessions/viz/a.html']
    assert held == ['sessions/m08-x/day-01-y/source.md']


def test_apply_holds_without_globs_is_identity():
    paths = ['sessions/viz/a.html']
    assert pubp.apply_holds(paths, []) == (paths, [])


def test_partition_warm_holds_recently_modified():
    """A file written seconds ago is probably still being written."""
    cold, warm = pubp.partition_warm(
        ['old.html', 'new.html'], window=120,
        mtime={'old.html': 1000.0, 'new.html': 1990.0}.__getitem__, now=2000.0)
    assert cold == ['old.html'] and warm == ['new.html']


def test_partition_warm_window_zero_disables():
    cold, warm = pubp.partition_warm(['new.html'], window=0,
                                     mtime=lambda p: 1999.0, now=2000.0)
    assert cold == ['new.html'] and warm == []


def test_partition_warm_treats_unstattable_as_cold():
    def boom(p):
        raise OSError('gone')
    cold, warm = pubp.partition_warm(['x.html'], window=120, mtime=boom, now=1.0)
    assert cold == ['x.html'] and warm == []


def test_held_path_is_not_staged_or_committed(capsys):
    g = FakeGit(untracked=['sessions/viz/toy.html',
                           'sessions/m08-x/day-01-y/source.md'])
    assert pubp.main(['--publish', '--hold', 'sessions/m08-x/**',
                      '--quiet-window', '0'], git=g, runner=ok_runner) == pubp.EX_OK
    adds = g.calls_matching('add --')
    assert list(adds[0][2:]) == ['sessions/viz/toy.html']
    assert 'held by --hold' in capsys.readouterr().out


@pytest.mark.skipif(shutil.which('git') is None, reason='git required')
def test_warm_path_is_held_by_default(tmp_path, capsys):
    """End-to-end through main with a real mtime: a just-written asset is held."""
    root = _init_repo(tmp_path)
    os.makedirs(os.path.join(root, 'sessions', 'viz'), exist_ok=True)
    open(os.path.join(root, 'sessions', 'viz', 'fresh.html'), 'w').write('<html></html>')
    g = FakeGit(root=root, untracked=['sessions/viz/fresh.html'])
    pubp.main([], git=g, runner=ok_runner)
    out = capsys.readouterr().out
    assert 'another session may still be writing it' in out
    assert not g.calls_matching('add --')


# ---------------------------------------------------------------------------
# step ORDER — git itself enforces this, and getting it wrong aborted a real
# publish run: `git merge` refuses to run with a dirty index, so staging the new
# assets before the merge makes the merge fail.
# ---------------------------------------------------------------------------
def _order(g, *prefixes):
    """Index of the first call matching each prefix, in log order."""
    keys = [' '.join(a) for a in g.log]
    out = []
    for pre in prefixes:
        out.append(next((i for i, k in enumerate(keys) if k.startswith(pre)), None))
    return out


def test_merge_happens_before_staging():
    g = FakeGit(untracked=['sessions/viz/toy.html'],
                codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})
    pubp.main(['--publish', '--quiet-window', '0'], git=g, runner=ok_runner)
    i_merge, i_add = _order(g, 'merge --no-edit', 'add --')
    assert i_merge is not None and i_add is not None
    assert i_merge < i_add, 'staged before merging; git merge needs a clean index'


def test_gates_run_after_the_merge_so_they_check_the_published_tree():
    order = []
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})

    def tracking_runner(gate):
        order.append(('gate', len(g.log)))
        return True, ''

    pubp.main(['--publish'], git=g, runner=tracking_runner)
    i_merge = _order(g, 'merge --no-edit')[0]
    assert order and i_merge is not None
    assert order[0][1] > i_merge, 'gates ran before the merge'


def test_commit_happens_after_the_gates():
    g = FakeGit(untracked=['sessions/viz/toy.html'],
                codes={'diff --cached --quiet': 1})
    seen = {'gated': False, 'committed_before_gates': False}

    def runner(gate):
        seen['gated'] = True
        return True, ''

    orig_mutate = g.mutate

    def mutate(*args):
        if args[0] == 'commit' and not seen['gated']:
            seen['committed_before_gates'] = True
        return orig_mutate(*args)

    g.mutate = mutate
    pubp.main(['--publish', '--quiet-window', '0'], git=g, runner=runner)
    assert not seen['committed_before_gates']


def test_commit_all_commits_before_the_merge():
    """git merge needs a clean WORKTREE too, so work-in-progress must land first."""
    g = FakeGit(codes={'diff --quiet': 1,
                       'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1},
                captures={'diff --name-only -z': 'sessions/x.html'})
    pubp.main(['--publish', '--commit-all', 'wip'], git=g, runner=ok_runner)
    i_commit, i_merge = _order(g, 'commit -m', 'merge --no-edit')
    assert i_commit is not None and i_merge is not None
    assert i_commit < i_merge


def test_merge_blocked_by_dirty_index_is_reported_distinctly(capsys):
    """git's 'Your local changes...' and 'untracked working tree files...' share
    the phrase 'would be overwritten by merge' but need different fixes."""
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1},
                fail={'merge --no-edit':
                      'error: Your local changes to the following files would be '
                      'overwritten by merge:\n  sessions/viz/toy.html'})
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_RECONCILE
    out = capsys.readouterr().out
    assert 'index or worktree is not clean' in out
    assert 'concurrent' in out
    assert not g.calls_matching('merge --abort')


# ---------------------------------------------------------------------------
# dry run must not mutate anything
# ---------------------------------------------------------------------------
def test_dry_run_never_merges_commits_or_pushes(capsys):
    g = FakeGit(untracked=['sessions/viz/toy.html'],
                codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_OK
    for forbidden in ('merge', 'commit', 'push', 'branch'):
        assert forbidden not in g.subcommands(), '%s ran in a dry run' % forbidden
    assert 'DRY RUN OK' in capsys.readouterr().out


def test_dry_run_stages_then_unstages_the_same_paths():
    """Staging is required (the trackedness gate reads the INDEX), so a dry run
    must put the index back exactly as it found it."""
    g = FakeGit(untracked=['sessions/viz/toy.html', 'sessions/viz/two.html'])
    pubp.main([], git=g, runner=ok_runner)
    adds = g.calls_matching('add --')
    resets = g.calls_matching('reset -q --')
    assert len(adds) == 1 and len(resets) == 1
    assert list(adds[0][2:]) == list(resets[0][3:])


def test_dry_run_with_nothing_untracked_does_not_touch_the_index():
    g = FakeGit(untracked=[])
    pubp.main([], git=g, runner=ok_runner)
    assert not g.calls_matching('add')
    assert not g.calls_matching('reset')


# ---------------------------------------------------------------------------
# publish path
# ---------------------------------------------------------------------------
def test_publish_pushes_branch_before_main():
    """Backup first: if the main push loses a race, the work is already safe."""
    g = FakeGit()
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_OK
    pushes = [' '.join(a) for a in g.calls_matching('push')]
    assert len(pushes) == 2
    assert 'refs/heads/build/capability-spiral' in pushes[0]
    assert 'refs/heads/main' in pushes[1]


def test_publish_never_forces():
    g = FakeGit()
    pubp.main(['--publish'], git=g, runner=ok_runner)
    joined = ' '.join(' '.join(a) for a in g.log)
    for danger in ('--force', '--force-with-lease', 'stash', 'checkout',
                   'restore', 'add -A', 'reset --hard'):
        assert danger not in joined, 'publish used %s' % danger


def test_publish_commits_allowlisted_assets_then_pushes():
    g = FakeGit(untracked=['sessions/viz/toy.html'],
                codes={'diff --cached --quiet': 0})
    # after staging, diff --cached must report "something staged" -> nonzero
    g.codes['diff --cached --quiet'] = 0
    calls = {'n': 0}

    real_code = g.code

    def code(*args):
        # clean index at preflight, dirty after we stage
        if ' '.join(args) == 'diff --cached --quiet':
            calls['n'] += 1
            return 0 if calls['n'] == 1 else 1
        return real_code(*args)

    g.code = code
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_OK
    assert g.calls_matching('commit -m')
    assert len(g.calls_matching('push')) == 2


def test_publish_with_no_assets_creates_no_commit():
    g = FakeGit(untracked=[])
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_OK
    assert not g.calls_matching('commit')
    assert len(g.calls_matching('push')) == 2


def test_publish_merges_when_diverged():
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_OK
    assert g.calls_matching('merge --no-edit')


def test_publish_skips_merge_when_already_ahead():
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 0,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})
    pubp.main(['--publish'], git=g, runner=ok_runner)
    assert not g.calls_matching('merge --no-edit')


def test_publish_fast_forwards_local_main():
    g = FakeGit()
    pubp.main(['--publish'], git=g, runner=ok_runner)
    assert g.calls_matching('branch -f main HEAD')


def test_publish_leaves_local_main_alone_when_not_an_ancestor(capsys):
    g = FakeGit(codes={'merge-base --is-ancestor refs/heads/main HEAD': 1})
    pubp.main(['--publish'], git=g, runner=ok_runner)
    assert not g.calls_matching('branch -f')
    assert 'left alone' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------
def test_gate_failure_returns_2_unstages_and_does_not_commit(capsys):
    g = FakeGit(untracked=['sessions/viz/toy.html'])
    assert pubp.main(['--publish'], git=g, runner=bad_runner) == pubp.EX_GATE
    assert not g.calls_matching('commit')
    assert not g.calls_matching('push')
    assert g.calls_matching('reset -q --')          # index restored
    assert 'gate(s) failed' in capsys.readouterr().out


def test_skip_gates_bypasses_them_and_warns(capsys):
    g = FakeGit()
    assert pubp.main(['--publish', '--skip-gates'], git=g,
                     runner=bad_runner) == pubp.EX_OK
    out = capsys.readouterr().out
    assert 'nothing is checking the site' in out
    assert len(g.calls_matching('push')) == 2


def test_merge_conflict_aborts_and_returns_3(capsys):
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1},
                fail={'merge --no-edit': 'CONFLICT (content): Merge conflict'})
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_RECONCILE
    out = capsys.readouterr().out
    assert 'conflicted' in out
    assert 'git reset --hard' in out               # printed, never executed
    assert not g.calls_matching('push')


def test_untracked_would_be_overwritten_does_not_abort_the_merge(capsys):
    """No merge started, so `merge --abort` would itself fail. Just report.
    Message text is git's real wording for this case."""
    g = FakeGit(codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1},
                fail={'merge --no-edit':
                      'error: The following untracked working tree files would be '
                      'overwritten by merge:\n\tsessions/viz/toy.html\n'
                      'Please move or remove them before you merge.'})
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_RECONCILE
    out = capsys.readouterr().out
    assert 'No merge was started' in out
    assert 'sessions/viz/toy.html' in out
    assert not g.calls_matching('merge --abort')
    assert not g.calls_matching('push')


def test_fetch_failure_returns_3(capsys):
    g = FakeGit(fail={'fetch origin': 'could not read Username'})
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_RECONCILE
    assert 'fetch failed' in capsys.readouterr().out


def test_push_rejection_returns_4(capsys):
    g = FakeGit(fail={'push origin HEAD:refs/heads/main':
                      '! [rejected] main -> main (non-fast-forward)'})
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_PUSH
    assert 'push rejected' in capsys.readouterr().out


def test_head_moved_before_push_returns_5(capsys):
    """A concurrent agent session committing mid-run must not be published blind."""
    g = FakeGit(heads=[SHA_A, SHA_B])
    assert pubp.main(['--publish'], git=g, runner=ok_runner) == pubp.EX_CONCURRENT
    assert not g.calls_matching('push')
    assert 'HEAD moved' in capsys.readouterr().out


def test_unclassified_untracked_blocks_with_1(capsys):
    g = FakeGit(untracked=['sessions/m24-review-cmp'])
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_PREFLIGHT
    out = capsys.readouterr().out
    assert 'sessions/m24-review-cmp' in out
    assert 'COMMIT_GLOBS' in out and '.gitignore' in out


def test_detached_head_blocks_with_1(capsys):
    g = FakeGit(branch='HEAD')
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_PREFLIGHT
    assert 'detached' in capsys.readouterr().out


def test_dirty_tracked_files_block_with_1(capsys):
    g = FakeGit(codes={'diff --quiet': 1})
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_PREFLIGHT
    out = capsys.readouterr().out
    assert 'tracked files are modified' in out
    assert '--commit-all' in out


def test_prestaged_index_blocks_with_1(capsys):
    g = FakeGit(codes={'diff --cached --quiet': 1})
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_PREFLIGHT
    assert 'index already has staged changes' in capsys.readouterr().out


def test_commit_all_uses_add_u_not_add_A():
    """add -u touches tracked modifications only. add -A would sweep in a
    concurrent session's untracked work and defeat the allowlist."""
    g = FakeGit(codes={'diff --quiet': 1},
                captures={'diff --name-only -z': 'sessions/x.html'})
    pubp.main(['--publish', '--commit-all', 'my message'], git=g, runner=ok_runner)
    assert g.calls_matching('add -u --')
    assert not [a for a in g.log if ' '.join(a).startswith('add -A')]


def test_commit_all_dry_run_unstages_the_tracked_files_it_staged():
    """A dry run must put the index back, including --commit-all's `add -u`."""
    g = FakeGit(codes={'diff --quiet': 1},
                captures={'diff --name-only -z': 'sessions/x.html\0sessions/y.html'})
    pubp.main(['--commit-all', 'msg'], git=g, runner=ok_runner)
    resets = g.calls_matching('reset -q --')
    assert resets, 'dry run left tracked modifications staged'
    assert set(resets[0][3:]) == {'sessions/x.html', 'sessions/y.html'}


def test_commit_all_with_nothing_modified_does_not_stage():
    g = FakeGit(codes={'diff --quiet': 1}, captures={'diff --name-only -z': ''})
    pubp.main(['--commit-all', 'msg'], git=g, runner=ok_runner)
    assert not g.calls_matching('add -u')


def test_commit_all_requires_a_message():
    with pytest.raises(SystemExit):
        pubp.main(['--commit-all'], git=FakeGit(), runner=ok_runner)


def test_gates_only_does_not_fetch_or_push(capsys):
    g = FakeGit()
    assert pubp.main(['--gates-only'], git=g, runner=ok_runner) == pubp.EX_OK
    assert not g.calls_matching('fetch')
    assert not g.calls_matching('push')
    assert 'GATES ONLY' in capsys.readouterr().out


def test_missing_remote_ref_is_treated_as_first_push(capsys):
    g = FakeGit(has_remote_ref=False)
    assert pubp.main([], git=g, runner=ok_runner) == pubp.EX_OK
    assert 'the first push will create it' in capsys.readouterr().out


def test_report_prints_cache_busted_url(capsys):
    g = FakeGit()
    pubp.main(['--publish'], git=g, runner=ok_runner)
    out = capsys.readouterr().out
    assert 'https://owner.github.io/repo/sessions/index.html' in out
    assert '?v=' in out
    assert 'Cmd+Shift+R' in out


def test_divergence_report_names_paths_a_merge_would_restore(capsys):
    g = FakeGit(added='A\t.claude/skills/frontier-old/SKILL.md',
                codes={'merge-base --is-ancestor refs/remotes/origin/main HEAD': 1,
                       'merge-base --is-ancestor HEAD refs/remotes/origin/main': 1})
    pubp.main([], git=g, runner=ok_runner)
    out = capsys.readouterr().out
    assert 'a merge would restore these): 1' in out
    assert 'frontier-old/SKILL.md' in out


# ---------------------------------------------------------------------------
# integration — a real throwaway repo, no remote, no network
# ---------------------------------------------------------------------------
needs_git = pytest.mark.skipif(shutil.which('git') is None, reason='git required')


def _init_repo(tmp_path):
    root = str(tmp_path)
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    os.makedirs(os.path.join(root, 'sessions'), exist_ok=True)
    open(os.path.join(root, '.nojekyll'), 'w').close()
    open(os.path.join(root, 'index.html'), 'w').write('<html></html>')
    open(os.path.join(root, 'sessions', 'index.html'), 'w').write('<html></html>')
    subprocess.run(['git', 'add', '-A'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.email=t@t', '-c', 'user.name=t',
                    'commit', '-qm', 'init'], cwd=root, check=True)
    return root


@needs_git
def test_stage_then_reset_restores_exact_index_and_worktree(tmp_path):
    root = _init_repo(tmp_path)
    os.makedirs(os.path.join(root, 'sessions', 'viz'), exist_ok=True)
    open(os.path.join(root, 'sessions', 'viz', 'toy.html'), 'w').write('<html>t</html>')
    git = pubp.Git(root, enabled=False)

    def snapshot():
        return (pubp.tracked_files(git), pubp.untracked_files(git))

    before = snapshot()
    git.index_op('add', '--', 'sessions/viz/toy.html')
    assert 'sessions/viz/toy.html' in pubp.tracked_files(git)   # staged => in index
    git.index_op('reset', '-q', '--', 'sessions/viz/toy.html')
    assert snapshot() == before


@needs_git
def test_gitignored_asset_is_invisible_to_classification(tmp_path):
    """The repo ignores *.json globally. An ignored asset never appears in
    `ls-files --others --exclude-standard`, so classification cannot see it at
    all — which is exactly why the trackedness gate resolves against the INDEX
    rather than trusting this list."""
    root = _init_repo(tmp_path)
    open(os.path.join(root, '.gitignore'), 'w').write('*.json\n')
    open(os.path.join(root, 'sessions', 'steps.json'), 'w').write('{}')
    git = pubp.Git(root, enabled=False)
    assert 'sessions/steps.json' not in pubp.untracked_files(git)


@needs_git
def test_index_lock_blocks_with_exit_5(tmp_path, capsys):
    root = _init_repo(tmp_path)
    open(os.path.join(root, '.git', 'index.lock'), 'w').close()
    git = pubp.Git(root, enabled=False)
    try:
        assert pubp.main([], git=git, runner=ok_runner) == pubp.EX_CONCURRENT
        assert 'index.lock' in capsys.readouterr().out
    finally:
        os.remove(os.path.join(root, '.git', 'index.lock'))


@needs_git
def test_tracked_files_handles_spaces(tmp_path):
    root = _init_repo(tmp_path)
    open(os.path.join(root, 'sessions', 'a file.html'), 'w').write('<html></html>')
    subprocess.run(['git', 'add', '-A'], cwd=root, check=True)
    git = pubp.Git(root, enabled=False)
    assert 'sessions/a file.html' in pubp.tracked_files(git)


@needs_git
def test_verify_live_returns_false_without_network(tmp_path):
    root = _init_repo(tmp_path)
    ok = pubp.verify_live('http://127.0.0.1:1/nope',
                          os.path.join(root, 'sessions', 'index.html'),
                          seconds=0, sleep=lambda s: None)
    assert ok is False
