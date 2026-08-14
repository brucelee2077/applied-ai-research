#!/usr/bin/env python3
# =============================================================================
# publish_pages — put the current branch's work on the GitHub Pages site.
# =============================================================================
# GitHub Pages for this repo is a BRANCH DEPLOY: Settings -> Pages, "Deploy from
# a branch", branch `main`, folder `/` (root). There is no CI and no build step —
# whatever is committed on `main` IS the site. `.nojekyll` at the root means the
# tree is served raw, including `_`-prefixed paths.
#
# Authoring happens on a long-lived branch (build/capability-spiral), so the site
# goes stale until someone moves the work onto `main`. Doing that by hand is four
# git commands with three ways to quietly break the live site:
#
#   1. a link whose CASE is wrong resolves on macOS and 404s on Pages
#   2. an asset that exists on disk but was never `git add`-ed 404s on Pages
#   3. `git push origin HEAD:main` is rejected when main carries merge commits
#      the branch never absorbed
#
# This script does the whole dance, refuses to publish a site that fails a gate,
# and is safe to re-run.
#
#   python3 scripts/publish_pages.py            # DRY RUN: gates + report, no push
#   python3 scripts/publish_pages.py --publish  # merge, commit, push
#
# Exit codes are distinct on purpose — you should not have to read the log to
# tell "my content is broken" from "the network is broken":
#   0 ok / nothing to do   1 preflight   2 gate failed
#   3 reconcile (fetch or merge)   4 push rejected   5 concurrent git activity
#
# SAFETY: never force-pushes, never rewrites history, never runs `git stash`,
# `git checkout`, `git restore`, `git add -A`, or `git commit -a`. Another agent
# session may be working in this repo at the same time; every mutation names its
# paths explicitly, and HEAD is re-checked before each commit and push. Untracked
# assets modified in the last 120s are held back (--quiet-window) because another
# session is probably still writing them; --hold GLOB holds specific paths.
# =============================================================================
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# is_root_absolute + scan_links are extracted primitives, shared so the two
# publish gates cannot drift apart. Do NOT use validate_self_contained here:
# it also rejects "../", and sessions/ has ~779 legitimate ../day-NN links.
from publish_portfolio import is_root_absolute, scan_links   # noqa: E402

PAGES_BRANCH = 'main'
SITE_ENTRY = 'sessions/index.html'
# Files whose absence silently breaks the whole site. `.nojekyll` is the big one:
# without it GitHub runs Jekyll, which drops every `_`-prefixed path from the
# built site. (The inverse is why _coldgen/ and _compare/ must never be committed:
# with .nojekyll present, underscore directories ARE publicly served.)
REQUIRED_IN_INDEX = ('.nojekyll', 'index.html', SITE_ENTRY)

# Untracked paths we are willing to commit for you, by pattern. Deny wins.
SKIP_GLOBS = (
    'sessions/_coldgen/**',        # cold-generation A/B trials
    'sessions/_compare/**',        # gold-vs-engine A/B trials
    'sessions/**/_coverage.md',    # generated per compile, not source
    '**/__pycache__/**',
    '**/*.pyc',
    '**/.DS_Store',
)
COMMIT_GLOBS = (
    'sessions/viz/*.html',
    'sessions/*/day-*/lesson.html',
    'sessions/*/day-*.html',
    'sessions/*/review*.html',
    'sessions/*/day-*/experiment.py',
    'sessions/*/day-*/source.md',
    'sessions/*/day-*/expected_output.txt',
    'sessions/_compiler/shells/*.donor',
    'sessions/_compiler/shells/js/*.js',
)

EX_OK, EX_PREFLIGHT, EX_GATE, EX_RECONCILE, EX_PUSH, EX_CONCURRENT = 0, 1, 2, 3, 4, 5

Classification = namedtuple('Classification', 'commit skip unclassified')
Gate = namedtuple('Gate', 'name argv fn')
GateResult = namedtuple('GateResult', 'name ok detail')


class Abort(Exception):
    """Stop with a specific exit code and an explanation the operator can act on."""

    def __init__(self, code, *lines):
        super().__init__(lines[0] if lines else '')
        self.code = code
        self.lines = list(lines)


class GitError(RuntimeError):
    def __init__(self, args, returncode, stderr):
        super().__init__('git %s -> %d: %s' % (' '.join(args), returncode, stderr))
        self.argv = list(args)
        self.returncode = returncode
        self.stderr = stderr or ''


# ---------------------------------------------------------------------------
# pure logic — no subprocess, no filesystem, no network
# ---------------------------------------------------------------------------
_GLOB_CACHE = {}


def _glob_re(pattern):
    """Compile a glob that RESPECTS '/' boundaries.

    fnmatch.fnmatch is wrong for path allowlists: its '*' happily crosses '/', so
    'sessions/*/day-*.html' would also match 'sessions/a/b/c/day-1.html' and pull
    an unrelated deep file into an auto-commit."""
    if pattern in _GLOB_CACHE:
        return _GLOB_CACHE[pattern]
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith('**/', i):
            out.append('(?:.*/)?')
            i += 3
        elif pattern.startswith('**', i):
            out.append('.*')
            i += 2
        elif pattern[i] == '*':
            out.append('[^/]*')
            i += 1
        elif pattern[i] == '?':
            out.append('[^/]')
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    rx = re.compile(r'\A' + ''.join(out) + r'\Z')
    _GLOB_CACHE[pattern] = rx
    return rx


def glob_match(path, pattern):
    return _glob_re(pattern).match(path) is not None


def classify_untracked(paths):
    """Split untracked paths into (commit, skip, unclassified).

    FAILS CLOSED: a path matching neither list lands in `unclassified` and blocks
    the publish. Silently ignoring unknown files is how a lesson asset gets left
    behind and 404s in production; silently committing them is how scratch work
    gets published. Neither is acceptable, so the operator decides once and
    encodes the answer in a glob or in .gitignore."""
    commit, skip, unclassified = [], [], []
    for p in paths:
        if any(glob_match(p, g) for g in SKIP_GLOBS):
            skip.append(p)
        elif any(glob_match(p, g) for g in COMMIT_GLOBS):
            commit.append(p)
        else:
            unclassified.append(p)
    return Classification(sorted(commit), sorted(skip), sorted(unclassified))


def build_commit_message(paths, gates_skipped=False):
    """Deterministic commit message naming every auto-committed path."""
    paths = sorted(paths)
    n = len(paths)
    lines = ['chore(publish): commit %d untracked site asset%s for Pages'
             % (n, '' if n == 1 else 's'), '']
    lines += ['  ' + p for p in paths]
    lines += ['',
              'Auto-committed by scripts/publish_pages.py: every path matched the',
              'site-asset allowlist and the tree passed the publish gates.']
    if gates_skipped:
        lines += ['', 'Publish-Gates: SKIPPED']
    return '\n'.join(lines) + '\n'


def reconcile_action(remote_in_head, head_in_remote):
    """What has to happen before `git push HEAD:main` can fast-forward."""
    if remote_in_head and head_in_remote:
        return 'identical'
    if remote_in_head:
        return 'ahead'          # push is already a fast-forward
    if head_in_remote:
        return 'behind'         # our branch is contained in main; merge is a ff
    return 'merge'              # diverged: a real merge commit is needed


def pages_url(remote_url, entry=SITE_ENTRY):
    """Map an origin URL to its GitHub Pages URL. None if unrecognised."""
    m = re.match(r'(?:https?://[^/]+/|git@[^:]+:|ssh://[^/]+/)'
                 r'([^/]+)/(.+?)(?:\.git)?/?\Z', (remote_url or '').strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    host = '%s.github.io' % owner.lower()
    base = 'https://' + host
    if repo.lower() != host:          # user/org site lives at the domain root
        base += '/' + repo
    return '%s/%s' % (base, entry)


def missing_required(tracked):
    """Which REQUIRED_IN_INDEX paths are absent from the git index."""
    return [p for p in REQUIRED_IN_INDEX if p not in set(tracked)]


def absolute_link_violations(files, root, read=None):
    """Root-absolute src=/href= across the published HTML.

    "/sessions/x.html" resolves to <user>.github.io/sessions/x.html and drops the
    /<repo>/ segment, so it is dead on a PROJECT page. "../" is fine here — the
    whole prev/next chain is built from it — which is exactly why this gate uses
    is_root_absolute and not _escapes_portfolio."""
    return scan_links(files, is_root_absolute, root, read=read)


def apply_holds(commit, hold_globs):
    """Pull explicitly held paths out of the commit set. Returns (keep, held)."""
    if not hold_globs:
        return list(commit), []
    held = [p for p in commit if any(glob_match(p, g) for g in hold_globs)]
    hs = set(held)
    return [p for p in commit if p not in hs], held


def partition_warm(paths, window, mtime, now):
    """Split paths into (cold, warm) by modification age.

    A file another agent session wrote seconds ago is very likely still being
    written; committing it captures a half-authored lesson. This repo genuinely
    has concurrent sessions — one added two commits and a 154-line source.md
    while this script was being built — so holding warm files back is the safe
    default rather than an exotic option. window <= 0 disables it."""
    if window <= 0:
        return list(paths), []
    cold, warm = [], []
    for p in paths:
        try:
            age = now - mtime(p)
        except OSError:
            cold.append(p)
            continue
        (warm if age < window else cold).append(p)
    return cold, warm


def gate_specs(root, skip_names=(), checks=None):
    """The ordered blocking gates. Cheapest and most specific first, so a
    trackedness bug surfaces in seconds rather than after the test suites.

    DELIBERATELY ABSENT (test_gate_specs_exclude_destructive_scripts pins this):
      _density_scan.py, _body_baseline.py  — argv[1] is an OUTPUT path; both have
                                             destroyed source files
      lesson_audit.py (no args)            — writes sessions/_recover_set.json
      _experiment_check.py --all           — exits 1 by design while stub days
                                             remain, and executes 115 scripts
      staff_lens_audit.js                  — exits 2 on 100% of lessons; it looks
                                             for the obsolete `.sec` class
      coverage_audit.py                    — globs the retired week-* layout
    A publish gate has to be green when the site is fine. These are not."""
    py = sys.executable or 'python3'
    checks = checks or {}
    specs = [
        Gate('nav-audit-published',
             [py, 'sessions/nav_audit.py', '--published-only'], None),
        Gate('absolute-links', None, checks.get('absolute-links')),
        Gate('required-files', None, checks.get('required-files')),
        # explicit path: pytest.ini sets testpaths=interview_app/tests, so bare
        # pytest would cheerfully pass while running the wrong suite.
        Gate('compiler-tests',
             [py, '-m', 'pytest', 'sessions/_compiler/tests', '-q',
              '-p', 'no:cacheprovider'], None),
        Gate('node-reveal', ['node', 'sessions/_compiler/tests/test_reveal.mjs'], None),
        Gate('node-sr', ['node', 'sessions/_compiler/tests/test_sr.mjs'], None),
        Gate('self-tests',
             [py, '-m', 'pytest', 'scripts', '-q', '-p', 'no:cacheprovider'], None),
    ]
    return [g for g in specs if g.name not in set(skip_names)]


def run_gates(gates, runner, fail_fast=False):
    results = []
    for g in gates:
        ok, detail = runner(g)
        results.append(GateResult(g.name, ok, detail))
        if fail_fast and not ok:
            break
    return results


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
class Git:
    def __init__(self, root, enabled=False, echo=False, timeout=120):
        self.root = root
        self.enabled = enabled      # False => mutate() is a logged no-op
        self.echo = echo
        self.timeout = timeout

    def _run(self, args, timeout=None):
        if self.echo:
            print('    $ git ' + ' '.join(args))
        env = dict(os.environ)
        env['GIT_TERMINAL_PROMPT'] = '0'    # fail fast instead of hanging on auth
        return subprocess.run(['git'] + list(args), cwd=self.root, env=env,
                              capture_output=True, text=True,
                              timeout=timeout or self.timeout)

    def capture(self, *args):
        r = self._run(list(args))
        if r.returncode != 0:
            raise GitError(args, r.returncode, r.stderr)
        return r.stdout.strip()

    def code(self, *args):
        return self._run(list(args)).returncode

    def index_op(self, *args):
        """An index-only mutation (add/reset) that runs even in dry-run mode.

        Staging is not optional: the trackedness gate asks the INDEX what exists,
        so the assets have to be staged for it to see them. Always undone."""
        r = self._run(list(args))
        if r.returncode != 0:
            raise GitError(args, r.returncode, r.stderr)
        return r.stdout.strip()

    def mutate(self, *args):
        """A real mutation. No-op (logged) unless --publish was passed."""
        if not self.enabled:
            print('    [dry-run] git ' + ' '.join(args))
            return ''
        r = self._run(list(args))
        if r.returncode != 0:
            raise GitError(args, r.returncode, r.stderr)
        return r.stdout.strip()


def make_runner(root, verbose=False):
    def runner(gate):
        if gate.fn is not None:
            return gate.fn()
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'   # do not litter the tree mid-publish
        try:
            r = subprocess.run(gate.argv, cwd=root, env=env,
                               capture_output=True, text=True, timeout=900)
        except FileNotFoundError:
            return False, '%s not found on PATH' % gate.argv[0]
        except subprocess.TimeoutExpired:
            return False, 'timed out'
        if r.returncode == 0:
            return True, ''
        tail = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
        return False, 'exit %d\n%s' % (r.returncode,
                                       '\n'.join('      ' + l for l in tail[-12:]))
    return runner


def _z(out):
    return [p for p in out.split('\0') if p]


def tracked_files(git, *pathspec):
    """git ls-files -z. -z because core.quotepath quotes names with spaces and
    181 tracked paths in this repo contain one."""
    return _z(git.capture('ls-files', '-z', *pathspec))


def untracked_files(git):
    return _z(git.capture('ls-files', '--others', '--exclude-standard', '-z'))


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def check_no_index_lock(git):
    if os.path.exists(os.path.join(git.root, '.git', 'index.lock')):
        raise Abort(EX_CONCURRENT,
                    'another git process holds .git/index.lock.',
                    'A concurrent session is mid-commit. Wait for it and re-run.')


def preflight(git, commit_all=None):
    check_no_index_lock(git)
    branch = git.capture('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'HEAD':
        raise Abort(EX_PREFLIGHT,
                    'HEAD is detached; there is no branch to publish.',
                    'git switch <branch> and re-run.')
    if git.code('diff', '--cached', '--quiet') != 0:
        raise Abort(EX_PREFLIGHT,
                    'the index already has staged changes.',
                    'This script stages and unstages by exact path, so it will not',
                    'touch a pre-staged file. Commit or unstage it, then re-run.')
    if git.code('diff', '--quiet') != 0 and commit_all is None:
        stat = git.capture('diff', '--stat')
        raise Abort(EX_PREFLIGHT,
                    'tracked files are modified:',
                    stat,
                    '',
                    'This is a soundness requirement, not tidiness: the trackedness',
                    'gate reads page CONTENT from the worktree but resolves link',
                    'TARGETS from the index, so a dirty tree makes it validate a',
                    'state that will never be served.',
                    'Commit them yourself, or pass --commit-all "your message".')
    return branch, git.capture('rev-parse', 'HEAD')


def assert_unmoved(git, expected_sha, what):
    now = git.capture('rev-parse', 'HEAD')
    if now != expected_sha:
        raise Abort(EX_CONCURRENT,
                    'HEAD moved while we were %s (%s -> %s).' % (what, expected_sha[:7], now[:7]),
                    'Another session committed. Nothing was pushed; re-run.')
    return now


def report_divergence(git, branch):
    """Always printed. In a dry run this IS the deliverable: it tells you what a
    merge would absorb before anything mutates."""
    remote_ref = 'refs/remotes/origin/%s' % PAGES_BRANCH
    if git.code('rev-parse', '--verify', '--quiet', remote_ref) != 0:
        print('  no %s yet — the first push will create it' % remote_ref)
        return 'ahead'
    remote_in_head = git.code('merge-base', '--is-ancestor', remote_ref, 'HEAD') == 0
    head_in_remote = git.code('merge-base', '--is-ancestor', 'HEAD', remote_ref) == 0
    action = reconcile_action(remote_in_head, head_in_remote)

    incoming = git.capture('log', '--oneline', '--no-decorate', 'HEAD..' + remote_ref)
    outgoing = git.capture('rev-list', '--count', remote_ref + '..HEAD')
    print('  branch   %s @ %s' % (branch, git.capture('rev-parse', '--short', 'HEAD')))
    print('  target   origin/%s @ %s' % (PAGES_BRANCH,
                                         git.capture('rev-parse', '--short', remote_ref)))
    print('  delta    %s commit(s) to publish, %d to absorb  [%s]'
          % (outgoing, len(incoming.splitlines()) if incoming else 0, action))
    if incoming:
        print('  incoming:')
        for line in incoming.splitlines()[:20]:
            print('    ' + line)
    # Files main has that we do not: these are what a merge could RESURRECT.
    # Re-derived every run instead of trusting a previous analysis.
    added = [l for l in git.capture('diff', '--name-status', 'HEAD', remote_ref).splitlines()
             if l.startswith('A')]
    print('  paths on origin/%s absent from HEAD (a merge would restore these): %d'
          % (PAGES_BRANCH, len(added)))
    for line in added[:20]:
        print('    ' + line)
    if len(added) > 20:
        print('    ... and %d more' % (len(added) - 20))
    return action


def reconcile(git, action):
    if action in ('identical', 'ahead'):
        print('  nothing to merge (push will fast-forward)')
        return None
    remote_ref = 'refs/remotes/origin/%s' % PAGES_BRANCH
    pre = git.capture('rev-parse', 'HEAD')
    try:
        git.mutate('merge', '--no-edit', remote_ref)
    except GitError as e:
        blob = (e.stderr or '') + ' '
        # Two different failures share the phrase "would be overwritten by merge".
        # They have different causes and different fixes, so do not conflate them.
        if 'untracked working tree file' in blob.lower():
            raise Abort(EX_RECONCILE,
                        'merge blocked: it would overwrite untracked files.',
                        e.stderr.strip(),
                        '',
                        'No merge was started, so nothing needs undoing.',
                        'origin/%s carries a file you also have untracked locally.'
                        % PAGES_BRANCH,
                        'Commit yours or move it aside, then re-run.')
        if 'local changes' in blob.lower():
            raise Abort(EX_RECONCILE,
                        'merge blocked: the index or worktree is not clean.',
                        e.stderr.strip(),
                        '',
                        'No merge was started, so nothing needs undoing.',
                        'git merge requires a clean tree. This script stages only',
                        'AFTER merging, so if you are seeing this, something else',
                        'dirtied the tree mid-run — most likely a concurrent',
                        'session. Re-run once it settles.')
        conflicted = ''
        try:
            conflicted = git.capture('diff', '--name-only', '--diff-filter=U')
        except GitError:
            pass
        git.code('merge', '--abort')      # code(): must not raise on top of a raise
        raise Abort(EX_RECONCILE,
                    'merge conflicted; aborted, tree restored.',
                    'conflicted paths:', conflicted or '(none reported)',
                    '',
                    'Resolve by hand:  git merge %s' % remote_ref,
                    'then re-run with --publish.',
                    'If anything looks wrong: git reset --hard %s' % pre)
    return pre


def push(git, branch):
    # Backup FIRST: if the main push races and is rejected, the work is already
    # safe on its own branch. Never --force / --force-with-lease — a rejection
    # means someone else pushed, which is exactly the signal we want.
    try:
        if branch != PAGES_BRANCH:
            print('  pushing %s (backup)' % branch)
            git.mutate('push', 'origin', 'HEAD:refs/heads/%s' % branch)
        print('  pushing %s (the publish)' % PAGES_BRANCH)
        git.mutate('push', 'origin', 'HEAD:refs/heads/%s' % PAGES_BRANCH)
    except GitError as e:
        raise Abort(EX_PUSH,
                    'push rejected or failed:',
                    (e.stderr or '').strip(),
                    '',
                    'If this is a non-fast-forward, origin/%s moved after our fetch.'
                    % PAGES_BRANCH,
                    'Just re-run: the next run fetches and merges the new commits.')


def sync_local_main(git, branch):
    if branch == PAGES_BRANCH:
        return
    if git.code('show-ref', '--verify', '--quiet', 'refs/heads/%s' % PAGES_BRANCH) != 0:
        return
    if git.code('merge-base', '--is-ancestor', 'refs/heads/%s' % PAGES_BRANCH, 'HEAD') == 0:
        git.mutate('branch', '-f', PAGES_BRANCH, 'HEAD')
    else:
        print('  note: local %s is not an ancestor of HEAD; left alone' % PAGES_BRANCH)


def verify_live(url, local_path, seconds, sleep=time.sleep):
    """Poll the live URL until the served bytes match local. Best-effort: a
    failure here is a warning, never a non-zero exit — the push already
    succeeded. (brucelee2077.github.io is blocked from the Claude Code sandbox,
    so this only works when a human runs it.)"""
    import urllib.request
    try:
        want = hashlib.sha256(open(local_path, 'rb').read()).hexdigest()
    except OSError as e:
        print('  [verify] cannot read %s: %s' % (local_path, e))
        return False
    deadline = time.time() + seconds
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers={'Accept-Encoding': 'identity',
                                                       'Cache-Control': 'no-cache'})
            body = urllib.request.urlopen(req, timeout=20).read()
            got = hashlib.sha256(body).hexdigest()
            if got == want:
                print('  [verify] live bytes match local after %d attempt(s)' % attempt)
                return True
            print('  [verify] attempt %d: served %d bytes, sha %s != %s'
                  % (attempt, len(body), got[:12], want[:12]))
        except Exception as e:                      # noqa: BLE001 - best effort
            print('  [verify] attempt %d: %s' % (attempt, e))
        sleep(10)
    print('  [verify] gave up after %ds. The push succeeded; Pages may still be'
          ' building, or your network blocks the site.' % seconds)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_lines(lines):
    for l in lines:
        if l:
            for sub in str(l).splitlines():
                print('  ' + sub)
        else:
            print()


def main(argv=None, git=None, runner=None):
    ap = argparse.ArgumentParser(
        description='Publish the current branch to the GitHub Pages branch (%s).'
                    % PAGES_BRANCH)
    ap.add_argument('--publish', action='store_true',
                    help='actually merge, commit and push (default is a dry run)')
    ap.add_argument('--gates-only', action='store_true',
                    help='run the gates and stop; no fetch, no push')
    ap.add_argument('--commit-all', metavar='MSG', default=None,
                    help='also commit modified/deleted TRACKED files with MSG')
    ap.add_argument('--skip-gates', action='store_true',
                    help='skip the blocking gates (recorded in the commit trailer)')
    ap.add_argument('--message', metavar='MSG', default=None,
                    help='override the generated auto-commit message')
    ap.add_argument('--verify', nargs='?', type=int, const=180, default=None,
                    metavar='SECONDS', help='after pushing, poll the live URL')
    ap.add_argument('--fail-fast', action='store_true',
                    help='stop at the first failing gate')
    ap.add_argument('--hold', action='append', default=[], metavar='GLOB',
                    help='keep matching untracked paths out of the auto-commit '
                         'this run (repeatable); use when another session owns them')
    ap.add_argument('--quiet-window', type=int, default=120, metavar='SECONDS',
                    help='hold back untracked assets modified within this many '
                         'seconds, on the assumption another session is still '
                         'writing them (default 120; 0 disables)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='echo every git command')
    a = ap.parse_args(argv)

    if git is None:
        git = Git(ROOT, enabled=a.publish, echo=a.verbose)
    staged = []
    try:
        # ---- 1 preflight -------------------------------------------------
        branch, expected = preflight(git, commit_all=a.commit_all)
        mode = 'PUBLISH' if a.publish else 'DRY RUN'
        print('=' * 68)
        print('publish_pages — %s' % mode)
        print('=' * 68)
        if a.skip_gates:
            print('!' * 68)
            print('!! --skip-gates: nothing is checking the site before it goes live')
            print('!' * 68)

        # ---- 2 classify --------------------------------------------------
        cls = classify_untracked(untracked_files(git))
        if cls.unclassified:
            raise Abort(EX_PREFLIGHT,
                        'these untracked paths match neither the site-asset',
                        'allowlist nor the skip list, so I will not guess:',
                        *['    ' + p for p in cls.unclassified],
                        '',
                        'Pick one per path: add a COMMIT_GLOBS entry (it is a site',
                        'asset), add a .gitignore rule (it is scratch), or delete it.')
        print('\n[1] untracked: %d to commit, %d skipped'
              % (len(cls.commit), len(cls.skip)))
        to_commit, held = apply_holds(cls.commit, a.hold)
        to_commit, warm = partition_warm(
            to_commit, a.quiet_window,
            mtime=lambda p: os.path.getmtime(os.path.join(git.root, p)),
            now=time.time())
        for p in to_commit:
            print('    + ' + p)
        for p in cls.skip:
            print('    - %s (skipped)' % p)
        for p in held:
            print('    . %s (held by --hold)' % p)
        for p in warm:
            print('    . %s (modified <%ds ago — another session may still be '
                  'writing it; held)' % (p, a.quiet_window))
        if warm:
            print('      pass --quiet-window 0 to commit warm files anyway')

        # ---- 3 commit_all -------------------------------------------------
        # This has to precede the merge: `git merge` refuses to run with a dirty
        # worktree, so work-in-progress must be committed first.
        if a.commit_all is not None:
            modified = _z(git.capture('diff', '--name-only', '-z'))
            if modified:
                git.index_op('add', '-u', '--')     # tracked mods/deletes only
                if a.publish:
                    check_no_index_lock(git)
                    print('\n[1b] committing %d modified tracked file(s)' % len(modified))
                    git.mutate('commit', '-m', a.commit_all)
                    expected = git.capture('rev-parse', 'HEAD')
                else:
                    # dry run: keep the index dirty only long enough to gate it,
                    # then the finally-block restores it by exact pathspec.
                    staged += modified

        # ---- 4/5 fetch + divergence --------------------------------------
        action = 'ahead'
        if not a.gates_only:
            print('\n[2] fetching origin/%s' % PAGES_BRANCH)
            try:
                git.index_op('fetch', 'origin',
                             '+refs/heads/%s:refs/remotes/origin/%s'
                             % (PAGES_BRANCH, PAGES_BRANCH))
            except (GitError, subprocess.TimeoutExpired) as e:
                raise Abort(EX_RECONCILE, 'fetch failed: %s' % e,
                            'Check the network and re-run.')
            print('\n[3] divergence')
            action = report_divergence(git, branch)

        # ---- 6 reconcile --------------------------------------------------
        if a.publish and not a.gates_only:
            print('\n[4] reconcile')
            assert_unmoved(git, expected, 'fetching')
            # Refresh `expected` ONLY if we actually created a merge commit. If no
            # merge happened, HEAD must still be where preflight found it —
            # refreshing unconditionally would silently absorb a concurrent
            # session's commit into this publish.
            if reconcile(git, action) is not None:
                expected = git.capture('rev-parse', 'HEAD')
        elif not a.gates_only and action == 'merge':
            print('\n[4] reconcile — [dry-run] would merge origin/%s into %s'
                  % (PAGES_BRANCH, branch))

        # ---- 7 stage, AFTER the merge -------------------------------------
        # Ordering is load-bearing, and git enforces it: `git merge` requires a
        # clean index, so staging first makes the merge fail with "Your local
        # changes to the following files would be overwritten by merge". Staging
        # here also means the gates below see the exact tree that gets pushed
        # (merge result + new assets) rather than a pre-merge approximation.
        if to_commit:
            check_no_index_lock(git)
            git.index_op('add', '--', *to_commit)
            staged += list(to_commit)

        # ---- 7 gates ------------------------------------------------------
        if not a.skip_gates:
            print('\n[5] gates')
            checks = _inproc_checks(git)
            gates = gate_specs(git.root, checks=checks)
            use = runner if runner is not None else make_runner(git.root, a.verbose)
            results = run_gates(gates, use, fail_fast=a.fail_fast)
            for r in results:
                print('    %-22s %s' % (r.name, 'ok' if r.ok else 'FAIL'))
                if not r.ok and r.detail:
                    _print_lines([r.detail])
            if any(not r.ok for r in results):
                raise Abort(EX_GATE,
                            '%d gate(s) failed; nothing was committed or pushed.'
                            % sum(1 for r in results if not r.ok))

        if a.gates_only:
            print('\nGATES ONLY — no fetch, no push.')
            return EX_OK

        # ---- 8 commit -----------------------------------------------------
        # --commit-all already committed above (it had to precede the merge), so
        # this only handles the allowlisted untracked assets.
        if a.publish:
            if to_commit:
                check_no_index_lock(git)
                assert_unmoved(git, expected, 'running the gates')
                msg = a.message or build_commit_message(to_commit, a.skip_gates)
                if git.code('diff', '--cached', '--quiet') != 0:
                    print('\n[6] committing %d site asset(s)' % len(to_commit))
                    git.mutate('commit', '-m', msg)
                    staged = []           # the commit consumed the staged paths
                    expected = git.capture('rev-parse', 'HEAD')
                else:
                    print('\n[6] nothing staged to commit')
            else:
                print('\n[6] no new site assets to commit')

            # ---- 9/10 push + local main ----------------------------------
            print('\n[7] push')
            assert_unmoved(git, expected, 'preparing to push')
            push(git, branch)
            sync_local_main(git, branch)
        else:
            print('\n[6] [dry-run] would commit %d asset(s) and push %s -> origin/%s'
                  % (len(to_commit), branch, PAGES_BRANCH))

        # ---- 11 report ----------------------------------------------------
        sha = git.capture('rev-parse', '--short', 'HEAD')
        url = None
        try:
            url = pages_url(git.capture('remote', 'get-url', 'origin'))
        except GitError:
            pass
        print('\n' + '=' * 68)
        if a.publish:
            print('PUBLISHED  %s -> origin/%s' % (sha, PAGES_BRANCH))
        else:
            print('DRY RUN OK — re-run with --publish to go live')
        if url:
            print('  %s' % url)
            print('  cache-busted (open this one):')
            print('  %s?v=%s' % (url, sha))
            print('')
            print('  Branch deploys usually go live in 30-120s. There is no build log.')
            print('  A tab you already have open will keep serving the old bytes:')
            print('  use the ?v= URL, or Cmd+Shift+R.')
        if a.publish and a.verify and url:
            verify_live(url, os.path.join(git.root, SITE_ENTRY), a.verify)
        return EX_OK

    except Abort as e:
        print('\n' + '=' * 68)
        print('ABORTED (exit %d)' % e.code)
        _print_lines(e.lines)
        return e.code
    except GitError as e:
        print('\ngit failed unexpectedly: %s' % e)
        return EX_PREFLIGHT
    finally:
        # Unstage exactly what we staged, never more. A gate failure or a dry run
        # must leave the index byte-identical to how we found it.
        if staged:
            try:
                git.index_op('reset', '-q', '--', *staged)
            except GitError as e:
                print('  WARNING: could not unstage %d path(s): %s' % (len(staged), e))


def _inproc_checks(git):
    """The two gates that are cheaper in-process than as a subprocess."""
    def absolute_links():
        files = [os.path.join(git.root, p) for p in tracked_files(git, '*.html')]
        viol = absolute_link_violations(files, git.root)
        if not viol:
            return True, ''
        return False, '\n'.join('      %s -> %s' % (f, l) for f, l in viol[:20])

    def required_files():
        gone = missing_required(tracked_files(git))
        if not gone:
            return True, ''
        return False, '      missing from the git index: %s' % ', '.join(gone)

    return {'absolute-links': absolute_links, 'required-files': required_files}


if __name__ == '__main__':
    sys.exit(main())
