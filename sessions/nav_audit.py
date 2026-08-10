#!/usr/bin/env python3
"""nav_audit.py — verify every Frontier-Lab session page is wired together.

The hub `index.html` MODULES array is the single source of truth for module
ORDER. Every built lesson/review page is expected to form ONE continuous chain
in that order: each page's "← Prev" points to the previous page, "Next →" to the
next, and the hub "Map" button to index.html. The first page keeps a disabled
"Start" placeholder; the last page's Next returns to the hub.

Checks, over all pages under sessions/:
  1. CHAIN     — actual prev/next/hub hrefs resolve to the correct neighbor
  2. BROKEN    — every href/src (attrs + JS ".html" literals) resolves to a file
  3. CASE      — resolves case-sensitively (macOS is lenient, GitHub Pages is not)
  4. ORPHANS   — built pages nothing links to

Usage:  python3 sessions/nav_audit.py            # from repo root
        python3 nav_audit.py                     # from sessions/
        python3 sessions/nav_audit.py --published-only    # PUBLISH gate
Exit code 0 iff CHAIN, BROKEN and CASE are all clean (orphans are informational).

--published-only — what GitHub Pages will actually serve
------------------------------------------------------------------
Default mode asks the FILESYSTEM "does this link target exist?". Pages serves the
git branch, not your disk, so a file you never `git add`-ed exists locally and
404s in production. `--published-only` therefore:

  * restricts the page set to files in the git index (329 -> 322 today: the
    _coldgen/ and _compare/ A/B scratch trees drop out, and their deliberately
    broken neighbour links stop being reported as real failures), and
  * resolves link targets against the git index instead of the filesystem.

Resolving against the index buys three checks with one predicate: broken links,
trackedness, AND case-exactness for free — git index names are byte-exact, so a
link to viz/foo.html against an index entry viz/Foo.html is a miss, which is
precisely what Pages does. The case_ok() filesystem walk is redundant here and is
skipped; default mode still runs it.

SOUNDNESS PRECONDITION: published mode reads page CONTENT from the worktree but
resolves TARGETS from the index. That is only sound when the worktree matches the
index for tracked files. scripts/publish_pages.py enforces a clean tree in
preflight; running this by hand on a dirty tree prints a warning.

Scope note: the link pattern is deliberately href/src attributes + JS ".html"
string literals. Do NOT widen it to .json/.csv/.png/.py/.md — those appear in
lesson prose and code blocks as filenames the LEARNER will create, and every one
of the 23 hits a widened scan produces is a false positive.
"""
import re, os, sys, glob, argparse, subprocess
from urllib.parse import urldefrag, unquote


def tracked_paths(repo_root):
    """Contents of the git index, POSIX-relative to repo_root.

    -z (not splitlines) because core.quotepath defaults to on and 181 tracked
    paths in this repo contain spaces — splitlines() on quoted output silently
    yields mangled names that then look 'untracked'."""
    r = subprocess.run(['git', 'ls-files', '-z'], cwd=repo_root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('nav_audit: git ls-files failed (%d): %s'
                         % (r.returncode, r.stderr.strip()))
    return {p for p in r.stdout.split('\0') if p}


def _is_gitignored(repo_root, abspath):
    """True if .gitignore excludes abspath. Used only to LABEL a broken link:
    an ignored asset is present on disk yet unpublishable, and 'BROKEN' with no
    explanation sends the reader hunting for a file that is sitting right there.
    The repo ignores *.json / *.csv / *.pdf globally, so this is a live trap."""
    return subprocess.run(['git', 'check-ignore', '--quiet', abspath],
                          cwd=repo_root, capture_output=True).returncode == 0


def _worktree_dirty(repo_root):
    """True if tracked files differ from the index (see SOUNDNESS PRECONDITION)."""
    return subprocess.run(['git', 'diff', '--quiet'], cwd=repo_root,
                          capture_output=True).returncode != 0


def audit(sess, published=False):
    """Run the audit over `sess` (a sessions/ directory). Returns 0 or 1."""
    repo_root = os.path.dirname(sess)
    INDEX = os.path.join(sess, "index.html")
    rel = lambda p: os.path.relpath(p, sess)
    # index membership is keyed from the REPO root, not from sessions/
    relroot = lambda p: os.path.relpath(p, repo_root).replace(os.sep, '/')

    tracked = tracked_paths(repo_root) if published else None
    # case-insensitive view of the index, so a published-mode miss can say
    # "you spelled it foo.html, the index says Foo.html" instead of "untracked".
    # macOS resolves both, GitHub Pages resolves only the index spelling.
    tracked_ci = {p.lower(): p for p in tracked} if published else None

    # ---------- build canonical order from index.html ----------
    txt = open(INDEX, encoding="utf-8").read()
    mod_re = re.compile(r"\{n:\s*('?)([^,']+)\1\s*,.*?lessons:\s*\[(.*?)\]\s*\}", re.S)
    les = re.compile(r"\[\s*'((?:[^'\\]|\\.)*)'\s*,\s*'((?:[^'\\]|\\.)*)'\s*,\s*(null|'([^']*)')")
    mod_dirs = []
    for mm in mod_re.finditer(txt):
        pages = [lm.group(4) for lm in les.finditer(mm.group(3)) if lm.group(3) != "null"]
        if pages:
            mod_dirs.append(pages[0].split("/")[0])
    def daynum(p):
        m = re.search(r"day-(\d+)", p); return int(m.group(1)) if m else 0
    canon = []
    for d in mod_dirs:
        ad = os.path.join(sess, d)
        days = sorted(glob.glob(ad + "/day-*/lesson.html"), key=daynum) or sorted(glob.glob(ad + "/day-*.html"), key=daynum)
        seq = list(days)
        rp = os.path.join(ad, "review-part-a.html")
        if os.path.exists(rp):
            pos = next((i for i, s in enumerate(seq) if daynum(s) == 4), len(seq)); seq.insert(pos, rp)
        rv = os.path.join(ad, "review.html")
        if os.path.exists(rv):
            seq.append(rv)
        canon += seq

    def rez(fp, h): return os.path.normpath(os.path.join(os.path.dirname(fp), h.split("#")[0].split("?")[0]))
    def prevs(fp): return set(re.findall(r'<a class="lnav prev"[^>]*href="([^"]+)"', open(fp, encoding="utf-8").read()))
    def nexts(fp): return set(re.findall(r'<a class="lnav next"[^>]*href="([^"]+)"', open(fp, encoding="utf-8").read()))
    def hubs(fp):  return set(re.findall(r'<a class="lnav-hub" href="([^"]+)"', open(fp, encoding="utf-8").read()))

    chain = []
    for i, fp in enumerate(canon):
        exp_prev = INDEX if i == 0 else canon[i - 1]
        exp_next = INDEX if i == len(canon) - 1 else canon[i + 1]
        ph = prevs(fp)
        if i == 0:
            if ph and all(rez(fp, h) != INDEX for h in ph):
                chain.append(f"PREV {rel(fp)} (first page should keep Start/hub)")
        else:
            if not ph: chain.append(f"NO-PREV {rel(fp)}")
            elif not all(rez(fp, h) == exp_prev for h in ph):
                chain.append(f"PREV {rel(fp)} -> {[rel(rez(fp,h)) for h in ph]} (exp {rel(exp_prev)})")
        nh = nexts(fp)
        if not nh: chain.append(f"NO-NEXT {rel(fp)}")
        elif not all(rez(fp, h) == exp_next for h in nh):
            chain.append(f"NEXT {rel(fp)} -> {[rel(rez(fp,h)) for h in nh]} (exp {rel(exp_next)})")
        if not all(rez(fp, h) == INDEX for h in hubs(fp)):
            chain.append(f"HUB {rel(fp)}")

    # ---------- broken + case + orphans over all html ----------
    attr_re = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.I)
    js_re = re.compile(r'["\']([^"\']*?\.html)["\']', re.I)
    ext = lambda l: re.match(r'^(https?:|mailto:|data:|javascript:|tel:|#|//)', l, re.I)
    def case_ok(abspath):
        if not os.path.exists(abspath): return True  # non-existence handled by BROKEN
        cur = os.sep
        for part in abspath.split(os.sep)[1:]:
            if not part: continue
            try: entries = os.listdir(cur)
            except OSError: return False
            if part not in entries: return False
            cur = os.path.join(cur, part)
        return True

    def target_exists(abspath):
        """Default: does it exist on disk? Published: is it in the git index?"""
        if not published:
            return os.path.exists(abspath)
        return relroot(abspath) in tracked

    def why_missing(abspath):
        """Suffix explaining a published-mode BROKEN entry. '' in default mode."""
        if not published:
            return ''
        want = relroot(abspath)
        actual = tracked_ci.get(want.lower())
        if actual and actual != want:
            return ' (CASE mismatch — index has %s)' % actual
        if not os.path.exists(abspath):
            return ' (not in index and not on disk)'
        if _is_gitignored(repo_root, abspath):
            return ' (ignored by .gitignore — present on disk, unpublishable)'
        return ' (untracked, exists on disk — git add it)'

    html_files = [p for p in glob.glob(sess + "/**/*.html", recursive=True)
                  if "__pycache__" not in p
                  and (not published or relroot(p) in tracked)]
    broken, case_bad, linked = [], [], set()
    for src in html_files:
        t = open(src, encoding="utf-8", errors="replace").read()
        # sorted(): set-union iteration order varies with PYTHONHASHSEED, which made
        # the BROKEN/CASE line order unstable run-to-run (verified on the original:
        # 2 of 6 seeds produced a different ordering). Sorting costs nothing and is
        # what lets a golden-output test be meaningful.
        for raw in sorted(set(attr_re.findall(t)) | set(js_re.findall(t))):
            link = raw.strip()
            if not link or ext(link): continue
            pp = unquote(urldefrag(link)[0].split("?")[0])
            if not pp: continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(src), pp))
            if not resolved.startswith(os.path.dirname(sess)): continue
            if not target_exists(resolved):
                broken.append(f"{rel(src)} -> {link}{why_missing(resolved)}")
            else:
                if resolved.endswith(".html"): linked.add(os.path.abspath(resolved))
                # published mode gets case-exactness from the index lookup itself
                if not published and not case_ok(resolved):
                    case_bad.append(f"{rel(src)} -> {link}")
    orphans = sorted(rel(p) for p in ({os.path.abspath(x) for x in html_files} - linked))

    # canon pages the index does not carry: BROKEN catches these transitively via
    # a neighbour's prev/next, but say it directly instead of making it inferred.
    unpublished = []
    if published:
        unpublished = sorted(rel(p) for p in canon if relroot(p) not in tracked)

    def section(title, items):
        print(f"\n### {title}: {len(items)}")
        for x in items[:60]: print("   ", x)

    print("=" * 64)
    print(f"nav_audit — {len(html_files)} pages, chain of {len(canon)}"
          + (" [published-only]" if published else ""))
    print("=" * 64)
    if published and _worktree_dirty(repo_root):
        print("WARNING: tracked files are modified. Published mode reads page content")
        print("         from the worktree but targets from the index, so this run")
        print("         validates a state that will never be served. Commit first.")
    section("CHAIN problems", chain)
    section("BROKEN links", broken)
    section("CASE mismatches (GitHub Pages)", case_bad)
    if published:
        section("UNPUBLISHED (in chain, not in git index)", unpublished)
    section("ORPHANS (informational)", orphans)
    ok = not (chain or broken or case_bad or unpublished)
    print("\n" + ("PASS — all pages wired together." if ok else "FAIL — see problems above."))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Verify every session page is wired together.')
    ap.add_argument('--published-only', action='store_true',
                    help='resolve links against the git index (what Pages serves) '
                         'and skip pages that are not tracked')
    ap.add_argument('--root', default=None,
                    help='repo root to audit (default: derived from this file); '
                         'sessions/ is expected directly beneath it')
    a = ap.parse_args(argv)

    if a.root:
        sess = os.path.join(os.path.abspath(a.root), 'sessions')
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        sess = here if os.path.basename(here) == 'sessions' else os.path.join(here, 'sessions')
    return audit(sess, published=a.published_only)


if __name__ == '__main__':
    sys.exit(main())
