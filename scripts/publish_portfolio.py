#!/usr/bin/env python3
# =============================================================================
# publish_portfolio (Plan 2)  — validate + publish the self-contained portfolio.
# =============================================================================
# Gate before publishing: the portfolio/ tree must be SELF-CONTAINED. Every
# src=/href= in every portfolio/**/index.html AND every copied viz
# (portfolio/**/assets/*.html) must stay inside portfolio/ — no root-absolute
# paths ("/x", dead under a Pages project subpath) and no "../" parent escapes.
# If it isn't self-contained, publishing anywhere else would produce dead links,
# so we refuse. External links (http/https/protocol-relative //), in-page
# anchors (#), mailto:, and data: URIs are fine on a public site.
#
# validate_self_contained(root) -> [(file, offending_link), ...]  ([] = clean)
#
# Reusable primitives (scripts/publish_pages.py gates the whole site on these):
#   is_root_absolute(link)                 -> the "/x" half, applies site-wide
#   scan_links(files, predicate, root)     -> the scanner, predicate-injectable
#
# CLI:
#   python3 scripts/publish_portfolio.py                 # dry-run, validate only
#   python3 scripts/publish_portfolio.py --to <dir> --no-dry-run   # copy portfolio/
# Always validates first; exits 1 (printing violations) if not self-contained.
# =============================================================================
import sys, os, re, glob, shutil, argparse

# repo root = one dir up from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# capture the value of any src=/href= — quoted (double or single) OR unquoted.
_LINK_RE = re.compile(
    r'''(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))''', re.IGNORECASE)

# link values that never escape the site (safe on a public Pages deploy)
_ALLOWED_PREFIXES = ('http://', 'https://', '//', '#', 'mailto:', 'data:', 'tel:')


def is_root_absolute(link):
    """True if a link is ROOT-ABSOLUTE ("/x" but NOT "//host" protocol-relative).
    Root-absolute links are dead under a Pages PROJECT subpath: "/sessions/x.html"
    resolves to <user>.github.io/sessions/x.html, dropping the /<repo>/ segment.

    This is the half of _escapes_portfolio() that applies to the WHOLE site, not
    just portfolio/. scripts/publish_pages.py gates on this predicate alone —
    "../" is perfectly legal outside portfolio/ (the sessions/ prev/next chain is
    built from ../day-NN/lesson.html), so do NOT gate the site on
    _escapes_portfolio or validate_self_contained."""
    l = link.strip()
    if not l:
        return False
    if l.lower().startswith(_ALLOWED_PREFIXES):
        return False
    return l.startswith('/')       # root-absolute (// already allowed above)


def _escapes_portfolio(link):
    """True if a link leaves the portfolio tree. Offenses:
      (a) root-absolute ("/x" but NOT "//host" protocol-relative) — dead under a
          Pages project subpath, and
      (b) any "../" parent escape.
    Allowed (never flagged): http(s)://, protocol-relative //, in-page #,
    mailto:, data:, tel:."""
    l = link.strip()
    if not l:
        return False
    if l.lower().startswith(_ALLOWED_PREFIXES):
        return False
    if is_root_absolute(l):        # (a)
        return True
    if '../' in l:                 # (b) parent escape anywhere in the value
        return True
    return False


def scan_links(files, predicate, root, read=None):
    """Scan `files` (absolute paths) for src=/href= values satisfying `predicate`.
    Returns [(path_relative_to_root, offending_link), ...] in sorted-file order.

    `read` is injectable for tests: read(path) -> html string. Unreadable files
    are skipped, matching the original behaviour (a binary blob named .html must
    not crash a publish gate)."""
    if read is None:
        def read(path):
            return open(path, encoding='utf-8').read()
    violations = []
    for path in sorted(files):
        try:
            html = read(path)
        except Exception:
            continue
        for m in _LINK_RE.finditer(html):
            link = m.group(1) or m.group(2) or m.group(3) or ''
            if predicate(link):
                violations.append((os.path.relpath(path, root), link))
    return violations


def _portfolio_files(root):
    """Every portfolio/**/index.html plus every portfolio/**/assets/*.html."""
    patterns = [os.path.join(root, 'portfolio', '**', 'index.html'),
                os.path.join(root, 'portfolio', '**', 'assets', '*.html')]
    seen = set()
    files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if p not in seen:
                seen.add(p)
                files.append(p)
    return files


def validate_self_contained(root):
    """Scan every portfolio/**/index.html AND portfolio/**/assets/*.html; return
    (file, offending_link) tuples for any src=/href= that escapes the portfolio
    (root-absolute or ../ parent escape). [] = clean."""
    return scan_links(_portfolio_files(root), _escapes_portfolio, root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description='Validate + publish the portfolio tree.')
    ap.add_argument('--to', default=None, help='target directory to copy portfolio/ into')
    ap.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                    help='actually copy (default is dry-run)')
    ap.set_defaults(dry_run=True)
    a = ap.parse_args(argv)

    violations = validate_self_contained(ROOT)
    if violations:
        print('PORTFOLIO NOT SELF-CONTAINED — %d offending link(s):' % len(violations))
        for f, link in violations:
            print('  %s -> %s' % (f, link))
        return 1
    print('portfolio is self-contained (no absolute / ../sessions links).')

    if a.to and not a.dry_run:
        target = os.path.join(a.to, 'portfolio')
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(os.path.join(ROOT, 'portfolio'), target)
        print('published portfolio/ -> %s' % target)
    elif a.to:
        print('[dry-run] would copy portfolio/ -> %s (pass --no-dry-run to publish)'
              % os.path.join(a.to, 'portfolio'))
    else:
        print('[dry-run] validation only; pass --to <dir> --no-dry-run to publish')
    return 0


if __name__ == '__main__':
    sys.exit(main())
