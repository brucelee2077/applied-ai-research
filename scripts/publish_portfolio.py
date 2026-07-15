#!/usr/bin/env python3
# =============================================================================
# publish_portfolio (Plan 2)  — validate + publish the self-contained portfolio.
# =============================================================================
# Gate before publishing: the portfolio/ tree must be SELF-CONTAINED. Every
# src=/href= in every portfolio/**/index.html must stay inside portfolio/ —
# no absolute paths ("/...") and no "../sessions" escapes. If it isn't self-
# contained, publishing anywhere else would produce dead links, so we refuse.
#
# validate_self_contained(root) -> [(file, offending_link), ...]  ([] = clean)
#
# CLI:
#   python3 scripts/publish_portfolio.py                 # dry-run, validate only
#   python3 scripts/publish_portfolio.py --to <dir> --no-dry-run   # copy portfolio/
# Always validates first; exits 1 (printing violations) if not self-contained.
# =============================================================================
import sys, os, re, glob, shutil, argparse

# repo root = one dir up from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# capture the value of any src="..."/href="..." (or single-quoted)
_LINK_RE = re.compile(r'''(?:src|href)\s*=\s*["']([^"']*)["']''', re.IGNORECASE)


def _escapes_portfolio(link):
    """True if a link leaves the portfolio tree: absolute path or ../sessions."""
    l = link.strip()
    if l.startswith('/'):
        return True
    if '../sessions' in l:
        return True
    return False


def validate_self_contained(root):
    """Scan every portfolio/**/index.html; return (file, offending_link) tuples for
    any src=/href= that escapes the portfolio (absolute or ../sessions). [] = clean."""
    violations = []
    pattern = os.path.join(root, 'portfolio', '**', 'index.html')
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            html = open(path, encoding='utf-8').read()
        except Exception:
            continue
        for link in _LINK_RE.findall(html):
            if _escapes_portfolio(link):
                violations.append((os.path.relpath(path, root), link))
    return violations


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
