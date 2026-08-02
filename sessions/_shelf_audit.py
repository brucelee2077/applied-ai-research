#!/usr/bin/env python3
"""
_shelf_audit.py — assert every Toy Shelf row points at something real.

Checks:
  1. every TOYS page exists on disk AND is tracked by git
  2. every TOYS owner quest-id appears as a data-quest-id on some tracked page
  3. no viz page that IS embedded by a lesson is silently missing from TOYS
  4. the copy of shelf.js inlined in index.html matches the module source

Run: python3 sessions/_shelf_audit.py      (exit 0 = pass)
"""
import re, os, sys, glob, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
SHELF_JS = os.path.join(BASE, "_compiler", "shells", "js", "shelf.js")
INDEX = os.path.join(BASE, "index.html")
EXCLUDED = {"attention-heatmap", "attention-multihead", "attention-pipeline",
            "softmax-scaling", "leaky-slope"}

fails = []


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=os.path.dirname(BASE),
                         capture_output=True, text=True).stdout
    # splitlines(), not split() — 181 tracked paths contain spaces ("ML Design/…")
    return set(out.splitlines())


def parse_toys(js):
    """Pull the 5-string rows out of the TOYS array literal."""
    m = re.search(r"export var TOYS = \[(.*?)\n\];", js, re.S)
    if not m:
        fails.append("could not find the TOYS array in shelf.js")
        return []
    rows = re.findall(r"\[\s*((?:'[^']*'\s*,\s*){4}'[^']*')\s*\]", m.group(1))
    return [[f.strip().strip("'") for f in re.findall(r"'([^']*)'", r)] for r in rows]


def main():
    js = open(SHELF_JS, encoding="utf-8").read()
    toys = parse_toys(js)
    tracked = tracked_files()

    pages = sorted(glob.glob(os.path.join(BASE, "m*", "day-*", "lesson.html")) +
                   glob.glob(os.path.join(BASE, "week-m*", "day-*.html")) +
                   glob.glob(os.path.join(BASE, "m*", "review*.html")) +
                   glob.glob(os.path.join(BASE, "week-m*", "review*.html")))
    pages = [p for p in pages
             if os.path.relpath(p, os.path.dirname(BASE)) in tracked]

    qids, embedded = set(), set()
    for p in pages:
        t = open(p, encoding="utf-8", errors="ignore").read()
        q = re.search(r'data-quest-id="([^"]+)"', t)
        if q:
            qids.add(q.group(1))
        embedded.update(re.findall(r"viz/([a-z0-9-]+)\.html", t))

    # 1 + 2
    for name, label, qid, verb, page in toys:
        full = os.path.join(BASE, page)
        if not os.path.isfile(full):
            fails.append(f"{name}: page missing on disk -> {page}")
        elif os.path.relpath(full, os.path.dirname(BASE)) not in tracked:
            fails.append(f"{name}: page is UNTRACKED (would 404 on Pages) -> {page}")
        if qid not in qids:
            fails.append(f"{name}: owner quest-id '{qid}' is on no tracked page")

    # 3 — an embedded viz page that nobody put on the shelf
    on_shelf = {os.path.basename(t[4])[:-5] for t in toys}
    for v in sorted(embedded - on_shelf - EXCLUDED):
        fails.append(f"viz/{v}.html is embedded by a lesson but missing from TOYS")
    # and the reverse: an excluded page that sneaked on
    for v in sorted(on_shelf & EXCLUDED):
        fails.append(f"viz/{v}.html has no owning lesson but is on the shelf")

    # 4 — inlined copy parity
    idx = open(INDEX, encoding="utf-8").read()
    m = re.search(r"/\* SHELF-LOGIC:BEGIN \*/(.*?)/\* SHELF-LOGIC:END \*/", idx, re.S)
    if not m:
        fails.append("index.html has no SHELF-LOGIC:BEGIN/END markers")
    else:
        # Must match the inliner exactly: sed 's/^export //'
        # i.e. strip `export ` only at line starts, not anywhere in the line.
        expect = re.sub(r"^export ", "", js, flags=re.M).strip()
        got = m.group(1).strip()
        if got != expect:
            fails.append("index.html's inlined shelf logic has DRIFTED from shelf.js "
                         "(re-inline with: sed 's/^export //' "
                         "sessions/_compiler/shells/js/shelf.js)")

    if fails:
        print(f"SHELF AUDIT: {len(fails)} problem(s)")
        for f in fails:
            print("  ✗", f)
        return 1
    print(f"SHELF AUDIT: OK — {len(toys)} toys, all pages tracked, all owners real, inline copy in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
