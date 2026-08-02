#!/usr/bin/env python3
"""
_shelf_audit.py — assert every Toy Shelf row points at something real.

Checks:
  1. every TOYS page exists on disk AND is tracked by git
  2. every TOYS owner quest-id appears as a data-quest-id on some tracked page
  3. no viz page that IS embedded by a lesson is silently missing from TOYS
  4. the copy of shelf.js inlined in index.html matches the module source
  5. shelf.js's viz message type still matches the donor the lessons compile against

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
    try:
        r = subprocess.run(["git", "ls-files"], cwd=os.path.dirname(BASE),
                           capture_output=True, text=True)
    except FileNotFoundError:
        fails.append("git is not on PATH — cannot tell tracked pages from stray ones")
        return set()
    if r.returncode != 0:
        fails.append(f"git ls-files failed (exit {r.returncode}): "
                     f"{r.stderr.strip() or '<no stderr>'}")
        return set()
    # splitlines(), not split() — 181 tracked paths contain spaces ("ML Design/…")
    return set(r.stdout.splitlines())


def parse_toys(js):
    """Pull the 5-string rows out of the TOYS array literal.

    Fails CLOSED: a row this cannot parse is a reported failure, never a silent
    skip. A silently dropped row is never checked for existence, tracking or a
    real owner, yet the audit would still print OK — the exact hole this script
    exists to close.
    """
    m = re.search(r"export var TOYS = \[(.*?)\n\];", js, re.S)
    if not m:
        fails.append("could not find the TOYS array in shelf.js")
        return []
    body = m.group(1)
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        # escape-aware, same idiom as nav_audit.py: '((?:[^'\\]|\\.)*)'
        rm = re.match(r"\[\s*((?:'(?:[^'\\]|\\.)*'\s*,\s*){4}'(?:[^'\\]|\\.)*')\s*\],?$", line)
        if not rm:
            fails.append(f"unparseable TOYS row: {line[:60]}")
            continue
        rows.append(re.findall(r"'((?:[^'\\]|\\.)*)'", rm.group(1)))
    # Independent count of row-opening lines, so no future parser rewrite can
    # drop a row in silence either.
    opened = sum(1 for ln in body.splitlines() if ln.strip().startswith("["))
    if opened != len(rows):
        fails.append(f"TOYS row count mismatch: {opened} line(s) open a row with '[' "
                     f"but only {len(rows)} parsed cleanly")
    return rows


def report(n_toys):
    if fails:
        print(f"SHELF AUDIT: {len(fails)} problem(s)")
        for f in fails:
            print("  ✗", f)
        return 1
    print(f"SHELF AUDIT: OK — {n_toys} toys, all pages tracked, all owners real, inline copy in sync")
    return 0


def main():
    js = open(SHELF_JS, encoding="utf-8").read()
    toys = parse_toys(js)
    tracked = tracked_files()
    if not tracked:
        # Without the tracked set every row below looks broken; that flood would
        # bury the real cause reported by tracked_files().
        fails.append("no tracked files to check against — stopping before the noise")
        return report(len(toys))

    pages = sorted(glob.glob(os.path.join(BASE, "m*", "day-*", "lesson.html")) +
                   glob.glob(os.path.join(BASE, "week-m*", "day-*.html")) +
                   glob.glob(os.path.join(BASE, "m*", "review*.html")) +
                   glob.glob(os.path.join(BASE, "week-m*", "review*.html")))
    pages = [p for p in pages
             if os.path.relpath(p, os.path.dirname(BASE)) in tracked]

    qids, embedded = set(), set()
    for p in pages:
        t = open(p, encoding="utf-8", errors="ignore").read()
        qids.update(re.findall(r'data-quest-id="([^"]+)"', t))
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
    on_shelf = {os.path.basename(t[4]).removesuffix(".html") for t in toys}
    for v in sorted(embedded - on_shelf - EXCLUDED):
        fails.append(f"viz/{v}.html is embedded by a lesson but missing from TOYS")
    # and the reverse: an excluded page that sneaked on
    for v in sorted(on_shelf & EXCLUDED):
        fails.append(f"viz/{v}.html has no owning lesson but is on the shelf")
    # EXCLUDED is a hand judgement about today's repo — re-check it, or a page that
    # gains an owner stays invisible to check 3 forever.
    for v in sorted(embedded & EXCLUDED):
        fails.append(f"viz/{v}.html is now embedded by a lesson — it has an owner, "
                     f"remove it from EXCLUDED and add a TOYS row")

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

    # 5 — the message type must match the donor the lessons compile against
    donor = os.path.join(BASE, "_compiler", "shells", "v9-base.donor")
    dtxt = open(donor, encoding="utf-8").read()
    dm = re.search(r"d\.type\s*!==\s*'([a-z-]+)'", dtxt)
    jm = re.search(r"VIZ_MSG_TYPE = '([a-z-]+)'", js)
    if not dm or not jm:
        fails.append("could not read the viz message type from donor and/or shelf.js")
    elif dm.group(1) != jm.group(1):
        fails.append(f"message type drift: donor says '{dm.group(1)}', "
                     f"shelf.js says '{jm.group(1)}'")

    return report(len(toys))


if __name__ == "__main__":
    sys.exit(main())
