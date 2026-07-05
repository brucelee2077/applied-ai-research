#!/usr/bin/env python3
"""
refactor_to_modules.py — migrate one Frontier-Lab module from the legacy
`sessions/week-*` layout to the module layout, co-locating each day's
experiment + log inside a per-day folder:

    sessions/week-XX/day-NN-slug.html   ->   sessions/mNN/day-NN-slug/lesson.html
    sessions/week-XX/review.html        ->   sessions/mNN/review.html
    (new)                                    sessions/mNN/day-NN-slug/experiment.py
    (new)                                    sessions/mNN/day-NN-slug/log.md

It also rewrites every relative link affected by the +1 folder depth
(hub, sibling-day nav, review, viz iframes), rewrites the PRODUCE section so
experiments live in the module/day folder, updates the `index.html` MODULES
block + hero CTA, and fixes inbound nav links from the following module.

Config-driven: run one module at a time. Quest-ids are left UNCHANGED so
learners' localStorage progress is preserved.

Usage:  python3 sessions/refactor_to_modules.py m01
"""
import os, re, subprocess, sys

SESS = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- module configs
# Each config maps one legacy week folder -> one module folder.
# `days` are ordered slugs (filename without .html). `inbound` lists other
# lesson files (relative to sessions/) that link INTO this module and the
# (old -> new) path substring to fix in them.
CONFIGS = {
    "m01": {
        "src": "week-f1",
        "dst": "m01-shape-of-data",  # module folders carry a descriptive kebab slug after the number
        "label": "Module 1",
        "days": [
            "day-01-arrays",
            "day-02-indexing-slicing",
            "day-03-broadcasting-dtypes",
            "day-04-matmul-and-shapes",
            "day-05-logs-and-exponents",
            "day-06-random-seeds",
        ],
        "review": "review",
        # next module's first lesson links back to this module's review gate
        "inbound": [
            ("week-f2/day-01-single-neuron.html", "../week-f1/review.html", "../m01-shape-of-data/review.html"),
        ],
    },
}


def sh(args):
    subprocess.run(args, cwd=SESS, check=True)


def rewrite_day(text, dst, dayslug):
    """Rewrites for a lesson moved DOWN one folder (week-f1/x.html -> mNN/slug/lesson.html)."""
    qualified = f"sessions/{dst}/{dayslug}/experiment.py"
    # 1. hub link is now one level deeper
    text = text.replace('"../index.html"', '"../../index.html"')
    # 2. viz iframes / links are one level deeper
    text = text.replace('"../viz/', '"../../viz/')
    # 3. sibling-day nav: day-XX-slug.html -> ../day-XX-slug/lesson.html
    text = re.sub(r'href="(day-\d\d-[a-z0-9-]+)\.html"', r'href="../\1/lesson.html"', text)
    # 4. any direct review link (rare in day files) -> ../review.html
    text = text.replace('href="review.html"', 'href="../review.html"')
    # 5. PRODUCE target path -> co-located, fully qualified from repo root
    text = re.sub(r'experiments/foundations/m\d\d_[a-z0-9_]+\.py', qualified, text)
    # 6. bare run command -> co-located
    text = re.sub(r'python3 m\d\d_[a-z0-9_]+\.py', f'python3 {qualified}', text)
    # 7. log callout path -> co-located (any sessions/week-*/day-NN-*.md, not just -log.md)
    text = re.sub(r'sessions/week-[a-z0-9]+/day-\d\d-[a-z0-9-]+\.md', f'sessions/{dst}/{dayslug}/log.md', text)
    return text


def rewrite_review(text):
    """review.html stays at module root (same depth as before): only sibling-day
    links change, since days are now in subfolders."""
    text = re.sub(r'href="(day-\d\d-[a-z0-9-]+)\.html"', r'href="\1/lesson.html"', text)
    return text


EXP_STUB = """# {label} - {slug} - experiment
#
# Placeholder. Fill this from the lesson's PRODUCE step (open lesson.html):
#   Option A: write it yourself, or
#   Option B: paste the frontier-experiment-lab prompt from the lesson.
# Then run:  python3 sessions/{dst}/{slug}/experiment.py
"""

LOG_STUB = """# {label} - {slug} - log

_After you run `experiment.py`, write one line here. The reflection prompt is in `lesson.html`._
"""


def migrate(cfg):
    src, dst, label = cfg["src"], cfg["dst"], cfg["label"]
    src_abs = os.path.join(SESS, src)
    if not os.path.isdir(src_abs):
        sys.exit(f"! source folder {src} not found (already migrated?) — aborting")

    print(f"== migrating {src} -> {dst} ({label}) ==")

    # --- move + rewrite each day lesson ---
    for slug in cfg["days"]:
        daydir = os.path.join(SESS, dst, slug)
        os.makedirs(daydir, exist_ok=True)
        sh(["git", "mv", f"{src}/{slug}.html", f"{dst}/{slug}/lesson.html"])
        p = os.path.join(daydir, "lesson.html")
        with open(p, encoding="utf-8") as f:
            t = f.read()
        with open(p, "w", encoding="utf-8") as f:
            f.write(rewrite_day(t, dst, slug))
        # scaffold co-located experiment + log stubs (rewire-only: no code is run)
        exp = os.path.join(daydir, "experiment.py")
        log = os.path.join(daydir, "log.md")
        if not os.path.exists(exp):
            open(exp, "w", encoding="utf-8").write(EXP_STUB.format(label=label, slug=slug, dst=dst))
        if not os.path.exists(log):
            open(log, "w", encoding="utf-8").write(LOG_STUB.format(label=label, slug=slug))
        print(f"  moved {slug}.html -> {dst}/{slug}/lesson.html  (+ experiment.py, log.md)")

    # --- move + rewrite review gate (stays at module root) ---
    if cfg.get("review"):
        rv = cfg["review"]
        sh(["git", "mv", f"{src}/{rv}.html", f"{dst}/{rv}.html"])
        p = os.path.join(SESS, dst, f"{rv}.html")
        with open(p, encoding="utf-8") as f:
            t = f.read()
        with open(p, "w", encoding="utf-8") as f:
            f.write(rewrite_review(t))
        print(f"  moved {rv}.html -> {dst}/{rv}.html")

    # --- remove now-empty legacy folder ---
    try:
        os.rmdir(src_abs)
        print(f"  removed empty {src}/")
    except OSError:
        print(f"  NOTE: {src}/ not empty — left in place")

    # --- update index.html (MODULES paths + hero CTA) ---
    idxp = os.path.join(SESS, "index.html")
    with open(idxp, encoding="utf-8") as f:
        idx = f.read()
    for slug in cfg["days"]:
        idx = idx.replace(f"'{src}/{slug}.html'", f"'{dst}/{slug}/lesson.html'")
    if cfg.get("review"):
        idx = idx.replace(f"'{src}/{cfg['review']}.html'", f"'{dst}/{cfg['review']}.html'")
    with open(idxp, "w", encoding="utf-8") as f:
        f.write(idx)
    print("  updated index.html (MODULES paths + CTA)")

    # --- fix inbound nav links from the following module ---
    for relfile, old, new in cfg.get("inbound", []):
        p = os.path.join(SESS, relfile)
        with open(p, encoding="utf-8") as f:
            t = f.read()
        n = t.count(old)
        t = t.replace(old, new)
        with open(p, "w", encoding="utf-8") as f:
            f.write(t)
        print(f"  fixed inbound link in {relfile} ({n}x: {old} -> {new})")

    print("== done ==")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "m01"
    if which not in CONFIGS:
        sys.exit(f"unknown module '{which}'. known: {', '.join(CONFIGS)}")
    migrate(CONFIGS[which])
