#!/usr/bin/env python3
"""
refactor_all_modules.py — migrate ALL remaining Frontier-Lab modules from the
legacy sessions/week-* layout to module folders with per-day subfolders.

Strategy: build a complete old-path -> new-path map for every lesson/review file,
then GLOBAL-REMAP every relative href/src by resolving it against the file's OLD
location and re-emitting it relative to its NEW location. This handles the
continuous cross-week nav chain, the many-to-many week splits, and merges
uniformly. Quest-ids are never touched (localStorage progress preserved).

  sessions/week-XX/day-YY-slug.html  ->  sessions/mNN-slug/day-ZZ-slug/lesson.html
  sessions/week-XX/review.html       ->  sessions/mNN-slug/review.html

DEFERRED (concurrent edits / not wired into index): m9b, m9e, m10b, m11, m22a-c.
m01 (week-f1) already migrated to m01-shape-of-data.

Usage:  python3 sessions/refactor_all_modules.py            # dry-run (default)
        python3 sessions/refactor_all_modules.py --apply     # perform migration
"""
import os, re, glob, subprocess, sys

SESS = os.path.dirname(os.path.abspath(__file__))

# module_slug -> ordered source spec; selector: 'all' | ('range', lo, hi) 1-based
# review_map: source-week -> review filename inside the module folder
SPEC = [
    ("m02-the-neuron",            [("week-f2","all"),("week-f3","all")],
        {"week-f3":"review.html","week-f2":"review-part-a.html"}),
    ("m03-attention",             [("week-f4","all")], {"week-f4":"review.html"}),
    ("m04-first-model-mlp",       [("week-f5","all")], {"week-f5":"review.html"}),
    ("m05a-text-transformer",     [("week-f6","all")], {"week-f6":"review.html"}),
    ("m05b-vision-transformer",   [("week-f7","all")], {"week-f7":"review.html"}),
    ("m06-cnns-vision-encoders",  [("week-f8","all")], {"week-f8":"review.html"}),
    ("m07-thinking-in-jax",       [("week-01","all")], {}),
    ("m08-transformer-math",      [("week-03","all")], {}),
    ("m09a-hardware-physics",     [("week-02","all")], {}),
    ("m09c-sharding-parallelism", [("week-04","all")], {}),
    ("m10a-scaling-laws",         [("week-05","all")], {}),
    ("m12-addition-transformer",  [("week-08","all"),("week-09","all"),("week-10",("range",1,3))], {}),
    ("m13-isoflops-scaling-law",  [("week-10",("range",4,6)),("week-11",("range",1,3))], {}),
    ("m14a-mixture-of-experts",   [("week-07","all")], {}),
    ("m14b-capstone-moe",         [("week-11",("range",4,6)),("week-12","all"),("week-17","all")], {}),
    ("m15a-custom-kernels-pallas",[("week-13","all")], {}),
    ("m15b-native-kernels",       [("week-14","all")], {}),
    ("m16a-inference-economics",  [("week-06","all")], {}),
    ("m17a-quantization",         [("week-15","all")], {}),
    ("m17b-long-context-decoding",[("week-16","all")], {}),
    ("m26-adrs",                  [("week-19","all"),("week-20","all")], {}),
    ("m28-formal-methods",        [("week-21","all")], {}),
    ("m29-ship-outreach",         [("week-23","all"),("week-24","all")], {}),
]
DEFERRED = {"week-m9b","week-m9e","week-m10b","week-m11","week-m22a","week-m22b","week-m22c"}
DONE_SRC = {"week-f1"}  # already migrated

def day_files(week):
    return sorted(glob.glob(os.path.join(SESS, week, "day-*.html")))

def stem_slug(path):
    return re.sub(r"^day-\d+-", "", os.path.splitext(os.path.basename(path))[0])

def build_newloc():
    """Return (newloc: old_relpath->new_relpath, per_module: slug->[(new_rel, is_review)])."""
    newloc, per_module = {}, {}
    for slug, sources, review_map in SPEC:
        seq = []
        for week, sel in sources:
            files = day_files(week)
            if sel != "all":
                _, lo, hi = sel
                files = files[lo-1:hi]
            seq.extend(files)
        rows = []
        for i, src in enumerate(seq, 1):
            new_rel = f"{slug}/day-{i:02d}-{stem_slug(src)}/lesson.html"
            newloc[os.path.relpath(src, SESS)] = new_rel
            rows.append((new_rel, False))
        for week, revname in review_map.items():
            rp = os.path.join(SESS, week, "review.html")
            if os.path.exists(rp):
                new_rel = f"{slug}/{revname}"
                newloc[os.path.relpath(rp, SESS)] = new_rel
                rows.append((new_rel, True))
        per_module[slug] = rows
    return newloc, per_module

# ---------- link/content rewriting ----------
def remap_href(u, old_dir_rel, new_dir_rel, newloc):
    """Given a relative href/src, return its rewritten form for the moved file."""
    if u.startswith(("http://","https://","mailto:","#","data:","tel:")): return u
    if any(c in u for c in ("'","+")) or u.startswith("{"): return u  # JS template junk
    base = u.split("#"); frag = "#"+base[1] if len(base) > 1 else ""
    path = base[0]
    if not path: return u
    old_target = os.path.normpath(os.path.join(old_dir_rel, path))     # sessions-relative
    new_target = newloc.get(old_target, old_target)                    # moved? else unchanged
    new_href = os.path.relpath(os.path.join(SESS,new_target), os.path.join(SESS,new_dir_rel))
    return new_href + frag

def rewrite_file(new_rel, old_rel, newloc):
    """Rewrite links + PRODUCE paths inside a migrated file at its NEW location."""
    p = os.path.join(SESS, new_rel)
    old_dir = os.path.dirname(old_rel)
    new_dir = os.path.dirname(new_rel)
    t = open(p, encoding="utf-8").read()
    # 1. every href/src
    def _sub(m):
        attr, q, val = m.group(1), m.group(2), m.group(3)
        return f'{attr}={q}{remap_href(val, old_dir, new_dir, newloc)}{q}'
    t = re.sub(r'(href|src)=(")([^"]*)"', _sub, t)
    # 2. PRODUCE: experiment paths + run commands + log callouts -> co-located
    mod_day_dir = os.path.dirname(new_rel)  # e.g. m10a-.../day-01-.../
    exp = f"sessions/{mod_day_dir}/experiment.py"
    log = f"sessions/{mod_day_dir}/log.md"
    t = re.sub(r'experiments/[A-Za-z0-9_./-]+\.py', exp, t)
    t = re.sub(r'python3 (?:experiments/[A-Za-z0-9_./-]+|m\d\d_[a-z0-9_]+)\.py', f'python3 {exp}', t)
    # any remaining experiments/<grouping>/<tail> (reports, dirs, .md/.json) -> co-located under the day folder
    def _exprel(mm):
        rest = mm.group(0)[len("experiments/"):]
        tail = rest.split("/", 1)[1] if "/" in rest else ""
        return f"sessions/{mod_day_dir}/{tail}"
    t = re.sub(r'experiments/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*/?', _exprel, t)
    t = re.sub(r'sessions/week-[a-z0-9]+/day-\d+-[a-z0-9-]+\.md', log, t)
    open(p, "w", encoding="utf-8").write(t)
    return exp in t  # whether this lesson has a co-located experiment

EXP_STUB = ("# {slug} - experiment\n#\n# Placeholder. Fill this from the lesson's PRODUCE step (open lesson.html):\n"
            "#   Option A: write it yourself, or  Option B: paste the frontier-experiment-lab prompt.\n"
            "# Then run:  python3 sessions/{d}/experiment.py\n")
LOG_STUB = "# {slug} - log\n\n_After you run `experiment.py`, write one line here (prompt is in `lesson.html`)._\n"

def main(apply):
    newloc, per_module = build_newloc()

    # coverage: every non-deferred, non-done week-*/*.html must be assigned
    all_html = glob.glob(os.path.join(SESS, "week-*", "*.html"))
    uncovered = []
    for f in all_html:
        rel = os.path.relpath(f, SESS)
        wk = rel.split("/")[0]
        if wk in DEFERRED or wk in DONE_SRC: continue
        if rel not in newloc: uncovered.append(rel)

    print(f"{'APPLY' if apply else 'DRY-RUN'} — {len(newloc)} files -> {len(per_module)} modules\n")
    for slug, rows in per_module.items():
        print(f"  {slug}  ({len(rows)} files)")
    print(f"\n  DEFERRED (untouched): {', '.join(sorted(DEFERRED))}")
    if uncovered:
        print(f"\n  !!! {len(uncovered)} UNCOVERED week files (would be orphaned):")
        for u in sorted(uncovered): print("     -", u)
    else:
        print("\n  coverage OK: every non-deferred week file is assigned.")

    if not apply:
        print("\n(dry-run only; pass --apply to perform)")
        return 0

    # 1. move every file (git mv preserves history + any working-tree edits)
    for old_rel, new_rel in newloc.items():
        dst = os.path.join(SESS, new_rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = subprocess.run(["git","mv",old_rel,new_rel], cwd=SESS, capture_output=True, text=True)
        if r.returncode != 0:
            os.replace(os.path.join(SESS,old_rel), dst)  # fallback
    # 2. rewrite links + PRODUCE inside every moved file; scaffold stubs
    for old_rel, new_rel in newloc.items():
        has_exp = rewrite_file(new_rel, old_rel, newloc)
        if new_rel.endswith("/lesson.html") and has_exp:
            d = os.path.dirname(new_rel); slug = os.path.basename(d)
            for fn, stub in (("experiment.py",EXP_STUB),("log.md",LOG_STUB)):
                fp = os.path.join(SESS, d, fn)
                if not os.path.exists(fp):
                    open(fp,"w",encoding="utf-8").write(stub.format(slug=slug, d=d))
    # 3. index.html — retarget every wired lesson/review path
    idxp = os.path.join(SESS,"index.html"); idx = open(idxp,encoding="utf-8").read()
    for old_rel, new_rel in newloc.items():
        idx = idx.replace(f"'{old_rel}'", f"'{new_rel}'")
    open(idxp,"w",encoding="utf-8").write(idx)
    # 4. progress.json — retarget page fields
    pj = os.path.join(SESS,"progress.json"); j = open(pj,encoding="utf-8").read()
    for old_rel, new_rel in newloc.items():
        j = j.replace(f"sessions/{old_rel}", f"sessions/{new_rel}")
    open(pj,"w",encoding="utf-8").write(j)
    # 5. remove emptied week dirs
    for wk in {r.split("/")[0] for r in newloc}:
        d = os.path.join(SESS, wk)
        if os.path.isdir(d) and not os.listdir(d): os.rmdir(d)
        elif os.path.isdir(d): print(f"  NOTE: {wk}/ not empty (leftover non-lesson files)")
    print("\n== migration applied ==")
    return 0

if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
