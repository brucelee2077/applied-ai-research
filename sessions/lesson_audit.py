#!/usr/bin/env python3
"""
Lesson integrity audit for the Frontier Lab curriculum (Coach Layer aware).

Discovers lessons on the CURRENT module-folder layout (sessions/m*/day-*/lesson.html and
sessions/week-*/**/lesson.html) rather than the old flat week-NN/slug.html scheme, and works on
BOTH lesson shells: the old top-nav shell (`class="sec"`) and the new sidebar + appearance-switcher
shell (`class="module-section"`). Structural counts are therefore based on shell-agnostic markers
(`data-sec=`, `class="gotit"`, `data-demo=`, the quiz literal) instead of a shell-specific class.

Status per file:
  OK        — exists, has a quest-id, 7 sections / 7 got-it / 3 demos / 4 quiz, a BUILD array, and
              no vmap-template leftover. (Chinese characters are NOT a defect — they are the
              intended bilingual Coach Layer scaffold.)
  MISSING   — a requested path does not exist.
  LEFTOVER  — still contains day-03-vmap template fingerprints (a failed copy/verify).
  DEGRADED  — exists but structurally off (wrong section/gotit/demo/quiz counts, no quest-id,
              missing BUILD array, or an old click-stepper leftover).

Separately, every OK/DEGRADED lesson also gets soft **COACH advisories** — non-failing hints about
missing Coach Layer elements (no analogy, math-heavy without a Math Ladder, no Produce artifact, no
Staff Lens, no interview-ready explanation, no acceptance criteria, no bilingual scaffold, ...).
Advisories never change a file's status and never fail the run; they are guidance for the refactor.

Usage:
  python3 sessions/lesson_audit.py                     # audit every discovered lesson (writes _recover_set.json)
  python3 sessions/lesson_audit.py <path> [<path> ...] # audit only these files/dirs (does NOT write _recover_set.json)
"""
import os
import re
import glob
import json
import sys

BASE = os.path.dirname(os.path.abspath(__file__))  # sessions/

# ---------------------------------------------------------------------------
# vmap-template fingerprints. A file is contaminated if it has the vmap quest id, the vmap quiz
# text, OR all three vmap playground demo keys together (a lone 'loop' key is legitimate).
# ---------------------------------------------------------------------------
STRONG_MARKERS = ['w01-d03-vmap', 'jax.vmap(f)(X)']
VMAP_DEMO_TRIO = ['data-demo="loop"', 'data-demo="vmap"', 'data-demo="inaxes"']


def discover(argv):
    """Return a sorted list of lesson.html paths to audit.

    With no CLI args: glob the whole current layout. With args: each arg is a lesson.html file, or a
    directory to search recursively. The distinction matters — full-scan mode owns _recover_set.json,
    targeted mode leaves it alone (so a targeted pilot run never clobbers another session's state).
    """
    if argv:
        paths = []
        for a in argv:
            if os.path.isdir(a):
                paths += glob.glob(os.path.join(a, "**", "lesson.html"), recursive=True)
            elif a.endswith(".html"):
                paths.append(a)
            else:
                # bare module/day slug -> try to resolve under BASE
                paths += glob.glob(os.path.join(BASE, a, "**", "lesson.html"), recursive=True)
        return sorted(set(os.path.abspath(p) for p in paths)), False  # targeted: don't write recover set
    paths = glob.glob(os.path.join(BASE, "m*", "**", "lesson.html"), recursive=True)
    paths += glob.glob(os.path.join(BASE, "week-*", "**", "lesson.html"), recursive=True)
    return sorted(set(os.path.abspath(p) for p in paths)), True  # full scan: owns _recover_set.json


def rel(path):
    try:
        return os.path.relpath(path, BASE)
    except ValueError:
        return path


def classify(path):
    """Return (status, hard_reasons, coach_advisories, info_notes)."""
    if not os.path.exists(path):
        return "MISSING", ["file does not exist"], [], []
    t = open(path, encoding="utf-8").read()
    hard, coach, info = [], [], []

    quest_id = None
    m = re.search(r'data-quest-id="([^"]*)"', t)
    if m:
        quest_id = m.group(1)
        info.append(f"quest-id: {quest_id}")
    else:
        hard.append("no data-quest-id on <body> (localStorage progress key is missing)")

    # ---- vmap leftover (skip the genuine vmap lesson itself) ----
    is_vmap_lesson = (quest_id == "w01-d03-vmap") or ("day-03-vmap" in path)
    leftover = False
    if not is_vmap_lesson:
        for mk in STRONG_MARKERS:
            if mk in t:
                hard.append(f"leftover-marker: {mk}")
                leftover = True
        if all(mk in t for mk in VMAP_DEMO_TRIO):
            hard.append("leftover-marker: vmap demo trio (loop/vmap/inaxes)")
            leftover = True

    # ---- structural counts (shell-agnostic) ----
    secs = len(re.findall(r'data-sec="\w+"', t))
    gotit = t.count('class="gotit"')
    demos = t.count('data-demo=')
    quiz = len(re.findall(r'\{\s*"?q"?\s*:', t))
    if secs != 7:
        hard.append(f"sections (data-sec)={secs} (want 7)")
    if gotit != 7:
        hard.append(f"gotit={gotit} (want 7)")
    if demos != 3:
        hard.append(f"data-demo={demos} (want 3)")
    if quiz != 4:
        hard.append(f"quiz={quiz} (want 4)")
    if 'var BUILD=' not in t:
        hard.append("missing BUILD array (not scroll-reveal format)")
    if 'renderStep' in t or 'data-sec="code"' in t:
        hard.append("old click-stepper leftover")

    # orphaned demo keys: every data-demo value must appear as a DEMOS key (quoted or bare)
    for dv in set(re.findall(r'data-demo="([^"]+)"', t)):
        if not re.search(r'''['"]?''' + re.escape(dv) + r'''['"]?\s*:\s*\{''', t):
            hard.append(f"orphaned demo key '{dv}' (button has no DEMOS entry)")

    # ---- Chinese = bilingual scaffold, NOT a defect ----
    zh = len(re.findall(r'[一-鿿]', t))
    if zh:
        info.append(f"bilingual scaffold: {zh} Chinese chars present (intended, not a defect)")

    # ---- COACH advisories (soft; never change status) ----
    low = t.lower()
    has_analogy = ('class="relate"' in t) or ('class="card"' in t)
    if not has_analogy:
        coach.append("no analogy block (.relate cards) — Reader A has no everyday hook")
    math_heavy = ('🧮' in t) or (t.count('<code>') > 12)
    if math_heavy and ('math ladder' not in low):
        coach.append("math-heavy but no Math Ladder (words -> labeled formula -> tiny numbers -> sanity check)")
    if 'experiment.py' not in low:
        coach.append("no concrete Produce artifact (experiment.py) referenced")
    if not any(k in low for k in ('staff lens', 'staff / research', 'staff engineer', 'the staff lens')):
        coach.append("no Staff / Research Engineer Lens (silent failure + trade-off)")
    if 'interview' not in low and '🎤' not in t:
        coach.append("no interview-ready explanation")
    if 'acceptance criteria' not in low:
        coach.append("no explicit Acceptance criteria in Produce")
    if not any(k in t for k in ('容易卡住', 'Why this trips', '😕')):
        coach.append("no pain-point block (why this is confusing)")
    if 'research log' not in low and '📓' not in t:
        coach.append("no 5-minute research log")
    if zh == 0:
        coach.append("no bilingual scaffold (add 2-4 light Chinese intuition touches)")

    if leftover:
        return "LEFTOVER", hard, coach, info
    if hard:
        return "DEGRADED", hard, coach, info
    return "OK", [], coach, info


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    paths, write_recover = discover(argv)
    if not paths:
        print("No lesson.html files found. (Looked under sessions/m*/ and sessions/week-*/.)")
        return 1

    buckets = {"OK": [], "MISSING": [], "LEFTOVER": [], "DEGRADED": []}
    coach_map = {}
    for p in paths:
        status, hard, coach, info = classify(p)
        buckets[status].append((rel(p), hard, info))
        if coach:
            coach_map[rel(p)] = coach

    total = sum(len(v) for v in buckets.values())
    print(f"TOTAL lessons audited: {total}")
    for k in ("OK", "MISSING", "LEFTOVER", "DEGRADED"):
        print(f"  {k}: {len(buckets[k])}")

    for k in ("MISSING", "LEFTOVER", "DEGRADED"):
        if buckets[k]:
            print(f"\n=== {k} ===")
            for r, hard, info in buckets[k]:
                print(f"  {r}")
                for reason in hard:
                    print(f"       - {reason}")

    # Coach advisories (do not affect pass/fail)
    if coach_map:
        print(f"\n=== COACH advisories (soft — not failures) ===")
        for r in sorted(coach_map):
            print(f"  {r}")
            for c in coach_map[r]:
                print(f"       ~ {c}")

    hard_fail = [r for k in ("MISSING", "LEFTOVER", "DEGRADED") for r, _, _ in buckets[k]]
    if write_recover:
        open(os.path.join(BASE, "_recover_set.json"), "w").write(json.dumps(hard_fail, indent=2))
        print(f"\nRecovery set ({len(hard_fail)}) written to sessions/_recover_set.json")
    else:
        print(f"\n(targeted run — _recover_set.json left unchanged)")

    return 0 if not hard_fail else 1


if __name__ == "__main__":
    sys.exit(main())
