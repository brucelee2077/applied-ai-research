#!/usr/bin/env python3
"""
One-off sweep: fix the shell-migration metadata bug across the whole
`sessions/` curriculum (see sessions/coach_layer_pilot_report.md §11-12,
risk #1). `_shell_migrate.py`'s render() substituted into a `.eyebrow` span
that doesn't exist in the new-shell template (only `.kicker` does), and never
touched the sidebar `.nav-group-label` or the `.fin` finale at all — so every
migrated lesson except the 4 hand-fixed Coach Layer pilots still shows the
template's own leftover identity ("Module 1 · Represent · Day 3", the
broadcasting/dtypes finale).

This script re-derives the correct kicker / nav-group-label / finale <h3> for
every lesson from that lesson's own <title> (day/week identity) plus
sessions/index.html's MODULES table (module display name) — see
_metadata_derive.py. The finale <p> body is only replaced when it is still
byte-identical to the template's leftover paragraph (never overwrites
hand-written recap text), with a generic-but-accurate replacement built from
this file's own <h1> and its own sidebar "next" link — no new lesson content
is authored.

Usage:
  python3 sessions/_fix_shell_metadata_sweep.py --check   # dry run, prints a diff summary
  python3 sessions/_fix_shell_metadata_sweep.py --apply   # write changes
"""
import re
import sys
import glob
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _metadata_derive as md

TEMPLATE_PATH = os.path.join(md.REPO, "sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html")

LEFTOVER_FINALE_P = (
    '<p>You can predict a broadcast from two shapes and explain why '
    '<code>float32</code> halves the memory of <code>float64</code>.<br>'
    'Next up: <b>Matmul &amp; Shapes</b> — the one operation every neural '
    'network is built from.</p>'
)


def esc(s):
    return s.replace('&', '&amp;')


def process(path, modules, apply):
    text = open(path, encoding="utf-8").read()
    orig = text

    tm = re.search(r'<title>(.*?)</title>', text, re.S)
    if not tm:
        return None, f"no <title> found"
    title = tm.group(1)

    num = md.module_num_from_path(path)
    if num is None or num not in modules:
        return None, f"could not resolve module number for path (got {num!r})"

    got = md.derive_metadata(title, num, modules)
    if got is None:
        return None, f"title did not match expected pattern: {title!r}"

    changes = []

    new_kicker = esc(got['kicker'])
    m = re.search(r'<span class="kicker">(.*?)</span>', text, re.S)
    if m and m.group(1) != new_kicker:
        text = text[:m.start(1)] + new_kicker + text[m.end(1):]
        changes.append(("kicker", m.group(1), new_kicker))

    new_navlabel = esc(got['nav_group_label'])
    m = re.search(r'(<nav aria-label="Sections">\s*<div class="nav-group-label">)(.*?)(</div>)', text, re.S)
    if m and m.group(2) != new_navlabel:
        text = text[:m.start(2)] + new_navlabel + text[m.end(2):]
        changes.append(("nav_group_label", m.group(2), new_navlabel))

    new_h3 = esc(got['finale_h3'])
    m = re.search(r'(<div class="fin" id="fin".*?<h3>)(.*?)(</h3>)', text, re.S)
    if m and m.group(2).strip() != new_h3:
        text = text[:m.start(2)] + new_h3 + text[m.end(2):]
        changes.append(("finale_h3", m.group(2), new_h3))

    if LEFTOVER_FINALE_P in text:
        h1m = re.search(r'<h1>(.*?)<span class="sub">', text, re.S)
        h1_main = h1m.group(1).strip() if h1m else None
        nextm = re.search(r'<a class="lnav next"[^>]*><span class="d">[^<]*</span><span class="t">(.*?)</span></a>', text, re.S)
        next_disabled_m = re.search(r'<span class="lnav next disabled">.*?<span class="t">(.*?)</span></span>', text, re.S)
        if h1_main and nextm:
            new_p = f"<p>Nice work — you've completed <b>{h1_main}</b>.<br>Next up: <b>{nextm.group(1)}</b>.</p>"
        elif h1_main and next_disabled_m:
            new_p = f"<p>Nice work — you've completed <b>{h1_main}</b>. That's the last lesson in this run — check the curriculum map for what's next.</p>"
        else:
            new_p = None
        if new_p:
            text = text.replace(LEFTOVER_FINALE_P, new_p, 1)
            changes.append(("finale_p", LEFTOVER_FINALE_P, new_p))

    if not changes:
        return [], None

    if apply and text != orig:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    return changes, None


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "--check"
    apply = mode == "--apply"

    modules = md.load_modules()
    files = sorted(glob.glob(os.path.join(md.REPO, "sessions/**/lesson.html"), recursive=True))
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(TEMPLATE_PATH)]

    total_changed = 0
    field_counts = {}
    errors = []

    for f in files:
        rel = os.path.relpath(f, md.REPO)
        changes, err = process(f, modules, apply)
        if err:
            errors.append((rel, err))
            continue
        if changes:
            total_changed += 1
            for field, before, after in changes:
                field_counts[field] = field_counts.get(field, 0) + 1

    print(f"mode: {'APPLY' if apply else 'CHECK (dry run)'}")
    print(f"files scanned: {len(files)}")
    print(f"files changed: {total_changed}")
    print(f"by field: {field_counts}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for rel, err in errors:
            print(f"  {rel}: {err}")


if __name__ == "__main__":
    main()
