#!/usr/bin/env python3
"""
Audit a migrated review gate against its pre-migration original: confirms the
CHECKS/QS data literals and all three section bodies (self-check / quiz /
verdict) transferred byte-for-byte, hero + nav + finale content is present, and
the new file is on the sidebar shell (no old top-nav chrome).

Usage: python3 sessions/_review_shell_audit.py <old_review.html> <new_review.html>
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shell_migrate as sm
import _review_shell_migrate as rm


def new_section_bodies(c):
    pat = re.compile(
        r'<section class="module-section" id="(s\d)" data-sec="(\w+)">\s*'
        r'<div class="sec-head">.*?</div>\s*'
        r'<div class="sec-body">(.*?)</div>\s*</section>', re.S)
    return {m.group(2): m.group(3) for m in pat.finditer(c)}


def audit_pair(old_path, new_path):
    problems = []
    old = rm.extract(old_path)          # structured extraction of the original
    new_c = sm.read(new_path)

    # 1. data literals byte-identical
    for var, ch in [("var CHECKS", "["), ("var QS", "[")]:
        try:
            nl = sm.extract_js_literal(new_c, var, ch, new_path)
        except Exception as e:
            problems.append(f"{var}: not found in new: {e}")
            continue
        ol = old["checks_js"] if var == "var CHECKS" else old["qs_js"]
        if ol != nl:
            problems.append(f"{var}: literal DIFFERS (old {len(ol)}B, new {len(nl)}B)")

    # 2. section bodies byte-identical
    nb = new_section_bodies(new_c)
    for s in old["sections"]:
        if s["key"] not in nb:
            problems.append(f'section {s["key"]}: missing in new')
        elif nb[s["key"]] != s["body"]:
            problems.append(f'section {s["key"]}: body DIFFERS (old {len(s["body"])}B, new {len(nb[s["key"]])}B)')

    # 3. content present (hero / nav / finale / sec headers)
    present = {
        "title": old["title"], "quest-id": old["qid"], "nav_title": old["nav_title"],
        "h1": old["h1"], "lead": old["lead"],
        "prev_href": old["prev_href"], "prev_label": old["prev_label"],
        "next_href": old["next_href"], "next_label": old["next_label"],
        "fin_em": old["fin_em"], "fin_h3": old["fin_h3"], "fin_p": old["fin_p"],
    }
    for label, val in present.items():
        if val and val not in new_c:
            problems.append(f"content missing in new: {label} ({val[:40]!r})")
    for s in old["sections"]:
        if s["sec_h"] and s["sec_h"] not in new_c:
            problems.append(f'sec-h missing in new: {s["key"]} ({s["sec_h"][:40]!r})')

    # 4. structural — new file is on the sidebar shell
    checks = {
        "3 module-section blocks": new_c.count('class="module-section"') == 3,
        "sidebar present": 'id="sidebar"' in new_c,
        "content column present": 'id="content"' in new_c,
        "4 appearance modes": new_c.count("theme-btn") >= 4,
        "verdict section kept (hub signal)": 'data-sec="verdict"' in new_c,
        "no old top-nav bar": '<nav class="nav">' not in new_c,
        "no old lesson-nav": 'lesson-nav' not in new_c,
        "side-nav prev/next/map": new_c.count('class="lnav') >= 2 and 'lnav-hub' in new_c,
    }
    for label, ok in checks.items():
        if not ok:
            problems.append(f"structural check failed: {label}")

    return problems


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    problems = audit_pair(sys.argv[1], sys.argv[2])
    if problems:
        print(f"FAIL {sys.argv[2]}")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK   {sys.argv[2]}")


if __name__ == "__main__":
    main()
