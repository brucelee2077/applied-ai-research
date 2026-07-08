#!/usr/bin/env python3
"""
Audit a migrated lesson.html (or lesson.new.html) against its pre-migration
original: confirms every pedagogical data block (DEMOS/BUILD/QS literals, all
7 section bodies) transferred byte-for-byte, and that hero/nav fields are
present and sane. Used to verify sessions/_shell_migrate.py output.

Usage: python3 sessions/_shell_audit.py <old_lesson.html> <new_lesson.html>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shell_migrate as m


def audit_pair(old_path, new_path):
    old_c = m.read(old_path)
    new_c = m.read(new_path)
    problems = []

    for var, ch in [("var DEMOS", "{"), ("var BUILD", "["), ("var QS", "[")]:
        try:
            ol = m.extract_js_literal(old_c, var, ch, old_path)
            nl = m.extract_js_literal(new_c, var, ch, new_path)
        except Exception as e:
            problems.append(f"{var}: extraction failed: {e}")
            continue
        if ol != nl:
            problems.append(f"{var}: literal DIFFERS (old {len(ol)}B, new {len(nl)}B)")

    old_d = m.extract_old(old_path)
    for s in old_d["sections"]:
        sec_pat = m.re.compile(
            r'<section class="module-section" id="' + s["id"] + r'" data-sec="[a-z]+">.*?<div class="sec-body">(.*?)</div>\s*</section>',
            m.re.S,
        )
        mm = sec_pat.search(new_c)
        if not mm:
            problems.append(f'{s["id"]}: not found in new file')
            continue
        if mm.group(1) != s["body"]:
            problems.append(f'{s["id"]}: body DIFFERS (old {len(s["body"])}B, new {len(mm.group(1))}B)')

    # structural sanity on the new file
    checks = {
        "7 module-section blocks": new_c.count('class="module-section"') == 7,
        "sidebar present": 'id="sidebar"' in new_c,
        "4 appearance modes": new_c.count("theme-btn") >= 4,
        "quest-id preserved": old_d["quest_id"] in new_c,
        "title preserved": old_d["title"] in new_c,
    }
    for label, ok in checks.items():
        if not ok:
            problems.append(f"structural check failed: {label}")

    return problems


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    old_path, new_path = sys.argv[1], sys.argv[2]
    problems = audit_pair(old_path, new_path)
    if problems:
        print(f"FAIL {new_path}")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK   {new_path}")


if __name__ == "__main__":
    main()
