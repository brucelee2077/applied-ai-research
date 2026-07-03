#!/usr/bin/env python3
"""
wire_index.py — fill the page + questId fields in sessions/index.html's WEEKS object
from the actual lesson files on disk. Idempotent; preserves the curated week/day labels.

For each lesson id 'wNN-dMM' present in index.html, it looks for
sessions/week-NN/day-MM-*.html, reads that file's data-quest-id, and sets the entry's
3rd (page, relative to sessions/) and 4th (questId) fields.
"""
import re, glob, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE, "index.html")

def find_file_and_qid(lesson_id):
    m = re.match(r'w(\d{2})-d(\d{2})', lesson_id)
    if not m: return None, None
    wk, dy = m.group(1), m.group(2)
    hits = sorted(glob.glob(os.path.join(BASE, f"week-{wk}", f"day-{dy}-*.html")))
    if not hits: return None, None
    path = hits[0]
    t = open(path, encoding="utf-8").read()
    qm = re.search(r'data-quest-id="([^"]+)"', t)
    rel = os.path.relpath(path, BASE)
    return rel, (qm.group(1) if qm else None)

def main():
    t = open(INDEX, encoding="utf-8").read()
    ids = sorted(set(re.findall(r"\['(w\d{2}-d\d{2})'", t)))
    filled = 0; missing = 0
    for lid in ids:
        page, qid = find_file_and_qid(lid)
        if not page or not qid:
            missing += 1; continue
        # replace this entry's page,qid fields: ['id','label', <p>, <q>],
        pat = re.compile(r"(\['" + re.escape(lid) + r"'\s*,\s*'(?:[^'\\]|\\.)*'\s*,\s*)(?:null|'[^']*')(\s*,\s*)(?:null|'[^']*')(\s*\])")
        def repl(mm):
            return mm.group(1) + "'" + page + "'" + mm.group(2) + "'" + qid + "'" + mm.group(3)
        t2, n = pat.subn(repl, t)
        if n:
            t = t2; filled += 1
        else:
            print(f"  ! could not splice entry for {lid}")
    open(INDEX, "w", encoding="utf-8").write(t)
    print(f"wired {filled} entries; {missing} ids had no file yet")

if __name__ == "__main__":
    sys.exit(main())
