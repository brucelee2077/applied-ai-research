#!/usr/bin/env python3
"""nav_audit.py — verify every Frontier-Lab session page is wired together.

The hub `index.html` MODULES array is the single source of truth for module
ORDER. Every built lesson/review page is expected to form ONE continuous chain
in that order: each page's "← Prev" points to the previous page, "Next →" to the
next, and the hub "Map" button to index.html. The first page keeps a disabled
"Start" placeholder; the last page's Next returns to the hub.

Checks, over all pages under sessions/:
  1. CHAIN     — actual prev/next/hub hrefs resolve to the correct neighbor
  2. BROKEN    — every href/src (attrs + JS ".html" literals) resolves to a file
  3. CASE      — resolves case-sensitively (macOS is lenient, GitHub Pages is not)
  4. ORPHANS   — built pages nothing links to

Usage:  python3 sessions/nav_audit.py            # from repo root
        python3 nav_audit.py                     # from sessions/
Exit code 0 iff CHAIN, BROKEN and CASE are all clean (orphans are informational).
"""
import re, os, sys, glob
from urllib.parse import urldefrag, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = HERE if os.path.basename(HERE) == "sessions" else os.path.join(HERE, "sessions")
INDEX = os.path.join(SESS, "index.html")
rel = lambda p: os.path.relpath(p, SESS)

# ---------- build canonical order from index.html ----------
txt = open(INDEX, encoding="utf-8").read()
mod_re = re.compile(r"\{n:\s*('?)([^,']+)\1\s*,.*?lessons:\s*\[(.*?)\]\s*\}", re.S)
les = re.compile(r"\[\s*'((?:[^'\\]|\\.)*)'\s*,\s*'((?:[^'\\]|\\.)*)'\s*,\s*(null|'([^']*)')")
mod_dirs = []
for mm in mod_re.finditer(txt):
    pages = [lm.group(4) for lm in les.finditer(mm.group(3)) if lm.group(3) != "null"]
    if pages:
        mod_dirs.append(pages[0].split("/")[0])
def daynum(p):
    m = re.search(r"day-(\d+)", p); return int(m.group(1)) if m else 0
canon = []
for d in mod_dirs:
    ad = os.path.join(SESS, d)
    days = sorted(glob.glob(ad + "/day-*/lesson.html"), key=daynum) or sorted(glob.glob(ad + "/day-*.html"), key=daynum)
    seq = list(days)
    rp = os.path.join(ad, "review-part-a.html")
    if os.path.exists(rp):
        pos = next((i for i, s in enumerate(seq) if daynum(s) == 4), len(seq)); seq.insert(pos, rp)
    rv = os.path.join(ad, "review.html")
    if os.path.exists(rv):
        seq.append(rv)
    canon += seq

def rez(fp, h): return os.path.normpath(os.path.join(os.path.dirname(fp), h.split("#")[0].split("?")[0]))
def prevs(fp): return set(re.findall(r'<a class="lnav prev"[^>]*href="([^"]+)"', open(fp, encoding="utf-8").read()))
def nexts(fp): return set(re.findall(r'<a class="lnav next"[^>]*href="([^"]+)"', open(fp, encoding="utf-8").read()))
def hubs(fp):  return set(re.findall(r'<a class="lnav-hub" href="([^"]+)"', open(fp, encoding="utf-8").read()))

chain = []
for i, fp in enumerate(canon):
    exp_prev = INDEX if i == 0 else canon[i - 1]
    exp_next = INDEX if i == len(canon) - 1 else canon[i + 1]
    ph = prevs(fp)
    if i == 0:
        if ph and all(rez(fp, h) != INDEX for h in ph):
            chain.append(f"PREV {rel(fp)} (first page should keep Start/hub)")
    else:
        if not ph: chain.append(f"NO-PREV {rel(fp)}")
        elif not all(rez(fp, h) == exp_prev for h in ph):
            chain.append(f"PREV {rel(fp)} -> {[rel(rez(fp,h)) for h in ph]} (exp {rel(exp_prev)})")
    nh = nexts(fp)
    if not nh: chain.append(f"NO-NEXT {rel(fp)}")
    elif not all(rez(fp, h) == exp_next for h in nh):
        chain.append(f"NEXT {rel(fp)} -> {[rel(rez(fp,h)) for h in nh]} (exp {rel(exp_next)})")
    if not all(rez(fp, h) == INDEX for h in hubs(fp)):
        chain.append(f"HUB {rel(fp)}")

# ---------- broken + case + orphans over all html ----------
attr_re = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.I)
js_re = re.compile(r'["\']([^"\']*?\.html)["\']', re.I)
ext = lambda l: re.match(r'^(https?:|mailto:|data:|javascript:|tel:|#|//)', l, re.I)
def case_ok(abspath):
    if not os.path.exists(abspath): return True  # non-existence handled by BROKEN
    cur = os.sep
    for part in abspath.split(os.sep)[1:]:
        if not part: continue
        try: entries = os.listdir(cur)
        except OSError: return False
        if part not in entries: return False
        cur = os.path.join(cur, part)
    return True

html_files = [p for p in glob.glob(SESS + "/**/*.html", recursive=True) if "__pycache__" not in p]
broken, case_bad, linked = [], [], set()
for src in html_files:
    t = open(src, encoding="utf-8", errors="replace").read()
    for raw in set(attr_re.findall(t)) | set(js_re.findall(t)):
        link = raw.strip()
        if not link or ext(link): continue
        pp = unquote(urldefrag(link)[0].split("?")[0])
        if not pp: continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(src), pp))
        if not resolved.startswith(os.path.dirname(SESS)): continue
        if not os.path.exists(resolved):
            broken.append(f"{rel(src)} -> {link}")
        else:
            if resolved.endswith(".html"): linked.add(os.path.abspath(resolved))
            if not case_ok(resolved): case_bad.append(f"{rel(src)} -> {link}")
orphans = sorted(rel(p) for p in ({os.path.abspath(x) for x in html_files} - linked))

def section(title, items):
    print(f"\n### {title}: {len(items)}")
    for x in items[:60]: print("   ", x)

print("=" * 64); print(f"nav_audit — {len(html_files)} pages, chain of {len(canon)}"); print("=" * 64)
section("CHAIN problems", chain)
section("BROKEN links", broken)
section("CASE mismatches (GitHub Pages)", case_bad)
section("ORPHANS (informational)", orphans)
ok = not (chain or broken or case_bad)
print("\n" + ("PASS — all pages wired together." if ok else "FAIL — see problems above."))
sys.exit(0 if ok else 1)
