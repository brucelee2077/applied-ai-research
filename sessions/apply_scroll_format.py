#!/usr/bin/env python3
"""
apply_scroll_format.py — convert old click-stepper lessons to the new scroll-reveal
"BUILD" engine, in place. Deterministic, no network.

For each old-format lesson (has '/* STEPPER */') it:
  1. replaces the whole <style>...</style> with the canonical new CSS (from the template),
  2. replaces the <section id="s5">...</section> markup with the new build scaffold,
  3. replaces the JS STEPPER block with the new BUILD engine, seeding a baseline BUILD
     array auto-derived from the lesson's existing walkthrough step-lines (so the exact
     same content is preserved, now revealed on scroll instead of by clicking Next).

The baseline BUILD is a safe fallback; a later agent pass upgrades high-value lessons
to richer visual diagrams by replacing the BUILD array only.

Usage:
    python3 sessions/apply_scroll_format.py [--dry FILE] [--all]
"""
import re, sys, glob, json, os, html as _html

BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE, "week-01", "day-03-vmap.html")

def _slice(t, start, end):
    i = t.find(start); j = t.find(end, i+len(start))
    if i == -1 or j == -1: raise ValueError(f"markers not found: {start!r}..{end!r}")
    return t[i:j], i, j

def canonical_css(tmpl):
    a = tmpl.find("<style>"); b = tmpl.find("</style>")
    return tmpl[a:b+len("</style>")]

def s5_scaffold(tmpl):
    # the new build section, from its comment marker to its </section>
    i = tmpl.find('<!-- ===== 5. BUILD IT UP')
    j = tmpl.find('</section>', i) + len('</section>')
    return tmpl[i:j]

def build_engine_tail(tmpl):
    # everything from 'var buildWrap' up to (not incl) the '/* QUIZ */' marker
    i = tmpl.find("var buildWrap=document.getElementById('build')")
    j = tmpl.find("/* QUIZ */", i)
    return tmpl[i:j].rstrip() + "\n"

def steplines_to_build(orig):
    """Auto-derive a baseline BUILD array from the old walkthrough step-lines."""
    m = re.search(r'<div class="step-screen"[^>]*>(.*?)</div>\s*<div class="step-ctrl"', orig, re.S)
    if not m:
        return [{"viz":'<div class="dgram"><span class="node dim">walkthrough</span></div>',
                 "note":"<b>Walkthrough.</b> A richer visual build-up for this lesson is being prepared."}]
    body = m.group(1)
    spans = re.findall(r'<span class="step-line"[^>]*>(.*?)</span>\s*(?=<span class="step-line"|$)', body, re.S)
    build = []
    for raw in spans:
        code_html = raw.strip()
        if not code_html:
            continue
        # plain-text version for the note
        plain = re.sub(r'<[^>]+>', '', code_html)
        plain = _html.unescape(plain).strip()
        if not plain:
            continue
        # split trailing "# comment" as the note if present
        note = plain
        m2 = re.search(r'#\s*(.+)$', plain)
        if m2 and len(m2.group(1)) > 3:
            note = m2.group(1).strip()
        viz = ('<div style="width:100%;background:#1E1E2E;color:#CDD6F4;font-family:\'JetBrains Mono\',monospace;'
               'font-size:.82rem;line-height:1.6;padding:.6rem .85rem;border-radius:8px;white-space:pre-wrap;text-align:left">'
               + code_html + '</div>')
        build.append({"viz": viz, "note": "<b>" + _html.escape(note[:80]) + "</b>"})
    if not build:
        build = [{"viz":'<div class="dgram"><span class="node dim">walkthrough</span></div>',
                  "note":"<b>Walkthrough.</b> A richer visual build-up for this lesson is being prepared."}]
    return build

def transform(path, css, scaffold, engine_tail):
    t = open(path, encoding="utf-8").read()
    if "/* STEPPER */" not in t:
        return False, "no stepper (already new or broken)"
    # 0. derive baseline BUILD from existing step-lines BEFORE we wipe s5
    build = steplines_to_build(t)
    build_js = "/* BUILD — scroll-reveal visual build-up (auto-baseline; upgrade later) */\nvar BUILD=" + \
               json.dumps(build, ensure_ascii=False) + ";\n" + engine_tail
    # 1. replace <style>
    t = re.sub(r'<style>.*?</style>', lambda _: css, t, count=1, flags=re.S)
    # 2. replace <section id="s5">...</section>
    t = re.sub(r'<section class="sec" id="s5".*?</section>', lambda _: scaffold, t, count=1, flags=re.S)
    # 3. replace STEPPER block: from '/* STEPPER */' to the standalone final 'renderStep();'
    #    (renderStep(); also appears inside the click handlers, so anchor on the newline-led final call)
    t = re.sub(r'/\* STEPPER \*/.*?\nrenderStep\(\);', lambda _: build_js, t, count=1, flags=re.S)
    open(path, "w", encoding="utf-8").write(t)
    return True, f"{len(build)} build steps"

def main():
    tmpl = open(TEMPLATE, encoding="utf-8").read()
    css = canonical_css(tmpl); scaffold = s5_scaffold(tmpl); engine_tail = build_engine_tail(tmpl)
    args = sys.argv[1:]
    if args and args[0] == "--dry":
        path = args[1]
        ok, msg = transform(path, css, scaffold, engine_tail)
        print(f"{'OK' if ok else 'SKIP'}: {path} — {msg}")
        return
    files = sorted(glob.glob(os.path.join(BASE, "week-*", "day-*.html")))
    done = skipped = 0
    for f in files:
        ok, msg = transform(f, css, scaffold, engine_tail)
        rel = os.path.relpath(f, BASE)
        if ok: done += 1
        else: skipped += 1;
        print(f"  {'✓' if ok else '·'} {rel} — {msg}")
    print(f"\nTransformed {done}, skipped {skipped}")

if __name__ == "__main__":
    main()
