#!/usr/bin/env python3
"""
Shell migration: swap the old top-nav card-stack shell for the new sidebar +
Appearance-switcher (Light/Dim/Dark/Midnight) shell across sessions/ lesson.html
files, byte-preserving every lesson's own pedagogical content (hero text,
section bodies, quiz/build/playground data). The new shell was hand-built and
verified for sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html;
that file is the template this script parametrizes.

Usage:
  python3 sessions/_shell_migrate.py --check <lesson.html>         # dry-run, extraction only
  python3 sessions/_shell_migrate.py --pilot <f1> <f2> ...         # write .new.html next to each
  python3 sessions/_shell_migrate.py --apply <f1> <f2> ...         # overwrite in place
  python3 sessions/_shell_migrate.py --apply-all                   # every lesson.html except the template
"""
import re
import sys
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(REPO, "sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html")

SEC_ORDER = ["what", "intuition", "play", "why", "build", "quiz", "produce"]
BADGE_CLASS = {
    "what": "s-what", "intuition": "s-study", "play": "s-play",
    "why": "s-study", "build": "s-code", "quiz": "s-quiz", "produce": "s-produce",
}


def find_matching_bracket(text, open_pos):
    """Given text[open_pos] in '{['/, return the index just past its matching
    close bracket, correctly skipping over string literals (single/double
    quoted, backslash-escaped) so brackets inside strings don't miscount."""
    open_ch = text[open_pos]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    i = open_pos
    n = len(text)
    in_str = None  # None, or the quote char currently inside
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    raise ValueError(f"no matching {close_ch!r} found for {open_ch!r} at pos {open_pos}")


def extract_js_literal(c, var_prefix, open_ch, path):
    """Extract 'var NAME=<literal>;' robustly regardless of internal
    whitespace/quoting style (multi-line JS-literal vs single-line JSON)."""
    start = c.find(var_prefix)
    if start == -1:
        raise ValueError(f"{var_prefix!r} not found in {path}")
    open_pos = c.index(open_ch, start)
    close_pos = find_matching_bracket(c, open_pos)
    # consume the trailing ';' (skip whitespace, none expected here)
    semi = c.index(";", close_pos)
    return c[start:semi + 1]


def replace_js_literal(text, var_prefix, open_ch, new_literal):
    """Replace an existing 'var NAME=<literal>;' block in text with
    new_literal, locating the existing block via bracket-matching (so it
    works regardless of the template's own internal formatting)."""
    start = text.find(var_prefix)
    if start == -1:
        raise ValueError(f"{var_prefix!r} not found in template")
    open_pos = text.index(open_ch, start)
    close_pos = find_matching_bracket(text, open_pos)
    semi = text.index(";", close_pos)
    return text[:start] + new_literal + text[semi + 1:]


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def req(pattern, text, path, flags=re.S):
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"pattern not found in {path}: {pattern[:80]}")
    return m


def extract_old(path):
    c = read(path)
    d = {"path": path}

    d["title"] = req(r"<title>(.*?)</title>", c, path).group(1)
    d["quest_id"] = req(r'data-quest-id="([^"]*)"', c, path).group(1)
    d["brand_sub"] = req(r'<span class="nav-title">([^<]*)</span>', c, path).group(1)
    d["eyebrow"] = req(r'<span class="eyebrow">(.*?)</span>', c, path).group(1)

    h1 = req(r"<h1>(.*?)</h1>", c, path).group(1)
    if "<br>" in h1:
        main, sub = h1.split("<br>", 1)
    else:
        main, sub = h1, ""
    d["h1_main"] = main.strip()
    d["h1_sub"] = sub.strip()

    d["lead"] = req(r'<p class="lead">(.*?)</p>', c, path).group(1)

    goal_raw = req(r'<div class="goal">\s*(.*?)\s*</div>', c, path).group(1)
    goal_inner = re.sub(r"(<b>)🎯\s*", r"\1", goal_raw, count=1)
    d["goal_inner"] = goal_inner

    secs = []
    for i, key in enumerate(SEC_ORDER, start=1):
        sid = f"s{i}"
        head_pat = (
            r'<section class="sec" id="' + sid + r'" data-sec="' + key + r'">\s*'
            r'<div class="sec-head"><span class="sec-badge (s-\w+)">\s*\d+\s*·\s*([^<]*)</span>'
            r'<span class="sec-h">(.*?)</span>'
        )
        hm = req(head_pat, c, path)
        badge_class, label_raw, heading = hm.groups()

        body_start = c.index('<div class="sec-body">', hm.end())
        body_pat = re.compile(r'<div class="sec-body">(.*?)</div>\s*</section>', re.S)
        bm = body_pat.search(c, body_start)
        if not bm:
            raise ValueError(f"sec-body close not found for {sid}/{key} in {path}")
        body = bm.group(1)

        label = label_raw.strip().lower().capitalize()
        if len(body.strip()) < 30:
            raise ValueError(f"suspiciously short body ({len(body)} chars) for {sid}/{key} in {path}")
        secs.append({"id": sid, "key": key, "badge_class": badge_class, "label": label,
                      "heading": heading.strip(), "body": body})
    d["sections"] = secs

    # prev / next / hub — handle the disabled-span edge case (start of curriculum)
    pm = re.search(r'<a class="lnav prev" href="([^"]*)"><span class="lnav-dir">([^<]*)</span><span class="lnav-t">([^<]*)</span></a>', c)
    if pm:
        d["prev"] = {"disabled": False, "href": pm.group(1), "dir": pm.group(2), "text": pm.group(3)}
    else:
        pm2 = req(r'<span class="lnav prev disabled"><span class="lnav-dir">([^<]*)</span><span class="lnav-t">([^<]*)</span></span>', c, path)
        d["prev"] = {"disabled": True, "href": "", "dir": pm2.group(1), "text": pm2.group(2)}

    nm = re.search(r'<a class="lnav next" href="([^"]*)"><span class="lnav-dir">([^<]*)</span><span class="lnav-t">([^<]*)</span></a>', c)
    if nm:
        d["next"] = {"disabled": False, "href": nm.group(1), "dir": nm.group(2), "text": nm.group(3)}
    else:
        nm2 = req(r'<span class="lnav next disabled"><span class="lnav-dir">([^<]*)</span><span class="lnav-t">([^<]*)</span></span>', c, path)
        d["next"] = {"disabled": True, "href": "", "dir": nm2.group(1), "text": nm2.group(2)}

    hm2 = req(r'<a class="lnav-hub" href="([^"]*)"', c, path)
    d["hub_href"] = hm2.group(1)

    d["demos_literal"] = extract_js_literal(c, "var DEMOS", "{", path)
    d["build_literal"] = extract_js_literal(c, "var BUILD", "[", path)
    d["qs_literal"] = extract_js_literal(c, "var QS", "[", path)

    return d


def nav_link_html(entry, cls_extra=""):
    label = f'{entry["dir"]}<span class="t">{entry["text"]}</span>'
    if entry["disabled"]:
        return f'<span class="lnav {cls_extra} disabled"><span class="d">{entry["dir"]}</span><span class="t">{entry["text"]}</span></span>'
    return f'<a class="lnav {cls_extra}" href="{entry["href"]}"><span class="d">{entry["dir"]}</span><span class="t">{entry["text"]}</span></a>'


def sub1(pattern, replacement, text, flags=0):
    """re.sub with a literal replacement — never treats backslashes in
    `replacement` as backreferences (raw-string re.sub replacements do)."""
    return re.sub(pattern, lambda m: replacement, text, count=1, flags=flags)


def render(d, template):
    out = template

    out = sub1(r"<title>.*?</title>", f'<title>{d["title"]}</title>', out, flags=re.S)
    out = sub1(r'data-quest-id="[^"]*"', f'data-quest-id="{d["quest_id"]}"', out)
    out = sub1(r'<div class="brand-sub">[^<]*</div>', f'<div class="brand-sub">{d["brand_sub"]}</div>', out)
    out = sub1(r'<span class="eyebrow">.*?</span>', f'<span class="eyebrow">{d["eyebrow"]}</span>', out, flags=re.S)
    out = sub1(
        r"<h1>.*?</h1>",
        f'<h1>{d["h1_main"]}<span class="sub">{d["h1_sub"]}</span></h1>',
        out, flags=re.S,
    )
    out = sub1(r'<p class="lede">.*?</p>', f'<p class="lede">{d["lead"]}</p>', out, flags=re.S)
    out = sub1(
        r'<div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>.*?</div></div>',
        f'<div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>{d["goal_inner"]}</div></div>',
        out, flags=re.S,
    )

    # sidebar section-nav button labels ("1 · What is it" etc.)
    for s in d["sections"]:
        out = re.sub(
            r'(<button class="nav-link" data-target="' + s["id"] + r'"><span class="nl-dot"></span>)[^<]*(</button>)',
            lambda m, s=s: m.group(1) + f'{s["id"][1:]} · {s["label"]}' + m.group(2),
            out, count=1,
        )

    # each section: head (badge class/number/label/heading) + body (verbatim)
    for i, s in enumerate(d["sections"], start=1):
        sec_pat = re.compile(
            r'<section class="module-section" id="' + s["id"] + r'" data-sec="[a-z]+">.*?</section>',
            re.S,
        )
        m = sec_pat.search(out)
        if not m:
            raise ValueError(f'template section {s["id"]} not found while rendering {d["path"]}')
        new_sec = (
            f'<section class="module-section" id="{s["id"]}" data-sec="{s["key"]}">\n'
            f'  <div class="sec-head"><span class="sec-num {s["badge_class"]}">{i}</span>'
            f'<span class="sec-h">{s["heading"]}</span><span class="sec-tag">{s["label"]}</span></div>\n'
            f'  <div class="sec-body">{s["body"]}</div>\n'
            f'</section>'
        )
        out = out[: m.start()] + new_sec + out[m.end():]

    # sidebar footer: prev / next / hub
    out = sub1(
        r'<a class="lnav prev" href="[^"]*"><span class="d">[^<]*</span><span class="t">[^<]*</span></a>',
        nav_link_html(d["prev"], "prev"), out,
    )
    out = sub1(
        r'<a class="lnav next" href="[^"]*"><span class="d">[^<]*</span><span class="t">[^<]*</span></a>',
        nav_link_html(d["next"], "next"), out,
    )
    out = sub1(r'<a class="lnav-hub" href="[^"]*">', f'<a class="lnav-hub" href="{d["hub_href"]}">', out)

    out = replace_js_literal(out, "var DEMOS", "{", d["demos_literal"])
    out = replace_js_literal(out, "var BUILD", "[", d["build_literal"])
    out = replace_js_literal(out, "var QS", "[", d["qs_literal"])

    return out


def migrate_one(path, template, out_path=None):
    d = extract_old(path)
    new_content = render(d, template)
    # sanity: section count preserved, quiz/build counts preserved
    assert new_content.count('class="module-section"') == 7, f"section count drift in {path}"
    old_c = read(path)
    old_q = len(re.findall(r"ans:\d+", old_c))
    new_q = len(re.findall(r"ans:\d+", new_content))
    assert old_q == new_q, f"quiz count drift in {path}: {old_q} -> {new_q}"
    target = out_path or path
    with open(target, "w", encoding="utf-8") as f:
        f.write(new_content)
    return d


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    template = read(TEMPLATE_PATH)
    mode = args[0]

    if mode == "--check":
        for p in args[1:]:
            d = extract_old(p)
            print(f"OK  {p}  title={d['title']!r}  sections={len(d['sections'])}  "
                  f"prev_disabled={d['prev']['disabled']}  next_disabled={d['next']['disabled']}")
        return

    if mode == "--pilot":
        for p in args[1:]:
            outp = p.replace(".html", ".new.html")
            migrate_one(p, template, out_path=outp)
            print(f"PILOT wrote {outp}")
        return

    if mode == "--apply":
        for p in args[1:]:
            migrate_one(p, template)
            print(f"APPLIED {p}")
        return

    if mode == "--apply-all":
        files = sorted(glob.glob(os.path.join(REPO, "sessions/**/lesson.html"), recursive=True))
        files = [f for f in files if os.path.abspath(f) != os.path.abspath(TEMPLATE_PATH)]
        ok, fail = 0, []
        for p in files:
            try:
                migrate_one(p, template)
                ok += 1
            except Exception as e:
                fail.append((p, str(e)))
        print(f"applied: {ok}/{len(files)}")
        if fail:
            print("FAILURES:")
            for p, e in fail:
                print(f"  {p}: {e}")
        return

    print(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
