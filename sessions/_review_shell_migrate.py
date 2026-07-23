#!/usr/bin/env python3
"""
Review-gate shell migration: swap the old fixed-top-nav card-stack shell of
sessions/**/review.html for the SAME sidebar + Appearance-switcher shell every
lesson.html already uses (nav-links + auto-built checklist + theme + prev/next/map
in the sidebar). Byte-preserves every review's own content: hero text, the
CHECKS self-check list, the scored QS quiz, the verdict prose, the pass banner,
and the real prev/next links. Only the chrome and the section wrappers change.

The CSS is read live from the canonical lesson template so it is guaranteed
identical to the lessons; a tiny review-only block (self-check list, quiz score,
banner CTA) is appended.

Usage:
  python3 sessions/_review_shell_migrate.py --check <review.html>       # dry-run, extraction report
  python3 sessions/_review_shell_migrate.py --pilot <f1> ...            # write .new.html next to each
  python3 sessions/_review_shell_migrate.py --apply <f1> ...            # overwrite in place
  python3 sessions/_review_shell_migrate.py --apply-all                 # every sessions/**/review.html
"""
import re
import sys
import glob
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shell_migrate as sm  # reuse read()/extract_js_literal()/find_matching_bracket()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_LESSON = os.path.join(REPO, "sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html")

# stable head bits, identical across every lesson/review (default appearance = dim, matching lessons)
PREPAINT = ("<script>/* set appearance before paint (no flash) */"
            "(function(){try{var t=localStorage.getItem('frontier-theme');"
            "if(['light','dim','dark','midnight'].indexOf(t)<0)t='dim';"
            "document.documentElement.setAttribute('data-theme',t);}"
            "catch(e){document.documentElement.setAttribute('data-theme','dim');}})();</script>")
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800'
         '&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700'
         '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">')

# review-only additions layered on top of the shared lesson CSS (uses lesson vars)
REVIEW_ADD = r"""
/* ---------- review-gate additions (self-check list, quiz score, banner CTA) ---------- */
.check{display:flex;flex-direction:column;gap:.6rem;margin:1.1rem 0}
.check-item{display:flex;align-items:flex-start;gap:.7rem;padding:.7rem .9rem;border:1.5px solid var(--line2);border-radius:var(--r-md);background:var(--panel2);cursor:pointer;transition:all .15s}
.check-item:hover{border-color:var(--accent)}
.check-item.on{border-color:var(--ok);background:var(--ok-soft)}
.check-box{width:20px;height:20px;border-radius:6px;border:2px solid var(--line2);flex-shrink:0;margin-top:1px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:.8rem}
.check-item.on .check-box{background:var(--ok);border-color:var(--ok)}
.check-item .ct{font-size:.92rem;color:var(--ink2)}.check-item .ct b{color:var(--ink)}
.check-lesson{font-family:var(--mono);font-size:.68rem;color:var(--muted)}
.q-score{font-family:var(--mono);font-size:.8rem;color:var(--muted);margin-top:.4rem}
.fin .cta{display:inline-block;margin-top:1rem;font-family:var(--body);font-weight:700;color:#fff;background:var(--ok);border-radius:var(--r-md);padding:.6rem 1.2rem;text-decoration:none;border:none}
.fin .cta:hover{text-decoration:none;filter:brightness(1.05)}
"""

# section key -> (sec-num color class, number, tag label)
SEC_MAP = {
    "check":   ("s-study",   "1", "Self-check"),
    "quiz":    ("s-quiz",    "2", "Mixed quiz"),
    "verdict": ("s-produce", "3", "Verdict"),
}

JS = r"""<script>
(function(){
"use strict";
var QID = document.body.getAttribute('data-quest-id') || 'quest';
var KEY = 'frontier-lesson:'+QID;
var state = load();
function load(){ try{ return JSON.parse(localStorage.getItem(KEY))||{done:{}} }catch(e){ return {done:{}} } }
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)) }catch(e){} }

var secs = Array.prototype.slice.call(document.querySelectorAll('.module-section'));
var bar = document.getElementById('progress-bar');
var count = document.getElementById('count'), fin = document.getElementById('fin');
var RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var navLinks = Array.prototype.slice.call(document.querySelectorAll('.nav-link'));

/* build the sidebar checklist from sections (same as lessons) */
var checklist = document.getElementById('checklist'), checkItems = {};
function shorten(t){ return t.length>34 ? t.slice(0,32).trim()+'…' : t; }
secs.forEach(function(sec){
  var key = sec.getAttribute('data-sec');
  var title = sec.querySelector('.sec-h') ? sec.querySelector('.sec-h').textContent : key;
  var num = sec.querySelector('.sec-num') ? sec.querySelector('.sec-num').textContent : '';
  var li = document.createElement('li');
  li.innerHTML = '<span class="box" aria-hidden="true"></span><span>'+num+' · '+shorten(title)+'</span>';
  li.addEventListener('click', function(){ sec.scrollIntoView({behavior:RM?'auto':'smooth', block:'start'}); });
  checklist.appendChild(li);
  checkItems[key] = li;
});

function markDone(sec, scrollNext){
  sec.classList.add('done'); state.done[sec.getAttribute('data-sec')]=true; save(); refresh();
  if(scrollNext){
    var i = secs.indexOf(sec);
    if(i>-1 && i+1<secs.length) setTimeout(function(){ secs[i+1].scrollIntoView({behavior:RM?'auto':'smooth',block:'start'}); },180);
    else setTimeout(function(){ fin.scrollIntoView({behavior:RM?'auto':'smooth',block:'center'}); },180);
  }
}
secs.forEach(function(sec){
  if(state.done[sec.getAttribute('data-sec')]) sec.classList.add('done');
  var btn = sec.querySelector('.gotit');
  if(btn) btn.addEventListener('click', function(){ if(btn.disabled) return; markDone(sec, true); });
});
function refresh(){
  var total = secs.length, done = secs.filter(function(s){return s.classList.contains('done')}).length;
  var pct = Math.round(done/total*100);
  if(bar){ bar.style.width = pct+'%'; bar.setAttribute('aria-valuenow', String(pct)); bar.setAttribute('aria-valuetext', pct+'% complete'); }
  if(count) count.innerHTML = '<b>'+done+'</b>/'+total+' sections done';
  secs.forEach(function(s){
    var k = s.getAttribute('data-sec'), on = s.classList.contains('done');
    if(checkItems[k]) checkItems[k].classList.toggle('done', on);
    var nl = navLinks.filter(function(l){return l.getAttribute('data-target')===s.id;})[0];
    if(nl) nl.classList.toggle('done', on);
  });
  var all = done===total; fin.classList.toggle('show', all); fin.setAttribute('aria-hidden', all?'false':'true');
}
var resetBtn = document.getElementById('reset');
if(resetBtn) resetBtn.addEventListener('click', function(){
  if(!confirm('Reset review progress?')) return;
  state={done:{}}; save(); secs.forEach(function(s){s.classList.remove('done')});
  document.querySelectorAll('.check-item.on').forEach(function(c){c.classList.remove('on'); var b=c.querySelector('.check-box'); if(b) b.textContent='';});
  refresh();
});

/* appearance switcher (writes the shared frontier-theme key) */
var themeBtns = Array.prototype.slice.call(document.querySelectorAll('.theme-btn'));
function setTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  try{ localStorage.setItem('frontier-theme', t); }catch(e){}
  themeBtns.forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-theme-set')===t); });
}
themeBtns.forEach(function(b){ b.addEventListener('click', function(){ setTheme(b.getAttribute('data-theme-set')); }); });
setTheme(document.documentElement.getAttribute('data-theme') || 'dim');

/* mobile sidebar toggle */
var sidebar = document.getElementById('sidebar'), toggle = document.getElementById('nav-toggle'), scrim = document.getElementById('scrim');
function openSidebar(){ sidebar.classList.add('open'); scrim.classList.add('show'); toggle.setAttribute('aria-expanded','true'); }
function closeSidebar(){ sidebar.classList.remove('open'); scrim.classList.remove('show'); toggle.setAttribute('aria-expanded','false'); }
if(toggle) toggle.addEventListener('click', function(){ sidebar.classList.contains('open')?closeSidebar():openSidebar(); });
if(scrim) scrim.addEventListener('click', closeSidebar);

/* sidebar nav: click to scroll, scrollspy to highlight active */
navLinks.forEach(function(l){
  l.addEventListener('click', function(){
    var t = document.getElementById(l.getAttribute('data-target'));
    if(t) t.scrollIntoView({behavior:RM?'auto':'smooth', block:'start'});
    closeSidebar();
  });
});
var spyTargets = [document.getElementById('home')].concat(secs).filter(Boolean);
if('IntersectionObserver' in window){
  var spy = new IntersectionObserver(function(es){
    es.forEach(function(e){
      if(e.isIntersecting){
        var id = e.target.id;
        navLinks.forEach(function(l){ l.classList.toggle('active', l.getAttribute('data-target')===id); });
      }
    });
  }, {threshold:0.01, rootMargin:'-45% 0px -50% 0px'});
  spyTargets.forEach(function(t){ spy.observe(t); });
}

/* SELF-CHECK */
@@CHECKS@@
var checkWrap=document.getElementById('check'), checkSec=document.getElementById('s1'), ticked=0;
if(checkWrap) CHECKS.forEach(function(c){
  var el=document.createElement('div'); el.className='check-item';
  el.innerHTML='<span class="check-box"></span><span class="ct"><span class="check-lesson">'+c[0]+'</span><br>'+c[1]+'</span>';
  el.addEventListener('click', function(){
    el.classList.toggle('on');
    el.querySelector('.check-box').textContent = el.classList.contains('on') ? '✓' : '';
    ticked = checkWrap.querySelectorAll('.check-item.on').length;
    var g=checkSec.querySelector('.gotit');
    if(ticked>=CHECKS.length){ g.disabled=false; g.textContent='All ticked — I\'m ready ✓'; }
    else { g.disabled=true; g.textContent='tick all to continue'; }
  });
  checkWrap.appendChild(el);
});

/* QUIZ (scored) */
@@QS@@
var quizWrap=document.getElementById('quiz'), quizSec=document.getElementById('s2'), scoreEl=document.getElementById('score'), answered={}, correct=0;
if(quizWrap) QS.forEach(function(item,qi){
  var block=document.createElement('div'); block.className='q';
  var ask=document.createElement('div'); ask.className='q-ask'; ask.innerHTML=item.q; block.appendChild(ask);
  var opts=document.createElement('div'); opts.className='q-opts';
  var fb=document.createElement('div'); fb.className='q-fb';
  item.opts.forEach(function(text,oi){
    var o=document.createElement('button'); o.type='button'; o.className='q-opt';
    o.innerHTML='<span class="mark"></span><span>'+text+'</span>';
    o.addEventListener('click', function(){
      if(answered[qi]) return; answered[qi]=true;
      Array.prototype.slice.call(opts.children).forEach(function(c){c.classList.add('locked')});
      if(oi===item.ans){ o.classList.add('correct'); fb.className='q-fb good show'; fb.innerHTML='✓ '+item.fb; correct++; }
      else { o.classList.add('wrong'); opts.children[item.ans].classList.add('correct'); fb.className='q-fb bad show'; fb.innerHTML='Correct answer is green. '+item.fb; }
      if(scoreEl) scoreEl.textContent = 'Score: '+correct+'/'+Object.keys(answered).length+' answered';
      if(Object.keys(answered).length>=QS.length){ var g=quizSec.querySelector('.gotit'); g.disabled=false; g.textContent='Done — score '+correct+'/'+QS.length+' ✓'; }
    });
    opts.appendChild(o);
  });
  block.appendChild(opts); block.appendChild(fb); quizWrap.appendChild(block);
});

refresh();
})();
</script>"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
@@PREPAINT@@
<title>@@TITLE@@</title>
@@FONTS@@
<style>@@STYLE@@</style>
</head>

<body data-quest-id="@@QID@@">

<div id="progress-track" aria-hidden="true"><div id="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" aria-label="review progress"></div></div>
<button id="nav-toggle" type="button" aria-label="Toggle sections" aria-expanded="false">☰ Sections</button>
<div id="scrim" aria-hidden="true"></div>

<div class="layout">

  <aside id="sidebar" aria-label="review navigation">
    <div class="sidebar-scroll">
    <div class="brand">
      <div class="brand-title">Frontier<span class="brand-dot">.</span>Lab</div>
      <div class="brand-sub">@@BRAND_SUB@@</div>
      <div class="nav-count" id="count"><b>0</b>/3 sections done</div>
    </div>

    <div class="theme-row" role="group" aria-label="appearance mode">
      <span class="nav-group-label" style="padding:0">Appearance</span>
      <div class="theme-btns">
        <button class="theme-btn" data-theme-set="light" type="button">Light</button>
        <button class="theme-btn" data-theme-set="dim" type="button">Dim</button>
        <button class="theme-btn" data-theme-set="dark" type="button">Dark</button>
        <button class="theme-btn" data-theme-set="midnight" type="button">Midnight</button>
      </div>
    </div>

    <nav aria-label="Sections">
      <div class="nav-group-label">Review gate</div>
      <button class="nav-link" data-target="home"><span class="nl-dot"></span>Start here</button>
      <button class="nav-link" data-target="s1"><span class="nl-dot"></span>1 · Self-check</button>
      <button class="nav-link" data-target="s2"><span class="nl-dot"></span>2 · Mixed quiz</button>
      <button class="nav-link" data-target="s3"><span class="nl-dot"></span>3 · Verdict</button>
    </nav>

    <div class="checklist-wrap">
      <div class="nav-group-label">Progress checklist</div>
      <ul id="checklist"></ul>
      <button id="reset" class="ghost-btn" type="button">↺ Reset progress</button>
    </div>
    </div><!-- /.sidebar-scroll -->

    <div class="side-nav">
      <a class="lnav prev" href="@@PREV_HREF@@"><span class="d">← Back</span><span class="t">@@PREV_LABEL@@</span></a>
      <a class="lnav next" href="@@NEXT_HREF@@"><span class="d">Next →</span><span class="t">@@NEXT_LABEL@@</span></a>
      <a class="lnav-hub" href="../index.html"><span class="d">▦ Map</span><span class="t">Back to curriculum</span></a>
    </div>
  </aside>

  <main id="content">

    <section id="home" class="hero">
      <span class="kicker">@@KICKER@@</span>
      <h1>@@H1@@</h1>
      <p class="lede">@@LEAD@@</p>
      <div class="goal">@@GOAL@@</div>
    </section>

@@SECTIONS@@

    <div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">@@FIN_EM@@</span>
      <h3>@@FIN_H3@@</h3>
      <p>@@FIN_P@@</p>
      <a class="cta" href="../index.html">Back to the map →</a>
    </div>

    <footer class="site-footer">Frontier Lab · Capability Spiral — review gate. Progress is saved locally in your browser (localStorage), works offline, just double-click to open.</footer>

  </main>
</div>

@@JS@@
</body>
</html>
"""


def _grp(pattern, text, path, flags=re.S, n=1):
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"pattern not found in {path}: {pattern[:70]}")
    return m.group(n)


def extract(path):
    c = sm.read(path)
    d = {}
    d["title"] = _grp(r"<title>(.*?)</title>", c, path)
    d["qid"] = _grp(r'data-quest-id="(.*?)"', c, path)
    d["nav_title"] = _grp(r'<span class="nav-title">(.*?)</span>', c, path)
    # hero
    d["kicker"] = _grp(r'<span class="eyebrow">(.*?)</span>', c, path)
    d["h1"] = _grp(r"<h1>(.*?)</h1>", c, path)
    d["lead"] = _grp(r'<p class="lead">(.*?)</p>', c, path)
    d["goal"] = _grp(r'<div class="goal">(.*?)</div>\s*</header>', c, path).strip()
    # sections (in document order)
    sec_pat = re.compile(
        r'<section class="sec" id="(s\d)" data-sec="(\w+)">\s*'
        r'<div class="sec-head">.*?<span class="sec-h">(.*?)</span>.*?</div>\s*'
        r'<div class="sec-body">(.*?)</div>\s*</section>', re.S)
    secs = sec_pat.findall(c)
    if len(secs) != 3:
        raise ValueError(f"{path}: expected 3 sections, found {len(secs)}")
    d["sections"] = [{"id": sid, "key": key, "sec_h": sec_h, "body": body}
                     for sid, key, sec_h, body in secs]
    # top-nav prev/next (first occurrence; top and bottom are identical)
    d["prev_href"] = _grp(r'<a class="lnav prev" href="(.*?)">', c, path)
    d["prev_label"] = _grp(r'<a class="lnav prev"[^>]*>.*?<span class="lnav-t">(.*?)</span>', c, path)
    d["next_href"] = _grp(r'<a class="lnav next" href="(.*?)">', c, path)
    d["next_label"] = _grp(r'<a class="lnav next"[^>]*>.*?<span class="lnav-t">(.*?)</span>', c, path)
    # finale ( span.em may carry attributes, e.g. aria-hidden — stay tolerant so re-runs don't false-fail )
    fin = re.search(r'<div class="fin" id="fin"[^>]*>\s*<span class="em"[^>]*>(.*?)</span>\s*<h3>(.*?)</h3>\s*<p>(.*?)</p>', c, re.S)
    if not fin:
        raise ValueError(f"{path}: fin block not found")
    d["fin_em"], d["fin_h3"], d["fin_p"] = fin.group(1), fin.group(2), fin.group(3)
    # data literals (verbatim, opaque)
    d["checks_js"] = sm.extract_js_literal(c, "var CHECKS", "[", path)
    d["qs_js"] = sm.extract_js_literal(c, "var QS", "[", path)
    return d


def _load_style():
    tpl = sm.read(TEMPLATE_LESSON)
    style = _grp(r"<style>(.*?)</style>", tpl, TEMPLATE_LESSON)
    return style + REVIEW_ADD


def render(d):
    style = _load_style()
    sections = []
    for s in d["sections"]:
        numcls, num, tag = SEC_MAP.get(s["key"], ("s-study", "?", s["key"].capitalize()))
        sections.append(
            f'    <section class="module-section" id="{s["id"]}" data-sec="{s["key"]}">\n'
            f'  <div class="sec-head"><span class="sec-num {numcls}">{num}</span>'
            f'<span class="sec-h">{s["sec_h"]}</span><span class="sec-tag">{tag}</span></div>\n'
            f'  <div class="sec-body">{s["body"]}</div>\n'
            f'</section>')
    sections_block = "\n\n".join(sections)
    js = JS.replace("@@CHECKS@@", d["checks_js"]).replace("@@QS@@", d["qs_js"])
    out = PAGE
    repl = {
        "@@PREPAINT@@": PREPAINT, "@@FONTS@@": FONTS, "@@STYLE@@": style,
        "@@TITLE@@": d["title"], "@@QID@@": d["qid"], "@@BRAND_SUB@@": d["nav_title"],
        "@@PREV_HREF@@": d["prev_href"], "@@PREV_LABEL@@": d["prev_label"],
        "@@NEXT_HREF@@": d["next_href"], "@@NEXT_LABEL@@": d["next_label"],
        "@@KICKER@@": d["kicker"], "@@H1@@": d["h1"], "@@LEAD@@": d["lead"], "@@GOAL@@": d["goal"],
        "@@SECTIONS@@": sections_block,
        "@@FIN_EM@@": d["fin_em"], "@@FIN_H3@@": d["fin_h3"], "@@FIN_P@@": d["fin_p"],
        "@@JS@@": js,
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    if "@@" in out:
        leftover = set(re.findall(r"@@\w+@@", out))
        raise ValueError(f"unreplaced tokens: {leftover}")
    return out


def migrate_one(path, out_path=None):
    d = extract(path)
    html = render(d)
    # structural guards on the produced file
    assert [s["key"] for s in d["sections"]] == ["check", "quiz", "verdict"], \
        f"section keys/order drift in {path}: {[s['key'] for s in d['sections']]}"
    assert html.count('class="module-section"') == 3, f"section count drift in {path}"
    assert 'id="sidebar"' in html, f"sidebar missing in {path}"
    assert 'data-sec="verdict"' in html, f"verdict section missing in {path} (hub completion detection)"
    assert d["qid"] in html, f"quest-id lost in {path}"
    assert 'class="nav"' not in html and 'lesson-nav' not in html, f"old chrome leaked in {path}"
    target = out_path or path
    with open(target, "w", encoding="utf-8") as f:
        f.write(html)
    return target


def all_reviews():
    return sorted(glob.glob(os.path.join(REPO, "sessions", "**", "review.html"), recursive=True))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    mode = args[0]
    if mode == "--check":
        for p in args[1:]:
            d = extract(p)
            print(f"OK extract {p}")
            print(f"   title={d['title']!r}")
            print(f"   qid={d['qid']}  nav_title={d['nav_title']!r}")
            print(f"   prev={d['prev_href']!r} ({d['prev_label']!r})")
            print(f"   next={d['next_href']!r} ({d['next_label']!r})")
            print(f"   sections={[s['key'] for s in d['sections']]}")
            print(f"   checks={len(d['checks_js'])}B  qs={len(d['qs_js'])}B")
    elif mode == "--pilot":
        for p in args[1:]:
            out = p.replace(".html", ".new.html")
            migrate_one(p, out)
            print(f"wrote {out}")
    elif mode == "--apply":
        for p in args[1:]:
            migrate_one(p)
            print(f"migrated {p}")
    elif mode == "--apply-all":
        files = all_reviews()
        for p in files:
            migrate_one(p)
            print(f"migrated {p}")
        print(f"\n{len(files)} review gates migrated.")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
