#!/usr/bin/env python3
"""Render ROADMAP.md -> sessions/roadmap.html in the Capability Spiral shell.

Self-contained page (no CDN/runtime deps): same 4-theme palette + Appearance
switcher + no-flash script as the hub, dark by default, shared 'frontier-theme'
localStorage key. Re-run this whenever ROADMAP.md changes:

    python3 sessions/_build_roadmap.py
"""
import os, re, sys, html
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "ROADMAP.md")
OUT  = os.path.join(HERE, "roadmap.html")


def gh_slugify(value, separator="-"):
    """GitHub-flavored heading slug so the doc's own #anchor links keep working."""
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)  # drop punctuation (— & ( ) . etc.)
    return value.replace(" ", separator)


def build():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()

    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        extension_configs={"toc": {"slugify": gh_slugify, "permalink": False}},
    )
    body = md.convert(md_text)

    # Sidebar TOC from the rendered <h2 id=...> headings (ids already match anchors).
    toc = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.DOTALL)
    toc_links = "\n".join(
        '        <a class="nav-link" href="#{sid}"><span class="nl-dot"></span>{txt}</a>'.format(
            sid=sid, txt=re.sub(r"<[^>]+>", "", t).strip()
        )
        for sid, t in toc
    )

    page = TEMPLATE.replace("{{TOC}}", toc_links).replace("{{BODY}}", body)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote {} ({} bytes, {} h2 sections)".format(OUT, len(page), len(toc)))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>/* set appearance before paint (no flash) — defaults to dark, shares the hub/lesson key */(function(){try{var t=localStorage.getItem('frontier-theme');if(['light','dim','dark','midnight'].indexOf(t)<0)t='dark';document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();</script>
<title>Roadmap · The Capability Spiral</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --accent:#8091FF; --accent-d:#6B7BF0; --accent-soft:#1B2340; --on-soft:#EDF1F7;
  --ok:#5BC98A; --ok-ink:#8FE3AE; --ok-soft:#12271C; --ok-line:#1E7E4C;
  --radius:16px; --r-sm:8px; --r-md:12px; --r-full:9999px;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
  --shadow-sm:0 2px 8px rgba(0,0,0,.28);
  --display:'Bricolage Grotesque',Georgia,sans-serif;
  --body:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono','Fira Code',Consolas,monospace;
  --ease:cubic-bezier(.16,1,.3,1);
  --bg:#161C26; --bg2:#0F141C; --panel:#1F2836; --panel2:#273140; --panel3:#2F3A4A;
  --ink:#EDF1F7; --ink2:#BFC9D8; --muted:#93A0B2; --line:#313C4C; --line2:#3E4A5B;
}
html[data-theme="dim"]{--bg:#161C26; --bg2:#0F141C; --panel:#1F2836; --panel2:#273140; --panel3:#2F3A4A;
  --ink:#EDF1F7; --ink2:#BFC9D8; --muted:#93A0B2; --line:#313C4C; --line2:#3E4A5B;}
html[data-theme="dark"]{--bg:#10151D; --bg2:#0A0E15; --panel:#18202B; --panel2:#202A36; --panel3:#28323F;
  --ink:#E9EEF5; --ink2:#B8C2D1; --muted:#8B97A8; --line:#28313F; --line2:#35414F;}
html[data-theme="midnight"]{--bg:#090C12; --bg2:#05070B; --panel:#10151D; --panel2:#161D28; --panel3:#1E2732;
  --ink:#E6ECF4; --ink2:#AEB9C8; --muted:#828FA1; --line:#212A37; --line2:#2C3644;}
html[data-theme="light"]{
  --bg:#FAF7F2; --bg2:#F1EBE0; --panel:#FFFFFF; --panel2:#FBF7F0; --panel3:#F3EDE3;
  --ink:#2C2A28; --ink2:#55504A; --muted:#6E675F; --line:#E5DFD6; --line2:#D8D0C4;
  --accent:#2A7B9B; --accent-d:#1F6280; --accent-soft:#E4F2F7; --on-soft:#1F6280;
  --ok:#2D8B55; --ok-ink:#1A5C38; --ok-soft:#E8F5EE; --ok-line:#2D8B55;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--body);color:var(--ink);background:var(--bg);line-height:1.6;font-size:16.5px;
  -webkit-font-smoothing:antialiased;
  background-image:
    radial-gradient(ellipse 110% 55% at 82% -8%,rgba(128,145,255,.10),transparent 60%),
    radial-gradient(ellipse 90% 50% at 6% 108%,rgba(255,155,92,.06),transparent 60%);
  background-attachment:fixed;min-height:100vh}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--line2);border-radius:var(--r-full)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}

.layout{display:flex;align-items:flex-start;max-width:1320px;margin:0 auto}
#sidebar{position:sticky;top:0;height:100vh;width:288px;flex:0 0 288px;
  display:flex;flex-direction:column;padding:22px 0 0;border-right:1px solid var(--line);background:var(--panel)}
.sidebar-scroll{flex:1 1 auto;overflow-y:auto;min-height:0;padding:0 16px}
.brand{padding:6px 8px 14px;border-bottom:1px solid var(--line);margin-bottom:12px}
.brand-title{font-family:var(--display);font-weight:800;font-size:1.2rem;letter-spacing:-.02em;color:var(--ink)}
.brand-dot{color:var(--accent)}
.brand-sub{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.theme-row{padding:10px 8px 0}
.theme-btns{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:6px;background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:3px}
.theme-btn{font-family:var(--body);font-size:.72rem;font-weight:600;color:var(--muted);background:none;border:0;border-radius:7px;padding:5px 4px;cursor:pointer;transition:.15s}
.theme-btn:hover{color:var(--ink)}
.theme-btn.active{background:var(--accent);color:#fff}
.nav-group-label{display:block;font-size:.68rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;padding:4px 8px;margin-top:14px}
.toc{display:flex;flex-direction:column;gap:1px}
.nav-link{display:flex;align-items:center;gap:.5rem;padding:7px 10px;margin:1px 0;border-radius:9px;color:var(--ink2);
  font-size:.8rem;border-left:2px solid transparent;text-align:left;background:none;border:0;cursor:pointer}
.nav-link:hover{background:var(--panel2);text-decoration:none;color:var(--ink)}
.nav-link .nl-dot{width:7px;height:7px;border-radius:50%;background:var(--line2);flex:0 0 auto}
.side-nav{flex:0 0 auto;padding:12px 16px 16px;border-top:1px solid var(--line);background:var(--panel2);display:flex;flex-direction:column;gap:6px}
.side-nav a{display:flex;flex-direction:column;gap:1px;padding:8px 10px;border:1px solid var(--line2);border-radius:9px;background:var(--panel);color:var(--ink);transition:.15s}
.side-nav a:hover{border-color:var(--accent);text-decoration:none;transform:translateY(-1px)}
.side-nav .d{font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.side-nav .t{font-size:.82rem;font-weight:600}

#content{flex:1 1 auto;min-width:0;padding:44px clamp(20px,5vw,60px) 90px;max-width:940px}
.eyebrow{display:block;font-family:var(--mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:.8rem}

/* rendered-markdown styles */
.md h1{font-family:var(--display);font-size:2rem;font-weight:800;margin:0 0 .8rem;color:var(--ink);line-height:1.15;scroll-margin-top:1rem}
.md h2{font-family:var(--display);font-size:1.4rem;font-weight:800;margin:2.1rem 0 .7rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--ink);scroll-margin-top:1rem}
.md h3{font-family:var(--display);font-size:1.12rem;font-weight:700;margin:1.4rem 0 .5rem;color:var(--ink);scroll-margin-top:1rem}
.md h4{font-size:1rem;font-weight:700;margin:1.1rem 0 .4rem;color:var(--ink2);scroll-margin-top:1rem}
.md p{margin:.7rem 0;color:var(--ink2)}
.md ul,.md ol{margin:.6rem 0 .6rem 1.35rem;color:var(--ink2)}
.md li{margin:.3rem 0}
.md li>ul,.md li>ol{margin-top:.3rem}
.md a{color:var(--accent)}
.md strong{color:var(--ink);font-weight:700}
.md em{color:var(--ink)}
.md hr{border:0;border-top:1px solid var(--line);margin:1.8rem 0}
.md blockquote{border-left:3px solid var(--accent);background:var(--panel2);padding:.7rem 1rem;margin:1rem 0;border-radius:0 var(--r-sm) var(--r-sm) 0;color:var(--ink2)}
.md blockquote p{margin:.35rem 0}
.md blockquote strong{color:var(--ink)}
.md code{font-family:var(--mono);font-size:.86em;background:var(--panel3);padding:.1em .35em;border-radius:5px;color:var(--ink)}
.md pre{background:var(--panel3);border:1px solid var(--line2);border-radius:var(--r-md);padding:1rem;overflow-x:auto;margin:1rem 0}
.md pre code{background:none;padding:0;color:var(--ink);font-size:.82rem;line-height:1.6}
.md table{border-collapse:collapse;width:100%;margin:1.1rem 0;font-size:.86rem;display:block;overflow-x:auto}
.md th,.md td{border:1px solid var(--line);padding:.5rem .7rem;text-align:left;vertical-align:top}
.md thead th{background:var(--panel2);color:var(--ink);font-weight:700;white-space:nowrap}
.md tbody tr:nth-child(even){background:var(--panel2)}
.md tbody td{color:var(--ink2)}

#nav-toggle{display:none}
#scrim{display:none}
@media(max-width:900px){
  .layout{display:block}
  #sidebar{position:fixed;z-index:70;top:0;left:0;transform:translateX(-100%);transition:transform .22s var(--ease);
    box-shadow:var(--shadow);width:290px;height:100vh}
  #sidebar.open{transform:translateX(0)}
  #nav-toggle{display:inline-flex;align-items:center;gap:.4rem;position:fixed;top:12px;right:12px;z-index:65;
    background:var(--accent);color:#fff;border:none;padding:9px 15px;border-radius:10px;font-weight:700;font-family:var(--body);font-size:.85rem;cursor:pointer;box-shadow:var(--shadow)}
  #scrim.show{display:block;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:69}
  #content{padding:60px 18px 70px;max-width:100%}
  .md h1{font-size:1.7rem}
}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>

<button id="nav-toggle" type="button" aria-label="Toggle menu" aria-expanded="false">☰ Contents</button>
<div id="scrim" aria-hidden="true"></div>

<div class="layout">
  <aside id="sidebar" aria-label="roadmap navigation">
    <div class="sidebar-scroll">
      <div class="brand">
        <div class="brand-title">Frontier<span class="brand-dot">.</span>Lab</div>
        <div class="brand-sub">Roadmap &amp; Build Status</div>
      </div>

      <div class="theme-row" role="group" aria-label="appearance mode">
        <span class="nav-group-label" style="padding:0;margin-top:0">Appearance</span>
        <div class="theme-btns">
          <button class="theme-btn" data-theme-set="light" type="button">Light</button>
          <button class="theme-btn" data-theme-set="dim" type="button">Dim</button>
          <button class="theme-btn" data-theme-set="dark" type="button">Dark</button>
          <button class="theme-btn" data-theme-set="midnight" type="button">Midnight</button>
        </div>
      </div>

      <div>
        <span class="nav-group-label">On this page</span>
        <nav class="toc" id="toc" aria-label="Sections">
{{TOC}}
        </nav>
      </div>
    </div>

    <div class="side-nav">
      <a href="index.html"><span class="d">▦ Map</span><span class="t">Back to the curriculum</span></a>
    </div>
  </aside>

  <main id="content">
    <span class="eyebrow">Single source of truth · generated from ROADMAP.md</span>
    <article class="md">
{{BODY}}
    </article>
  </main>
</div>

<script>
// ── appearance switcher (same shared localStorage key as the hub & every lesson) ──
var themeBtns = Array.prototype.slice.call(document.querySelectorAll('.theme-btn'));
function setTheme(t){
  try{ localStorage.setItem('frontier-theme', t); }catch(e){}
  document.documentElement.setAttribute('data-theme', t);
  themeBtns.forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-theme-set')===t); });
}
themeBtns.forEach(function(b){ b.addEventListener('click', function(){ setTheme(b.getAttribute('data-theme-set')); }); });
setTheme(document.documentElement.getAttribute('data-theme') || 'dark');

// ── mobile sidebar toggle ──
var sidebar = document.getElementById('sidebar'), toggle = document.getElementById('nav-toggle'), scrim = document.getElementById('scrim');
function openSidebar(){ sidebar.classList.add('open'); scrim.classList.add('show'); toggle.setAttribute('aria-expanded','true'); }
function closeSidebar(){ sidebar.classList.remove('open'); scrim.classList.remove('show'); toggle.setAttribute('aria-expanded','false'); }
if(toggle) toggle.addEventListener('click', function(){ sidebar.classList.contains('open')?closeSidebar():openSidebar(); });
if(scrim) scrim.addEventListener('click', closeSidebar);
// close the mobile drawer after picking a section
Array.prototype.slice.call(document.querySelectorAll('#toc a')).forEach(function(a){
  a.addEventListener('click', function(){ if(window.innerWidth<=900) closeSidebar(); });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
