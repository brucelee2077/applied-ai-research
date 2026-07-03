#!/usr/bin/env python3
"""
inject_viz.py — splice agent-authored visual BUILD steps (JSON sidecars in
sessions/_viz_steps/) into their lesson files, replacing the code-only baseline BUILD.
Validates each sidecar (>=5 steps, each has viz+note, has visual markers) before injecting.
"""
import glob, re, json, os, sys
BASE=os.path.dirname(os.path.abspath(__file__))
VISUAL=('<svg','dgram','class="node','<rect','<circle','<path','<line','<polyline')

def visual_ok(steps):
    if not isinstance(steps,list) or len(steps)<5: return False,"<5 steps"
    for s in steps:
        if not (isinstance(s,dict) and s.get('viz') and s.get('note')): return False,"step missing viz/note"
    nvis=sum(1 for s in steps if any(k in s['viz'] for k in VISUAL))
    if nvis < max(3, len(steps)-1): return False, f"only {nvis}/{len(steps)} steps visual"
    if any('background:#1E1E2E' in s['viz'] for s in steps): return False,"contains dark code block"
    return True,"ok"

def lesson_for_sidecar(sc):
    # sidecar name = week-NN_day-MM-slug_html.json  -> sessions/week-NN/day-MM-slug.html
    base=os.path.basename(sc)[:-5]  # strip .json
    # reverse the re.sub([/.]->_): the path was week-NN/day-...html
    m=re.match(r'week-(\d2|\d{2})_day-(.+)_html$', base)
    # simpler: split on first '_day-'
    m=re.match(r'(week-\d{2})_(day-.+)_html$', base)
    if not m: return None
    return os.path.join(BASE, m.group(1), m.group(2)+".html")

def main():
    scs=sorted(glob.glob(os.path.join(BASE,"_viz_steps","*.json")))
    injected=skipped=0; bad=[]
    for sc in scs:
        try: steps=json.load(open(sc,encoding='utf-8'))
        except Exception as e: bad.append((sc,f"json err {e}")); continue
        ok,why=visual_ok(steps)
        if not ok: bad.append((sc,why)); continue
        lp=lesson_for_sidecar(sc)
        if not lp or not os.path.exists(lp): bad.append((sc,"lesson not found")); continue
        t=open(lp,encoding='utf-8').read()
        i=t.find('var BUILD'); j=t.find("\nvar buildWrap=document.getElementById('build')",i)
        if i<0 or j<0: bad.append((sc,"BUILD block not found")); continue
        newjs='var BUILD='+json.dumps(steps,ensure_ascii=False)+';'
        t=t[:i]+newjs+t[j:]
        open(lp,'w',encoding='utf-8').write(t)
        injected+=1
    print(f"injected {injected}; skipped/bad {len(bad)}")
    for sc,why in bad[:40]: print("  ✗",os.path.basename(sc),"::",why)
    return 0 if injected else 1

if __name__=="__main__":
    sys.exit(main())
