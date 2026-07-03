#!/usr/bin/env python3
"""
inject_quiz.py — splice agent-authored scenario quizzes (JSON sidecars in
sessions/_quiz_steps/) into their lessons' QS array. Validates each sidecar
(exactly 4 questions, 4 opts each, ans in 0..3, non-empty q/fb) before injecting.
If a sidecar is invalid, the lesson keeps its original quiz (no regression).
"""
import glob, re, json, os, sys
BASE=os.path.dirname(os.path.abspath(__file__))

def valid(qs):
    if not isinstance(qs,list) or len(qs)!=4: return False,"!=4 questions"
    for q in qs:
        if not isinstance(q,dict): return False,"non-object question"
        if not q.get('q') or not q.get('fb'): return False,"missing q/fb"
        o=q.get('opts')
        if not (isinstance(o,list) and len(o)==4): return False,"opts!=4"
        a=q.get('ans')
        if not (isinstance(a,int) and 0<=a<=3): return False,f"ans {a} invalid"
    return True,"ok"

def lesson_for(sc):
    base=os.path.basename(sc)[:-5]
    m=re.match(r'(week-\d{2})_(day-.+)_html$', base)
    return os.path.join(BASE, m.group(1), m.group(2)+".html") if m else None

def main():
    scs=sorted(glob.glob(os.path.join(BASE,"_quiz_steps","*.json")))
    inj=0; bad=[]
    for sc in scs:
        try: qs=json.load(open(sc,encoding='utf-8'))
        except Exception as e: bad.append((sc,f"json {e}")); continue
        ok,why=valid(qs)
        if not ok: bad.append((sc,why)); continue
        lp=lesson_for(sc)
        if not lp or not os.path.exists(lp): bad.append((sc,"lesson missing")); continue
        t=open(lp,encoding='utf-8').read()
        i=t.find('var QS='); j=t.find("\nvar quizWrap",i)
        if i<0 or j<0: bad.append((sc,"QS block not found")); continue
        newjs='var QS='+json.dumps(qs,ensure_ascii=False)+';'
        t=t[:i]+newjs+t[j:]
        open(lp,'w',encoding='utf-8').write(t)
        inj+=1
    print(f"injected {inj}; skipped {len(bad)}")
    for sc,w in bad: print("  ~",os.path.basename(sc),w)
    return 0

if __name__=="__main__": sys.exit(main())
