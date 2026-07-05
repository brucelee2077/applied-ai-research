// Staff Lens audit — scans every built module lesson (sessions/m*/day-*/lesson.html)
// and reports, per lesson, whether it meets the Staff Lens bar and renders clean.
//
// Bar (for lessons whose section id="s4" is a Mechanism/math section):
//   - a named silent failure-mode callout  (string "Failure mode (silent)")
//   - a named trade-off callout             (string "Trade-off:")
// Render (jsdom): 7 .sec sections, 4 quiz Qs, 16 options, 0 empty options, no page JS errors.
//
// Usage:  node sessions/staff_lens_audit.js            (audit all)
//         node sessions/staff_lens_audit.js m17a        (audit one module prefix)
//
// Requires jsdom at /tmp/jsdomcheck/node_modules/jsdom
//   (npm install jsdom --no-save --prefix /tmp/jsdomcheck)
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('/tmp/jsdomcheck/node_modules/jsdom');

const SESS = __dirname;
const filterPrefix = process.argv[2] || '';

function listLessons() {
  const out = [];
  for (const mod of fs.readdirSync(SESS)) {
    if (!/^m\d/.test(mod)) continue;
    if (filterPrefix && !mod.startsWith(filterPrefix)) continue;
    const modDir = path.join(SESS, mod);
    if (!fs.statSync(modDir).isDirectory()) continue;
    for (const day of fs.readdirSync(modDir)) {
      const f = path.join(modDir, day, 'lesson.html');
      if (fs.existsSync(f)) out.push({ mod, day, f });
    }
  }
  return out.sort((a, b) => a.f.localeCompare(b.f));
}

function renderCheck(html) {
  return new Promise((resolve) => {
    const errs = [];
    const vc = new VirtualConsole();
    vc.on('jsdomError', (e) => { if (!/Could not load link|fonts\.googleapis/.test(e.message)) errs.push(e.message); });
    const dom = new JSDOM(html, { runScripts: 'dangerously', virtualConsole: vc, pretendToBeVisual: true });
    setTimeout(() => {
      const d = dom.window.document;
      const r = {
        secs: d.querySelectorAll('.sec').length,
        q: d.querySelectorAll('#quiz .q').length,
        o: d.querySelectorAll('#quiz .q-opt').length,
        empty: [...d.querySelectorAll('#quiz .q-opt span:last-child')].filter((s) => !s.textContent.trim()).length,
        warn: d.querySelectorAll('#s4 .callout.c-warn').length,
        errs,
      };
      dom.window.close();
      resolve(r);
    }, 150);
  });
}

(async () => {
  const lessons = listLessons();
  let done = 0, gap = 0, broken = 0;
  const gaps = [], breaks = [];
  for (const { mod, day, f } of lessons) {
    const html = fs.readFileSync(f, 'utf8');
    const hasFail = html.includes('Failure mode (silent)');
    const hasTrade = /Trade-off:/.test(html);
    const jsonQuiz = /var QS=\[\{/.test(html);
    const r = await renderCheck(html);
    const renderOk = r.secs === 7 && r.q === 4 && r.o === 16 && r.empty === 0 && r.errs.length === 0;
    const compliant = hasFail && hasTrade;
    if (!renderOk) { broken++; breaks.push(`${mod}/${day}  render=${JSON.stringify(r)}`); }
    if (compliant) { done++; } else { gap++; gaps.push(`${mod}/${day}  fail=${hasFail} trade=${hasTrade} jsonQuiz=${jsonQuiz}`); }
    const mark = compliant ? (renderOk ? 'OK ' : 'OK*') : 'gap';
    console.log(`${mark}  ${mod}/${day}  [fail:${hasFail ? 1 : 0} trade:${hasTrade ? 1 : 0} warn:${r.warn} q:${r.q} render:${renderOk ? 'ok' : 'BROKEN'}]`);
  }
  console.log('\n==== SUMMARY ====');
  console.log(`lessons: ${lessons.length}  |  staff-lens callouts present: ${done}  |  gap: ${gap}  |  render-broken: ${broken}`);
  if (breaks.length) { console.log('\nRENDER-BROKEN:'); breaks.forEach((b) => console.log('  ' + b)); }
  process.exit(broken > 0 ? 2 : 0);
})();
