// Verify the DONOR's __revealBuild scroll-reveal against REAL compiled lessons
// in a REAL DOM, including the accessibility fallbacks.
//
// Why this exists alongside tests/test_reveal.mjs: that test exercises the pure
// shells/js/reveal.js MODULE against a hand-rolled stub. This one exercises the
// DONOR MIRROR as it actually ships inside a compiled lesson.html — the code a
// reader's browser really runs — together with the CSS rule
// `.build-step:not(.revealed){opacity:0}`. If the mirror ever drifts from the
// module, or a %%% steps ladder lands in a container the observer never arms,
// the worked example renders as BLANK SPACE. That is invisible to every offline
// gate and to the LLM judges (they read the source text, not the rendered DOM).
//
// Requires jsdom at /tmp/jsdomcheck/node_modules/jsdom
//   (npm install jsdom --no-save --prefix /tmp/jsdomcheck)
// — the same convention staff_lens_audit.js uses; public CDNs are 403-blocked
// in this sandbox, so install from the internal npm registry.
//
// Usage: node sessions/_reveal_audit.js <lesson.html> [more.html ...]
// Exit 0 = every lesson passes (a lesson with no .build-step is SKIPped).

const fs = require('fs');
const { JSDOM } = require('/tmp/jsdomcheck/node_modules/jsdom');

// Boot one lesson in jsdom. `io` false removes IntersectionObserver entirely;
// `rm` true makes the reduced-motion media query match.
function boot(html, { rm = false, io = true } = {}) {
  const observers = [];
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    beforeParse(w) {
      if (!io) {
        delete w.IntersectionObserver;
      } else {
        w.IntersectionObserver = class {
          constructor(cb) { this.cb = cb; this.targets = []; observers.push(this); }
          observe(t) { this.targets.push(t); }
          unobserve(t) { this.targets = this.targets.filter((x) => x !== t); }
          disconnect() { this.targets = []; }
          // fire as if the reader scrolled every step into view
          fireAll() { this.cb(this.targets.map((t) => ({ isIntersecting: true, target: t })), this); }
        };
      }
      w.matchMedia = (q) => ({
        matches: rm && /reduced-motion/.test(q),
        addEventListener() {}, addListener() {},
      });
      w.scrollTo = () => {};
    },
  });
  return { dom, observers };
}

function counts(doc) {
  const steps = [...doc.querySelectorAll('.build-step')];
  return {
    steps,
    builds: [...doc.querySelectorAll('.build')],
    revealed: () => steps.filter((s) => s.classList.contains('revealed')).length,
    armed: (d) => [...d.querySelectorAll('.build')].filter((b) => b.classList.contains('armed')).length,
  };
}

function audit(file) {
  const html = fs.readFileSync(file, 'utf8');
  const name = file.split('/').slice(-2).join('/');

  // Probe once just to see whether this lesson has any steps ladders at all.
  const probe = boot(html);
  const nSteps = probe.dom.window.document.querySelectorAll('.build-step').length;
  probe.dom.window.close();
  if (!nSteps) {
    console.log(`SKIP ${name} — no .build-step`);
    return null;
  }

  const problems = [];

  // (1) normal: scrolling reveals every step
  {
    const { dom, observers } = boot(html);
    const d = dom.window.document;
    const c = counts(d);
    const hasFn = typeof dom.window.__revealBuild === 'function';
    observers.forEach((o) => o.fireAll && o.fireAll());
    const rev = c.revealed();
    if (!hasFn) problems.push('__revealBuild missing from the donor');
    if (rev !== c.steps.length) {
      problems.push(`scroll revealed ${rev}/${c.steps.length} steps — the rest render as blank space`);
    }
    if (c.armed(d) !== c.builds.length) {
      problems.push(`armed ${c.armed(d)}/${c.builds.length} .build containers`);
    }
    console.log(`  scroll: builds=${c.builds.length} armed=${c.armed(d)} steps=${c.steps.length} revealed=${rev}`);
    dom.window.close();
  }

  // (2)+(3) fallbacks: with reduced motion, or with no IntersectionObserver,
  // every step must already be revealed (never left at opacity:0).
  for (const [label, opts] of [['reduced-motion', { rm: true }], ['no-IO', { io: false }]]) {
    const { dom } = boot(html, opts);
    const d = dom.window.document;
    const c = counts(d);
    const rev = c.revealed();
    if (rev !== c.steps.length) {
      problems.push(`${label}: only ${rev}/${c.steps.length} steps revealed — invisible for these readers`);
    }
    if (c.armed(d) !== 0) problems.push(`${label}: ${c.armed(d)} container(s) armed (should degrade, not animate)`);
    console.log(`  ${label}: revealed=${rev}/${c.steps.length} armed=${c.armed(d)}`);
    dom.window.close();
  }

  const ok = problems.length === 0;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}`);
  problems.forEach((p) => console.log(`   ! ${p}`));
  return ok;
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node sessions/_reveal_audit.js <lesson.html> [...]');
  process.exit(2);
}
let failed = 0, checked = 0;
for (const f of files) {
  const r = audit(f);
  if (r === null) continue;
  checked++;
  if (!r) failed++;
}
console.log(`\n${checked} lesson(s) with steps ladders checked, ${failed} failed`);
process.exit(failed ? 1 : 0);
