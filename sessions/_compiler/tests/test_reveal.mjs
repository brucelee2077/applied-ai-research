// Run: node sessions/_compiler/tests/test_reveal.mjs   (exit 0 = pass)
// Tests the pure reveal.js scroll-reveal with a hand-rolled DOM stub + a fake
// IntersectionObserver — no jsdom (mirrors test_sr.mjs's bare-node approach).
import { revealBuild } from '../shells/js/reveal.js'
import assert from 'node:assert'

// --- tiny DOM stub -----------------------------------------------------------
function node(cls) {
  const classes = new Set((cls || '').split(' ').filter(Boolean))
  const n = {
    _classes: classes,
    children: [],
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
    },
  }
  n.querySelectorAll = (sel) => {
    const want = sel.replace(/^\./, '')
    const out = []
    const walk = (m) => (m.children || []).forEach((ch) => { if (ch._classes.has(want)) out.push(ch); walk(ch) })
    walk(n)
    return out
  }
  return n
}
function mkRoot() {
  const root = node('root')
  for (let i = 0; i < 2; i++) {
    const b = node('build')
    b.children = [node('build-step'), node('build-step')]
    root.children.push(b)
  }
  return root
}
function created() { const list = []; class FakeIO {
    constructor(cb) { this.cb = cb; this.obs = []; list.push(this) }
    observe(el) { this.obs.push(el) }
    unobserve(el) { this.obs = this.obs.filter((x) => x !== el) }
    fire() { this.cb(this.obs.map((el) => ({ isIntersecting: true, target: el }))) }
  }
  return { IO: FakeIO, list }
}
const allSteps = (root) => root.querySelectorAll('.build-step')
const allBuilds = (root) => root.querySelectorAll('.build')

// --- (a) IO + motion: containers arm, steps reveal only after intersection ---
{
  const root = mkRoot(); const { IO, list } = created()
  revealBuild(root, { IO, reducedMotion: false })
  assert.ok(allBuilds(root).every((b) => b.classList.contains('armed')), 'each .build armed')
  assert.ok(allSteps(root).every((s) => !s.classList.contains('revealed')), 'steps hidden until scroll')
  list.forEach((o) => o.fire())
  assert.ok(allSteps(root).every((s) => s.classList.contains('revealed')), 'steps revealed after intersect')
}

// --- (b) reduced motion: reveal all, never arm -------------------------------
{
  const root = mkRoot(); const { IO } = created()
  revealBuild(root, { IO, reducedMotion: true })
  assert.ok(allBuilds(root).every((b) => !b.classList.contains('armed')), 'no arm under reduced motion')
  assert.ok(allSteps(root).every((s) => s.classList.contains('revealed')), 'all revealed under reduced motion')
}

// --- (c) no IntersectionObserver: reveal all, never arm ----------------------
{
  const root = mkRoot()
  revealBuild(root, { IO: null, reducedMotion: false })
  assert.ok(allBuilds(root).every((b) => !b.classList.contains('armed')), 'no arm without IO')
  assert.ok(allSteps(root).every((s) => s.classList.contains('revealed')), 'all revealed without IO')
}

// --- (d) idempotent: a 2nd call creates no new observers ---------------------
{
  const root = mkRoot(); const { IO, list } = created()
  revealBuild(root, { IO, reducedMotion: false })
  const n = list.length
  revealBuild(root, { IO, reducedMotion: false })
  assert.equal(list.length, n, 'armed containers are skipped on re-run')
}

console.log('ok: reveal.js multi-container scroll-reveal')
