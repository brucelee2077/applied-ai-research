// Scroll-reveal for %%% steps build-up scaffolds — pure logic, no globals, so it is
// unit-testable under bare node (like sr.js) and can be inlined into v9-base.donor.
//
// Reveals `.build-step` children of every `.build` container as they scroll into view,
// turning a narrated worked-example into a satisfying assemble-as-you-read build. It is a
// MULTI-container implementation (a lesson may have many %%% steps blocks in different
// concept sections) and is DECOUPLED from `.gotit` / section unlock. Graceful: with no
// IntersectionObserver or reduced motion, every step is revealed at once (never hidden).
//
// Deps are INJECTED (root + IntersectionObserver ctor + reducedMotion flag) so the donor
// passes the real ones and tests pass stubs.
export function revealBuild(root, opts) {
  opts = opts || {}
  const IO = opts.IO                       // IntersectionObserver constructor (or falsy)
  const reducedMotion = !!opts.reducedMotion
  const containers = root.querySelectorAll('.build')
  for (let i = 0; i < containers.length; i++) {
    const c = containers[i]
    if (c.classList.contains('armed')) continue   // idempotent — already wired
    const steps = c.querySelectorAll('.build-step')
    if (reducedMotion || !IO) {
      // no animation path: show everything immediately (graceful degradation)
      for (let j = 0; j < steps.length; j++) steps[j].classList.add('revealed')
      continue
    }
    c.classList.add('armed')
    const obs = new IO(function (entries) {
      for (let k = 0; k < entries.length; k++) {
        if (entries[k].isIntersecting) {
          entries[k].target.classList.add('revealed')
          obs.unobserve(entries[k].target)
        }
      }
    }, { threshold: 0.25 })
    for (let j = 0; j < steps.length; j++) obs.observe(steps[j])
  }
}
