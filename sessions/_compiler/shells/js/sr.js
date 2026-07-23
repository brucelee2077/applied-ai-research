// SM-2 spaced repetition — browser mirror of coach/core.py:667-714 (record_quiz_answer)
// + get_due_concepts (641-665). Pure logic, no DOM/network, so it is unit-testable and
// can be inlined into v9-base.donor to run over the existing frontier-lesson localStorage.
// The XP side-effects (core.py:702-707) are intentionally OMITTED.
//
// `today` is an integer DAY INDEX (e.g. Math.floor(Date.now()/86400000)) so the recurrence
// is deterministic and testable; the donor supplies the real day index.
export const SR_INITIAL_EASE = 2.5
export const SR_MIN_EASE = 1.3

// Update one concept's SR record after a self-rated answer.
// state: { [conceptId]: {ease, interval, next, reps} }  (mutated + returned)
// quality: 0-5 (>=3 = correct). Returns the same state object.
export function review(state, conceptId, quality, today) {
  const prev = state[conceptId] || {}
  const ease = prev.ease != null ? prev.ease : SR_INITIAL_EASE
  const interval = prev.interval != null ? prev.interval : 1
  const reps = prev.reps != null ? prev.reps : 0

  let newEase = ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
  newEase = Math.max(SR_MIN_EASE, newEase)

  let newInterval
  if (quality >= 3) {
    if (reps === 0) newInterval = 1
    else if (reps === 1) newInterval = 6
    else newInterval = Math.round(interval * newEase)
  } else {
    // wrong answer: reset (don't penalize ease too harshly, per core.py)
    newInterval = 1
    newEase = SR_INITIAL_EASE
  }

  state[conceptId] = {
    ease: Math.round(newEase * 1000) / 1000,
    interval: newInterval,
    next: today + newInterval,
    reps: reps + 1,
  }
  return state
}

// Concepts due for review at `today`, most-overdue first (mirrors get_due_concepts).
export function dueConcepts(state, today) {
  return Object.keys(state || {})
    .filter((cid) => state[cid] && state[cid].next != null && state[cid].next <= today)
    .sort((a, b) => (today - state[a].next) - (today - state[b].next))
    .reverse()
}
