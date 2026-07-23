// Run: node sessions/_compiler/tests/test_sr.mjs   (exit 0 = pass)
import { review, dueConcepts, SR_INITIAL_EASE } from '../shells/js/sr.js'
import assert from 'node:assert'

let s = {}
s = review(s, 'c1', 5, 0)                 // reps 0, perfect -> interval 1
assert.equal(s.c1.interval, 1); assert.equal(s.c1.next, 1); assert.equal(s.c1.reps, 1)

s = review(s, 'c1', 5, 1)                 // reps 1 -> interval 6
assert.equal(s.c1.interval, 6); assert.equal(s.c1.next, 7); assert.equal(s.c1.reps, 2)

s = review(s, 'c1', 5, 7)                 // reps 2 -> round(interval*ease), grows past 6
assert.ok(s.c1.interval > 6, 'interval should grow on 3rd success')

s = review(s, 'c1', 1, 20)                // wrong -> reset interval to 1, ease to initial
assert.equal(s.c1.interval, 1)
assert.equal(s.c1.ease, SR_INITIAL_EASE)

// due selection: next <= today, most overdue first
assert.deepEqual(dueConcepts({ a: { next: 0 }, b: { next: 5 } }, 3), ['a'])
assert.deepEqual(dueConcepts({ a: { next: 1 }, b: { next: 2 } }, 5), ['a', 'b'])
assert.deepEqual(dueConcepts({}, 10), [])

console.log('ok: sr.js SM-2 mirror')
