// Run: node sessions/_compiler/tests/test_shelf.mjs   (exit 0 = pass)
import { isUnlocked, TOYS, shelfSummary } from '../shells/js/shelf.js'
import assert from 'node:assert'

// isUnlocked takes the RAW localStorage string (or null) — never touches localStorage itself.
assert.equal(isUnlocked(null), false, 'no stored state -> locked')
assert.equal(isUnlocked(''), false, 'empty string -> locked')
assert.equal(isUnlocked('not json{'), false, 'corrupt JSON must not throw')
assert.equal(isUnlocked('{}'), false, 'no done key -> locked')
assert.equal(isUnlocked('{"sr":{"x":{"next":1}}}'), false, 'sr alone does not unlock')
assert.equal(isUnlocked('{"done":{}}'), false, 'visited but nothing done -> locked')
assert.equal(isUnlocked('{"done":{"c1":true}}'), true, 'any done section -> unlocked')
assert.equal(isUnlocked('{"done":{"produce":true}}'), true, 'produce also unlocks')
// deliberately looser than pillStatus(): started is enough
assert.equal(isUnlocked('{"done":{"c1":true},"sr":{"x":{"next":1}}}'), true, 'sr key is ignored')

console.log('test_shelf: unlock rule OK')

assert.equal(TOYS.length, 17, 'exactly 17 shelf-eligible viz pages')

// every row is well-formed
for (const t of TOYS) {
  assert.equal(t.length, 5, 'row shape [name, label, qid, verb, page]: ' + t[0])
  for (const f of t) assert.ok(typeof f === 'string' && f.length, 'no empty field in ' + t[0])
  assert.ok(t[4].startsWith('viz/') && t[4].endsWith('.html'), 'page path: ' + t[4])
}

// no duplicate pages, no duplicate names
assert.equal(new Set(TOYS.map(t => t[4])).size, TOYS.length, 'no duplicate page paths')
assert.equal(new Set(TOYS.map(t => t[0])).size, TOYS.length, 'no duplicate display names')

// the five unembedded pages must NOT be present
for (const bad of ['attention-heatmap','attention-multihead','attention-pipeline',
                   'softmax-scaling','leaky-slope']) {
  assert.ok(!TOYS.some(t => t[4] === 'viz/' + bad + '.html'), bad + ' has no owner, must be excluded')
}
// but diffusion-noising IS embedded (by a flat week-m22b page) and must be present
assert.ok(TOYS.some(t => t[4] === 'viz/diffusion-noising.html'), 'diffusion-noising is owned')

// curated ownership overrides — `|| []` so a rename shows an assertion diff, not a TypeError
const owner = n => (TOYS.find(t => t[4] === 'viz/' + n + '.html') || [])[2]
assert.equal(owner('neuron-boundary'), 'wf2-d01-neuron', 'curated: day-01 holds it inline')
assert.equal(owner('xor-limit'),       'wf2-d01-neuron', 'curated: day-01 holds xor-c inline')
assert.equal(owner('matmul'),          'wf1-d04-matmul', 'simple case: earliest embedder')

// shelfSummary takes a reader fn so it never touches localStorage
const REAL_TODAY = ['wf1-d01-arrays','wf1-d02-indexing','wf1-d03-broadcasting','wf1-d04-matmul',
                    'wf1-d05-logs','wf1-d06-seeds','wf1-review','wf2-d01-neuron','wf2-d02-activations']
const read = qid => REAL_TODAY.includes(qid) ? '{"done":{"c1":true}}' : null

const sum = shelfSummary(read)
assert.equal(sum.total, TOYS.length)
assert.equal(sum.unlocked.length, 5, 'learner has exactly 5 toys today')
assert.deepEqual(sum.unlocked.map(t => t[0]).sort(),
  ['Activation curves','Broadcasting','Matmul shapes','Neuron weights','XOR limit'])
assert.equal(sum.locked.length, TOYS.length - 5)
// order is preserved from TOYS (curriculum order): Broadcasting is m01, so it comes first
assert.equal(sum.unlocked[0][0], 'Broadcasting', 'unlocked list keeps TOYS order')
assert.equal(sum.unlocked[sum.unlocked.length - 1][0], 'Activation curves', 'and ends at m02 Day 2')

// empty store -> nothing unlocked, still no throw
const none = shelfSummary(() => null)
assert.equal(none.unlocked.length, 0)
assert.equal(none.locked.length, TOYS.length)

// blocked site data makes localStorage.getItem throw SecurityError. That throw must
// not escape shelfSummary, or one browser setting stops the whole hub rendering.
const blocked = shelfSummary(() => { throw new Error('SecurityError: storage blocked') })
assert.equal(blocked.unlocked.length, 0, 'blocked storage -> nothing unlocked')
assert.equal(blocked.locked.length, TOYS.length, 'blocked storage -> everything locked')

console.log('test_shelf: TOYS table + summary OK')
