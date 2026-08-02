// Run: node sessions/_compiler/tests/test_shelf.mjs   (exit 0 = pass)
import { isUnlocked } from '../shells/js/shelf.js'
import assert from 'node:assert'

// isUnlocked takes the RAW localStorage string (or null) — never touches localStorage itself.
assert.equal(isUnlocked(null), false, 'no stored state -> locked')
assert.equal(isUnlocked(''), false, 'empty string -> locked')
assert.equal(isUnlocked('not json{'), false, 'corrupt JSON must not throw')
assert.equal(isUnlocked('{"done":{}}'), false, 'visited but nothing done -> locked')
assert.equal(isUnlocked('{"done":{"c1":true}}'), true, 'any done section -> unlocked')
assert.equal(isUnlocked('{"done":{"produce":true}}'), true, 'produce also unlocks')
// deliberately looser than pillStatus(): started is enough
assert.equal(isUnlocked('{"done":{"c1":true},"sr":{"x":{"next":1}}}'), true, 'sr key is ignored')

console.log('test_shelf: unlock rule OK')
