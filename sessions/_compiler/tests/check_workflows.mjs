// Validate a Workflow script the way the runtime actually evaluates it: the body
// runs inside an async function (so top-level `return` and `await` are legal),
// with `export const meta = {...}` hoisted out.
//
// Neither plain `node --check` nor module-mode `--check` is correct here:
//   * `node --check` returns exit 0 on a BROKEN file that contains an ESM
//     `export` — verified: a double comma in an object literal passes. Every
//     workflow file starts with `export const meta`, so every node --check on one
//     was vacuous.
//   * module-mode --check rejects the top-level `return` these scripts legitimately
//     use, so it fails all 7 files whether or not they are broken.
import { readFileSync } from 'fs'
let bad = 0
for (const f of process.argv.slice(2)) {
  const src = readFileSync(f, 'utf8').replace(/^export\s+const\s+meta\s*=/m, 'const meta =')
  try {
    new Function(`return (async () => {\n${src}\n})`)
    console.log(`  ok    ${f.split('/').pop()}`)
  } catch (e) {
    bad++
    console.log(`  FAIL  ${f.split('/').pop()}: ${e.message.slice(0, 100)}`)
  }
}
process.exit(bad ? 1 : 0)
