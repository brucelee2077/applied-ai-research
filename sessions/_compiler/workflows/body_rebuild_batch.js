export const meta = {
  name: 'body-rebuild-batch',
  description: 'Rebuild the remaining m02+m03 concept bodies under the Build-Up Register, accept-gating each day',
  whenToUse: 'Task 10 of the concept-body engagement plan: chunk dense build-ups without losing coverage or visuals.',
  phases: [
    { title: 'Batch 1' },
    { title: 'Batch 2' },
    { title: 'Batch 3' },
    { title: 'Batch 4' },
    { title: 'Summary' },
  ],
}

const LB = '/Users/ruifengli/Desktop/applied-ai-research/sessions/_compiler/workflows/lesson_build.js'
const A = args || {}

// Batches keep concurrency ~3 so the account-wide floodgate cap (320 req/4min)
// is never approached; each lesson_build itself fans out to ~7 agents.
const BATCHES = A.batches || [
  [['m02-the-neuron', 'day-05-gradients-backprop'],
   ['m02-the-neuron', 'day-06-training-loop'],
   ['m02-the-neuron', 'day-09-train-val-test']],
  [['m02-the-neuron', 'day-01-single-neuron'],
   ['m02-the-neuron', 'day-02-activations'],
   ['m02-the-neuron', 'day-03-layers-forward-pass']],
  [['m02-the-neuron', 'day-04-loss'],
   ['m02-the-neuron', 'day-08-learning-rate'],
   ['m03-attention', 'day-01-embeddings']],
  [['m03-attention', 'day-02-qkv'],
   ['m03-attention', 'day-03-attention-scores'],
   ['m03-attention', 'day-04-multihead'],
   ['m03-attention', 'day-05-positional']],
]

const PREP_SCHEMA = {
  type: 'object',
  properties: {
    frozen: { type: 'string' },
    inventory: { type: 'object' },
    concepts: { type: 'integer' },
    mean_prose: { type: 'integer' },
    densest: { type: 'array', items: { type: 'object' } },
  },
  required: ['frozen', 'inventory', 'concepts', 'mean_prose', 'densest'],
}

const ACCEPT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['KEEP', 'REVERT', 'ERROR'] },
    fails: { type: 'array', items: { type: 'string' } },
    warns: { type: 'array', items: { type: 'string' } },
    before_after: { type: 'string' },
  },
  required: ['verdict', 'fails', 'warns'],
}

// Load one day's frozen front-matter + chunking mandate.
async function prep(module_, day, phaseName) {
  return await agent(
    `Read /tmp/rebuild_args.json and return the entry for key "${day}".
If that file does not exist, first run:
  cd /Users/ruifengli/Desktop/applied-ai-research && python3 sessions/_rebuild_args.py
Return the "frozen" string EXACTLY as stored (verbatim — do not reformat, re-indent,
or re-escape it; it is YAML that will be pasted into source.md between --- fences),
plus inventory, concepts, mean_prose, and densest. JSON only.`,
    { label: `args:${day}`, phase: phaseName, schema: PREP_SCHEMA })
}

// The chunking mandate, handed to round 0 as seedFindings so the author targets
// the densest build-ups immediately instead of discovering them in a fix round.
function mandate(p) {
  return [{
    kind: 'body_engagement',
    severity: 'P0',
    why: `USER REPORT 2026-07-24: each concept's BODY is "a little bit hard to digest, tedious, boring". `
      + `The voice judge already grades this day's bodies GOOD — the defect is DENSITY, not warmth. `
      + `This day averages ${p.mean_prose} characters of build-up PROSE per concept. `
      + `Densest build-ups: ${p.densest.map(d => `"${d.concept}" (${d.prose_chars} chars, longest unbroken paragraph ${d.longest_wall})`).join('; ')}.`,
    fix: `CHUNK every dense build-up; do NOT cut coverage. Turn runs of prose/equations into `
      + `%%% steps ladders (one move + its plain-English "why" per rung), add %%% insight re-hooks `
      + `mid-mechanism so the body answers a live question, and add predict: lines to %%% demo blocks `
      + `so worked examples become discoveries. Put a one-line "so far..." breath between chunks. `
      + `Give steps rungs SEMANTIC names ("the collapse", "the fix"), never a bare Step 1/2/3. `
      + `PRESERVE this day's inventory (${JSON.stringify(p.inventory)}): ship AT LEAST as many `
      + `svg+viz+demo blocks, all ${p.concepts} concepts, and every coverage_topic in the frozen `
      + `front-matter. A multi-line demo out: may repeat the "out:" key — each line is kept.`,
  }]
}

// Accept-gate one day. The gate itself decides KEEP/REVERT; this agent only reports.
async function acceptGate(module_, day, phaseName) {
  return await agent(
    `Run this EXACT command from /Users/ruifengli/Desktop/applied-ai-research:

  python3 sessions/_rebuild_accept.py ${module_}/${day} --json /tmp/accept_${day}.json

It compares the working tree against the committed version and prints KEEP or REVERT
with per-check reasons. It takes several minutes (two LLM judge calls) — wait for it.
Do NOT modify, revert, or stage any file. Report the verdict, every FAIL line, every
warn line, and the before->after numbers it prints.`,
    { label: `accept:${day}`, phase: phaseName, schema: ACCEPT_SCHEMA })
}

// One day, end to end: load mandate -> rebuild -> accept-gate.
async function rebuildDay(module_, day, phaseName) {
  const p = await prep(module_, day, phaseName)
  if (!p) return { module: module_, day, error: 'prep failed' }

  let built = null, err = null
  try {
    built = await workflow({ scriptPath: LB }, {
      module: module_, day, frozen: p.frozen, seedFindings: mandate(p), maxRounds: 4,
    })
  } catch (e) {
    err = String(e)
  }
  if (err) {
    log(`${day}: REBUILD ERRORED — ${err}`)
    return { module: module_, day, error: err }
  }

  const accept = await acceptGate(module_, day, phaseName)
  const verdict = accept ? accept.verdict : 'ERROR'
  log(`${day}: converged=${built && built.converged} rounds=${built && built.rounds} accept=${verdict}`)
  return {
    module: module_, day,
    converged: !!(built && built.converged),
    rounds: built ? built.rounds : null,
    residual_p0: (built && built.routing) ? built.routing.p0.length : null,
    accept: accept || { verdict: 'ERROR', fails: ['accept agent died'], warns: [] },
    report: built ? built.report : null,
  }
}

const all = []
for (let i = 0; i < BATCHES.length; i++) {
  const phaseName = `Batch ${i + 1}`
  phase(phaseName)
  const batch = BATCHES[i]
  log(`=== ${phaseName}: ${batch.map(b => b[1]).join(', ')} ===`)
  const results = await parallel(batch.map(([m, d]) => () => rebuildDay(m, d, phaseName)))
  results.filter(Boolean).forEach(r => all.push(r))
  const kept = results.filter(r => r && r.accept && r.accept.verdict === 'KEEP').length
  log(`=== ${phaseName} done: ${kept}/${batch.length} KEEP ===`)
}

phase('Summary')
const keep = all.filter(r => r.accept && r.accept.verdict === 'KEEP')
const revert = all.filter(r => r.accept && r.accept.verdict === 'REVERT')
const errored = all.filter(r => r.error || (r.accept && r.accept.verdict === 'ERROR'))

log(`TOTAL ${all.length} days: ${keep.length} KEEP, ${revert.length} REVERT, ${errored.length} ERROR`)
revert.forEach(r => log(`REVERT ${r.day}: ${(r.accept.fails || []).join(' | ')}`))
errored.forEach(r => log(`ERROR  ${r.day}: ${r.error || 'accept error'}`))

return {
  total: all.length,
  keep: keep.map(r => r.day),
  revert: revert.map(r => ({ day: r.day, fails: r.accept.fails })),
  errored: errored.map(r => ({ day: r.day, error: r.error })),
  days: all,
}
