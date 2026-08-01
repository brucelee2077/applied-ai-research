export const meta = {
  name: 'doing-leg-build',
  description: 'Per-module DOING leg engine (the learner-facing sessions/<module>/<day>/experiment.py, which lesson_build.js does NOT write): triage skips every day whose artifact already passes the gate AND drops any day whose produce spec cannot be read (terminal SPEC_GAP for a human — a writer with no spec invents requirements, the one failure this engine exists to prevent) -> ONE writer agent per day in parallel (spec from _produce_spec.py, env from _experiment_env.md, loop until _experiment_check.py passes AND two runs are byte-identical AND its own planted defects are caught) -> ONE module-wide adversarial reviewer that plants semantic defects and names vacuous claims -> a cross-day consistency pass that is ON BY DEFAULT and whose absence blocks VERIFIED on any multi-day module (CROSS_DAY_UNCHECKED) -> one bounded repair round (every P0, plus P1 cross-day collisions) -> a final deterministic gate sweep. args {module, days:[...], maxRounds, cumulative, force, plants, repair}.',
  whenToUse: 'To backfill or rebuild a module\'s doing leg after its lessons pass. NOT for portfolio evidence — that is evidence_build.js, which writes a different file under portfolio/ to a different contract.',
  phases: [
    { title: 'Triage' },
    { title: 'Write' },
    { title: 'Review' },
    { title: 'Cross-day' },
    { title: 'Repair' },
    { title: 'Route' },
  ],
}

// ---------------------------------------------------------------------------
// ARGS
//   module     (required) "m0X-name" — the module directory under sessions/.
//   days       (required) explicit list of day directory names. See the note below.
//   maxRounds  bounded write/fix rounds per day (default 3).
//   cumulative DEFAULT TRUE — run the cross-day consistency pass. Only pass
//              cumulative:false for a module of genuinely independent days, and
//              know that saying false does NOT buy a VERIFIED verdict: a
//              multi-day module without the cross-day pass comes back
//              CROSS_DAY_UNCHECKED.
//              ONE-LINE TEST: does any later day reuse an earlier day's helper,
//              identifier, formula or printed shape? Then it is cumulative. A
//              capstone / synthesis / "full model" day is an automatic yes.
//   force      rewrite every listed day even if its artifact already passes.
//   plants     minimum semantic defects each day's self-check must catch (default 4).
//   repair     false to skip the bounded repair round (findings then just stand).
//
// ⚠️ OBSERVED FAILURE MODE — THE MODULE-WIDE REVIEWER STALLS ON A BIG MODULE.
// This engine funnels the whole module through ONE adversarial reviewer, which is
// deliberate (a per-day reviewer cannot see cross-day collisions). But on a 9-day
// module that is 70+ plants in a single agent, and on 2026-08-01 an equivalent
// hand-run reviewer for a 5-day module STALLED SIX TIMES (no progress for 180s
// each) and its workflow failed with the repairs already applied — leaving the
// module repaired but UNVERIFIED, which is the worst state to leave silently.
// Mitigations, in order of preference:
//   1. Run the engine on a SUBSET of days per invocation (the day list is already
//      explicit) and let the cross-day pass run once over the full module after.
//   2. Lower `plants` for the first pass, then re-run for depth.
//   3. If a run dies after the writers, DO NOT report the module as verified.
//      Re-plant by hand: copy each day to /tmp, apply one recorded plant at a
//      time, and diff FULL stdout. A stalled prover is missing evidence, not
//      evidence of success.
// ---------------------------------------------------------------------------
const A = args || {}
const module_ = A.module || ''
const days = A.days || []
const MAX_ROUNDS = A.maxRounds || 3
// Opting OUT is the deliberate act. Wave 2 shipped six REAL cross-day
// collisions with every day green because this defaulted false and nothing
// derived it, so the default invocation {module, days} ran no cross-day pass.
const cumulative = A.cumulative !== false
const force = !!A.force
const PLANTS_MIN = A.plants || 4
const REPAIR = A.repair !== false

// The day list is EXPLICIT on purpose. Wave 1 assumed a module's days were all
// stubs, and five write agents overwrote artifacts that were already real. The
// caller enumerates the days it means (workflow scripts have no filesystem
// access), and Triage below still re-checks every one of them before writing.
if (!module_ || !Array.isArray(days) || days.length === 0) {
  throw new Error('doing-leg-build requires args {module: "m0X-name", days: ["day-01-...", ...]}. '
    + 'Enumerate the days deliberately — never let the engine guess which ones are stubs. '
    + 'To see the current state first: python3 sessions/_experiment_check.py --module <module>')
}
// These land inside shell paths in agent prompts.
for (const name of [module_].concat(days)) {
  if (typeof name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) || name.includes('..')) {
    throw new Error(`Unsafe module/day name ${JSON.stringify(name)} — must be a plain directory name`)
  }
}

const GOLD = 'sessions/m02-the-neuron/day-02-activations/experiment.py'

// Backlog IDs use the SHORT day form the manifests and frontier-refactor-qa
// already speak (SEED-LX-D2-FORMULA-FIRST, DOING-D3-VACUOUS-CHECK). The full
// directory name would produce DOING-day-03-attention-scores-STUB-ARTIFACT,
// which no consumer of the manifest recognises.
function shortDay(day) {
  const m = /(?:^|[-_])d(?:ay)?[-_]?(\d+)/i.exec(day)
  return m ? `D${String(Number(m[1]))}` : day.toUpperCase()
}

// ---------------------------------------------------------------------------
// Shared prompt fragments. Every one of these is a rule that was learned by
// shipping the mistake; AUTHORING.md section 10 holds the full catalogue with
// worked examples and stays the single source of truth for it.
// ---------------------------------------------------------------------------

const SAFETY = `HARD CONSTRAINTS. Never run a git write command (no add / commit / checkout / stash / restore / rm / mv); read-only git (status, log, diff, show) is fine. Do not touch source.md, lesson.html, any other day's files, anything under portfolio/, any gate, or any skill. Work from the repo root /Users/ruifengli/Desktop/applied-ai-research and use system python3.`

const VACUITY = `A PASSING ✅ IS NOT EVIDENCE THE CHECK MEANS ANYTHING. Every one of these was proven on a real day by planting the bug and watching the script still print "✅ you got it" — read sessions/_compiler/AUTHORING.md section 10 for the catalogue with worked examples, and do not reproduce any of them:
  (1) circular — the expected value is re-derived from the code path under test; (2) too weak to see the bug (a sorted multiset after a reshape); (3) fake prediction — a hardcoded "predict" string, so compute the prediction from the inputs; (4) coupled to a library's error WORDING — assert the exception TYPE instead; (5) entailed clause — a tight pin that implies the loose one next to it; (6) constant fold — both sides are literals; (7) dead branch as a prediction; (8) self-derived tautology (argmin vs min of the same list); (9) identity / scale-invariant claim (cosine(v, v) == 1.0); (10) print/assert divergence — the printed line and the asserted line are two SEPARATE expressions for one quantity, so corrupting only the print is invisible: compute once, bind a name, print that name and assert that same name.
Pin every claim against a value WRITTEN DOWN in the self-check — prefer an exact round(x, 4) == <literal>, because a tolerance can only ever be widened.
MAKE THE CODE PATH OBSERVABLE. A parameter initialised to zero cannot be tested (deleting a bias that is np.zeros changes nothing). A transpose is invisible on symmetric data (Q = K = V = x). /err.size vs /err.shape[0] is invisible when Y is (8, 1). A UNIFORM OUTPUT hides an axis bug exactly as well as a symmetric input — run the reduction on a lopsided map too. Floor division absorbs a changed kernel size ((32-4+2)//2+1 and (32-3+2)//2+1 are both 16), so evaluate several settings computed from ONE shared literal. A strictness (> 0.85 vs >= 0.85) is untestable unless a value sits exactly ON the boundary — add the tie. A symmetric aggregation hides an internal transpose, and "the diagonal is always the row max" hides reading the max instead of the LABELLED diagonal — pin the directions separately.`

const PLANT_METHOD = `HOW TO PLANT. Copy the file to /tmp first (cp <path> /tmp/<day>.plant.py) and plant THERE — never in the repo, so a killed agent cannot leave a planted defect behind. Each plant changes exactly ONE thing and must be a real SEMANTIC defect, not a numeric tweak: drop a bias / a normalisation / a /N; transpose an operand; swap mean for sum; set lr = 0; remove zero_grad; wrap the forward in no_grad; flip an update sign; drop the ragged final batch; use the row max instead of the labelled diagonal. Run the planted copy and require it to FAIL (non-zero exit, or a printed ❌).
BEFORE YOU CALL A PLANT "NOT CAUGHT", DIFF THE FULL STDOUT (run both, redirect to two files, diff them). A plant that reproduces the same VALUE is a NO-OP and proves nothing about the check — hardcoding exact_steps = 13.5 where the code computes (28-3+2)/2 is byte-identical by construction, so pick a different plant instead of reporting a gap. Some no-ops are CORRECT and should be recorded rather than chased: subtracting the max inside a softmax cannot change the result, and deleting eps is invisible except on a flat channel.`

const ENV = `THE ENVIRONMENT IS BINDING. Read sessions/_experiment_env.md before you write a line — it was measured on this machine. In short: no network (the gate injects a shim that makes socket connects RAISE, so a download fails loudly); no file writes, no savefig, no plot window (MPLBACKEND=Agg is forced); deterministic — seed everything (np.random.default_rng(0), torch.manual_seed(0)); fast (aim under 10s, hard timeout 180s); NO CUDA (torch.cuda.is_available() is False — never print a number that implies a GPU, and treat roofline/FLOP/KV-cache days as ARITHMETIC over published spec numbers, not benchmarks). NO dataset is cached (no MNIST/CIFAR) — synthesize a small seeded stand-in, keep the shapes the lesson taught (784 -> 128 -> 10), and say so in a comment AT THE POINT OF USE. Several HF models ARE cached (gpt2, distilbert-base-uncased, all-MiniLM-L6-v2, bge-*, Qwen*) with HF_HUB_OFFLINE=1; bert-base-uncased is NOT cached — use distilbert-base-uncased. If the produce step asks for something impossible here, build the closest honest thing, say what changed in a comment, and report it in substitutions — never silently drop it and never fake it.`

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const TRIAGE_SCHEMA = {
  type: 'object',
  properties: {
    days: { type: 'array', items: { type: 'object',
      properties: {
        day: { type: 'string' },
        gate_pass: { type: 'boolean' },
        stub: { type: 'boolean' },
        lines: { type: 'integer' },
        reasons: { type: 'array', items: { type: 'string' } },
        spec_found: { type: 'boolean',
          description: 'false ONLY when _produce_spec.py printed "no produce section found"' },
        spec_prompt: { type: 'boolean',
          description: 'a buildable requirement list was extracted: claude_prompt, OR the produce block itself carries the numbered requirements' },
        spec_acceptance: { type: 'boolean',
          description: 'the acceptance field is non-null (a WARNING when false, never an exclusion)' },
        spec_note: { type: 'string', description: 'what the spec pre-flight printed / what is missing' },
      }, required: ['day', 'gate_pass'] } },
    output_tail: { type: 'string' },
  },
  required: ['days'],
}

const WRITE_SCHEMA = {
  type: 'object',
  properties: {
    wrote: { type: 'boolean' },
    gate_pass: { type: 'boolean' },
    gate_output: { type: 'string' },
    deterministic: { type: 'boolean' },
    lines: { type: 'integer' },
    claims: { type: 'array', items: { type: 'string' },
      description: 'one line per pinned claim the self-check asserts' },
    plants: { type: 'array', items: { type: 'object',
      properties: {
        defect: { type: 'string' },
        caught: { type: 'boolean' },
        how: { type: 'string', description: 'non-zero exit / printed ❌ / assert message' },
        stdout_diffed: { type: 'boolean' },
      }, required: ['defect', 'caught', 'how'] } },
    substitutions: { type: 'string', description: 'what the produce step asked for that this box cannot do, and what was built instead' },
    stdout_tail: { type: 'string' },
  },
  required: ['wrote', 'gate_pass', 'gate_output', 'deterministic', 'plants'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['PASS', 'GAPS'] },
    findings: { type: 'array', items: { type: 'object',
      properties: {
        day: { type: 'string' },
        kind: { type: 'string', description: 'vacuous_claim | uncaught_plant | not_deterministic | env_violation | spec_gap | lesson_mismatch | stub | correctness' },
        severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
        why: { type: 'string' }, fix: { type: 'string' },
      }, required: ['day', 'kind', 'severity', 'why'] } },
    plants: { type: 'array', items: { type: 'object',
      properties: {
        day: { type: 'string' }, defect: { type: 'string' },
        caught: { type: 'boolean' },
        no_op: { type: 'boolean', description: 'true when the plant reproduced the same values (proves nothing — not a gap)' },
        evidence: { type: 'string' },
      }, required: ['day', 'defect', 'caught'] } },
    correct_no_ops: { type: 'array', items: { type: 'string' },
      description: 'value-preserving changes that are CORRECT, recorded so a later reviewer does not chase them' },
    summary: { type: 'string' },
  },
  required: ['verdict', 'findings', 'plants'],
}

const XDAY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['PASS', 'GAPS'] },
    findings: { type: 'array', items: { type: 'object',
      properties: {
        days: { type: 'array', items: { type: 'string' } },
        kind: { type: 'string', description: 'formula_alias | name_collision | word_meaning | lost_knobs | notation_split | misplaced_concept' },
        severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
        why: { type: 'string' }, fix: { type: 'string' },
      }, required: ['days', 'kind', 'severity', 'why'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'findings'],
}

const SWEEP_SCHEMA = {
  type: 'object',
  properties: {
    passed: { type: 'integer' },
    failed: { type: 'integer' },
    per_day: { type: 'array', items: { type: 'object',
      properties: { day: { type: 'string' }, ok: { type: 'boolean' }, reasons: { type: 'array', items: { type: 'string' } } },
      required: ['day', 'ok'] } },
    output_tail: { type: 'string' },
  },
  required: ['passed', 'failed', 'per_day'],
}

// ---------------------------------------------------------------------------
// Stage 0 — Triage. Never write over an artifact that already works, and never
// hand a writer a day whose spec cannot be read.
// ---------------------------------------------------------------------------

phase('Triage')
async function triage() {
  return await agent(
    `You are the DOING-LEG TRIAGE agent (READ-ONLY — write nothing, run no git write command).
Run this ONE command from /Users/ruifengli/Desktop/applied-ai-research (it takes one path per day and both runs and contract-checks each file):

  python3 sessions/_experiment_check.py ${days.map(d => `sessions/${module_}/${d}/experiment.py`).join(' ')} --json /tmp/doing_triage_${module_}.json

THEN A SPEC PRE-FLIGHT, one command per day:

  python3 sessions/_produce_spec.py sessions/${module_}/<day>

The writer is forbidden from editing source.md, so a day with no readable produce spec cannot be written by anyone — it would send its writer into INVENTING requirements, the one failure this engine exists to prevent. Read each day's JSON and report:
  - spec_found: false ONLY when the command printed "no produce section found" (or the "produce" field is empty). That day is a terminal SPEC_GAP for a human to fix in the lesson source.
  - spec_prompt: is there a buildable requirement list? TRUE when "claude_prompt" is non-null, and ALSO true when claude_prompt is null but the "produce" block itself spells out numbered requirements (measured: sessions/m03-attention/day-03-attention-scores has claude_prompt: null and a 5-step numbered produce block — that is NOT a spec gap). FALSE only when neither exists.
  - spec_acceptance: is the "acceptance" field non-null? A false here is a WARNING you report, NOT a reason to stop: quote what is missing in spec_note. (Measured on this repo today: acceptance resolves on real V9 days — day-03-attention-scores returns the "What you should see by the end: …" line — and the summary run reports 0 of 69 stub days without acceptance criteria. So a null here is now unusual and worth naming.)
  - spec_note: one line — which fields were null and the first line of the produce block.

For EACH of the ${days.length} days also report: day, gate_pass (the "ok" line — contract passed AND it ran: exit 0, a ✅ with no ❌, no network, no timeout), stub (true when a reason says it is the placeholder stub or the file is missing), lines (wc -l of the file, 0 if missing), and every FAIL reason verbatim.
Report exactly what the commands printed. Do NOT fix anything and do NOT judge quality — this is a state read.`,
    { label: `triage:${module_}`, phase: 'Triage', schema: TRIAGE_SCHEMA })
}

let tri = await triage()
if (!tri) {
  log('triage agent died — retrying once')
  tri = await triage()
}
if (!tri && !force) {
  const msg = `ABORT: triage failed twice, so the engine cannot tell which of ${days.length} days already have a REAL experiment.py. `
    + 'Refusing to write — overwriting working artifacts is the wave-1 mistake this stage exists to prevent. '
    + 'Re-run after checking by hand: python3 sessions/_experiment_check.py --module ' + module_ + ' (or pass force:true to rewrite every listed day).'
  log(msg)
  return { module: module_, aborted: true, reason: msg, days: [], skipped: [], spec_gaps: [], spec_warnings: [], review: null, cross_day: null, cross_day_ran: false, cross_day_unchecked: true, sweep: null, verified_count: 0, total: days.length, report: msg }
}

const triMap = {}
for (const t of ((tri && tri.days) || [])) triMap[t.day] = t

const skipped = []
const targets = []
// Days the engine REFUSES to write because nothing states what they must do.
// Terminal on purpose: a writer told to build to a missing spec invents the
// requirements, and a reviewer's spec_gap P0 would route straight back to a
// writer who is not allowed to touch source.md, so the loop cannot converge.
const specGaps = []
const specWarnings = []
for (const day of days) {
  const t = triMap[day]
  if (t && t.spec_prompt === false) {
    const note = (t.spec_note || '').slice(0, 300)
      || (t.spec_found === false ? '_produce_spec.py found no produce section' : 'no buildable requirement list in the produce block')
    specGaps.push({ day, reason: note })
    log(`SPEC_GAP ${module_}/${day} — no extractable produce prompt (${note}). NOT handed to a writer: fix the lesson source by hand, then re-run.`)
    continue
  }
  if (t && t.spec_acceptance === false) {
    specWarnings.push({ day, reason: (t.spec_note || 'no acceptance section extracted').slice(0, 300) })
    log(`WARNING ${module_}/${day} — the produce spec has a prompt but NO acceptance section; the writer pins claims from the prompt and the lesson's own numbers instead. Not a blocker.`)
  }
  if (t && t.gate_pass && !force) {
    skipped.push({ day, lines: t.lines || null, reason: 'already passes _experiment_check.py' })
    log(`SKIP ${module_}/${day} — experiment.py already passes the gate (${t.lines || '?'} lines). Pass force:true to rewrite.`)
  } else {
    targets.push(day)
    const why = !t ? 'no triage row (treated as needing work)'
      : force ? 'force:true' : (t.stub ? 'stub/missing' : `gate FAIL: ${(t.reasons || []).join('; ').slice(0, 200)}`)
    log(`WRITE ${module_}/${day} — ${why}`)
  }
}
log(`triage: ${targets.length} to write, ${skipped.length} skipped, ${specGaps.length} SPEC_GAP (of ${days.length})`)
if (specGaps.length) {
  log(`SPEC_GAP days are excluded from the Write stage and cannot be VERIFIED by any engine run: ${specGaps.map(g => g.day).join(', ')}. `
    + 'A human (or a source pass) must add the produce prompt to the lesson source first.')
}

// ---------------------------------------------------------------------------
// Stage 1 — one WRITER per day, in parallel. Per-day bounded round loop.
// ---------------------------------------------------------------------------

phase('Write')
async function writeRound(day, round, findings) {
  const path = `sessions/${module_}/${day}/experiment.py`
  const fb = findings
    ? `\n\nFIX ROUND ${round}. Address EVERY finding below in ${path}, then re-run the gate, the determinism diff and your plants from scratch. A finding you cannot reproduce must still be answered — say why it does not hold, with the command output that shows it:\n${JSON.stringify(findings, null, 2)}`
    : ''
  return await agent(
    `You are the DOING-LEG WRITER for ${module_}/${day}. You own EXACTLY ONE file: ${path} — the learner's DOING leg, the script the lesson's produce section tells them to run. ${SAFETY}

1. THE SPEC ALREADY EXISTS — NEVER INVENT REQUIREMENTS. Run:
     python3 sessions/_produce_spec.py sessions/${module_}/${day}
   It prints the day's produce block, its Option-B numbered requirement list (claude_prompt) and its acceptance criteria. That IS your contract; build to it, not to what you would have chosen. Triage already confirmed this day HAS a buildable prompt — a day without one never reaches you. Two shapes are normal: claude_prompt may be null while the produce block itself carries the numbered steps (build to those), and a null acceptance means you pin claims from the numbered steps and the lesson's own printed numbers instead. If you nonetheless find NOTHING to build to, STOP and report it in substitutions — do not invent requirements and do not edit source.md. Then read the day's lesson (sessions/${module_}/${day}/source.md, or lesson.html when there is no source.md) for the exact names, shapes and NUMBERS the lesson promises. The lesson's %%% demo out: lines and your stdout are two copies of the same claim and no gate compares them — print the same expression the lesson shows, keeping the spelling that produces it (dtype included: np.maximum(0, z) on an int array prints array([0, 0, 0, 2, 5]); np.where(z > 0, z, 0.0) does not). If the lesson and the produce spec disagree, follow the produce spec and report the mismatch.

2. ${ENV}

3. SHAPE — follow the gold standard ${GOLD}: a header naming the day, "Today's big idea in two lines of output", the exact run command (python3 ${path}); imports with a reason each; small named helpers; then if __name__ == "__main__": split into "# --- Part N ---" sections; PRINTED SHAPES and intermediate values at EVERY step; then a self-check that computes ONE BOOLEAN PER CLAIM, prints "✅ you got it" or "❌ not yet — expected …", and asserts each with a message. 85-140 lines is a TARGET, not a cap: extra lines are justified by a printed step, a contrast control or a pinned claim, never by commentary. If the file is long and the claim count is not, cut.

4. ${VACUITY}

5. THE GATE. Run it and iterate until it exits 0:
     python3 sessions/_experiment_check.py ${path}
   It runs gates/experiment_contract.py (not the placeholder stub, parses, >=1 import, a __main__ guard, >=1 assert, a ✅/❌ print) and THEN EXECUTES the file, requiring exit 0, a ✅ with no ❌, under 180s, and no network. Passing it is necessary and NOT sufficient — steps 6 and 7 are what make the ✅ mean something.

6. DETERMINISM. Run the file twice into two files under /tmp and diff them; the output must be BYTE-IDENTICAL. If it is not, seed the source of the difference (never widen a tolerance to hide it).

7. PLANT YOUR OWN DEFECTS — at least ${PLANTS_MIN}, and choose them where THIS day's mechanism could plausibly be misspelled. ${PLANT_METHOD}
   Optional review aid: python3 sessions/_experiment_mutate.py ${path} perturbs one numeric literal at a time. It edits the file IN PLACE and restores it, so if it is interrupted, check git diff --stat before you finish. A surviving mutant is a LEAD, not a verdict, and it is blind to every structural case in step 4.

Report: wrote; gate_pass (only true if _experiment_check.py exited 0 — paste its output in gate_output); deterministic (the two-run diff was empty); lines; claims (one line per pinned claim); plants (defect, caught, how, stdout_diffed); substitutions; stdout_tail (last ~20 lines of a real run).${fb}`,
    { label: `write ${day} r${round}`, phase: 'Write', schema: WRITE_SCHEMA, agentType: 'general-purpose' })
}

function selfFindings(r) {
  const out = []
  if (!r.gate_pass) {
    out.push({ day: '', kind: 'gate', severity: 'P0',
      why: `_experiment_check.py did not exit 0: ${(r.gate_output || '(no output)').slice(-700)}`,
      fix: 'fix the artifact until the gate passes — contract, exit 0, a ✅ with no ❌, no network, no timeout' })
  }
  if (!r.deterministic) {
    out.push({ day: '', kind: 'not_deterministic', severity: 'P0',
      why: 'two runs were not byte-identical — the reviewer diffs two runs, and a learner cannot tell a real change from noise',
      fix: 'seed every source of randomness; do not widen a tolerance to hide it' })
  }
  const plants = r.plants || []
  const caught = plants.filter(p => p.caught)
  if (caught.length < PLANTS_MIN) {
    out.push({ day: '', kind: 'uncaught_plant', severity: 'P0',
      why: `only ${caught.length} of the required ${PLANTS_MIN} planted SEMANTIC defects were caught (${plants.length} planted). A self-check that survives a real defect certifies broken code as correct.`,
      fix: 'pin the surviving claim against a value written down in the self-check, and choose shapes/values where the wrong spelling gives a DIFFERENT answer. Re-check each "not caught" by diffing the FULL stdout first — a value-preserving plant is a no-op, not a gap.' })
  }
  return out
}

// One day, end to end: bounded write -> gate -> determinism -> self-plants.
async function writeDay(day) {
  let r = await writeRound(day, 0, null)
  let round = 0, deaths = 0, lastFindings = null
  while (round < MAX_ROUNDS) {
    if (!r) {
      deaths += 1
      log(`${day}: writer died (API error) — retry ${deaths}/2`)
      if (deaths > 2) break
      r = await writeRound(day, round, lastFindings)
      continue
    }
    const f = selfFindings(r).map(x => ({ ...x, day }))
    if (!f.length) break
    round += 1
    if (round >= MAX_ROUNDS) break
    lastFindings = f
    log(`${day}: r${round - 1} not clean (${f.map(x => x.kind).join(', ')}) -> fix round ${round}`)
    r = await writeRound(day, round, lastFindings)
  }
  if (!r) {
    log(`${day}: NO RESULT (writer died on every retry)`)
    return { day, result: null, rounds: round, verdict: 'ERROR', findings: [] }
  }
  const f = selfFindings(r).map(x => ({ ...x, day }))
  const verdict = f.length ? (r.gate_pass ? 'GATE_ONLY' : 'FAILED') : 'WRITTEN'
  log(`${day}: rounds=${round} gate=${r.gate_pass} deterministic=${r.deterministic} plants_caught=${(r.plants || []).filter(p => p.caught).length}/${(r.plants || []).length} -> ${verdict}`)
  return { day, result: r, rounds: round, verdict, findings: f }
}

// Per-day and independent -> pipeline. Concurrency is capped for us.
const written = targets.length
  ? (await pipeline(targets, day => writeDay(day))).filter(Boolean)
  : []
if (targets.length && written.length !== targets.length) {
  log(`WARNING: ${targets.length - written.length} writer(s) returned nothing`)
}

// ---------------------------------------------------------------------------
// Stages 2+3 — module-wide. Both need EVERY day's artifact to exist first, so
// they run after the pipeline, and in parallel with each other.
// ---------------------------------------------------------------------------

const specGapSet = {}
for (const g of specGaps) specGapSet[g.day] = g.reason

const inScope = days.filter(d => !specGapSet[d] && (targets.indexOf(d) >= 0 || (triMap[d] && triMap[d].gate_pass)))
const scopeNote = `Days IN SCOPE: ${inScope.join(', ')}. Written this run: ${targets.join(', ') || '(none)'}. `
  + `Pre-existing and NOT rewritten: ${skipped.map(s => s.day).join(', ') || '(none)'} — review these too: an artifact written before sessions/_experiment_check.py existed is UNVERIFIED, not done (m03 day-03-attention-scores ran clean, exited 0, and printed no ✅/❌ at all).`
  + (specGaps.length ? ` OUT OF SCOPE (no readable produce spec — terminal SPEC_GAP awaiting a source fix, do not review or repair): ${specGaps.map(s => s.day).join(', ')}.` : '')
  + (specWarnings.length ? ` Spec WARNING (a prompt but no acceptance section — still in scope, judge the claims against the prompt and the lesson's numbers): ${specWarnings.map(s => s.day).join(', ')}.` : '')

async function review() {
  return await agent(
    `You are the ADVERSARIAL DOING-LEG REVIEWER for the WHOLE module ${module_} — ONE reviewer for all its days, not one per day, because the blind spots are cross-file. Spend real effort: this stage empirically catches what the writers missed, INCLUDING in files a writer already reported verified. ${SAFETY} You are READ-ONLY on the repo: plant only on copies under /tmp, and if you run sessions/_experiment_mutate.py (which edits in place and restores) confirm with git diff --stat that every repo file is unchanged before you finish.

${scopeNote}

Your question is NOT "does it run" — the gate already answered that. It is: WOULD THIS SELF-CHECK NOTICE IF THE MECHANISM WERE WRONG?

For EACH day in scope:
1. Read sessions/${module_}/<day>/experiment.py and the day's produce spec (python3 sessions/_produce_spec.py sessions/${module_}/<day>). Does the artifact actually do what the produce section promised, with the numbers the lesson shows?
2. ${VACUITY}
3. ${PLANT_METHOD}
   Plant at least FOUR semantic defects per day, each changing exactly one thing.
4. Run the gate yourself (python3 sessions/_experiment_check.py sessions/${module_}/<day>/experiment.py) and diff two runs for determinism.
5. Check the environment rules are really honored: no network, no file writes/savefig, no CUDA assumption, no uncached dataset or bert-base-uncased, seeded.

Report: verdict (GAPS if ANY P0); findings (day, kind, severity, why, fix) — a claim that cannot fail is a P0 vacuous_claim, a stub or a missing ✅ is a P0, a lesson/artifact number mismatch is a P0 lesson_mismatch; a spec_gap means the LESSON SOURCE is missing the contract, so write the fix as a source/human action (no writer is allowed to edit source.md, and this engine routes a spec_gap to a terminal SPEC_GAP instead of to a repair round); plants (day, defect, caught, no_op, evidence) listing EVERY plant you tried including the no-ops; correct_no_ops (value-preserving changes that are CORRECT, so the next reviewer does not chase them); summary.`,
    { label: `adversarial review:${module_}`, phase: 'Review', schema: REVIEW_SCHEMA, agentType: 'general-purpose' })
}

async function crossDay() {
  return await agent(
    `You are the CROSS-DAY CONSISTENCY reviewer for the cumulative module ${module_}. ${SAFETY} You are READ-ONLY: report, do not edit.

${scopeNote}

NO GATE IS CROSS-FILE, so every collision below passes every gate and every judge on both days and still teaches the learner a contradiction. All six were REAL in wave 2. Read all the in-scope experiment.py files together (and the days' produce specs) and check:
1. formula_alias — one formula appearing under three different names across the days that define it (the conv output-size formula did, across three m06 days).
2. name_collision — a shared identifier naming a DIFFERENT object on different days (causal_mask was an additive -inf grid on m05a days 05/06 and a boolean keep-list on day 08).
3. word_meaning — a word changing meaning between days (channel = axis 0 on m06 day-03, axis 1 on day-04).
4. lost_knobs — a reused helper silently dropping the knobs an earlier day taught AS the mechanism (layer_norm losing gamma/beta).
5. notation_split — a convention spelled two ways (a softmax axis, a batch-first vs time-first order, a shape comment order).
6. misplaced_concept — a comment placing a concept in the wrong module or day.
Also check the ladder: does a later day reuse the earlier day's helper and printed shapes, or silently re-derive them differently?

Report verdict (GAPS if any P0), findings (days[], kind, severity, why, fix) and a one-paragraph summary. Quote the two conflicting lines for every finding — a cross-day claim without both sides quoted is not actionable.`,
    { label: `cross-day:${module_}`, phase: 'Cross-day', schema: XDAY_SCHEMA, agentType: 'general-purpose' })
}

phase('Review')
const moduleAgents = [() => review()]
if (cumulative) moduleAgents.push(() => crossDay())
else log('cross-day pass SKIPPED (args.cumulative:false was passed explicitly) — no day of a multi-day module can come back VERIFIED without it; it is reported CROSS_DAY_UNCHECKED instead')
const moduleResults = inScope.length ? await parallel(moduleAgents) : []
const reviewRes = moduleResults[0] || null
phase('Cross-day')
const xdayRes = cumulative ? (moduleResults[1] || null) : null
if (!reviewRes) log('WARNING: the module-wide adversarial reviewer returned nothing — no day may be reported as VERIFIED this run')
log(`review: verdict=${reviewRes ? reviewRes.verdict : 'ERROR'} findings=${reviewRes ? reviewRes.findings.length : 0} plants=${reviewRes ? reviewRes.plants.length : 0}`)
if (cumulative) log(`cross-day: verdict=${xdayRes ? xdayRes.verdict : 'ERROR'} findings=${xdayRes ? xdayRes.findings.length : 0}`)

// The cross-day pass is the ONLY stage that can see a collision, because no gate
// and no judge is cross-file. If it did not run (or its agent died) on a module
// of more than one day, that module is not verified — it is UNCHECKED, and the
// difference has to survive into the verdict, not just into a log line. Wave 2
// shipped six real collisions with every single day reported green.
const crossDayRan = !!xdayRes
const crossDayUnchecked = inScope.length > 1 && !crossDayRan
const crossDayWhy = cumulative
  ? 'every per-day signal is clean, but the cross-day consistency pass returned NOTHING on a multi-day module — the only stage that can see a formula alias, a name collision or a word changing meaning between days. Re-run it before calling any day done.'
  : 'every per-day signal is clean, but the cross-day consistency pass did not run (cumulative:false) on a multi-day module. No gate and no judge is cross-file, so nothing here has looked for a collision. Re-run with cumulative:true (the default) to reach VERIFIED.'
if (crossDayUnchecked) {
  log(`CROSS-DAY UNCHECKED: ${inScope.length} days in scope and no cross-day result (${cumulative ? 'the cross-day agent returned nothing' : 'cumulative:false'}). `
    + 'Every otherwise-clean day is reported CROSS_DAY_UNCHECKED, not VERIFIED.')
}

// ---------------------------------------------------------------------------
// Stage 4 — one bounded repair round for days the module-wide stages faulted.
// ---------------------------------------------------------------------------

phase('Repair')
// reviewByDay holds the findings that must be REPAIRED before a day can be
// called done. Two severity rules, because the two stages have different
// blindness:
//   - the per-day reviewer: P0 only (its P1/P2 are quality notes for the backlog);
//   - the cross-day reviewer: P0 AND P1, because a P1 word_meaning or
//     notation_split IS the collision — nothing downstream can see it, so
//     printing it and moving on is how wave 2's six collisions shipped green.
// A spec_gap is never repairable by a writer (no writer may edit source.md), so
// it becomes a terminal SPEC_GAP instead of a repair round.
const reviewByDay = {}
for (const f of ((reviewRes && reviewRes.findings) || [])) {
  if (f.severity !== 'P0') continue
  if (typeof f.kind === 'string' && f.kind.indexOf('spec_gap') >= 0) {
    if (!specGapSet[f.day]) {
      specGapSet[f.day] = `reviewer P0 spec_gap: ${(f.why || '').slice(0, 300)}`
      specGaps.push({ day: f.day, reason: specGapSet[f.day] })
      log(`SPEC_GAP ${module_}/${f.day} — raised by the adversarial reviewer. NOT routed to a writer (source.md is off-limits to every agent here): a human fixes the lesson's produce section, then re-run.`)
    }
    continue
  }
  reviewByDay[f.day] = (reviewByDay[f.day] || []).concat([f])
}
for (const f of ((xdayRes && xdayRes.findings) || [])) {
  if (f.severity !== 'P0' && f.severity !== 'P1') continue
  for (const d of (f.days || [])) {
    reviewByDay[d] = (reviewByDay[d] || []).concat([{ ...f, day: d, kind: `cross_day:${f.kind}` }])
  }
}
function worstSeverity(day) {
  const fs = reviewByDay[day] || []
  return fs.some(f => f.severity === 'P0') ? 'P0' : (fs.length ? 'P1' : null)
}

const dayState = {}
for (const w of written) dayState[w.day] = w
for (const s of skipped) {
  if (!dayState[s.day]) dayState[s.day] = { day: s.day, result: null, rounds: 0, verdict: 'PRE_EXISTING', findings: [] }
}
for (const g of specGaps) {
  dayState[g.day] = { day: g.day, result: null, rounds: 0, verdict: 'SPEC_GAP', findings: [], spec_gap: g.reason }
}

const repairDays = Object.keys(reviewByDay).filter(d => days.indexOf(d) >= 0 && !specGapSet[d])
if (!REPAIR) {
  log(`repair SKIPPED (args.repair === false); ${repairDays.length} day(s) carry routable review findings`)
} else if (!repairDays.length) {
  log('repair: nothing to repair (no P0 per-day findings and no P0/P1 cross-day findings from the module-wide stages)')
} else {
  log(`repair: ${repairDays.length} day(s) with routable review findings -> ${repairDays.map(d => `${d} (${worstSeverity(d)})`).join(', ')}`)
  const repaired = await pipeline(repairDays, async day => {
    const prior = dayState[day]
    const startRound = (prior && prior.rounds ? prior.rounds : 0) + 1
    const r = await writeRound(day, startRound, reviewByDay[day])
    return { day, result: r, rounds: startRound, repaired: true }
  })
  for (const rep of repaired.filter(Boolean)) {
    if (!rep.result) { log(`repair ${rep.day}: writer died — findings stand`); continue }
    const f = selfFindings(rep.result).map(x => ({ ...x, day: rep.day }))
    dayState[rep.day] = { day: rep.day, result: rep.result, rounds: rep.rounds, repaired: true,
      verdict: f.length ? (rep.result.gate_pass ? 'GATE_ONLY' : 'FAILED') : 'REPAIRED', findings: f }
    log(`repair ${rep.day}: gate=${rep.result.gate_pass} deterministic=${rep.result.deterministic} -> ${dayState[rep.day].verdict}`)
  }
}

// ---------------------------------------------------------------------------
// Stage 5 — Route. One deterministic sweep, then honest per-day verdicts.
// ---------------------------------------------------------------------------

phase('Route')
const sweep = await agent(
  `You are the FINAL GATE SWEEP (READ-ONLY — write nothing, run no git write command). Run from /Users/ruifengli/Desktop/applied-ai-research:

  python3 sessions/_experiment_check.py --module ${module_} --json /tmp/doing_sweep_${module_}.json

It contract-checks AND RUNS every experiment.py under sessions/${module_}/. Report passed, failed, per_day (day, ok, reasons verbatim for every FAIL) and the tail of its output. Report exactly what it printed — this number is the module's machine truth and is deliberately independent of what the writer agents claimed about their own files.`,
  { label: `final sweep:${module_}`, phase: 'Route', schema: SWEEP_SCHEMA })

// Matched by substring, not by equality: the sweep prints PATHS, so an agent may
// report "sessions/<module>/<day>/experiment.py" instead of the bare day name.
// Getting this wrong would silently downgrade every day, so be liberal here.
function sweepFor(day) {
  const rows = (sweep && sweep.per_day) || []
  for (const s of rows) {
    if (typeof s.day === 'string' && (s.day === day || s.day.indexOf(day) >= 0)) return s
  }
  return null
}

// A day is VERIFIED only when every independent signal agrees: the gate, two-run
// determinism, its own caught plants, the module-wide adversarial reviewer, the
// cross-day pass (on a multi-day module) and the final sweep. The contract gate
// alone is never enough: it cannot tell a real check from one that cannot fail,
// which is exactly how a vacuous ✅ shipped before.
function finalVerdict(day) {
  const st = dayState[day]
  const sw = sweepFor(day)
  const r = st && st.result
  const sweptOk = sw ? sw.ok : null
  // First, and ahead of the sweep: a spec gap is terminal and human-actionable,
  // so it must not be reported as a mere gate FAILED (the stub does fail the
  // sweep, and that reason would hide the real one).
  if (specGapSet[day]) {
    return { verdict: 'SPEC_GAP', why: `no readable produce spec — ${String(specGapSet[day]).slice(0, 300)}. No writer ran (and none may edit source.md): a human adds the produce prompt to the lesson source, then re-run this engine.` }
  }
  if (sweptOk === false) return { verdict: 'FAILED', why: `final sweep FAIL: ${(sw.reasons || []).join('; ').slice(0, 300)}` }
  if (!st) return { verdict: 'NOT_ATTEMPTED', why: 'day was listed but produced no state' }
  if (st.verdict === 'ERROR') return { verdict: 'ERROR', why: 'the writer agent died on every retry' }
  if (!r && st.verdict === 'PRE_EXISTING') {
    if (!reviewRes) return { verdict: 'UNVERIFIED', why: 'pre-existing artifact; the module-wide reviewer did not run' }
    if ((reviewByDay[day] || []).length) {
      return worstSeverity(day) === 'P0'
        ? { verdict: 'REVIEW_P0', why: 'pre-existing artifact with P0 review findings' }
        : { verdict: 'REVIEW_P1', why: 'pre-existing artifact with P1 cross-day findings (a collision no gate can see)' }
    }
    if (crossDayUnchecked) return { verdict: 'CROSS_DAY_UNCHECKED', why: crossDayWhy }
    return { verdict: 'VERIFIED', why: 'pre-existing artifact, gate-passing and adversarially reviewed clean' }
  }
  if (!r || !r.gate_pass) return { verdict: 'FAILED', why: 'the acceptance gate did not pass' }
  if (!r.deterministic) return { verdict: 'NONDETERMINISTIC', why: 'two runs were not byte-identical' }
  if ((r.plants || []).filter(p => p.caught).length < PLANTS_MIN) {
    return { verdict: 'GATE_ONLY', why: `gate passes but fewer than ${PLANTS_MIN} planted semantic defects were caught — the ✅ is not evidence` }
  }
  if ((reviewByDay[day] || []).length) {
    // A repaired day is NOT verified: the reviewer that raised the finding never
    // saw the repair. Say so instead of implying the finding still stands, and
    // instead of implying a fix was confirmed.
    const worst = worstSeverity(day)
    return st.repaired
      ? { verdict: 'REPAIRED_UNREVIEWED', why: `repair applied for ${(reviewByDay[day] || []).length} routed review finding(s) (worst ${worst}) and the gate passes again, but the adversarial reviewer did NOT re-run — re-run this engine (or a review-only pass) before calling the day done` }
      : worst === 'P0'
        ? { verdict: 'REVIEW_P0', why: 'the module-wide reviewer left P0 findings on this day' }
        : { verdict: 'REVIEW_P1', why: 'a P1 cross-day collision stands on this day — no gate and no judge is cross-file, so nothing downstream will catch it' }
  }
  if (sweptOk === null) return { verdict: 'GATE_ONLY', why: 'the final sweep did not report this day' }
  if (!reviewRes) return { verdict: 'GATE_ONLY', why: 'gate + determinism + self-plants pass, but the adversarial reviewer never ran' }
  if (crossDayUnchecked) return { verdict: 'CROSS_DAY_UNCHECKED', why: crossDayWhy }
  return { verdict: 'VERIFIED', why: `gate + two-run determinism + self-plants + module-wide adversarial review${crossDayRan ? ' + cross-day consistency' : ''} + final sweep all agree` }
}

const finals = days.map(day => {
  const st = dayState[day] || null
  const fv = finalVerdict(day)
  const sw = sweepFor(day)
  const r = st && st.result
  return {
    day,
    verdict: fv.verdict,
    why: fv.why,
    written_this_run: targets.indexOf(day) >= 0,
    skipped: skipped.some(s => s.day === day),
    repaired: !!(st && st.repaired),
    rounds: st ? st.rounds : null,
    gate_pass: r ? r.gate_pass : (sw ? sw.ok : null),
    deterministic: r ? r.deterministic : null,
    plants_caught: r ? (r.plants || []).filter(p => p.caught).length : null,
    plants_tried: r ? (r.plants || []).length : null,
    lines: r ? r.lines : (triMap[day] ? triMap[day].lines : null),
    claims: r ? (r.claims || []) : [],
    substitutions: r ? (r.substitutions || '') : '',
    spec_gap: specGapSet[day] || '',
    spec_acceptance_missing: specWarnings.some(w => w.day === day),
    review_p0: (reviewByDay[day] || []).filter(f => f.severity === 'P0').length,
    review_p1: (reviewByDay[day] || []).filter(f => f.severity === 'P1').length,
    backlog_id_prefix: `DOING-${shortDay(day)}`,
  }
})

const verified = finals.filter(f => f.verdict === 'VERIFIED')
const notVerified = finals.filter(f => f.verdict !== 'VERIFIED')
log(`doing-leg ${module_}: ${verified.length}/${finals.length} VERIFIED`)
notVerified.forEach(f => log(`  ${f.verdict.padEnd(20)} ${f.day} — ${f.why}`))

const report = [
  `# Doing-leg build report — ${module_}`,
  ``,
  `- Days listed: ${finals.length} | written this run: ${targets.length} | skipped (already passing): ${skipped.length} | SPEC_GAP (not writable): ${specGaps.length}`,
  `- VERIFIED: ${verified.length}/${finals.length}  (verdict requires gate + two-run determinism + >=${PLANTS_MIN} caught semantic plants + module-wide adversarial review + the cross-day pass on a multi-day module + final sweep)`,
  `- Final sweep over sessions/${module_}/: ${sweep ? `${sweep.passed} passed, ${sweep.failed} failed` : 'DID NOT RUN'}`,
  `- Module-wide adversarial review: ${reviewRes ? `${reviewRes.verdict}, ${reviewRes.findings.length} findings, ${reviewRes.plants.length} plants tried` : 'DID NOT RUN — no day can be called verified'}`,
  `- Cross-day consistency pass: ${crossDayRan
    ? `RAN — ${xdayRes.verdict}, ${xdayRes.findings.length} findings`
    : (cumulative
      ? `DID NOT RUN (the agent returned nothing)${crossDayUnchecked ? ' — so NO day of this multi-day module is VERIFIED; each otherwise-clean day is CROSS_DAY_UNCHECKED' : ' (only one day in scope, so there is no cross-day surface)'}`
      : `NOT RUN — cumulative:false was passed explicitly${crossDayUnchecked ? ', so NO day of this multi-day module is VERIFIED; each otherwise-clean day is CROSS_DAY_UNCHECKED. Re-run with cumulative:true (the default) to close it' : ' (only one day in scope, so there is no cross-day surface)'}`)}`,
  crossDayUnchecked
    ? `- WHY THIS MATTERS: no gate and no judge is cross-file. Wave 2 shipped six real cross-day collisions with every day green because this pass was opt-in and its absence changed no verdict.`
    : '',
  ``,
  `## Per-day`,
  finals.map(f => `- ${f.day}: **${f.verdict}** — ${f.why}`
    + ` (gate=${f.gate_pass}, deterministic=${f.deterministic}, plants=${f.plants_caught}/${f.plants_tried}, lines=${f.lines}, rounds=${f.rounds}${f.skipped ? ', skipped' : ''}${f.repaired ? ', repaired' : ''}${f.review_p1 ? `, cross-day P1=${f.review_p1}` : ''})`).join('\n'),
  specGaps.length ? `\n## SPEC_GAP — terminal, needs a human source fix (NOT handed to any writer)\n${specGaps.map(g => `- ${g.day}: ${g.reason}`).join('\n')}\nA writer may not edit source.md, so there is nothing an agent can do here: add the produce prompt to the lesson source, then re-run this engine on these days.` : '',
  specWarnings.length ? `\n## Spec WARNING — a produce prompt but no acceptance section (built anyway)\n${specWarnings.map(w => `- ${w.day}: ${w.reason}`).join('\n')}` : '',
  skipped.length ? `\n## Skipped (already passed the gate — NOT rewritten)\n${skipped.map(s => `- ${s.day} (${s.lines || '?'} lines): ${s.reason}`).join('\n')}` : '',
  reviewRes && reviewRes.findings.length ? `\n## Adversarial review findings\n${reviewRes.findings.map(f => `- [${f.severity}/${f.kind}] ${f.day}: ${f.why}`).join('\n')}` : `\n(no adversarial review findings)`,
  reviewRes && reviewRes.correct_no_ops && reviewRes.correct_no_ops.length ? `\n## Correct no-ops (recorded so the next reviewer does not chase them)\n${reviewRes.correct_no_ops.map(s => `- ${s}`).join('\n')}` : '',
  xdayRes && xdayRes.findings.length ? `\n## Cross-day findings (no gate catches these — P0 AND P1 are routed to repair)\n${xdayRes.findings.map(f => `- [${f.severity}/${f.kind}] ${(f.days || []).join(' + ')}: ${f.why}`).join('\n')}` : '',
  finals.some(f => f.substitutions) ? `\n## Honest substitutions (what this box could not do)\n${finals.filter(f => f.substitutions).map(f => `- ${f.day}: ${f.substitutions}`).join('\n')}` : '',
  `\nBacklog IDs for the module manifest use the SHORT day form the manifests already speak (D<n> from day-0<n>, as in SEED-LX-D2-FORMULA-FIRST):`
    + ` DOING-D<n>-STUB-ARTIFACT / DOING-D<n>-VACUOUS-CHECK / DOING-XDAY-<NAME>-COLLISION`
    + ` — for example ${finals.length ? `DOING-${shortDay(finals[0].day)}-STUB-ARTIFACT` : 'DOING-D1-STUB-ARTIFACT'}, DOING-${finals.length ? shortDay(finals[0].day) : 'D1'}-VACUOUS-CHECK, DOING-XDAY-NAME-COLLISION.`
    + ` Never the full directory name (DOING-day-03-attention-scores-STUB-ARTIFACT is not a form frontier-refactor-qa or any manifest recognises).`,
].join('\n')
log(report)

return {
  module: module_,
  days: finals,
  skipped,
  spec_gaps: specGaps,
  spec_warnings: specWarnings,
  review: reviewRes,
  cross_day: xdayRes,
  cross_day_ran: crossDayRan,
  cross_day_unchecked: crossDayUnchecked,
  sweep,
  verified_count: verified.length,
  total: finals.length,
  converged: verified.length === finals.length,
  report,
}
