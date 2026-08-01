export const meta = {
  name: 'evidence-build',
  description: 'Per-day frontier evidence: producer writes+runs a REAL experiment.py (deterministic, offline, gated by sessions/_experiment_check.py) + blog.md (reuses the lesson viz) -> evidence_compile assembles a self-contained portfolio page -> evidence-judge (frontier-staff bar + numbers_match) -> bounded loop -> checkpoint. args {module, day, maxRounds}.',
  whenToUse: 'After a lesson passes, to produce its frontier-facing evidence artifact (blog + reproducible experiment + reused demo).',
  phases: [{ title: 'Produce' }, { title: 'Compile' }, { title: 'Judge' }, { title: 'Route' }],
}

const A = args || {}
const module_ = A.module || 'm04-first-model-mlp'
const day     = A.day    || 'day-01-mlp-mnist'
const MAX_ROUNDS = A.maxRounds || 3
const src    = `sessions/${module_}/${day}/source.md`
const lesson = `sessions/${module_}/${day}/lesson.html`
const pdir   = `portfolio/${module_}/${day}`
const doing  = `sessions/${module_}/${day}/experiment.py`

const PRODUCE_SCHEMA = {
  type: 'object',
  properties: {
    wrote: { type: 'boolean' },
    ran_ok: { type: 'boolean' },
    gate_ok: { type: 'boolean' },
    gate_output: { type: 'string' },
    compiled: { type: 'boolean' },
    output_tail: { type: 'string' },
    claim: { type: 'string' },
  },
  required: ['wrote', 'ran_ok', 'gate_ok', 'compiled'],
}

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['STRONG', 'OK', 'WEAK'] },
    numbers_match: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'object',
      properties: { axis: { type: 'string' }, severity: { type: 'string' }, why: { type: 'string' }, fix: { type: 'string' } },
      required: ['axis', 'severity', 'why'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'numbers_match', 'findings'],
}

// --- Produce: ONE write-capable agent writes+runs experiment.py + blog.md, then compiles the page
phase('Produce')
async function produce(round, findings) {
  const fb = findings
    ? `\n\nFIX ROUND ${round}. Address these evidence-judge findings (regenerate blog/experiment as needed; EVERY number in blog.md MUST come from the real experiment_out.txt):\n${JSON.stringify(findings, null, 2)}`
    : ''
  return await agent(
    `You are the EVIDENCE-PRODUCER sub-agent — Reader-B / frontier-staff register (NOT beginner). You own this day's evidence.
The lesson at ${src} + ${lesson} has PASSED. Produce its frontier-facing evidence into ${pdir}/ :
1. Write ${pdir}/experiment.py — a SMALL, self-contained, REAL python3 script demonstrating ONE concrete claim from the lesson (e.g. a from-scratch numerical check, a training/loss curve, a benchmark). It MUST print its key result to stdout, plus the shapes and intermediate values that make the claim readable.
  SEPARATE ARTIFACT: ${pdir}/experiment.py is NOT ${doing} (the learner's DOING leg, authored from the day's produce spec). Neither substitutes for the other — do not copy, move, or overwrite the sessions file, and do not treat an existing sessions artifact as discharging this one.
  SHAPE — write it so the gate in step 3 can judge it: an \`if __name__ == "__main__":\` block, one boolean per claim, one \`assert\` per claim with a message, and a printed \`✅\` on success. Print \`❌ not yet — expected …\` ONLY on the failure branch: the gate fails on ANY \`❌\` in stdout.
  DETERMINISTIC + OFFLINE: seed every source of randomness (numpy / torch / random) and never touch the network — no downloads, no HF hub fetch, no dataset pull; synthesize a seeded stand-in and say so in a comment at the point of use. The gate RE-RUNS the script with sockets blocked, so a script that is non-deterministic or online either fails outright or prints numbers that no longer match blog.md.
  It MAY savefig a PNG into ${pdir}/assets/ (it is a standalone script, NOT a notebook, so savefig is allowed — this is the ONE place the portfolio artifact is deliberately looser than the learner-facing rules in sessions/_experiment_env.md). Resolve \`assets/\` from the SCRIPT, never from the cwd — \`HERE = os.path.dirname(os.path.abspath(__file__)); ASSETS = os.path.join(HERE, 'assets'); os.makedirs(ASSETS, exist_ok=True)\` — because the two runs use DIFFERENT working directories: step 2 runs it from the repo root, while the step-3 gate runs it with cwd set to the script's own directory. A bare \`'assets'\` therefore writes the PNG to the repo root in step 2, where evidence_compile cannot find it. Do not require a display; MPLBACKEND=Agg is forced.
  THE SELF-CHECK MUST BE ABLE TO FAIL. Pin the claim against a value WRITTEN DOWN in the check (prefer an exact \`round(x, 4) == <literal>\`), never against one re-derived from the code path under test; compute any "predicted" value from the inputs instead of hardcoding it; assert exception TYPES, never a library's error wording; and choose shapes and values where a wrong spelling gives a DIFFERENT answer — symmetric test data, a uniform output, a zero-initialised parameter, floor division, and a threshold with nothing sitting on the boundary each make their own code path untestable. The failure classes are circular, too weak to see the bug, fake prediction, coupled to library wording, entailed clause, constant fold, dead branch, self-derived tautology, identity/scale-invariant claim; read sessions/_compiler/AUTHORING.md section 10 ("A passing ✅ is not evidence the check means anything") for the catalogue with worked examples BEFORE writing the check.
2. RUN it: first \`mkdir -p ${pdir}/assets\`, then \`python3 ${pdir}/experiment.py > ${pdir}/experiment_out.txt 2>&1\`. Confirm it exits 0; if it errors, FIX the script and re-run until exit 0. Use system python3.
3. GATE it: \`python3 sessions/_experiment_check.py ${pdir}/experiment.py\` (exit 0). It runs the contract (parses, >=1 import, a \`__main__\` guard, an assert, a ✅/❌ print) and THEN executes the script under a socket-blocking shim, requiring exit 0, a ✅, no ❌, and completion inside 180s. A gate failure is a P0: fix the script, re-run step 2 so experiment_out.txt matches the gated run, then gate again. Do NOT continue with gate_ok false.
4. Write ${pdir}/blog.md — a staff-depth technical write-up (the mechanism + at least one failure mode + one design trade-off) that EMBEDS the REAL numbers from experiment_out.txt. NO fabricated figures — a number not in the run output is a fabrication the judge will catch. Repurpose the lesson's staff-depth content; do not re-teach it beginner-style. Open with a \`# \` title.
5. Assemble the self-contained page: \`python3 sessions/_compiler/evidence_compile.py ${module_} ${day}\` (exit 0).
Return: wrote (all files written), ran_ok (experiment.py exited 0), gate_ok (_experiment_check.py exited 0), gate_output (tail of the gate output), compiled (evidence_compile exited 0), output_tail (last ~15 lines of experiment_out.txt), claim (the one claim in a sentence).${fb}`,
    { label: `produce r${round}`, phase: 'Produce', schema: PRODUCE_SCHEMA, agentType: 'general-purpose' })
}

let prod = await produce(0, null)
log(`produce r0: wrote=${prod.wrote} ran_ok=${prod.ran_ok} gate_ok=${prod.gate_ok} compiled=${prod.compiled}`)

// --- Judge: read-only, frontier-staff bar + numbers_match
async function judge() {
  return await agent(
    `You are the EVIDENCE-JUDGE (read-only). Run: python3 sessions/_compiler/evidence_judge.py ${pdir}. Parse the "Evidence Judge" panel and report verdict (STRONG|OK|WEAK), numbers_match (bool), findings[] (axis/severity/why/fix), summary. If the bridge is unavailable, return verdict OK + numbers_match true and note the fallback.`,
    { label: 'evidence-judge', phase: 'Judge', schema: JUDGE_SCHEMA })
}

// --- the bounded loop (produce <-> judge)
let round = 0, verdict = null
while (round < MAX_ROUNDS) {
  if (!prod.ran_ok || !prod.gate_ok || !prod.compiled) {
    log(`r${round}: experiment/gate/compile did not succeed -> regenerate`)
    round += 1
    prod = await produce(round, [{ axis: 'reproducibility', severity: 'P0', why: 'experiment.py did not exit 0, sessions/_experiment_check.py did not pass, or evidence_compile failed', fix: prod.gate_output || prod.output_tail || 'fix the script until it runs AND passes python3 sessions/_experiment_check.py' }])
    continue
  }
  phase('Judge')
  verdict = await judge()
  phase('Route')
  const pass = verdict.numbers_match && verdict.verdict !== 'WEAK'
  log(`r${round}: verdict=${verdict.verdict} numbers_match=${verdict.numbers_match} findings=${verdict.findings.length} pass=${pass}`)
  if (pass) break
  round += 1
  if (round >= MAX_ROUNDS) break
  const extra = verdict.numbers_match ? [] : [{ axis: 'numbers_match', severity: 'P0', why: 'a blog number is NOT supported by experiment_out.txt', fix: 'make every blog figure come from the real run' }]
  prod = await produce(round, verdict.findings.concat(extra))
}

const converged = !!(verdict && verdict.numbers_match && verdict.verdict !== 'WEAK' && prod.ran_ok && prod.gate_ok && prod.compiled)
if (!converged) log(`NOT converged after ${round} rounds — evidence flagged at checkpoint`)

const report = [
  `# Evidence build report — ${module_}/${day}`,
  ``,
  `- Converged: ${converged}  (rounds: ${round}/${MAX_ROUNDS})`,
  `- Experiment ran_ok: ${prod.ran_ok} | _experiment_check gate: ${prod.gate_ok} | page compiled: ${prod.compiled}`,
  verdict ? `- Judge: verdict=${verdict.verdict}, numbers_match=${verdict.numbers_match}` : `- Judge: (never reached — experiment/gate/compile failed)`,
  !prod.ran_ok || !prod.gate_ok || !prod.compiled ? `\n## Blocker\n\`\`\`\n${(prod.gate_output || prod.output_tail || '(no output captured)').slice(-1500)}\n\`\`\`` : '',
  verdict && verdict.findings.length ? `\n## Findings\n${verdict.findings.map(f => `- [${f.severity}/${f.axis}] ${f.why}`).join('\n')}` : `\n(no findings)`,
  `\nArtifact: ${pdir}/index.html`,
].join('\n')
log(report)

return { module: module_, day, pdir, converged, rounds: round, verdict, produce: prod, report }
