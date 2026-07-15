export const meta = {
  name: 'evidence-build',
  description: 'Per-day frontier evidence: producer writes+runs a REAL experiment.py + blog.md (reuses the lesson viz) -> evidence_compile assembles a self-contained portfolio page -> evidence-judge (frontier-staff bar + numbers_match) -> bounded loop -> checkpoint. args {module, day, maxRounds}.',
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

const PRODUCE_SCHEMA = {
  type: 'object',
  properties: {
    wrote: { type: 'boolean' },
    ran_ok: { type: 'boolean' },
    compiled: { type: 'boolean' },
    output_tail: { type: 'string' },
    claim: { type: 'string' },
  },
  required: ['wrote', 'ran_ok', 'compiled'],
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
1. Write ${pdir}/experiment.py — a SMALL, self-contained, REAL python3 script demonstrating ONE concrete claim from the lesson (e.g. a from-scratch numerical check, a training/loss curve, a benchmark). It MAY savefig a PNG into ${pdir}/assets/ (it is a standalone script, NOT a notebook, so savefig is allowed). It MUST print its key result to stdout.
2. RUN it: first \`mkdir -p ${pdir}/assets\`, then \`python3 ${pdir}/experiment.py > ${pdir}/experiment_out.txt 2>&1\`. Confirm it exits 0; if it errors, FIX the script and re-run until exit 0. Use system python3.
3. Write ${pdir}/blog.md — a staff-depth technical write-up (the mechanism + at least one failure mode + one design trade-off) that EMBEDS the REAL numbers from experiment_out.txt. NO fabricated figures — a number not in the run output is a fabrication the judge will catch. Repurpose the lesson's staff-depth content; do not re-teach it beginner-style. Open with a \`# \` title.
4. Assemble the self-contained page: \`python3 sessions/_compiler/evidence_compile.py ${module_} ${day}\` (exit 0).
Return: wrote (all files written), ran_ok (experiment.py exited 0), compiled (evidence_compile exited 0), output_tail (last ~15 lines of experiment_out.txt), claim (the one claim in a sentence).${fb}`,
    { label: `produce r${round}`, phase: 'Produce', schema: PRODUCE_SCHEMA, agentType: 'general-purpose' })
}

let prod = await produce(0, null)
log(`produce r0: wrote=${prod.wrote} ran_ok=${prod.ran_ok} compiled=${prod.compiled}`)

// --- Judge: read-only, frontier-staff bar + numbers_match
async function judge() {
  return await agent(
    `You are the EVIDENCE-JUDGE (read-only). Run: python3 sessions/_compiler/evidence_judge.py ${pdir}. Parse the "Evidence Judge" panel and report verdict (STRONG|OK|WEAK), numbers_match (bool), findings[] (axis/severity/why/fix), summary. If the bridge is unavailable, return verdict OK + numbers_match true and note the fallback.`,
    { label: 'evidence-judge', phase: 'Judge', schema: JUDGE_SCHEMA })
}

// --- the bounded loop (produce <-> judge)
let round = 0, verdict = null
while (round < MAX_ROUNDS) {
  if (!prod.ran_ok || !prod.compiled) {
    log(`r${round}: experiment/compile did not succeed -> regenerate`)
    round += 1
    prod = await produce(round, [{ axis: 'reproducibility', severity: 'P0', why: 'experiment.py did not exit 0 or evidence_compile failed', fix: prod.output_tail || 'fix the script until it runs' }])
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

const converged = !!(verdict && verdict.numbers_match && verdict.verdict !== 'WEAK' && prod.ran_ok && prod.compiled)
if (!converged) log(`NOT converged after ${round} rounds — evidence flagged at checkpoint`)

const report = [
  `# Evidence build report — ${module_}/${day}`,
  ``,
  `- Converged: ${converged}  (rounds: ${round}/${MAX_ROUNDS})`,
  `- Experiment ran_ok: ${prod.ran_ok} | page compiled: ${prod.compiled}`,
  verdict ? `- Judge: verdict=${verdict.verdict}, numbers_match=${verdict.numbers_match}` : `- Judge: (never reached — experiment/compile failed)`,
  !prod.ran_ok || !prod.compiled ? `\n## Blocker\n\`\`\`\n${(prod.output_tail || '(no output captured)').slice(-1500)}\n\`\`\`` : '',
  verdict && verdict.findings.length ? `\n## Findings\n${verdict.findings.map(f => `- [${f.severity}/${f.axis}] ${f.why}`).join('\n')}` : `\n(no findings)`,
  `\nArtifact: ${pdir}/index.html`,
].join('\n')
log(report)

return { module: module_, day, pdir, converged, rounds: round, verdict, produce: prod, report }
