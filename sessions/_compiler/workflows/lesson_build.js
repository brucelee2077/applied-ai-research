export const meta = {
  name: 'lesson-build',
  description: 'Self-correcting per-lesson engine: draft coverage (skill-blind) -> author V9 concept source -> compile -> judge panel -> regenerate until pass -> checkpoint. args {module, day}. Writes source.md; compiles; reports.',
  whenToUse: 'To (re)generate one lesson-day into the V9 concept structure with an autonomous judge-gated fix loop.',
  phases: [
    { title: 'Coverage' },
    { title: 'Author' },
    { title: 'Compile' },
    { title: 'Evaluate' },
    { title: 'Route' },
  ],
}

const A = args || {}
const module_ = A.module || 'm04-first-model-mlp'
const day     = A.day    || 'day-01-mlp-mnist'
const source  = `sessions/${module_}/${day}/source.md`
const lesson  = `sessions/${module_}/${day}/lesson.html`
const MAX_ROUNDS = A.maxRounds || 3

const SPEC_SCHEMA = {
  type: 'object',
  properties: {
    covers: { type: 'array', items: { type: 'string' } },
    deferred: { type: 'array', items: { type: 'object',
      properties: { topic: { type: 'string' }, where: { type: 'string' } }, required: ['topic', 'where'] } },
    out_of_scope: { type: 'array', items: { type: 'object',
      properties: { topic: { type: 'string' }, reason: { type: 'string' } }, required: ['topic', 'reason'] } },
    reasoning: { type: 'string' },
  },
  required: ['covers', 'deferred', 'out_of_scope', 'reasoning'],
}

const COMPILE_SCHEMA = {
  type: 'object',
  properties: {
    wrote_source: { type: 'boolean' },
    compiled: { type: 'boolean' },
    compile_exit_code: { type: 'integer' },
    gate_output: { type: 'string' },
    concept_count: { type: 'integer' },
  },
  required: ['wrote_source', 'compiled', 'compile_exit_code', 'gate_output'],
}

// --- Phase 1: blind coverage draft (reused role from coverage_review.js) ----
phase('Coverage')
const draft = await agent(
  `You are the coverage SPEC-DRAFTER sub-agent.
Read ONLY the "Coverage Spec Rule" section of .claude/skills/frontier-curriculum-architect/SKILL.md.
Do NOT read any notebook or existing lesson — blind draft.
Draft the coverage spec for a BEGINNER lesson for module "${module_}", day "${day}".
Apply every rung: core mechanism family; historical ancestor when it motivates the modern form;
for EVERY failure mode its REMEDY; capability limits; forward-pointers -> deferred; out_of_scope with reason.
Return covers, deferred, out_of_scope, and one-paragraph reasoning.`,
  { label: 'spec-draft (blind)', phase: 'Coverage', schema: SPEC_SCHEMA })

// Reconcile happens deterministically after we can read the manifest; for the
// happy path we pass the blind draft to the author as the working spec and let
// the committed manifest (read by the author) be authoritative.

// --- Phase 2+3: author writes V9 concept source, then compiles ---------------
phase('Author')
async function authorAndCompile(round, findings) {
  const findingsBlock = findings
    ? `\n\nThis is FIX ROUND ${round}. FULLY REGENERATE the lesson (do not patch) addressing these findings:\n${JSON.stringify(findings, null, 2)}`
    : ''
  return await agent(
    `You are the AUTHOR sub-agent — the ONLY writer for this lesson; you own its voice end to end.
Author the V9 concept-mode lesson at ${source} for module "${module_}", day "${day}".
Follow the authoring grammar in sessions/_compiler/AUTHORING.md EXACTLY (mode: concept; @@@ hero/concept/quiz/produce/fin; %%% svg|viz|demo|quiz widgets).
BEGINNER-FRIENDLINESS IS THE #1 REQUIREMENT (this lesson is for a curious 12-year-old for whom English may be a second language — repo CLAUDE.md §5/§7 + frontier-lesson-builder's Beginner Intuition Register). For EVERY concept unit, IN THIS ORDER: (1) OPEN with a plain-words intuition + a CONCRETE everyday analogy (a physical, experienced thing a kid knows) INCLUDING where it breaks down — BEFORE any formula, notation, or undefined jargon; (2) its OWN inline visual; (3) a step-by-step build-up. Warm "brilliant friend" voice, one idea per sentence, define every term before first use, normalize confusion, victory laps. A companion notebook exists (see notebook_yardstick) — MATCH its warmth and analogy density; it is the gold standard. Litmus test: if a sentence sounds like a textbook or an interview answer, rewrite it for a 12-year-old.
Coverage to realize (committed manifest is authoritative; this blind draft is guidance):
${JSON.stringify(draft, null, 2)}
Write source.md INCREMENTALLY — do NOT compose the whole lesson silently and write it in one shot (that stalls the agent on dense days). Instead: FIRST create ${source} with the front-matter + the @@@ hero block, THEN append each @@@ concept unit one at a time with repeated edits, THEN append @@@ quiz / @@@ produce / @@@ fin. Keep making steady tool-call progress. When the file is complete, compile and report:
  python3 sessions/_compiler/compile_lesson.py ${source}
Also run: python3 sessions/_compiler/gates/concept_structure_gate.py ${source}
Return: wrote_source; compiled = TRUE only if BOTH commands exit 0 (if EITHER compile_lesson.py OR concept_structure_gate.py fails, set compiled=false and put both outputs in gate_output); compile_exit_code (compile_lesson.py's); gate_output (tail of BOTH commands); concept_count.${findingsBlock}`,
    { label: `author r${round}`, phase: 'Author', schema: COMPILE_SCHEMA, agentType: 'general-purpose' })
}

phase('Compile')
// A.seedFindings lets a "polish round" re-run seed round-0 with known findings
// (e.g. P1s surfaced at a prior checkpoint the user chose to fix). Null on a fresh run.
let compileRes = await authorAndCompile(0, (A.seedFindings && A.seedFindings.length) ? A.seedFindings : null)
log(`author r0: compiled=${compileRes.compiled} exit=${compileRes.compile_exit_code} concepts=${compileRes.concept_count}`)

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object',
      properties: {
        concept: { type: 'string' },
        kind: { type: 'string', description: 'exec_gap | intuition | analogy | buildup | tone | correctness | skill_gap' },
        severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
        why: { type: 'string' }, fix: { type: 'string' },
      }, required: ['kind', 'severity', 'why'] } },
    verdict: { type: 'string', enum: ['PASS', 'GAPS'] },
    lens: { type: 'string' },
  },
  required: ['findings', 'verdict', 'lens'],
}

const LENSES = [
  { key: 'coverage', prompt: `Run: python3 sessions/_compiler/gates/coverage_judge.py ${lesson} --source ${source}. Parse the "Coverage Judge" section: report each MENTIONED/ABSENT spec concept as an exec_gap finding (P0). ALSO parse its "skill gaps (notebook teaches; spec missed)" subsection: report EACH listed concept as a finding with kind="skill_gap" (severity P1) — these route to a user-approved skill proposal, NOT the author. If the bridge is unavailable, say so with verdict PASS (structural fallback) and note it.` },
  { key: 'tone', prompt: `Run: python3 sessions/_compiler/gates/coverage_judge.py ${lesson} --source ${source}. Parse the "Beginner-Friendliness Judge" section. If overall is BELOW_NOTEBOOK or WORSE_THAN_NOTEBOOK, emit ONE P0 tone finding "lesson is not as beginner-friendly as the notebook" (use the top_fixes as its fix); ALSO report each BELOW/WORSE dimension as a P1 tone finding. Bridge unavailable -> verdict PASS, note it.` },
  { key: 'structure', prompt: `Run: python3 sessions/_compiler/gates/coverage_judge.py ${lesson} --source ${source}. Parse the "Concept-Structure Judge" section (per-concept intuition_first / analogy / buildup). Beginner-friendliness is the bar, so grade strictly: report analogy MISSING or WEAK as a P0 analogy finding (every concept needs a concrete everyday analogy WITH where-it-breaks-down); intuition_first MISSING as P0, WEAK as P1; buildup MISSING/WEAK as P1. Bridge unavailable -> verdict PASS, note it.` },
  { key: 'correctness', prompt: `Adversarially read ${lesson} (strip HTML) for TECHNICAL errors, numeric self-inconsistency, and broken narrative spine. Report each as a correctness finding (P0). Default to reporting if unsure.` },
]

async function evaluate() {
  const results = await parallel(LENSES.map(l => () =>
    agent(`You are the ${l.key.toUpperCase()} evaluator (read-only). ${l.prompt}
Return findings[], a verdict (PASS iff no P0), and lens="${l.key}".`,
      { label: `judge:${l.key}`, phase: 'Evaluate', schema: JUDGE_SCHEMA })))
  return results.filter(Boolean)
}

// deterministic router: split loop-back findings from skill-gap escalations
function route(evals) {
  const all = evals.flatMap(e => (e.findings || []).map(f => ({ ...f, lens: e.lens })))
  const skillGaps = all.filter(f => f.kind === 'skill_gap')
  const fixable = all.filter(f => f.kind !== 'skill_gap')
  const p0 = fixable.filter(f => f.severity === 'P0')
  return { all, skillGaps, fixable, p0, pass: p0.length === 0 }
}

// --- the self-correcting loop (2 <-> 4) -------------------------------------
let round = 0, routing = null, lastEvals = []
while (round < MAX_ROUNDS) {
  // hard-gate failure short-circuits the LLM panel: loop straight back to author
  if (!compileRes.compiled) {
    log(`r${round}: hard gate failed (exit ${compileRes.compile_exit_code}) -> regenerate`)
    round += 1
    compileRes = await authorAndCompile(round, [{ kind: 'compile_gate', severity: 'P0', why: compileRes.gate_output }])
    continue
  }
  phase('Evaluate')
  lastEvals = await evaluate()
  phase('Route')
  routing = route(lastEvals)
  log(`r${round}: P0=${routing.p0.length} fixable=${routing.fixable.length} skill_gaps=${routing.skillGaps.length} pass=${routing.pass}`)
  if (routing.pass) break
  round += 1
  if (round >= MAX_ROUNDS) break
  compileRes = await authorAndCompile(round, routing.fixable)
}

const converged = !!(routing && routing.pass && compileRes.compiled)
if (!converged) log(`NOT converged after ${round} rounds — blocker report at checkpoint`)

let skillProposal = null
if (routing && routing.skillGaps.length) {
  phase('Route')
  skillProposal = await agent(
    `You are the SKILL-GAP PROPOSER (read-only; you propose, you do NOT edit any skill).
The blind coverage draft + judges found concepts the committed spec/lesson lack, that are NOT correctly deferred/out-of-scope:
${JSON.stringify(routing.skillGaps, null, 2)}
Read .claude/skills/frontier-curriculum-architect/SKILL.md "Coverage Spec Rule".
Draft the SMALLEST concrete edit (a unified-diff-style before/after snippet) to that rule and/or the manifest coverage.<day> that would make the blind draft reproduce these concepts next time. Explain in one paragraph why. Output the proposal text only — it will be shown to the user for approval.`,
    { label: 'skill-gap proposal', phase: 'Route', schema: {
      type: 'object',
      properties: { proposal_diff: { type: 'string' }, rationale: { type: 'string' }, targets: { type: 'array', items: { type: 'string' } } },
      required: ['proposal_diff', 'rationale'],
    } })
}

const report = [
  `# Lesson build report — ${module_}/${day}`,
  ``,
  `- Converged: ${converged}  (rounds: ${round}/${MAX_ROUNDS})`,
  `- Final compile: exit ${compileRes.compile_exit_code}, ${compileRes.concept_count || '?'} concepts`,
  !compileRes.compiled
    ? `\n## Hard-gate blocker (lesson never compiled — the LLM judge panel did NOT run)\n\`\`\`\n${(compileRes.gate_output || '(no gate output captured)').slice(-2000)}\n\`\`\``
    : '',
  `- Residual P0 (if any): ${routing ? routing.p0.length : 'n/a (lesson never compiled)'}`,
  routing && routing.p0.length ? `\n## Residual findings\n${routing.p0.map(f => `- [${f.severity}/${f.lens}] ${f.kind}: ${f.why}`).join('\n')}` : `\n(no residual P0)`,
  routing && routing.fixable.filter(f => f.severity !== 'P0').length ? `\n## Advisory findings (P1/P2 — surfaced, not looped this run)\n${routing.fixable.filter(f => f.severity !== 'P0').map(f => `- [${f.severity}/${f.lens}] ${f.kind}${f.concept ? ' (' + f.concept + ')' : ''}: ${f.why}`).join('\n')}` : '',
  skillProposal ? `\n## Skill-gap proposal (needs your approval)\n${skillProposal.rationale}\n\n\`\`\`diff\n${skillProposal.proposal_diff}\n\`\`\`` : `\n(no skill-gap proposals)`,
].join('\n')

log(report)

return {
  module: module_, day, source, lesson, converged, rounds: round,
  blind_draft: draft, final_compile: compileRes,
  evaluations: lastEvals, routing, skill_proposal: skillProposal, report,
}
