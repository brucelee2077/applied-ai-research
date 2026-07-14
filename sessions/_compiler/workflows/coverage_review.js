export const meta = {
  name: 'coverage-review',
  description: 'Skill-driven, LLM-judged coverage review for one lesson-day: blind spec-draft from the architect skill (no notebook) → extract notebook concepts → judge coverage + intuition → route gaps. Read-only (reports; never writes).',
  whenToUse: 'After (re)building a lesson day, to verify the SKILL-DRAFTED coverage is complete and genuinely taught. Reusable for any module via args {topic, source, lesson, notebook}. Skills are the source of truth; the notebook is a held-out test.',
  phases: [
    { title: 'Draft',      detail: 'blind spec-drafter: architect Coverage Spec Rule, NO notebook' },
    { title: 'Oracle',     detail: 'extract the concepts the notebook (test oracle) teaches' },
    { title: 'Judge',      detail: 'LLM judge: spec vs notebook (skill gaps) + lesson (exec/intuition)' },
    { title: 'Synthesize', detail: 'route findings — skill gap vs exec gap vs curation' },
  ],
}

// -------------------------------------------------------------------------
// Reusable coverage-review workflow.  Skills define the sub-agent roles
// (see frontier-refactor-qa "Coverage Review Workflow").  Parameterize per
// module via args; defaults target m02 Day 2 (the proof case).
//   args = { topic, source, lesson, notebook }
// -------------------------------------------------------------------------
const A = args || {}
const topic    = A.topic    || 'activation functions'
const source   = A.source   || 'sessions/m02-the-neuron/day-02-activations/source.md'
const lesson   = A.lesson   || 'sessions/m02-the-neuron/day-02-activations/lesson.html'
const notebook = A.notebook || '00-neural-networks/fundamentals/03_activation_functions.ipynb'

const SPEC_SCHEMA = {
  type: 'object',
  properties: {
    covers:       { type: 'array', items: { type: 'string' } },
    deferred:     { type: 'array', items: { type: 'object',
                      properties: { topic: { type: 'string' }, where: { type: 'string' } },
                      required: ['topic', 'where'] } },
    out_of_scope: { type: 'array', items: { type: 'object',
                      properties: { topic: { type: 'string' }, reason: { type: 'string' } },
                      required: ['topic', 'reason'] } },
    reasoning:    { type: 'string' },
  },
  required: ['covers', 'deferred', 'out_of_scope', 'reasoning'],
}

const ORACLE_SCHEMA = {
  type: 'object',
  properties: { concepts: { type: 'array', items: { type: 'string' } } },
  required: ['concepts'],
}

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    skill_gaps:       { type: 'array', items: { type: 'object',
                          properties: { concept: { type: 'string' }, why: { type: 'string' } },
                          required: ['concept', 'why'] } },
    exec_gaps:        { type: 'array', items: { type: 'object',
                          properties: { concept: { type: 'string' }, why: { type: 'string' } },
                          required: ['concept', 'why'] } },
    intuition_issues: { type: 'array', items: { type: 'object',
                          properties: { concept: { type: 'string' }, issue: { type: 'string' } },
                          required: ['concept', 'issue'] } },
    verdict:          { type: 'string', enum: ['PASS', 'GAPS'] },
    summary:          { type: 'string' },
  },
  required: ['skill_gaps', 'exec_gaps', 'intuition_issues', 'verdict', 'summary'],
}

// --- Phase 1: BLIND spec draft — the proof the skill drives coverage --------
phase('Draft')
const draft = await agent(
  `You are the coverage SPEC-DRAFTER sub-agent.
Read ONLY the "Coverage Spec Rule" section of .claude/skills/frontier-curriculum-architect/SKILL.md.
Do NOT read any notebook and do NOT read the existing lesson — this is a blind draft.
Using that rule and your own domain knowledge, draft the coverage spec for a BEGINNER lesson on "${topic}".
Apply every rung of the rule: (1) core mechanism family; (2) historical ancestor IF it motivates the
modern form; (3) for EVERY failure mode you would teach, its REMEDY (a disease with no cure is an
incomplete spec); (4) capability limits; (5) forward-pointers -> deferred{topic,where};
(6) out-of-scope{topic,reason} (a remedy for a failure you teach can never be out of scope).
Return covers (concept names), deferred, out_of_scope, and one-paragraph reasoning.`,
  { label: 'spec-draft (blind, no notebook)', phase: 'Draft', schema: SPEC_SCHEMA })

// --- Phase 2: notebook concepts (the held-out test oracle) ------------------
phase('Oracle')
const oracle = await agent(
  `You are the TEST-ORACLE sub-agent. Read the notebook at ${notebook}.
List the distinct ML CONCEPTS it teaches about "${topic}". Include first-class concepts and named
techniques (e.g. specific activation functions, failure modes, remedies). EXCLUDE analogy labels
(e.g. "light switch", "signal processing") and pedagogical chrome ("When to Use This", "Key Takeaways",
"Visualizing X", "What's Next"). Return concept names only.`,
  { label: 'oracle: notebook concepts', phase: 'Oracle', schema: ORACLE_SCHEMA })

// --- Phase 3: LLM judge — spec vs oracle (skill gaps) + lesson (execution) --
phase('Judge')
const verdict = await agent(
  `You are the COVERAGE JUDGE sub-agent. Be skeptical: a concept named in one clause is MENTIONED,
not TAUGHT. Use TWO DIFFERENT references — do not conflate them:

BLIND-DRAFTED SPEC (from the skill, without the notebook) — used ONLY to test the SKILL's completeness:
${JSON.stringify(draft, null, 2)}

NOTEBOOK CONCEPTS (held-out test oracle):
${JSON.stringify(oracle.concepts, null, 2)}

COMMITTED SPEC: read the module manifest for this day (derive it from ${source}: the file
sessions/<module>/_refactor/manifest.yaml, key coverage.<day>.covers / deferred / out_of_scope).
The committed spec — NOT the blind draft — is what the lesson is meant to realize.

Now read the compiled lesson at ${lesson} (strip HTML to text) and report:
- skill_gaps: notebook concepts ABSENT from the BLIND-DRAFTED spec's covers/deferred/out_of_scope
  (this scores the ARCHITECT skill's derivation). A concept that the COMMITTED manifest already
  defers or scopes out is NOT a skill gap — the skill just under-enumerated a correctly-deferred
  long-tail; note it but do not treat it as a hole.
- exec_gaps: concepts in the COMMITTED manifest 'covers' that are NOT genuinely taught in the lesson
  (mentioned-only counts). Judge execution against the COMMITTED spec, never against the blind draft.
- intuition_issues: committed 'covers' concepts NOT introduced intuition-first (formula/notation
  before a felt picture).
verdict = PASS iff exec_gaps is empty AND every skill_gap is something the committed manifest does
NOT already defer/scope-out.`,
  { label: 'judge: committed-spec exec + blind-draft skill test', phase: 'Judge', schema: JUDGE_SCHEMA })

// --- Phase 4: synthesize + route --------------------------------------------
phase('Synthesize')
const covers = (draft.covers || []).map(c => String(c).toLowerCase())
const blindHasStep  = covers.some(c => c.includes('step'))
const blindHasLeaky = covers.some(c => c.includes('leaky'))

const routing = {
  skill_fixes: verdict.skill_gaps,        // -> frontier-curriculum-architect Coverage Spec Rule
  lesson_fixes: [...verdict.exec_gaps, ...verdict.intuition_issues], // -> frontier-lesson-builder
  proof_skill_drives_coverage: {
    blind_draft_reproduced_step_ancestor: blindHasStep,
    blind_draft_reproduced_leaky_remedy: blindHasLeaky,
    note: 'If both true, the skill produced the ancestor + the failure-remedy WITHOUT reading the notebook.',
  },
}

log(`coverage-review "${topic}": verdict=${verdict.verdict}; ` +
    `skill_gaps=${verdict.skill_gaps.length}; exec_gaps=${verdict.exec_gaps.length}; ` +
    `blind-draft step=${blindHasStep} leaky=${blindHasLeaky}`)

return { topic, source, lesson, notebook, blind_drafted_spec: draft,
         notebook_concepts: oracle.concepts, verdict, routing }
