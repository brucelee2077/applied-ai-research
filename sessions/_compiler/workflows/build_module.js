export const meta = {
  name: 'build-module',
  description: 'Run the lesson_build engine across a list of days for one module, sequentially; report per-day convergence. args {module, days:[...], maxRounds}.',
  whenToUse: 'To (re)generate multiple days of a module into V9 concept structure in one background run.',
  phases: [{ title: 'Build days' }],
}

const A = args || {}
const module_ = A.module || 'm04-first-model-mlp'
const days = A.days || []
const maxRounds = A.maxRounds || 3
const LB = '/Users/ruifengli/Desktop/applied-ai-research/sessions/_compiler/workflows/lesson_build.js'

phase('Build days')
const results = []
for (const day of days) {
  log(`=== building ${module_}/${day} ===`)
  let r = null, err = null
  try {
    r = await workflow({ scriptPath: LB }, { module: module_, day, maxRounds })
  } catch (e) {
    err = String(e)
  }
  results.push({
    day,
    converged: !!(r && r.converged),
    rounds: r ? r.rounds : null,
    residual_p0: (r && r.routing) ? r.routing.p0.length : null,
    skill_proposal: !!(r && r.skill_proposal),
    concepts: (r && r.final_compile) ? r.final_compile.concept_count : null,
    error: err,
    // Copied through so the module summary can report the Chinese outcome. The
    // first version read r.zh_status off THIS object, which never carried it, so
    // the summary always said "Chinese 0 converged, 0 not required".
    zh_status: r && r.zh_status,
    zh_rounds: r && r.zh_rounds,
  })
  log(`=== ${day}: converged=${!!(r && r.converged)} rounds=${r ? r.rounds : '?'} concepts=${(r && r.final_compile) ? r.final_compile.concept_count : '?'} ===`)
}

const done = results.filter(x => x.converged).length
const zhDone = results.filter(r => r && r.zh_status === 'converged').length
const zhSkip = results.filter(r => r && r.zh_status === 'not-required').length
log(`build-module ${module_}: ${done}/${results.length} days converged; Chinese ${zhDone} converged, ${zhSkip} not required`)
if (zhSkip === results.length && results.length) {
  log(`NOTE: no day produced a Chinese twin because this module never opted in. Add zh.require to sessions/${module_}/_refactor/manifest.yaml to make the build bilingual.`)
}
return { module: module_, results, converged_count: done, total: results.length }
