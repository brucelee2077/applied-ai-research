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
  })
  log(`=== ${day}: converged=${!!(r && r.converged)} rounds=${r ? r.rounds : '?'} concepts=${(r && r.final_compile) ? r.final_compile.concept_count : '?'} ===`)
}

const done = results.filter(x => x.converged).length
log(`build-module ${module_}: ${done}/${results.length} days converged`)
return { module: module_, results, converged_count: done, total: results.length }
