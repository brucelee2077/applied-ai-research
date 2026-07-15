export const meta = {
  name: 'build-evidence',
  description: 'Run the evidence_build engine across a list of days for one module, sequentially; report per-day convergence. args {module, days:[...], maxRounds}.',
  whenToUse: 'After lessons pass, to produce frontier evidence for multiple days in one background run.',
  phases: [{ title: 'Build evidence' }],
}

const A = args || {}
const module_ = A.module || 'm04-first-model-mlp'
const days = A.days || []
const maxRounds = A.maxRounds || 3
const EB = '/Users/ruifengli/Desktop/applied-ai-research/sessions/_compiler/workflows/evidence_build.js'

phase('Build evidence')
const results = []
for (const day of days) {
  log(`=== evidence for ${module_}/${day} ===`)
  let r = null, err = null
  try {
    r = await workflow({ scriptPath: EB }, { module: module_, day, maxRounds })
  } catch (e) {
    err = String(e)
  }
  results.push({
    day,
    converged: !!(r && r.converged),
    rounds: r ? r.rounds : null,
    verdict: (r && r.verdict) ? r.verdict.verdict : null,
    numbers_match: (r && r.verdict) ? r.verdict.numbers_match : null,
    error: err,
  })
  log(`=== ${day}: converged=${!!(r && r.converged)} verdict=${(r && r.verdict) ? r.verdict.verdict : '?'} ===`)
}
const done = results.filter(x => x.converged).length
log(`build-evidence ${module_}: ${done}/${results.length} days converged`)
return { module: module_, results, converged_count: done, total: results.length }
