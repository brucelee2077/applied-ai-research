export const meta = {
  name: 'quiz-depth',
  description: 'Upgrade quizzes of interview-critical concept lessons to scenario/reasoning questions (staff-interview flavor); each agent writes a JSON sidecar of 4 questions.',
  phases: [{ title: 'Quiz' }],
}
const ROOT=(args&&args.root)||'.';
const ITEMS=[{"path": "week-02/day-01-rooflines.html", "sidecar": "sessions/_quiz_steps/week-02_day-01-rooflines_html.json"}, {"path": "week-02/day-03-sharding.html", "sidecar": "sessions/_quiz_steps/week-02_day-03-sharding_html.json"}, {"path": "week-02/day-05-memory-footprint.html", "sidecar": "sessions/_quiz_steps/week-02_day-05-memory-footprint_html.json"}, {"path": "week-03/day-01-transformer-arithmetic.html", "sidecar": "sessions/_quiz_steps/week-03_day-01-transformer-arithmetic_html.json"}, {"path": "week-03/day-03-kv-cache.html", "sidecar": "sessions/_quiz_steps/week-03_day-03-kv-cache_html.json"}, {"path": "week-04/day-01-data-parallel-fsdp.html", "sidecar": "sessions/_quiz_steps/week-04_day-01-data-parallel-fsdp_html.json"}, {"path": "week-04/day-03-pipeline-parallel.html", "sidecar": "sessions/_quiz_steps/week-04_day-03-pipeline-parallel_html.json"}, {"path": "week-05/day-02-chinchilla-correction.html", "sidecar": "sessions/_quiz_steps/week-05_day-02-chinchilla-correction_html.json"}, {"path": "week-05/day-03-isoflops-methodology.html", "sidecar": "sessions/_quiz_steps/week-05_day-03-isoflops-methodology_html.json"}, {"path": "week-06/day-02-memory-wall.html", "sidecar": "sessions/_quiz_steps/week-06_day-02-memory-wall_html.json"}, {"path": "week-06/day-03-batching-economics.html", "sidecar": "sessions/_quiz_steps/week-06_day-03-batching-economics_html.json"}, {"path": "week-07/day-01-moe-fundamentals.html", "sidecar": "sessions/_quiz_steps/week-07_day-01-moe-fundamentals_html.json"}, {"path": "week-07/day-03-load-balancing-problem.html", "sidecar": "sessions/_quiz_steps/week-07_day-03-load-balancing-problem_html.json"}, {"path": "week-13/day-04-flashattention-paradigm.html", "sidecar": "sessions/_quiz_steps/week-13_day-04-flashattention-paradigm_html.json"}, {"path": "week-15/day-01-quantization-theory-int8.html", "sidecar": "sessions/_quiz_steps/week-15_day-01-quantization-theory-int8_html.json"}, {"path": "week-16/day-04-speculative-decoding.html", "sidecar": "sessions/_quiz_steps/week-16_day-04-speculative-decoding_html.json"}];
const SCHEMA={type:'object',properties:{file:{type:'string'},done:{type:'boolean'}},required:['file','done']};
function prompt(it){
  return `Upgrade ONE lesson's quiz from recall questions to STAFF-INTERVIEW-style SCENARIO/REASONING questions.

STEP 1 — Read the lesson: ${ROOT}/${it.path}  (understand its exact topic + the failure modes/trade-offs in section 4).
STEP 2 — Author EXACTLY 4 questions that make the learner REASON, not recite — the kind a Frontier-lab staff interviewer
asks. Good shapes: "You observe <symptom>. What is the most likely cause?" / "You change <X>. What breaks and why?" /
"Which trade-off decides <design choice>?" / "A teammate does <Y>; the model trains but <subtle wrongness> — where do
you look first?". Each question must have exactly 4 options with ONE correct, a 0-based ans index, and a fb that
explains WHY the right answer is right AND why the most tempting wrong option is wrong. Ground the numbers/mechanisms
in THIS lesson; be technically accurate (staff-grade).
STEP 3 — WRITE a JSON array to ${ROOT}/${it.sidecar} with the Write tool, EXACTLY:
[{"q":"<question html>","opts":["a","b","c","d"],"ans":<0-3>,"fb":"<why-right + why-tempting-wrong-is-wrong>"}, ... x4]
Valid JSON (double quotes, escape internal quotes as \\", no trailing commas). English only, zero Chinese.
Return {file:"${it.path}", done:true}.`;
}
const results = await pipeline(ITEMS,
  (it)=>agent(prompt(it),{label:'quiz:'+it.path.replace('week-','w').replace('/day-','-').replace('.html',''),phase:'Quiz',schema:SCHEMA,model:'sonnet',effort:'medium'}).then(r=>({it,r}))
);
return results.map(x=>x&&({file:x.it.path,done:x.r&&x.r.done}));
