#!/usr/bin/env python3
"""Builder for M24 — Eval Science & LLM-as-a-Judge.
Source: 06-evaluation/README.md + metrics/ + benchmarks/. Builds on M21a (judges/labels).
4 lessons + review. Run: python3 sessions/_build_m24.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m24")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — Contamination & Saturation ----------------
L("day-01-contamination.html", dict(
  qid="m24-d01-contam", title="Module 24 · Day 1 — Contamination & Saturation", nav_title="Spiral · M24 Day 1",
  eyebrow="Module 24 · Eval · Day 1", h1="Contamination & Saturation",
  lead="A model scores 92% on a hard benchmark. Impressive — until you find those exact questions sitting in its training data. It didn't reason; it remembered. This is <em>contamination</em>, the number-one way benchmark scores lie. Today you learn how the leak happens, three ways to catch it, and why every good benchmark eventually stops telling you anything.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how a test set leaks into pretraining, use n-gram overlap to detect it, compute a simple overlap fraction, and say why benchmarks saturate.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-elicitation.html", next_label="Elicitation & Prompt Sensitivity",
  sections=[
    {"title":"When the test is in the textbook","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a model is <span class="term" data-tip="Training on a huge text corpus to predict the next token. This corpus is scraped from the internet and can accidentally include benchmark questions and answers.">pretrained</span> on huge amounts of internet text (M21a), and a <span class="term" data-tip="A standardized, fixed set of questions used to compare models fairly, like a shared exam.">benchmark</span> is a fixed shared exam. Today: why exam scores can be fake.</div></div>
     <h4>The core problem</h4>
     <p>A benchmark is only trustworthy if the model has never seen the questions before. But benchmarks are published on the internet — and the internet is exactly what models train on. When benchmark questions (and their answers) end up in the training data, we call it <span class="term" data-tip="When benchmark questions or answers leak into a model's training data, so the model can recall them instead of solving them. Its score is inflated and no longer measures real skill.">contamination</span>.</p>
     <h4>Why it inflates the score</h4>
     <p>A contaminated model does not <em>reason</em> to the answer. It <em>recalls</em> it. The score looks like skill, but it measures memory of the answer key. This is silent: the number goes up, and nothing on the surface tells you it is fake.</p>
     <h4>Saturation</h4>
     <p>Even with clean data, a benchmark has a second death. Once the best models score near the ceiling (say 95%+), the benchmark stops separating them — a 0.3% gap is noise, not signal. We call this <span class="term" data-tip="When top models all score near the maximum on a benchmark, so it can no longer tell them apart. The benchmark is 'full' and needs replacing.">saturation</span>. GLUE saturated, so researchers built SuperGLUE; that saturated, so they built MMLU. This is the benchmark treadmill.</p>''',
     "gotit":"Got contamination & saturation"},
    {"title":"The stolen answer key","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔑</span><h5>A student who saw the answer key</h5><p>Imagine a student who found the exam answer key the night before. They score 100% — but they learned nothing. Their grade is real on paper and meaningless in truth. A contaminated model is that student.</p></div>
       <div class="card"><span class="big">📚</span><h5>An exam everyone has memorized</h5><p>Now imagine an exam so old that every student has seen every past paper. Everyone scores 98%. The exam can no longer tell a great student from a good one. That is saturation — the test is used up.</p></div>
     </div>
     <p><strong>In one line:</strong> contamination is seeing the answers early; saturation is the test becoming too easy for everyone. Both make the score stop meaning what you think.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a student <i>knows</i> whether they cheated. A model has no memory of "studying" — the leak is buried in trillions of tokens, so nobody set out to cheat and nobody can easily tell it happened. You have to go looking.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Detect a leak","body":'''<p>Watch three detection methods run on one suspicious benchmark item: n-gram overlap, a canary string, and a perturbation test. <strong>Click all three.</strong></p>'''},
    {"title":"Measuring the overlap","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>To catch contamination, you check how much a benchmark question <em>textually overlaps</em> with the training data. The standard tool is <span class="term" data-tip="An n-gram is a run of n consecutive words. A 13-gram is 13 words in a row. Long exact matches between a test question and training text are strong evidence of a leak.">n-gram overlap</span>: count how many long word-runs (say 13 words in a row) from the test item also appear in the training set.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       overlap = (# of n-grams in the test item ALSO found in training) / (total n-grams in the test item)<br>
       <span class="dim"># n-gram = n words in a row; a high overlap = likely leaked</span><br>
       flag the item as contaminated if overlap &gt; threshold (e.g. 0.5)
     </div></div>
     <p>Symbols: an <em>n-gram</em> is n consecutive words. You slide a window of size n across the test item, and for each window ask "does this exact run appear anywhere in training?"</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       test item has 10 distinct n-grams<br>
       6 of them are found, word-for-word, in the training data<br><br>
       overlap = 6 / 10 = <span class="hl">0.6</span><br>
       0.6 &gt; 0.5 threshold → <span class="hl">FLAGGED as contaminated</span>
     </div></div>
     <p>Two other cheap checks back this up. A <span class="term" data-tip="A rare, unique marker string a benchmark author hides in the data. If a model can reproduce the canary, it must have trained on the benchmark file.">canary string</span> is a unique marker hidden in the benchmark; if the model can recite it, it trained on the file. A <span class="term" data-tip="Slightly changing a question (rename variables, reword) so a memorizing model fails but a reasoning model still succeeds. A big score drop means the model relied on memory.">perturbation test</span> lightly rewrites the question; if a memorizing model's score crashes while a reasoning model's holds, memory was doing the work.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — silent contamination.</b> Contamination raises the score with <b>no error, no warning, no crash</b>. The leaderboard just shows a bigger number, and everyone celebrates. You only find it if you deliberately run overlap / canary / perturbation checks. If you never look, a memorized model looks exactly like a smart one — and you ship the wrong model.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — public vs held-out benchmarks.</b> Public benchmarks (MMLU, HumanEval) are <i>comparable</i> — everyone reports the same number, so you can rank models. But being public means they are on the internet, so they leak and saturate. A private held-out set is <i>trustworthy</i> (nobody trained on it) but not comparable (no one else can reproduce your number). Serious teams use both: public for comparison, private for the truth.</div></div>''',
     "gotit":"Got overlap detection"},
    {"title":"Contamination, built up","body":'''<p>Here is the whole leak-and-detect story assembled: the clean setup, the leak, the inflated score, the overlap check, and saturation. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a contamination probe","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m24_contamination.py</code>. Take ~30 benchmark questions and a small "training" corpus. Deliberately paste 10 of the questions into the corpus. Compute 13-gram overlap for every question against the corpus, and show that the 10 planted ones flag (overlap high) while the rest do not.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 24 Day 1 artifact.
Create experiments/foundations/m24_contamination.py: build a tiny "training corpus" and a 30-item benchmark, plant 10 items verbatim into the corpus, then compute 13-gram overlap = (matched n-grams)/(total n-grams) for each item. Print which items exceed a 0.5 threshold and show the 10 planted ones flag while the clean ones do not. Add a perturbation test that reworders 5 items and shows overlap drops.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m24/day-01-log.md</code>: what overlap threshold cleanly separated planted from clean items, and did any clean item false-flag?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "ngram":{"btn":"① n-gram overlap","html":'''<span class="prompt">&gt;&gt;&gt;</span> test = "the mitochondria is the powerhouse of the cell"
<span class="prompt">&gt;&gt;&gt;</span> overlap = ngram_match(test, training_data, n=5)
<span class="hl">matched 6 / 10 five-grams</span>
<span class="hl">overlap = 0.60</span>   <span class="dim"># &gt; 0.5 → FLAGGED as leaked</span>''',
      "take":"<b>①  Long exact word-runs give it away.</b> 6 of 10 five-grams from the test item appear verbatim in training. Overlap 0.6 → almost certainly memorized, not reasoned."},
    "canary":{"btn":"② canary string","html":'''<span class="prompt">&gt;&gt;&gt;</span> canary = "BENCH-CANARY-8f3a-DO-NOT-TRAIN"
<span class="prompt">&gt;&gt;&gt;</span> model("Repeat: BENCH-CANARY-8f3a-...")
<span class="bad">"...DO-NOT-TRAIN"</span>   <span class="dim"># it completed the secret marker!</span>
<span class="dim"># the model saw the benchmark FILE itself</span>''',
      "take":"<b>②  The canary is a hidden fingerprint.</b> The benchmark author buried a unique string. The model can finish it — so it trained on the benchmark file. Proof of a leak."},
    "perturb":{"btn":"③ perturbation test","html":'''<span class="prompt">&gt;&gt;&gt;</span> original: score = 0.92
<span class="prompt">&gt;&gt;&gt;</span> reworded (same meaning): score = 0.61
<span class="bad">Δ = -0.31</span>   <span class="dim"># memorizer collapses when text changes</span>
<span class="dim"># a true reasoner would barely move</span>''',
      "take":"<b>③  Rewrite it and watch it fall.</b> A memorizing model scores 0.92 on the exact wording but 0.61 when reworded. A big drop means it relied on remembered text, not understanding."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='170' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='145' y='37' fill='#1a5c38'>held-out test</text><rect x='290' y='18' width='170' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='375' y='37' fill='#1F6280'>training data</text></g></svg>",
     "note":"<b>The clean setup.</b> A held-out test the model has never seen, kept separate from the training data. This is the only way a score means 'real skill.'"},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='20' width='170' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='145' y='39' fill='#C93B3B'>test questions</text><path d='M230 35 L288 35' stroke='#C93B3B' stroke-width='2' marker-end='url(#a1)'/><rect x='290' y='20' width='170' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='375' y='39' fill='#9A7208'>training data (leaked!)</text><defs><marker id='a1' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L6,3 L0,6 Z' fill='#C93B3B'/></marker></defs></g></svg>",
     "note":"<b>The leak.</b> Benchmark questions were published online, scraped, and swept into training. Now the model has quietly seen the exam."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='18' width='220' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='37' fill='#C93B3B'>score 92% (recalled, not solved)</text></g></svg>",
     "note":"<b>The inflated score.</b> The model recites remembered answers. The number looks like brilliance but measures memory of the answer key. Nothing warns you."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='120' y='18' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='37' fill='#1F6280'>overlap = 6/10 = 0.6 &gt; 0.5 → FLAG</text></g></svg>",
     "note":"<b>The overlap check catches it.</b> Count long word-runs shared between the test item and training. High overlap (0.6) flags the item as contaminated. Back it up with a canary and a perturbation test."},
    {"viz":"<svg viewBox='0 0 520 96'><g font-family='monospace' font-size='10'><text x='260' y='14' fill='#6B645E' text-anchor='middle'>top model score climbs to the ceiling</text><line x1='40' y1='75' x2='470' y2='75' stroke='#E5DFD6'/><line x1='40' y1='25' x2='470' y2='25' stroke='#C99A12' stroke-dasharray='4,3'/><text x='476' y='28' fill='#9A7208'>100%</text><polyline points='40,68 150,50 260,34 370,28 460,26' fill='none' stroke='#7C6DAA' stroke-width='2.5'/><text x='300' y='50' fill='#5E5191'>saturation</text></g></svg>",
     "note":"<b>Saturation ends the benchmark.</b> Once every top model scores near 100%, the test can't separate them. You build a harder one — GLUE → SuperGLUE → MMLU. The treadmill never stops."},
  ],
  quiz=[
    {"q":"1. What is benchmark contamination?","opts":["the benchmark file is corrupted on disk","benchmark questions or answers leaked into the model's training data, so it recalls instead of reasons","the model runs out of memory","the benchmark has too many questions"],"ans":1,"fb":"Contamination = the test leaked into training. The model recites remembered answers, so its score is inflated and no longer measures skill."},
    {"q":"2. A test item has 20 n-grams; 12 are found verbatim in the training data. The overlap is:","opts":["0.4","0.6","1.2","20"],"ans":1,"fb":"overlap = matched / total = 12 / 20 = 0.6. Above a 0.5 threshold, you'd flag this item as likely contaminated."},
    {"q":"3. You reword a benchmark question (same meaning) and the model's score drops from 0.90 to 0.55. This suggests:","opts":["the model got smarter","the model was relying on memorized text (contamination), since a reasoner would barely move","the benchmark is too easy","nothing — scores always vary"],"ans":1,"fb":"A perturbation test: a big drop from a light reword means the model leaned on remembered wording, not understanding — a contamination signal."},
    {"q":"4. Why do teams keep a private held-out eval AND report public benchmarks?","opts":["to waste compute","public benchmarks are comparable across teams but leak/saturate; private held-out is trustworthy but not comparable — you need both","private evals are illegal to share","public benchmarks are always wrong"],"ans":1,"fb":"Public = comparable but contamination-prone. Private held-out = trustworthy but only you can run it. Serious teams use public for ranking and private for the truth."},
  ],
  fin={"em":"🔑","h3":"Day 1 complete — you can spot a fake score!",
       "p":"You now know the two ways benchmark scores lie: contamination (the model saw the answers) and saturation (the test got too easy for everyone). You can detect a leak with n-gram overlap, a canary string, and a perturbation test — and you know why every good benchmark eventually dies. Next: <b>Day 2</b> — the same model scoring wildly differently just because you changed the prompt."},
))
# ---------------- Day 2 — Elicitation & Prompt Sensitivity ----------------
L("day-02-elicitation.html", dict(
  qid="m24-d02-elicit", title="Module 24 · Day 2 — Elicitation & Prompt Sensitivity", nav_title="Spiral · M24 Day 2",
  eyebrow="Module 24 · Eval · Day 2", h1="Elicitation & Prompt Sensitivity",
  lead="The exact same model, on the exact same questions, can score 40% or 75% — nothing changed but the prompt. Ask nicely, add a few examples, tweak the answer format, and the number moves. So when a benchmark says a model 'can't do X', is that the model's limit, or just a bad prompt? Today you learn to tell capability apart from elicitation.",
  goal="<b>🎯 By the end, you'll be able to:</b> define elicitation, explain why the same model scores differently under different prompts, compute a score range across prompts, and choose between a fixed prompt and a best-prompt protocol.",
  prev_href="day-01-contamination.html", prev_label="Contamination & Saturation", next_href="day-03-llm-as-judge.html", next_label="LLM-as-a-Judge & Its Biases",
  sections=[
    {"title":"Two different questions","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (a benchmark is a fixed exam), and that a model's answer depends on the exact prompt text you feed it. Today: why one model gives you many scores.</div></div>
     <h4>Capability vs elicitation</h4>
     <p>There are two very different questions hiding inside "how good is this model?" The first is <span class="term" data-tip="What the model can do at its best, given the ideal prompt and setup. Its true ceiling of skill.">capability</span> — what the model <em>can</em> do at its best. The second is <span class="term" data-tip="How much of the model's true capability a given prompt actually pulls out. A weak prompt elicits little; a strong prompt elicits more.">elicitation</span> — how much of that capability a particular prompt actually pulls out.</p>
     <h4>Why the score moves</h4>
     <p>The model is the same, but the prompt is the lens. A weak prompt (vague instruction, no examples, awkward answer format) elicits little, so the score looks low. A strong prompt (clear instruction, a few worked examples, a clean format the parser understands) elicits more, so the score rises. You measured elicitation, but you <em>reported</em> capability.</p>
     <h4>What makes prompts sensitive</h4>
     <ul>
       <li><b>Format:</b> "Answer: B" vs "The answer is (B)." — if your <span class="term" data-tip="The small piece of code that reads the model's raw text output and extracts the final answer to grade. If it expects a shape the model didn't use, it marks correct answers wrong.">parser</span> expects one and gets the other, a correct answer is scored wrong.</li>
       <li><b>Few-shot examples:</b> showing 0, 1, or 5 solved examples changes the score a lot. This is <span class="term" data-tip="Putting a few solved examples in the prompt before the real question, so the model copies the pattern. More examples usually lifts the score.">few-shot</span> prompting.</li>
       <li><b>Instruction wording:</b> "Think step by step" — a <span class="term" data-tip="Chain-of-thought: asking the model to reason out loud step by step before the final answer. It often raises accuracy on reasoning tasks.">chain-of-thought</span> cue — can lift reasoning scores sharply.</li>
       <li><b>Option order:</b> even shuffling multiple-choice letters shifts results.</li>
     </ul>''',
     "gotit":"Got capability vs elicitation"},
    {"title":"A shy expert in an interview","body":
     '''<div class="relate">
       <div class="card"><span class="big">🙊</span><h5>The bad interviewer</h5><p>A brilliant engineer is asked a vague, rushed question by a nervous interviewer. They freeze and give a weak answer. The interviewer writes "can't do it." The skill was there — it just wasn't pulled out. That's under-elicitation.</p></div>
       <div class="card"><span class="big">🎤</span><h5>The good interviewer</h5><p>A calm interviewer asks the same thing clearly, gives an example, and says "take your time." The same engineer shines. Same person, same knowledge — a different result, purely from how the question was asked.</p></div>
     </div>
     <p><strong>In one line:</strong> the prompt is the interviewer. A good prompt lets the model show what it knows; a bad prompt makes a capable model look incapable.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a human can say "let me try again" or ask for clarification. A model in a benchmark gets one fixed prompt and no chance to push back — so the prompt you pick <i>is</i> the whole interview.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Same model, three prompts","body":'''<p>Watch one fixed model answer the same questions under three prompts — zero-shot, few-shot, and a format fix — and see the score swing. <strong>Click all three.</strong></p>'''},
    {"title":"Measuring the swing","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>To separate capability from elicitation, you don't run one prompt. You run <em>several</em> reasonable prompts and look at the whole spread of scores. The bottom of the spread is what a lazy setup reports; the top is closer to the model's real capability.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       run the SAME model under k prompt variants → scores s₁ … s_k<br>
       elicitation gap = max(s₁ … s_k) − min(s₁ … s_k)<br>
       <span class="dim"># a big gap = the score depends more on the prompt than on the model</span>
     </div></div>
     <p>Symbols: <code>s_i</code> is the benchmark score under prompt variant <code>i</code>; <code>k</code> is how many variants you tried. The <em>gap</em> is how much room the prompt alone controls.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       same model, 3 prompts:<br>
       zero-shot        → s₁ = 0.42<br>
       5-shot examples  → s₂ = 0.68<br>
       5-shot + format fix → s₃ = 0.75<br><br>
       gap = 0.75 − 0.42 = <span class="hl">0.33</span>  <span class="dim"># 33 points swing, one model!</span><br>
       reporting only 0.42 = a false ceiling; the true capability is ≥ 0.75
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — under-elicitation (the false ceiling).</b> You run a weak, default prompt, the model scores 42%, and you conclude "this model can't do math." No error, no warning — just a wrong conclusion. A better prompt would have shown 75%. You cancel the model, or over-claim a rival's lead, all because you measured your prompt, not the model. The only fix is to <b>try hard to elicit</b> before reporting a ceiling.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — fixed prompt vs best-prompt.</b> A single <i>fixed</i> prompt for every model is <b>reproducible and fair</b> — same lens for all — but may under-elicit some models and misrank them. Searching for the <i>best</i> prompt per model reveals true capability but is <b>not reproducible</b> (a different searcher gets a different number) and can quietly overfit the prompt to the test. Good practice: fix the prompt for head-to-head comparison, and separately report a best-effort number to show the ceiling.</div></div>''',
     "gotit":"Got the swing"},
    {"title":"Elicitation, built up","body":'''<p>Here it is assembled: one fixed model, several prompts, a spread of scores, the gap, and the false-ceiling trap. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a prompt sweep","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m24_elicitation.py</code>. Take one model and one small multiple-choice set. Run it under at least 4 prompt variants (zero-shot, few-shot, "think step by step", and a format fix). Print each score and the max−min gap. Show that a single default prompt would have reported a false ceiling.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 24 Day 2 artifact.
Create experiments/foundations/m24_elicitation.py: evaluate ONE model on a small multiple-choice benchmark under >=4 prompt variants (zero-shot, 5-shot, chain-of-thought, and a strict answer-format fix). Also add an answer-parsing step and show that a mismatched format scores correct answers as wrong. Print each variant's score and the elicitation gap = max - min. Conclude which number you'd report for a fair comparison vs for the true ceiling.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m24/day-02-log.md</code>: how big was your elicitation gap, and did a format mismatch alone cost real points?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "zero":{"btn":"① zero-shot","html":'''<span class="prompt">&gt;&gt;&gt;</span> prompt = question   <span class="dim"># no examples, no instructions</span>
<span class="prompt">&gt;&gt;&gt;</span> evaluate(model, benchmark)
<span class="hl">score = 0.42</span>   <span class="dim"># looks weak — but is it the model?</span>''',
      "take":"<b>①  Bare prompt, low score.</b> No examples, no guidance. The model scores 0.42. A lazy report stops here and calls the model weak."},
    "fewshot":{"btn":"② + few-shot examples","html":'''<span class="prompt">&gt;&gt;&gt;</span> prompt = 5_solved_examples + question
<span class="prompt">&gt;&gt;&gt;</span> evaluate(model, benchmark)
<span class="hl">score = 0.68</span>   <span class="dim"># +26 points, same model, same questions</span>''',
      "take":"<b>②  Show it how, score jumps.</b> Five worked examples lift the SAME model to 0.68. Nothing changed but the prompt — that 26-point jump is pure elicitation."},
    "format":{"btn":"③ + format fix","html":'''<span class="prompt">&gt;&gt;&gt;</span> parser expected "Answer: X"
<span class="prompt">&gt;&gt;&gt;</span> model was writing "The answer is (X)."   <span class="dim"># parsed as wrong!</span>
<span class="prompt">&gt;&gt;&gt;</span> fix the format instruction
<span class="hl">score = 0.75</span>   <span class="dim"># correct answers now counted</span>''',
      "take":"<b>③  A parsing mismatch was hiding points.</b> The model was right but wrote the answer in a shape the grader didn't recognize. Fix the format → 0.75. Range for one model: 0.42 → 0.75."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>ONE fixed model (frozen)</text></g></svg>",
     "note":"<b>Start with one frozen model.</b> Its weights never change through this whole experiment. Any score difference we see is not the model — it is the prompt."},
    {"viz":"<svg viewBox='0 0 520 74'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='24' width='140' height='28' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='100' y='42' fill='#5A544E'>zero-shot</text><rect x='190' y='24' width='140' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='42' fill='#1F6280'>few-shot</text><rect x='350' y='24' width='140' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='420' y='42' fill='#1a5c38'>+ format fix</text></g></svg>",
     "note":"<b>Feed it several prompts.</b> Same questions, different lenses: bare, with examples, with a clean answer format. Each is a reasonable way to ask."},
    {"viz":"<svg viewBox='0 0 520 96'><g font-family='monospace' font-size='10'><line x1='40' y1='80' x2='470' y2='80' stroke='#E5DFD6'/><rect x='70' y='58' width='40' height='22' fill='#F5F0E8' stroke='#6B645E'/><text x='90' y='52' fill='#5A544E' text-anchor='middle'>.42</text><rect x='230' y='34' width='40' height='46' fill='#E4F2F7' stroke='#2A7B9B'/><text x='250' y='28' fill='#1F6280' text-anchor='middle'>.68</text><rect x='390' y='26' width='40' height='54' fill='#E8F5EE' stroke='#2D8B55'/><text x='410' y='20' fill='#1a5c38' text-anchor='middle'>.75</text></g></svg>",
     "note":"<b>A whole spread of scores.</b> One model gives 0.42, 0.68, 0.75. The score you 'get' depends entirely on which prompt you chose to run."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>gap = 0.75 − 0.42 = 0.33</text></g></svg>",
     "note":"<b>The elicitation gap.</b> 33 points controlled by the prompt alone. A big gap warns you: this score is more about your setup than about the model."},
    {"viz":"<svg viewBox='0 0 520 68'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='20' width='210' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='145' y='39' fill='#C93B3B'>report 0.42 = false ceiling ✗</text><rect x='270' y='20' width='210' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='375' y='39' fill='#1a5c38'>true capability ≥ 0.75 ✓</text></g></svg>",
     "note":"<b>The trap and the fix.</b> Reporting the low number is a false ceiling — it under-sells the model. Try hard to elicit before you claim a limit, and say which number is comparable vs which is the ceiling."},
  ],
  quiz=[
    {"q":"1. What is the difference between capability and elicitation?","opts":["they are the same thing","capability is what the model can do at its best; elicitation is how much of that a given prompt pulls out","capability is speed; elicitation is accuracy","elicitation only matters for images"],"ans":1,"fb":"Capability = the model's true ceiling. Elicitation = how much of it a particular prompt reveals. A weak prompt under-elicits and hides real capability."},
    {"q":"2. Same model, prompts scoring 0.50, 0.66, and 0.71. The elicitation gap is:","opts":["0.21","0.16","0.71","0.05"],"ans":0,"fb":"gap = max − min = 0.71 − 0.50 = 0.21. That 21-point range is controlled by the prompt, not the model."},
    {"q":"3. Your parser expects 'Answer: B' but the model writes 'The answer is (B).' What happens?","opts":["it's counted correct anyway","correct answers get scored as wrong (a format mismatch), so the score is artificially low","the model crashes","the benchmark auto-fixes it"],"ans":1,"fb":"A format mismatch makes the grader reject right answers. This is a classic under-elicitation bug — the model knew it, but the number hid it."},
    {"q":"4. Why report BOTH a fixed-prompt score and a best-effort score?","opts":["to confuse readers","the fixed prompt is reproducible and comparable across models; the best-effort shows the true capability ceiling","best-effort is always higher, so it's the real one","fixed prompts are illegal"],"ans":1,"fb":"Fixed = same lens for all models (fair ranking) but may under-elicit. Best-effort = the ceiling, but not reproducible. Reporting both keeps the comparison honest and shows the true limit."},
  ],
  fin={"em":"🎤","h3":"Day 2 complete — you can tell the model from the prompt!",
       "p":"You now separate capability (what the model can do) from elicitation (what a prompt pulls out). You can measure the gap across prompt variants, you know format, few-shot, and wording all move the score, and you know the false-ceiling trap of under-elicitation. Fixed prompts for fair ranking; best-effort for the true ceiling. Next: <b>Day 3</b> — using an LLM itself as the grader, and the biases that come with it."},
))
# ---------------- Day 3 — LLM-as-a-Judge & Its Biases ----------------
L("day-03-llm-as-judge.html", dict(
  qid="m24-d03-judge", title="Module 24 · Day 3 — LLM-as-a-Judge & Its Biases", nav_title="Spiral · M24 Day 3",
  eyebrow="Module 24 · Eval · Day 3", h1="LLM-as-a-Judge & Its Biases",
  lead="For open-ended tasks — is this answer helpful? is this summary faithful? — there is no answer key a computer can match. Human judges are the gold standard but slow and costly. So we ask another LLM to be the judge. It is fast and cheap and scales to millions of comparisons. It is also biased in strange, sneaky ways — and if you don't correct for them, your winner might just be the answer that came first.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain model-graded eval, name four judge biases, run a position-swap to cancel order bias, and calibrate a judge against humans.",
  prev_href="day-02-elicitation.html", prev_label="Elicitation & Prompt Sensitivity", next_href="day-04-held-out.html", next_label="Held-Out-by-Construction",
  sections=[
    {"title":"Ask a model to grade","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> from M21a, models can compare two answers and pick a preferred one. Here we use that same skill to <em>grade</em> other models' outputs. Some tasks have no exact answer key (Day 1's automatic grading can't reach them).</div></div>
     <h4>Model-graded evaluation</h4>
     <p>For open-ended tasks, we use <span class="term" data-tip="Using a capable LLM to score or compare other models' outputs, instead of an exact-match grader or a human. Also called model-graded evaluation.">LLM-as-a-judge</span>: give a strong model two answers (or one answer and a rubric) and ask it which is better, or to rate quality 1–10. It plays the role a human rater would.</p>
     <h4>Two common shapes</h4>
     <ul>
       <li><b>Pairwise:</b> "Here are answers A and B. Which is better?" — used for head-to-head rankings (like Chatbot Arena).</li>
       <li><b>Pointwise:</b> "Rate this answer 1–10 on helpfulness." — a single score per answer.</li>
     </ul>
     <h4>Why we bother</h4>
     <p>A human rater costs money and minutes per comparison. An LLM judge costs a fraction of a cent and returns in a second, so you can run millions. That scale is what makes it worth learning — and what makes its biases dangerous, because a biased judge is biased a million times over.</p>''',
     "gotit":"Got model-graded eval"},
    {"title":"A judge with quirky habits","body":
     '''<div class="relate">
       <div class="card"><span class="big">⚖️</span><h5>A fast but quirky judge</h5><p>Imagine a contest judge who is quick and cheap but has odd habits: they slightly favor whoever presents <i>first</i>, they think <i>longer</i> speeches sound smarter, and they secretly like contestants who talk like <i>them</i>. None of these habits are about who is actually best.</p></div>
       <div class="card"><span class="big">🔁</span><h5>The fix: judge both orders</h5><p>To cancel the first-speaker habit, you have every pair perform twice — once in each order — and average. If a contestant only wins when they go first, that win was the habit, not the merit.</p></div>
     </div>
     <p><strong>In one line:</strong> an LLM judge is a fast, cheap grader with predictable biases — so you design the test to cancel them out, then check the judge against humans.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a human judge can be told "ignore length" and try to comply. An LLM judge's biases are baked into its weights — telling it "be fair" helps only a little. You need mechanical fixes (swap positions, control length), not a polite request.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"A pairwise judge with a swap","body":'''<p>Watch a pairwise judge pick a winner, then run the same pair in swapped order and reveal position bias. <strong>Click all three.</strong></p>'''},
    {"title":"Position bias and the swap fix","body":
     '''<h4>Step 1 — the biases, named</h4>
     <ul>
       <li><span class="term" data-tip="The judge tends to favor whichever answer is shown first (or in a fixed slot), regardless of quality. Also called order bias.">Position / order bias</span> — favors the answer in a certain slot.</li>
       <li><span class="term" data-tip="The judge rates longer answers higher even when the extra length adds nothing, or is padding.">Length / verbosity bias</span> — longer looks better.</li>
       <li><span class="term" data-tip="The judge prefers answers written by itself or its own model family over equally-good answers from other models.">Self-preference</span> — likes its own family's style.</li>
       <li>Style/tone bias — confident, formatted, or flattering answers get a boost.</li>
     </ul>
     <h4>Step 2 — the swap fix (for position bias)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       judge the pair TWICE:  order (A,B) and order (B,A)<br>
       A wins only if it wins in BOTH orders → "consistent" preference<br>
       if the two orders disagree → position bias; count it a tie (or drop it)<br>
       <span class="dim"># averaging over both orders cancels the slot advantage</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       order (A,B): judge picks the <span class="hl">first</span> one → A wins<br>
       order (B,A): judge picks the <span class="hl">first</span> one → B wins<br><br>
       the two orders <span class="hl">disagree</span> → the judge just picked position 1 both times<br>
       verdict: <span class="hl">TIE (position bias detected)</span> — NOT a real win for A<br>
       <span class="dim"># without the swap, you'd have crowned A the winner by pure luck of order</span>
     </div></div>
     <p>To judge quality, we compare against the gold standard — human ratings — and measure <span class="term" data-tip="How well the judge's verdicts match human raters. Usually measured as agreement rate or a correlation; you accept the judge only if this is high enough.">calibration</span>: does the judge agree with humans often enough (say 80%+ agreement)? If not, you fix the prompt/rubric, or you don't trust the judge.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — position bias flips the winner.</b> Run each pair in only <b>one</b> order and a judge with position bias will systematically crown whoever you happened to list first. The leaderboard looks authoritative — clean percentages, clear ranking — but a chunk of the "wins" are just slot luck. Reorder your inputs and the ranking changes. You catch it only by running both orders and checking for disagreement.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — LLM judge vs human judge.</b> An LLM judge is <b>scalable and cheap</b> (millions of comparisons, cents each) but <b>biased and needs calibration</b> to humans; its biases repeat identically at scale. A human panel is the <b>trusted ground truth</b> but slow, expensive, and itself noisy (raters disagree). Best practice: use the LLM judge for volume, with swap + length controls, and calibrate it on a human-rated sample before you trust the rankings.</div></div>''',
     "gotit":"Got the swap fix"},
    {"title":"The judge, built up","body":'''<p>Here it is assembled: the pair, the judge's first verdict, the swapped rerun, the disagreement → tie, and human calibration. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a de-biased judge","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m24_llm_judge.py</code>. Build a pairwise LLM judge (use the local bridge). For each pair, query it in both orders (A,B) and (B,A). Count a win only if both orders agree; otherwise call it a tie. Compare the raw single-order win rate to the swap-consistent win rate, and show how many "wins" were actually position bias.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 24 Day 3 artifact.
Create experiments/foundations/m24_llm_judge.py: a pairwise LLM-as-judge over ~20 answer pairs using the local bridge at http://localhost:11211/api/openai/v1 (model aws:anthropic.claude-sonnet-4-6). For each pair, query BOTH orders (A,B) and (B,A); count a win only if both orders agree, else a tie. Report raw single-order win rate vs swap-consistent win rate, the position-bias rate (fraction that disagreed), and a length-bias check (does the judge favor the longer answer?). Wrap calls in try/except and degrade gracefully if the bridge is down.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m24/day-03-log.md</code>: what fraction of pairs disagreed across orders (your position-bias rate), and did the judge favor longer answers?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "judge":{"btn":"① judge picks a winner","html":'''<span class="prompt">&gt;&gt;&gt;</span> judge("Q: ... \\nA: [answer A]\\nB: [answer B]\\nWhich is better?")
<span class="hl">"Answer A is better."</span>
<span class="dim"># looks decisive — but is it merit, or is A just first?</span>''',
      "take":"<b>①  The judge is confident.</b> It cleanly picks A over B. One order, one verdict — this is what most naive pipelines do, and it's exactly where the trap hides."},
    "swap":{"btn":"② swap the order","html":'''<span class="prompt">&gt;&gt;&gt;</span> judge("Q: ... \\nA: [answer B]\\nB: [answer A]\\nWhich is better?")   <span class="dim"># B now first</span>
<span class="hl">"Answer A is better."</span>   <span class="dim"># = the answer now in slot A = original B!</span>
<span class="bad"># it picked the FIRST slot again, not the same content</span>''',
      "take":"<b>②  Swap reveals the truth.</b> Both times it picked whatever sat in the first slot. The verdict flipped between the two contents — that's position bias, not judgment."},
    "calib":{"btn":"③ calibrate to humans","html":'''<span class="prompt">&gt;&gt;&gt;</span> agree = compare(judge_verdicts, human_verdicts)
<span class="hl">agreement = 0.83</span>   <span class="dim"># 83% match with human raters</span>
<span class="dim"># above ~0.80 → trust the judge; below → fix or drop it</span>''',
      "take":"<b>③  Check it against humans.</b> After the swap fix, compare the judge to human raters. 83% agreement means it's usable; a low number means the judge — not the models — is the problem."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='170' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='145' y='37' fill='#1F6280'>answer A</text><rect x='290' y='18' width='170' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='375' y='37' fill='#5E5191'>answer B</text></g></svg>",
     "note":"<b>Start with two answers to one prompt.</b> No answer key exists — the task is open-ended. We need a judge to say which is better."},
    {"viz":"<svg viewBox='0 0 520 76'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='34' fill='#9A7208'>LLM judge, order (A,B)</text><text x='260' y='64' fill='#1F6280'>→ picks A (first slot)</text></g></svg>",
     "note":"<b>First verdict, order (A,B).</b> The judge picks A. Decisive and clean — but we don't yet know if it judged content or just position."},
    {"viz":"<svg viewBox='0 0 520 76'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='34' fill='#9A7208'>same pair, order (B,A)</text><text x='260' y='64' fill='#C93B3B'>→ picks B (first slot again!)</text></g></svg>",
     "note":"<b>Rerun swapped.</b> Now B sits first, and the judge picks B. It chose the first slot both times — the content preference flipped."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='36' fill='#C93B3B'>orders disagree → TIE (position bias)</text></g></svg>",
     "note":"<b>Disagreement = a caught bias.</b> Because the two orders disagree, this pair is a tie, not a win for A. The swap just saved you from a fake result."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='36' fill='#1a5c38'>agreement with humans = 0.83 ✓</text></g></svg>",
     "note":"<b>Calibrate to the gold standard.</b> Compare the de-biased judge to human raters. High agreement (0.83) means you can trust it at scale; low agreement means fix the judge first."},
  ],
  quiz=[
    {"q":"1. Why use an LLM as a judge instead of exact-match grading?","opts":["it's always more accurate than humans","many tasks (helpfulness, summary quality) have no single answer key, and an LLM judge scales far cheaper than human raters","exact-match is illegal","LLMs have no biases"],"ans":1,"fb":"Open-ended tasks can't be matched to a key. An LLM judge fills the human-rater role at a fraction of the cost and scales to millions — its downside is bias."},
    {"q":"2. A pairwise judge picks the FIRST answer in order (A,B), and again the FIRST answer in order (B,A). This is:","opts":["a genuine win for A","position/order bias — it favored the slot, not the content, so this pair should be a tie","length bias","self-preference"],"ans":1,"fb":"Picking the first slot in both orders means the content preference flipped — that's position bias. The swap catches it; the pair is a tie."},
    {"q":"3. Which of these is a real, common LLM-judge bias?","opts":["preferring shorter answers always","length/verbosity bias — rating longer answers higher even when length adds nothing","refusing to judge","judging faster over time"],"ans":1,"fb":"Length/verbosity bias is classic: judges over-reward longer answers. Position bias and self-preference are the other big ones."},
    {"q":"4. Your LLM judge agrees with human raters only 55% of the time (near coin-flip on 2 options). You should:","opts":["ship the rankings anyway","not trust the rankings — fix the judge prompt/rubric or fall back to humans, because the judge itself is unreliable","double the number of pairs","remove the swap step"],"ans":1,"fb":"Calibration this low means the judge — not the models — is the problem. A near-50% agreement on a 2-choice task is barely above chance; don't trust its verdicts until it calibrates."},
  ],
  fin={"em":"⚖️","h3":"Day 3 complete — you can judge the judge!",
       "p":"You now know LLM-as-a-judge: model-graded eval for open-ended tasks, scalable and cheap but carrying position, length, self-preference, and style biases. You can cancel order bias with a position swap, and you calibrate the judge against human ratings before trusting it. Next: <b>Day 4</b> — the deepest fix of all: designing evals that <em>cannot</em> be contaminated in the first place."},
))
# ---------------- Day 4 — Held-Out-by-Construction ----------------
L("day-04-held-out.html", dict(
  qid="m24-d04-heldout", title="Module 24 · Day 4 — Held-Out-by-Construction", nav_title="Spiral · M24 Day 4",
  eyebrow="Module 24 · Eval · Day 4", h1="Held-Out-by-Construction",
  lead="Day 1 taught you to <em>detect</em> contamination after the fact. But detection is a losing game — you're always chasing a leak that already happened. There is a stronger move: build evals that <em>cannot</em> be contaminated in the first place, because the questions did not exist when the model was trained. This is held-out-by-construction, and it's how frontier teams get signal they can actually trust.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain held-out-by-construction, name three ways to build one (fresh, procedurally generated, time-gated), say why it beats decontamination, and reason about its cost.",
  prev_href="day-03-llm-as-judge.html", prev_label="LLM-as-a-Judge & Its Biases", next_href="review.html", next_label="Module 24 · Review Gate",
  sections=[
    {"title":"Contamination-proof by design","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (contamination inflates scores; you can detect it with overlap/canary/perturbation). Today: how to make contamination <em>impossible</em>, not just detectable.</div></div>
     <h4>The problem with detecting after the fact</h4>
     <p>Day 1's tools find leaks <em>after</em> training. But you can only decontaminate what you can detect — paraphrased or translated leaks slip through, and you can't rebuild the model just because you found a leak. <span class="term" data-tip="Removing benchmark items from training data after you find them. It only catches leaks you can detect, and can't undo a model that already trained on them.">Decontamination-after-the-fact</span> is always one step behind.</p>
     <h4>The stronger idea</h4>
     <p><span class="term" data-tip="Designing an eval whose questions could not have been in any training data — because they didn't exist yet, or are generated fresh each run. Contamination is impossible by design, not merely detected.">Held-out-by-construction</span> flips it: instead of scrubbing leaks out, you make sure the eval questions were never available to train on. If a question did not exist when the model finished training, the model cannot have memorized it — full stop.</p>
     <h4>Three ways to build one</h4>
     <ul>
       <li><span class="term" data-tip="Writing brand-new questions after a model's training cutoff, and keeping them private until eval time.">Fresh</span> — write new items after the training cutoff and keep them private.</li>
       <li><span class="term" data-tip="A generator makes new problem instances on demand from a template (e.g. random numbers into a math template), so every run uses unseen items.">Procedurally generated</span> — a program mints new instances each run from a template.</li>
       <li><span class="term" data-tip="Only score questions created after the model's training cutoff date (e.g. news, exams, or code commits from after the cutoff). By construction the model never saw them.">Time-gated</span> — use only questions dated after the training cutoff (new exams, new commits, new events).</li>
     </ul>''',
     "gotit":"Got the idea"},
    {"title":"A fresh exam every time","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎲</span><h5>New questions, every sitting</h5><p>Some driving tests draw from a giant pool and generate a fresh mix for every candidate. You can't memorize "the answers" because your paper is new. That's procedural generation — the test builds itself on the spot.</p></div>
       <div class="card"><span class="big">📅</span><h5>Only questions from after you studied</h5><p>Imagine an exam that only asks about events from <i>after</i> the day you stopped studying. There is no way you could have crammed for it — the material didn't exist yet. That's time-gating.</p></div>
     </div>
     <p><strong>In one line:</strong> instead of guarding an old answer key, you use a key that was minted after the model was frozen — so memorization is off the table by design.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a driving-test pool can still be reverse-engineered if the template is simple. A too-predictable generator can be "learned" as a pattern, so held-out-by-construction only holds if the items are genuinely varied and truly private/new.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Build an uncontaminable eval","body":'''<p>Watch three constructions guarantee freshness: a procedural generator, a time-gate on the cutoff, and what happens if you reuse the set. <strong>Click all three.</strong></p>'''},
    {"title":"Why construction beats detection","body":
     '''<h4>Step 1 — the guarantee, in words</h4>
     <p>A model trained up to a <span class="term" data-tip="The date after which no new data went into a model's training. Anything created after this date is guaranteed unseen by the model.">training cutoff</span> can only have memorized data that existed before that date. So if every eval item is created <em>after</em> the cutoff (or freshly generated at eval time), memorization is logically impossible.</p>
     <h4>Step 2 — the rule</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       if  item_creation_date &gt; model_training_cutoff  →  item is UNSEEN (safe)<br>
       or  item = generator(fresh_random_seed)          →  item is UNSEEN (safe)<br>
       <span class="dim"># contamination probability = 0 by construction, no detection needed</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       model cutoff = Jan 2025<br><br>
       benchmark A: written 2023 (public)     → 2023 &lt; Jan 2025 → <span class="bad">could be contaminated</span><br>
       benchmark B: exam released Mar 2025     → Mar 2025 &gt; Jan 2025 → <span class="hl">guaranteed unseen ✓</span><br>
       benchmark C: math items generated fresh at run time → <span class="hl">guaranteed unseen ✓</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — reusing an eval erodes it.</b> A fresh eval is only fresh <b>once</b>. The moment you publish it, run it repeatedly, or let its items into the next training run, it starts leaking — and it erodes <i>silently</i>: scores drift up over time with no error, and you slowly mistake memorization for progress again. A held-out set that has been public for a year is no longer held out. You must keep minting new items (or rotate/retire the set).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — fresh-generation cost vs contamination-proof signal.</b> Held-out-by-construction gives you the <b>most trustworthy signal there is</b> — a number contamination cannot fake. But it <b>costs continuously</b>: you must keep writing fresh items, build and maintain a generator, or wait for new time-gated data, and you lose the easy comparability of a fixed public benchmark. You pay ongoing effort in exchange for a score you can actually believe.</div></div>''',
     "gotit":"Got construction over detection"},
    {"title":"Held-out-by-construction, built up","body":'''<p>Here it is assembled: the leaky public set, the training cutoff line, the time-gate, the fresh generator, and the erosion warning. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a fresh generator","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m24_held_out.py</code>. Write a procedural generator for a simple task (e.g. multi-step arithmetic or a templated logic puzzle) that mints a fresh, unique instance from a random seed each call. Show that two runs produce disjoint item sets, and add a check that the generated items do NOT overlap (n-gram) with a small "training corpus."</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 24 Day 4 artifact.
Create experiments/foundations/m24_held_out.py: a procedural benchmark generator for templated multi-step arithmetic (random operands/seed) that mints fresh, unique instances each run. Show that two seeded runs produce disjoint item sets, verify 13-gram overlap with a toy "training corpus" is ~0 (contamination-proof), and add a time-gate helper that filters items to those dated after a given training cutoff. Print a note on the ongoing cost of keeping it fresh.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m24/day-04-log.md</code>: did your generator's two runs share any items, and what was the n-gram overlap with the training corpus?</div></div>''',
     "gotit":"Done — on to the review"},
  ],
  demos={
    "gen":{"btn":"① procedural generator","html":'''<span class="prompt">&gt;&gt;&gt;</span> item = generate(seed=random())
<span class="hl">"A train leaves at 3:47 going 61mph... (new numbers each call)"</span>
<span class="prompt">&gt;&gt;&gt;</span> run_A ∩ run_B = <span class="hl">∅</span>   <span class="dim"># zero shared items</span>''',
      "take":"<b>①  A generator mints unseen items.</b> Every call makes a fresh instance from a random seed. No two runs share questions, so nothing can have been memorized."},
    "gate":{"btn":"② time-gate the cutoff","html":'''<span class="prompt">&gt;&gt;&gt;</span> model_cutoff = "2025-01"
<span class="prompt">&gt;&gt;&gt;</span> keep = [q for q in items if q.date &gt; model_cutoff]
<span class="hl">kept: exam from 2025-03 ✓</span>   <span class="dim"># created AFTER cutoff → unseen</span>
<span class="dim">dropped: 2023 items (could be contaminated)</span>''',
      "take":"<b>②  Time-gating guarantees freshness.</b> Keep only items created after the model's training cutoff. The model literally could not have seen them."},
    "reuse":{"btn":"③ reuse erodes it","html":'''<span class="prompt">&gt;&gt;&gt;</span> published fresh eval, then reran for a year
<span class="hl">2025-03: score 0.61</span>   <span class="dim"># honest, first use</span>
<span class="hl">2026-03: score 0.79</span>   <span class="bad"># eroded — now in training data</span>
<span class="dim"># fresh only ONCE; keep minting new items</span>''',
      "take":"<b>③  Freshness expires.</b> Once you publish and reuse an eval, it leaks into future training and silently inflates. A held-out set stays held out only if you keep it new."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='18' width='220' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='37' fill='#C93B3B'>public benchmark (leaks + saturates)</text></g></svg>",
     "note":"<b>Start with the problem.</b> A public benchmark is comparable but leaks into training and saturates. Detecting the leak (Day 1) is always one step behind."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10'><line x1='260' y1='16' x2='260' y2='78' stroke='#C99A12' stroke-width='2' stroke-dasharray='4,3'/><text x='260' y='12' fill='#9A7208' text-anchor='middle'>training cutoff</text><text x='130' y='50' fill='#C93B3B' text-anchor='middle'>before →</text><text x='130' y='66' fill='#C93B3B' text-anchor='middle'>could be seen</text><text x='395' y='50' fill='#1a5c38' text-anchor='middle'>after →</text><text x='395' y='66' fill='#1a5c38' text-anchor='middle'>guaranteed unseen</text></g></svg>",
     "note":"<b>Draw the cutoff line.</b> The model only knows data from before its training cutoff. Anything after that date is, by logic, impossible to have memorized."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='18' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='37' fill='#1F6280'>time-gate: keep items dated &gt; cutoff</text></g></svg>",
     "note":"<b>Time-gating.</b> Use only questions created after the cutoff — a new exam, fresh news, recent code commits. Contamination probability is exactly zero."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='18' width='280' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='37' fill='#1a5c38'>generator(seed) → fresh unseen items</text></g></svg>",
     "note":"<b>Procedural generation.</b> A program mints new instances on demand from a template. Every eval run uses items that never existed before — trustworthy without any detection."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10'><text x='260' y='14' fill='#6B645E' text-anchor='middle'>same 'fresh' set, reused over time</text><line x1='40' y1='75' x2='470' y2='75' stroke='#E5DFD6'/><polyline points='40,62 180,58 320,44 460,30' fill='none' stroke='#C93B3B' stroke-width='2.5' stroke-dasharray='5,3'/><text x='300' y='40' fill='#C93B3B'>erodes ↗ (leaks back in)</text></g></svg>",
     "note":"<b>The catch: freshness expires.</b> Reuse or publish the set and it slowly leaks into future training, inflating scores silently. The ongoing cost of held-out-by-construction is that you must keep minting new items."},
  ],
  quiz=[
    {"q":"1. What does held-out-by-construction mean?","opts":["deleting the test set after use","designing the eval so its questions could not have been in training (didn't exist yet, or generated fresh) — contamination is impossible by design","hiding the benchmark on a private server only","using a bigger model as judge"],"ans":1,"fb":"By construction the items were never available to train on, so memorization is logically impossible — no detection needed."},
    {"q":"2. Why does held-out-by-construction beat decontamination-after-the-fact?","opts":["it's cheaper","detection only catches leaks you can find (paraphrases slip through) and can't undo a trained model; construction removes the leak's possibility entirely","decontamination is illegal","it needs no benchmark at all"],"ans":1,"fb":"Detection is always one step behind — it misses disguised leaks and can't retrain the model. Construction guarantees zero contamination up front."},
    {"q":"3. A model's training cutoff is Jan 2025. Which item is guaranteed unseen?","opts":["a benchmark published in 2022","a puzzle from a book printed in 2024","an exam released in March 2025, after the cutoff","a famous riddle from 2019"],"ans":2,"fb":"Only the March 2025 exam was created after the Jan 2025 cutoff, so the model could not have trained on it. The others predate the cutoff and could be contaminated."},
    {"q":"4. You built a fresh eval, published it, and reran it monthly for a year. Scores keep rising. Most likely:","opts":["the models genuinely improved that much","the eval eroded — once public and reused it leaked into new training, so it's no longer held out","the generator broke","nothing, scores always rise"],"ans":1,"fb":"A fresh eval is fresh only once. Publishing and reusing it lets it leak back into training, inflating scores silently. Keep minting new items or rotate the set."},
  ],
  fin={"em":"🎲","h3":"Day 4 complete — you can build trustworthy signal!",
       "p":"You now know the strongest defense against contamination: held-out-by-construction. Build evals from fresh, procedurally generated, or time-gated items so memorization is impossible by design — better than chasing leaks after the fact. The price is ongoing freshness: reuse erodes any eval. Next: the <b>review gate</b> — prove you can spot a fake score, tell the model from the prompt, judge the judge, and build an eval that can't be gamed."},
))
# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m24-review", title="Module 24 · Review Gate", nav_title="Spiral · M24 Review",
  eyebrow="Module 24 · The Gate — You Can Trust (or Break) a Score", h1="Module 24 — Review Gate",
  lead="Checkpoint. The gate question: <b>when someone shows me a benchmark number, can I say whether it's real — and if not, why?</b> Contamination, elicitation, judge bias, and freshness are the four ways a score lies. Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Evaluation is the highest-frequency real task — every model claim, paper, and launch rests on it, so this gate matters.",
  prev_href="day-04-held-out.html", prev_label="Day 4 · Held-Out-by-Construction",
  checks=[
    ["Day 1","I can explain contamination and saturation, and detect a leak with n-gram overlap, a canary string, and a perturbation test."],
    ["Day 2","I can separate capability from elicitation, measure the score gap across prompts, and explain fixed-prompt vs best-prompt."],
    ["Day 3","I can explain LLM-as-a-judge, name its biases (position, length, self-preference), run a position swap, and calibrate to humans."],
    ["Day 4","I can explain held-out-by-construction (fresh, procedural, time-gated), why it beats decontamination, and why reuse erodes it."],
  ],
  quiz=[
    {"q":"1. A model scores 95% on a public benchmark, but a lightly reworded version drops it to 60%. The most likely cause is:","opts":["the model improved","contamination — it memorized the exact wording, so a reasoner-level drop reveals it wasn't reasoning","the benchmark is broken","random noise"],"ans":1,"fb":"A big drop from a small reword is a perturbation-test signal for contamination (Day 1): the model relied on remembered text, not understanding."},
    {"q":"2. Same model, three prompts score 0.45, 0.60, 0.78. The elicitation gap is:","opts":["0.33","0.18","0.78","0.15"],"ans":0,"fb":"gap = max − min = 0.78 − 0.45 = 0.33 (Day 2). A 33-point range means the prompt, not the model, controls most of the reported score."},
    {"q":"3. A pairwise LLM judge picks the first-listed answer in BOTH orders of a swap. This pair is:","opts":["a clear win for the first answer","a tie caused by position bias — the judge favored the slot, not the content","proof of length bias","a self-preference win"],"ans":1,"fb":"Picking the first slot in both orders flips the content preference — that's position bias (Day 3). The swap catches it; score the pair as a tie."},
    {"q":"4. A model's training cutoff is Jan 2025. Which eval gives a contamination-proof number?","opts":["a benchmark published in 2021","a coding exam released in June 2025, after the cutoff","a well-known 2020 riddle set","any benchmark with a canary string"],"ans":1,"fb":"Only the June 2025 exam was created after the cutoff, so it's held-out-by-construction (Day 4) — the model could not have seen it."},
    {"q":"5. Which pairing correctly matches the failure to its fix?","opts":["contamination → position swap; judge bias → n-gram overlap","contamination → held-out-by-construction; under-elicitation → try harder prompts; judge position bias → order swap; eval reuse → mint fresh items","all four are fixed by a bigger model","none of these have fixes"],"ans":1,"fb":"Each failure has its own fix: build fresh evals (contamination), elicit harder (false ceiling), swap orders (judge bias), and keep minting new items (erosion). Mixing them up means you'll 'fix' the wrong thing."},
  ],
  verdict_pass="you can look at any benchmark claim and interrogate it: is it contaminated? is it under-elicited? is the judge biased? is the eval still fresh? You can also build an eval that resists all four. This is the daily craft of evaluation science — the skill every model claim depends on.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The four failure modes — <code>contamination</code> → <code>elicitation</code> → <code>judge bias</code> → <code>eval erosion</code> — and their fixes should feel automatic before you trust (or make) any model claim.",
  complete_label="Mark Module 24 complete",
  fin={"em":"🏆","h3":"Module 24 — passed!",
       "p":"You can now trust or break a benchmark score on sight. You know the four ways scores lie — contamination (memorized answers), under-elicitation (a false ceiling from a weak prompt), judge bias (position, length, self-preference), and eval erosion (freshness expiring) — and the fix for each, up to designing held-out-by-construction evals that can't be gamed. Evaluation is the highest-frequency real task in an AI lab, and you now do it with eyes open."},
  score_target=4,
))
print("M24 built: 4 lessons + review")

