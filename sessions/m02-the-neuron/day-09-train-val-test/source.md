---
quest_id: wf3-d06-split
donor: m02-day-09.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 9 — Train / Val / Test &amp; Overfitting"
spine: "exam vs answer-key"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 2 · Day 9 — Train / Val / Test &amp; Overfitting</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 9</div>
@@@ region name=sidebar_nav
<nav aria-label="Sections">
      <div class="nav-group-label">Module 02 · Train</div>
      <button class="nav-link" data-target="home"><span class="nl-dot"></span>Start here</button>
      <button class="nav-link" data-target="s1"><span class="nl-dot"></span>1 · What is it</button>
      <button class="nav-link" data-target="s2"><span class="nl-dot"></span>2 · Intuition</button>
      <button class="nav-link" data-target="s3"><span class="nl-dot"></span>3 · Playground</button>
      <button class="nav-link" data-target="s4"><span class="nl-dot"></span>4 · Mechanism &amp; why</button>
      <button class="nav-link" data-target="s5"><span class="nl-dot"></span>5 · Build it up</button>
      <button class="nav-link" data-target="s6"><span class="nl-dot"></span>6 · Quiz</button>
      <button class="nav-link" data-target="s7"><span class="nl-dot"></span>7 · Produce</button>
    </nav>
@@@ region name=nav_prev
<a class="lnav prev" href="../day-08-learning-rate/lesson.html"><span class="d">← Prev</span><span class="t">Learning-Rate Intuition</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../review.html"><span class="d">Next →</span><span class="t">Review Gate</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 9</span>
      <h1>Train / Val / Test &amp; Overfitting<span class="sub">Did the model learn, or just memorize?</span></h1>
      <p class="lede">A low training loss feels like success. But a model can score perfectly on the data it studied and still fail on anything new — that's <strong>overfitting</strong>. Today you split your data into three parts, keep some hidden, and use it to catch overfitting the moment it starts. This is how you know a model actually learned.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, you'll be able to name the three data splits and what each is for, explain why you must hold data out, define overfitting in one line, and read a train/val loss chart to spot the moment it begins.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Split the data into three parts</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> the training loop and loss (Days 1–3), and that the loss measures how wrong the model is. Today we ask a new question: is the model learning real patterns or just memorizing? <b>Nothing new needed.</b></div></div>

      <h4>First, the picture — hold this before the definitions</h4>
      <p>Think about studying for a big exam. You learn on a set of practice problems, you check yourself with a mock exam full of fresh questions, and your real grade comes from the final exam you sit exactly once. And there's a trap: if you just memorize the practice answer key, you'll ace practice and bomb the real thing. A model faces the exact same setup — and the exact same trap. Keep that picture; the three "splits" below are just practice, mock, and final exam.</p>

      <h4>What are the three splits?</h4>
      <p>Before training, you split your data into three groups. Each has one job.</p>
      <ol>
        <li><strong><span class="term" data-tip="The portion of data the model learns from — the only data it updates its weights on. Usually 60–80%.">Training set</span></strong> — the data the model learns from. Weights only ever update on this. (About 60–80%.)</li>
        <li><strong><span class="term" data-tip="Data held out from training, used to check progress and tune choices like learning rate. The model never learns on it. Usually 10–20%.">Validation set</span></strong> — held out from training. You check the loss on it to see if the model works on data it has not seen, and to tune choices like the learning rate. (About 10–20%.)</li>
        <li><strong><span class="term" data-tip="Data used only once, at the very end, for a final honest score. Never used to train or tune. Usually 10–20%.">Test set</span></strong> — locked away until the very end. Used once, for a final honest score. (About 10–20%.)</li>
      </ol>

      <h4>Why split at all?</h4>
      <p>If you only ever measure the loss on the data the model trained on, you cannot tell learning from memorizing. Held-out data is the only fair test. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> the Llama 3 and GPT-4-class technical reports both describe checking their training data for overlap with benchmarks like MMLU and GSM8K before trusting a score — the same held-out-data discipline you're about to learn.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 训练 loss 一直往下掉，看着像成功了，很容易就此收手 — a falling training loss feels like success, so it's tempting to stop there. In plain terms: a model can score perfectly on the data it studied and still fail on anything new. The only honest test is data it has never seen — that's the whole reason we hold some out.</div></div>
      <button class="gotit" type="button">Got the splits — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">Practice questions vs the real exam</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the exam picture from Section 1 — here it is as four cards, mapping each split (and the trap) precisely:</p>
      <div class="relate">
        <div class="card"><span class="big">📚</span><h5>Training set = practice problems</h5><p>You study these and learn from them. Getting them right is a good start — but it does not prove you understand.</p></div>
        <div class="card"><span class="big">📝</span><h5>Validation set = a mock exam</h5><p>Fresh questions you have not studied. You take a mock to check if you really learned the ideas, not just the practice answers.</p></div>
      </div>
      <div class="relate">
        <div class="card"><span class="big">🎓</span><h5>Test set = the real exam</h5><p>The final, official test. You sit it once. It gives an honest score, because you never saw these questions before.</p></div>
        <div class="card"><span class="big">🦜</span><h5>Memorizing the answer key</h5><p>If you only memorize the practice answers, you ace practice but fail the real exam. A model that does this is <b>overfitting</b>.</p></div>
      </div>
      <p><strong>In one line:</strong> training is your practice set, and the held-out data is the exam that tells you whether you learned or just memorized.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a student memorizes on purpose, but a model does not "decide" to overfit — it simply keeps lowering its training loss, and if it has enough freedom it will fit the noise in the practice set too. You catch this by watching the <span class="term" data-tip="The gap between the loss on training data and the loss on held-out validation data; a growing gap means overfitting.">train–val gap</span>, coming up next.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，train / val / test 就是"练习题 / 模拟考 / 真正的高考" — 用练习题学，用模拟考调整，真正的分数只看那场没见过的考试。The technical point: you fit on the train split, tune choices on validation, and report the final number on a test split touched exactly once, so the score reflects generalization, not memorization.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The train–val gap, and why holding data out is sacred</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>Overfitting, in one line</h4>
      <p><span class="term" data-tip="When a model fits the training data very well but does poorly on new, held-out data — it memorized instead of learning general patterns.">Overfitting</span> is when the model fits the training data very well but does poorly on new data. It has memorized the practice set instead of learning the general pattern.</p>

      <h4>How you spot it: watch two curves</h4>
      <p>During training you plot two losses against the training steps:</p>
      <ul>
        <li>The <b>training loss</b> almost always keeps falling — the model gets better and better at the data it studies.</li>
        <li>The <b>validation loss</b> falls too, at first. Then, if the model starts memorizing, it <b>turns around and rises</b>.</li>
      </ul>
      <p>The moment the validation loss stops falling and starts rising is the moment overfitting begins. The growing space between the two curves is the <b>train–val gap</b>.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>gap = validation_loss − training_loss</code><br><span style="color:var(--muted)">small gap → the model generalizes · large, growing gap → it is memorizing (overfitting)</span></div></div>
      <p><b>Worked example:</b> suppose at step 100 the training loss is 0.30 and the validation loss is 0.35 — a small gap of 0.05, healthy. Validation keeps improving to <b>0.28 at step 200</b> (its lowest), then turns around: by step 500 training loss has dropped to 0.02 but validation has climbed to 0.40 — a gap of 0.38 and growing. The model has started memorizing the training data. The best model is the one at the validation minimum, <b>step 200</b>, not the final step.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — the train–val gap</b>
      <br><b>1 · In words:</b> track the gap between validation loss and training loss as training proceeds; a small, steady gap is healthy, a growing gap means the model is memorizing.
      <br><b>2 · The formula:</b> <code>gap = validation_loss − training_loss</code>, watched as a function of training steps (or model complexity).
      <br><b>3 · Tiny numbers:</b> step 100 — train <code>0.30</code>, val <code>0.35</code> → gap <code>0.05</code> (healthy). Val keeps falling to <code>0.28</code> at step 200 (its lowest), then rises. Step 500 — train <code>0.02</code>, val <code>0.40</code> → gap <code>0.38</code> and rising. Early stopping keeps the <b>step-200</b> model — the validation minimum, not the last step.
      <br><b>4 · Sanity check:</b> the training loss almost always keeps falling, so it's the rising <em>validation</em> loss (not training loss) that signals overfitting. The model you keep is the one at the validation minimum — that's early stopping.</div></div>

      <h4>Two rules that keep the test honest</h4>
      <ul>
        <li><b>Never train on validation or test data.</b> If the model ever updates its weights on that data, it is no longer "unseen", and it can no longer catch overfitting.</li>
        <li><b>Touch the test set once.</b> If you tune your choices against the test set again and again, you slowly overfit to it too, and your final score becomes a lie. That is why the test set is locked away until the very end.</li>
      </ul>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>The stopping trick — early stopping:</b> keep the model from the step where validation loss was lowest, and stop once it has been rising for a while. This is the simplest, most common cure for overfitting, and you will use it constantly.</div></div>
      <div class="callout c-info"><span class="ic">💡</span><div><b>The cause is capacity — and its cures.</b> Overfitting grows with a model's <b>capacity</b>: the more freedom it has (more parameters, a higher-degree polynomial, more training steps), the more it can memorize noise instead of the pattern. So the U-shaped validation curve appears both as you <em>train longer</em> (the time axis above) and as you <em>add capacity</em> (the complexity axis — the knob your artifact sweeps with polynomial degree). Beyond early stopping, the standard gap-closers are <b>L2 / weight decay</b> (penalize large weights → a smoother, less wiggly function) and <b>dropout</b> (randomly zero units during training → forces redundancy, roughly an ensemble of sub-networks). More data helps too. Their mechanics are a later module; today just know they shrink the gap.</div></div>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a huge model can memorize enormous amounts of data. Held-out evaluation is how a lab knows its model truly generalizes and is not just parroting its training set — exactly why Llama 3 and GPT-4-class reports (the teaser above) go a step further and check their training data against benchmarks like MMLU and GSM8K. This practice is called <span class="term" data-tip="Removing or flagging training examples that overlap with a benchmark's test set, so the reported score reflects genuine generalization rather than memorized answers.">decontamination</span>. Every serious result is reported on data the model never trained on. That completes the "how a model learns — and how we trust it" story.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>The whole point of a test set is to give an honest score — which is exactly why the way it silently becomes dishonest is worth naming.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): data leakage.</b> Leakage is when information from the test set sneaks into training, so the model has secretly already seen the answers — the mock-exam questions slipped into your practice packet, so you "ace" the mock only because you had already seen them. It happens easily: the same or near-duplicate rows land in both train and test, or you compute a preprocessing step (like scaling by the mean and standard deviation) over the <em>whole</em> dataset <em>before</em> you split it, so the training data quietly absorbs facts about the test data. Nothing crashes, and the test score looks fantastic — but it is a lie, and the model does worse in the real world. A senior engineer catches this in <b>code review</b> by checking that the split happens <em>first</em>, that no example appears in two splits, and that every fit-on-data step (scalers, vocabularies) is learned from the training set only.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: how much data to hold out.</b> Every example you move into validation and test is an example the model cannot learn from. A large held-out set gives a stable, trustworthy score but leaves less data to train on, which can weaken the model. A small held-out set keeps most data for training but gives a noisy, unreliable estimate you cannot trust. Picking the split (and whether to use cross-validation when data is scarce) is a <b>design-review</b> decision that trades model strength against how much you can believe your own evaluation.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "I split data into train / validation / test: fit on train, tune hyperparameters on validation, and report once on test. Overfitting shows up as a growing train–val gap — training loss keeps falling while validation loss bottoms out and rises — so I keep the model at the validation minimum (early stopping). The subtle, expensive failure is data leakage: validation or test examples (or near-duplicates) sneaking into training makes the held-out score a lie, so the first thing I verify is that the splits are truly disjoint and de-duplicated before I trust any number."</div></div>
      <button class="gotit" type="button">Got the gap</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Predict the winner, then make overfitting appear</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the experiment worth running. The playground swept the <b>training-steps</b> axis (the U-shaped validation curve over time); here we sweep the <b>model-capacity</b> axis instead — same overfitting U-turn, just measured against how flexible the model is rather than how long it trained. <b>Before you run it, predict:</b> you'll fit polynomials of degree 1, 3, 5, and 9 to the same noisy points — which degree wins on the held-out data? Jot your guess. Then run it and watch the validation loss bottom out at a <em>middle</em> degree while the training loss just keeps falling with every extra degree. That U-turn in the validation curve is the exam-vs-answer-key moment made visible. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-09-train-val-test/experiment.py</code>. Make a tiny noisy dataset of 30 points and split it into <b>three disjoint sets</b> — 18 train / 6 validation / 6 test — printing the three sizes and checking they sum to 30 with no shared points. Fit a polynomial by least squares to the <b>training points only</b>, once each for degrees 1, 3, 5, 9. For every degree print the training loss and the validation loss (never fit on validation or test). You'll see the training loss keep falling as the degree rises while the validation loss bottoms out at a middle degree and then climbs — that turning point is overfitting. Pick the degree with the lowest <b>validation</b> loss (not the highest degree), and only then report the <b>test</b> loss for that one model. Run with <code>python3 sessions/m02-the-neuron/day-09-train-val-test/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 9 artifact.

Create sessions/m02-the-neuron/day-09-train-val-test/experiment.py that, with a comment on each step:
1. Makes a small noisy dataset of 30 points from a smooth curve plus random noise (fixed seed for determinism).
2. Splits it into THREE disjoint sets: 18 training, 6 validation, 6 test. Print the three sizes, and assert they are disjoint and sum to 30.
3. For degrees 1, 3, 5, 9: fit a polynomial by least squares to the TRAINING points only, and print the training loss (MSE on train) and the validation loss (MSE on val).
4. Show that training loss keeps falling as degree rises, but validation loss bottoms out at a middle degree and then climbs — the signature of overfitting.
5. Select the degree with the lowest VALIDATION loss (not the lowest training loss, and not necessarily the highest degree).
6. Only for that selected degree, evaluate and print the TEST loss once — the honest final score.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>You split 30 noisy points into <b>three disjoint sets</b> — 18 train / 6 validation / 6 test — and print the sizes summing to 30 with no shared points.</li>
        <li>For degrees 1, 3, 5, 9 your printout shows the training loss falling while the validation loss bottoms out and then rises — the turning point is overfitting.</li>
        <li>The best-validation degree is <b>not</b> the highest degree, and you report the <b>test</b> loss only for that one selected model (the test set is touched exactly once).</li>
        <li>Running twice under a fixed seed prints identical losses (deterministic).</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-09-train-val-test/log.md</code>: (1) why a low training loss alone doesn't prove a model learned; (2) at which complexity the validation loss stopped improving, and why that's the model you'd keep; (3) what data leakage is and why it makes a held-out score meaningless.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 9 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Train / Val / Test &amp; Overfitting</b>.<br>Next up: <b>Review Gate</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  split:{html:'<span class="prompt">&gt;&gt;&gt;</span> data = load()             <span class="dim"># 100 examples</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> train, val, test = split(data, <span class="num">0.7</span>, <span class="num">0.15</span>, <span class="num">0.15</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> len(train), len(val), len(test)\n'+
    '<span class="ok">(70, 15, 15)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># weights will ONLY update on train</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># val is measured, never trained on</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># test stays locked until the very end</span>',
    take:'<b>①  Three groups, three jobs.</b> 70 points to learn from (train), 15 to check progress (val), 15 locked away for one final honest score (test). The model never learns on val or test.'},
  gap:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> step <span class="kw">in</span> [<span class="num">100</span>, <span class="num">200</span>, <span class="num">300</span>, <span class="num">400</span>, <span class="num">500</span>]:\n'+
    '<span class="dim">...</span>     print(step, train_loss, val_loss)\n'+
    '<span class="hl">step  train   val</span>\n'+
    '<span class="ok">100   0.30    0.35</span>   <span class="dim"># small gap — healthy</span>\n'+
    '<span class="ok">200   0.15    0.28</span>\n'+
    '<span class="hl">300   0.08    0.30</span>   <span class="dim"># val stops falling…</span>\n'+
    '<span class="bad">400   0.04    0.36</span>   <span class="dim"># val rising — overfitting</span>\n'+
    '<span class="bad">500   0.02    0.40</span>   <span class="dim"># gap = 0.38 and growing</span>',
    take:'<b>②  The gap opens.</b> Training loss keeps falling, but after its step-200 low the validation loss turns around and climbs (by step 300 it is clearly rising). That turning point is where the model starts memorizing instead of learning.'},
  earlystop:{html:'<span class="prompt">&gt;&gt;&gt;</span> best_val = min over all steps    <span class="dim"># lowest val loss seen</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> best_val\n'+
    '<span class="ok">0.28  (at step 200)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># val kept rising for many steps after 200</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> stop_and_restore(step=<span class="num">200</span>)\n'+
    '<span class="ok"># keep the step-200 weights, throw away the rest</span>',
    take:'<b>③  Early stopping.</b> Keep the weights from the step where validation loss was lowest (step 200), and stop once it has been rising for a while. This simple rule is the most common cure for overfitting.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Data split into training, validation, and test sets'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='40' width='260' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='170' y='62' fill='#1F6280'>train (70%)</text><rect x='308' y='40' width='96' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='356' y='62' fill='#5E5191'>val (15%)</text><rect x='412' y='40' width='68' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='446' y='62' fill='#9A7208'>test</text><text x='260' y='96' fill='#2C2A28'>one dataset, three jobs</text></g></svg>",
  note:"<b>Split the data into three.</b> Most goes to <b>train</b> (learn from it), some to <b>validation</b> (check progress on unseen data), and a small piece to <b>test</b> (a final honest score). Each piece has exactly one job."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Weights update only on the training set'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='40' width='150' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='115' y='62' fill='#1F6280'>train → update w</text><rect x='210' y='40' width='130' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2' stroke-dasharray='4 3'/><text x='275' y='62' fill='#5E5191'>val → measure</text><rect x='360' y='40' width='120' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2' stroke-dasharray='4 3'/><text x='420' y='62' fill='#9A7208'>test → later</text><text x='260' y='96' fill='#2C2A28'>weights change on train only</text></g></svg>",
  note:"<b>Train only on the training set.</b> The weights update on the training data alone. Validation is only <b>measured</b>, never learned from — that is exactly what lets it tell you the truth about new data."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Early on both losses fall together'><g font-family='monospace'><line x1='55' y1='108' x2='470' y2='108' stroke='#E5DFD6' stroke-width='1.5'/><path d='M65 40 C 140 78, 250 96, 300 100' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><path d='M65 46 C 150 82, 250 96, 300 98' fill='none' stroke='#7C6DAA' stroke-width='2.5' stroke-dasharray='5 4'/><text x='230' y='124' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='70' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 70)'>loss</text><text x='330' y='96' font-size='10' fill='#1F6280'>train</text><text x='330' y='112' font-size='10' fill='#5E5191'>val</text></g></svg>",
  note:"<b>At first, both losses fall.</b> The solid line is training loss, the dashed line is validation loss. Early in training they drop together — the model is learning real patterns that help on unseen data too."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Validation loss turns around while training loss keeps falling'><g font-family='monospace'><line x1='55' y1='108' x2='470' y2='108' stroke='#E5DFD6' stroke-width='1.5'/><path d='M65 40 C 160 85, 300 100, 460 104' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><path d='M65 46 C 150 82, 230 92, 300 90 C 380 88, 430 60, 460 40' fill='none' stroke='#C93B3B' stroke-width='2.5' stroke-dasharray='5 4'/><line x1='300' y1='30' x2='300' y2='108' stroke='#C93B3B' stroke-width='1' stroke-dasharray='2 3'/><text x='300' y='24' font-size='10' fill='#C93B3B' text-anchor='middle'>overfitting starts</text><text x='40' y='70' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 70)'>loss</text><text x='420' y='100' font-size='10' fill='#1F6280'>train ↓</text><text x='420' y='36' font-size='10' fill='#C93B3B'>val ↑</text></g></svg>",
  note:"<b>The gap opens — this is overfitting.</b> Training loss keeps falling, but validation loss turns around and <b>rises</b>. The model is memorizing the training set. The vertical line marks the moment it started — the best model was just before it."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Early stopping keeps the model from the lowest validation point'><g font-family='monospace'><line x1='55' y1='108' x2='470' y2='108' stroke='#E5DFD6' stroke-width='1.5'/><path d='M65 46 C 150 82, 230 92, 300 90 C 380 88, 430 60, 460 40' fill='none' stroke='#C93B3B' stroke-width='2.5' stroke-dasharray='5 4'/><circle cx='270' cy='92' r='7' fill='#2D8B55' stroke='#1a5c38' stroke-width='2'/><text x='270' y='78' font-size='10' fill='#1a5c38' text-anchor='middle'>lowest val — keep this</text><line x1='320' y1='30' x2='320' y2='108' stroke='#6B645E' stroke-width='1' stroke-dasharray='2 3'/><text x='360' y='30' font-size='10' fill='#6B645E'>stop here</text><text x='40' y='70' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 70)'>val loss</text></g></svg>",
  note:"<b>Early stopping.</b> Keep the weights from the step where validation loss was lowest (green dot), and stop once it has been rising for a while (dashed line). This simple rule is the most common, most reliable cure for overfitting."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='The test set gives one final honest score'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='60' y='40' width='170' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='145' y='62' fill='#9A7208'>test set (used once)</text><text x='255' y='62' fill='#6B645E'>→</text><rect x='290' y='40' width='170' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='375' y='62' fill='#1a5c38'>honest final score ✓</text><text x='260' y='98' fill='#2C2A28'>never trained on → you can trust it</text></g></svg>",
  note:"<b>Why a frontier lab cares.</b> Held-out evaluation is the only way to know a huge model truly generalizes instead of memorizing. Real labs go further and decontaminate their training data against benchmarks first. You now know how a model learns — and how we trust it. Foundations done."}
];
@@@ region name=QS
var QS=[
 {q:'1. Your model scores almost perfectly on the test set, and the code ran with no error. But when you deploy it, it does much worse on real users. You then find that you scaled all your data by its mean and standard deviation <em>before</em> splitting into train and test. What most likely happened?',
  opts:['The model is simply excellent and real users are the problem','Data leakage: computing the scaling over the whole dataset before splitting let training absorb facts about the test data, so the test score was inflated and dishonest — with no crash to warn you','The test set was too small, which always inflates scores','A perfect test score with no error guarantees good real-world performance'],
  ans:1, fb:'Right. Fitting a preprocessing step on the full dataset before splitting leaks test-set information into training, so the test score is a lie. Nothing crashes and the number looks great. You check that the split happens first and that anything fit-on-data (scalers, vocabularies) is learned from the training set only.'},
 {q:'2. What is overfitting?',
  opts:['using too large a learning rate','fitting the training data well but failing on new, unseen data','training for too few steps','having too few weights'],
  ans:1, fb:'Right. Overfitting is when the model memorizes the training set and does poorly on data it has not seen.'},
 {q:'3. During training, the training loss keeps falling but the validation loss starts rising. This means:',
  opts:['the model is learning perfectly','overfitting has begun','the learning rate is too small','the test set is too big'],
  ans:1, fb:'Right. When validation loss turns around and rises while training loss still falls, the model has started memorizing — overfitting.'},
 {q:'4. Why is the test set used only once, at the very end?',
  opts:['because it is smaller','so it stays truly unseen and gives an honest final score','to make training faster','because it has no labels'],
  ans:1, fb:'Right. If you tuned against the test set repeatedly, you would slowly overfit to it and the final score would no longer be honest. So it is locked away until the end.'}
];
