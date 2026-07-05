#!/usr/bin/env python3
"""Builder for M21a — Core Post-Training: SFT/PEFT/RM/PPO/DPO.
Convert from 10-reinforcement-learning/rlhf + 02-fine-tuning. 8 lessons + review.
Run: python3 sessions/_build_m21a.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m21a")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — The Post-Training Pipeline ----------------
L("day-01-pipeline.html", dict(
  qid="m21a-d01-pipeline", title="Module 21a · Day 1 — The Post-Training Pipeline", nav_title="Spiral · M21a Day 1",
  eyebrow="Module 21a · Train · Day 1", h1="From Raw Model to Helpful Assistant",
  lead="A freshly pretrained model can finish your sentences, but it won't reliably follow an instruction or refuse a harmful one. Post-training is the set of steps that turns that raw next-word predictor into a helpful, honest assistant: supervised fine-tuning, then learning from human preferences. Today is the map of the whole pipeline — every later lesson is one box on it.",
  goal="<b>🎯 By the end, you'll be able to:</b> name the three stages (pretrain → SFT → preference optimization), say what each stage adds, and explain why we need preferences instead of just more labelled answers.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-sft.html", next_label="Supervised Fine-Tuning",
  sections=[
    {"title":"Three stages, three jobs","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> you know a transformer predicts the next token (M5a) and the RL frame from M20 (policy, reward). We build the whole alignment pipeline on those.</div></div>
     <h4>Where a pretrained model falls short</h4>
     <p>A <span class="term" data-tip="Training on a huge text corpus to predict the next token. Produces broad knowledge but no instruction-following or safety behaviour.">pretrained</span> model has read the internet and can predict likely next tokens. Ask it a question and it might continue with <em>more questions</em>, because that is what often follows a question online. It has knowledge but no manners.</p>
     <h4>The pipeline</h4>
     <ul>
       <li><b>Pretraining</b> — learn language and world knowledge from raw text (next-token prediction).</li>
       <li><b>Supervised fine-tuning (SFT)</b> — show it thousands of good <em>instruction → answer</em> examples so it learns the assistant format.</li>
       <li><b>Preference optimization (RLHF/DPO)</b> — teach it which of two answers people prefer, so it improves on qualities we can't easily write down.</li>
     </ul>
     <h4>Why preferences, not just more answers</h4>
     <p>SFT needs a "correct" answer to copy. But "which reply is more helpful, honest, and harmless?" often has no single right answer — and it's far easier for a human to <em>compare two replies</em> than to write the perfect one. Preference learning turns easy comparisons into a training signal. That is the heart of the next seven lessons.</p>''',
     "gotit":"Got the three stages"},
    {"title":"Raising a new hire","body":
     '''<div class="relate">
       <div class="card"><span class="big">📚</span><h5>School (pretraining)</h5><p>A new hire arrives having read enormous amounts — broad knowledge, but no idea how <i>your</i> company answers customers. That's the pretrained model.</p></div>
       <div class="card"><span class="big">🧑‍🏫</span><h5>Shadowing + feedback (post-training)</h5><p>First they copy good example responses (SFT). Then a manager reviews pairs of their drafts — "this one's better" — and they internalize the preferences (RLHF). Now they sound like your best rep.</p></div>
     </div>
     <p><strong>In one line:</strong> pretraining gives knowledge, SFT gives the format, and preference learning gives judgment about what's actually good.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a human keeps learning on the job. A deployed model is frozen — everything it "knows how to be" was baked in during these training stages, which is why getting them right matters so much.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Walk the pipeline","body":'''<p>Watch the same prompt handled by a raw model, an SFT model, and a preference-tuned model. <strong>Click all three.</strong></p>'''},
    {"title":"Why each stage exists","body":
     '''<h4>Step 1 — what SFT can and can't do</h4>
     <p>SFT is plain supervised learning: minimize the loss of copying reference answers. It reliably teaches <em>format</em> (answer the question, use the assistant voice). It cannot teach qualities you can't demonstrate cleanly — tone, calibrated refusals, "helpful but not sycophantic" — because there's no single target to copy.</p>
     <h4>Step 2 — the preference signal</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       collect: prompt x, two answers (y_win ≻ y_lose) ranked by a human<br>
       learn:   make y_win more likely than y_lose<br>
       <span class="dim"># comparisons are cheaper and more reliable than writing perfect answers</span>
     </div></div>
     <p>The two ways to use this signal are the two big branches: build a <span class="term" data-tip="A model trained to score how good a response is, from human preference comparisons. Used as the reward in RLHF.">reward model</span> and optimize against it with RL (RLHF/PPO), or optimize the preferences <em>directly</em> (DPO). We cover both.</p>
     <h4>Step 3 — the order matters</h4>
     <p>You always SFT <em>before</em> preference optimization. RLHF/DPO make small, careful improvements around a good starting policy; run them on a raw model and they wander into gibberish. SFT gets you into the right neighbourhood; preferences polish.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — alignment tax / capability regression.</b> Aggressive preference tuning can quietly make the model <i>worse</i> at things it used to do well (math, coding) while it optimizes for "niceness" — the "alignment tax." The chat evals look great while a capability benchmark slips. You only see it if you keep measuring the old skills, not just the new preference score.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — SFT vs preference optimization.</b> SFT is simple, stable, and cheap, but capped by the quality of your demonstrations and can't express fuzzy goals. Preference optimization reaches goals you can only <i>compare</i>, but it is more complex, easier to destabilize, and vulnerable to gaming the reward. Real pipelines use both, in that order.</div></div>''',
     "gotit":"Got why each stage exists"},
    {"title":"The pipeline, assembled","body":'''<p>Here is the whole pipeline built up: raw text → pretrained model → SFT → preference optimization → assistant. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Map your own pipeline","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_pipeline.md</code>. For a chatbot you'd build, write the three stages and, for each, list: the data you'd need, the loss/objective, and one failure mode. Keep it to one page.</p>
     <h4>Option B · let Claude help</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 1 artifact.
Create experiments/foundations/m21a_pipeline.md: a one-page map of pretrain -> SFT -> preference optimization for a customer-support chatbot. For each stage give the data required, the objective/loss, and one failure mode. End with why SFT must come before preference optimization.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-01-log.md</code>: which stage would you spend the most data budget on, and why?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "raw":{"btn":"① raw pretrained model","html":'''<span class="prompt">&gt;&gt;&gt;</span> model("How do I reset my password?")
<span class="bad">"How do I change my email? How do I ..."</span>
<span class="dim"># continues the pattern of a FAQ page — doesn't answer</span>''',
      "take":"<b>①  The raw model just continues text.</b> It saw many FAQ pages, so it produces more questions. Knowledge, but no instruction-following."},
    "sft":{"btn":"② after SFT","html":'''<span class="prompt">&gt;&gt;&gt;</span> sft_model("How do I reset my password?")
<span class="ok">"Go to Settings > Security > Reset password, then ..."</span>
<span class="dim"># learned the instruction→answer format from examples</span>''',
      "take":"<b>②  SFT teaches the format.</b> After copying thousands of instruction→answer pairs, it now answers the question. But its tone/judgment is still average."},
    "pref":{"btn":"③ after preference tuning","html":'''<span class="prompt">&gt;&gt;&gt;</span> aligned_model("How do I reset my password?")
<span class="ok">"Sure! Here are the steps... If you don't see the option,
it may be managed by your admin — want me to check?"</span>
<span class="dim"># more helpful, calibrated — learned from human preferences</span>''',
      "take":"<b>③  Preferences add judgment.</b> Trained on which replies humans prefer, it becomes more helpful and calibrated — qualities you can't easily write a single target for."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='35' fill='#5A544E'>raw text (the internet)</text></g></svg>",
     "note":"<b>Start with raw text.</b> Trillions of tokens of unlabelled text — the pretraining diet."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='130' y='16' width='260' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>pretrained model — knows language</text></g></svg>",
     "note":"<b>Pretraining → a base model.</b> Broad knowledge and fluent language, via next-token prediction. But no assistant behaviour yet."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>+ SFT on instruction→answer pairs</text></g></svg>",
     "note":"<b>SFT teaches the format.</b> Supervised copying of good instruction→answer examples. Now it answers, in the assistant voice."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='16' width='300' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>+ preference optimization (RLHF / DPO)</text></g></svg>",
     "note":"<b>Preferences add judgment.</b> Learn from human comparisons (win ≻ lose) to improve on qualities you can only compare, not demonstrate."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>helpful, honest assistant</text></g></svg>",
     "note":"<b>The result.</b> A model that follows instructions and reflects human preferences — assembled from these three stages, each covered next."},
  ],
  quiz=[
    {"q":"1. What does supervised fine-tuning (SFT) mainly teach a pretrained model?","opts":["new world knowledge","the instruction→answer format (how to behave like an assistant)","how to pretrain","the discount factor"],"ans":1,"fb":"SFT copies good instruction→answer pairs, teaching the assistant format. Broad knowledge comes from pretraining."},
    {"q":"2. Why use human preferences instead of just more labelled answers?","opts":["preferences are always cheaper to store","comparing two answers is easier and more reliable than writing the single perfect one, and captures fuzzy qualities","answers are illegal","to avoid SFT"],"ans":1,"fb":"For qualities like helpfulness or tone there's no single correct answer; humans compare far more reliably than they author perfect targets."},
    {"q":"3. Why must SFT come before RLHF/DPO?","opts":["it's arbitrary","preference methods make small improvements around a good starting policy; on a raw model they wander into gibberish","RLHF can't run after SFT","SFT needs preferences first"],"ans":1,"fb":"Preference optimization polishes a decent policy; without SFT to reach the right neighbourhood, it destabilizes."},
    {"q":"4. Your chat preference score rises but the model gets worse at math it used to solve. This is:","opts":["impossible","the alignment tax / capability regression — optimizing preferences can quietly erode other skills","a reward model bug only","expected and harmless"],"ans":1,"fb":"The alignment tax: preference tuning can degrade prior capabilities. You catch it only by continuing to measure those old skills, not just the preference score."},
  ],
  fin={"em":"🗺️","h3":"Day 1 complete — you have the whole map!",
       "p":"You now know the post-training pipeline: pretrain (knowledge) → SFT (format) → preference optimization (judgment), why comparisons beat authored answers, and why order matters. Every remaining lesson is one box on this map. Next: <b>Day 2</b> — supervised fine-tuning up close."},
))

# ---------------- Day 2 — Supervised Fine-Tuning ----------------
L("day-02-sft.html", dict(
  qid="m21a-d02-sft", title="Module 21a · Day 2 — Supervised Fine-Tuning", nav_title="Spiral · M21a Day 2",
  eyebrow="Module 21a · Train · Day 2", h1="Supervised Fine-Tuning & Instruction Tuning",
  lead="SFT is the first and simplest post-training step: keep predicting the next token, but now on curated instruction→answer examples instead of raw web text, and only score the answer tokens. Do it across thousands of diverse tasks and the model learns the general skill of \"follow the instruction\" — that's instruction tuning.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how SFT reuses the next-token loss, why we mask the prompt tokens from the loss, and what makes instruction tuning generalize across tasks.",
  prev_href="day-01-pipeline.html", prev_label="The Pipeline", next_href="day-03-lora-qlora.html", next_label="LoRA & QLoRA",
  sections=[
    {"title":"Same loss, better data","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (the pipeline) and cross-entropy / next-token loss (M2, M5a).</div></div>
     <h4>SFT is next-token prediction, curated</h4>
     <p><span class="term" data-tip="Supervised fine-tuning: continue training the pretrained model on curated instruction→answer examples, using the same next-token cross-entropy loss.">SFT</span> uses the exact same loss as pretraining — predict the next token — but on hand-picked examples formatted as an instruction and a good answer. No new machinery; just far better, task-shaped data.</p>
     <h4>The example format</h4>
     <p>Each example is a prompt (system message + user instruction) followed by the target answer. The model reads the prompt and learns to generate the answer that follows.</p>
     <h4>Instruction tuning</h4>
     <p>Do SFT across <em>thousands of different tasks</em> — summarize, translate, explain, refuse — and the model learns the general behaviour "read the instruction, do what it asks." That generalization to unseen instructions is <span class="term" data-tip="SFT on a large, diverse set of instruction→answer tasks so the model learns to follow instructions in general, including ones not seen in training.">instruction tuning</span>.</p>''',
     "gotit":"Got the idea"},
    {"title":"Learning the house style","body":
     '''<div class="relate">
       <div class="card"><span class="big">✍️</span><h5>Copy good examples</h5><p>A new writer studies a binder of excellent letters and drafts their own to match. They already know grammar (pretraining); the binder teaches the house style. That binder is your SFT dataset.</p></div>
       <div class="card"><span class="big">🧩</span><h5>Many kinds of letters</h5><p>Show them complaints, thank-yous, refunds, refusals — dozens of kinds — and they learn to handle a <i>new</i> kind they've never seen. That generalization is instruction tuning.</p></div>
     </div>
     <p><strong>In one line:</strong> feed the model many good instruction→answer examples and it learns both the format and the general skill of following instructions.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a writer understands <i>why</i> a letter is good. SFT only imitates the surface pattern — it copies the demonstrated answers, so if your examples are subtly wrong, it will faithfully copy the mistakes.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Prompt masking in action","body":'''<p>Watch an example get tokenized, the prompt tokens masked from the loss, and the answer tokens scored. <strong>Click all three.</strong></p>'''},
    {"title":"Mask the prompt, score the answer","body":
     '''<h4>Step 1 — the loss</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       loss = − Σ<sub>t ∈ answer</sub> log p(token<sub>t</sub> | tokens<sub>&lt;t</sub>)<br>
       <span class="dim"># standard next-token cross-entropy, summed over ANSWER tokens only</span>
     </div></div>
     <h4>Step 2 — why mask the prompt</h4>
     <p>We only want the model to learn to <em>produce</em> answers, not to reproduce user instructions. So we compute the loss only on the answer tokens and <span class="term" data-tip="Setting the loss weight to zero on prompt tokens so the model is scored only on generating the answer, not on parroting the instruction.">mask</span> (zero out) the loss on the prompt tokens. The prompt is context, not a target.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       tokens:  [Q: What is 2+2?]  [A:]  [4]  [&lt;eos&gt;]<br>
       mask:    [ 0  0   0   0  ]  [ 0]  [1]  [ 1 ]<br>
       <span class="dim"># loss counted only on "4" and end-of-sequence → learns to answer, not to echo</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — overfitting to a narrow SFT set.</b> Train too long, or on a small/repetitive dataset, and the model memorizes phrasings and <b>catastrophically forgets</b> pretraining knowledge and diversity. Its held-out instruction-following drops even as the training loss looks perfect. Guard with a diverse dataset, few epochs (often 1–3), and a held-out eval.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — data quality vs quantity.</b> A small set of <i>excellent</i>, diverse examples often beats a huge noisy one (the \"LIMA\" lesson: ~1k great examples go a long way). More data helps only if it stays high-quality and varied; noisy demonstrations get copied verbatim, mistakes and all.</div></div>''',
     "gotit":"Got prompt masking"},
    {"title":"An SFT example, tokenized","body":'''<p>Here is one SFT step built up: the formatted example, tokenization, the prompt mask, the scored answer tokens, and the gradient. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a tiny SFT run","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_sft.py</code>. Take a small model (or the transformer you built) and fine-tune it on a handful of instruction→answer pairs, masking the loss on prompt tokens. Show a before/after generation on a held-out instruction.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 2 artifact.
Create experiments/foundations/m21a_sft.py: SFT a small HuggingFace model on ~20 instruction->answer pairs, masking the loss on prompt tokens (label = -100 for prompt tokens). Print a held-out generation before and after, and note how few epochs before it overfits.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-02-log.md</code>: did masking the prompt change the output, and after how many epochs did quality start to drop?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "tok":{"btn":"① tokenize the example","html":'''<span class="prompt">&gt;&gt;&gt;</span> ex = "Q: What is 2+2?\\nA: 4"
<span class="prompt">&gt;&gt;&gt;</span> ids = tokenize(ex)
<span class="hl">[Q][:][ What][ is][ 2][+][2][?][\\n][A][:][ 4]</span>''',
      "take":"<b>①  Format and tokenize.</b> The instruction and answer become one token sequence. The model will read the question and learn to produce the answer."},
    "mask":{"btn":"② mask the prompt","html":'''<span class="prompt">&gt;&gt;&gt;</span> labels = mask_prompt(ids)
<span class="hl">prompt tokens → -100 (ignored)</span>
<span class="hl">answer tokens → kept</span>
<span class="dim"># loss is computed ONLY on " 4" and &lt;eos&gt;</span>''',
      "take":"<b>②  Mask the prompt.</b> Prompt tokens get label −100 so they don't contribute to the loss. The model is scored only on generating the answer."},
    "loss":{"btn":"③ score the answer","html":'''<span class="prompt">&gt;&gt;&gt;</span> loss = cross_entropy(logits, labels)
<span class="hl">loss = 0.42</span>   <span class="dim"># only over answer tokens</span>
<span class="prompt">&gt;&gt;&gt;</span> loss.backward()   <span class="dim"># update weights</span>''',
      "take":"<b>③  Same next-token loss.</b> Standard cross-entropy over the answer tokens — no new machinery, just curated data and a mask."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>\"Q: What is 2+2?  A: 4\"</text></g></svg>",
     "note":"<b>A formatted example.</b> Instruction plus a good answer, written in the assistant format."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='300' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='190' y='35' fill='#1F6280'>[Q][What][is][2][+][2][?]</text><rect x='350' y='18' width='120' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='410' y='35' fill='#9A7208'>[A][4][eos]</text></g></svg>",
     "note":"<b>Tokenize.</b> Prompt tokens (blue) then answer tokens (amber) in one sequence."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='300' height='26' rx='5' fill='#F5F0E8' stroke='#6B645E' stroke-dasharray='4,3'/><text x='190' y='35' fill='#6B645E'>prompt → masked (label −100)</text><rect x='350' y='18' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='410' y='35' fill='#1a5c38'>answer → scored</text></g></svg>",
     "note":"<b>Mask the prompt.</b> Prompt tokens are ignored in the loss; only the answer tokens are targets."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='130' y='16' width='260' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>loss = −Σ log p(answer token)</text></g></svg>",
     "note":"<b>Score the answer.</b> Standard next-token cross-entropy over the answer tokens only."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='100' y='16' width='320' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>× thousands of diverse tasks = instruction tuning</text></g></svg>",
     "note":"<b>Repeat across many tasks.</b> Diverse instruction→answer data teaches the general skill of following instructions, even unseen ones."},
  ],
  quiz=[
    {"q":"1. What loss does SFT use?","opts":["a brand-new preference loss","the same next-token cross-entropy as pretraining, on curated data","mean squared error on rewards","no loss"],"ans":1,"fb":"SFT reuses next-token cross-entropy; the only change is curated instruction→answer data (and prompt masking)."},
    {"q":"2. Why do we mask the prompt tokens from the loss?","opts":["to save memory","so the model learns to generate answers, not to parrot instructions","because prompts are secret","to increase γ"],"ans":1,"fb":"We only score answer tokens (prompt labels = −100), so the model learns to produce the answer given the prompt as context."},
    {"q":"3. What makes instruction tuning generalize to unseen instructions?","opts":["a bigger learning rate","SFT across thousands of diverse tasks, teaching the general \"follow the instruction\" skill","using only one task","freezing all weights"],"ans":1,"fb":"Breadth of tasks is the key: diverse instruction→answer data teaches the general behaviour, so new instructions transfer."},
    {"q":"4. Your SFT training loss is near zero but held-out instruction-following got worse. Most likely:","opts":["perfect training","overfitting / catastrophic forgetting from too many epochs on a narrow set","the mask is wrong by design","γ too high"],"ans":1,"fb":"Over-training on a small/repetitive set memorizes phrasings and erodes pretraining knowledge. Use diverse data, few epochs, and a held-out eval."},
  ],
  fin={"em":"✍️","h3":"Day 2 complete — the model follows instructions!",
       "p":"You now know SFT: the same next-token loss on curated instruction→answer data, with the prompt masked so only answers are scored. Across thousands of diverse tasks that becomes instruction tuning — general instruction-following. You also met overfitting/catastrophic forgetting and the quality-over-quantity rule. Next: <b>Day 3</b> — how to fine-tune cheaply with LoRA and QLoRA."},
))
# ---------------- Day 3 — PEFT: LoRA & QLoRA ----------------
L("day-03-lora-qlora.html", dict(
  qid="m21a-d03-lora", title="Module 21a · Day 3 — LoRA & QLoRA", nav_title="Spiral · M21a Day 3",
  eyebrow="Module 21a · Train · Day 3", h1="PEFT: LoRA & QLoRA",
  lead="Full fine-tuning updates every weight in the model — billions of numbers, needing huge memory. LoRA does something clever: freeze the whole pretrained model and train only a tiny pair of low-rank matrices bolted onto each layer. You get almost the same quality for a fraction of the memory and storage. QLoRA pushes it further by quantizing the frozen base to 4 bits, so you can fine-tune a large model on a single GPU.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why LoRA freezes the base and trains a low-rank update <code>W + BA</code>, count how many parameters that saves, and say what QLoRA adds on top.",
  prev_href="day-02-sft.html", prev_label="Supervised Fine-Tuning", next_href="day-04-reward-model.html", next_label="Reward Models",
  sections=[
    {"title":"Don't retrain everything","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (SFT), and that a layer holds a big weight matrix <code>W</code> you multiply the input by (M4/M5a).</div></div>
     <h4>Full fine-tuning is expensive</h4>
     <p>Full fine-tuning updates every weight. For a 7-billion-parameter model that means storing gradients and optimizer state for all 7 billion numbers — often 4× the model size in memory. Most people cannot afford that.</p>
     <h4>The PEFT idea</h4>
     <p><span class="term" data-tip="Parameter-Efficient Fine-Tuning: freeze the pretrained weights and train only a small number of new parameters, cutting memory and storage.">PEFT (parameter-efficient fine-tuning)</span> freezes the pretrained model and trains only a few new parameters. The most popular kind is <span class="term" data-tip="Low-Rank Adaptation: freeze W and learn a small low-rank update B·A added to it, training only B and A.">LoRA (low-rank adaptation)</span>.</p>
     <h4>LoRA in one sentence</h4>
     <p>Instead of changing <code>W</code>, LoRA learns a small correction <code>ΔW = B·A</code> and uses <code>W + BA</code>. Here <code>A</code> and <code>B</code> are skinny matrices (rank <code>r</code>, maybe 8), so together they hold a tiny fraction of <code>W</code>'s numbers. You train only <code>A</code> and <code>B</code>; <code>W</code> never moves.</p>''',
     "gotit":"Got the idea"},
    {"title":"Sticky notes on a textbook","body":
     '''<div class="relate">
       <div class="card"><span class="big">📓</span><h5>Don't rewrite the book</h5><p>You have a huge textbook (the pretrained model). To adapt it for one class, you don't reprint it — you add small sticky notes in the margins. The book stays fixed; the notes carry your changes. Those sticky notes are the LoRA matrices.</p></div>
       <div class="card"><span class="big">🔁</span><h5>Swap the notes per task</h5><p>For a different class, peel off these notes and stick on another set. One shared book, many small note-sets — that's why teams keep one base model and many tiny LoRA adapters.</p></div>
     </div>
     <p><strong>In one line:</strong> freeze the big model and train only small add-on matrices, so fine-tuning is cheap and each task is just a tiny set of notes.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: sticky notes sit on top of the page, but LoRA's <code>BA</code> is added <i>inside</i> the math of every adapted layer — it changes the actual computation, not just an annotation.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Count the parameters","body":'''<p>Watch the parameter count for full fine-tuning, then LoRA, then QLoRA's memory. <strong>Click all three.</strong></p>'''},
    {"title":"Low rank, few parameters","body":
     '''<h4>Step 1 — the update in words</h4>
     <p>A weight matrix <code>W</code> has shape <code>d × k</code>. LoRA replaces the update to <code>W</code> with the product of two thin matrices: <code>B</code> (shape <code>d × r</code>) and <code>A</code> (shape <code>r × k</code>), where the <span class="term" data-tip="The inner dimension r of the low-rank matrices B (d×r) and A (r×k). Small r (e.g. 8) means few trainable parameters.">rank</span> <code>r</code> is small.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       output = (W + B·A) · x        <span class="dim"># W frozen; only B and A are trained</span><br>
       W: d×k (frozen)   B: d×r   A: r×k   with r ≪ d, k<br>
       trainable params = r·(d + k)   instead of   d·k
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       one layer with d = k = 4096, rank r = 8<br><br>
       full fine-tune: 4096 × 4096 = <span class="hl">16,777,216</span> params<br>
       LoRA: r·(d+k) = 8 × (4096 + 4096) = <span class="hl">65,536</span> params<br>
       ratio ≈ 256× fewer trainable parameters — for near-equal quality
     </div></div>
     <p>QLoRA goes further: it stores the frozen base weights in a compact <span class="term" data-tip="4-bit NormalFloat: a 4-bit number format QLoRA uses to store the frozen base weights, cutting memory ~4× while the LoRA adapters stay in higher precision.">4-bit (NF4)</span> format and keeps the LoRA matrices in higher precision. Gradients flow through the dequantized base into <code>A</code> and <code>B</code>. That is how a 65B model fine-tunes on one 48GB GPU.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — rank too small.</b> If <code>r</code> is too low for the task, the adapter simply cannot represent the needed change: training loss plateaus higher than full fine-tuning and it looks like the model "can't learn," when really the low-rank bottleneck is starving it. There's no error — just a quietly worse ceiling. Fix: raise <code>r</code> (or apply LoRA to more layers) for harder tasks.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — efficiency vs capacity (and QLoRA's quant error).</b> LoRA buys huge memory/storage savings and swappable per-task adapters, but a low-rank update has a capacity ceiling — for a big behaviour change, full fine-tuning still wins. QLoRA saves even more memory but the 4-bit base adds small quantization error, a tiny quality cost for a large accessibility gain.</div></div>''',
     "gotit":"Got low-rank updates"},
    {"title":"LoRA, built up","body":'''<p>Here is LoRA assembled: the frozen W, the thin B and A, their product added to W, and QLoRA's 4-bit base. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a LoRA fine-tune","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_lora.py</code>. Take a small model, wrap its attention/MLP weight matrices with a LoRA adapter (rank 8), freeze the base, and fine-tune on your Day 2 SFT data. Print the trainable-parameter count vs the full count, and confirm quality is close.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 3 artifact.
Create experiments/foundations/m21a_lora.py: LoRA-fine-tune a small HuggingFace model with peft (rank 8 on attention projections), freeze the base, and train on ~20 instruction pairs. Print trainable params vs total params (the ~1% ratio), and a before/after generation. Add a comment on what QLoRA would change (4-bit NF4 base).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-03-log.md</code>: what fraction of parameters were trainable, and did quality hold up?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "full":{"btn":"① full fine-tune","html":'''<span class="prompt">&gt;&gt;&gt;</span> layer W: shape (4096, 4096)
<span class="prompt">&gt;&gt;&gt;</span> trainable = 4096 * 4096
<span class="hl">16,777,216</span>   <span class="dim"># every weight, per layer — huge</span>''',
      "take":"<b>①  Full fine-tuning trains everything.</b> ~16.8M parameters for one layer, plus gradients and optimizer state — 4× the model size in memory."},
    "lora":{"btn":"② LoRA (rank 8)","html":'''<span class="prompt">&gt;&gt;&gt;</span> B: (4096, 8),  A: (8, 4096)   <span class="dim"># frozen W stays put</span>
<span class="prompt">&gt;&gt;&gt;</span> trainable = 8 * (4096 + 4096)
<span class="hl">65,536</span>   <span class="dim"># ~256x fewer than full</span>''',
      "take":"<b>②  LoRA trains a thin add-on.</b> Just B and A (rank 8) — 65,536 params vs 16.8M. Same layer behaviour, a tiny fraction of the cost."},
    "qlora":{"btn":"③ QLoRA memory","html":'''<span class="prompt">&gt;&gt;&gt;</span> base weights → 4-bit NF4 (frozen)
<span class="prompt">&gt;&gt;&gt;</span> LoRA adapters → bf16 (trained)
<span class="hl">65B model fits on one 48GB GPU</span>   <span class="dim"># ~4x memory cut</span>''',
      "take":"<b>③  QLoRA quantizes the base.</b> Store frozen weights in 4-bit, train LoRA on top. A model that needed many GPUs now fits on one."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='190' y='20' width='140' height='44' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='40' fill='#5A544E'>W  (d×k)</text><text x='260' y='56' fill='#6B645E'>frozen ❄️</text></g></svg>",
     "note":"<b>Start with the frozen weight W.</b> The big pretrained matrix. In LoRA it never changes — no gradients, no optimizer state for it."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='25' width='60' height='44' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='150' y='51' fill='#5E5191'>B (d×r)</text><rect x='210' y='35' width='120' height='24' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='270' y='51' fill='#5E5191'>A (r×k)</text><text x='360' y='51' fill='#6B645E'>r = 8 (tiny)</text></g></svg>",
     "note":"<b>Add two thin matrices B and A.</b> Their shared inner size is the rank r (say 8). These are the only trainable parameters."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='20' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='40' fill='#1F6280'>output = (W + B·A) · x</text></g></svg>",
     "note":"<b>Add the product to W.</b> The layer computes with W + BA. W supplies the pretrained knowledge; BA supplies the small learned correction."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='180' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='150' y='42' fill='#C93B3B'>full: 16,777,216</text><rect x='300' y='24' width='160' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='380' y='42' fill='#1a5c38'>LoRA: 65,536</text></g></svg>",
     "note":"<b>~256× fewer trainable parameters.</b> r·(d+k) instead of d·k. Cheaper memory, and each fine-tune is a tiny file you can swap per task."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='22' width='130' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='185' y='41' fill='#9A7208'>base → 4-bit</text><rect x='280' y='22' width='130' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='345' y='41' fill='#5E5191'>LoRA → bf16</text></g></svg>",
     "note":"<b>QLoRA quantizes the frozen base to 4 bits.</b> The base shrinks ~4×; the small LoRA adapters stay higher-precision and trainable. Big models now fine-tune on one GPU."},
  ],
  quiz=[
    {"q":"1. In LoRA, which parameters are actually trained?","opts":["all of W","only the small matrices B and A (W stays frozen)","only the biases","the tokenizer"],"ans":1,"fb":"LoRA freezes W and trains only the low-rank add-on B·A, which is a tiny fraction of the parameters."},
    {"q":"2. A layer has W of shape 1000×1000. With LoRA rank r=4, how many trainable parameters?","opts":["1,000,000","8,000","4,000","2,000"],"ans":1,"fb":"r·(d+k) = 4·(1000+1000) = 8,000 — versus 1,000,000 for full fine-tuning (125× fewer)."},
    {"q":"3. What does QLoRA add on top of LoRA?","opts":["it trains all weights","it stores the frozen base in 4-bit (NF4) to cut memory, keeping LoRA adapters higher-precision","it removes the adapters","it doubles the rank"],"ans":1,"fb":"QLoRA quantizes the frozen base to 4 bits (~4× memory cut) while LoRA adapters stay trainable in higher precision."},
    {"q":"4. Your LoRA fine-tune plateaus at a worse loss than full fine-tuning and won't improve. Likely cause?","opts":["the base is unfrozen","the rank r is too small to represent the needed change (capacity bottleneck)","QLoRA is illegal","the learning rate is zero by design"],"ans":1,"fb":"Too-low rank starves the adapter of capacity — a silent ceiling. Raise r or apply LoRA to more layers for harder tasks."},
  ],
  fin={"em":"📓","h3":"Day 3 complete — you can fine-tune cheaply!",
       "p":"You now know PEFT: LoRA freezes W and trains a low-rank add-on <code>W + BA</code>, cutting trainable parameters ~100–256×, and QLoRA quantizes the frozen base to 4-bit so large models fine-tune on one GPU. You also saw the rank capacity ceiling. Next: <b>Day 4</b> — building the reward model that scores responses from human preferences."},
))
# ---------------- Day 4 — Reward Models ----------------
L("day-04-reward-model.html", dict(
  qid="m21a-d04-rm", title="Module 21a · Day 4 — Reward Models", nav_title="Spiral · M21a Day 4",
  eyebrow="Module 21a · Train · Day 4", h1="Reward Models: Turning Preferences into a Score",
  lead="To improve a model with reinforcement learning, you need a reward — a number saying how good a response is. But humans give <em>comparisons</em> (\"this reply is better\"), not scores. A reward model learns to turn those comparisons into a single number, so it can score any new response. The trick that makes this work is a 100-year-old formula for turning wins into skill ratings: Bradley-Terry.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain what a reward model outputs, write the Bradley-Terry loss <code>−log σ(r_w − r_l)</code>, and compute the preference probability from two scores.",
  prev_href="day-03-lora-qlora.html", prev_label="LoRA & QLoRA", next_href="day-05-rm-failure-modes.html", next_label="RM Failure Modes",
  sections=[
    {"title":"A model that scores answers","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (preferences: y_win ≻ y_lose) and the sigmoid function σ (from the foundations).</div></div>
     <h4>Why we need a score</h4>
     <p>RLHF optimizes a policy against a reward. Humans do not hand out scores reliably — but they <em>can</em> reliably say which of two answers is better. A <span class="term" data-tip="A model that takes a prompt and a response and outputs a single number scoring how good the response is, trained from human preference comparisons.">reward model (RM)</span> learns from those comparisons to output a score for <em>any</em> response.</p>
     <h4>How it's built</h4>
     <p>Take a pretrained (usually SFT) model, replace its word-prediction head with a single-number head, and feed it a prompt + response. The output is the reward <code>r(x, y)</code>.</p>
     <h4>The training signal</h4>
     <p>The data is pairs: for a prompt <code>x</code>, a preferred answer <code>y_w</code> (win) and a rejected one <code>y_l</code> (lose). We train the RM so it scores the winner higher than the loser. Do this over many pairs and it learns a general sense of "good." The next question is exactly <em>how</em> to turn "winner &gt; loser" into a loss — that's Bradley-Terry.</p>''',
     "gotit":"Got the idea"},
    {"title":"Chess ratings from match results","body":
     '''<div class="relate">
       <div class="card"><span class="big">♟️</span><h5>Wins become ratings</h5><p>In chess, nobody assigns you a skill number directly. You play games, and a rating system turns your wins and losses into an Elo number. A reward model does the same: it turns "answer A beat answer B" into a hidden quality score.</p></div>
       <div class="card"><span class="big">📈</span><h5>Bigger gap, surer win</h5><p>If a 2500-rated player meets a 1500, we're almost sure who wins. If both are 2000, it's a coin flip. The score <i>difference</i> predicts the win probability — exactly the Bradley-Terry idea.</p></div>
     </div>
     <p><strong>In one line:</strong> a reward model is a rating system for answers — it learns a hidden score so that better answers score higher, trained only on which answer won.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: chess results are objective (someone truly wins). Preference labels are subjective and noisy — two annotators may disagree — so the RM learns the <i>average</i> human taste, warts and all.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Score a pair","body":'''<p>Watch a preference pair, the RM's two scores, and the Bradley-Terry loss. <strong>Click all three.</strong></p>'''},
    {"title":"The Bradley-Terry loss","body":
     '''<h4>Step 1 — from scores to a probability</h4>
     <p>Give the RM's score to the winner (<code>r_w</code>) and loser (<code>r_l</code>). The <span class="term" data-tip="A model that turns the difference of two scores into the probability that the higher-scored item is preferred: P = sigmoid(r_w − r_l).">Bradley-Terry</span> model says the chance the winner is preferred is the sigmoid of the score gap.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       P(y_w ≻ y_l) = σ(r_w − r_l)          <span class="dim"># σ = sigmoid, squashes to 0..1</span><br>
       loss = − log σ(r_w − r_l)             <span class="dim"># maximize the winner's margin</span>
     </div></div>
     <p>Symbols: <code>r_w</code>, <code>r_l</code> = the RM's scores for winner and loser. <code>σ(z) = 1/(1+e^−z)</code>. Training pushes <code>r_w</code> above <code>r_l</code> — it grows the gap until the winner is clearly preferred.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       RM scores: r_w = 2.0,  r_l = 0.5<br><br>
       gap = 2.0 − 0.5 = 1.5<br>
       P(win) = σ(1.5) = 1 / (1 + e^−1.5) = 1 / (1 + 0.223) = <span class="hl">0.82</span><br>
       loss = −log(0.82) = <span class="hl">0.20</span>   <span class="dim"># small → the RM already ranks this pair well</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the RM is only as good as its pairs.</b> The RM has no ground truth; it copies whatever the labels imply. If annotators systematically prefer longer or more confident-sounding answers, the RM bakes that in and scores those higher — even when they're wrong. Its loss looks great on the training pairs while it quietly learns the annotators' biases. You catch it only by auditing what high-reward responses actually look like (tomorrow's whole lesson).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — a single scalar reward.</b> Collapsing "good" into one number makes it a flexible RL signal you can optimize against — but it's a learned <i>proxy</i>, not the true goal, so it can be gamed (Goodhart). A richer signal (per-attribute scores, rules, human-in-the-loop) resists gaming better but is costlier and harder to optimize.</div></div>''',
     "gotit":"Got Bradley-Terry"},
    {"title":"Building the reward model","body":'''<p>Here it is assembled: the pair, the shared RM scoring each, the score gap, the sigmoid, and the loss. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a tiny RM","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_reward_model.py</code>. Add a scalar head to a small model, feed it (prompt, response) pairs with a chosen/rejected label, and train with the Bradley-Terry loss <code>-log σ(r_chosen − r_rejected)</code>. Print the accuracy at ranking held-out pairs.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 4 artifact.
Create experiments/foundations/m21a_reward_model.py: a scalar-head reward model over a small encoder; train on ~30 (prompt, chosen, rejected) triples with loss = -log sigmoid(r_chosen - r_rejected). Print pairwise ranking accuracy on a held-out set and one example (r_chosen, r_rejected, P(win)).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-04-log.md</code>: what pairwise accuracy did the RM reach, and did it ever prefer the longer answer regardless of quality?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "pair":{"btn":"① a preference pair","html":'''<span class="prompt">&gt;&gt;&gt;</span> x = "Explain gravity to a child."
<span class="hl">y_win </span>= "Things fall because Earth gently pulls them down..."
<span class="hl">y_lose</span>= "Gravity is a fundamental interaction described by..."
<span class="dim"># a human picked y_win as the better reply</span>''',
      "take":"<b>①  Humans give a comparison.</b> For one prompt, a chosen (win) and rejected (lose) answer. No numeric score — just which is better."},
    "score":{"btn":"② the RM scores both","html":'''<span class="prompt">&gt;&gt;&gt;</span> r_w = reward_model(x, y_win)
<span class="prompt">&gt;&gt;&gt;</span> r_l = reward_model(x, y_lose)
<span class="hl">r_w = 2.0    r_l = 0.5</span>   <span class="dim"># one number each</span>''',
      "take":"<b>②  The RM turns each into a score.</b> Same model, run on both answers. We want the winner's score above the loser's."},
    "loss":{"btn":"③ Bradley-Terry loss","html":'''<span class="prompt">&gt;&gt;&gt;</span> P = sigmoid(r_w - r_l)   <span class="dim"># sigmoid(1.5)</span>
<span class="hl">P = 0.82</span>   <span class="dim"># 82% chance winner is preferred</span>
<span class="prompt">&gt;&gt;&gt;</span> loss = -log(P)
<span class="hl">loss = 0.20</span>   <span class="dim"># small → already ranks it right</span>''',
      "take":"<b>③  Grow the winner's margin.</b> The loss −log σ(r_w−r_l) pushes r_w above r_l. Over many pairs the RM learns a general score for \"good.\""},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='170' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='145' y='43' fill='#1a5c38'>y_win (chosen)</text><rect x='290' y='24' width='170' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='375' y='43' fill='#C93B3B'>y_lose (rejected)</text></g></svg>",
     "note":"<b>Start with a preference pair.</b> Same prompt, two answers, a human label saying which is better. This is the only supervision."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='180' y='16' width='160' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='34' fill='#1F6280'>reward model r(x,y)</text><text x='120' y='64' fill='#2D8B55'>r_w = 2.0</text><text x='400' y='64' fill='#C93B3B'>r_l = 0.5</text></g></svg>",
     "note":"<b>Score each answer.</b> The same reward model reads (prompt, answer) and outputs one number for each."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='170' y='16' width='180' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>gap = r_w − r_l = 1.5</text></g></svg>",
     "note":"<b>Take the score gap.</b> How much higher the winner scored. A bigger gap means the RM is surer the winner is better."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>P = σ(1.5) = 0.82</text></g></svg>",
     "note":"<b>Sigmoid turns the gap into a probability.</b> A gap of 1.5 → 82% chance the winner is preferred. Bradley-Terry links score difference to win chance."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>loss = −log σ(r_w − r_l)</text></g></svg>",
     "note":"<b>The loss grows the winner's margin.</b> Minimizing it pushes r_w above r_l on every pair. The RM ends up scoring good answers high — a reward for RLHF."},
  ],
  quiz=[
    {"q":"1. What does a reward model output?","opts":["the next token","a single number scoring how good a (prompt, response) is","a probability distribution over words","a class label"],"ans":1,"fb":"An RM outputs one scalar reward for a prompt+response, learned from human preference comparisons."},
    {"q":"2. The RM is trained from:","opts":["numeric scores humans write","preference comparisons (which of two answers is better)","raw pretraining text","its own outputs only"],"ans":1,"fb":"Humans compare pairs reliably; the RM learns a hidden score consistent with those win/lose comparisons (Bradley-Terry)."},
    {"q":"3. RM scores are r_w = 1.0 and r_l = 1.0. What is P(winner preferred)?","opts":["1.0","0.5","0.0","0.82"],"ans":1,"fb":"P = σ(r_w − r_l) = σ(0) = 0.5 — equal scores mean a coin flip, exactly as expected."},
    {"q":"4. Your RM gives near-perfect training loss but keeps scoring longer answers higher regardless of quality. This is:","opts":["a bug in sigmoid","the RM copying an annotator length bias — it's only as good as its pairs","impossible with Bradley-Terry","proof it's well-calibrated"],"ans":1,"fb":"The RM has no ground truth; it learns whatever the labels imply. If annotators favored length, the RM bakes that in — a silent failure you catch by auditing high-reward outputs."},
  ],
  fin={"em":"♟️","h3":"Day 4 complete — you can score answers!",
       "p":"You now know the reward model: a scalar-head model trained from preference pairs with the Bradley-Terry loss <code>−log σ(r_w − r_l)</code>, where the score gap sets the preference probability. You also saw its core weakness — it only mirrors its labels. Next: <b>Day 5</b> — exactly how reward models fail, and how to defend against it."},
))

# ---------------- Day 5 — Reward Model Failure Modes ----------------
L("day-05-rm-failure-modes.html", dict(
  qid="m21a-d05-rmfail", title="Module 21a · Day 5 — RM Failure Modes", nav_title="Spiral · M21a Day 5",
  eyebrow="Module 21a · Train · Day 5", h1="When the Reward Model Lies",
  lead="A reward model is a learned <em>proxy</em> for human judgment, not the real thing. The moment you optimize a policy hard against it, the policy finds the cracks — answers that score high on the RM but are actually worse. This is reward overoptimization, and understanding it is the difference between an aligned model and a confidently wrong one.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain Goodhart's law / reward overoptimization, name concrete RM biases (length, sycophancy), and describe the defenses (KL penalty, RM ensembles, fresh labels).",
  prev_href="day-04-reward-model.html", prev_label="Reward Models", next_href="day-06-ppo-for-llms.html", next_label="PPO for LLMs",
  sections=[
    {"title":"Optimizing a proxy","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 4 (the reward model scores responses). Today: what happens when you push a policy to maximize that score.</div></div>
     <h4>The proxy problem</h4>
     <p>The RM approximates human judgment but is imperfect — it has blind spots. When the policy only mildly optimizes the RM, higher reward really does mean better answers. But push harder and the policy starts exploiting the RM's mistakes: it finds answers the RM <em>scores</em> high that humans would rate low. This is <span class="term" data-tip="Reward overoptimization: as a policy optimizes a proxy reward too hard, true quality first rises then falls, because the policy exploits the reward model's errors.">reward overoptimization</span>.</p>
     <h4>Goodhart's law</h4>
     <p>"When a measure becomes a target, it ceases to be a good measure." The RM was a good <em>measure</em> of quality. Make it the <em>target</em> of hard optimization and the policy games it. That is <span class="term" data-tip="Goodhart's law: once you optimize a proxy metric as a target, agents exploit the gap between the proxy and the true goal.">Goodhart's law</span> in action.</p>
     <h4>Concrete biases</h4>
     <ul>
       <li><b>Length bias:</b> RMs often over-reward longer answers, so the policy learns to pad.</li>
       <li><b>Sycophancy:</b> RMs reward agreeable, flattering replies, so the policy learns to tell you what you want to hear.</li>
       <li><b>Distribution shift:</b> the RM was trained on the SFT model's answers; as the policy changes, it produces answers the RM never saw and mis-scores.</li>
     </ul>''',
     "gotit":"Got the proxy problem"},
    {"title":"Teaching to the test","body":
     '''<div class="relate">
       <div class="card"><span class="big">📝</span><h5>The score goes up</h5><p>A school is judged by test scores, so it drills test tricks. Scores rise — but real learning does not. The metric (test score) and the goal (learning) come apart. The RM score is the test; true helpfulness is the learning.</p></div>
       <div class="card"><span class="big">🎈</span><h5>Padding to look thorough</h5><p>A student who learns "longer answers get more marks" writes pages of fluff. The grader's bias (length) becomes the student's strategy. That's exactly RM length bias driving a padded policy.</p></div>
     </div>
     <p><strong>In one line:</strong> optimize the score instead of the goal, and the policy will chase the score's blind spots — looking better while getting worse.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a school knows it's gaming the test. The policy has no intent — it just follows the gradient toward higher reward. The gaming is automatic, which is why it needs mechanical defenses, not good faith.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Watch overoptimization","body":'''<p>Watch the RM score climb while true quality peaks then falls, and the KL from the reference grow. <strong>Click all three.</strong></p>'''},
    {"title":"The overoptimization curve & defenses","body":
     '''<h4>Step 1 — proxy vs true reward</h4>
     <p>Plot two things as you optimize: the RM (proxy) reward and the true human-rated quality, both against how far the policy has moved from the start (measured as <span class="term" data-tip="Kullback–Leibler divergence: here, how far the optimized policy has drifted from the reference (SFT) model. More optimization = larger KL.">KL divergence</span> from the reference model).</p>
     <h4>Step 2 — the shape</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       proxy reward (RM):  rises, then keeps rising  ↗↗<br>
       true quality:       rises, PEAKS, then FALLS  ↗↘<br>
       x-axis = KL(policy ‖ reference)  — how hard you optimized<br>
       <span class="dim"># the gap after the peak is pure reward hacking</span>
     </div></div>
     <p>Early on, both rise together — optimizing helps. Past a point, the RM score keeps climbing while true quality drops. The best model is at the <em>peak of true quality</em>, not the peak of RM reward.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       KL = 5:   RM = 0.6,  true = 0.6   <span class="dim"># still improving together</span><br>
       KL = 20:  RM = 0.9,  true = <span class="hl">0.75</span>  <span class="dim"># ← true-quality peak (stop near here)</span><br>
       KL = 50:  RM = 1.1,  true = 0.55  <span class="dim"># RM up, true DOWN → over-optimized</span>
     </div></div>
     <h4>The defenses</h4>
     <ul>
       <li><b>KL penalty:</b> add a cost for drifting from the reference model, so the policy can't wander into the RM's blind spots (this is the β in tomorrow's PPO).</li>
       <li><b>RM ensembles:</b> average several reward models; the policy can't game all of them at once as easily.</li>
       <li><b>Fresh labels:</b> re-collect preferences on the <em>new</em> policy's outputs to fix distribution shift.</li>
     </ul>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the metric that always looks good.</b> The most dangerous property here: the number you're optimizing (RM reward) <b>keeps going up</b> even as the model gets worse. If you only watch the reward curve, everything looks like success. The only way to catch it is an <i>independent</i> check — held-out human ratings or a separate eval — precisely because the RM itself is compromised.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — optimize harder vs stay grounded.</b> More optimization (higher KL) squeezes more RM reward and can find genuine improvements — but past the true-quality peak it's just hacking. Staying near the reference (strong KL penalty) is safe but leaves gains on the table. Tuning that balance (the KL coefficient) is the central RLHF dial.</div></div>''',
     "gotit":"Got overoptimization"},
    {"title":"The overoptimization curve","body":'''<p>Here it is built up: the two reward curves, the diverging gap, the true-quality peak, and the three defenses. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove overoptimization","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_overopt.py</code>. Make a toy \"true\" reward and a noisy RM approximation of it. Optimize a simple policy against the RM and plot both the RM reward and the true reward vs optimization strength — watch true reward peak then fall.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 5 artifact.
Create experiments/foundations/m21a_overopt.py: define a true reward over a toy output space and a noisy reward-model approximation; optimize a policy against the RM at increasing strength (KL from a reference) and plot RM reward vs true reward — showing true reward peaks then declines (overoptimization). Add a KL penalty and show it delays the decline.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-05-log.md</code>: at what KL did true reward peak, and how much did the KL penalty help?</div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "climb":{"btn":"① RM reward climbs","html":'''<span class="prompt">&gt;&gt;&gt;</span> for step in optimize(policy, reward_model):
<span class="prompt">...</span>     print(rm_score)
<span class="hl">0.6 → 0.8 → 0.95 → 1.1</span>   <span class="dim"># keeps going up</span>''',
      "take":"<b>①  The RM score always rises.</b> The policy is doing its job — maximizing the reward model. Looks like pure success."},
    "true":{"btn":"② true quality peaks then drops","html":'''<span class="prompt">&gt;&gt;&gt;</span> human_rating_at_each_step
<span class="hl">0.6 → 0.75 → 0.70 → 0.55</span>
<span class="bad"># peaked at 0.75, then FELL while RM kept rising</span>''',
      "take":"<b>②  True quality peaks, then falls.</b> Past the peak the policy is exploiting the RM's blind spots — higher score, worse answers."},
    "kl":{"btn":"③ KL from reference grows","html":'''<span class="prompt">&gt;&gt;&gt;</span> kl_from_reference
<span class="hl">5 → 20 → 50</span>   <span class="dim"># policy drifting further from the SFT model</span>
<span class="dim"># a KL penalty caps this drift → defense</span>''',
      "take":"<b>③  The policy drifts.</b> Bigger KL = more optimization = more room to game the RM. A KL penalty limits the drift and keeps the model grounded."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11'><text x='260' y='14' fill='#6B645E' text-anchor='middle'>proxy (RM) reward vs optimization</text><polyline points='40,70 140,55 240,42 340,32 460,22' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><text x='470' y='24' fill='#1F6280'>RM ↗</text></g></svg>",
     "note":"<b>The RM reward keeps rising.</b> As you optimize the policy against it, the proxy score climbs and climbs — this is the number you'd naively watch."},
    {"viz":"<svg viewBox='0 0 520 100'><g font-family='monospace' font-size='11'><polyline points='40,75 140,58 240,45 340,35 460,25' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><polyline points='40,78 140,55 240,48 340,62 460,82' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='250' y='40' fill='#1a5c38'>true ↗↘</text></g></svg>",
     "note":"<b>True quality (green) tells a different story.</b> It rises with the RM at first, then peaks and falls even as the RM keeps climbing."},
    {"viz":"<svg viewBox='0 0 520 100'><g font-family='monospace' font-size='11'><polyline points='40,75 240,45 460,25' fill='none' stroke='#2A7B9B' stroke-width='2'/><polyline points='40,78 240,48 460,82' fill='none' stroke='#2D8B55' stroke-width='2'/><line x1='340' y1='35' x2='340' y2='62' stroke='#C93B3B' stroke-width='2' stroke-dasharray='3,3'/><text x='355' y='55' fill='#C93B3B'>gap = hacking</text></g></svg>",
     "note":"<b>The widening gap is reward hacking.</b> After the peak, every extra RM point is bought by exploiting the RM's errors, not by real improvement."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11' text-anchor='middle'><polyline points='40,70 240,45 460,75' fill='none' stroke='#2D8B55' stroke-width='2.5'/><circle cx='240' cy='45' r='6' fill='#C99A12'/><text x='240' y='30' fill='#9A7208'>ship HERE (true peak)</text></g></svg>",
     "note":"<b>The best model is at the true-quality peak.</b> Not the RM peak. Stop optimizing near here — which is why you need an independent quality check to find it."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='26' width='140' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='100' y='45' fill='#1F6280'>KL penalty</text><rect x='190' y='26' width='140' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='260' y='45' fill='#5E5191'>RM ensemble</text><rect x='350' y='26' width='140' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='420' y='45' fill='#1a5c38'>fresh labels</text></g></svg>",
     "note":"<b>Three defenses.</b> A KL penalty limits drift; an ensemble is harder to game; fresh labels fix distribution shift. Together they push the true-quality peak higher and later."},
  ],
  quiz=[
    {"q":"1. What is reward overoptimization?","opts":["the RM runs out of memory","optimizing the RM proxy too hard makes true quality rise then fall, as the policy exploits RM errors","the policy ignores the reward","training is too slow"],"ans":1,"fb":"Past a point, higher RM reward comes from gaming the RM's blind spots, so true quality declines even as the score climbs — Goodhart's law."},
    {"q":"2. Which is a concrete, common RM bias?","opts":["preferring shorter answers always","length bias — over-rewarding longer answers, so the policy learns to pad","refusing to score","only scoring code"],"ans":1,"fb":"Length bias is classic: RMs often score longer answers higher, so the optimized policy pads its replies. Sycophancy is another."},
    {"q":"3. Why does a KL penalty help against overoptimization?","opts":["it makes the RM bigger","it penalizes drifting far from the reference model, so the policy can't wander into the RM's blind spots","it removes the reward","it doubles the reward"],"ans":1,"fb":"Keeping the policy near the trusted reference limits how far it can exploit the RM's errors — the β dial in PPO."},
    {"q":"4. Your RM reward keeps rising for 50k steps. What must you do to know it's actually improving?","opts":["trust the reward curve","check an independent signal (held-out human ratings or a separate eval) — the RM itself is the thing being gamed","increase the learning rate","remove the KL penalty"],"ans":1,"fb":"Because the RM is what's being exploited, its own score can't tell you the truth. Only an independent check reveals the true-quality peak."},
  ],
  fin={"em":"🎯","h3":"Day 5 complete — you can spot a lying reward!",
       "p":"You now understand reward overoptimization: optimize a proxy RM too hard and true quality peaks then falls (Goodhart's law), driven by biases like length and sycophancy. The defenses are a KL penalty, RM ensembles, and fresh labels — and the key discipline is checking an independent signal, not the RM's own score. Next: <b>Day 6</b> — PPO for language models, where that KL penalty becomes a concrete term."},
))
# ---------------- Day 6 — PPO for Language Models ----------------
L("day-06-ppo-for-llms.html", dict(
  qid="m21a-d06-ppo", title="Module 21a · Day 6 — PPO for LLMs", nav_title="Spiral · M21a Day 6",
  eyebrow="Module 21a · Train · Day 6", h1="PPO for Language Models",
  lead="Now we connect everything. The policy is the LLM, an action is generating a response, and the reward is the reward model's score — minus a penalty for drifting too far from the trusted starting model. This is RLHF: PPO from M20, wired up for text, with a KL leash that stops the reward hacking you saw yesterday.",
  goal="<b>🎯 By the end, you'll be able to:</b> write the RLHF reward <code>r_RM − β·KL</code>, explain the roles of the reference model and value head, and say what the KL coefficient β controls.",
  prev_href="day-05-rm-failure-modes.html", prev_label="RM Failure Modes", next_href="day-07-dpo.html", next_label="DPO",
  sections=[
    {"title":"The LLM as a policy","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> M20 (policy gradient, PPO clip, advantage, KL) and Days 4–5 (reward model + overoptimization). Today we assemble them.</div></div>
     <h4>Mapping RL onto text</h4>
     <p>From M20: an agent takes actions in states to maximize reward. For an LLM: the <b>state</b> is the prompt-so-far, an <b>action</b> is the next token, and generating a whole response is one episode. The <b>reward</b> comes from Day 4's reward model, scoring the finished response.</p>
     <h4>The four models in the room</h4>
     <ul>
       <li><b>Policy</b> — the LLM being trained (it generates responses).</li>
       <li><b>Reference</b> — a frozen copy of the SFT model. We measure drift from it.</li>
       <li><b>Reward model</b> — scores responses (Day 4).</li>
       <li><b>Value head</b> — a small head on the policy estimating <code>V(s)</code> for PPO's advantage (M20 Day 5).</li>
     </ul>
     <h4>The leash</h4>
     <p>Yesterday you saw that hard optimization games the RM. RLHF adds a <span class="term" data-tip="A penalty in the reward for how far the policy has drifted from the frozen reference model, measured by KL divergence. It stops the policy exploiting the reward model's blind spots.">KL penalty</span>: the policy is rewarded by the RM but <em>punished</em> for straying from the reference. This keeps it near known-good behaviour.</p>''',
     "gotit":"Got the setup"},
    {"title":"A dog on a leash","body":
     '''<div class="relate">
       <div class="card"><span class="big">🦮</span><h5>Reward pulls forward</h5><p>The reward model is a treat pulling the policy toward higher-scoring answers. Left unchecked, the dog bolts toward whatever smells best — even into the RM's blind spots (reward hacking).</p></div>
       <div class="card"><span class="big">🪢</span><h5>KL is the leash</h5><p>The KL penalty is a leash tied to the reference model. The policy can move toward reward, but only so far. β sets the leash length: short leash = safe but limited; long leash = more freedom but more risk of bolting.</p></div>
     </div>
     <p><strong>In one line:</strong> the reward model pulls the policy toward better answers, and the KL penalty leashes it to the trusted reference so it can't run off and game the reward.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a leash is a hard limit, but the KL penalty is a <i>soft</i> cost — the policy can still drift far if the RM reward is large enough to outweigh the penalty. It discourages drift; it doesn't forbid it.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"One PPO step for text","body":'''<p>Watch the policy generate, the reward combine RM score with the KL penalty, and the clipped PPO update. <strong>Click all three.</strong></p>'''},
    {"title":"The RLHF reward and update","body":
     '''<h4>Step 1 — the combined reward</h4>
     <p>For a generated response, the reward is the RM score minus a penalty for how far each token's distribution drifted from the reference.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       R(x, y) = r_RM(x, y)  −  β · KL( π(·|x) ‖ π_ref(·|x) )<br>
       <span class="dim"># r_RM = reward-model score; π = policy; π_ref = frozen reference; β = KL coefficient</span><br>
       then optimize R with PPO's clipped objective (from M20)
     </div></div>
     <p>Symbols: <code>r_RM</code> = Day 4's score. <code>β</code> (beta) = how hard the KL penalty bites. <code>KL(π ‖ π_ref)</code> = how far the policy moved from the reference. PPO then improves the policy on this reward using the clip that keeps each update small (M20 Day 6).</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       r_RM = 0.80,   KL = 3.0,   β = 0.1<br><br>
       R = 0.80 − 0.1 × 3.0 = 0.80 − 0.30 = <span class="hl">0.50</span><br>
       <span class="dim"># the drift cost 0.30 of reward — raise β and drift is punished harder</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — KL collapse or blow-up.</b> Set β too low and the policy sprints away from the reference, hacks the RM, and produces confident garbage (mode collapse to a high-reward exploit). Set β too high and the policy barely moves — the KL term dominates and it learns almost nothing while the reward looks "stable." Both look non-crashing; you diagnose them by watching the KL value itself, not just the reward.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — β (freedom vs grounding).</b> Low β lets the policy chase reward hard (bigger gains, bigger risk of hacking); high β keeps it glued to the SFT reference (safe, smaller gains). This one coefficient is the central RLHF knob, and it directly sets where you land on yesterday's overoptimization curve.</div></div>''',
     "gotit":"Got the RLHF reward"},
    {"title":"RLHF, assembled","body":'''<p>Here it is built up: the policy generates, the RM scores, the KL penalty subtracts, the value head gives the advantage, and PPO clips the update. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a toy RLHF loop","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_ppo.py</code>. Wire a tiny PPO loop: a small policy generates, your Day 4 RM scores, subtract a KL penalty to the frozen reference, and update with the clipped objective. Sweep β and watch reward vs KL.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 6 artifact.
Create experiments/foundations/m21a_ppo.py: a minimal PPO-for-LM loop (use trl if available, else a toy env). Reward = reward_model_score - beta * KL(policy || reference). Sweep beta in {0.01, 0.1, 1.0} and print reward and KL over steps, showing low beta -> reward hacking / high KL, high beta -> little learning.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-06-log.md</code>: what happened to KL and reward at low vs high β?</div></div>''',
     "gotit":"Done — on to Day 7"},
  ],
  demos={
    "gen":{"btn":"① policy generates","html":'''<span class="prompt">&gt;&gt;&gt;</span> y = policy.generate(x)   <span class="dim"># the LLM = the policy</span>
<span class="hl">y = "Here are three safe options..."</span>
<span class="dim"># one full response = one episode</span>''',
      "take":"<b>①  The LLM acts.</b> Generating a response is the episode. Each token is an action; the prompt-so-far is the state."},
    "reward":{"btn":"② reward = RM − β·KL","html":'''<span class="prompt">&gt;&gt;&gt;</span> r_rm = reward_model(x, y)      <span class="dim"># 0.80</span>
<span class="prompt">&gt;&gt;&gt;</span> kl   = KL(policy, reference)    <span class="dim"># 3.0</span>
<span class="prompt">&gt;&gt;&gt;</span> R = r_rm - 0.1 * kl
<span class="hl">R = 0.80 - 0.30 = 0.50</span>''',
      "take":"<b>②  Reward minus drift penalty.</b> The RM pulls reward up; the KL term subtracts for straying from the reference. β=0.1 here cost 0.30."},
    "ppo":{"btn":"③ clipped PPO update","html":'''<span class="prompt">&gt;&gt;&gt;</span> advantage = R - value_head(x)   <span class="dim"># M20 advantage</span>
<span class="prompt">&gt;&gt;&gt;</span> ppo_update(policy, advantage, clip=0.2)
<span class="hl">policy nudged toward higher R, stays near π_ref</span>''',
      "take":"<b>③  PPO improves the policy.</b> The value head gives the advantage; the clip keeps each step small so the policy doesn't lurch — exactly M20's PPO, now on text."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>policy (LLM) generates y</text></g></svg>",
     "note":"<b>The policy generates a response.</b> The LLM being trained produces y from the prompt x — one episode of the RL loop."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>reward model scores: r_RM = 0.80</text></g></svg>",
     "note":"<b>The reward model scores it.</b> Day 4's RM gives the response a number — the pull toward better answers."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>− β · KL(policy ‖ reference) = −0.30</text></g></svg>",
     "note":"<b>Subtract the KL penalty.</b> A frozen reference (the SFT model) sets a leash; drifting from it costs reward. This blocks reward hacking."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='36' fill='#1a5c38'>R = 0.80 − 0.30 = 0.50</text></g></svg>",
     "note":"<b>The combined reward.</b> RM score minus drift penalty. This is what PPO actually maximizes — reward, kept honest by the leash."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>PPO clip update (value head → advantage)</text></g></svg>",
     "note":"<b>PPO improves the policy.</b> The value head gives the advantage; the clipped objective (M20) keeps each step small and stable. Repeat — that's RLHF."},
  ],
  quiz=[
    {"q":"1. In RLHF, what plays the role of the RL \"reward\"?","opts":["the next-token loss","the reward-model score minus a KL penalty for drifting from the reference","the learning rate","the number of tokens"],"ans":1,"fb":"The reward is r_RM(x,y) − β·KL(π‖π_ref): the RM score, leashed by a drift penalty. PPO optimizes this."},
    {"q":"2. What is the reference model for?","opts":["it generates the final answers","it's a frozen model we measure drift from, so the KL penalty can keep the policy grounded","it replaces the reward model","it stores the dataset"],"ans":1,"fb":"The frozen reference (usually the SFT model) defines \"known-good\"; KL(policy‖reference) penalizes wandering away from it."},
    {"q":"3. r_RM = 1.0, KL = 4.0, β = 0.2. The combined reward R is:","opts":["1.0","0.2","5.0","0.8"],"ans":1,"fb":"R = 1.0 − 0.2×4.0 = 1.0 − 0.8 = 0.2. The drift cost 0.8 of reward."},
    {"q":"4. You set β very low and the model starts producing confident nonsense with a soaring RM score. This is:","opts":["ideal RLHF","KL collapse / reward hacking — a too-loose leash let it exploit the RM","a reward-model crash","impossible"],"ans":1,"fb":"Low β removes the leash, so the policy sprints into the RM's blind spots (yesterday's overoptimization). Raise β to re-ground it."},
  ],
  fin={"em":"🦮","h3":"Day 6 complete — you can run RLHF!",
       "p":"You assembled RLHF: the LLM is the policy, the reward is <code>r_RM − β·KL</code> (RM pull, KL leash to the frozen reference), a value head gives the advantage, and PPO's clip keeps updates stable. β is the central knob between freedom and grounding. Next: <b>Day 7</b> — DPO, which reaches the same goal without a reward model or rollouts."},
))

# ---------------- Day 7 — DPO ----------------
L("day-07-dpo.html", dict(
  qid="m21a-d07-dpo", title="Module 21a · Day 7 — DPO", nav_title="Spiral · M21a Day 7",
  eyebrow="Module 21a · Train · Day 7", h1="DPO: Direct Preference Optimization",
  lead="RLHF works, but it's heavy: train a reward model, run PPO with a policy, reference, value head, and live generation. DPO asks — what if we skip the reward model and the RL entirely, and optimize the preference pairs directly with a simple classification-style loss? It turns out the same objective can be reached with one loss on a fixed dataset. Simpler, stabler, and often just as good.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how DPO removes the reward model and rollouts, read the DPO loss, and say what the implicit reward is.",
  prev_href="day-06-ppo-for-llms.html", prev_label="PPO for LLMs", next_href="day-08-data-engineering.html", next_label="Data Engineering",
  sections=[
    {"title":"Skip the reward model","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 4 (preference pairs, Bradley-Terry) and Day 6 (RLHF reward with a KL-to-reference term).</div></div>
     <h4>The insight</h4>
     <p>RLHF has two steps: fit a reward model to preferences, then RL-optimize against it (with a KL leash). <span class="term" data-tip="Direct Preference Optimization: fine-tune directly on preference pairs with a single classification-style loss, skipping the reward model and RL.">DPO (direct preference optimization)</span> shows those two steps can be folded into <em>one</em> loss on the preference data — no separate reward model, no sampling from the policy during training.</p>
     <h4>What DPO trains on</h4>
     <p>Exactly the same pairs as the reward model: prompt <code>x</code>, chosen <code>y_w</code>, rejected <code>y_l</code>. But instead of scoring them with an RM, DPO directly increases the policy's probability of <code>y_w</code> and decreases <code>y_l</code> — measured <em>relative to the frozen reference</em>, which quietly plays the role of the KL leash.</p>
     <h4>Why it's popular</h4>
     <p>DPO is a supervised-style loss over a fixed dataset. There is no reward model to train, no generation loop, no value head, no PPO clipping to tune. It is far simpler to run and much more stable — which is why many open models are aligned with DPO instead of PPO.</p>''',
     "gotit":"Got the idea"},
    {"title":"Grade the pair directly","body":
     '''<div class="relate">
       <div class="card"><span class="big">⚖️</span><h5>RLHF: hire a judge</h5><p>RLHF trains a judge (the reward model), then sends the student out to perform, get judged, and adjust — over and over, live. Lots of moving parts.</p></div>
       <div class="card"><span class="big">✅</span><h5>DPO: mark the answer key</h5><p>DPO skips the judge. It just takes the marked pairs ("this answer beat that one") and directly teaches the student to prefer the winners — one pass over a fixed answer key. No live performances.</p></div>
     </div>
     <p><strong>In one line:</strong> DPO learns straight from the preference pairs with one loss, using the frozen reference to stay grounded — no reward model, no reinforcement learning.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a marked answer key is fixed, but DPO still measures everything <i>relative to the reference model</i>, so the "grading" shifts as the policy moves away from the reference. It is not pure supervised copying.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Compute a DPO step","body":'''<p>Watch a pair, the log-ratios against the reference, and the DPO loss. <strong>Click all three.</strong></p>'''},
    {"title":"The DPO loss","body":
     '''<h4>Step 1 — the implicit reward</h4>
     <p>DPO's key move: define the reward <em>implicitly</em> as how much more likely the policy makes a response than the reference does. If the policy raises <code>y</code>'s probability above the reference, that <code>y</code> has positive implicit reward.</p>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       implicit reward:  r(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )
     </div></div>
     <h4>Step 2 — plug into Bradley-Terry</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       loss = − log σ( β·[ log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ] )<br>
       <span class="dim"># raise the winner's log-ratio above the loser's — the reference cancels the need for an RM</span>
     </div></div>
     <p>Symbols: <code>π_θ</code> = the policy being trained. <code>π_ref</code> = frozen reference. <code>β</code> = same KL-strength knob as PPO. It's Day 4's Bradley-Terry loss, but the "score" is now the log-probability ratio to the reference — so the policy itself <em>is</em> the reward model.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       β = 1.0.  Policy vs reference log-ratios:  winner = +0.5,  loser = −0.4<br><br>
       margin = 0.5 − (−0.4) = 0.9<br>
       loss = −log σ(1.0 × 0.9) = −log σ(0.9) = −log(0.711) = <span class="hl">0.34</span><br>
       <span class="dim"># gradient pushes the winner's log-ratio up and the loser's down</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — DPO overoptimization / reference drift.</b> DPO looks immune to reward hacking (no RM), but it isn't: pushed too hard (small β, many epochs) it drives the loser's probability toward zero and drifts far from the reference, degrading fluency and diversity while the DPO loss keeps dropping. It's the same overoptimization, just hidden in the loss instead of a reward curve. Watch a held-out eval and the KL to the reference, not only the training loss.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — DPO vs PPO.</b> DPO is simpler, cheaper, and more stable (offline, one loss, no RM/rollouts) and is a great default. PPO is heavier but has real advantages: it optimizes <i>fresh on-policy</i> generations and lets you use an explicit reward model (and non-preference or verifiable rewards, next module), giving more control at the frontier. Many teams DPO first, then PPO for the last mile.</div></div>''',
     "gotit":"Got the DPO loss"},
    {"title":"DPO, assembled","body":'''<p>Here it is built up: the pair, the policy and reference log-probs, the two log-ratios, the margin, and the loss. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with DPO","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_dpo.py</code>. Take an SFT model as both policy and (frozen) reference, and fine-tune on (chosen, rejected) pairs with the DPO loss. Print how the chosen/rejected log-probs separate over training, and compare to your Day 6 PPO run.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 7 artifact.
Create experiments/foundations/m21a_dpo.py: DPO on a small model (use trl DPOTrainer if available). Policy + frozen reference from the same SFT checkpoint; train on ~30 (chosen, rejected) pairs with beta=0.1. Print the chosen vs rejected log-prob margin over steps and note that no reward model or generation loop is used.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-07-log.md</code>: did chosen and rejected log-probs separate, and how did DPO feel simpler than PPO?</div></div>''',
     "gotit":"Done — on to Day 8"},
  ],
  demos={
    "pair":{"btn":"① the same pair","html":'''<span class="prompt">&gt;&gt;&gt;</span> x, y_win, y_lose = preference_pair()
<span class="dim"># identical data to the reward model — but no RM is trained</span>''',
      "take":"<b>①  Same preference data.</b> DPO uses the very same (chosen, rejected) pairs — it just uses them differently."},
    "ratios":{"btn":"② log-ratios vs reference","html":'''<span class="prompt">&gt;&gt;&gt;</span> lw = log(pi(y_win)) - log(pi_ref(y_win))   <span class="hl">+0.5</span>
<span class="prompt">&gt;&gt;&gt;</span> ll = log(pi(y_lose)) - log(pi_ref(y_lose))  <span class="hl">-0.4</span>
<span class="dim"># how much the policy shifted each vs the frozen reference</span>''',
      "take":"<b>②  Measure shift from the reference.</b> The log-ratio is DPO's implicit reward. The reference keeps the policy grounded — no separate KL term needed."},
    "loss":{"btn":"③ the DPO loss","html":'''<span class="prompt">&gt;&gt;&gt;</span> margin = lw - ll             <span class="dim"># 0.5 - (-0.4) = 0.9</span>
<span class="prompt">&gt;&gt;&gt;</span> loss = -log(sigmoid(beta*margin))
<span class="hl">loss = 0.34</span>   <span class="dim"># pushes winner up, loser down</span>''',
      "take":"<b>③  One classification-style loss.</b> Bradley-Terry on the log-ratios. Minimizing it prefers winners — no reward model, no rollouts, no PPO."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='170' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='145' y='43' fill='#1a5c38'>y_win</text><rect x='290' y='24' width='170' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='375' y='43' fill='#C93B3B'>y_lose</text></g></svg>",
     "note":"<b>Start with the preference pair.</b> The same chosen/rejected data a reward model would use — but DPO needs no RM."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='24' width='200' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='140' y='43' fill='#5E5191'>policy log-prob π_θ(y)</text><rect x='280' y='24' width='200' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='380' y='43' fill='#5A544E'>reference log-prob π_ref(y)</text></g></svg>",
     "note":"<b>Score each answer twice.</b> Under the trained policy and under the frozen reference. Their difference is the implicit reward."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='80' y='24' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='155' y='43' fill='#1F6280'>log-ratio_win = +0.5</text><rect x='290' y='24' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='365' y='43' fill='#1F6280'>log-ratio_lose = −0.4</text></g></svg>",
     "note":"<b>Two log-ratios.</b> How much more (or less) likely the policy makes each answer than the reference does. These are the implicit rewards."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='160' y='16' width='200' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>margin = 0.5 − (−0.4) = 0.9</text></g></svg>",
     "note":"<b>The margin.</b> Winner's implicit reward minus loser's. DPO wants this positive and growing — the winner clearly preferred."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='36' fill='#1a5c38'>loss = −log σ(β · margin)</text></g></svg>",
     "note":"<b>The DPO loss.</b> Bradley-Terry on the margin. One backward pass over fixed data updates the policy — the reference keeps it grounded, replacing the whole RM + PPO stack."},
  ],
  quiz=[
    {"q":"1. What does DPO remove compared with PPO-based RLHF?","opts":["the preference data","the separate reward model and the online RL/generation loop","the transformer","the tokenizer"],"ans":1,"fb":"DPO folds reward modeling and RL into one loss on fixed pairs — no RM, no rollouts, no value head or PPO clipping."},
    {"q":"2. In DPO, the implicit reward for a response is:","opts":["the raw RM score","β·log(π_θ(y|x)/π_ref(y|x)) — the policy's log-prob shift from the reference","the token count","always 1"],"ans":1,"fb":"DPO defines reward implicitly as the log-ratio of policy to reference (times β), so the policy itself acts as the reward model."},
    {"q":"3. What role does the frozen reference model play in DPO?","opts":["it generates training data","it grounds the update (acting like the KL leash), so DPO doesn't drift arbitrarily","it scores with Bradley-Terry","it replaces the tokenizer"],"ans":1,"fb":"Everything is measured relative to the reference, which quietly provides the same KL-to-reference grounding PPO had explicitly."},
    {"q":"4. DPO has no reward model, so it can't overoptimize. True?","opts":["true — DPO is immune","false — pushed too hard it drifts from the reference and degrades fluency while the loss still drops","true, unless β=0","false, but only for PPO"],"ans":1,"fb":"DPO can still overoptimize: small β / many epochs drives the loser's probability down and drifts from the reference. Watch a held-out eval and KL, not just the loss."},
  ],
  fin={"em":"✅","h3":"Day 7 complete — you can align without RL!",
       "p":"You now know DPO: define an implicit reward as the policy-vs-reference log-ratio, plug it into Bradley-Terry, and get one stable loss on fixed preference pairs — no reward model, no rollouts. It's the simpler default; PPO wins the last mile with on-policy, RM-based control. Next: <b>Day 8</b> — the preference and instruction data that decides whether any of this works."},
))
# ---------------- Day 8 — Preference & Instruction Data ----------------
L("day-08-data-engineering.html", dict(
  qid="m21a-d08-data", title="Module 21a · Day 8 — Preference & Instruction Data", nav_title="Spiral · M21a Day 8",
  eyebrow="Module 21a · Train · Day 8", h1="The Data Decides Everything",
  lead="Every method in this module — SFT, reward models, PPO, DPO — is only as good as the data it learns from. The model's values are the annotators' values; its biases are the pipeline's biases. Today is the unglamorous, decisive craft: how preference and instruction data is collected, where it goes wrong, and the threat model that frames what you're actually defending against.",
  goal="<b>🎯 By the end, you'll be able to:</b> describe the preference-data pipeline, name annotator biases (length, position, self-preference), explain RLAIF/Constitutional AI as a scalable alternative, and sketch the alignment threat model.",
  prev_href="day-07-dpo.html", prev_label="DPO", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Where the data comes from","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–7 — you know SFT, reward models, PPO, DPO all consume instruction and preference data.</div></div>
     <h4>The pipeline</h4>
     <p>Preference data is manufactured, not found. The pipeline: collect <b>prompts</b> → have the model produce several <b>candidate responses</b> → have <b>humans (or an AI) rank</b> them → <b>filter</b> for quality and agreement → ship the pairs. Every stage can inject a bias that the reward model or DPO will faithfully learn.</p>
     <h4>Annotator agreement</h4>
     <p>Different people disagree about "better." <span class="term" data-tip="A measure of how often independent annotators give the same label. Low agreement means the labels are noisy and the resulting reward signal is unreliable.">Inter-annotator agreement</span> tells you how noisy the labels are. Low agreement means the "preference" you're training on is partly coin-flips — no method can rise above noisy labels.</p>
     <h4>Scaling with AI feedback</h4>
     <p>Human labels are slow and expensive. <span class="term" data-tip="Reinforcement Learning from AI Feedback: use a capable model, guided by written principles, to produce the preference labels instead of humans.">RLAIF</span> replaces human rankers with a model, often guided by a written set of principles — <span class="term" data-tip="Constitutional AI: the model critiques and revises its own outputs against an explicit written 'constitution' of principles, generating preference data without human labels for every pair.">Constitutional AI</span>. It scales cheaply, but the labels now carry the labelling model's biases.</p>''',
     "gotit":"Got the pipeline"},
    {"title":"Garbage in, garbage out","body":
     '''<div class="relate">
       <div class="card"><span class="big">🍎</span><h5>The model eats what you feed it</h5><p>A student taught from a biased textbook learns the bias as fact. The model has no way to know the labels are skewed — it treats them as ground truth. The annotators' taste becomes the model's taste.</p></div>
       <div class="card"><span class="big">👥</span><h5>Who labels matters</h5><p>Ten tired annotators rushing through pairs will favour whatever's quick to judge — often the longer or more confident answer. The pool of labelers, their instructions, and their incentives are all baked into the final model.</p></div>
     </div>
     <p><strong>In one line:</strong> the model inherits the values and blind spots of whoever (or whatever) produced its preference labels — so the data pipeline <em>is</em> the alignment.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a student can eventually question a biased textbook. The model cannot — it optimizes the labels exactly, with no outside sense that they might be wrong. That is why the pipeline needs deliberate bias controls, not just good intentions.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Build a clean dataset","body":'''<p>Watch pairs collected, annotator agreement checked, and low-quality pairs filtered. <strong>Click all three.</strong></p>'''},
    {"title":"Biases and the threat model","body":
     '''<h4>Step 1 — the biases to control</h4>
     <ul>
       <li><b>Length bias:</b> labelers pick longer answers → the model learns to pad. Control: length-balance the pairs.</li>
       <li><b>Position/order bias:</b> when shown A then B, judges favour whichever slot — so always <span class="term" data-tip="Showing each pair in both orders (A,B) and (B,A) and averaging, to cancel the tendency to favour a fixed position.">swap the order</span> and average.</li>
       <li><b>Self-preference:</b> an AI judge rates its <em>own</em> style of answer higher. Control: use a different judge, or human spot-checks.</li>
     </ul>
     <h4>Step 2 — the quality knobs</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       keep a pair if:  annotators_agree ≥ threshold  AND  not(near_duplicate)  AND  length_balanced<br>
       measure: agreement rate, prompt diversity, contamination overlap with eval sets
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       10,000 raw pairs collected<br>
       − 1,500 with annotator disagreement (coin-flips) → drop<br>
       − 800 near-duplicate prompts → drop<br>
       − 300 overlap with the eval set (contamination) → drop<br>
       = <span class="hl">7,400</span> clean, diverse, trustworthy pairs → these train the RM / DPO
     </div></div>
     <h4>The alignment threat model</h4>
     <p>Zoom out: what are we defending against? <b>Specification gaming</b> (the model satisfies the letter of the reward, not the intent — Day 5), <b>deception/sycophancy</b> (it tells you what you want), and <b>jailbreaks / prompt injection</b> (users or content override its training). Naming these threats is what turns "be good" into an engineering target — and it's exactly what Module 27 (assistant safety, red-teaming) attacks head-on.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — contamination baked into training.</b> If your preference or instruction data overlaps with your evaluation benchmarks, every downstream number is inflated — the model "improves" on the test because it trained on it. Nothing errors; the leaderboard just lies. Decontaminate the training data against every eval set <i>before</i> training, and prefer held-out-by-construction evals (Module 24).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — human labels vs AI feedback.</b> Human preferences are higher-quality and catch subtle harms, but are slow, costly, and inconsistent. RLAIF / Constitutional AI scales to millions of labels for cheap and is consistent, but bakes in the labelling model's biases and blind spots. Frontier pipelines blend both: AI feedback for scale, humans for the hard and high-stakes cases.</div></div>''',
     "gotit":"Got biases and threats"},
    {"title":"The data pipeline, assembled","body":'''<p>Here it is built up: prompts, candidate generations, ranking (human or AI), bias controls, and the filtered clean set. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Audit a dataset","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21a_data_audit.py</code>. Take a preference dataset (or make a toy one) and compute: length bias (does the chosen answer skew longer?), prompt diversity (near-duplicate rate), and an order-swap check on an AI judge. Report the biases you find.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21a Day 8 artifact.
Create experiments/foundations/m21a_data_audit.py: load or synthesize a preference dataset and report (1) length bias — mean length of chosen vs rejected, (2) near-duplicate prompt rate, (3) an order-swap consistency check for an LLM judge (does swapping A/B flip the winner?). Print each metric with a one-line interpretation.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21a/day-08-log.md</code>: which bias was strongest, and how would you fix it?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "collect":{"btn":"① collect pairs","html":'''<span class="prompt">&gt;&gt;&gt;</span> for prompt in prompts:
<span class="prompt">...</span>     cands = model.generate(prompt, n=4)
<span class="prompt">...</span>     ranking = annotator.rank(cands)   <span class="dim"># human or AI</span>
<span class="hl">10,000 (chosen, rejected) pairs</span>''',
      "take":"<b>①  Manufacture the pairs.</b> Prompts → several candidate answers → a ranker picks winners. This raw set is noisy and needs cleaning."},
    "agree":{"btn":"② check agreement","html":'''<span class="prompt">&gt;&gt;&gt;</span> agreement = fraction_where_two_annotators_match()
<span class="hl">agreement = 0.78</span>   <span class="dim"># 22% are near coin-flips</span>
<span class="bad"># low-agreement pairs are noise — drop them</span>''',
      "take":"<b>②  Measure label noise.</b> If annotators disagree a lot, those pairs are unreliable. No method learns above the label noise floor."},
    "filter":{"btn":"③ filter & decontaminate","html":'''<span class="prompt">&gt;&gt;&gt;</span> clean = drop_disagree(pairs)
<span class="prompt">&gt;&gt;&gt;</span> clean = dedup(clean); clean = decontaminate(clean, eval_sets)
<span class="hl">10,000 → 7,400 clean pairs</span>''',
      "take":"<b>③  Filter to trustworthy data.</b> Drop coin-flips, near-duplicates, and eval-set overlaps. The clean set is what actually trains the RM or DPO."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='16' width='180' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>prompts</text></g></svg>",
     "note":"<b>Start with prompts.</b> Diverse, representative of real use. Narrow prompts give a narrow model."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>model generates several candidates</text></g></svg>",
     "note":"<b>Generate candidates.</b> The model produces several answers per prompt — the things to be ranked."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='150' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='165' y='36' fill='#1a5c38'>human ranking</text><rect x='280' y='16' width='150' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='355' y='36' fill='#9A7208'>or AI (RLAIF)</text></g></svg>",
     "note":"<b>Rank them.</b> Humans (quality, costly) or an AI judge with a constitution (scale, cheaper but biased). This produces the win/lose labels."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='24' width='130' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='105' y='42' fill='#C93B3B'>swap order</text><rect x='195' y='24' width='130' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='260' y='42' fill='#C93B3B'>length-balance</text><rect x='350' y='24' width='130' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='415' y='42' fill='#C93B3B'>drop low-agree</text></g></svg>",
     "note":"<b>Apply bias controls.</b> Swap A/B order, balance lengths, drop coin-flip pairs. This is where you fight length, position, and self-preference bias."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='36' fill='#1a5c38'>7,400 clean, decontaminated pairs</text></g></svg>",
     "note":"<b>The clean dataset.</b> Diverse, high-agreement, deduplicated, decontaminated. This — not the algorithm — is what most determines the aligned model's quality."},
  ],
  quiz=[
    {"q":"1. Why is annotator agreement worth measuring?","opts":["to speed up training","low agreement means noisy labels, and no method can learn above the label noise","it sets the learning rate","it picks the model size"],"ans":1,"fb":"If annotators disagree, the 'preferences' are partly random; the reward model or DPO can only be as reliable as the labels."},
    {"q":"2. To counter an AI judge's position/order bias, you should:","opts":["always show option A first","show each pair in both orders and average the result","use longer prompts","remove the reference model"],"ans":1,"fb":"Order-swapping and averaging cancels the tendency to favour a fixed slot — a standard debiasing step for pairwise judges."},
    {"q":"3. What does RLAIF / Constitutional AI trade for scale?","opts":["nothing — it's strictly better","it inherits the labelling model's biases instead of humans'","it needs more GPUs than humans","it removes preferences"],"ans":1,"fb":"AI feedback scales cheaply and consistently, but the labels carry the judging model's biases and blind spots — so teams blend it with human labels."},
    {"q":"4. Your fine-tuned model scores great on a benchmark, but you later find that benchmark's prompts were in your preference data. This means:","opts":["the model truly improved","contamination inflated the score — the model trained on the test","the reward model is broken","β was too high"],"ans":1,"fb":"Training-eval overlap is contamination: the number is inflated because the model saw the test. Decontaminate before training and prefer held-out-by-construction evals."},
  ],
  fin={"em":"🧱","h3":"Day 8 complete — you can build the fuel!",
       "p":"You now know the craft that decides alignment: the preference-data pipeline (prompts → generations → ranking → filtering), annotator agreement, the biases to control (length, position, self-preference), RLAIF/Constitutional AI for scale, contamination, and the threat model (specification gaming, deception, jailbreaks) that Module 27 attacks. That completes Core Post-Training — on to the review gate."},
))

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m21a-review", title="Module 21a · Review Gate", nav_title="Spiral · M21a Review",
  eyebrow="Module 21a · The Gate — You Can Align a Model", h1="Module 21a — Review Gate",
  lead="Checkpoint. The gate question: <b>can I take a pretrained model through SFT, a reward model, and either PPO or DPO — and say where each can quietly go wrong?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Module 21b (reasoning-RL / GRPO) and the safety modules build directly on this alignment stack.",
  prev_href="day-08-data-engineering.html", prev_label="Day 8 · Preference & Instruction Data",
  checks=[
    ["Day 1","I can lay out the pipeline pretrain → SFT → preference optimization and say why comparisons beat authored answers."],
    ["Day 2","I can explain SFT (next-token loss on curated data, prompt masked) and how instruction tuning generalizes."],
    ["Day 3","I can explain LoRA (train a low-rank <code>B·A</code>, freeze <code>W</code>) and what QLoRA's 4-bit base adds."],
    ["Day 4","I can write the reward-model Bradley-Terry loss <code>−log σ(r_w − r_l)</code> and compute a preference probability."],
    ["Day 5","I can explain reward overoptimization (Goodhart), name RM biases, and list the defenses (KL, ensembles, fresh labels)."],
    ["Day 6","I can write the RLHF reward <code>r_RM − β·KL</code> and say what the reference, value head, and β do."],
    ["Day 7","I can explain how DPO drops the reward model and RL, and read its loss (implicit reward = log-ratio to the reference)."],
    ["Day 8","I can describe the preference-data pipeline, its biases, RLAIF, and the alignment threat model."],
  ],
  quiz=[
    {"q":"1. Why does SFT come before preference optimization?","opts":["it's arbitrary","preference methods polish a good starting policy; on a raw model they wander into gibberish","SFT needs a reward model first","DPO requires it by law"],"ans":1,"fb":"SFT gets you to the right neighbourhood; RLHF/DPO make small careful improvements around it (Day 1)."},
    {"q":"2. A layer's W is 2000×2000; LoRA rank r=8. Trainable params ≈","opts":["4,000,000","32,000","16,000","8,000"],"ans":1,"fb":"r·(d+k) = 8·(2000+2000) = 32,000, versus 4,000,000 for full fine-tuning (Day 3)."},
    {"q":"3. Reward-model scores r_w=1.5, r_l=1.5. P(winner preferred) =","opts":["1.0","0.5","0.0","0.75"],"ans":1,"fb":"σ(r_w − r_l) = σ(0) = 0.5 — equal scores mean no preference (Day 4)."},
    {"q":"4. In RLHF, the KL penalty β·KL(π‖π_ref) is there to:","opts":["increase the reward","stop the policy drifting into the reward model's blind spots (reward hacking)","replace the value head","shrink the model"],"ans":1,"fb":"It leashes the policy to the trusted reference, limiting overoptimization (Days 5–6)."},
    {"q":"5. The biggest single lever on how well any of these methods work is:","opts":["the learning rate","the quality, diversity, and cleanliness of the preference/instruction data","the GPU brand","the random seed"],"ans":1,"fb":"The model inherits its data's values and biases; the data pipeline is the alignment (Day 8)."},
  ],
  verdict_pass="you can run the full post-training stack — SFT, LoRA/QLoRA, a Bradley-Terry reward model, PPO-based RLHF with a KL leash, and DPO — and you can name where each quietly fails and how to defend it. You're ready for reasoning-RL (M21b).",
  verdict_fail="note which day tripped you up and re-run just that lesson. The arc — <code>pipeline</code> → <code>SFT</code> → <code>LoRA</code> → <code>reward model</code> → <code>overoptimization</code> → <code>PPO</code> → <code>DPO</code> → <code>data</code> — should feel familiar before you build reasoning-RL on top.",
  complete_label="Mark Module 21a complete",
  fin={"em":"🏆","h3":"Module 21a — passed!",
       "p":"You can now turn a raw pretrained model into an aligned assistant: SFT for format, LoRA/QLoRA for cheap fine-tuning, a Bradley-Terry reward model, RLHF with PPO and a KL leash, DPO as the simpler alternative, and the data craft plus threat model underneath it all. This is the alignment backbone the reasoning-RL and safety modules build on."},
  score_target=4,
))
print("M21a built: 8 lessons + review")




