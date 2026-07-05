#!/usr/bin/env python3
"""Builder for M25 — Interpretability Foundations (author from scratch).
Working literacy: probes, logit lens, activation patching; superposition/SAE/circuits awareness.
6 lessons + review. Run: python3 sessions/_build_m25.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m25")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — Linear Probes ----------------
L("day-01-linear-probes.html", dict(
  qid="m25-d01-probes", title="Module 25 · Day 1 — Linear Probes", nav_title="Spiral · M25 Day 1",
  eyebrow="Module 25 · Automate · Day 1", h1="Linear Probes: What Does a Layer Know?",
  lead="A trained transformer is a black box of numbers. Interpretability is the craft of reading those numbers to understand what the model computes. The simplest tool: a linear probe. Freeze the model, grab the activations at some layer, and train a tiny linear classifier to predict a property — say, sentiment. If the probe succeeds, that information is present and easy to read at that layer. It's a flashlight into the dark box.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain what a linear probe measures, train one on a layer's activations in principle, and say why a high probe score does NOT prove the model uses that information.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-logit-lens.html", next_label="The Logit Lens",
  sections=[
    {"title":"Read the hidden layer","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a transformer has hidden activations at each layer (M5a), and a linear classifier maps a vector to a label (M2). Nothing about interpretability yet.</div></div>
     <h4>The black-box problem</h4>
     <p>A model works, but we can't see <em>why</em>. Interpretability tries to open the box. The first question: what information is sitting in the activations at a given layer?</p>
     <h4>The linear probe</h4>
     <p>A <span class="term" data-tip="A small linear classifier trained on a frozen model's hidden activations to test whether a property (e.g. sentiment) is linearly readable at that layer.">linear probe</span> is a tiny classifier trained on top of a <em>frozen</em> model. You run text through the model, capture the hidden vector at layer L, and train a linear map from that vector to a label you care about — is this sentence positive or negative? If the probe predicts well, the property is <span class="term" data-tip="Readable by a single linear layer — the information sits along a direction in activation space, no complex nonlinearity needed to extract it.">linearly decodable</span> at that layer.</p>
     <h4>Why linear, and why frozen</h4>
     <p>The model stays frozen so you study <em>it</em>, not a new network. The probe is linear on purpose: if a mere linear map can read the property, the model has made that information easy to access. Probe accuracy across layers shows <em>where</em> in the network a concept becomes available.</p>''',
     "gotit":"Got the idea"},
    {"title":"A metal detector over the layers","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔍</span><h5>Scan for a signal</h5><p>A metal detector sweeps the ground and beeps where metal sits. A probe sweeps a layer's activations and "beeps" (high accuracy) when a specific concept is present and easy to read there.</p></div>
       <div class="card"><span class="big">📶</span><h5>Where the signal is strongest</h5><p>Move the detector deeper (later layers) and the beep can get stronger — abstract concepts often become linearly readable only after several layers. The probe tells you at what depth the concept "shows up."</p></div>
     </div>
     <p><strong>In one line:</strong> a linear probe scans a frozen layer to check whether a chosen concept is present and easy to read — and where in the network it appears.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a metal detector beeping means metal is really there and matters. A probe beeping means the info is <i>readable</i> — but the model might never actually <i>use</i> it. Presence is not use (that's today's big warning).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Train a probe","body":'''<p>Watch activations captured at a layer, a linear probe trained on them, and its accuracy read out. <strong>Click all three.</strong></p>'''},
    {"title":"What the score means (and doesn't)","body":
     '''<h4>Step 1 — the probe</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       h = activations(model, text)[layer_L]     <span class="dim"># frozen model, hidden vector</span><br>
       ŷ = softmax(W · h + b)                     <span class="dim"># a LINEAR probe, only W,b are trained</span><br>
       train W,b on labeled (text, property) pairs; the model never updates
     </div></div>
     <h4>Step 2 — read the accuracy</h4>
     <p>High probe accuracy means the property is linearly present at layer L. Low accuracy means it isn't readable there (maybe it appears deeper, or nonlinearly, or not at all). Sweeping L shows the concept emerging with depth.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       probe accuracy for "is this sentence about sports?" by layer:<br>
       layer 2 → 55%  (barely above chance 50%)<br>
       layer 6 → 78%<br>
       layer 10 → <span class="hl">94%</span>  (the concept is clearly readable here)<br>
       <span class="dim"># the topic becomes linearly available around the later layers</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — presence is not use.</b> The single biggest trap in probing: a high probe score shows a concept is <i>readable</i> in the activations, NOT that the model <i>relies</i> on it to make predictions. The information could be an idle by-product the model ignores. Worse, a high-capacity probe can "find" structure that is really the probe learning, not the model representing it. You can only prove the model <i>uses</i> a feature with a causal test — like tomorrow's logit lens and Day 3's activation patching.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — probe capacity.</b> A stronger probe (nonlinear, or high-dimensional) reads more, so it scores higher — but it also risks reading structure the <i>probe</i> invented rather than the <i>model</i> encoded, giving false positives. A weak linear probe is conservative and interpretable (if it reads it, the model made it easy) but misses nonlinearly-encoded features. Keep probes deliberately simple, and treat their scores as evidence of presence, never of causation.</div></div>''',
     "gotit":"Got what the score means"},
    {"title":"The probe, built up","body":'''<p>Here it is assembled: text into the frozen model, the layer's activations, the linear probe, its prediction, and the accuracy curve across layers. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a probe","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_probe.py</code>. Run a small frozen model on labelled sentences, capture activations at several layers, and train a logistic-regression probe for a property (e.g. sentiment) at each layer. Plot probe accuracy vs layer.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 1 artifact.
Create experiments/foundations/m25_probe.py: run a small frozen HuggingFace model on a labelled sentiment dataset, extract mean-pooled hidden states at each layer, train a logistic-regression linear probe per layer, and plot probe accuracy vs layer. Add a comment: high accuracy shows the info is READABLE, not that the model USES it.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-01-log.md</code>: at which layer did the concept become linearly readable, and why doesn't that prove the model uses it?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "extract":{"btn":"① capture activations","html":'''<span class="prompt">&gt;&gt;&gt;</span> h = model(text, output_hidden=True).layers[10]
<span class="hl">h.shape = (768,)</span>   <span class="dim"># hidden vector at layer 10, model frozen</span>''',
      "take":"<b>①  Grab the hidden vector.</b> Run text through the frozen model and capture activations at a chosen layer. This is the thing we probe."},
    "train":{"btn":"② train the linear probe","html":'''<span class="prompt">&gt;&gt;&gt;</span> probe = LogisticRegression()   <span class="dim"># just W, b</span>
<span class="prompt">&gt;&gt;&gt;</span> probe.fit(activations, labels)   <span class="dim"># model stays frozen</span>
<span class="hl">a linear map from h → property</span>''',
      "take":"<b>②  Train only the probe.</b> A tiny linear classifier learns to read the property from h. The model does not change — we study it, not retrain it."},
    "read":{"btn":"③ read the accuracy","html":'''<span class="prompt">&gt;&gt;&gt;</span> probe.score(test_activations, test_labels)
<span class="hl">0.94</span>   <span class="dim"># concept is linearly readable at layer 10</span>
<span class="dim"># but readable ≠ the model actually uses it</span>''',
      "take":"<b>③  High score = present, not used.</b> 94% means the property is easy to read at this layer. It does NOT prove the model relies on it — that needs a causal test."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='12' width='220' height='28' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='31' fill='#5A544E'>input text → frozen model</text></g></svg>",
     "note":"<b>Run text through the frozen model.</b> We never update the model — we only observe what it already computes."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>hidden activations h at layer L</text></g></svg>",
     "note":"<b>Capture activations at a layer.</b> A vector of numbers — the model's internal representation of the input at depth L."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>linear probe: ŷ = W·h + b</text></g></svg>",
     "note":"<b>Train a linear probe.</b> Only W and b are learned. If a linear map can read the property, the model made it easy to access."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='14' width='180' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='33' fill='#1a5c38'>accuracy = 0.94</text></g></svg>",
     "note":"<b>Read the accuracy.</b> High → the concept is linearly present at this layer. Low → it isn't readable here."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10'><polyline points='40,58 140,55 240,42 340,28 460,20' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><text x='250' y='68' fill='#6B645E' text-anchor='middle'>probe accuracy rises with layer depth</text></g></svg>",
     "note":"<b>Sweep the layers.</b> The accuracy curve shows where a concept becomes readable. But remember: readable ≠ used — proving use needs a causal test (Days 2–3)."},
  ],
  quiz=[
    {"q":"1. What does a linear probe test?","opts":["whether the model is accurate","whether a property is linearly readable from a layer's activations","the learning rate","the token count"],"ans":1,"fb":"A probe trains a linear classifier on frozen activations; high accuracy means the property is present and linearly decodable at that layer."},
    {"q":"2. During probing, the model is:","opts":["fine-tuned along with the probe","kept frozen — only the probe (W, b) is trained","deleted","retrained from scratch"],"ans":1,"fb":"The point is to study the existing model, so it's frozen; only the small probe learns."},
    {"q":"3. A probe reaches 95% accuracy for 'is this about sports?' at layer 10. This proves:","opts":["the model uses sports-topic info to predict","the info is linearly readable there — NOT that the model relies on it","the model was trained on sports","layer 10 is the output"],"ans":1,"fb":"Presence is not use: the concept is readable, but the model might ignore it. Only a causal intervention shows actual use."},
    {"q":"4. Why keep the probe linear (not a big nonlinear network)?","opts":["linear is faster to type","a powerful probe can 'find' structure it invented, giving false positives; linear is conservative","nonlinear probes are illegal","to increase accuracy"],"ans":1,"fb":"A high-capacity probe risks reading structure the probe learned rather than the model encoded. A simple linear probe keeps the finding conservative and interpretable."},
  ],
  fin={"em":"🔦","h3":"Day 1 complete — you can read a layer!",
       "p":"You now know linear probes: train a tiny linear classifier on a frozen layer's activations to test whether a concept is linearly readable, and sweep layers to see where it emerges. The crucial caveat — presence is not use; a high probe score never proves the model relies on the feature. Next: <b>Day 2</b> — the logit lens, watching the model's prediction form layer by layer."},
))

# ---------------- Day 2 — The Logit Lens ----------------
L("day-02-logit-lens.html", dict(
  qid="m25-d02-logitlens", title="Module 25 · Day 2 — The Logit Lens", nav_title="Spiral · M25 Day 2",
  eyebrow="Module 25 · Automate · Day 2", h1="The Logit Lens: Watch the Answer Form",
  lead="A transformer builds its prediction gradually across layers. The logit lens is a beautifully simple trick to watch that happen: take the activations at any middle layer and push them through the model's own output projection early, as if the model had to answer right now. You see the model's 'current best guess' at each depth — often gibberish early, snapping to the right answer in later layers.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the logit lens (project a mid-layer activation through the unembedding), read how a prediction forms across layers, and say when the lens misleads.",
  prev_href="day-01-linear-probes.html", prev_label="Linear Probes", next_href="day-03-activation-patching.html", next_label="Activation Patching",
  sections=[
    {"title":"Ask the model early","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a transformer's final layer projects the hidden vector to token logits via an unembedding matrix (M5a), and the residual stream carries information across layers.</div></div>
     <h4>The residual stream</h4>
     <p>In a transformer, each layer <em>adds</em> to a running vector called the <span class="term" data-tip="The running sum of vectors flowing through a transformer; each layer reads from it and adds its contribution. The final residual-stream vector is projected to token predictions.">residual stream</span>. The final residual vector is turned into a prediction by the <span class="term" data-tip="The output matrix (often the transpose of the embedding) that maps a hidden vector to a score for every token in the vocabulary.">unembedding</span> matrix.</p>
     <h4>The logit lens</h4>
     <p>The <span class="term" data-tip="An interpretability technique: apply the model's final unembedding to the residual stream at an intermediate layer, to read the model's 'current guess' at that depth.">logit lens</span> asks: what if we applied that final unembedding to a <em>middle</em> layer's residual vector, not just the last one? You get the token the model would predict if it had to stop at that layer. Do it at every layer and you watch the prediction form.</p>
     <h4>Why it's useful</h4>
     <p>It shows <em>where</em> in the network the answer gets decided. Often early layers give unrelated tokens, middle layers get warmer, and the last few layers lock onto the answer. It's a free, no-training window into the model's evolving belief — a perfect complement to yesterday's probes.</p>''',
     "gotit":"Got the idea"},
    {"title":"Peeking at the scratch work","body":
     '''<div class="relate">
       <div class="card"><span class="big">📝</span><h5>Scratch work at each stage</h5><p>A student solving a problem scribbles intermediate guesses. Peek over their shoulder at stage 3, stage 6, stage 9 and you see the answer taking shape. The logit lens peeks at the model's scratch work (residual stream) at each layer.</p></div>
       <div class="card"><span class="big">🎯</span><h5>Warmer, warmer, got it</h5><p>Early peeks look random; later peeks converge on the final answer. Watching the guesses sharpen tells you which layers actually did the deciding.</p></div>
     </div>
     <p><strong>In one line:</strong> apply the model's output projection to a middle layer to read its guess-so-far, and watch the answer sharpen across depth.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a student's scratch work is meant to be read. The residual stream was never designed to be decoded early — later layers may transform it in ways the lens can't see, so early "guesses" can be misleading, not just immature.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Watch a prediction form","body":'''<p>Watch a mid-layer activation, its projection through the unembedding, and the top token at several layers. <strong>Click all three.</strong></p>'''},
    {"title":"Projecting the residual early","body":
     '''<h4>Step 1 — the operation</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       logits_L = Unembed( LayerNorm( residual_L ) )     <span class="dim"># normally only done at the last layer</span><br>
       guess_L  = argmax(logits_L)                        <span class="dim"># the model's 'current' top token at layer L</span>
     </div></div>
     <p>Symbols: <code>residual_L</code> = the residual-stream vector at layer L. <code>Unembed</code> = the final output matrix. We borrow the final projection and apply it early. Repeat for every L to get a guess per layer.</p>
     <h4>Step 2 — read the trajectory</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       prompt: "The capital of France is" — watch the top token by layer:<br>
       layer 4  → " the"      <span class="dim"># generic, unresolved</span><br>
       layer 8  → " a"        <span class="dim"># still generic</span><br>
       layer 12 → " Paris"    <span class="dim"># the answer appears</span><br>
       layer 16 → <span class="hl"> Paris</span>   <span class="dim"># locked in</span>
     </div></div>
     <p>The lens shows the answer "Paris" emerging around layer 12 and firming up after — evidence that the late-middle layers do the factual recall here.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the lens lies on late-binding models.</b> The logit lens assumes intermediate residuals live in the same space the final unembedding expects. On many models that's only roughly true — heavy final LayerNorm, or features that are only "decoded" by the last layers (late binding), make the early projections systematically wrong. The lens can show a confident "guess" that the model never actually held. It's a cheap heuristic, not ground truth; a tuned lens (a learned per-layer transform) or activation patching (Day 3) is needed to trust it.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — free intuition vs fidelity.</b> The logit lens costs nothing (no training, one matrix multiply per layer) and gives instant, vivid intuition about where predictions form. But it's approximate — it reads the residual through the wrong-for-that-layer lens. You trade fidelity for zero cost: great for exploration, not for claims you'd stake a paper on without a causal follow-up.</div></div>''',
     "gotit":"Got the projection"},
    {"title":"The logit lens, built up","body":'''<p>Here it is assembled: the residual stream across layers, the borrowed unembedding, the per-layer top token, and the answer sharpening. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a logit lens","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_logit_lens.py</code>. On a small model, run a factual prompt, and at each layer apply the final layernorm + unembedding to the residual stream. Print the top token per layer and watch the answer emerge.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 2 artifact.
Create experiments/foundations/m25_logit_lens.py: on a small GPT-style model, capture the residual stream at each layer for a factual prompt, apply the model's final layernorm + unembedding to each, and print the top predicted token per layer to show the answer forming. Add a comment on why the lens can mislead (late binding / final layernorm).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-02-log.md</code>: at which layer did the answer appear, and where might the lens have misled you?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "resid":{"btn":"① a mid-layer residual","html":'''<span class="prompt">&gt;&gt;&gt;</span> r = residual_stream(model, prompt)[layer_12]
<span class="hl">r.shape = (768,)</span>   <span class="dim"># the running vector at layer 12</span>''',
      "take":"<b>①  Grab the residual stream.</b> The vector each layer adds to. Normally only the final one becomes a prediction — we'll decode a middle one."},
    "project":{"btn":"② project through unembedding","html":'''<span class="prompt">&gt;&gt;&gt;</span> logits = unembed(layernorm(r))   <span class="dim"># borrow the final projection</span>
<span class="prompt">&gt;&gt;&gt;</span> top = argmax(logits)
<span class="hl">top = " Paris"</span>   <span class="dim"># the guess at layer 12</span>''',
      "take":"<b>②  Decode early.</b> Apply the model's own output matrix to the mid-layer vector to read its 'current guess' — here 'Paris' already at layer 12."},
    "trace":{"btn":"③ trace across layers","html":'''<span class="prompt">&gt;&gt;&gt;</span> [logit_lens(model, prompt, L) for L in layers]
<span class="hl">L4:" the"  L8:" a"  L12:" Paris"  L16:" Paris"</span>
<span class="dim"># the answer emerges around layer 12 and locks in</span>''',
      "take":"<b>③  Watch it form.</b> Early layers are generic; the answer appears in the late-middle layers. That tells you which layers do the deciding."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='20' width='70' height='22' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='65' y='35' fill='#1F6280'>L4</text><rect x='130' y='20' width='70' height='22' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='165' y='35' fill='#1F6280'>L8</text><rect x='230' y='20' width='70' height='22' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='265' y='35' fill='#1F6280'>L12</text><rect x='330' y='20' width='70' height='22' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='365' y='35' fill='#1F6280'>L16</text><text x='450' y='35' fill='#6B645E'>→ out</text></g></svg>",
     "note":"<b>The residual stream flows through layers.</b> Each layer adds to it. Only the final vector is normally turned into a prediction."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>borrow the final unembedding</text></g></svg>",
     "note":"<b>Borrow the output projection.</b> The logit lens applies the final unembedding (and layernorm) to a middle layer instead of only the last."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='33' fill='#9A7208'>top token at layer L</text></g></svg>",
     "note":"<b>Read the guess at each layer.</b> The argmax token is the model's 'current answer' if it had to stop there."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='75' y='32' fill='#6B645E'>the</text><text x='190' y='32' fill='#6B645E'>a</text><text x='300' y='32' fill='#2D8B55'>Paris</text><text x='420' y='32' fill='#2D8B55'>Paris</text></g></svg>",
     "note":"<b>The answer sharpens.</b> Generic early, 'Paris' appears around layer 12 and locks in — the late-middle layers did the recall."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='90' y='16' width='340' height='26' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>caution: late binding / final LayerNorm can fool the lens</text></g></svg>",
     "note":"<b>Trust it carefully.</b> The lens is a cheap heuristic. On late-binding models it can show guesses the model never held — confirm with a causal test (Day 3)."},
  ],
  quiz=[
    {"q":"1. What does the logit lens do?","opts":["retrain the model","apply the final unembedding to an intermediate layer to read its 'current guess'","delete a layer","add noise"],"ans":1,"fb":"It borrows the model's output projection and applies it early, revealing the top token the model would predict at that depth."},
    {"q":"2. Across layers, a typical logit-lens trajectory looks like:","opts":["the answer is correct from layer 1","generic/unrelated early, converging to the answer in later layers","random at every layer","always the same token"],"ans":1,"fb":"Predictions usually start generic and sharpen with depth, showing where the answer gets decided."},
    {"q":"3. The logit lens requires:","opts":["training a new probe per layer","no training — just the model's existing unembedding applied to each layer","a reward model","fine-tuning"],"ans":1,"fb":"It's training-free: one matrix multiply (plus layernorm) per layer using the model's own output projection."},
    {"q":"4. On a model with heavy final LayerNorm and late binding, the logit lens can:","opts":["always be trusted","show confident intermediate 'guesses' the model never actually held","only fail at the last layer","never produce a token"],"ans":1,"fb":"Intermediate residuals may not live in the space the final unembedding expects, so early projections can be systematically wrong — verify causally."},
  ],
  fin={"em":"🔭","h3":"Day 2 complete — you can watch a prediction form!",
       "p":"You now know the logit lens: apply the model's final unembedding to intermediate residual-stream vectors to read its 'current guess' at each layer, and watch the answer sharpen with depth — for free, no training. But it's a heuristic that can mislead on late-binding models. Next: <b>Day 3</b> — activation patching, a causal test that proves which component actually matters."},
))
# ---------------- Day 3 — Activation Patching ----------------
L("day-03-activation-patching.html", dict(
  qid="m25-d03-patching", title="Module 25 · Day 3 — Activation Patching", nav_title="Spiral · M25 Day 3",
  eyebrow="Module 25 · Automate · Day 3", h1="Activation Patching: A Causal Test",
  lead="Probes and the logit lens show what's <em>present</em> — but not what the model actually <em>uses</em>. Activation patching closes that gap with a genuine causal experiment. Run the model on a clean prompt and a corrupted one, then surgically copy a single internal activation from the clean run into the corrupted run. If the output snaps back toward the clean answer, that activation truly caused the behaviour. It's the interpretability equivalent of a controlled A/B test on the model's own guts.",
  goal="<b>🎯 By the end, you'll be able to:</b> describe the clean/corrupted patching setup, explain why it gives causal (not just correlational) evidence, and say what it still can't tell you.",
  prev_href="day-02-logit-lens.html", prev_label="The Logit Lens", next_href="day-04-superposition.html", next_label="Superposition",
  sections=[
    {"title":"From correlation to cause","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–2 (probes and the logit lens read activations) and that a transformer's activations can be captured and replaced at a chosen layer/position.</div></div>
     <h4>The gap probes leave</h4>
     <p>A probe says "this info is here"; the logit lens says "the guess looks like this here". Neither proves the model <em>relies</em> on a given piece. To show cause, you must intervene and watch what changes.</p>
     <h4>The patching idea</h4>
     <p><span class="term" data-tip="A causal interpretability method: run a clean and a corrupted input, then copy (patch) one internal activation from the clean run into the corrupted run and measure how much the output recovers.">Activation patching</span> (also called causal tracing) runs two inputs: a <span class="term" data-tip="The input that produces the behaviour you want to explain, e.g. the prompt with the correct fact.">clean</span> prompt (correct answer) and a <span class="term" data-tip="A minimally-changed input that breaks the behaviour, e.g. the same prompt with the key fact swapped.">corrupted</span> prompt (wrong answer). You cache the clean activations, then re-run the corrupted prompt but overwrite <em>one</em> activation with its clean value. If the answer moves back toward the clean one, that activation carried the needed information.</p>
     <h4>Why it's powerful</h4>
     <p>Because you change exactly one thing and measure the effect, you get <span class="term" data-tip="Evidence that a component actually causes an output, obtained by intervening on it — stronger than correlation from just observing activations.">causal</span> evidence, not correlation. Sweep the patch across every layer and position and you get a heat-map of <em>which components matter</em> for that behaviour.</p>''',
     "gotit":"Got the idea"},
    {"title":"Swap one part to find the culprit","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔧</span><h5>A working and a broken machine</h5><p>You have one radio that works and one that's silent. Swap parts one at a time from the working radio into the broken one. The moment it plays, the last swapped part was the culprit. Patching swaps activations to find the part that carries the answer.</p></div>
       <div class="card"><span class="big">🗺️</span><h5>A map of what matters</h5><p>Do this swap at every component and you get a map: green where a swap fixed the output (it mattered), grey where nothing changed (it didn't). That map is the causal trace.</p></div>
     </div>
     <p><strong>In one line:</strong> copy one clean activation into a corrupted run and see if the answer recovers — the components that fix it are the ones that carry the behaviour.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: radio parts are independent. Model components interact and back each other up — sometimes several must be swapped together to see an effect, so a single-part swap can under-count what matters.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run a patch","body":'''<p>Watch a clean run cached, a corrupted run, and one activation patched in to measure the recovery. <strong>Click all three.</strong></p>'''},
    {"title":"The intervention, measured","body":
     '''<h4>Step 1 — set up clean vs corrupted</h4>
     <p>Pick a clean prompt with the behaviour and a minimally-different corrupted prompt that breaks it (e.g. swap one key token). Record the model's output on each.</p>
     <h4>Step 2 — patch and measure</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       clean_acts = run(clean_prompt)                       <span class="dim"># cache every activation</span><br>
       out = run(corrupted_prompt, patch={layer L, pos p: clean_acts[L][p]})<br>
       recovery = ( logit(clean_answer | out) − corrupted_baseline ) / ( clean_baseline − corrupted_baseline )<br>
       <span class="dim"># recovery ≈ 1 → this activation fully restores the behaviour; ≈ 0 → it doesn't matter</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       clean: "The Eiffel Tower is in" → " Paris"   (clean logit for Paris = 9.0)<br>
       corrupted: "The Colosseum is in" → " Rome"   (corrupted logit for Paris = 1.0)<br><br>
       patch layer 8 / last position → logit for Paris jumps to 7.4<br>
       recovery = (7.4 − 1.0) / (9.0 − 1.0) = 6.4 / 8.0 = <span class="hl">0.80</span><br>
       <span class="dim"># layer 8 at that position carries ~80% of the needed information</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — sufficiency is not the whole circuit.</b> A high recovery from patching one component shows it is <i>sufficient</i> to restore the behaviour here — it does NOT prove it's the only path or reveal the full mechanism. Models have <b>backup</b> components (e.g. backup attention heads) that take over when the main one is ablated, so patching can <i>under</i>-attribute (a component looks unimportant because a backup silently compensates) or <i>over</i>-attribute. A clean heat-map can quietly hide a redundant circuit. Confirm with ablation, path patching, and multiple corrupted prompts.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — causal strength vs combinatorial cost.</b> Patching gives strong causal evidence, but the search space is huge: layers × positions × components, and interactions may need patching several things at once. Full coverage is expensive. You trade completeness for tractability — usually by patching at a coarse granularity first, then zooming into the hot regions.</div></div>''',
     "gotit":"Got the intervention"},
    {"title":"Activation patching, built up","body":'''<p>Here it is assembled: the clean run cached, the corrupted run, one activation swapped in, the recovery measured, and the causal heat-map. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a patch","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_patching.py</code>. On a small model, set up a clean/corrupted factual prompt pair, cache clean activations, and patch each layer at the final position one at a time, plotting the recovery. Find the layers that carry the fact.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 3 artifact.
Create experiments/foundations/m25_patching.py: use a small model + a hooks library (e.g. TransformerLens if available) to do activation patching on a clean/corrupted factual prompt pair. Cache clean residual-stream activations, patch each layer at the last token position, and plot the recovery metric per layer to locate where the fact lives. Comment on why this is causal but shows sufficiency, not the full circuit.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-03-log.md</code>: which layers recovered the answer, and why doesn't that fully explain the circuit?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "clean":{"btn":"① cache the clean run","html":'''<span class="prompt">&gt;&gt;&gt;</span> clean = run("The Eiffel Tower is in")
<span class="hl">answer " Paris", logit 9.0</span>
<span class="prompt">&gt;&gt;&gt;</span> cache = clean.activations   <span class="dim"># save every layer/position</span>''',
      "take":"<b>①  Record the clean run.</b> The prompt that works, plus every internal activation cached for later patching."},
    "corrupt":{"btn":"② run the corrupted prompt","html":'''<span class="prompt">&gt;&gt;&gt;</span> corr = run("The Colosseum is in")
<span class="hl">answer " Rome", Paris-logit 1.0</span>
<span class="dim"># the behaviour is broken — baseline for comparison</span>''',
      "take":"<b>②  Break it minimally.</b> A tiny change flips the answer. Now we'll see which activation, restored, brings the clean answer back."},
    "patch":{"btn":"③ patch one activation","html":'''<span class="prompt">&gt;&gt;&gt;</span> out = run("The Colosseum is in", patch=cache[L8][last])
<span class="hl">Paris-logit jumps 1.0 → 7.4</span>
<span class="dim"># recovery = (7.4-1.0)/(9.0-1.0) = 0.80 → L8 carries the fact</span>''',
      "take":"<b>③  Swap in one clean activation.</b> Layer 8 restores 80% of the answer — causal evidence that this component carries the needed information."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='33' fill='#1a5c38'>clean run → \" Paris\" (cache activations)</text></g></svg>",
     "note":"<b>Run the clean prompt and cache everything.</b> This is the working behaviour and the source of the activations we'll transplant."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>corrupted run → wrong answer</text></g></svg>",
     "note":"<b>Run the corrupted prompt.</b> A minimal change breaks the behaviour. This is the baseline we try to repair."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='100' y='35' fill='#1a5c38'>clean L8</text><text x='185' y='35' fill='#C93B3B'>patch →</text><rect x='270' y='18' width='210' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='375' y='35' fill='#9A7208'>into corrupted run at L8</text></g></svg>",
     "note":"<b>Patch one activation.</b> Overwrite a single layer/position in the corrupted run with its clean value. Everything else stays corrupted."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>recovery = 0.80</text></g></svg>",
     "note":"<b>Measure the recovery.</b> How far the output snapped back toward clean. High recovery = this component causally carried the behaviour."},
    {"viz":"<svg viewBox='0 0 520 65'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='40' y='20' width='40' height='22' fill='#F5F0E8' stroke='#ccc'/><rect x='90' y='20' width='40' height='22' fill='#F5F0E8' stroke='#ccc'/><rect x='140' y='20' width='40' height='22' fill='#cfe8d8' stroke='#2D8B55'/><rect x='190' y='20' width='40' height='22' fill='#2D8B55'/><rect x='240' y='20' width='40' height='22' fill='#cfe8d8' stroke='#2D8B55'/><rect x='290' y='20' width='40' height='22' fill='#F5F0E8' stroke='#ccc'/><text x='260' y='58' fill='#6B645E'>recovery by layer — hot layers carry the behaviour</text></g></svg>",
     "note":"<b>Sweep everything → a causal map.</b> Patch each component and colour by recovery. But beware: backup components can hide, so confirm hot spots with ablation and path patching."},
  ],
  quiz=[
    {"q":"1. Why is activation patching causal rather than correlational?","opts":["it uses a bigger probe","it intervenes — changing one activation and measuring the effect — instead of only observing","it reads the final layer","it trains the model"],"ans":1,"fb":"Patching changes exactly one thing and measures the output change, giving cause-and-effect evidence a probe (pure observation) can't."},
    {"q":"2. In the setup, what is the 'corrupted' prompt?","opts":["random noise","a minimally-changed input that breaks the behaviour, used as the baseline to repair","the correct prompt","the model's output"],"ans":1,"fb":"The corrupted prompt is a small edit of the clean one that flips the answer; you patch clean activations into it and see what recovers."},
    {"q":"3. Patching layer 8 gives recovery 0.9; patching layer 3 gives 0.05. This suggests:","opts":["layer 3 carries the behaviour","layer 8 carries most of the needed information for this behaviour","both are equal","neither matters"],"ans":1,"fb":"High recovery at layer 8 means restoring it largely restores the behaviour — it causally carries the information here; layer 3 barely matters."},
    {"q":"4. A component patches to near-zero recovery, so you conclude it's unimportant. What's the risk?","opts":["none, that's correct","a backup component may silently compensate, so patching under-attributes — it shows sufficiency, not the full circuit","recovery can't be zero","the model is broken"],"ans":1,"fb":"Redundancy/backup heads can mask a component's role, so a single patch can under-count. Confirm with ablation, path patching, and multiple prompts."},
  ],
  fin={"em":"🔬","h3":"Day 3 complete — you can test causally!",
       "p":"You now know activation patching: cache a clean run, run a corrupted one, transplant a single clean activation, and measure how much the behaviour recovers — genuine causal evidence of which components matter, mapped across layers. But it shows sufficiency, not the whole circuit, and backup components can hide. Next: <b>Day 4</b> — why single neurons are so hard to read: superposition."},
))

# ---------------- Day 4 — Superposition ----------------
L("day-04-superposition.html", dict(
  qid="m25-d04-superpos", title="Module 25 · Day 4 — Superposition", nav_title="Spiral · M25 Day 4",
  eyebrow="Module 25 · Automate · Day 4", h1="Superposition: More Features Than Neurons",
  lead="Here's a puzzle: a layer might have 4,000 neurons, yet the model seems to track tens of thousands of distinct concepts. How? Superposition. When features are rare (they seldom appear together), a network can cram many of them into fewer neurons by giving each its own direction and tolerating a little interference. The cost: individual neurons become polysemantic — one neuron lights up for several unrelated things — which is exactly why reading neurons one at a time is so misleading.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain superposition, define a polysemantic neuron, and say why superposition makes single-neuron interpretation unreliable.",
  prev_href="day-03-activation-patching.html", prev_label="Activation Patching", next_href="day-05-saes.html", next_label="Sparse Autoencoders",
  sections=[
    {"title":"Cramming features into neurons","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a layer's activation is a vector; a "feature" is a direction in that vector space that means something (a concept the model tracks).</div></div>
     <h4>The counting puzzle</h4>
     <p>If each neuron stored one concept, a 4,000-neuron layer could track 4,000 concepts. But models seem to represent far more. The trick is <span class="term" data-tip="Storing more features than there are neurons by assigning each feature its own direction in activation space and tolerating small interference — works when features are sparse.">superposition</span>.</p>
     <h4>Why it works: sparsity</h4>
     <p>Most features are <span class="term" data-tip="A feature is sparse if it is active only rarely (most inputs don't trigger it). Sparsity is what lets many features share the same neurons with little interference.">sparse</span> — any given input activates only a few of them. If two features almost never fire together, they can share the same neurons: on most inputs only one is active, so they rarely interfere. The network packs many near-orthogonal feature directions into a lower-dimensional space.</p>
     <h4>The cost: polysemanticity</h4>
     <p>Because features share neurons, a single neuron ends up responding to several unrelated features — it is <span class="term" data-tip="A neuron that responds to several unrelated features, because multiple features were packed into overlapping directions (superposition). The opposite is a monosemantic neuron.">polysemantic</span>. Look at one neuron and you'll see it fire for "French text", "the color green", and "code indentation" — no single clean meaning. That is why neuron-by-neuron interpretation fails.</p>''',
     "gotit":"Got the idea"},
    {"title":"Too few drawers","body":
     '''<div class="relate">
       <div class="card"><span class="big">🗄️</span><h5>Ten labels, three drawers</h5><p>You must store 10 kinds of item in 3 drawers. If you rarely need two kinds at once, you can share drawers — mixing items that seldom clash. You fit far more than 3 kinds, at the price of occasional confusion when two share a drawer.</p></div>
       <div class="card"><span class="big">🎚️</span><h5>Overlapping signals</h5><p>Like many radio stations sharing a band: as long as two strong ones rarely broadcast at once, you can pack them close. When they do collide, you get interference — the model's small, tolerated error.</p></div>
     </div>
     <p><strong>In one line:</strong> when concepts rarely co-occur, the network stores many of them in few neurons by overlapping their directions, so each neuron ends up meaning several things.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: drawers are discrete slots. Feature directions are continuous vectors that can point anywhere and partly overlap, so the "sharing" is a matter of geometry (angles between directions), not fixed compartments.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Pack two features into one neuron","body":'''<p>Watch two sparse features stored in a single neuron, read back cleanly when only one is active, and interfere when both fire. <strong>Click all three.</strong></p>'''},
    {"title":"The geometry of packing","body":
     '''<h4>Step 1 — features as directions</h4>
     <p>Represent each feature as a direction (unit vector) in activation space. Reading a feature = projecting the activation onto its direction. With <code>n</code> neurons you have <code>n</code> truly orthogonal directions — but you can fit <em>more</em> nearly-orthogonal ones if you accept a little overlap.</p>
     <h4>Step 2 — the sparsity condition</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       activation ≈ Σ (feature_value_i · direction_i)          <span class="dim"># a sum of active features</span><br>
       if features are SPARSE (few active at once) and directions nearly orthogonal,<br>
       you can pack #features ≫ #neurons with small interference<br>
       <span class="dim"># interference ∝ overlap between the directions that happen to co-activate</span>
     </div></div>
     <h4>Step 3 — worked toy example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       1 neuron, 2 sparse features A, B (each active ~5% of the time):<br>
       store A as +1, B as −1 along the single axis<br><br>
       only A active → read +1 → recover A ✓<br>
       only B active → read −1 → recover B ✓   (they rarely collide, ~0.25% of inputs)<br>
       BOTH active → read 0 → <span class="hl">interference: neither read correctly</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — reading single neurons misleads.</b> Because of polysemanticity, the natural move — "find the neuron for concept X and watch it" — is unreliable. A neuron fires for a jumble of features, so a clean-looking "neuron for French" may also encode three other things, and a concept you care about may be split across many neurons. Interpretations built on single neurons quietly draw wrong conclusions. This is the core reason the field moved to feature-based methods (tomorrow's SAEs).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — capacity vs interpretability (and interference).</b> Superposition lets a network represent far more features than neurons — a big capacity win the model exploits for free when features are sparse. The costs: occasional interference (errors when packed features collide) and, for us, badly tangled representations that are hard to read. The model optimizes for capability, not for our ability to understand it.</div></div>''',
     "gotit":"Got the geometry"},
    {"title":"Superposition, built up","body":'''<p>Here it is assembled: features as directions, sparse activation, packing more than neurons, clean read-back, and the interference case. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove superposition","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_superposition.py</code>. Reproduce the toy superposition demo: train a tiny autoencoder to compress more sparse features than it has hidden dims, and show it packs them into overlapping directions (and interferes as sparsity drops).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 4 artifact.
Create experiments/foundations/m25_superposition.py: the Anthropic toy model of superposition — a small linear+ReLU autoencoder compressing N sparse features into m<N hidden dims. Vary feature sparsity and show that with high sparsity the model packs >m features into near-orthogonal directions (plot the feature-direction overlap matrix), and interference rises as sparsity falls.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-04-log.md</code>: how did sparsity change how many features got packed, and when did interference appear?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "store":{"btn":"① store two features in one neuron","html":'''<span class="prompt">&gt;&gt;&gt;</span> # 1 neuron, features A and B (both rare)
<span class="prompt">&gt;&gt;&gt;</span> encode: A → +1,  B → −1   <span class="dim"># opposite directions on one axis</span>
<span class="hl">two features, one neuron</span>''',
      "take":"<b>①  Pack two into one.</b> Two rarely-co-occurring features share a single neuron by using opposite directions. More features than neurons."},
    "read":{"btn":"② read back (one active)","html":'''<span class="prompt">&gt;&gt;&gt;</span> only A active → neuron = +1 → recover A ✓
<span class="prompt">&gt;&gt;&gt;</span> only B active → neuron = −1 → recover B ✓
<span class="hl">clean when one fires at a time</span>''',
      "take":"<b>②  Reads cleanly when sparse.</b> Because A and B almost never fire together, each is recovered correctly on nearly every input."},
    "interfere":{"btn":"③ interference (both active)","html":'''<span class="prompt">&gt;&gt;&gt;</span> A and B both active → neuron = +1 + (−1) = 0
<span class="bad">reads 0 → neither feature recovered</span>
<span class="dim"># the rare collision is the tolerated cost of packing</span>''',
      "take":"<b>③  Collisions cause interference.</b> On the rare input where both fire, they cancel. The model tolerates this because it almost never happens — the price of superposition."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><line x1='60' y1='40' x2='460' y2='40' stroke='#6B645E'/><line x1='140' y1='40' x2='200' y2='15' stroke='#2A7B9B' stroke-width='2'/><text x='205' y='14' fill='#1F6280'>feature A</text><line x1='140' y1='40' x2='90' y2='18' stroke='#7C6DAA' stroke-width='2'/><text x='60' y='14' fill='#5E5191'>feature B</text></g></svg>",
     "note":"<b>Features are directions.</b> Each concept is a direction in activation space. Reading it = projecting onto that direction."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>sparse: only a few features active per input</text></g></svg>",
     "note":"<b>Sparsity is the key.</b> On any input, only a handful of features fire. Rare co-activation is what lets features share neurons safely."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='14' width='340' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>#features ≫ #neurons (near-orthogonal packing)</text></g></svg>",
     "note":"<b>Pack more than neurons.</b> Many nearly-orthogonal directions fit into few dimensions — the network stores far more concepts than it has neurons."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>one neuron → many meanings (polysemantic)</text></g></svg>",
     "note":"<b>Neurons become polysemantic.</b> Sharing means a single neuron responds to several unrelated features — so reading one neuron tells you little."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='16' width='200' height='26' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='140' y='33' fill='#1a5c38'>capacity: more features</text><rect x='280' y='16' width='200' height='26' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='380' y='33' fill='#C93B3B'>cost: interference + unreadable</text></g></svg>",
     "note":"<b>The trade.</b> Superposition buys capacity for free, at the cost of occasional interference and representations that are hard for us to read — motivating the feature-extraction tools next."},
  ],
  quiz=[
    {"q":"1. What is superposition?","opts":["stacking many layers","representing more features than neurons by packing them into overlapping directions","adding noise to activations","a type of attention"],"ans":1,"fb":"Superposition packs many (sparse) feature directions into fewer neurons, tolerating small interference — more features than dimensions."},
    {"q":"2. What makes superposition possible with little damage?","opts":["large batch size","feature sparsity — features rarely fire at the same time, so shared neurons rarely collide","a high learning rate","using ReLU only"],"ans":1,"fb":"If features are sparse (seldom co-active), packed features rarely interfere, so the network can share neurons cheaply."},
    {"q":"3. A polysemantic neuron is one that:","opts":["never activates","responds to several unrelated features at once","is in the last layer","has no weights"],"ans":1,"fb":"Because features share neurons, a single neuron fires for multiple unrelated concepts — polysemantic — which is why single-neuron reading misleads."},
    {"q":"4. Two packed features A and B are both (rarely) active at once. What happens?","opts":["the model crashes","interference — their directions combine and neither is read cleanly","they merge into one feature permanently","nothing ever"],"ans":1,"fb":"On the rare collision, the overlapping directions add and corrupt the read-out. The model tolerates this because it almost never happens."},
  ],
  fin={"em":"🧩","h3":"Day 4 complete — you know why neurons are messy!",
       "p":"You now understand superposition: networks pack more features than neurons by giving each a direction and exploiting sparsity, which makes individual neurons polysemantic and unreliable to read. The trade is capacity (many features, for free) against interference and tangled, hard-to-read representations. Next: <b>Day 5</b> — Sparse Autoencoders, the tool built to untangle superposition into clean features."},
))
# ---------------- Day 5 — Sparse Autoencoders ----------------
L("day-05-saes.html", dict(
  qid="m25-d05-sae", title="Module 25 · Day 5 — Sparse Autoencoders", nav_title="Spiral · M25 Day 5",
  eyebrow="Module 25 · Automate · Day 5", h1="Sparse Autoencoders: Untangling Superposition",
  lead="Yesterday's problem: neurons are polysemantic because features are crammed together in superposition. The leading fix is a sparse autoencoder (SAE). Train a small network to take a layer's activation, expand it into a much wider but mostly-zero hidden layer, and reconstruct the original. Each of those wide hidden units learns to fire for one clean feature. The SAE effectively un-mixes the smoothie back into ingredients — turning tangled neurons into a dictionary of interpretable features.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain what an SAE does, read its reconstruction + sparsity objective, and name its failure modes (dead features, feature splitting).",
  prev_href="day-04-superposition.html", prev_label="Superposition", next_href="day-06-circuits.html", next_label="Circuits & Synthesis",
  sections=[
    {"title":"A dictionary of clean features","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 4 (superposition packs many features into few polysemantic neurons) and the autoencoder idea (encode → bottleneck → reconstruct) from M22a.</div></div>
     <h4>The goal: monosemantic features</h4>
     <p>We want <span class="term" data-tip="A unit that responds to exactly one human-understandable feature (the opposite of polysemantic). SAEs aim to recover monosemantic features.">monosemantic</span> features — units that each mean one thing. Superposition hid them by overlapping directions. An SAE tries to recover them.</p>
     <h4>What an SAE is</h4>
     <p>A <span class="term" data-tip="Sparse Autoencoder: a network trained to reconstruct a layer's activations through a much wider hidden layer that is forced to be sparse, so each hidden unit learns one clean feature.">sparse autoencoder (SAE)</span> is trained on a model's activations. It <em>encodes</em> each activation into a hidden layer that is much <em>wider</em> than the original (e.g. 8× more units) but forced to be <em>sparse</em> (only a few units active), then <em>decodes</em> back to reconstruct the activation. Because the hidden layer is wide and sparse, each unit can specialize in a single feature instead of sharing.</p>
     <h4>Why wide and sparse</h4>
     <p>Wide gives room for each packed feature to get its own unit (undoing superposition). Sparse forces the SAE to explain each activation with just a few features — the same sparsity that made superposition possible now makes the features separable. The result is a <span class="term" data-tip="The set of learned feature directions (decoder columns) of an SAE — the vocabulary of features it found in the activations.">dictionary</span> of interpretable features.</p>''',
     "gotit":"Got the idea"},
    {"title":"Un-mixing the smoothie","body":
     '''<div class="relate">
       <div class="card"><span class="big">🥤</span><h5>Back into ingredients</h5><p>A neuron's activation is a smoothie — many features blended together. The SAE is a machine that separates the smoothie back into labelled ingredients: strawberry, banana, spinach. Each ingredient is one clean feature.</p></div>
       <div class="card"><span class="big">🎛️</span><h5>A wide, mostly-off panel</h5><p>Picture a mixing desk with thousands of sliders, but only a handful raised for any given sound. The SAE learns such a panel: a huge feature vocabulary where each input lights up just a few sliders.</p></div>
     </div>
     <p><strong>In one line:</strong> an SAE expands tangled activations into a wide, sparse set of units so each one captures a single clean feature — a readable dictionary.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a smoothie has a true, fixed recipe. An SAE only <i>infers</i> features that reconstruct well and stay sparse — it may split one true concept into several units, or miss features, so its "ingredients" are a useful approximation, not ground truth.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run an SAE","body":'''<p>Watch an activation encoded into sparse features, and decoded back to reconstruct it. <strong>Click all three.</strong></p>'''},
    {"title":"Reconstruct and stay sparse","body":
     '''<h4>Step 1 — encode, then decode</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       f = ReLU(W_enc · a + b_enc)      <span class="dim"># a = activation; f = wide sparse feature codes (mostly 0)</span><br>
       â = W_dec · f + b_dec            <span class="dim"># reconstruct the activation from the few active features</span>
     </div></div>
     <h4>Step 2 — the two-part loss</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       loss = ‖ a − â ‖²   +   λ · ‖ f ‖₁<br>
       <span class="dim"># term 1: reconstruct well.  term 2: L1 penalty pushes most features to 0 (sparsity)</span>
     </div></div>
     <p>Symbols: <code>a</code> = the model's activation. <code>f</code> = the SAE's wide feature codes. <code>‖f‖₁</code> = sum of absolute values — the <span class="term" data-tip="An L1 penalty (sum of absolute values) added to the loss; it pushes most feature activations to exactly zero, enforcing sparsity.">L1 sparsity penalty</span>. <code>λ</code> sets how hard we push toward sparsity. The two terms fight: perfect reconstruction wants many active features; sparsity wants few. The balance yields a few clean, meaningful features per input.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       activation dim = 512, SAE hidden width = 4096  (8× wider)<br>
       on one input, only 20 of 4096 features are non-zero (sparse)<br>
       those 20 features reconstruct a with small error → each names one concept<br>
       <span class="dim"># inspect a feature by finding which inputs make it fire strongest</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — dead features & feature splitting.</b> Two quiet SAE pathologies. <b>Dead features:</b> many hidden units never activate on any input — wasted dictionary, and the loss looks fine because the live features still reconstruct. <b>Feature splitting:</b> one true concept gets split across several SAE units (e.g. "the same feature at different strengths"), so your dictionary over-counts and fragments meaning. Both make the SAE look healthy by loss while its <i>interpretation</i> is off. Track the fraction of live features and inspect for near-duplicate features.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — dictionary size & the reconstruction–sparsity tension.</b> A wider dictionary and stronger sparsity give cleaner, more monosemantic features — but cost more compute, produce more dead features, and if λ is too high the reconstruction degrades (you lose information about the model). Too small/weak and features stay polysemantic. There is no free lunch: you tune width and λ to trade reconstruction fidelity against interpretability, and the "right" setting is itself a research question.</div></div>''',
     "gotit":"Got the objective"},
    {"title":"The SAE, built up","body":'''<p>Here it is assembled: the activation, the wide sparse encoding, the few active features, the reconstruction, and the reconstruction–sparsity balance. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with an SAE","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_sae.py</code>. Collect activations from a small model, train a sparse autoencoder (wide hidden layer + L1 penalty), and inspect a few learned features by finding the inputs that maximally activate them. Track the fraction of dead features.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 5 artifact.
Create experiments/foundations/m25_sae.py: collect residual-stream activations from a small model on a text corpus, train a sparse autoencoder (hidden width ~8x the activation dim, loss = MSE + lambda*L1). Report reconstruction error, average number of active features per input, and the fraction of dead features; print the top-activating tokens for 3 learned features.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-05-log.md</code>: how many features stayed alive, and did any learned feature look monosemantic?</div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "encode":{"btn":"① encode to sparse features","html":'''<span class="prompt">&gt;&gt;&gt;</span> a = activation(model, text)   <span class="dim"># dim 512, tangled</span>
<span class="prompt">&gt;&gt;&gt;</span> f = relu(W_enc @ a + b)        <span class="dim"># width 4096, mostly zero</span>
<span class="hl">20 of 4096 features active</span>''',
      "take":"<b>①  Expand into a wide, sparse code.</b> The 512-dim activation becomes a 4096-wide code with only ~20 non-zeros — room for each feature to get its own unit."},
    "decode":{"btn":"② decode back","html":'''<span class="prompt">&gt;&gt;&gt;</span> a_hat = W_dec @ f + b
<span class="hl">||a - a_hat|| small</span>   <span class="dim"># the few active features reconstruct the activation</span>''',
      "take":"<b>②  Reconstruct from the few features.</b> Just those ~20 features rebuild the original activation — so they capture what it encoded."},
    "inspect":{"btn":"③ inspect a feature","html":'''<span class="prompt">&gt;&gt;&gt;</span> top_inputs(feature_137)
<span class="hl">fires on: "Paris", "Lyon", "Marseille" ...</span>
<span class="dim"># a monosemantic feature: French cities</span>''',
      "take":"<b>③  Read a clean feature.</b> Find the inputs that fire a feature hardest — often one clean concept (here: French cities). Superposition, untangled."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>tangled activation a (dim 512)</text></g></svg>",
     "note":"<b>Start with a tangled activation.</b> A polysemantic vector where features overlap — the thing we want to untangle."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='14' width='300' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>encode → wide feature layer (4096, sparse)</text></g></svg>",
     "note":"<b>Encode into a wide sparse layer.</b> Much wider than the activation, but forced mostly to zero — each unit can specialize."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='33' fill='#1a5c38'>only ~20 features active</text></g></svg>",
     "note":"<b>Few features fire.</b> The L1 penalty keeps the code sparse, so each input is explained by a small set of clean features."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>decode → reconstruct a  (small error)</text></g></svg>",
     "note":"<b>Reconstruct the activation.</b> The few active features rebuild the original, so they truly capture its content."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='200' height='26' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='140' y='35' fill='#1a5c38'>reconstruct well ↔</text><rect x='280' y='18' width='200' height='26' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='380' y='35' fill='#9A7208'>stay sparse (λ)</text></g></svg>",
     "note":"<b>The balance.</b> Reconstruction wants many features; sparsity wants few. Tuning width and λ trades fidelity against clean, monosemantic features — and risks dead features if pushed too hard."},
  ],
  quiz=[
    {"q":"1. What problem does an SAE try to solve?","opts":["slow inference","superposition — it untangles polysemantic neurons into monosemantic features","overfitting","tokenization"],"ans":1,"fb":"An SAE expands activations into a wide, sparse feature layer so each unit captures one clean feature, undoing superposition."},
    {"q":"2. The SAE loss has two terms. They are:","opts":["cross-entropy and KL","reconstruction error (MSE) plus an L1 sparsity penalty","policy and value","attention and MLP"],"ans":1,"fb":"loss = ‖a − â‖² + λ‖f‖₁: reconstruct the activation while an L1 penalty forces most features to zero."},
    {"q":"3. Why is the SAE hidden layer much WIDER than the activation?","opts":["to slow it down","to give each packed feature room for its own unit, undoing superposition","to reduce memory","to match the vocabulary"],"ans":1,"fb":"Superposition packed many features into few neurons; a wide dictionary gives each feature its own unit so it can be monosemantic."},
    {"q":"4. Your SAE reconstructs well, but half its features never activate on any input. This is:","opts":["ideal","dead features — wasted dictionary capacity that the loss doesn't flag","feature merging","a reconstruction bug"],"ans":1,"fb":"Dead features are a silent SAE pathology: many units never fire, so the effective dictionary is smaller than it looks. Track the live-feature fraction."},
  ],
  fin={"em":"📖","h3":"Day 5 complete — you can untangle features!",
       "p":"You now know sparse autoencoders: expand a layer's activations into a wide, sparse dictionary trained with reconstruction + L1 sparsity, so each unit becomes a monosemantic feature that undoes superposition. Watch for dead features and feature splitting, and the reconstruction–sparsity tension. Next: <b>Day 6</b> — circuits, and tying the whole interpretability toolkit together."},
))

# ---------------- Day 6 — Circuits & Synthesis ----------------
L("day-06-circuits.html", dict(
  qid="m25-d06-circuits", title="Module 25 · Day 6 — Circuits & Synthesis", nav_title="Spiral · M25 Day 6",
  eyebrow="Module 25 · Automate · Day 6", h1="Circuits: The Wiring of a Behaviour",
  lead="A single feature or neuron isn't the whole story — real behaviours come from circuits: connected computations spanning several attention heads and layers that together implement an algorithm. The classic example is the induction head, a pair of heads that learn to copy patterns and power in-context learning. Today you'll meet circuits and pull the whole toolkit together — probes, logit lens, patching, and SAEs — into one way of reverse-engineering a model.",
  goal="<b>🎯 By the end, you'll be able to:</b> define a circuit, explain what an induction head does, and combine the module's tools into a single interpretability workflow.",
  prev_href="day-05-saes.html", prev_label="Sparse Autoencoders", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Behaviours are circuits","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–5 (probes, logit lens, patching, superposition, SAEs) and attention heads (M3).</div></div>
     <h4>Beyond single components</h4>
     <p>A concept might live in one feature, but a <em>behaviour</em> (like "copy a name you saw earlier") usually needs several components working together. A <span class="term" data-tip="A connected set of model components (attention heads, MLP features) across layers that together implement a specific algorithm or behaviour.">circuit</span> is that connected computation — the wiring diagram of a behaviour across heads and layers.</p>
     <h4>The famous example: induction heads</h4>
     <p>An <span class="term" data-tip="A pair of attention heads that implement pattern-copying: when the current token appeared before, they attend to what followed it last time and predict that token again — the basis of in-context learning.">induction head</span> is a two-head circuit that copies patterns. Seeing <code>[A][B] … [A]</code>, it looks back to the previous <code>[A]</code>, finds what came after it (<code>[B]</code>), and predicts <code>[B]</code> again. This simple copy mechanism is a big part of how models do <em>in-context learning</em> — learning from examples in the prompt.</p>
     <h4>How circuits are found</h4>
     <p>You combine the tools: patching (Day 3) to find which heads matter, attention-pattern analysis to see what they attend to, SAE features (Day 5) to name what flows between them, and the logit lens (Day 2) to watch the prediction form. A circuit is the story that ties those clues together.</p>''',
     "gotit":"Got the idea"},
    {"title":"A reverse-engineered wiring diagram","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔌</span><h5>Trace the wires</h5><p>To understand a mystery gadget, you trace which parts connect to which and what each does. A circuit analysis traces which heads feed which, and what computation the chain performs — a wiring diagram for one behaviour.</p></div>
       <div class="card"><span class="big">📼</span><h5>Copy what came before</h5><p>Induction heads are like a tape machine: "last time I saw this token, the next one was X — so predict X." That copy trick, learned automatically, underlies learning from examples in the prompt.</p></div>
     </div>
     <p><strong>In one line:</strong> a circuit is the connected set of components that implement a behaviour, and induction heads are a clear, real example that powers in-context learning.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a gadget's wiring is fixed and discrete. Model circuits are soft, distributed, and overlapping — the same head can be part of several circuits, so a clean diagram is an idealization of messier reality.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Trace a circuit","body":'''<p>Watch an induction pattern detected, the responsible heads found by patching, and the tools combined. <strong>Click all three.</strong></p>'''},
    {"title":"The interpretability workflow","body":
     '''<h4>Step 1 — the induction mechanism</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       sequence: … [A] [B] … [A] → predict ?<br>
       induction head: attend from the current [A] to the token AFTER the previous [A],<br>
       &nbsp;&nbsp;copy it (= [B]) → predict [B]<br>
       <span class="dim"># a 2-head circuit: a 'previous-token' head + a 'copy' head</span>
     </div></div>
     <h4>Step 2 — the combined workflow</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       1. probe (Day 1)      → is the concept present at this layer?<br>
       2. logit lens (Day 2) → where does the prediction form?<br>
       3. patching (Day 3)   → which heads/positions CAUSE it?<br>
       4. SAE (Day 5)        → name the features flowing through those components<br>
       5. circuit            → connect them into one mechanism you can state in words
     </div></div>
     <p>No single tool is enough: probes show presence, the lens shows the trajectory, patching gives causation, SAEs give clean features, and the circuit is the narrative that unifies them.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — cherry-picked circuits over-generalize.</b> A circuit found on a few hand-picked examples can look crisp and complete, then fail to explain the behaviour on the full distribution — you found <i>a</i> path, not <i>the</i> mechanism. Confounds: backup heads (Day 3) fill in when you ablate, and the same behaviour may use different circuits on different inputs. A tidy diagram can quietly be a story you told yourself. Validate on held-out inputs, ablate the whole proposed circuit, and check it's necessary, not just sufficient.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — mechanistic depth vs coverage.</b> Full circuit analysis buys deep, causal understanding of <i>one</i> behaviour — genuinely explaining how the model does it. But it's labour-intensive and doesn't scale to every behaviour of a large model. You trade breadth for depth: pick the behaviours worth the effort (safety-relevant ones), and accept that most of the model stays unexplored.</div></div>''',
     "gotit":"Got the workflow"},
    {"title":"Circuits, built up","body":'''<p>Here it is assembled: the induction pattern, the two heads, the tools that reveal them, the unified circuit, and the validation caveat. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it — find an induction head","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m25_circuits.py</code>. On a small model, feed a repeated random sequence and measure per-head "induction scores" (how much each head attends from a token to the position after its previous occurrence). Identify the induction heads.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 25 Day 6 artifact.
Create experiments/foundations/m25_circuits.py: using a small model + hooks (TransformerLens if available), feed a repeated random token sequence and compute an induction score per attention head (attention from position i to the token after the previous occurrence of token i). Print the top induction heads and note this circuit underlies in-context learning; add a caveat about validating on held-out inputs.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m25/day-06-log.md</code>: which heads scored as induction heads, and how would you check the circuit generalizes?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "pattern":{"btn":"① spot the induction pattern","html":'''<span class="prompt">&gt;&gt;&gt;</span> seq = "... cat sat ... cat ___"
<span class="hl">predict: "sat"</span>   <span class="dim"># copy what followed 'cat' last time</span>''',
      "take":"<b>①  The copy pattern.</b> When a token repeats, the model predicts whatever followed it before — the signature of an induction circuit."},
    "find":{"btn":"② find the heads (patching)","html":'''<span class="prompt">&gt;&gt;&gt;</span> patch each head; measure recovery of the copy behaviour
<span class="hl">head L5.H3 + head L6.H7 = the induction circuit</span>
<span class="dim"># a previous-token head feeding a copy head</span>''',
      "take":"<b>②  Locate the circuit.</b> Activation patching (Day 3) reveals the two heads whose interaction causes the copy — the induction circuit."},
    "combine":{"btn":"③ combine the tools","html":'''<span class="prompt">&gt;&gt;&gt;</span> probe → present? · lens → forming? · patch → causal? · SAE → which feature?
<span class="hl">one mechanism, stated in words</span>''',
      "take":"<b>③  Unify into a circuit.</b> Each tool gives one clue; together they let you state the mechanism plainly — that's mechanistic interpretability."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='100' y='14' width='320' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>[A][B] … [A] → predict [B]</text></g></svg>",
     "note":"<b>The induction pattern.</b> Seeing a token repeat, the model predicts what followed it last time — the behaviour we want to explain."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='18' width='160' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='140' y='35' fill='#1F6280'>previous-token head</text><text x='245' y='35' fill='#6B645E'>→</text><rect x='290' y='18' width='160' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='370' y='35' fill='#1F6280'>copy head</text></g></svg>",
     "note":"<b>Two heads make the circuit.</b> One head marks the previous token; the next head copies what followed it. Their composition is the induction circuit."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='20' y='20' width='95' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='67' y='36' fill='#1a5c38'>probe</text><rect x='125' y='20' width='95' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='172' y='36' fill='#1a5c38'>lens</text><rect x='230' y='20' width='95' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='277' y='36' fill='#1a5c38'>patching</text><rect x='335' y='20' width='95' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='382' y='36' fill='#1a5c38'>SAE</text></g></svg>",
     "note":"<b>The tools combine.</b> Presence (probe) + trajectory (lens) + cause (patching) + clean features (SAE) — each contributes one piece of the picture."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>one mechanism, stated in words</text></g></svg>",
     "note":"<b>The circuit is the story.</b> The clues unify into a mechanism you can explain plainly — the goal of mechanistic interpretability."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='90' y='16' width='340' height='26' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>validate: held-out inputs + full-circuit ablation</text></g></svg>",
     "note":"<b>Then validate.</b> A tidy circuit can be cherry-picked. Confirm it on held-out inputs and by ablating the whole proposed circuit — necessary, not just sufficient."},
  ],
  quiz=[
    {"q":"1. What is a circuit in mechanistic interpretability?","opts":["a single neuron","a connected set of components (heads, features) across layers implementing a behaviour","the output layer","a training loss"],"ans":1,"fb":"A circuit is the wiring of a behaviour — several components working together to implement an algorithm, not one unit."},
    {"q":"2. What does an induction head do?","opts":["it normalizes activations","it copies: attend to what followed a token last time it appeared, and predict that again","it computes the loss","it stores the vocabulary"],"ans":1,"fb":"Induction heads implement pattern-copying ([A][B]…[A]→[B]), a core mechanism behind in-context learning."},
    {"q":"3. Which tool gives CAUSAL evidence about which components matter?","opts":["a linear probe","activation patching","the logit lens","an SAE"],"ans":1,"fb":"Patching (Day 3) intervenes and measures the effect — causal. Probes/lens/SAE are observational or descriptive."},
    {"q":"4. You find a crisp induction circuit on 5 hand-picked prompts. What's the risk before claiming it's THE mechanism?","opts":["none","it may over-generalize — validate on held-out inputs and ablate the whole circuit (backup heads can hide, behaviour may use other paths)","it must be retrained","circuits can't be wrong"],"ans":1,"fb":"A cherry-picked circuit can be a story you told yourself. Check necessity on held-out data and ablate the full circuit, since backups and alternate paths can mislead."},
  ],
  fin={"em":"🕸️","h3":"Day 6 complete — you can reverse-engineer behaviours!",
       "p":"You now know circuits: connected components (like induction heads) that implement a behaviour, found by combining probes, the logit lens, activation patching, and SAEs into one workflow — then validated on held-out inputs. Mechanistic depth trades against coverage. That completes Interpretability Foundations — on to the review gate."},
))
# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m25-review", title="Module 25 · Review Gate", nav_title="Spiral · M25 Review",
  eyebrow="Module 25 · The Gate — You Can Read a Model", h1="Module 25 — Review Gate",
  lead="Checkpoint. The gate question: <b>can I read what a layer knows, watch a prediction form, test a component causally, and explain why single neurons mislead — then tie it into a circuit?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Interpretability is a flagship safety direction — this working literacy is what lets you reason about what a model is actually doing.",
  prev_href="day-06-circuits.html", prev_label="Day 6 · Circuits & Synthesis",
  checks=[
    ["Day 1","I can explain a linear probe (read a frozen layer) and why a high probe score shows presence, NOT that the model uses the feature."],
    ["Day 2","I can explain the logit lens (project a mid-layer residual through the unembedding) and when it misleads (late binding)."],
    ["Day 3","I can describe activation patching (clean/corrupted, patch one activation, measure recovery) and why it's causal but shows sufficiency, not the full circuit."],
    ["Day 4","I can explain superposition (more features than neurons via sparsity) and why it makes single neurons polysemantic and hard to read."],
    ["Day 5","I can explain a sparse autoencoder (wide + sparse, reconstruction + L1) and its failure modes (dead features, feature splitting)."],
    ["Day 6","I can define a circuit, explain induction heads, and combine probes/lens/patching/SAEs into one workflow (then validate)."],
  ],
  quiz=[
    {"q":"1. A linear probe scores 92% for a concept at layer 8. This proves:","opts":["the model uses that concept","the concept is linearly readable there — not that the model relies on it","layer 8 is the output","the model was trained on it"],"ans":1,"fb":"Presence is not use — only a causal test (patching) shows the model actually relies on a feature (Day 1)."},
    {"q":"2. The logit lens works by:","opts":["training a probe per layer","applying the final unembedding to an intermediate residual to read its current guess","ablating a head","adding an L1 penalty"],"ans":1,"fb":"It borrows the model's output projection and applies it early to watch the prediction form (Day 2)."},
    {"q":"3. Patching layer 8 gives recovery 0.85 on a clean/corrupted pair. This means:","opts":["layer 8 is irrelevant","layer 8 causally carries most of the information for this behaviour","the model is broken","the probe failed"],"ans":1,"fb":"High recovery from restoring one activation = causal evidence it carries the behaviour here (Day 3)."},
    {"q":"4. Why are individual neurons often unreadable?","opts":["they have no weights","superposition makes them polysemantic — each fires for several unrelated features","they are always zero","they are in the last layer"],"ans":1,"fb":"Sparse features are packed into overlapping directions, so a neuron responds to many concepts (Day 4)."},
    {"q":"5. A sparse autoencoder untangles superposition by:","opts":["shrinking the layer","using a wide, sparse hidden layer trained with reconstruction + L1 so each unit is monosemantic","removing attention","adding more neurons to the model"],"ans":1,"fb":"The wide + sparse dictionary gives each packed feature its own unit; L1 enforces sparsity (Day 5)."},
  ],
  verdict_pass="you have working interpretability literacy — probes for presence, the logit lens for the prediction trajectory, activation patching for causal attribution, superposition + SAEs for the feature story, and circuits (like induction heads) to tie it together. You can reason concretely about what a model is doing.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The toolkit — <code>probe</code> → <code>logit lens</code> → <code>activation patching</code> → <code>superposition</code> → <code>SAE</code> → <code>circuits</code> — should feel familiar.",
  complete_label="Mark Module 25 complete",
  fin={"em":"🏆","h3":"Module 25 — passed!",
       "p":"You built interpretability literacy from scratch: linear probes read what a layer knows, the logit lens watches predictions form, activation patching gives causal evidence, superposition explains why neurons are polysemantic, sparse autoencoders untangle clean features, and circuits (like induction heads) tie components into mechanisms — always validated on held-out inputs. This is the foundation of a flagship safety research direction."},
  score_target=4,
))
print("M25 built: 6 lessons + review")



