#!/usr/bin/env python3
"""Builder for M18 — Ranking & Recommendation at serving scale.
Three authored foundation lessons (two-tower, calibration/metrics, retrieval
architecture) + five WRAP case-study lessons (video-rec, ad-click, event-rec,
similar-listings, news-feed) that teach the idea self-contained then link to the
matching ML Design README + a review gate. Rendered via _lesson_gen.
Run: python3 sessions/_build_m18.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m18")
os.makedirs(OUT, exist_ok=True)

def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ==================== Day 1 — Two-Tower Retrieval ====================
L("day-01-two-tower.html", dict(
  qid="m18-d01-twotower", title="Module 18 · Day 1 — Two-Tower Retrieval", nav_title="Spiral · M18 Day 1",
  eyebrow="Module 18 · Serve · Day 1", h1="Two-Tower Retrieval: Finding Needles Fast",
  lead="A recommender may have a hundred million items and a few tens of milliseconds to pick the best ones for you. You cannot score every item with a heavy model. The trick that makes web-scale recommendation possible is <b>retrieve-then-rank</b>, and its workhorse is the <b>two-tower model</b>: turn the query and every item into vectors once, then score by a cheap dot product.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the retrieve-then-rank pattern, describe a two-tower model (query tower + item tower scored by dot product), say why negative sampling matters, and compute a dot-product score by hand.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-calibration-metrics.html", next_label="Calibration & Ranking Metrics",
  sections=[
    {"title":"Too many items, too little time","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> M8 embeddings (turning things into vectors) and M16a serving (latency budgets). Nothing else — we start from the problem.</div></div>
     <h4>The impossible scoring problem</h4>
     <p>Say you run a video app with 100 million videos. A user opens the app. You have maybe 100 milliseconds to choose what to show. Running a big neural network on all 100 million videos would take hours. So you cannot score everything with your best model.</p>
     <h4>The fix: do it in two stages</h4>
     <p>This is the <span class="term" data-tip="A two-stage recommender: a cheap retrieval model narrows millions of items to a few hundred candidates, then an expensive ranking model orders just those.">retrieve-then-rank</span> pattern:</p>
     <ul>
       <li><b>Retrieve</b> — a very cheap model quickly narrows 100 million items down to a few hundred good <span class="term" data-tip="A small set of items (a few hundred) that the cheap retrieval stage passes to the expensive ranking stage.">candidates</span>.</li>
       <li><b>Rank</b> — a slower, smarter model carefully orders just those few hundred.</li>
     </ul>
     <p>Today is about the retrieve stage. Its job: from a huge pile, grab a small set that almost surely contains the best items, and do it in a blink.</p>
     <h4>The two-tower model</h4>
     <p>The <span class="term" data-tip="A retrieval model with two separate networks: one turns the query (user/context) into a vector, the other turns an item into a vector. Score = dot product of the two vectors.">two-tower model</span> has two separate networks. One tower (the <b>query tower</b>) turns the user and their context into a vector. The other tower (the <b>item tower</b>) turns each item into a vector. The score of an item for a query is just the <span class="term" data-tip="A single number measuring how aligned two vectors are: multiply matching entries and add them up. Bigger = more similar direction.">dot product</span> of the two vectors. Big dot product means "good match."</p>''',
     "gotit":"Got the two-stage idea"},
    {"title":"A dating app with a matching score","body":
     '''<div class="relate">
       <div class="card"><span class="big">🧲</span><h5>Two profiles, one score</h5><p>Think of a dating app. It writes a short number-summary of you (what you like) and a number-summary of each other person. To see if you match, it just lines up the two summaries and gets one number. High number = likely match. It does not re-read every profile from scratch each time.</p></div>
       <div class="card"><span class="big">📇</span><h5>Summaries computed once</h5><p>Every person's number-summary is written down ahead of time and filed away. When you open the app, it only has to make <i>your</i> summary, then quickly compare it to the filed ones. That is why it feels instant even with millions of profiles.</p></div>
     </div>
     <p><strong>In one line:</strong> summarise the query and every item as vectors, precompute the item vectors, and match by a cheap dot product — so retrieval over millions of items stays fast.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: on a dating app both people fill in their own profile. In a two-tower model the query and item towers are trained <i>together</i> so their vectors land in the same space and dot products mean something — the summaries are not independent.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Score one match by hand","body":
     '''<p>Let us walk one retrieval: the query tower makes a vector, an item tower makes a vector, and we score the match with a dot product. <strong>Click all three.</strong></p>'''},
    {"title":"Dot-product scoring and negatives","body":
     '''<h4>Step 1 — the score, in words</h4>
     <p>Each tower outputs a vector of the same length. The score of an item for a query is how aligned the two vectors are. We measure alignment by the dot product: multiply matching entries, then add them up.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       q = f<sub>query</sub>(user, context)   <span class="dim"># query vector, length d</span><br>
       v = f<sub>item</sub>(item)              <span class="dim"># item vector, length d</span><br>
       score(q, v) = q · v = Σ<sub>i</sub> q<sub>i</sub> · v<sub>i</sub>   <span class="dim"># one number per (query, item)</span>
     </div></div>
     <p>Symbols: <code>f<sub>query</sub></code> and <code>f<sub>item</sub></code> are the two towers (neural networks). <code>q</code> and <code>v</code> are their output vectors, both length <code>d</code>. The score is their dot product — a single number.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       q = [0.9, 0.1, 0.4],  v_A = [0.8, 0.0, 0.5]<br><br>
       score(q, v_A) = 0.9·0.8 + 0.1·0.0 + 0.4·0.5<br>
       score(q, v_A) = 0.72 + 0.00 + 0.20 = <span class="hl">0.92</span><br><br>
       v_B = [0.1, 0.9, 0.0]  →  0.9·0.1 + 0.1·0.9 + 0.4·0.0 = <span class="hl">0.18</span>
     </div></div>
     <p>Item A scores 0.92, item B scores 0.18. Same query, so A is retrieved before B. Because item vectors are computed <em>once</em> and stored, at serve time you only build <code>q</code> and run fast dot products — often with an <span class="term" data-tip="Approximate Nearest Neighbour search: an index (like HNSW or IVF) that finds the top items by dot product without checking every item, trading a tiny bit of accuracy for huge speed.">ANN</span> index so you do not even score every item.</p>
     <h4>How the towers learn: negative sampling</h4>
     <p>To train, you show the model a query and the item the user actually liked (a <b>positive</b>) and pull their vectors together. You also need items the user did <em>not</em> like (<b>negatives</b>) to push apart. <span class="term" data-tip="Using the other items in the same training batch as negative examples for a query — cheap, because they are already loaded.">In-batch negatives</span> reuse the other items in the batch as negatives — nearly free. <span class="term" data-tip="Negatives that are deliberately similar to the positive, so the model must learn fine distinctions instead of only easy ones.">Hard negatives</span> are items that look similar to the positive but are wrong; they force the model to learn fine differences.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — no hard negatives.</b> If you only train with easy random negatives, the model learns to tell a cooking video from a car video, but not a <i>good</i> cooking video from a mediocre one. At serve time recall looks fine on easy cases while the model can't separate close, competing items — exactly the ones ranking cares about. The training loss looks great; the top of the list is mush. You catch this only by evaluating on hard, near-duplicate pairs.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — two-tower speed vs cross-encoder accuracy.</b> The two towers never look at the query and item together, so item vectors can be precomputed and searched with ANN — millions of items in milliseconds. A <span class="term" data-tip="A model that takes the query and item together as one input and outputs a match score. Much more accurate but cannot precompute item vectors, so it is far too slow to run over millions of items.">cross-encoder</span> reads query and item jointly and is far more accurate, but it must run fresh for every pair, so it can only handle the few hundred candidates in the rank stage. Speed at retrieve, accuracy at rank — that is why you use both.</div></div>''',
     "gotit":"Got dot-product scoring"},
    {"title":"Build the two-tower retriever","body":
     '''<p>Here is the retriever built one piece at a time: the two towers, the shared vector space, precomputed item vectors, the dot-product scores, and the ANN shortlist. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a tiny retriever","body":
     '''<p>Understanding is not the same as building. Make a small two-tower retriever. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_two_tower.py</code>. Make a query tower and an item tower (small MLPs) that output d=16 vectors. Train with in-batch negatives on a toy click dataset, then add hard negatives. Precompute all item vectors, score a query by dot product, and print recall@10 with and without hard negatives.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 1 artifact.
Create experiments/foundations/m18_two_tower.py: a two-tower retriever (query tower + item tower, both small MLPs to d=16). Train with in-batch negatives on a toy click dataset, then also with hard negatives. Precompute item vectors, score by dot product, and print recall@10 for both settings so the hard-negative gain is visible.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-01-log.md</code>: what was recall@10 with vs without hard negatives, and did the score for a near-duplicate item change?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "query":{"btn":"① query tower → q","html":'''<span class="prompt">&gt;&gt;&gt;</span> q = query_tower(user, context)
<span class="dim"># turns "who + when + where" into one vector</span>
<span class="hl">q = [0.9, 0.1, 0.4]</span>   <span class="dim"># length d = 3 (tiny, for show)</span>''',
      "take":"<b>①  The query tower makes one vector.</b> It reads the user and context and outputs a short list of numbers — the query's position in vector space."},
    "item":{"btn":"② item tower → v","html":'''<span class="prompt">&gt;&gt;&gt;</span> v_A = item_tower(video_A)   <span class="dim"># computed OFFLINE, once</span>
<span class="hl">v_A = [0.8, 0.0, 0.5]</span>
<span class="prompt">&gt;&gt;&gt;</span> v_B = item_tower(video_B)
<span class="hl">v_B = [0.1, 0.9, 0.0]</span>''',
      "take":"<b>②  Item vectors are precomputed.</b> Every item is turned into a vector ahead of time and stored, so serving does not re-run the item tower."},
    "score":{"btn":"③ dot-product score","html":'''<span class="prompt">&gt;&gt;&gt;</span> score(q, v_A) = 0.9*0.8 + 0.1*0.0 + 0.4*0.5
<span class="hl">= 0.92</span>   <span class="dim"># strong match</span>
<span class="prompt">&gt;&gt;&gt;</span> score(q, v_B) = 0.9*0.1 + 0.1*0.9 + 0.4*0.0
<span class="hl">= 0.18</span>   <span class="dim"># weak match → retrieved later</span>''',
      "take":"<b>③  Match by dot product.</b> A single multiply-and-add per item gives the score. A returns first (0.92), B later (0.18). Fast enough for millions."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='30' width='120' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='120' y='51' fill='#5E5191'>query tower</text><rect x='340' y='30' width='120' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='400' y='51' fill='#1a5c38'>item tower</text><text x='260' y='51' fill='#6B645E'>two separate nets</text></g></svg>",
     "note":"<b>Two towers.</b> One network reads the user and context, the other reads an item. They are separate, but trained together so their outputs are comparable."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='120' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='120' y='43' fill='#5E5191'>query tower</text><rect x='340' y='24' width='120' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='400' y='43' fill='#1a5c38'>item tower</text><text x='120' y='78' fill='#6B645E'>q ↓</text><text x='400' y='78' fill='#6B645E'>↓ v</text><rect x='210' y='62' width='100' height='22' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='78' fill='#1F6280'>same space</text></g></svg>",
     "note":"<b>One shared space.</b> Both towers output vectors of the same length into the same space, so a dot product between them is meaningful."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='260' y='18' fill='#6B645E'>item vectors precomputed OFFLINE and filed</text><rect x='60' y='30' width='70' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='95' y='47' fill='#1a5c38'>v₁</text><rect x='150' y='30' width='70' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='185' y='47' fill='#1a5c38'>v₂</text><rect x='240' y='30' width='70' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='275' y='47' fill='#1a5c38'>v₃</text><text x='355' y='47' fill='#6B645E'>… millions</text></g></svg>",
     "note":"<b>Precompute the item side.</b> Every item's vector is built once and stored in an index. Serving never re-runs the item tower — that is the whole speed win."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='32' width='70' height='26' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='75' y='49' fill='#5E5191'>q</text><text x='125' y='49' fill='#6B645E'>·</text><rect x='150' y='32' width='60' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='180' y='49' fill='#1a5c38'>v_A</text><text x='230' y='49' fill='#2D8B55'>= 0.92</text><rect x='320' y='32' width='60' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='350' y='49' fill='#1a5c38'>v_B</text><text x='400' y='49' fill='#C93B3B'>= 0.18</text></g></svg>",
     "note":"<b>Score by dot product.</b> Line up the query vector with each item vector and add up the products. One number per item, ranked from high to low."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='30' width='140' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='110' y='49' fill='#9A7208'>ANN index</text><text x='210' y='49' fill='#6B645E'>→ top-K</text><rect x='300' y='30' width='180' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='390' y='49' fill='#1F6280'>~hundreds of candidates</text></g></svg>",
     "note":"<b>ANN grabs the shortlist.</b> An approximate-nearest-neighbour index returns the top-K by dot product without scoring every item — a few hundred candidates handed to the ranking stage."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='260' y='16' fill='#6B645E'>training pulls positives close, pushes negatives away</text><circle cx='200' cy='50' r='6' fill='#7C6DAA'/><text x='200' y='72' fill='#5E5191'>q</text><circle cx='250' cy='48' r='6' fill='#2D8B55'/><text x='252' y='40' fill='#1a5c38'>+ (liked)</text><circle cx='140' cy='55' r='6' fill='#C93B3B'/><text x='110' y='55' fill='#C93B3B'>hard neg</text><circle cx='360' cy='60' r='5' fill='#C99A12'/><text x='395' y='62' fill='#9A7208'>easy neg</text></g></svg>",
     "note":"<b>Negatives shape the space.</b> Positives are pulled toward the query; easy negatives are pushed away cheaply; hard negatives (close but wrong) force the fine distinctions that decide the top of the list."},
  ],
  quiz=[
    {"q":"1. Why do web-scale recommenders split into retrieve then rank?","opts":["to use more GPUs","because scoring every item with the heavy model is too slow, so a cheap model first narrows millions to a few hundred","to avoid embeddings","ranking is optional"],"ans":1,"fb":"Retrieval is a cheap model that shortlists candidates; ranking is the expensive model that only runs on those few hundred. That keeps latency low over millions of items."},
    {"q":"2. In a two-tower model, why can item vectors be precomputed?","opts":["the item tower ignores the item","because the item tower never sees the query, so its output does not depend on who is asking","dot products are slow","items never change"],"ans":1,"fb":"The two towers are separate; the item tower reads only the item. So each item's vector is fixed and can be built once, stored, and searched with ANN at serve time."},
    {"q":"3. Query q=[1,0,2], item v=[3,4,1]. The dot-product score is:","opts":["7","5","10","3"],"ans":1,"fb":"q·v = 1·3 + 0·4 + 2·1 = 3 + 0 + 2 = 5. Multiply matching entries, then add."},
    {"q":"4. You trained with only easy random negatives and recall looks fine, but the top items are near-duplicates of each other. Most likely cause?","opts":["the dot product is broken","no hard negatives — the model never learned to separate close, competing items","the ANN index is too fast","the query tower is missing"],"ans":1,"fb":"Easy negatives teach coarse differences only. Without hard negatives the model can't tell good from mediocre among similar items, so the top of the list collapses. Add hard negatives and evaluate on near-duplicate pairs."},
  ],
  fin={"em":"🧲","h3":"Day 1 complete — you can retrieve at scale!",
       "p":"You now have retrieve-then-rank and the two-tower model: two towers make comparable vectors, item vectors are precomputed and searched with ANN, and the score is a dot product <code>q·v</code>. You saw why hard negatives matter and the two-tower-vs-cross-encoder speed/accuracy trade-off. Next: <b>Day 2</b> — how we measure whether the ranking is any good, and why probabilities must be calibrated."},
))
print("M18: Day 1 built")

# ==================== Day 2 — Calibration & Ranking Metrics ====================
L("day-02-calibration-metrics.html", dict(
  qid="m18-d02-metrics", title="Module 18 · Day 2 — Calibration & Ranking Metrics", nav_title="Spiral · M18 Day 2",
  eyebrow="Module 18 · Serve · Day 2", h1="Calibration & Ranking Metrics",
  lead="A recommender can be great at putting the best item first and still be dangerously wrong about <i>how likely</i> a click is. Getting the order right is <b>ranking quality</b>. Getting the probability right is <b>calibration</b>. Ads auctions and many downstream decisions need both — and a model can ace one while failing the other.",
  goal="<b>🎯 By the end, you'll be able to:</b> read AUC, NDCG, and recall@k, define calibration, explain why an ad auction needs calibrated click probabilities, and compute a small NDCG example by hand.",
  prev_href="day-01-two-tower.html", prev_label="Two-Tower Retrieval", next_href="day-03-retrieval-architecture.html", next_label="Retrieval as Architecture",
  sections=[
    {"title":"Two different questions","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (retrieve-then-rank) and the idea of a predicted probability. Nothing else.</div></div>
     <h4>Order vs. probability</h4>
     <p>Imagine your model scores three items. There are two separate questions you can ask:</p>
     <ul>
       <li><b>Did it get the order right?</b> Are the items the user will like near the top? This is <span class="term" data-tip="How well a model ranks relevant items above irrelevant ones. Only the order matters, not the exact score values.">ranking quality</span>.</li>
       <li><b>Are the numbers honest?</b> If the model says "0.30 chance of click," do about 30% of such items actually get clicked? This is <span class="term" data-tip="A model is calibrated if its predicted probabilities match reality: among items it scores 0.3, about 30% really happen.">calibration</span>.</li>
     </ul>
     <h4>Ranking metrics measure the order</h4>
     <ul>
       <li><span class="term" data-tip="Area Under the ROC Curve: the probability a random positive is scored above a random negative. 0.5 = coin flip, 1.0 = perfect ordering.">AUC</span> — the chance a random liked item outranks a random disliked one. Pure ordering.</li>
       <li><span class="term" data-tip="Normalized Discounted Cumulative Gain: a ranking score that rewards putting relevant items near the top, discounting items lower down. 1.0 = ideal order.">NDCG</span> — rewards putting relevant items high; things lower in the list count less.</li>
       <li><span class="term" data-tip="Of all the relevant items, the fraction that appear in your top-k results. Measures how many good items you caught.">recall@k</span> — of all the good items, how many landed in your top-k.</li>
     </ul>
     <p>None of these care about the exact score value — only who is above whom. That is the gap calibration fills.</p>''',
     "gotit":"Got order vs probability"},
    {"title":"A weather forecaster","body":
     '''<div class="relate">
       <div class="card"><span class="big">🌦️</span><h5>Ranking the days</h5><p>A forecaster ranks the week's days from driest to rainiest. If they always put the actually-rainy days first, their <i>ranking</i> is perfect — even if their numbers are off.</p></div>
       <div class="card"><span class="big">☔</span><h5>Honest percentages</h5><p>But you also want "30% chance of rain" to mean it rains on about 3 of every 10 such days. A forecaster who ranks perfectly but always says "90%" when it's really 30% is well-ordered yet badly calibrated — and you'd carry an umbrella for nothing.</p></div>
     </div>
     <p><strong>In one line:</strong> ranking asks "is the order right?", calibration asks "are the percentages honest?" — and you can be excellent at one while failing the other.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a forecaster's mistakes just annoy you. In an ad auction, miscalibrated probabilities directly change who wins and how much money moves — the error has an immediate price tag.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"See calibration vs ranking","body":
     '''<p>Let us compare two models: one ranks perfectly but lies about probabilities, one is honest. Then compute a small NDCG. <strong>Click all three.</strong></p>'''},
    {"title":"NDCG, and why auctions need calibration","body":
     '''<h4>Step 1 — NDCG in words</h4>
     <p>NDCG rewards putting relevant items near the top. Each item contributes a <b>gain</b> (how relevant it is), discounted by how far down the list it sits. You divide by the best possible arrangement so the score lands between 0 and 1.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       DCG = Σ<sub>i=1..k</sub>  rel<sub>i</sub> / log<sub>2</sub>(i + 1)   <span class="dim"># rel = relevance, i = position (1-based)</span><br>
       NDCG = DCG / IDCG   <span class="dim"># IDCG = DCG of the ideal (best) ordering</span>
     </div></div>
     <p>Symbols: <code>rel<sub>i</sub></code> is the relevance of the item at position <code>i</code>. The <code>log<sub>2</sub>(i+1)</code> in the bottom is the position discount: lower positions count less. <code>IDCG</code> is the DCG if everything were in the perfect order.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.7">
       relevances of your top 3 = [3, 1, 2]  (order the model produced)<br><br>
       DCG = 3/log₂(2) + 1/log₂(3) + 2/log₂(4)<br>
       DCG = 3/1 + 1/1.585 + 2/2 = 3 + 0.631 + 1 = <span class="hl">4.631</span><br><br>
       ideal order = [3, 2, 1]:<br>
       IDCG = 3/1 + 2/1.585 + 1/2 = 3 + 1.262 + 0.5 = <span class="hl">4.762</span><br><br>
       NDCG = 4.631 / 4.762 = <span class="hl">0.973</span>   <span class="dim"># close to ideal</span>
     </div></div>
     <h4>Why auctions need calibration</h4>
     <p>In an ad auction the platform ranks ads by <b>expected value</b> = bid × predicted click probability (pCTR). If the pCTR is a real probability, the ranking and the price are fair. But the auction uses the <em>number itself</em>, not just the order — so an inflated pCTR makes an ad look more valuable than it is and it wins slots it should not.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — great AUC, broken bidding.</b> A model can have excellent AUC (it orders clicks above non-clicks perfectly) yet output probabilities that are all, say, 3× too high. AUC never notices — it only looks at order. But <code>bid × pCTR</code> is now 3× inflated, the auction misprices every ad, budgets burn wrong, and revenue drops. The offline ranking dashboard stays green while money leaks. You catch this only with a calibration plot, not an AUC number.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — ranking quality vs calibrated probabilities.</b> Some training tricks that sharpen ranking (like hard-negative mining or pairwise losses) push scores apart to get the order right, which can distort the probability scale. Pure calibration objectives (log loss) keep probabilities honest but may not maximise top-of-list ordering. Teams often optimise ranking, then add a calibration layer (Platt scaling / isotonic regression) on top to recover honest probabilities without hurting the order.</div></div>''',
     "gotit":"Got NDCG and calibration"},
    {"title":"Build the metric picture","body":
     '''<p>Here it is built up: the ranked list, the position discount, DCG vs the ideal (NDCG), a calibration plot, and where the auction reads the number. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a metrics script","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_metrics.py</code>. On a toy dataset of scores + true labels, compute AUC, NDCG@10, and recall@10. Then take a well-ranked model and multiply all its probabilities by 3: show AUC is unchanged but the calibration curve (predicted vs observed rate, in bins) is badly off. Fit Platt scaling to fix it.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 2 artifact.
Create experiments/foundations/m18_metrics.py: on a toy scores+labels dataset compute AUC, NDCG@10, recall@10. Then multiply all predicted probabilities by 3 to show AUC is unchanged while the reliability/calibration curve breaks; add Platt scaling (or isotonic) and show calibration recovers with AUC intact.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-02-log.md</code>: what were AUC and NDCG@10, and how far off was calibration before vs after scaling?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "ranks":{"btn":"① two models, same order","html":'''<span class="prompt">&gt;&gt;&gt;</span> true_clicks   = [1, 0, 1, 0]
<span class="prompt">&gt;&gt;&gt;</span> model_A_pCTR  = [0.9, 0.4, 0.7, 0.2]   <span class="dim"># honest</span>
<span class="prompt">&gt;&gt;&gt;</span> model_B_pCTR  = [0.99, 0.85, 0.95, 0.80] <span class="dim"># inflated</span>
<span class="hl">both rank clicks above non-clicks → same AUC</span>''',
      "take":"<b>①  Same order, different honesty.</b> A and B rank the items identically (same AUC), but B's probabilities are all pushed toward 1 — inflated."},
    "calib":{"btn":"② check calibration","html":'''<span class="prompt">&gt;&gt;&gt;</span> # of items B scored ~0.9, what fraction clicked?
<span class="hl">B says 0.90  →  reality 0.50</span>   <span class="bad"># way off</span>
<span class="hl">A says 0.50  →  reality 0.50</span>   <span class="ok"># honest</span>''',
      "take":"<b>②  Only calibration catches it.</b> B's 0.90 items click only half the time — badly calibrated. A's numbers match reality. AUC saw no difference."},
    "ndcg":{"btn":"③ compute NDCG","html":'''<span class="prompt">&gt;&gt;&gt;</span> rel = [3, 1, 2]   <span class="dim"># relevance at positions 1,2,3</span>
<span class="prompt">&gt;&gt;&gt;</span> DCG  = 3/1 + 1/1.585 + 2/2 = 4.631
<span class="prompt">&gt;&gt;&gt;</span> IDCG = 3/1 + 2/1.585 + 1/2 = 4.762
<span class="hl">NDCG = 4.631 / 4.762 = 0.973</span>''',
      "take":"<b>③  NDCG scores the order.</b> DCG rewards relevant items up top with a position discount; dividing by the ideal (IDCG) gives 0–1. 0.973 is near-perfect ordering."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='26' width='90' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='85' y='45' fill='#1a5c38'>rel=3</text><rect x='150' y='26' width='90' height='30' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='195' y='45' fill='#9A7208'>rel=1</text><rect x='260' y='26' width='90' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='305' y='45' fill='#1F6280'>rel=2</text><text x='420' y='45' fill='#6B645E'>the ranked list</text></g></svg>",
     "note":"<b>Start with the ranked list.</b> Each item has a relevance (how much the user wants it). The model produced this particular order: 3, 1, 2."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='90' y='30' fill='#6B645E'>pos 1</text><text x='90' y='48' fill='#5E5191'>÷ log₂2 = 1</text><text x='250' y='30' fill='#6B645E'>pos 2</text><text x='250' y='48' fill='#5E5191'>÷ log₂3 = 1.585</text><text x='420' y='30' fill='#6B645E'>pos 3</text><text x='420' y='48' fill='#5E5191'>÷ log₂4 = 2</text></g></svg>",
     "note":"<b>The position discount.</b> Divide each item's relevance by log₂(position+1). Items lower down are discounted more — a click at rank 1 is worth more than at rank 3."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='200' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='140' y='43' fill='#1F6280'>DCG = 4.631</text><rect x='290' y='24' width='200' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='390' y='43' fill='#1a5c38'>IDCG = 4.762</text></g></svg>",
     "note":"<b>DCG vs the ideal.</b> Add the discounted gains to get DCG (4.631). Do the same for the perfect order [3,2,1] to get IDCG (4.762)."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='20' width='220' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='42' fill='#9A7208'>NDCG = DCG / IDCG = 0.973</text></g></svg>",
     "note":"<b>NDCG normalises to 0–1.</b> Divide by IDCG so a perfect order scores 1.0. Here 0.973 — the model's order is almost ideal, no matter what the raw scores were."},
    {"viz":"<svg viewBox='0 0 520 110'><g font-family='monospace' font-size='10' text-anchor='middle'><line x1='60' y1='90' x2='60' y2='20' stroke='#6B645E'/><line x1='60' y1='90' x2='470' y2='90' stroke='#6B645E'/><text x='40' y='25' fill='#6B645E'>real</text><text x='260' y='108' fill='#6B645E'>predicted</text><line x1='60' y1='90' x2='440' y2='25' stroke='#2D8B55' stroke-dasharray='4,3'/><text x='300' y='45' fill='#1a5c38'>ideal (y=x)</text><polyline points='60,90 160,82 260,72 360,60 440,50' fill='none' stroke='#C93B3B' stroke-width='2'/><text x='330' y='80' fill='#C93B3B'>inflated model</text></g></svg>",
     "note":"<b>The calibration plot.</b> Predicted probability on the x-axis, observed rate on the y-axis. The honest model hugs the diagonal; the inflated one sags below — same AUC, broken probabilities."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='135' y='43' fill='#1F6280'>bid × pCTR</text><text x='240' y='43' fill='#6B645E'>→</text><rect x='280' y='24' width='200' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='380' y='43' fill='#9A7208'>who wins &amp; the price</text></g></svg>",
     "note":"<b>Where the number is spent.</b> The auction multiplies bid by pCTR to rank and price ads. It uses the value itself — so if pCTR is inflated, the whole auction misprices. Calibration is not optional here."},
  ],
  quiz=[
    {"q":"1. AUC measures:","opts":["whether predicted probabilities match reality","how well the model orders positives above negatives (order only)","the click count","the number of ads"],"ans":1,"fb":"AUC = the chance a random positive outranks a random negative. It only sees order, so it is blind to whether the probabilities are honest."},
    {"q":"2. A model has AUC 0.95 but its calibration curve sags far below the diagonal. What is true?","opts":["it is both well-ranked and well-calibrated","it ranks well but its probabilities are dishonest (miscalibrated)","AUC must be wrong","recall is 0"],"ans":1,"fb":"High AUC means good ordering; a sagging calibration curve means the predicted probabilities don't match observed rates. Ordering and calibration are separate properties."},
    {"q":"3. Why does an ad auction specifically need calibrated pCTR, not just good ranking?","opts":["auctions are random","because it ranks and prices by bid × pCTR, using the probability value itself — inflated pCTR misprices ads","calibration is faster","ranking is illegal in auctions"],"ans":1,"fb":"The auction multiplies bid by the predicted click probability. If that probability is inflated, expected value is wrong, so the wrong ads win and prices are off — even with perfect ordering."},
    {"q":"4. Top-3 relevances are [2, 3, 1] (model's order). Using DCG = Σ relᵢ/log₂(i+1), the DCG is about:","opts":["6.00","4.39","3.00","2.00"],"ans":1,"fb":"DCG = 2/log₂2 + 3/log₂3 + 1/log₂4 = 2/1 + 3/1.585 + 1/2 = 2 + 1.893 + 0.5 ≈ 4.39."},
  ],
  fin={"em":"📏","h3":"Day 2 complete — you can measure ranking honestly!",
       "p":"You can now separate <b>ranking quality</b> (AUC, NDCG, recall@k — order only) from <b>calibration</b> (do the probabilities match reality?). You computed NDCG = DCG/IDCG by hand and saw why an ad auction, which spends <code>bid × pCTR</code>, breaks when pCTR is inflated even at perfect AUC. Next: <b>Day 3</b> — retrieval as architecture: contrastive loss, cross-encoders, ColBERT, and Matryoshka embeddings."},
))
print("M18: Day 2 built")

# ==================== Day 3 — Retrieval as Architecture ====================
L("day-03-retrieval-architecture.html", dict(
  qid="m18-d03-retrieval", title="Module 18 · Day 3 — Retrieval as Architecture", nav_title="Spiral · M18 Day 3",
  eyebrow="Module 18 · Serve · Day 3", h1="Retrieval as Architecture",
  lead="Day 1 gave you the two-tower idea. Today we open the toolbox that turns it into a real retrieval system: the <b>InfoNCE</b> contrastive loss that trains the towers, the <b>cross-encoder</b> rerank that fixes precision, <b>ColBERT</b> late interaction that sits between them, and <b>Matryoshka</b> embeddings that let one vector be cheap or accurate on demand.",
  goal="<b>🎯 By the end, you'll be able to:</b> read the InfoNCE loss, place cross-encoder rerank and ColBERT late interaction on the speed/accuracy line, explain Matryoshka nested dimensions, and reason about why reranking too few candidates caps recall.",
  prev_href="day-02-calibration-metrics.html", prev_label="Calibration & Ranking Metrics", next_href="day-04-video-rec.html", next_label="Video Recommendation",
  sections=[
    {"title":"The retrieval toolbox","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (two-tower, dot-product scoring, in-batch negatives) and Day 2 (recall@k). Nothing else.</div></div>
     <h4>Four tools, one system</h4>
     <p>A production retrieval stack mixes a few ideas. Here is the map before the details:</p>
     <ul>
       <li><span class="term" data-tip="A contrastive loss: for a query, treat its true match as the positive and all other in-batch items as negatives, then maximise the softmax probability of the positive.">InfoNCE</span> — the loss that <em>trains</em> the two towers, using in-batch negatives.</li>
       <li><span class="term" data-tip="A model that reads the query and item together and outputs one match score. Very accurate, but too slow to run over millions — used only to rerank a shortlist.">Cross-encoder rerank</span> — a slow, accurate model that re-scores the shortlist.</li>
       <li><span class="term" data-tip="A retrieval method that keeps a vector per token and scores a query-item pair by summing each query token's best match against item tokens (MaxSim). More accurate than a single vector, cheaper than a full cross-encoder.">ColBERT late interaction</span> — a middle ground: one vector per token, matched cheaply.</li>
       <li><span class="term" data-tip="Embeddings trained so that the first k dimensions are already a usable smaller embedding. One vector serves many sizes, like nested Russian dolls.">Matryoshka embeddings</span> — one embedding whose first few dimensions already work, so you pick cost vs accuracy at query time.</li>
     </ul>
     <p>The theme: a <span class="term" data-tip="A retriever where the query and item are encoded separately into one vector each, then scored by dot product. The two-tower model is a bi-encoder. Fast, precomputable.">bi-encoder</span> (two-tower) is fast but coarse; a cross-encoder is precise but slow. The tools let you place each stage exactly where you want on that line.</p>''',
     "gotit":"Got the toolbox"},
    {"title":"A library with two kinds of helper","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏃</span><h5>The fast helper (bi-encoder)</h5><p>A helper who has memorised a one-line summary of every book. Ask a question and they instantly hand you a shelf of maybe-relevant books. Fast, but their one-line summaries miss nuance, so a few duds sneak in.</p></div>
       <div class="card"><span class="big">🔍</span><h5>The careful helper (cross-encoder)</h5><p>A second helper who actually reads your question next to each book, cover to cover, and ranks them precisely. Far more accurate — but so slow you can only ask them about the small shelf the first helper picked.</p></div>
     </div>
     <p><strong>In one line:</strong> the fast bi-encoder shortlists from millions with cheap precomputed vectors; the careful cross-encoder reranks only that shortlist for precision — and ColBERT and Matryoshka are dials in between.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: the fast helper's summaries are not written by hand — they are learned by InfoNCE so that a dot product between summaries actually predicts relevance. The quality of retrieval lives or dies by that training loss.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Walk the InfoNCE loss","body":
     '''<p>Let us walk one InfoNCE step, then see the cross-encoder rerank and a Matryoshka cut. <strong>Click all three.</strong></p>'''},
    {"title":"InfoNCE, and the recall ceiling","body":
     '''<h4>Step 1 — the contrastive idea, in words</h4>
     <p>For a query, one item is the true match (the positive) and every other item in the batch is a negative. InfoNCE says: make the positive's score high and all the negatives' scores low. It is a softmax over the batch where the positive should win.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       L = − log [ exp(q·v⁺ / τ) / Σ<sub>j</sub> exp(q·v<sub>j</sub> / τ) ]<br>
       <span class="dim"># v⁺ = the positive item; v_j runs over positive + all in-batch negatives; τ = temperature</span>
     </div></div>
     <p>Symbols: <code>q·v⁺</code> is the score of the true match; the sum in the bottom runs over the positive and all negatives; <code>τ</code> (tau) is a <span class="term" data-tip="A number that scales the scores before softmax. Small τ makes the model separate items sharply; large τ softens the distribution.">temperature</span> that controls how sharply the model separates items. Minimising L pushes the positive's score above the negatives'.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.6">
       scores (already ÷τ): positive = 2.0, two negatives = 1.0, 0.0<br><br>
       exp: e² = 7.389, e¹ = 2.718, e⁰ = 1.000  →  sum = 11.107<br>
       softmax(positive) = 7.389 / 11.107 = 0.665<br>
       L = − log(0.665) = <span class="hl">0.408</span>   <span class="dim"># lower as the positive pulls further ahead</span>
     </div></div>
     <h4>The stack, on one line</h4>
     <p>bi-encoder (fast, coarse) → optional ColBERT (per-token, medium) → cross-encoder rerank (slow, precise) on the shortlist. Matryoshka lets the bi-encoder's vector be truncated to fewer dims for a cheaper first pass.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the recall ceiling.</b> The cross-encoder can only reorder the candidates retrieval already handed it. If a truly relevant item never made the shortlist, no reranker can bring it back. So reranking too few candidates (say top-20 instead of top-500) <b>caps recall</b>: your precision metrics on the shortlist look great while the best item silently sits at rank 60, never seen. The reranker cannot fix a retrieval miss — recall is set upstream.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — bi-encoder recall vs cross-encoder precision, and embedding dim vs cost.</b> More candidates and a heavier reranker raise precision but cost latency and compute. A single-vector bi-encoder is cheapest but coarsest; ColBERT's per-token vectors are more accurate but use much more storage; a full cross-encoder is most accurate but cannot scale past a shortlist. Matryoshka gives a third axis: bigger embedding dimension = more accuracy but more memory and slower ANN, and you can truncate to trade one for the other <i>without retraining</i>.</div></div>''',
     "gotit":"Got InfoNCE and the ceiling"},
    {"title":"Build the retrieval stack","body":
     '''<p>Here is the stack built up: InfoNCE training, the fast bi-encoder shortlist, ColBERT late interaction, the cross-encoder rerank, and a Matryoshka truncation. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a rerank sweep","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_rerank.py</code>. Train a bi-encoder with InfoNCE, retrieve top-N candidates, then rerank with a small cross-encoder. Sweep N ∈ {10, 50, 200, 1000} and plot final recall@10 vs N to see the recall ceiling. Bonus: truncate the embedding (Matryoshka style) to 8/16/32 dims and show the accuracy/cost trade.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 3 artifact.
Create experiments/foundations/m18_rerank.py: train a small bi-encoder with InfoNCE (in-batch negatives), retrieve top-N, rerank with a small cross-encoder. Sweep N in {10,50,200,1000} and plot final recall@10 vs N to show the recall ceiling set by retrieval. Also truncate embeddings to 8/16/32 dims (Matryoshka) and show accuracy vs cost.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-03-log.md</code>: at what N did recall@10 stop improving, and what did truncating the embedding cost you?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "infonce":{"btn":"① InfoNCE step","html":'''<span class="prompt">&gt;&gt;&gt;</span> scores = [2.0, 1.0, 0.0]   <span class="dim"># positive first, then negatives (÷τ)</span>
<span class="prompt">&gt;&gt;&gt;</span> p = softmax(scores)[0]
<span class="hl">p(positive) = 0.665</span>
<span class="prompt">&gt;&gt;&gt;</span> loss = -log(0.665) = <span class="hl">0.408</span>''',
      "take":"<b>①  InfoNCE makes the positive win.</b> A softmax over the batch; the loss drops as the true match pulls ahead of the in-batch negatives."},
    "rerank":{"btn":"② cross-encoder rerank","html":'''<span class="prompt">&gt;&gt;&gt;</span> shortlist = biencoder.retrieve(q, top=200)
<span class="prompt">&gt;&gt;&gt;</span> reordered = cross_encoder.score(q, shortlist)
<span class="dim"># cross-encoder reads q AND item together — accurate, slow</span>
<span class="hl">precision@10 up; only 200 items scored</span>''',
      "take":"<b>②  Rerank the shortlist.</b> The cross-encoder re-scores only the candidates retrieval passed it — accurate, but it can't recover an item that never made the shortlist."},
    "matryoshka":{"btn":"③ Matryoshka cut","html":'''<span class="prompt">&gt;&gt;&gt;</span> full = embed(item)   <span class="dim"># length 256, trained Matryoshka-style</span>
<span class="prompt">&gt;&gt;&gt;</span> cheap = full[:32]   <span class="hl"># first 32 dims already usable</span>
<span class="dim"># small = fast/cheap ANN; full = most accurate — no retrain</span>''',
      "take":"<b>③  One vector, many sizes.</b> Matryoshka trains so the first k dims are a working smaller embedding. Truncate for speed, keep full for accuracy — same model."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='22' width='280' height='32' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='42' fill='#5E5191'>InfoNCE: positive wins vs in-batch negatives</text></g></svg>",
     "note":"<b>Train with InfoNCE.</b> A softmax over the batch pushes the true match's score above every negative's. This is what makes dot products meaningful."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='120' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='100' y='43' fill='#1a5c38'>bi-encoder</text><text x='185' y='43' fill='#6B645E'>→</text><rect x='210' y='24' width='260' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='340' y='43' fill='#1F6280'>ANN top-N shortlist (fast, coarse)</text></g></svg>",
     "note":"<b>Fast shortlist.</b> The bi-encoder + ANN return the top-N candidates from millions in milliseconds. Coarse, but it sets the recall ceiling for everything after."},
    {"viz":"<svg viewBox='0 0 520 76'><g font-family='monospace' font-size='10' text-anchor='middle'><text x='260' y='16' fill='#6B645E'>ColBERT: per-token vectors, MaxSim match</text><circle cx='90' cy='45' r='5' fill='#7C6DAA'/><circle cx='120' cy='45' r='5' fill='#7C6DAA'/><circle cx='150' cy='45' r='5' fill='#7C6DAA'/><text x='120' y='68' fill='#5E5191'>query tokens</text><circle cx='330' cy='45' r='5' fill='#2D8B55'/><circle cx='360' cy='45' r='5' fill='#2D8B55'/><circle cx='390' cy='45' r='5' fill='#2D8B55'/><text x='360' y='68' fill='#1a5c38'>item tokens</text><line x1='150' y1='45' x2='330' y2='45' stroke='#C99A12' stroke-width='1.5'/></g></svg>",
     "note":"<b>ColBERT sits in the middle.</b> Keep a vector per token; score by summing each query token's best match against item tokens. More accurate than one vector, far cheaper than a full cross-encoder."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='170' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='125' y='43' fill='#9A7208'>cross-encoder</text><text x='230' y='43' fill='#6B645E'>reads q+item together</text><rect x='360' y='24' width='120' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='420' y='43' fill='#1F6280'>precise order</text></g></svg>",
     "note":"<b>Rerank for precision.</b> The cross-encoder reads query and item jointly — most accurate — but only on the shortlist. It reorders; it cannot rescue a missed item."},
    {"viz":"<svg viewBox='0 0 520 76'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='30' width='360' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><rect x='60' y='30' width='60' height='24' rx='4' fill='#2D8B55'/><text x='90' y='47' fill='#fff'>32</text><line x1='120' y1='24' x2='120' y2='60' stroke='#C93B3B' stroke-dasharray='3,2'/><text x='260' y='47' fill='#1a5c38'>… up to 256</text><text x='260' y='72' fill='#6B645E'>Matryoshka: first k dims already work</text></g></svg>",
     "note":"<b>Matryoshka truncation.</b> The embedding is trained so its first k dimensions are a usable smaller embedding. Cut to 32 dims for a cheap first pass, keep 256 for accuracy — no retraining."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10' text-anchor='middle'><line x1='60' y1='60' x2='470' y2='60' stroke='#6B645E' stroke-width='2'/><circle cx='90' cy='60' r='5' fill='#2D8B55'/><text x='90' y='80' fill='#1a5c38'>bi-encoder</text><text x='90' y='40' fill='#6B645E'>fast/coarse</text><circle cx='265' cy='60' r='5' fill='#C99A12'/><text x='265' y='80' fill='#9A7208'>ColBERT</text><circle cx='440' cy='60' r='5' fill='#C93B3B'/><text x='440' y='80' fill='#C93B3B'>cross-encoder</text><text x='440' y='40' fill='#6B645E'>slow/precise</text></g></svg>",
     "note":"<b>The whole line.</b> Speed on the left, accuracy on the right. Each stage picks a spot: retrieve cheaply, optionally refine with ColBERT, rerank precisely — recall set early, precision late."},
  ],
  quiz=[
    {"q":"1. What does the InfoNCE loss train the two towers to do?","opts":["predict the exact click probability","score the true match (positive) above the in-batch negatives via a softmax","memorise every item id","ignore negatives"],"ans":1,"fb":"InfoNCE is contrastive: for each query it maximises the softmax probability of the positive against the in-batch negatives, so dot products separate matches from non-matches."},
    {"q":"2. Why can't a cross-encoder replace the bi-encoder for first-stage retrieval over millions of items?","opts":["it is less accurate","it reads query and item together, so it can't precompute item vectors — far too slow at that scale","it has no parameters","it only works for images"],"ans":1,"fb":"A cross-encoder must run fresh for every query-item pair, so it can't be precomputed or ANN-searched. It is only affordable on the small shortlist."},
    {"q":"3. You rerank only the top-20 candidates and precision@10 looks great, but overall recall is low. Why?","opts":["the cross-encoder is broken","reranking too few candidates caps recall — a relevant item outside the top-20 can never be recovered","NDCG is undefined","the temperature is wrong"],"ans":1,"fb":"The reranker can only reorder what retrieval passed it. If a relevant item wasn't in the shortlist, no reranker brings it back. Recall is set by the retrieval stage, so widen N."},
    {"q":"4. What is the point of Matryoshka embeddings?","opts":["they encrypt the vectors","the first k dimensions are already a usable smaller embedding, so you can truncate for speed or keep full for accuracy without retraining","they remove the item tower","they double the temperature"],"ans":1,"fb":"Matryoshka nests smaller embeddings inside the full one, so a single trained vector serves many sizes — pick cost vs accuracy at query time, no retrain needed."},
  ],
  fin={"em":"🧰","h3":"Day 3 complete — you can architect a retriever!",
       "p":"You now hold the retrieval toolbox: <b>InfoNCE</b> trains the towers (positive over in-batch negatives), a <b>cross-encoder</b> reranks the shortlist for precision, <b>ColBERT</b> late interaction sits in between, and <b>Matryoshka</b> embeddings trade dimension for cost with no retrain. You saw the recall ceiling: a reranker can't rescue what retrieval missed. Next: five case studies putting this to work — starting with <b>video recommendation</b>."},
))
print("M18: Day 3 built")

# ==================== Day 4 — Video Recommendation (case study) ====================
L("day-04-video-rec.html", dict(
  qid="m18-d04-videorec", title="Module 18 · Day 4 — Video Recommendation", nav_title="Spiral · M18 Day 4",
  eyebrow="Module 18 · Serve · Day 4 · Case Study", h1="Video Recommendation",
  lead="Now we put retrieve-then-rank to work on a real system: the home feed of a video app. Millions of videos, one user, a few dozen slots. You will see how candidate generation, ranking by watch-time, offline and online metrics, and feedback loops fit together — the whole loop from Days 1–2 in one product.",
  goal="<b>🎯 By the end, you'll be able to:</b> lay out a video recommender as candidate generation + ranking, pick a watch-time/engagement target, name the offline (NDCG) and online (A/B) metrics, and spot the feedback-loop drift that quietly poisons it.",
  prev_href="day-03-retrieval-architecture.html", prev_label="Retrieval as Architecture", next_href="day-05-ad-click.html", next_label="Ad Click Prediction",
  sections=[
    {"title":"The video feed problem","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (retrieve-then-rank, two-tower) and Day 2 (NDCG, calibration). This lesson applies them.</div></div>
     <h4>What the system does</h4>
     <p>A user opens the app. The job: fill the feed with videos this user is likely to watch and enjoy, out of a catalogue of millions, in under ~100 ms. This is exactly the retrieve-then-rank pattern from Day 1, made concrete.</p>
     <h4>Two stages, again</h4>
     <ul>
       <li><span class="term" data-tip="The retrieval stage for videos: cheap models (often two-tower + ANN) narrow millions of videos to a few hundred candidates for this user.">Candidate generation</span> — cheap models pull a few hundred candidate videos from millions (two-tower + ANN, plus sources like "watched-by-similar-users").</li>
       <li><b>Ranking</b> — a heavier model scores those few hundred and orders them for the slots.</li>
     </ul>
     <h4>What are we ranking for?</h4>
     <p>Clicks are easy to game (clickbait thumbnails). Most video systems rank for <span class="term" data-tip="How long a user actually watches, often weighted by video length so a full short view and a partial long view compare fairly. A better proxy for satisfaction than a click.">watch-time</span> or a blend of engagement signals (watch-time, likes, shares, completion) — a better proxy for real satisfaction than a bare click.</p>''',
     "gotit":"Got the video setup"},
    {"title":"A TV channel that learns you","body":
     '''<div class="relate">
       <div class="card"><span class="big">📺</span><h5>A channel just for you</h5><p>Imagine a TV channel that rebuilds its whole schedule every time you sit down, guessing what you'll actually watch to the end — not just click on. It first grabs a big pile of maybe-shows (candidates), then carefully orders the best few for tonight.</p></div>
       <div class="card"><span class="big">🔁</span><h5>It watches you back</h5><p>Whatever it shows shapes what you watch, and what you watch trains tomorrow's schedule. If it only ever shows cooking videos, it will only ever learn you like cooking — even if you'd have loved travel.</p></div>
     </div>
     <p><strong>In one line:</strong> generate candidate videos cheaply, rank them by predicted watch-time/engagement, and remember that today's feed becomes tomorrow's training data.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a TV channel's viewers are fixed. A recommender changes user behaviour — it can create the very tastes it then claims to have discovered. That two-way loop is the source of the drift below.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Trace one recommendation","body":
     '''<p>Let us trace one feed: candidate generation, ranking by watch-time, and the metrics you'd check. <strong>Click all three.</strong></p>'''},
    {"title":"Metrics, feedback loops, and drift","body":
     '''<h4>Step 1 — measuring offline, in words</h4>
     <p>Before shipping, you replay logged data: given what the user later watched, did your ranking put those videos high? That is an <span class="term" data-tip="Evaluation on logged historical data before deploying — cheap and fast, but only sees what was already shown.">offline metric</span> like NDCG (from Day 2).</p>
     <h4>Step 2 — measuring online</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       offline:  NDCG@k, recall@k on logged plays   <span class="dim"># cheap, but only sees past exposures</span><br>
       online:   A/B test → Δ watch-time, Δ retention   <span class="dim"># the truth, but slow and costly</span>
     </div></div>
     <p>An <span class="term" data-tip="A live experiment: split users into a control group (old model) and a treatment group (new model), then compare real outcomes like watch-time and retention.">A/B test</span> splits users into control and treatment and compares real outcomes. Offline metrics guide iteration; the A/B test is the verdict.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.7">
       new model: offline NDCG@10 0.71 → 0.78  (looks great)<br>
       A/B test:  watch-time +4%,  BUT 7-day retention −1%<br>
       <span class="dim"># it won short-term watch-time by pushing addictive clickbait,</span><br>
       <span class="dim"># which cost long-term retention — offline never saw this</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the feedback loop.</b> The model only ever gets feedback on videos it chose to show. Videos it never surfaces get no clicks, look "bad," and get shown even less — a self-reinforcing spiral. Popular items get more popular, niche-but-loved items vanish, and the training data drifts toward the model's own past choices. Offline metrics improve because you are grading the model on the world it created. You catch this only with exploration (showing some random/uncertain items) and by watching diversity and long-term retention, not just replayed clicks.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — engagement now vs satisfaction later.</b> Optimising pure short-term watch-time reliably lifts the offline number and the first A/B week, but can push clickbait and autoplay traps that erode trust and retention. Blending in completion, explicit "not interested," and diversity, and holding out a long-horizon retention metric, trades a little immediate engagement for a healthier long-term product. There is no single metric that is safe alone.</div></div>
     <div class="callout c-info"><span class="ic">↗</span><div><b>Go deeper:</b> the full system design — candidate sources, feature engineering, serving, and monitoring — is in <a href="../../ML%20Design/06-video-recommendation/README.md">ML Design · 06 Video Recommendation</a>.</div></div>''',
     "gotit":"Got metrics and drift"},
    {"title":"Build the video recommender","body":
     '''<p>Here it is built up: the user request, candidate generation, watch-time ranking, offline vs online metrics, and the feedback loop that closes back on training. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a mini recommender","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_video_rec.py</code>. On a toy watch-log, build candidate generation (two-tower recall) + a watch-time ranker. Report offline NDCG@10. Then simulate a feedback loop: retrain only on the model's own shown items for a few rounds and watch catalogue diversity collapse.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 4 artifact.
Create experiments/foundations/m18_video_rec.py: toy video recommender with two-tower candidate generation + a watch-time ranker. Report offline NDCG@10. Then simulate a feedback loop (retrain only on the model's own shown items over several rounds) and plot how catalogue diversity / long-tail coverage collapses. Add an exploration knob and show it helps.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-04-log.md</code>: what NDCG@10 did you get, and how fast did diversity collapse without exploration?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "candidates":{"btn":"① candidate generation","html":'''<span class="prompt">&gt;&gt;&gt;</span> q = query_tower(user, context)
<span class="prompt">&gt;&gt;&gt;</span> cands = ann_index.search(q, top=300)
<span class="dim"># millions of videos → a few hundred, in a blink</span>
<span class="hl">300 candidate videos</span>''',
      "take":"<b>①  Narrow the catalogue.</b> Two-tower + ANN pull a few hundred candidates from millions — the cheap retrieve stage from Day 1."},
    "rank":{"btn":"② rank by watch-time","html":'''<span class="prompt">&gt;&gt;&gt;</span> scores = ranker.predict_watchtime(user, cands)
<span class="prompt">&gt;&gt;&gt;</span> feed = cands.sorted_by(scores)[:20]
<span class="dim"># heavier model, only 300 items → fast enough</span>
<span class="hl">top 20 fill the feed</span>''',
      "take":"<b>②  Order for satisfaction.</b> The ranker predicts watch-time (not just clicks) on the shortlist and fills the slots — the expensive rank stage."},
    "metrics":{"btn":"③ offline vs online","html":'''<span class="prompt">&gt;&gt;&gt;</span> offline NDCG@10: 0.71 → 0.78   <span class="ok"># better order</span>
<span class="prompt">&gt;&gt;&gt;</span> A/B: watch-time +4%, retention -1%
<span class="hl">offline ↑ but long-term ↓ — trust the A/B</span>''',
      "take":"<b>③  Two kinds of truth.</b> Offline NDCG guides iteration; the A/B test is the verdict — and it can disagree with offline when short-term engagement hurts retention."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='16' width='180' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>user opens the app</text></g></svg>",
     "note":"<b>A request arrives.</b> One user, their context (time, device, history), and a ~100 ms budget to fill the feed from millions of videos."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='16' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='115' y='35' fill='#1F6280'>two-tower + ANN</text><text x='215' y='35' fill='#6B645E'>→</text><rect x='250' y='16' width='230' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='365' y='35' fill='#1F6280'>~300 candidate videos</text></g></svg>",
     "note":"<b>Candidate generation.</b> Cheap retrieval (plus sources like co-watch) narrows millions to a few hundred candidates for this user."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='16' width='170' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='145' y='35' fill='#9A7208'>watch-time ranker</text><text x='255' y='35' fill='#6B645E'>→</text><rect x='290' y='16' width='190' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='385' y='35' fill='#1a5c38'>top 20 → the feed</text></g></svg>",
     "note":"<b>Ranking.</b> A heavier model scores the shortlist by predicted watch-time/engagement — not raw clicks — and fills the slots."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='16' width='190' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='135' y='35' fill='#1F6280'>offline: NDCG@k (fast)</text><rect x='290' y='16' width='190' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='385' y='35' fill='#9A7208'>online: A/B (the verdict)</text></g></svg>",
     "note":"<b>Two metrics.</b> Offline NDCG on logged plays for fast iteration; a live A/B test on watch-time and retention for the real decision."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='30' width='120' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='120' y='48' fill='#1a5c38'>feed shown</text><rect x='340' y='30' width='120' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='400' y='48' fill='#5E5191'>train data</text><path d='M180,44 H340' stroke='#6B645E' stroke-width='1.5'/><path d='M340,66 Q260,86 180,66' fill='none' stroke='#C93B3B' stroke-width='1.5' stroke-dasharray='4,3'/><text x='260' y='84' fill='#C93B3B'>loop: only shown items get feedback</text></g></svg>",
     "note":"<b>The feedback loop closes.</b> The feed becomes tomorrow's training data. Unshown videos get no feedback and fade — the drift you must fight with exploration and diversity metrics."},
  ],
  quiz=[
    {"q":"1. Why is a video recommender split into candidate generation then ranking?","opts":["to use two datacenters","because scoring millions of videos with the heavy ranker is too slow, so cheap retrieval shortlists a few hundred first","ranking is optional","to avoid watch-time"],"ans":1,"fb":"Same retrieve-then-rank pattern as Day 1: cheap candidate generation narrows millions to hundreds, then the heavy ranker orders just those within the latency budget."},
    {"q":"2. Why rank by watch-time/engagement instead of raw clicks?","opts":["clicks are illegal","clicks are easily gamed by clickbait; watch-time is a better proxy for real satisfaction","watch-time is free","clicks need a GPU"],"ans":1,"fb":"A click only says the thumbnail worked. Watch-time (weighted by length) better reflects whether the user actually valued the video, and is harder to game with clickbait."},
    {"q":"3. New model: offline NDCG@10 jumps 0.71→0.78, but the A/B shows watch-time +4% and 7-day retention −1%. What do you conclude?","opts":["ship it — offline improved","offline order improved but it may be pushing short-term-engaging clickbait that hurts long-term retention; trust the A/B","the A/B is broken","NDCG is wrong"],"ans":1,"fb":"Offline only sees replayed exposures and short-term order. The A/B reveals a long-horizon cost. When they disagree, the online long-term metric wins."},
    {"q":"4. Over months, popular videos dominate and niche-but-loved videos disappear from the feed. This is:","opts":["a rendering bug","the feedback loop — the model only gets feedback on what it shows, so unshown items fade and training data drifts toward its own past choices","too much exploration","a calibration error"],"ans":1,"fb":"Classic feedback-loop drift: unshown items get no positive signal, look worse, get shown less. Fight it with exploration and by monitoring diversity and long-term retention, not just replayed clicks."},
  ],
  fin={"em":"📺","h3":"Day 4 complete — a full video recommender!",
       "p":"You assembled the whole loop: candidate generation (retrieve) + watch-time ranking, judged by offline NDCG and online A/B, with the feedback loop as the silent enemy. This is Days 1–2 made into a product. For the full system design, follow the Go-deeper link. Next: <b>Day 5</b> — ad click prediction, where calibration becomes money."},
))
print("M18: Day 4 built")

# ==================== Day 5 — Ad Click Prediction (case study) ====================
L("day-05-ad-click.html", dict(
  qid="m18-d05-adclick", title="Module 18 · Day 5 — Ad Click Prediction", nav_title="Spiral · M18 Day 5",
  eyebrow="Module 18 · Serve · Day 5 · Case Study", h1="Ad Click Prediction",
  lead="Ads are where calibration stops being academic and becomes money. The system predicts the probability a user clicks an ad (pCTR), and an auction spends that number directly. Today you'll see CTR prediction as <b>calibrated ranking</b>, why feature crosses matter, how the auction uses pCTR, and why a miscalibrated model quietly bleeds revenue.",
  goal="<b>🎯 By the end, you'll be able to:</b> frame CTR prediction as calibrated ranking, explain feature crosses, walk a second-price auction using pCTR, and say exactly why calibration (not just AUC) decides the money.",
  prev_href="day-04-video-rec.html", prev_label="Video Recommendation", next_href="day-06-event-rec.html", next_label="Event Recommendation",
  sections=[
    {"title":"Predicting a click, spending the number","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (calibration, AUC) most of all, plus retrieve-then-rank from Day 1. This lesson makes calibration concrete.</div></div>
     <h4>What the system predicts</h4>
     <p>For each candidate ad, the model predicts <span class="term" data-tip="Predicted click-through rate: the model's estimated probability that this user clicks this ad in this context.">pCTR</span> — the probability this user clicks this ad here. Retrieval shortlists ads; the CTR model ranks them; the auction picks winners.</p>
     <h4>Why the number itself matters</h4>
     <p>Unlike the video feed, the ad system does not just use the <em>order</em> of pCTR — it uses the <em>value</em>. The auction ranks by <b>expected value = bid × pCTR</b>. So pCTR must be a real probability, not just a good ordering. That makes ad CTR a <span class="term" data-tip="A task where you must both rank items correctly AND output honest probabilities, because a downstream system spends the probability value.">calibrated ranking</span> problem — Day 2's calibration is now the core of the product.</p>
     <h4>Feature crosses</h4>
     <p>Click behaviour depends on <em>combinations</em>: "user in country=US" × "ad category=shoes" behaves differently from either alone. A <span class="term" data-tip="A new feature built by combining two raw features (e.g. country × ad-category), letting a linear model capture interactions it otherwise can't.">feature cross</span> multiplies signals so a simple model can capture these interactions — the classic trick behind Wide &amp; Deep and factorization machines.</p>''',
     "gotit":"Got CTR + auction"},
    {"title":"A market stall barker","body":
     '''<div class="relate">
       <div class="card"><span class="big">📣</span><h5>Whose sign goes up front</h5><p>A market has one prime billboard and many sellers. Each seller offers to pay per glance. The market puts up the sign that earns it the most per glance — but "per glance" depends on how likely a passer-by is to actually look and buy, not just what the seller offers.</p></div>
       <div class="card"><span class="big">⚖️</span><h5>Honest odds decide the price</h5><p>If the market wrongly believes shoe ads get looked at twice as often as they really do, it puts shoe ads up front and overcharges — until sellers notice they're paying for glances that never happen. Honest odds keep the market fair and profitable.</p></div>
     </div>
     <p><strong>In one line:</strong> the auction ranks and prices ads by bid × predicted click probability, so the probability must be honest — a miscalibrated pCTR misprices every ad.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a human barker can eyeball a crowd and adjust. The ad system commits to its pCTR for billions of auctions per hour with no human in the loop — a small calibration error, multiplied across that volume, is a large amount of money.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Walk one ad auction","body":
     '''<p>Let us walk one auction: candidate ads, pCTR prediction, expected value, and the winner + price. <strong>Click all three.</strong></p>'''},
    {"title":"The auction math and calibration","body":
     '''<h4>Step 1 — expected value, in words</h4>
     <p>An advertiser's value from a slot is what they'll pay if clicked (their bid), times how likely a click is (pCTR). The auction ranks ads by this expected value, so a lower bidder with a much higher click chance can still win.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       expected value  eCPM ∝ bid × pCTR<br>
       rank ads by eCPM, highest wins<br>
       <span class="dim"># second-price: winner pays the minimum bid that would still have won</span>
     </div></div>
     <p>Symbols: <code>bid</code> = what the advertiser pays per click; <code>pCTR</code> = predicted click probability; <code>eCPM</code> = expected earnings per thousand impressions. In a <span class="term" data-tip="An auction where the winner pays just enough to have beaten the runner-up, not their own bid. Encourages advertisers to bid their true value.">second-price auction</span> the winner pays the price that would just have beaten the next ad.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.6">
       Ad A: bid $2.00, pCTR 0.10 → eCPM ∝ 0.200<br>
       Ad B: bid $5.00, pCTR 0.03 → eCPM ∝ 0.150<br><br>
       A wins (0.200 &gt; 0.150), even though B bid more.<br>
       <span class="dim"># now inflate everyone's pCTR by 2×: order is unchanged (AUC fine)…</span><br>
       <span class="dim"># …but the price the auction charges is computed wrong → revenue leaks</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — great AUC, mispriced auction.</b> A CTR model can rank clicks above non-clicks perfectly (high AUC) while its probabilities are systematically off — say every pCTR is 1.5× too high. The <i>ranking</i> of ads may look unchanged, but the auction spends <code>bid × pCTR</code> to set prices and budgets, so it overcharges, exhausts advertiser budgets early, and mis-allocates slots. The offline AUC dashboard stays green while money quietly leaks. You catch it only with a calibration plot and monitoring predicted-vs-actual CTR, never with AUC.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — ranking quality vs calibrated probabilities.</b> Losses and tricks that sharpen ranking can distort the probability scale; a pure log-loss keeps probabilities honest but may not maximise top-slot ordering. In practice teams train for ranking, then add a calibration layer (Platt/isotonic) and continuously recalibrate as traffic drifts — because in ads, an un-calibrated model with perfect AUC is still broken.</div></div>
     <div class="callout c-info"><span class="ic">↗</span><div><b>Go deeper:</b> the full system — feature pipelines, model choices, serving, and calibration monitoring — is in <a href="../../ML%20Design/08-ad-click-prediction/README.md">ML Design · 08 Ad Click Prediction</a>.</div></div>''',
     "gotit":"Got the auction + calibration"},
    {"title":"Build the ad system","body":
     '''<p>Here it is built up: candidate ads, feature crosses, the calibrated pCTR model, expected value, and the second-price auction. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with an auction sim","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_ad_auction.py</code>. Train a small pCTR model with a feature cross, report AUC and a calibration curve. Then run a second-price auction over synthetic ads; inflate all pCTRs by 1.5× and show AUC is unchanged while auction revenue/prices go wrong. Recalibrate and show it recovers.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 5 artifact.
Create experiments/foundations/m18_ad_auction.py: a small pCTR model with a feature cross (report AUC + calibration curve), then a second-price auction over synthetic ads ranking by bid*pCTR. Inflate all pCTRs by 1.5x to show AUC unchanged but auction prices/revenue break; add Platt/isotonic recalibration and show it recovers.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-05-log.md</code>: how much did auction revenue distort under inflated pCTR while AUC stayed flat?</div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "cands":{"btn":"① candidate ads + features","html":'''<span class="prompt">&gt;&gt;&gt;</span> ads = retrieve(user, context, top=100)
<span class="prompt">&gt;&gt;&gt;</span> x = features(user, ad)
<span class="hl">cross: country=US × cat=shoes</span>   <span class="dim"># interaction feature</span>''',
      "take":"<b>①  Shortlist and featurize.</b> Retrieval narrows the ad pool; features include crosses like country×category that a simple model needs to capture interactions."},
    "pctr":{"btn":"② predict pCTR","html":'''<span class="prompt">&gt;&gt;&gt;</span> pCTR_A = model(user, ad_A) = 0.10
<span class="prompt">&gt;&gt;&gt;</span> pCTR_B = model(user, ad_B) = 0.03
<span class="dim"># must be an HONEST probability, not just ranked</span>''',
      "take":"<b>②  Predict the click probability.</b> The pCTR must be calibrated — the auction spends the number itself, so honest probabilities matter as much as order."},
    "auction":{"btn":"③ run the auction","html":'''<span class="prompt">&gt;&gt;&gt;</span> eCPM_A = 2.00 * 0.10 = 0.200
<span class="prompt">&gt;&gt;&gt;</span> eCPM_B = 5.00 * 0.03 = 0.150
<span class="hl">A wins (0.200 &gt; 0.150) despite B's higher bid</span>''',
      "take":"<b>③  Rank by bid × pCTR.</b> Expected value decides the winner and price. A lower bid with higher click odds beats a high bid with low odds."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='35' fill='#1F6280'>~100 candidate ads</text></g></svg>",
     "note":"<b>Candidate ads.</b> Retrieval shortlists eligible ads for this user and context — the same retrieve stage, now for ads."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='90' height='26' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='85' y='35' fill='#5E5191'>country=US</text><text x='140' y='35' fill='#6B645E'>×</text><rect x='160' y='18' width='90' height='26' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='205' y='35' fill='#5E5191'>cat=shoes</text><text x='265' y='35' fill='#6B645E'>→</text><rect x='290' y='18' width='190' height='26' rx='4' fill='#FCF3DC' stroke='#C99A12'/><text x='385' y='35' fill='#9A7208'>cross feature</text></g></svg>",
     "note":"<b>Feature crosses.</b> Combine raw features so the model captures interactions (US × shoes) that neither feature reveals alone."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>calibrated pCTR model → honest probability</text></g></svg>",
     "note":"<b>Predict calibrated pCTR.</b> The model outputs a real click probability, not just a rank — because the next step spends the value."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='130' y='16' width='260' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>eCPM ∝ bid × pCTR</text></g></svg>",
     "note":"<b>Expected value.</b> Multiply each ad's bid by its pCTR. This is where a calibration error turns directly into a pricing error."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='150' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='135' y='43' fill='#9A7208'>rank by eCPM</text><text x='230' y='43' fill='#6B645E'>→</text><rect x='260' y='24' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='370' y='43' fill='#1a5c38'>winner + second-price</text></g></svg>",
     "note":"<b>The auction decides.</b> Highest expected value wins; the winner pays just enough to have beaten the runner-up. Honest pCTR keeps this fair and profitable."},
  ],
  quiz=[
    {"q":"1. Why is ad CTR prediction a calibrated ranking problem, not just ranking?","opts":["ads are random","the auction spends the pCTR value itself (bid × pCTR), so the probability must be honest, not just correctly ordered","calibration is faster","clicks don't matter"],"ans":1,"fb":"The auction ranks and prices by expected value = bid × pCTR. It uses the number, so a well-ordered but inflated pCTR still misprices — calibration is core, not optional."},
    {"q":"2. What is a feature cross for?","opts":["to encrypt features","to combine features (e.g. country × ad-category) so the model captures interactions a single feature can't","to remove the auction","to lower latency"],"ans":1,"fb":"Click behaviour depends on combinations. A cross multiplies signals so even a simple model can represent 'US users click shoe ads' as its own feature."},
    {"q":"3. Ad A: bid $1.00, pCTR 0.20. Ad B: bid $4.00, pCTR 0.04. Ranking by bid × pCTR, who wins?","opts":["Ad B, it bid more","Ad A — 1.00×0.20 = 0.20 beats 4.00×0.04 = 0.16","they tie","neither"],"ans":1,"fb":"A: 1.00×0.20 = 0.20. B: 4.00×0.04 = 0.16. A wins on expected value despite B's higher bid — that is why calibrated pCTR is decisive."},
    {"q":"4. Your CTR model has excellent AUC but every pCTR is ~1.5× too high. What breaks?","opts":["nothing, AUC is what matters","the auction misprices ads and mis-allocates budgets because bid × pCTR is inflated — revenue leaks while AUC stays green","the retrieval stage","the feature cross"],"ans":1,"fb":"AUC only sees order. The auction spends the inflated probability, so prices and budgets go wrong. Monitor calibration (predicted vs actual CTR), not just AUC, and recalibrate."},
  ],
  fin={"em":"💰","h3":"Day 5 complete — calibration is money!",
       "p":"You framed ad CTR as <b>calibrated ranking</b>: retrieval shortlists ads, a calibrated pCTR model scores them (with feature crosses for interactions), and a second-price auction spends <code>bid × pCTR</code> — so an inflated but well-ranked model quietly bleeds revenue. Follow the Go-deeper link for the full design. Next: <b>Day 6</b> — event recommendation, where items expire and cold-start bites."},
))
print("M18: Day 5 built")

# ==================== Day 6 — Event Recommendation (case study) ====================
L("day-06-event-rec.html", dict(
  qid="m18-d06-eventrec", title="Module 18 · Day 6 — Event Recommendation", nav_title="Spiral · M18 Day 6",
  eyebrow="Module 18 · Serve · Day 6 · Case Study", h1="Event Recommendation",
  lead="Recommending events — concerts, meetups, games this weekend — breaks the usual playbook in two ways. Events <b>expire</b>: a concert is useless the day after. And most events are brand new with no history, so <b>cold-start</b> is the normal case, not the exception. Location and time become first-class features.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why time-bound items make cold-start the default, use location and time features, handle event freshness/expiry, and name the drift that comes from a constantly-churning item set.",
  prev_href="day-05-ad-click.html", prev_label="Ad Click Prediction", next_href="day-07-similar-listings.html", next_label="Similar Listings",
  sections=[
    {"title":"Items that expire","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (retrieve-then-rank, embeddings) and Day 4 (candidate generation + ranking). This lesson stresses the time and cold-start twist.</div></div>
     <h4>What makes events different</h4>
     <p>Most recommenders assume items live a long time and gather clicks. Events do not:</p>
     <ul>
       <li><span class="term" data-tip="An item that is only valid within a time window (an event has a start time and expires after it happens). Recommending it late is useless.">Time-bound</span> — an event has a date. After it passes, recommending it is worthless.</li>
       <li><span class="term" data-tip="The problem of recommending an item (or serving a user) with little or no interaction history to learn from.">Cold-start</span> — a brand-new event has zero clicks. And by the time it has history, it may be about to expire. So you must recommend well from features alone.</li>
     </ul>
     <h4>Features that carry the weight</h4>
     <p>With little interaction history, <span class="term" data-tip="Attributes of the item or user known without any interaction data — for events: category, price, description text, and especially where and when.">content features</span> do the heavy lifting: category, description text, price, and above all <b>location</b> (is it near the user?) and <b>time</b> (is it soon, and does it fit when the user is free?).</p>
     <h4>The retrieval twist</h4>
     <p>Candidate generation must filter by "near me" and "still upcoming" before ranking. A great event across the country next year is a bad candidate. Geo and time filters are part of retrieval, not an afterthought.</p>''',
     "gotit":"Got the event twist"},
    {"title":"A weekend what's-on board","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎟️</span><h5>This weekend, near you</h5><p>Think of a local "what's on" board. It only lists things happening soon and close by. A concert in another city next month is useless to post today — no matter how good it is.</p></div>
       <div class="card"><span class="big">🆕</span><h5>Everything is new</h5><p>Most listings went up this week and have no reviews yet. The board must guess who'll enjoy them from the description, price, place, and time — because waiting for reviews means the event is already over.</p></div>
     </div>
     <p><strong>In one line:</strong> events expire and start with no history, so retrieval filters by place and time and ranking leans on content features — you cannot wait for clicks to arrive.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a paper board is static. Here the candidate set changes hour by hour as events are added and expire, so the model's world is never the same twice — which is exactly what makes drift constant.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Recommend one event","body":
     '''<p>Let us walk one event feed: geo/time candidate filter, content-based ranking, and a freshness decay. <strong>Click all three.</strong></p>'''},
    {"title":"Cold-start, freshness, and drift","body":
     '''<h4>Step 1 — freshness decay, in words</h4>
     <p>An event's usefulness drops as its date nears and vanishes after it passes. We fold this into the score with a time-decay factor so far-off or expired events sink automatically.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       final_score = content_relevance × geo_fit × freshness(Δt)<br>
       freshness(Δt) = 0 if the event has passed, else decays as Δt grows<br>
       <span class="dim"># Δt = time until the event; content_relevance from features (cold-start-safe)</span>
     </div></div>
     <p>Symbols: <code>content_relevance</code> comes from item/user features (works with no clicks); <code>geo_fit</code> scores distance; <code>freshness(Δt)</code> zeroes expired events and downweights far-off ones.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.6">
       Event X: relevance 0.8, geo_fit 1.0, in 2 days → freshness 0.9 → score 0.72<br>
       Event Y: relevance 0.9, geo_fit 1.0, in 200 days → freshness 0.2 → score 0.18<br>
       Event Z: relevance 0.95, but already passed → freshness 0 → score <span class="hl">0.00</span><br>
       <span class="dim"># X wins: slightly less relevant, but soon and near — Z is dropped outright</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — cold-start collapse to popularity.</b> When a model can't learn a new event from clicks, it leans on whatever signal it has — usually popularity or organizer size. New, niche, or first-time events get almost no exposure, gather no clicks, and are judged "irrelevant," so they never surface. The catalogue silently narrows to big, known events. Offline metrics on logged (popular) events look fine. You catch it by tracking coverage of new/cold items and by forcing some exploration for fresh events.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — content features vs interaction signal.</b> Leaning on content (text, geo, time) solves cold-start and handles churn, but content features are weaker predictors than real behaviour once it exists. Leaning on clicks is more accurate but useless for new events and always lagging in a fast-churning catalogue. Event systems must blend both and shift weight from content to behaviour as an event ages — a moving target.</div></div>
     <div class="callout c-info"><span class="ic">↗</span><div><b>Go deeper:</b> the full design — data sources, geo indexing, cold-start modeling, and monitoring — is in <a href="../../ML%20Design/07-event-recommendation/README.md">ML Design · 07 Event Recommendation</a>.</div></div>''',
     "gotit":"Got cold-start + freshness"},
    {"title":"Build the event recommender","body":
     '''<p>Here it is built up: the request with a place and time, geo/time candidate filter, content-based ranking, freshness decay, and the cold-start/drift watch. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with an event ranker","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_event_rec.py</code>. Build content-based candidate generation (embed event text + geo + time), rank with a freshness decay, and simulate cold-start: add brand-new events with zero history and show whether they ever surface with vs without an exploration bonus.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 6 artifact.
Create experiments/foundations/m18_event_rec.py: content-based event recommender (embed text + geo + time), rank with score = relevance * geo_fit * freshness(delta_t), dropping expired events. Simulate cold-start (new events with zero history) and show coverage of new items with vs without an exploration bonus.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-06-log.md</code>: did new events ever surface without exploration, and how did freshness decay change the top list?</div></div>''',
     "gotit":"Done — on to Day 7"},
  ],
  demos={
    "filter":{"btn":"① geo + time filter","html":'''<span class="prompt">&gt;&gt;&gt;</span> cands = events.filter(near=user_loc, when=">= now")
<span class="dim"># drop far-away and already-passed events first</span>
<span class="hl">120 upcoming, nearby events</span>''',
      "take":"<b>①  Filter before ranking.</b> Retrieval keeps only events that are near the user and still upcoming — geo and time are part of candidate generation."},
    "rank":{"btn":"② content-based rank","html":'''<span class="prompt">&gt;&gt;&gt;</span> relevance = model(user_features, event_text, price, cat)
<span class="dim"># no clicks needed → cold-start safe</span>
<span class="hl">ranks new events from features alone</span>''',
      "take":"<b>②  Rank from content.</b> With little interaction history, the ranker uses the event's text, category, price, and the user's profile — so brand-new events can still be scored."},
    "fresh":{"btn":"③ apply freshness","html":'''<span class="prompt">&gt;&gt;&gt;</span> X: 0.8 * 1.0 * freshness(2 days=0.9)  = 0.72
<span class="prompt">&gt;&gt;&gt;</span> Y: 0.9 * 1.0 * freshness(200 days=0.2)= 0.18
<span class="prompt">&gt;&gt;&gt;</span> Z: passed → freshness 0 → <span class="hl">0.00 (dropped)</span>''',
      "take":"<b>③  Decay by time.</b> A soon-and-near event beats a slightly-more-relevant far-off one, and expired events are zeroed out entirely."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>request: user + place + time</text></g></svg>",
     "note":"<b>The request carries place and time.</b> Unlike a video feed, where and when the user is free are core inputs, not extras."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='16' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='115' y='35' fill='#1F6280'>geo + time filter</text><text x='215' y='35' fill='#6B645E'>→</text><rect x='250' y='16' width='230' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='365' y='35' fill='#1F6280'>nearby &amp; upcoming only</text></g></svg>",
     "note":"<b>Filter first.</b> Drop far-away and already-passed events before ranking — a brilliant event you can't attend is a bad candidate."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>content ranking (text, price, cat) — no clicks needed</text></g></svg>",
     "note":"<b>Rank from content.</b> Because most events are cold, the ranker leans on features, not interaction history — cold-start is the default case."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10' text-anchor='middle'><line x1='60' y1='70' x2='470' y2='70' stroke='#6B645E'/><line x1='60' y1='70' x2='60' y2='20' stroke='#6B645E'/><path d='M60,25 Q120,28 200,45 Q300,66 470,70' fill='none' stroke='#C99A12' stroke-width='2'/><text x='90' y='40' fill='#9A7208'>soon = high</text><text x='400' y='60' fill='#9A7208'>far = low</text><text x='260' y='88' fill='#6B645E'>freshness(Δt): decays to 0 by event time</text></g></svg>",
     "note":"<b>Freshness decay.</b> Multiply the score by a time factor that falls as the event nears and hits zero once it passes — expired events drop out automatically."},
    {"viz":"<svg viewBox='0 0 520 76'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='26' width='200' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='140' y='45' fill='#C93B3B'>cold-start → popularity collapse</text><rect x='300' y='26' width='180' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='390' y='45' fill='#1a5c38'>fix: explore new events</text></g></svg>",
     "note":"<b>Watch cold-start drift.</b> Without exploration the feed collapses to popular, known events and new ones never surface. Track new-item coverage and force some exploration."},
  ],
  quiz=[
    {"q":"1. Why is cold-start the normal case for event recommendation, not the exception?","opts":["events have too many clicks","most events are brand new with no interaction history, and by the time they gather history they may be expiring","events never expire","users are anonymous"],"ans":1,"fb":"Events are time-bound and constantly added, so most candidates have little or no history. You must recommend from content features, because waiting for clicks means the event is over."},
    {"q":"2. Why are geo and time filters part of retrieval, not just ranking?","opts":["to save memory","a far-away or already-passed event is a bad candidate regardless of relevance, so you filter it out before scoring","filtering is illegal in ranking","time is not a feature"],"ans":1,"fb":"No amount of relevance rescues an event you can't attend. Filtering by 'near me' and 'still upcoming' during candidate generation keeps the shortlist useful."},
    {"q":"3. Event X: relevance 0.8, geo 1.0, freshness 0.9. Event Z: relevance 0.95, but it already happened (freshness 0). Ranking by relevance×geo×freshness, who is shown?","opts":["Z, it's most relevant","X — 0.8×1.0×0.9 = 0.72, while Z is 0.95×1.0×0 = 0, so Z is dropped","they tie","neither"],"ans":1,"fb":"X scores 0.72; Z scores 0 because freshness zeroes an expired event. Relevance can't rescue an event that already happened."},
    {"q":"4. Over time the event feed shows only big, popular events and new listings never appear. This is:","opts":["a caching bug","cold-start collapse to popularity — new events get no clicks, look irrelevant, and are never surfaced; fix with exploration and new-item coverage tracking","too much freshness decay","a geo error"],"ans":1,"fb":"Without a way to give new events exposure, they get no feedback, look 'bad,' and vanish. Monitor coverage of cold items and force some exploration for fresh events."},
  ],
  fin={"em":"🎟️","h3":"Day 6 complete — you can recommend expiring items!",
       "p":"You handled the event twist: items are <b>time-bound</b> and <b>cold-start</b> is the default, so retrieval filters by place and time, ranking leans on content features, and a <b>freshness decay</b> drops expired events. You saw cold-start collapse to popularity and the content-vs-behaviour trade-off. Follow the Go-deeper link for the full design. Next: <b>Day 7</b> — similar listings, pure item-to-item retrieval."},
))
print("M18: Day 6 built")

# ==================== Day 7 — Similar Listings (case study) ====================
L("day-07-similar-listings.html", dict(
  qid="m18-d07-similar", title="Module 18 · Day 7 — Similar Listings", nav_title="Spiral · M18 Day 7",
  eyebrow="Module 18 · Serve · Day 7 · Case Study", h1="Similar Listings",
  lead="\"You're looking at this home — here are others like it.\" That is item-to-item recommendation: no user model needed, just a good notion of similarity between items. It is the purest use of embeddings + ANN from Days 1 and 3, with one hard extra: listings appear and vanish, so <b>freshness</b> of the index matters.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain item-to-item similarity via embeddings, use ANN retrieval to find nearest items, reason about index freshness for a churning catalogue, and pick a sensible offline metric.",
  prev_href="day-06-event-rec.html", prev_label="Event Recommendation", next_href="day-08-news-feed.html", next_label="Personalized News Feed",
  sections=[
    {"title":"Items like this one","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (embeddings, ANN, dot product) and Day 3 (bi-encoder retrieval). This is item-to-item, the simplest shape.</div></div>
     <h4>No user needed</h4>
     <p>On a listing page (a home, a product, a rental) the query is not a user — it is the <em>current item</em>. The job: find other items most similar to it. This is <span class="term" data-tip="Recommending items similar to a given item, using item embeddings and nearest-neighbour search — no per-user model required.">item-to-item recommendation</span>, and it needs only a good similarity between items.</p>
     <h4>Similarity is a nearest-neighbour search</h4>
     <p>Each listing gets an embedding (from its photos, text, price, location). "Similar" means "close in embedding space." So the whole system is: embed the current item, then run an <span class="term" data-tip="Approximate Nearest Neighbour search: quickly returns the closest vectors to a query without scanning every item, using an index like HNSW or IVF.">ANN</span> search for its nearest neighbours. This is Day 1's retrieve stage with the item itself as the query.</p>
     <h4>The catch: the catalogue churns</h4>
     <p>Listings sell, expire, or change price constantly. If the ANN index is stale, you recommend homes that are already gone. So <span class="term" data-tip="How up-to-date the retrieval index is. A stale index recommends items that have sold, expired, or changed — a real problem for fast-churning catalogues.">index freshness</span> — how often you re-embed and re-index — is a core design decision, not a detail.</p>''',
     "gotit":"Got item-to-item"},
    {"title":"A 'more like this' shelf","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏠</span><h5>More like this home</h5><p>You're viewing a two-bedroom near the park. A good shelf shows other two-bedrooms nearby in the same price range — things close to what you're already looking at. No profile of you required; the home you're on <i>is</i> the query.</p></div>
       <div class="card"><span class="big">🗂️</span><h5>A card catalogue that ages</h5><p>Each home has an index card describing it. To find similar homes you flip to nearby cards. But if a home sold last week and its card is still in the drawer, you'll recommend something no one can buy. Someone must keep the drawer current.</p></div>
     </div>
     <p><strong>In one line:</strong> embed each item, and "similar" is just its nearest neighbours by ANN — with the index kept fresh so you never recommend items that are gone.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: index cards are read by a person who might notice a sold home. The ANN index has no common sense — if a stale vector is closest, it is returned, sold or not. Freshness must be engineered.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Find similar items","body":
     '''<p>Let us walk one lookup: embed the current listing, ANN-search neighbours, and filter out stale/sold ones. <strong>Click all three.</strong></p>'''},
    {"title":"Similarity, ANN, and freshness","body":
     '''<h4>Step 1 — similarity, in words</h4>
     <p>Two listings are similar if their embeddings point in nearly the same direction. We measure that with a dot product (or cosine similarity, which is the dot product of unit-length vectors) — exactly Day 1's scoring, now item-vs-item.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       sim(a, b) = e<sub>a</sub> · e<sub>b</sub>          <span class="dim"># dot product of the two item embeddings</span><br>
       neighbours(a) = ANN.top_k( e<sub>a</sub> )   <span class="dim"># k closest items, excluding a itself</span>
     </div></div>
     <p>Symbols: <code>e<sub>a</sub></code>, <code>e<sub>b</sub></code> are item embeddings; <code>sim</code> is their dot product; the ANN index returns the top-k closest to <code>e<sub>a</sub></code> without scanning every item.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.7">
       current e_a = [0.6, 0.8]<br>
       e_b = [0.5, 0.8] → sim = 0.6·0.5 + 0.8·0.8 = 0.30 + 0.64 = <span class="hl">0.94</span><br>
       e_c = [0.9, 0.1] → sim = 0.6·0.9 + 0.8·0.1 = 0.54 + 0.08 = <span class="hl">0.62</span><br>
       <span class="dim"># b is more similar to a than c → shown first (if still available)</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the stale index.</b> Embeddings are computed in batches and the ANN index is rebuilt on a schedule. Between rebuilds, listings sell, get delisted, or change price — but their old vectors still sit in the index and still come back as top neighbours. You then recommend homes that are gone, quietly hurting trust and click-through. Nothing errors; the results just point at ghosts. You catch it by measuring the fraction of recommended items that are still valid, and by re-indexing (or invalidating) fast enough for your churn rate.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — index freshness vs cost.</b> Rebuilding the ANN index more often (or streaming updates) keeps recommendations current but costs compute and complexity; rebuilding rarely is cheap but serves stale items. High-churn catalogues (rentals, event tickets, marketplace listings) need near-real-time updates or a fast validity filter at serve time; slow-churn ones can rebuild nightly. Match freshness to how fast your items change.</div></div>
     <div class="callout c-info"><span class="ic">↗</span><div><b>Go deeper:</b> the full design — embedding features, ANN choice, freshness pipeline, and evaluation — is in <a href="../../ML%20Design/09-similar-listing/README.md">ML Design · 09 Similar Listing</a>.</div></div>''',
     "gotit":"Got similarity + freshness"},
    {"title":"Build the similar-listings system","body":
     '''<p>Here it is built up: item embeddings, the ANN index, the current-item query, nearest neighbours, and the freshness/validity filter. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a similarity index","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_similar_listings.py</code>. Embed toy listings (features → vectors), build an ANN index, and for a query item return its nearest neighbours by dot product. Then simulate churn: delist some items and show how a stale index returns sold listings until you re-index or add a validity filter.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 7 artifact.
Create experiments/foundations/m18_similar_listings.py: embed toy listings, build an ANN index (or brute-force top-k), return nearest neighbours by dot product for a query item. Simulate churn by delisting items and show the stale index returns sold listings until re-index; add a serve-time validity filter and measure the fraction of valid recommendations before vs after.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-07-log.md</code>: what fraction of recommendations were stale before the fix, and what did re-indexing / a validity filter recover?</div></div>''',
     "gotit":"Done — on to Day 8"},
  ],
  demos={
    "embed":{"btn":"① embed the current item","html":'''<span class="prompt">&gt;&gt;&gt;</span> e_a = item_tower(current_listing)
<span class="dim"># from photos, text, price, location</span>
<span class="hl">e_a = [0.6, 0.8]</span>   <span class="dim"># the query is the item itself</span>''',
      "take":"<b>①  The item is the query.</b> No user model — embed the listing you're viewing, then look for its neighbours."},
    "ann":{"btn":"② ANN nearest neighbours","html":'''<span class="prompt">&gt;&gt;&gt;</span> sim(a,b) = 0.6*0.5 + 0.8*0.8 = 0.94
<span class="prompt">&gt;&gt;&gt;</span> sim(a,c) = 0.6*0.9 + 0.8*0.1 = 0.62
<span class="hl">b (0.94) before c (0.62)</span>   <span class="dim"># ANN returns top-k fast</span>''',
      "take":"<b>②  Similar = nearest by dot product.</b> The ANN index returns the closest listings without scanning the whole catalogue — Day 1's retrieve stage, item-to-item."},
    "fresh":{"btn":"③ filter stale items","html":'''<span class="prompt">&gt;&gt;&gt;</span> neighbours = [b, c, d]
<span class="prompt">&gt;&gt;&gt;</span> d.status == "SOLD"   <span class="bad"># stale vector still in index</span>
<span class="prompt">&gt;&gt;&gt;</span> show = [n for n in neighbours if n.available]
<span class="hl">drop d → recommend only live listings</span>''',
      "take":"<b>③  Keep it fresh.</b> Between re-indexes, sold items linger in the ANN index. A validity filter (or fast re-index) stops you recommending ghosts."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='80' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='80' y='42' fill='#1a5c38'>e₁</text><rect x='140' y='24' width='80' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='180' y='42' fill='#1a5c38'>e₂</text><rect x='240' y='24' width='80' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='280' y='42' fill='#1a5c38'>e₃</text><text x='400' y='42' fill='#6B645E'>every item → a vector</text></g></svg>",
     "note":"<b>Embed every item.</b> Photos, text, price, and location become a vector per listing. Similarity will be distance in this space."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>ANN index of all item vectors</text></g></svg>",
     "note":"<b>Build the ANN index.</b> Store all item vectors in an index (HNSW/IVF) so nearest-neighbour lookups are fast over the whole catalogue."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>query = the current listing's vector</text></g></svg>",
     "note":"<b>The item is the query.</b> When a user views a listing, its own vector becomes the search query — no per-user model needed."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='260' cy='40' r='6' fill='#7C6DAA'/><text x='260' y='24' fill='#5E5191'>a</text><circle cx='300' cy='38' r='5' fill='#2D8B55'/><text x='320' y='36' fill='#1a5c38'>b 0.94</text><circle cx='210' cy='52' r='5' fill='#2A7B9B'/><text x='175' y='55' fill='#1F6280'>c 0.62</text></g></svg>",
     "note":"<b>Nearest neighbours.</b> ANN returns the closest vectors by dot product — b (0.94) before c (0.62). These are the 'more like this' items."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='120' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='100' y='43' fill='#C93B3B'>d = SOLD</text><text x='185' y='43' fill='#6B645E'>filter out →</text><rect x='300' y='24' width='180' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='390' y='43' fill='#1a5c38'>only live listings</text></g></svg>",
     "note":"<b>Freshness filter.</b> Drop items that sold or expired since the last re-index. Track the fraction of valid recommendations and re-index fast enough for your churn."},
  ],
  quiz=[
    {"q":"1. In similar-listings recommendation, what plays the role of the query?","opts":["the user's full profile","the current item (listing) the user is viewing — its own embedding","a random item","the ad auction"],"ans":1,"fb":"Item-to-item recommendation needs no user model: the listing being viewed is the query, and you retrieve its nearest neighbours in embedding space."},
    {"q":"2. How is 'similar' computed?","opts":["by alphabetical order","as nearest neighbours in embedding space (dot product / cosine), found with an ANN index","by price only","randomly"],"ans":1,"fb":"Each item is embedded; similarity is the dot product (or cosine) of embeddings, and an ANN index returns the closest items fast — Day 1's retrieve stage applied item-to-item."},
    {"q":"3. Current item e_a=[1,0]. Candidate e_b=[0.8,0.6], e_c=[0,1]. By dot product, which is more similar to a?","opts":["e_c","e_b — sim=1·0.8+0·0.6=0.8, versus e_c sim=1·0+0·1=0","they tie","neither"],"ans":1,"fb":"sim(a,b)=1·0.8+0·0.6=0.8; sim(a,c)=1·0+0·1=0. b is far more aligned with a, so b is the more similar listing."},
    {"q":"4. Users keep seeing recommended homes that are already sold. The core cause is:","opts":["the embeddings are wrong","a stale ANN index — sold items' old vectors linger between re-indexes and still come back as neighbours","the dot product is broken","too few candidates"],"ans":1,"fb":"Between scheduled re-indexes, delisted items stay in the index and get returned. Fix with faster re-indexing / streaming updates or a serve-time validity filter, and monitor the valid-recommendation rate."},
  ],
  fin={"em":"🏠","h3":"Day 7 complete — pure item-to-item retrieval!",
       "p":"You built similar-listings the simplest way: embed every item, treat the current listing as the query, and return nearest neighbours by ANN — Days 1 and 3 with no user model. The one hard extra was <b>index freshness</b>: a stale index recommends items that are gone, so match re-index speed to your churn. Follow the Go-deeper link for the full design. Next: <b>Day 8</b> — the personalized news feed, where many objectives collide."},
))
print("M18: Day 7 built")

# ==================== Day 8 — Personalized News Feed (case study) ====================
L("day-08-news-feed.html", dict(
  qid="m18-d08-newsfeed", title="Module 18 · Day 8 — Personalized News Feed", nav_title="Spiral · M18 Day 8",
  eyebrow="Module 18 · Serve · Day 8 · Case Study", h1="Personalized News Feed",
  lead="A news feed is where recommendation gets genuinely hard: you cannot optimise one number. You must balance <b>engagement</b>, <b>freshness</b>, and <b>diversity</b> at once, correct for <b>position bias</b> in your own logs, and keep watching for <b>drift</b> as the world changes hourly. It is the capstone of everything in this module.",
  goal="<b>🎯 By the end, you'll be able to:</b> combine multiple objectives into one ranking, explain position bias and how to correct it, say why diversity and freshness fight engagement, and name the drift a live feed must monitor.",
  prev_href="day-07-similar-listings.html", prev_label="Similar Listings", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"One list, many goals","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> all of Days 1–7 — retrieve-then-rank, metrics, feedback loops, freshness. This lesson combines them.</div></div>
     <h4>Why one number isn't enough</h4>
     <p>Rank purely by predicted clicks and you get a feed of clickbait and stale-but-popular posts. A healthy feed juggles several goals at once:</p>
     <ul>
       <li><b>Engagement</b> — will the user read, like, or comment?</li>
       <li><span class="term" data-tip="How recent an item is. News loses value fast, so a feed must favour recent posts even if older ones have more accumulated engagement.">Freshness</span> — is it recent? Yesterday's news is worth little.</li>
       <li><span class="term" data-tip="Showing a variety of topics and sources rather than many near-identical items, so the feed doesn't become repetitive or an echo chamber.">Diversity</span> — is the feed varied, not ten near-identical posts?</li>
     </ul>
     <p>This is <span class="term" data-tip="Ranking that balances several goals at once (engagement, freshness, diversity) instead of a single score, usually by combining objectives with weights.">multi-objective ranking</span>: combine the goals, usually as a weighted blend, and tune the weights to the product's values.</p>
     <h4>Your logs are biased</h4>
     <p>Items shown at the top of the feed get more clicks just because they are on top — that is <span class="term" data-tip="The tendency for items shown higher in a list to get more clicks regardless of relevance, because users see them first. It contaminates click logs used for training.">position bias</span>. If you train on raw click logs, the model learns 'top position = good' instead of 'relevant = good.' You must correct for it.</p>''',
     "gotit":"Got the multi-goal setup"},
    {"title":"A newspaper editor","body":
     '''<div class="relate">
       <div class="card"><span class="big">📰</span><h5>The editor's judgment</h5><p>A good editor doesn't just print whatever got the most clicks yesterday. They balance a big story (engagement), today's news (freshness), and a mix of topics (diversity) — so the front page is popular <i>and</i> worth reading.</p></div>
       <div class="card"><span class="big">👀</span><h5>The top of the page wins</h5><p>Whatever the editor puts at the top gets read most — not because it's best, but because it's first. If they judged stories only by 'what got read,' they'd keep promoting whatever happened to be on top. They must account for that.</p></div>
     </div>
     <p><strong>In one line:</strong> blend engagement, freshness, and diversity into one ranking, and correct your click logs for position bias so 'shown first' isn't mistaken for 'good.'</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: an editor sees the whole world and uses judgment. The model only sees what it already showed, filtered through biased clicks — so without care it amplifies its own past choices instead of reflecting the world.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Rank one feed","body":
     '''<p>Let us walk one feed: a multi-objective score, position-bias correction, and a diversity re-rank. <strong>Click all three.</strong></p>'''},
    {"title":"Blending objectives and fixing bias","body":
     '''<h4>Step 1 — combine goals, in words</h4>
     <p>Give each goal a weight and add them up into a single score. The weights encode what the product values — more freshness for news, more diversity to avoid echo chambers.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       score = w₁·engagement + w₂·freshness + w₃·diversity<br>
       train on click / P(seen at position p)   <span class="dim"># inverse-propensity weighting fixes position bias</span>
     </div></div>
     <p>Symbols: <code>w₁, w₂, w₃</code> are the objective weights; dividing each training click by <code>P(seen at position p)</code> — <span class="term" data-tip="Inverse-propensity weighting: divide each observed click by the probability the item was seen at its position, so clicks from high positions count less and the position bias is removed.">inverse-propensity weighting</span> — removes the advantage top items got just from position.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.6">
       weights: w₁=0.6 (engage), w₂=0.3 (fresh), w₃=0.1 (diverse)<br><br>
       Post P: engage 0.8, fresh 0.2, diverse 0.5 → 0.6·0.8+0.3·0.2+0.1·0.5 = <span class="hl">0.59</span><br>
       Post Q: engage 0.5, fresh 0.9, diverse 0.7 → 0.6·0.5+0.3·0.9+0.1·0.7 = <span class="hl">0.64</span><br>
       <span class="dim"># Q wins — fresher and more diverse outweighs P's higher raw engagement</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — position-bias feedback loop.</b> If you train on raw clicks without correcting for position, the model learns that top items are 'good' — so it keeps ranking them top, they keep getting clicks, and the belief self-reinforces. New or lower-ranked items never get a fair chance, the feed calcifies, and offline click metrics look great because you're grading the model on the biased world it created. You catch it only by inverse-propensity weighting (or randomized position experiments) and by monitoring exposure of lower-ranked items.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — engagement vs diversity/freshness (and short vs long term).</b> Cranking the engagement weight lifts clicks today but drifts toward clickbait, repetition, and echo chambers, quietly hurting long-term retention and trust. Adding freshness and diversity weight costs some immediate engagement to keep the feed healthy and varied. There is no single 'best' setting — it is a product-values decision, and it must be re-tuned as content and behaviour <b>drift</b> over time (breaking news, seasonal shifts, trends). A live feed monitors drift continuously and re-trains.</div></div>
     <div class="callout c-info"><span class="ic">↗</span><div><b>Go deeper:</b> the full design — objectives, feature and serving stack, position-bias handling, and drift monitoring — is in <a href="../../ML%20Design/10-personalized-news-feed/README.md">ML Design · 10 Personalized News Feed</a>.</div></div>''',
     "gotit":"Got objectives + bias"},
    {"title":"Build the news feed","body":
     '''<p>Here it is built up: candidate posts, per-objective scores, the weighted blend, position-bias correction, and a diversity re-rank with drift monitoring. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a multi-objective feed","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m18_news_feed.py</code>. Score toy posts on engagement, freshness, diversity; combine with weights and sweep them to see the feed change. Add position-bias: train on raw clicks vs inverse-propensity-weighted clicks and show the difference in what gets promoted. Add a diversity re-rank (e.g. MMR).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 18 Day 8 artifact.
Create experiments/foundations/m18_news_feed.py: score toy posts on engagement/freshness/diversity, combine with weights (sweep them), and add a diversity re-rank (MMR). Simulate position bias: train on raw clicks vs inverse-propensity-weighted clicks and show how the promoted items differ. Print how the feed changes as weights shift.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m18/day-08-log.md</code>: how did the top of the feed change when you raised the freshness weight, and did IPW correction change what got promoted?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "blend":{"btn":"① blend objectives","html":'''<span class="prompt">&gt;&gt;&gt;</span> w = [0.6, 0.3, 0.1]   <span class="dim"># engage, fresh, diverse</span>
<span class="prompt">&gt;&gt;&gt;</span> P: 0.6*0.8 + 0.3*0.2 + 0.1*0.5 = 0.59
<span class="prompt">&gt;&gt;&gt;</span> Q: 0.6*0.5 + 0.3*0.9 + 0.1*0.7 = 0.64
<span class="hl">Q ranks above P</span>''',
      "take":"<b>①  One score from many goals.</b> A weighted blend lets a fresher, more diverse post (Q) beat a higher-engagement one (P). The weights encode product values."},
    "bias":{"btn":"② correct position bias","html":'''<span class="prompt">&gt;&gt;&gt;</span> raw click at rank 1 counts as 1.0
<span class="prompt">&gt;&gt;&gt;</span> P(seen | rank 1) = 0.9, P(seen | rank 8) = 0.2
<span class="prompt">&gt;&gt;&gt;</span> weighted = click / P(seen)
<span class="hl">a rank-8 click is worth 1/0.2 = 5× a rank-1 click</span>''',
      "take":"<b>②  Un-bias the logs.</b> Inverse-propensity weighting divides each click by the chance the item was seen, so 'shown first' isn't mistaken for 'good.'"},
    "diverse":{"btn":"③ diversity re-rank","html":'''<span class="prompt">&gt;&gt;&gt;</span> top by score: [tech, tech, tech, sports]
<span class="prompt">&gt;&gt;&gt;</span> re-rank for variety (MMR)
<span class="hl">[tech, sports, tech, world]</span>   <span class="dim"># less repetition</span>''',
      "take":"<b>③  Spread the topics.</b> A diversity re-rank breaks up near-identical posts so the feed isn't ten of the same thing — trading a little score for variety."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='35' fill='#1F6280'>candidate posts</text></g></svg>",
     "note":"<b>Candidate posts.</b> Retrieval shortlists posts the user might want — the retrieve stage, now for a news feed."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='120' height='26' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='100' y='35' fill='#5E5191'>engagement</text><rect x='200' y='18' width='120' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='260' y='35' fill='#1a5c38'>freshness</text><rect x='360' y='18' width='120' height='26' rx='4' fill='#FCF3DC' stroke='#C99A12'/><text x='420' y='35' fill='#9A7208'>diversity</text></g></svg>",
     "note":"<b>Score each objective.</b> Predict engagement, measure freshness, and estimate diversity contribution — three separate signals per post."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>score = w₁·engage + w₂·fresh + w₃·diverse</text></g></svg>",
     "note":"<b>Blend with weights.</b> Combine the objectives into one score. The weights are a product decision — more freshness for news, more diversity against echo chambers."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='16' width='300' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>train on click / P(seen at position)</text></g></svg>",
     "note":"<b>Correct position bias.</b> Divide each click by the chance the item was seen at its position, so the model learns relevance, not 'it was on top.'"},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='150' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='115' y='43' fill='#1a5c38'>diversity re-rank</text><text x='210' y='43' fill='#6B645E'>+</text><rect x='240' y='24' width='240' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='360' y='43' fill='#C93B3B'>monitor drift continuously</text></g></svg>",
     "note":"<b>Re-rank and watch drift.</b> A final diversity pass breaks up repetition; then monitor drift (breaking news, trends, behaviour shifts) and re-train — the feed's world changes hourly."},
  ],
  quiz=[
    {"q":"1. Why can't a news feed rank by predicted clicks alone?","opts":["clicks are illegal","it drifts to clickbait and stale-but-popular posts; a healthy feed must also weigh freshness and diversity","clicks are too slow","freshness is free"],"ans":1,"fb":"A single engagement objective ignores whether content is recent or varied, producing clickbait and repetition. Multi-objective ranking blends engagement, freshness, and diversity."},
    {"q":"2. What is position bias?","opts":["a hardware fault","items shown higher get more clicks just for being on top, so raw click logs confuse 'shown first' with 'relevant'","a type of embedding","the freshness decay"],"ans":1,"fb":"Users click top items more regardless of relevance. Training on raw clicks teaches 'top = good.' Inverse-propensity weighting (or position randomization) corrects it."},
    {"q":"3. Weights w=[0.6,0.3,0.1] for [engage,fresh,diverse]. Post A: [0.9,0.1,0.0] → 0.57. Post B: [0.4,0.9,0.8] → ?. Who ranks higher?","opts":["A, higher engagement","B — 0.6·0.4+0.3·0.9+0.1·0.8 = 0.24+0.27+0.08 = 0.59 > 0.57","they tie","neither"],"ans":1,"fb":"B = 0.6·0.4 + 0.3·0.9 + 0.1·0.8 = 0.24 + 0.27 + 0.08 = 0.59, just above A's 0.57. Freshness and diversity tip it despite A's higher engagement."},
    {"q":"4. You raise the engagement weight and short-term clicks jump, but weeks later retention falls and the feed feels repetitive. This is:","opts":["a rendering bug","the engagement-vs-diversity/freshness trade-off plus drift — over-weighting engagement drifts to clickbait and echo chambers, hurting long-term health","a calibration error","position bias only"],"ans":1,"fb":"Over-weighting one objective optimizes the short term at the cost of variety and retention. Balance the weights, monitor long-term metrics, and re-tune as behaviour drifts."},
  ],
  fin={"em":"📰","h3":"Day 8 complete — the recommendation capstone!",
       "p":"You ranked a news feed the hard way: a <b>multi-objective</b> blend of engagement, freshness, and diversity; <b>position-bias</b> correction so 'shown first' isn't 'good'; a diversity re-rank; and continuous <b>drift</b> monitoring. This pulls together every idea in M18. Follow the Go-deeper link for the full design. Next: the <b>review gate</b> — prove the whole module stuck."},
))
print("M18: Day 8 built")

# ==================== Review Gate ====================
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m18-review", title="Module 18 · Review Gate", nav_title="Spiral · M18 Review",
  eyebrow="Module 18 · The Gate — Ranking & Recommendation", h1="Module 18 — Review Gate",
  lead="Checkpoint. The gate question: <b>can I design a web-scale recommender — retrieve-then-rank, honest metrics, and the failure modes that quietly break it?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Every case study in this module — and every serving system after it — assumes retrieve-then-rank, calibrated metrics, and drift awareness are solid.",
  prev_href="day-08-news-feed.html", prev_label="Day 8 · Personalized News Feed",
  checks=[
    ["Day 1","I can explain retrieve-then-rank and the two-tower model, score a match by dot product, and say why hard negatives matter."],
    ["Day 2","I can separate ranking metrics (AUC, NDCG, recall@k) from calibration, and say why an ad auction needs calibrated pCTR."],
    ["Day 3","I can read the InfoNCE loss and place cross-encoder rerank, ColBERT, and Matryoshka on the speed/accuracy line, and explain the recall ceiling."],
    ["Day 4","I can lay out a video recommender (candidate generation + watch-time ranking) with offline NDCG + online A/B, and name the feedback-loop drift."],
    ["Day 5","I can frame ad CTR as calibrated ranking, walk a bid × pCTR auction, and say why great AUC with inflated pCTR still bleeds revenue."],
    ["Day 6","I can handle event recommendation: time-bound items, cold-start, geo/time features, and a freshness decay."],
    ["Day 7","I can build similar-listings as item-to-item ANN retrieval and explain why index freshness matters for a churning catalogue."],
    ["Day 8","I can combine engagement + freshness + diversity, correct position bias, and name the drift a live feed must monitor."],
  ],
  quiz=[
    {"q":"1. The retrieve-then-rank pattern exists because:","opts":["ranking is optional","scoring millions of items with the heavy model is too slow, so a cheap retrieval stage shortlists candidates first","embeddings are illegal","auctions require it"],"ans":1,"fb":"Cheap retrieval narrows millions to a few hundred; the expensive ranker orders only those — the core idea behind every system in M18 (Day 1)."},
    {"q":"2. A CTR model has AUC 0.94 but every predicted probability is 1.5× too high. In an ad auction this:","opts":["is fine, AUC is high","misprices ads because the auction spends bid × pCTR — calibration, not AUC, decides the money","improves revenue","only affects retrieval"],"ans":1,"fb":"AUC sees only order. The auction uses the probability value, so inflated pCTR mis-prices and mis-allocates — you need calibration monitoring (Days 2, 5)."},
    {"q":"3. Query q=[2,1,0], item v=[1,3,4]. The dot-product retrieval score is:","opts":["5","9","6","2"],"ans":0,"fb":"q·v = 2·1 + 1·3 + 0·4 = 2 + 3 + 0 = 5. Multiply matching entries and add (Day 1)."},
    {"q":"4. You rerank only the top-10 candidates with a great cross-encoder, but overall recall stays low. Why?","opts":["the cross-encoder is broken","the reranker can only reorder the shortlist — a relevant item outside the top-10 can't be recovered, so retrieval caps recall","calibration is off","the temperature is wrong"],"ans":1,"fb":"Recall is set by the retrieval stage. No reranker rescues an item that never made the shortlist — widen the candidate set (Day 3)."},
    {"q":"5. A feed trained on raw click logs keeps promoting whatever is already on top. The core cause is:","opts":["too little compute","position bias uncorrected — top items get clicks for being on top, so the model learns 'top = good' and self-reinforces","the dot product","cold-start"],"ans":1,"fb":"Uncorrected position bias makes clicks reflect position, not relevance. Fix with inverse-propensity weighting or position randomization (Day 8)."},
  ],
  verdict_pass="you can design a web-scale recommender end to end: retrieve-then-rank with two towers, honest ranking + calibration metrics, the retrieval toolbox (InfoNCE, rerank, ColBERT, Matryoshka), and the five case studies — plus the silent failures (hard negatives, miscalibration, recall ceiling, feedback loops, cold-start, stale index, position bias) that separate a demo from production.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The chain — <code>two-tower</code> → <code>metrics/calibration</code> → <code>retrieval architecture</code> → <code>video</code> → <code>ads</code> → <code>events</code> → <code>similar listings</code> → <code>news feed</code> — should feel connected before you move on.",
  complete_label="Mark Module 18 complete",
  fin={"em":"🏆","h3":"Module 18 — passed!",
       "p":"You built the whole ranking-and-recommendation stack: retrieve-then-rank with two-tower models, calibrated ranking metrics, the modern retrieval toolbox, and five real case studies from video to news feeds — with a sharp eye for the failure modes that quietly break recommenders at scale."},
  score_target=4,
))
print("M18: review built — 8 lessons + review complete")
