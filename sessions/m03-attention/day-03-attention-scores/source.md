---
quest_id: wf4-d03-attention-scores
donor: m03-day-03.donor
mode: reference
source_mode: clean
notebook_yardstick: null   # Reference mode (Notebook Smoothness Gate = N/A)
spine: "meeting: who to listen to"
module_label: "Module 4 · Represent · Day 3"
title: "Attention Scores &amp; Softmax"
subtitle: "From Matches to a Weighted Blend"
nav_prev_href: "../day-02-qkv/lesson.html"
nav_next_href: "../day-04-multihead/lesson.html"
fin_title: "Module 4 · Day 3 complete! 🏆"
fin_body: "Nice work — you've completed <b>Attention Scores &amp; Softmax</b>.<br>Next up: <b>Multi-Head Attention</b>."
references:
  - "Vaswani et al. 2017, 'Attention Is All You Need' (arXiv:1706.03762) — scaled dot-product attention"
  - "Dao et al. 2022, 'FlashAttention' (arXiv:2205.14135) — the O(n²) score-grid cost"
---

@@@ region name=title
<title>Module 4 · Day 3 — Attention Scores &amp; Softmax</title>

@@@ region name=brand_sub
<div class="brand-sub">Foundations · M4 Day 3</div>

@@@ region name=sidebar_nav
<nav aria-label="Sections">
      <div class="nav-group-label">Module 04 · Represent</div>
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
<a class="lnav prev" href="../day-02-qkv/lesson.html"><span class="d">← Prev</span><span class="t">Query, Key, Value</span></a>

@@@ region name=nav_next
<a class="lnav next" href="../day-04-multihead/lesson.html"><span class="d">Next →</span><span class="t">Multi-Head Attention</span></a>

@@@ hero
@lede
This is the moment the meeting actually happens. Every word now has a Query (what it's looking for) and a Key (what it advertises) from yesterday — so each word compares its own question against everyone else's name-tag to see who's relevant. But a word only has so much attention to give, so it can't listen to everyone equally: it has to split a fixed budget across the room, leaning hard toward the words that matched and barely toward the rest. Then it blends together what those words offer. Score, share out a budget, blend — that is the entire attention formula, and by the end of this lesson you'll have written all three steps.
@goal
By the time you close this tab, you'll compute attention scores as `Q @ Kᵀ`, turn each row into weights with softmax, and produce the output as a weighted average of the Values.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="Score, share out, blend" gotit="Got the three steps — let's go"
!!! c-info 🧭
**What this assumes:** Day 2 (a word has a Query, a Key, and a Value), and the dot product / matmul from Module 1. **That's all** — this lesson wires them together.
!!!

#### First, the picture — hold this before any words
Here is the picture: one word in the meeting turns to face the room. It compares its **question** (Query) against every other word's **name-tag** (Key) and gives each a match score. Then — because it can't listen to everyone at full volume — it spreads a fixed **100% budget of attention** over the room, giving most of it to the words that matched best. Finally it **blends** together what those words offer (their Values), weighted by that budget. Score, share out the budget, blend. Every term below names one of those three moves.

#### The words you'll meet today — plain English first
Glance down the list so nothing feels new when it arrives.

%%% jargon
attention score | One match number: how well one word's Query lines up with another word's Key. It's just a dot product.
softmax | The step that turns a row of raw scores into positive **weights that add up to 1** — the attention budget. Bigger score → exponentially bigger share.
attention weights | The shared-out budget: one weight per word, all positive, summing to 100%.
weighted sum | Blend the Values by the weights — mostly the words you attended to, a little of the rest.
scaling (÷√d_k) | Shrinking the scores before softmax (divide by the square root of the key size) so softmax doesn't collapse to all-or-nothing.
context-aware vector | The output for a word after the blend: its new vector, now shaped by the words it listened to.
%%%

#### The three steps
- **Step 1 — scores.** `scores = Q @ Kᵀ`: dot every word's Query with every word's Key. Entry `(i, j)` says "how much should word `i` attend to word `j`." A [[matmul||Matrix multiply, from Module 1.]] gives the whole grid at once.
- **Step 2 — softmax.** [[softmax||Turns a row of scores into positive weights that sum to 1.]] turns each *row* of scores into an attention budget — positive weights that sum to 1.
- **Step 3 — weighted sum.** `output = weights @ V`: blend the Values by that budget. Each word walks away with a new, **context-aware** vector.

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 很多人把公式背下来，却说不出哪一步才是"注意力" — people memorize the formula but can't point to which step *is* the attention. In plain terms: "attention" isn't one number — it's a whole *row* of weights (one per other word) that must sum to 1. And the `÷√d_k` step looks arbitrary until you see softmax collapse to all-or-nothing without it.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="Splitting a budget of attention" gotit="Got the picture"
#### The meeting has a fixed budget
Back to the meeting: a word can't blast full attention at everyone — real attention is a limited resource, so it has to *divide* it. That single idea, "you have 100% to spend and must split it," is what softmax does for us.

%%% cards
💯 | Softmax = share out 100% | Each word has 100% of attention to give. Softmax turns raw match scores into percentages — more to the words it matches well, less to the rest, always adding to 100%.
🧪 | Output = a weighted mix | Blend the Values by those percentages. If "bank" gives 80% of its attention to "river," its new vector is mostly river's Value — so "bank" now means the riverside kind.
%%%

**In one line:** score every word, softmax the scores into a 100% budget, then mix the Values by that budget.

**直觉 / Intuition:** 每个词的注意力预算是 100%，softmax 把匹配分数变成百分比，再按这个预算把各词的 Value 混合起来。 The output is a weighted average of the Values, so a word's new representation literally *becomes* a blend of whoever it listened to.

!!! c-warn ⚠️
Where the analogy breaks: it's not a person consciously choosing how to spend attention — it's a smooth, differentiable formula, so the training loop (M3) can nudge it to get better. And the score grid is `seq × seq`, which is why long meetings (long sequences) get expensive — the cost we'll name in Section 4.
!!!

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="The formula, the scale, and the n² cost" gotit="Got the formula"
#### The formula

%%% formula
expr: Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
note: `Q · Kᵀ` = the (seq × seq) score grid; `√d_k` = the square root of the key dimension (the scaling constant); `softmax` runs across each **row** so it sums to 1; the final `· V` blends the Values into one context-aware vector per word.
%%%

Why divide by `√d_k`? Large dot products push softmax to *saturate* — it hands almost all the budget to a single word and near-zero to everyone else, which makes the gradient vanish and learning stall. Dividing by `√d_k` keeps the scores in a sane range so softmax stays smooth and trainable.

%%% mathladder
title: scaled dot-product attention
words: score every query-key pair, shrink the scores so softmax doesn't saturate, turn each row into a 100% budget, then blend the Values by that budget.
formula: `softmax(Q·Kᵀ / √d_k) · V` — `Q, K, V` have shape `(n, d_k)`; `Q·Kᵀ` is the `(n, n)` score grid; `√d_k` scales; `softmax` runs across the **last axis** so each row sums to 1; the final `·V` returns shape `(n, d_k)`.
numbers: one query, raw scores `[2, 6]`, `d_k = 4` → divide by `√4 = 2` → `[1, 3]`. `softmax([1, 3]) = [e¹, e³] / (e¹ + e³) ≈ [0.12, 0.88]`. Output = `0.12·V₀ + 0.88·V₁`.
sanity: each softmax row sums to 1 (`0.12 + 0.88 = 1.0`). If a row does **not** sum to 1, you softmaxed the wrong axis — the silent bug below.
%%%

#### Why a frontier lab cares
`Q @ Kᵀ` and the final `@ V` are the two big matmuls that dominate attention's FLOPs — exactly what **FlashAttention** (Dao et al., 2022) speeds up by never writing the full grid to memory. And the score grid is `seq × seq`, so the cost grows with the **square** of the sequence length: double the context and the grid quadruples. That `O(n²)` wall is the reason long context is hard at scale — models that hold a million-token context in production have to engineer around it, and a whole research area exists to soften this quadratic cost.

!!! c-info 🔗
**Remember this chain:** Q, K, V (Day 2) → scores `Q·Kᵀ` → scale by `√d_k` → softmax to a 100% budget → blend the Values → a context-aware vector per word. That one line, stacked and repeated, is a transformer. Next lesson: run several of these at once (multi-head).
!!!

#### The staff lens — one silent failure, one trade-off
The formula is short, which is why its worst bug hides in plain sight.

!!! c-warn ⚠️
**Failure mode (silent): softmax on the wrong axis.** Write `softmax(scores, axis=0)` (down the columns) or over the whole grid instead of `axis=-1` (across each row), and it **still** sums to 1 — so nothing crashes and the loss looks plausible. But each query no longer gets its own distribution over the keys, so the attention is quietly meaningless. This is the bug a senior engineer catches in **code review**, precisely because it never throws an error — you confirm it by asserting each *row* sums to 1.
!!!

!!! c-info ⚖️
**Trade-off: the seq×seq score grid.** Computing `Q @ Kᵀ` in full is exact and perfectly parallel — every token pair scored at once, ideal for a GPU. The price is **O(n²)** memory and compute: double the sequence length and the grid quadruples. That is the cost a staff engineer raises in a **design review** — it decides whether your model reads a whole book or just a paragraph, and it's exactly what FlashAttention and long-context research exist to soften.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Scaled dot-product attention scores every query against every key with <code>Q·Kᵀ</code>, divides by <code>√d_k</code> so the softmax doesn't saturate, normalizes each row into weights that sum to 1, then takes that weighted sum of the Values. The nuance is the <code>seq×seq</code> score grid makes it <code>O(n²)</code> in sequence length — that quadratic cost is why long context is hard and why FlashAttention and related work exist to avoid ever materializing the full grid."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict the weights, then run the whole formula" gotit="Done"
Here's the fun part. **Before you write any code, predict:** if a word's two raw scores are `[2, 6]` and `d_k = 4`, roughly what split will softmax give after scaling — a near-even 50/50, or something lopsided? Write your guess. Then implement the three steps and check whether each row of weights really sums to 1. Pick one path:

#### Option A · write it yourself
Create `sessions/m03-attention/day-03-attention-scores/experiment.py`. With small `Q, K, V`, compute `scores = Q @ K.T`, a row-wise `softmax`, and `output = weights @ V`. Print scores, weights (check each row sums to 1), and the output. Run with `python3 sessions/m03-attention/day-03-attention-scores/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 4 Day 3 artifact.

Create sessions/m03-attention/day-03-attention-scores/experiment.py that, with a comment on each step, implements scaled dot-product attention:
1. Defines softmax over the last axis (subtract the row max for numerical stability).
2. Uses small Q, K, V (shape (2,2)); computes scores = Q @ K.T; prints them.
3. Scales by 1/sqrt(d_k) and applies row-wise softmax to get weights; asserts each row sums to 1.
4. Computes output = weights @ V; prints scores, weights, and output, noting the output is a weighted blend of the Values.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- The score grid `Q @ Kᵀ` has shape `(seq, seq)` — one row per word, one column per word it could attend to.
- Each **row** of the softmax weights sums to `1.0` (softmax over `axis=-1`). Raw scores `[2, 6]` scaled to `[1, 3]` land near `[0.12, 0.88]` — lopsided, not 50/50. Did you call it?
- The output has the same shape as `V` — one context-aware vector per token, each a weighted blend of the words it listened to. That blend *is* attention.

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-03-attention-scores/log.md`: (1) which step turns raw matches into a 100% budget; (2) why we divide by √d_k; (3) one thing you're still unsure about (the O(n²) cost is fair game).
!!!
