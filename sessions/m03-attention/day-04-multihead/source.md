---
quest_id: wf4-d04-multihead
donor: m03-day-04.donor
mode: reference
source_mode: clean
notebook_yardstick: null   # Reference mode (Notebook Smoothness Gate = N/A)
spine: "meeting: who to listen to"
module_label: "Module 4 · Represent · Day 4"
title: "Multi-Head Attention"
subtitle: "Many Views at Once"
nav_prev_href: "../day-03-attention-scores/lesson.html"
nav_next_href: "../day-05-positional/lesson.html"
fin_title: "Module 4 · Day 4 complete! 🏆"
fin_body: "Nice work — you've completed <b>Multi-Head Attention</b>.<br>Next up: <b>Positional Encoding</b>."
references:
  - "Vaswani et al. 2017, 'Attention Is All You Need' (arXiv:1706.03762) — multi-head attention"
---

@@@ region name=title
<title>Module 4 · Day 4 — Multi-Head Attention</title>

@@@ region name=brand_sub
<div class="brand-sub">Foundations · M4 Day 4</div>

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
<a class="lnav prev" href="../day-03-attention-scores/lesson.html"><span class="d">← Prev</span><span class="t">Attention Scores &amp; Softmax</span></a>

@@@ region name=nav_next
<a class="lnav next" href="../day-05-positional/lesson.html"><span class="d">Next →</span><span class="t">Positional Encoding &amp; RoPE</span></a>

@@@ hero
@lede
You just built one attention (Day 3) — but here's the catch: a single conversation in the meeting can only listen for *one* kind of relationship at a time. And a real sentence has many at once. Who did what to whom? Which adjective belongs to which noun? What does that pronoun point back to? Asking one conversation to catch all of it is too much. So a transformer runs **several meetings in parallel** — each one a "head" free to focus on something different — and then merges their notes into one answer. That's multi-head attention, and it's why every real transformer has not one attention but many.
@goal
By the time you close this tab, you'll be able to explain why transformers use multiple heads, and describe the split → attend → concatenate → project pipeline.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="Several attentions, side by side" gotit="Got why many heads — let's go"
!!! c-info 🧭
**What this assumes:** Day 3 (one full attention: score → softmax → blend). **That's the only prerequisite** — a head is just one Day-3 attention.
!!!

#### First, the picture — hold this before any words
Here is the picture: instead of the whole meeting having one conversation, split the room into a few smaller **breakout meetings** running at the same time. Each breakout listens for its own thing — one tracks grammar, one tracks long-range references, one links describing words to what they describe. When they're done, everyone comes back and their notes are stitched into a single summary. Each breakout is a "head"; the stitch-and-summarize at the end is the merge. Hold that, and the vocabulary below is just names for the pieces.

#### The words you'll meet today — plain English first
Glance down the list first.

%%% jargon
head | One complete attention (the whole Day-3 formula) with its own Q/K/V projections. A breakout meeting.
multi-head attention | Running several heads in parallel, then merging their outputs into one vector.
d_model | The full width of a word's vector (the model size).
d_k | The width each head works in. Split the model width across heads: `d_k = d_model / h`.
h | The number of heads.
W_O | The learned output matrix that mixes the concatenated heads back into one `d_model` vector.
%%%

#### The limitation of one head, and the fix
A single head produces **one** attention pattern — one kind of relationship. Useful, but a sentence needs several at once. The fix: give each of `h` heads its own `W_Q, W_K, W_V`, each projecting into the smaller size [[d_k||The per-head width, d_model / h.]] `= d_model / h`. Every head runs the full Day-3 attention on its own slice. Then **concatenate** the `h` head outputs back to length `d_model` and mix them with one learned matrix [[W_O||The learned output projection that blends the heads back into a d_model vector.]].

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 听起来 h 个头应该是 h 倍计算量，其实差不多 — people expect "more heads = more compute," but total cost is roughly flat. Splitting `d_model` across heads makes each head work in the smaller size `d_k = d_model / h`, so you're *re-slicing the same budget*, not adding to it.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="A panel of specialists" gotit="Got the picture"
#### The breakout meetings are specialists
Back to the meeting: the breakouts aren't random — each becomes a specialist. Picture a panel reading the same sentence, each member with a different job.

%%% cards
🧑‍🔬 | Each head = a specialist | One head watches grammar, another tracks what a pronoun refers to across a long distance, another links adjectives to their nouns. Same sentence, different focus.
🖍️ | Many highlighters, one summary | Like reading a page with several colored highlighters, each marking a different pattern, then combining all the marks into one understanding.
%%%

**In one line:** split into several small attentions that each specialize, then merge their results into one vector.

**直觉 / Intuition:** 把注意力拆成几个小头，每个头专注一种关系，最后把它们的结果合并成一个向量。 No one hands out the specialties — the heads discover them on their own through training.

!!! c-warn ⚠️
Where the analogy breaks: nobody *assigns* each head a job — they specialize on their own through training (M3). And the heads run in parallel and cost about the same as one big attention, because each works in the smaller size `d_k`. A human panel would take longer with more members; heads don't.
!!!

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="Split, attend, concat, project — and why every model does it" gotit="Got multi-head"
#### The pipeline
1. **Split:** give each of `h` heads its own `W_Q, W_K, W_V`, projecting into size `d_k = d_model / h`.
2. **Attend:** each head runs the full scaled dot-product attention (Day 3) on its own Q, K, V.
3. **Concat:** stack the `h` head outputs side by side, back to length `d_model`.
4. **Project:** multiply by the learned matrix `W_O` to mix the heads into the final output.

One note on the interactive playground above: it uses deliberately tiny numbers so the shapes fit on screen, so *its* `W_O` also trims the width down to `d_model`. The idea is identical — concatenate the heads, then let `W_O` mix them into the model width — but in the standard design the concatenated width `h·d_k` already equals `d_model`, which makes `W_O` a square `(d_model, d_model)` matrix.

%%% mathladder
title: multi-head attention
words: split the model width into `h` smaller heads, let each run its own attention on a slice, stitch the results back together, then mix them with one matrix.
formula: `d_k = d_model / h`; `MultiHead(x) = Concat(head₁, …, head_h) · W_O`, where `headᵢ = Attention(x·W_Qⁱ, x·W_Kⁱ, x·W_Vⁱ)`. `d_model` = full width; `h` = number of heads; `d_k` = per-head width; `W_O` has shape `(d_model, d_model)`.
numbers: `d_model = 4`, `h = 2` → `d_k = 2`. Two heads each output a length-2 vector; concat → length 4; `· W_O` (4×4) → length 4. Same width in, same width out. (Real model: `d_model = 512`, `h = 8` → `d_k = 64`.)
sanity: `h` must divide `d_model`, and the concat width `h·d_k` must equal `d_model` before `W_O`. Push `h` too high (`d_model = 512`, `h = 64` → `d_k = 8`) and each head is too small to hold a useful subspace.
%%%

#### Why a frontier lab cares
Every real transformer is multi-head — small models use 8–16 heads, big ones 32–128 (Llama 3's largest runs 128). Heads visibly specialize: probing a trained model shows some heads tracking syntax, others copying earlier tokens. And multi-head sets up one of the biggest inference costs at scale: the Keys and Values for *every head* must be cached during generation, so a long context times many heads is a serious **production** memory bill — which is exactly why GQA/MQA (Grouped-Query / Multi-Query Attention) were invented to share K/V across groups of heads.

!!! c-info 🔗
**Remember this chain:** one attention = one head (Day 3) → `h` heads in parallel, each in `d_k = d_model/h` → concat → `W_O` mixes them → one `d_model` vector. Stack this block with the Module-2 feed-forward layers and you have a transformer. Next lesson: teach it *word order*.
!!!

#### The staff lens — one silent failure, one trade-off
Splitting into heads is a reshape — and reshapes are where silent bugs live.

!!! c-warn ⚠️
**Failure mode (silent): the wrong reshape/transpose when splitting heads.** Turning `(seq, d_model)` into `(h, seq, d_k)` takes a reshape *and* a transpose in the right order. Get the order wrong and the shapes can still line up — so nothing crashes — but each head now attends over scrambled slices of the vector. The model trains and the loss drops; the heads just quietly stop specializing. A senior engineer catches this in **code review** by printing the per-head shapes, not by waiting for a crash.
!!!

!!! c-info ⚖️
**Trade-off: more heads vs. wider heads.** Total compute is about the same however you slice it, because `d_k = d_model / h`. More heads means more independent relationship channels, but each is lower-resolution; fewer heads means richer per-head subspaces but fewer patterns at once. Push `h` too high and `d_k` shrinks until a head can't hold a useful subspace. That balance is a **design-review** call — and it's the same axis GQA / MQA trade along to shrink the KV cache at inference.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Multi-head attention runs <code>h</code> attentions in parallel, each with its own Q/K/V projections into a <code>d_k = d_model/h</code> subspace, then concatenates the heads and mixes them with an output matrix <code>W_O</code>. Different heads specialize on different relationships at roughly the same total cost as one big attention. The nuance is the head-count trade-off: too many heads shrinks <code>d_k</code> until each head is too low-dimensional to be useful — and it's the same axis GQA/MQA trade along to shrink the KV cache at inference."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict the shapes, then run two heads and merge" gotit="Done"
Here's the fun part. **Before you write any code, predict the shapes:** with input width `d_model = 4` and `h = 2` heads, what width does each head output, what width is the concatenation, and what width comes out after `W_O`? Write your three guesses. Then build two heads and watch the widths split and rejoin. Pick one path:

#### Option A · write it yourself
Create `sessions/m03-attention/day-04-multihead/experiment.py`. Reuse your Day-3 attention function. Run it twice with two different sets of `W_Q/W_K/W_V` (two heads) on the same input, `np.concatenate` the two outputs, then multiply by an output matrix `W_O`. Print each head's output shape, the concatenated shape, and the final shape (back to `d_model`). Run with `python3 sessions/m03-attention/day-04-multihead/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 4 Day 4 artifact.

Create sessions/m03-attention/day-04-multihead/experiment.py that, with a comment on each step:
1. Reuses a scaled dot-product attention function (softmax(Q@K.T/sqrt(d_k))@V).
2. Defines TWO heads, each with its own random Wq/Wk/Wv projecting an input x (seq=2, d_model=4) into d_k=2; runs attention in each head; prints each head's output shape (2,2).
3. Concatenates the two head outputs along the last axis (shape (2,4)).
4. Multiplies by an output matrix Wo (4,4) to get the final (2,4) output; prints shapes at each step and notes each head learned a different pattern.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Each head's output has width `d_k = d_model / h = 2` — the model width, split in two.
- The concatenated output has width `d_model = 4` again (before `W_O`). Then `W_O` returns the final shape to match the input — one vector per token.
- The thing worth noticing: two heads cost about the same as one big attention, because each worked in half the width — that "re-slice the budget" trick is why real models can afford 32–128 heads. Did your three shape guesses line up?

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-04-multihead/log.md`: (1) why more heads don't cost much more compute; (2) what `W_O` is for; (3) one thing you're still unsure about (why shrinking `d_k` too far hurts is fair game).
!!!
