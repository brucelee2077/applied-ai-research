---
quest_id: wf4-d05-position
donor: m03-day-05.donor
mode: reference
source_mode: clean
notebook_yardstick: null   # Reference mode (Notebook Smoothness Gate = N/A)
spine: "meeting: who to listen to"
module_label: "Module 4 · Represent · Day 5"
title: "Positional Encoding"
subtitle: "&amp; the Shuffle Problem"
nav_prev_href: "../day-04-multihead/lesson.html"
nav_next_href: "../review.html"
fin_title: "Module 4 · Day 5 complete! 🏆"
fin_body: "Nice work — you've completed <b>Positional Encoding</b>.<br>Next up: the <b>Module 4 Review Gate</b>."
references:
  - "Vaswani et al. 2017, 'Attention Is All You Need' (arXiv:1706.03762) — sinusoidal positional encoding"
  - "Su et al. 2021, 'RoFormer / RoPE' (arXiv:2104.09864); Press et al. 2021, 'ALiBi' (arXiv:2108.12409)"
---

@@@ region name=title
<title>Module 4 · Day 5 — Positional Encoding &amp; the Shuffle Problem</title>

@@@ region name=brand_sub
<div class="brand-sub">Foundations · M4 Day 5</div>

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
<a class="lnav prev" href="../day-04-multihead/lesson.html"><span class="d">← Prev</span><span class="t">Multi-Head Attention</span></a>

@@@ region name=nav_next
<a class="lnav next" href="../review.html"><span class="d">Next →</span><span class="t">Review Gate</span></a>

@@@ hero
@lede
Everything you built this week — the meeting where words decide who to listen to — has a hidden blind spot: it doesn't know word *order*. Shuffle the words and attention gives the exact same answer. So to the meeting, "dog bites man" and "man bites dog" look identical — which is a disaster, because they mean opposite things. Why does this happen? Because attention only computes weighted sums, and a sum doesn't care what order you add things in. **Positional encoding** fixes it by stamping each word with a signal that says where it sits — a seat number — so order finally carries meaning.
@goal
By the time you close this tab, you'll be able to explain why plain self-attention is permutation-invariant, and describe how a sinusoidal position signal added to each embedding restores word order.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="The shuffle problem, and the fix" gotit="Got the shuffle problem — let's go"
!!! c-info 🧭
**What this assumes:** Days 1–4 (embeddings, and attention as score → softmax → blend). **That's all** — this lesson fixes one blind spot in what you already built.
!!!

#### First, the picture — hold this before any words
Here is the picture: the meeting is a room full of people talking, but nobody has a **seat number**. You can tell *who* is in the room and *how* they relate, but not *where* each one sits — so if two people swap seats, nothing about the conversation changes. That's the bug: plain attention knows the words and their relationships, but not their order. The fix is to hand every word a seat number before the meeting starts. Every term below names a piece of that fix.

#### The words you'll meet today — plain English first
Glance down the list first.

%%% jargon
permutation-invariant | A fancy way to say "order doesn't matter": shuffle the inputs and the output is unchanged. Plain self-attention has this property — which here is a bug.
positional encoding | The seat number: a vector added to each word's embedding that says where the word sits.
sinusoidal encoding | The classic recipe for those seat numbers, built from sine and cosine waves at many different speeds.
position signal / fingerprint | The unique pattern each position gets, so no two positions look alike.
frequency | How fast a wave wiggles. Mixing fast and slow waves is what makes each position's fingerprint unique.
%%%

#### The blind spot, the fix, and how the signal is made
- **The blind spot.** Plain self-attention is [[permutation-invariant||Shuffle the input words and the output is unchanged.]] — it only looks at *what* each word is and *how* words relate, never *where* they sit. Shuffles pass straight through.
- **The fix.** Add a [[positional encoding||A per-position vector added to each embedding so word order carries into the model.]] — one vector per position — to each embedding *before* attention. Now the same word carries different numbers at different spots.
- **How the signal is made.** With [[sinusoidal encoding||Sine and cosine waves at many frequencies; each position gets a unique fingerprint.]]: sine and cosine waves at many frequencies, so each position gets its own fingerprint of wave values.

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 第一反应通常是"模型怎么可能不知道顺序？" — the first reaction is "how could a model *not* know order?" In plain terms: plain attention only computes weighted sums, and a sum doesn't care what order you add things in. Order isn't built in — it has to be *injected* as an extra signal.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="Numbered seats in a theater" gotit="Got the picture"
#### Give the meeting numbered seats
Back to the meeting without seat numbers. Here are two everyday pictures that make the fix click.

%%% cards
🎭 | Numbered seats in a theater | Without numbers you know *who* is in the room but not *where*. Print a seat number on each chair and every person now has a definite spot. The seat number is the position signal.
🕐 | Like a clock's hands | An hour hand and a minute hand turn at different speeds; together they name one exact time. Sine and cosine waves at different speeds work the same way — together they name one exact position.
%%%

**In one line:** without seat numbers you can't read order; positional encoding is the seat number added to each word.

**直觉 / Intuition:** 没有座位号就读不出顺序；位置编码就是加到每个词上的"座位号"。 Waves at many speeds give each position a unique fingerprint, the way an hour hand plus a minute hand pin down one exact time.

!!! c-warn ⚠️
Where the analogy breaks: a seat number is a single integer, but a position signal is a whole *list* of wave values (the same length as the embedding). And a person keeps their identity when they change seats — but a word's numbers actually *change* once you add the position signal, because it's added right into the embedding.
!!!

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="The wave formula, in three small steps — and why order matters" gotit="Got positional encoding"
#### Step 1 — in plain words
For a word at position `pos`, build a list the same length as the embedding. Even slots (0, 2, 4, …) get a **sine**; odd slots (1, 3, 5, …) get a **cosine**. Slots near the front use fast waves (short wavelength); slots near the back use slow waves (long wavelength). Mix enough speeds and every position gets a fingerprint no other position shares.

#### Step 2 — the formula

%%% formula
expr: PE(pos, 2i) = sin( pos / 10000^(2i / d_model) )   ·   PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
note: `pos` = which word (0, 1, 2, …); `i` = which pair of slots; `d_model` = embedding width; `10000` = the base that spreads the wave speeds from fast (front slots) to slow (back slots).
%%%

#### Step 3 — a worked example
Take `d_model = 4`, so there are two slot-pairs with divisors `10000^0 = 1` and `10000^(2/4) = 100`.

%%% mathladder
title: sinusoidal positional encoding
words: even slots get a sine, odd slots get a cosine; fast waves near the front, slow near the back — Steps 1–2 above.
formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(…)` — `pos` = position, `i` = slot pair, `d_model` = width, `10000` = frequency base.
numbers: position 0 → `[sin 0, cos 0, sin 0, cos 0] = [0, 1, 0, 1]`. position 1 → `[sin 1, cos 1, sin 0.01, cos 0.01] ≈ [0.84, 0.54, 0.01, 1.00]`. Add this row to the word's embedding — same size in, same size out.
sanity: every row must be *unique* — no two positions share a fingerprint — and adding `PE` keeps the vector the same length as the embedding. If two positions collide, your base or `d_model` is off.
%%%

#### Modern alternatives (just be aware)
Sinusoidal is the original. Modern LLMs often swap it for **RoPE** (rotate Q and K by a position-dependent angle) or **ALiBi** (a distance penalty added to the attention scores). Both encode *relative* distance and extrapolate better to longer inputs — same goal, different mechanics.

#### Why a frontier lab cares
Position is not a nicety — without it a transformer literally cannot tell "dog bites man" from "man bites dog," so *every* transformer injects it. And the *scheme* is a live research axis at scale: sinusoidal is free and length-agnostic but absolute; learned absolute positions have no defined values past the longest training length, so they degrade on longer inputs; RoPE/ALiBi encode relative distance and extrapolate to the long contexts that long-context models ship in production. Picking one is a real **design-review** decision for any frontier model that wants to read a whole book.

!!! c-info 🔗
**Remember this chain:** embeddings (Day 1) → Q/K/V (Day 2) → attention (Day 3) → many heads (Day 4) → **+ position** (today). That's the full input side of a transformer. From here the Module-2 layers stack on top. You've built the representation the whole model runs on.
!!!

#### The staff lens — one silent failure, one trade-off
Forgetting position is the classic silent bug — the model still trains, it just quietly ignores order.

!!! c-warn ⚠️
**Failure mode (silent): no position signal reaches the model.** Forget to add positional encoding — or add it *after* attention instead of to the input embeddings — and nothing errors. The model trains and the loss can even look fine on order-insensitive data. But it's now permutation-invariant: "the dog bit the man" and "the man bit the dog" produce the same representation. Word order is silently ignored. A senior engineer catches this in **code review** by asking one question: "where does position enter?"
!!!

!!! c-info ⚖️
**Trade-off: which position scheme.** Sinusoidal needs no parameters and works at any length, but encodes *absolute* position; learned-absolute is flexible but undefined past the longest sequence seen in training, so it degrades on longer inputs; RoPE / ALiBi encode *relative* distance and extrapolate to longer contexts, at the cost of acting inside attention and being more to implement. Choosing one for a long-context model is a **design-review** call.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Plain self-attention is permutation-invariant — it computes weighted sums, which ignore input order — so word order has to be injected. Sinusoidal positional encoding adds a per-position vector of sine/cosine waves at many frequencies to each embedding, giving every position a unique fingerprint. The nuance is that this absolute scheme extrapolates poorly beyond training lengths, which is why modern LLMs use relative schemes like RoPE — rotating Q and K by a position-dependent angle — or ALiBi, a distance penalty on the scores."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict whether the shuffle changes anything, then run it" gotit="Done"
Here's the fun part. **Before you write any code, predict:** run a tiny self-attention on a sentence, then shuffle the word rows and run it again — will each word's output change, or come out the same set of vectors? Write your guess. Then add positional encoding and predict again. Pick one path:

#### Option A · write it yourself
Create `sessions/m03-attention/day-05-positional/experiment.py`. First run a tiny self-attention on an input, then shuffle the rows and run it again — show the per-word outputs are the same set (permutation invariance). Then write `sinusoidal_encoding(max_len, d_model)` using the sine/cosine formula and print a small table (say 6 positions × 8 dims). Finally add the encoding to the embeddings and show two shuffled inputs now differ. Run with `python3 sessions/m03-attention/day-05-positional/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 4 Day 5 artifact.

Create sessions/m03-attention/day-05-positional/experiment.py that, with a comment on each step:
1. Defines a tiny scaled dot-product self_attention(x) with fixed random projections.
2. Runs it on an input (seq=3, d_model=4), then on a row-shuffled input, showing the per-word outputs stay the same set (permutation invariance).
3. Implements sinusoidal_encoding(max_len, d_model) with PE(pos,2i)=sin(pos/10000**(2i/d_model)) and PE(pos,2i+1)=cos(...); prints a table for max_len=6, d_model=8 confirming every row is unique.
4. Adds the encoding to the embeddings (x + pe[:seq]) and shows two different word orders now produce different outputs.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Shuffling the input rows leaves the *set* of per-word attention outputs unchanged — proof that plain attention is permutation-invariant. Did you predict that?
- `sinusoidal_encoding(max_len, d_model)` prints a table whose rows are all unique — position 0 starts `[0, 1, 0, 1]`, position 1 starts `[0.84, 0.54, …]`.
- After adding the encoding, two different word orders produce *different* outputs. That's the whole fix: order is now something the meeting can read. You've finished the input side of a transformer.

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-05-positional/log.md`: (1) why plain attention ignores order; (2) how the seat-number fix restores it; (3) one thing you're still unsure about (why RoPE/ALiBi extrapolate better is fair game).
!!!
