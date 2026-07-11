---
quest_id: wf4-d02-qkv
donor: m03-day-02.donor
mode: reference
source_mode: clean
notebook_yardstick: null   # Reference mode (Notebook Smoothness Gate = N/A)
spine: "meeting: who to listen to"
module_label: "Module 4 · Represent · Day 2"
title: "Query, Key, Value"
subtitle: "The Three Roles Every Word Plays"
nav_prev_href: "../day-01-embeddings/lesson.html"
nav_next_href: "../day-03-attention-scores/lesson.html"
fin_title: "Module 4 · Day 2 complete! 🏆"
fin_body: "Nice work — you've completed <b>Query, Key, Value</b>.<br>Next up: <b>Attention Scores &amp; Softmax</b>."
references:
  - "Vaswani et al. 2017, 'Attention Is All You Need' (arXiv:1706.03762)"
---

@@@ region name=title
<title>Module 4 · Day 2 — Query, Key, Value</title>

@@@ region name=brand_sub
<div class="brand-sub">Foundations · M4 Day 2</div>

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
<a class="lnav prev" href="../day-01-embeddings/lesson.html"><span class="d">← Prev</span><span class="t">Words to Numbers (Embeddings)</span></a>

@@@ region name=nav_next
<a class="lnav next" href="../day-03-attention-scores/lesson.html"><span class="d">Next →</span><span class="t">Attention Scores &amp; Softmax</span></a>

@@@ hero
@lede
Back in the meeting. Yesterday every word showed up with a profile card of numbers. Now the words have to actually decide *who to listen to* — and here's the catch: the word "bank" means one thing next to "river" and something else next to "money," so a word can only figure itself out by looking at its neighbors. To do that cleanly, each word turns its one card into **three** different versions of itself: a **Query** (what am I looking for?), a **Key** (what do I advertise about myself?), and a **Value** (what will I actually share if someone listens?). Three roles, one word — that split is the engine of a transformer, and today you build it.
@goal
By the time you close this tab, you'll be able to say what Q, K, and V each mean, and compute all three from an embedding with three matmuls.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="Three versions of one word" gotit="Got why three vectors — let's go"
!!! c-info 🧭
**What this assumes:** Day 1 (embeddings — a word is a vector), and the matmul from Module 1. **Nothing else** — this is the first moving part of attention.
!!!

#### First, the picture — hold this before any words
Here is the picture to hold: in the meeting, one word plays **three roles at the same time**. It *asks* a question ("who here is relevant to me?"), it *advertises* a name-tag so others can decide if it's relevant to them, and it *carries* some content it will hand over if listened to. Same word, three hats — asker, advertiser, contributor. Every term below is just a name for one of those three hats.

#### The words you'll meet today — plain English first
Glance down this list first, so no term feels new when it arrives.

%%% jargon
attention | The mechanism that lets each word gather information from the *other* words, weighting them by how relevant they are. The heart of a transformer.
Query (Q) | The "what am I looking for?" version of a word — the question it asks the room.
Key (K) | The "what do I advertise?" version — the name-tag other words check against their query.
Value (V) | The "what will I share?" version — the content a word hands over if it's listened to.
projection | Turning one vector into another by multiplying it by a learned matrix. Q, K, V are three projections of the same embedding.
W_Q, W_K, W_V | The three learned weight matrices that make Q, K, and V. Training tunes them.
%%%

#### Why three versions?
It's tempting to ask why one word needs three vectors — it feels like numbers appearing from nowhere. The reason is the three hats: *asking*, *advertising*, and *contributing* are genuinely different jobs, so they need to be genuinely different numbers. A word might advertise "I'm a noun about money" (its Key) while asking "which adjective describes me?" (its Query) — those shouldn't be the same vector. So each word takes its Day-1 [[embedding||A word's profile card: the vector from Day 1.]] and passes it through three separate learned matrices.

#### How are they made?
From the same embedding `x`, with three matmuls: `Q = x @ W_Q`, `K = x @ W_K`, `V = x @ W_V`. Each `W` is a [[projection||Multiplying a vector by a learned matrix to reshape it into a new role.]] matrix the model *learns* during training. That's the whole construction — three matmuls from one card.

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 一个词为什么要变成三个向量？感觉像凭空多出来的 — people accept "one embedding → three vectors" but can't say *why three*. The point is the three hats: a word asks (Q), advertises (K), and offers content (V), and those roles have to be different numbers — so each gets its own learned matrix. Today we only *build* Q, K, V; the matching and mixing come in the next two lessons.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="Searching a library" gotit="Got the picture"
#### The same meeting, seen as a library search
Here's the picture that makes Q, K, V click, and it's the one the playground below uses. Think of every word in the meeting as someone walking into a library:

%%% cards
🔎 | Query = your search terms | What you're after right now. You'll compare it against every book's label to see which ones are relevant to *you*.
🏷️ | Key = the label · Value = the book | Each book advertises a **Key** (its spine label) and holds a **Value** (its actual contents). Where your query matches a key well, you read more of that book's value.
%%%

**In one line:** match your *query* against everyone's *keys*, then take more of the *values* whose keys matched best.

**直觉 / Intuition:** 把一个词想成同时在图书馆里做三件事：查资料（Q）、贴标签（K）、借出内容（V）。 The technical point is that Q, K, and V are three *different projections* of the same embedding, so a word can look for one thing, advertise another, and hand over a third — three roles, one input.

!!! c-warn ⚠️
Where the analogy breaks: in a real library one person searches while the books sit still — but in the meeting every word is *simultaneously* a searcher (its Query), a book on the shelf (its Key and Value), and it does this for all words at once, not one reader at a time. Today we just build Q, K, V; the matching (Day 3) and the merging (Day 3) come next.
!!!

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="The projections — and why they're the heart of a transformer" gotit="Got Q, K, V"
#### The mechanism
- Input: the embeddings `x`, shape `(seq_len, d_model)` — one row per word (the profile cards from Day 1).
- Three learned matrices project them: `Q = x @ W_Q`, `K = x @ W_K`, `V = x @ W_V`. Each `W` has shape `(d_model, d_k)`, so each output has shape `(seq_len, d_k)`.
- A word's relevance to another = **its query · the other's key** (a dot product). Higher = attend more. Building that full score grid is the next lesson.

%%% mathladder
title: one embedding → three roles
words: take a word's card and pass it through three different learned matrices; each matmul reshapes it into one of the three roles.
formula: `Q = x·W_Q`, `K = x·W_K`, `V = x·W_V` — `x` = one embedding row, shape `(1, d_model)`; each `W` is learned, shape `(d_model, d_k)`; each output has shape `(1, d_k)`.
numbers: word 0 has `x = [1, 0]`. With `W_Q = [[1,0],[0,1]]` → `Q[0] = [1, 0]`. With `W_K = [[0,1],[1,0]]` → `K[0] = [0, 1]`. Word 0's query dotted with the two keys `K[0]=[0,1]`, `K[1]=[1,0]` gives scores `[1·0+0·1, 1·1+0·0] = [0, 1]`.
sanity: the three outputs should generally *differ*. Word 0's query matches word 1's key (score 1), not its own (score 0) — attention has found an asymmetric link. If Q and K come out identical for every word, your projections are tied or missing.
%%%

#### Why a frontier lab cares
Q/K/V is the core of attention, and at scale it dominates the parameter and memory budget. Every attention layer carries its own `W_Q, W_K, W_V` — a big share of a transformer's weights across dozens of layers. Real models push this hard: Llama 3's largest model runs 128 separate query heads per layer, each with its own learned projection. And because the Keys and Values of every past token get cached during generation, `W_K`/`W_V` sizing is a first-order **production** memory decision at scale — the exact axis GQA and MQA (later) trade along.

!!! c-info 🔗
**Remember this chain:** one embedding → three projections (Q, K, V) → a query·key dot product scores relevance → (next lesson) softmax those scores and blend the Values. Today you built the three roles; Day 3 makes them talk.
!!!

#### The staff lens — one silent failure, one trade-off
Three projections look like a formality. Two things about them are worth naming precisely.

!!! c-warn ⚠️
**Failure mode (silent): Q and K are the same.** Skip the learned projections and feed the raw embedding as both Query and Key (or tie `W_Q = W_K`). Shapes are fine, so nothing crashes — but now each word's strongest match is *itself*, and attention can't form asymmetric links (a pronoun can't preferentially look back at its noun). The model still trains; it just learns weakly. A senior engineer catches this in **code review** by confirming Q and K come from *separate* learned projections.
!!!

!!! c-info ⚖️
**Trade-off: three full projections vs. fewer.** Separate `W_Q, W_K, W_V` cost parameters and compute, but they let a word ask for one thing, advertise another, and carry a third. Tie or shrink them to save memory and you lose some of that freedom. This is exactly the axis **GQA / MQA** trade along — sharing Keys and Values across heads to cut the memory bill. It's a **design-review** call: expressiveness vs. footprint.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Attention projects each token's embedding through three learned matrices into a Query, a Key, and a Value: <code>Q=x·W_Q</code>, <code>K=x·W_K</code>, <code>V=x·W_V</code>. The query is what a token is looking for, keys advertise what each token offers, and a query·key dot product scores relevance. The nuance is that Q and K must come from <em>separate</em> projections — tie them and every token's best match is itself, so attention can't form the asymmetric links, like a pronoun looking back at its noun, that make it useful."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict which word attends to which, then run it" gotit="Done"
Here's the fun part. **Before you write any code, predict:** if word 0's Query is `[1, 0]` and the two Keys come out as `[0, 1]` (word 0) and `[1, 0]` (word 1), which word does word 0 attend to more — itself, or word 1? Jot it down. Then build Q, K, V and let the dot products settle it. Pick one path:

#### Option A · write it yourself
Create `sessions/m03-attention/day-02-qkv/experiment.py`. Start from an embedding matrix `x` of shape `(2, 2)`. Make three weight matrices `Wq, Wk, Wv`, compute `Q = x@Wq`, `K = x@Wk`, `V = x@Wv`, print their shapes, then compute one word's query dotted with every key (`K @ Q[0]`). Run with `python3 sessions/m03-attention/day-02-qkv/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 4 Day 2 artifact.

Create sessions/m03-attention/day-02-qkv/experiment.py that, with a comment on each step:
1. Defines x = np.array([[1.,0.],[0.,1.]]) as two word embeddings (rows).
2. Defines Wq=identity, Wk=[[0,1],[1,0]], Wv=2*identity; computes Q=x@Wq, K=x@Wk, V=x@Wv and prints each with its shape (2,2).
3. Explains in comments what Q, K, V represent (looking-for / advertise / contribute).
4. Computes scores = K @ Q[0] (word 0's query vs every key) and prints them (expect [0., 1.]), noting higher = attend more.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- `Q`, `K`, `V` each print with shape `(2, 2)` and are *not* all identical — three different views of the same two words.
- `scores = K @ Q[0]` prints `[0., 1.]` — word 0 attends to word 1, not to itself. Did you predict that asymmetric link?
- The thing worth noticing: one input word produced three different vectors, and the *query·key* dot product — not the word itself — decided who listens to whom. That decision is exactly what Day 3 turns into attention weights.

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-02-qkv/log.md`: (1) which of Q / K / V is the "search," the "label," and the "book"; (2) why Q and K must be different projections; (3) one thing you're still unsure about.
!!!
