---
quest_id: wf4-d01-embeddings
donor: m03-day-01.donor
mode: reference
source_mode: clean
notebook_yardstick: null   # no companion sessions/ notebook — Reference mode (Notebook Smoothness Gate = N/A)
spine: "meeting: who to listen to"
module_label: "Module 4 · Represent · Day 1"
title: "Words to Numbers"
subtitle: "Embeddings: Meaning as Geometry"
nav_prev_href: "../../m02-the-neuron/review.html"
nav_next_href: "../day-02-qkv/lesson.html"
fin_title: "Module 4 · Day 1 complete! 🏆"
fin_body: "Nice work — you've completed <b>Words to Numbers</b>.<br>Next up: <b>Query, Key, Value</b>."
references:
  - "Vaswani et al. 2017, 'Attention Is All You Need' (arXiv:1706.03762)"
---

@@@ region name=title
<title>Module 4 · Day 1 — Words to Numbers (Embeddings)</title>

@@@ region name=brand_sub
<div class="brand-sub">Foundations · M4 Day 1</div>

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
<a class="lnav prev" href="../../m02-the-neuron/review.html"><span class="d">← Prev</span><span class="t">M2 · Review Gate</span></a>

@@@ region name=nav_next
<a class="lnav next" href="../day-02-qkv/lesson.html"><span class="d">Next →</span><span class="t">Query, Key, Value</span></a>

@@@ hero
@lede
Picture a meeting where the smartest thing in the room is a sentence, and every word is a person who will soon have to decide *who else to listen to*. Over the next few days you'll build that meeting piece by piece. But there's a problem before anyone can even speak: a computer cannot do arithmetic on the letters `c-a-t`. So the very first step of every language model turns each word into a small list of numbers — a **profile card** called an *embedding* — arranged so that words that *mean* similar things get similar cards. Get the cards right and "cat" and "dog" end up sitting close together, while "cat" and "car" drift far apart. Everything the meeting does later runs on these cards.
@goal
By the time you close this tab, an embedding will feel obvious: you'll turn a word into a row of numbers by looking it up in a table, and measure how alike two words are with a single dot product.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="A word's profile card" gotit="Got what an embedding is — let's go"
!!! c-info 🧭
**What this assumes:** Module 1 — arrays, indexing a row, and the dot product. **No transformer background needed** — this is the very first step of one.
!!!

#### First, the picture — hold this before any words
Here is the one picture to hold in your head: every word walks into the meeting carrying a small **profile card**, and that card is just a short list of numbers. Words with similar meanings carry similar cards, so "cat" and "dog" hold nearly the same card, while "car" holds a very different one. That is the whole idea. Every term below is only a name for a part of that single picture — not a new thing to be scared of.

#### The words you'll meet today — plain English first
Before any formula, here is every term you'll run into today, said in plain words. You don't need to memorize these — just glance down the list so no word feels new when it shows up.

%%% jargon
token | One word (or word-piece) — the basic unit a language model reads. Splitting text into tokens is called *tokenization*.
embedding | The profile card: a list of numbers that stands for a token's meaning. Similar tokens get similar lists.
vector | Just a list of numbers. An embedding *is* a vector.
embedding matrix | The big lookup table that holds one embedding (one row) per token in the vocabulary.
d_model | How long each profile card is — the number of numbers in one embedding. A core size knob of the whole model.
similarity | How alike two cards are. We measure it with a **dot product**: bigger means more alike.
%%%

#### What is an embedding?
An [[embedding||A vector of numbers that represents a token's meaning. Similar tokens get nearby vectors. Stored as a row of a learned embedding matrix.]] is that profile card written as a [[vector||Just a list of numbers.]] — a list of numbers that stands for a [[token||A word or word-piece: the basic unit a language model reads.]]'s meaning. The trick is *how* the numbers are arranged: **similar words get similar vectors.** Nothing about the word's spelling matters; only its meaning, turned into position.

#### How is it stored?
As one big lookup table — the [[embedding matrix||A matrix of shape (vocabulary size, d_model). Row i is the embedding vector for token id i.]]. You hand it a token's id number `i` and it hands back row `i` (that's just indexing a row, from M1). Those numbers aren't typed in by a person; they are **learned during training** (the M2/M3 loop), so meaning slowly becomes geometry.

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 很多人以为 embedding 是某种复杂算法 — people expect the "embedding" to be some clever algorithm. In plain terms the lookup is trivial: it is just "grab row `i`". What feels like magic is that *meaning* ends up living in those numbers at all — and it does only because training slowly moved similar words' rows close together before you ever opened the table.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="A map of meaning" gotit="Got the picture"
#### Turn the profile cards into a map
You already have the picture from Section 1 — each word carries a card of numbers. Now do one more thing with it: treat every card as a set of *coordinates* and drop each word onto a map. Suddenly the cards have a shape. Words that mean similar things land in the same neighborhood; unrelated words sit far away. The meeting we're building doesn't care what a word looks like — it only ever reads *where each word sits on this map*.

%%% cards
🗺️ | Words as places on a map | Each word is a point. Related words cluster — animals in one area, vehicles in another. The *position* carries the meaning, and closeness means "similar."
🎨 | Like a color code | A color is just three numbers (R, G, B), and similar colors have similar numbers. A word is a longer list of numbers, and the same rule holds: nearby lists mean alike.
%%%

**In one line:** an embedding turns each word into a point in space so that meaning becomes distance — and that map is the seating chart for the whole meeting to come.

**直觉 / Intuition:** 把每个词想成地图上的一个点，意思越接近，点就越靠近。 The technical point is that an embedding places each token at coordinates in a `d_model`-dimensional space, so "similar meaning" becomes "small distance," and a dot product reads that closeness off directly.

!!! c-warn ⚠️
Where the map analogy breaks: a real embedding isn't 2-D — it's hundreds of dimensions (a size called `d_model`). We draw flat 2-D pictures to see the idea, but the model works in a much bigger space you can't fully picture. The rule "similar = close" still holds; you just can't sketch all the axes.
!!!

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="The lookup table — and why every model starts here" gotit="Got embeddings"
#### The mechanism
- The [[embedding matrix||Shape (vocab_size, d_model): one row per token, each row a vector of length d_model.]] has shape `(vocab_size, d_model)`: one row per token, each row a vector of length `d_model`.
- Embedding a token = **indexing its row**: `E[id]` (M1).
- [[Similarity||How alike two vectors are — a dot product; bigger = more alike.]] between two vectors = a **dot product** (M1): bigger means more alike.
- The numbers are **learned** by the training loop, so words used in similar contexts drift close together on the map.

%%% mathladder
title: similarity as a dot product
words: to score how alike two profile cards are, multiply them position-by-position and add it all up.
formula: `sim(a, b) = a · b = Σᵢ aᵢ·bᵢ` — `a`, `b` = two embedding vectors of length `d_model`; `aᵢ`, `bᵢ` = their i-th numbers; the sum runs over all `d_model` positions.
numbers: cat `= [0.9, 0.1]`, dog `= [0.8, 0.2]` → `0.9·0.8 + 0.1·0.2 = 0.72 + 0.02 = 0.74`. car `= [-0.7, 0.6]` → cat·car `= 0.9·(-0.7) + 0.1·0.6 = -0.63 + 0.06 = -0.57`.
sanity: the two animals (0.74) score far above cat-vs-car (−0.57). If a pair you *expect* to be similar scores lower than an unrelated pair, the vectors — not the formula — are wrong.
%%%

#### Why a frontier lab cares
This is **step 1 of every language model**: text → tokens → embeddings, and everything after — the whole listening meeting (attention), the layers you built in Module 2 — operates on these cards. Two facts make it matter at scale. First, `d_model` is a core size knob: it sets how much nuance each card can hold. Second, the embedding matrix is `(vocab × d_model)`, which in a small model is often the single **biggest** block of parameters — a real chunk of a model's millions or billions of weights, and a real slice of the memory bill in production. Real tokenizers show the scale directly: Llama 3 has a 128,256-token vocabulary and GPT-4o's is roughly 200,000 — that many rows of learned vectors, no more, no less. Get the input representation right and everything downstream has something meaningful to work with.

!!! c-info 🔗
**Remember this chain:** text → tokens → ids → embedding lookup → vectors that place meaning in space. Next lesson: how each word's card gets turned into three roles so it can look at the *other* words in the meeting — that's Query, Key, Value.
!!!

#### The staff lens — one silent failure, one trade-off
The lookup is simple, which is exactly why its failures hide. A staff engineer knows the one that never crashes, and the size knob they defend in review.

!!! c-warn ⚠️
**Failure mode (silent): a token indexes the wrong row.** If the tokenizer and the embedding matrix disagree on ids — or every rare word collapses to one shared `<unk>` row — nothing crashes. The model trains and the loss drops, but each affected word gets a consistent-but-wrong card, and distinct rare words become literally indistinguishable (identical vectors). A senior engineer catches this in **code review** by checking that `vocab_size` equals the number of embedding rows and that out-of-vocabulary tokens aren't silently merged.
!!!

!!! c-info ⚖️
**Trade-off: how wide to make `d_model`.** Wider cards hold more nuance, but the matrix is `(vocab × d_model)`, so width costs memory and can overfit on little data. Too narrow and distinct words can't get distinct-enough cards. Choosing the width for a given vocabulary and data budget is a **design-review** call, not a free knob.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "An embedding is a learned lookup table: token id <code>i</code> returns row <code>i</code> of a <code>(vocab_size, d_model)</code> matrix, and because training pulls together words used in similar contexts, a dot product between two rows measures how related they are. The nuance is that this matrix is often the single largest parameter block in a small model, so <code>d_model</code> and vocabulary size are a real memory-vs-expressiveness trade-off — and the silent bug to watch for is a tokenizer/embedding id mismatch, which never crashes but gives whole classes of words a consistent-but-wrong vector."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict which pair scores higher, then run it" gotit="Done"
Here's the fun part. **Before you write any code, predict:** you'll score "cat" against "dog" and "cat" against "car" with a dot product — which pair comes out higher, and will either score go *negative*? Jot your guess down. Then build the tiny table and find out. That little "I was right / oh, I was wrong" moment is what makes the day stick. Pick one path:

#### Option A · write it yourself
Create `sessions/m03-attention/day-01-embeddings/experiment.py`. Make a tiny `vocab` dict, an embedding matrix `E` of shape `(4, 2)` with hand-set rows so cat/dog are close and car is far, look up a couple of words by id, and compute the dot-product similarity of cat & dog versus cat & car. Print each step. Run with `python3 sessions/m03-attention/day-01-embeddings/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 4 Day 1 artifact.

Create sessions/m03-attention/day-01-embeddings/experiment.py that, with a comment on each step:
1. Defines vocab = {"cat":0,"dog":1,"car":2,"the":3} and maps a sentence to ids.
2. Builds an embedding matrix E of shape (4,2) with hand-set rows so cat/dog are close and car is far.
3. Looks up E[vocab["cat"]] and prints it (indexing a row = M1).
4. Defines sim(a,b)=np.dot(a,b) and prints sim(cat,dog) vs sim(cat,car), confirming the animals score higher than cat-vs-car.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Looking up "cat" (id 0) just returns row 0 of `E` — a vector like `[0.9, 0.1]`. That's indexing a row, exactly M1.
- `sim(cat, dog)` comes out clearly higher (around `0.74`) than `sim(cat, car)` (around `−0.57`). The animals win, and the unrelated pair even goes negative — did you predict the sign?
- The thing worth noticing: nothing about spelling was used. Meaning lived entirely in *where the rows sat*, and one dot product read it off. That's the whole reason embeddings come first.

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-01-embeddings/log.md`: (1) why it is useful that similar words end up with nearby vectors; (2) the similarity number that surprised you; (3) one thing you're still unsure about (why the numbers are *learned* and not typed in is fair game).
!!!
