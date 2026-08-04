---
quest_id: wf4-d02-qkv
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 2 — Query, Key, Value"
module_label: "Module 4 · Represent · Day 2"
title: "Query, Key, Value"
subtitle: "The Three Roles Every Word Plays"
brand_sub: "Foundations · M4 Day 2"
spine: "library"
nav_prev_href: "../day-01-embeddings/lesson.html"
nav_prev_label: "Words to Numbers (Embeddings)"
nav_next_href: "../day-03-attention-scores/lesson.html"
nav_next_label: "Attention Scores & Softmax"
fin_title: "Module 4 · Day 2 complete! 🏆"
fin_body: "Nice work — you've met the three roles every word plays: a <b>Query</b> (what it's looking for), a <b>Key</b> (the label it advertises), and a <b>Value</b> (the content it shares) — all three projected from its one embedding.<br>Next up: <b>Attention Scores &amp; Softmax</b> — how a word compares its Query to everyone's Key and decides who to listen to."
notebook_yardstick: null
coverage_topics:
  - {topic: query key value roles, keywords: [query, key, value]}
  - {topic: projection matrices, keywords: [W_Q, projection, learned matrix]}
  - {topic: attention score dot product, keywords: [dot product, score]}
  - {topic: softmax weights, keywords: [softmax, sum to 1, weights]}
  - {topic: weighted sum of values, keywords: [weighted sum, blend]}
  - {topic: scaled dot product, keywords: [sqrt, scale, d_k]}
  - {topic: full pipeline, keywords: [pipeline, end to end, output]}
  - {topic: self attention same sequence, keywords: [self-attention, same sentence, same sequence]}
---

@@@ hero
@lede You know how you find a book in the library? You hold one question in your head — "something about volcanoes" — and your eyes skip along the shelf reading the little labels on each spine, slowing down on the ones that match, and you pull *those* books down to read. You do it without thinking. Here's the surprise: that library walk — hold a question, scan the labels, take more from the best matches — is the single move that lets ChatGPT understand a sentence. (The same "compare a question to a label, then take more of the best match" idea also decides which faces in your camera roll belong together, and which song a music app queues up next.) It's called **attention**, and today you build its beating heart. Yesterday you turned each word into a little profile card of numbers. But a card just sits there. Today each word learns to *look around the room* — because "bank" can only tell if it means a river-bank or a money-bank by peeking at its neighbours. To do that, every word quietly splits its one card into **three** versions of itself: a **Query** (what am I looking for?), a **Key** (what do I advertise?), and a **Value** (what will I share?). Three roles, one word — and by the end of today it'll feel as easy as walking the aisle.
@goal Together we'll give one word three roles, score a word's question against everyone's label, keep those scores from getting too loud, turn them into a "how much to listen" recipe that always adds up to 1, and blend the room into a fresh, context-aware word. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.

%%% warmup
q: Yesterday you turned each word into its own short list of numbers. What is that list called? | a:1 | a token id | an embedding | a softmax | a one-hot | concept: embeddings | fb: An embedding is a word's profile card — a fixed-length list of numbers that stands for its meaning. Today each word splits this one list into three roles.
q: On yesterday's "map of meaning", what did it mean when two words sat CLOSE together? | a:2 | they are spelled alike | they have the same length | they mean similar things | they came from the same sentence | concept: map-of-meaning | fb: Close = similar meaning. That "how alike are these two arrows?" idea comes back today as the match score between a Query and a Key.
%%%

@@@ concept id=c1 tag="Three roles" title="One word plays three roles at once" gotit="Got the three roles"
Let's start with the surprise that trips up almost everyone: today each single word turns into *three* little lists of numbers, not one. That sounds like numbers appearing from thin air. It isn't — a word has three genuinely different jobs to do in a sentence, and three jobs need three tools.

Picture walking into a [[library||A room full of books; today's picture for a sentence full of words.]] with a question in your head. You do three separate things without noticing. You *ask* — you hold your question, "something about volcanoes." Every book *advertises* itself with a short label on its spine — "COOKING", "SPACE", "VOLCANOES". And once you find a match, you *take its actual contents* down to read. Asking, advertising, handing over content: three different jobs. A word in a sentence does all three at the same time, so it makes three versions of itself — a **Query** (its question), a **Key** (its spine-label), and a **Value** (its contents).

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="One word splits into three roles. On the left, a single word card holding the word bank. Three arrows fan out to three labelled cards: a Query card showing a magnifying glass and the words what am I looking for, a Key card showing a spine label and the words what do I advertise, and a Value card showing a book and the words what will I share."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One word → three roles it plays at the same time</text><rect x="24" y="80" width="96" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="72" y="100" text-anchor="middle" fill="#8A6D3B" font-weight="bold">word</text><text x="72" y="116" text-anchor="middle" fill="#6B645E" font-size="10">"bank"</text><line x1="120" y1="96" x2="180" y2="60" stroke="#B8AEA2"/><line x1="120" y1="102" x2="180" y2="102" stroke="#B8AEA2"/><line x1="120" y1="108" x2="180" y2="150" stroke="#B8AEA2"/><rect x="184" y="38" width="316" height="44" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="200" y="58" fill="#5E5191" font-size="16">🔎</text><text x="224" y="58" fill="#5E5191" font-weight="bold">Query</text><text x="224" y="73" fill="#6B645E" font-size="9">"what am I looking for?"</text><rect x="184" y="88" width="316" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="200" y="108" fill="#8A6D3B" font-size="16">🏷️</text><text x="224" y="108" fill="#8A6D3B" font-weight="bold">Key</text><text x="224" y="123" fill="#6B645E" font-size="9">"what do I advertise about myself?"</text><rect x="184" y="138" width="316" height="44" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="200" y="158" fill="#276b45" font-size="16">📖</text><text x="224" y="158" fill="#276b45" font-weight="bold">Value</text><text x="224" y="173" fill="#6B645E" font-size="9">"what content will I share if listened to?"</text></g></svg>
%%%

**Gets right:** asking, advertising and handing over content really are three separate jobs — so one word needs three separate versions of itself.
**Breaks down:** in a real library one person searches while the books sit still. In a sentence *every* word is a searcher, a spine-label and a book, all at once. Nobody waits their turn.

#### Walk the aisle once, and name each move
Stay in the aisle and go slowly. Each rung below is one move you already make without thinking — and one of the three roles.

%%% steps
step: You hold your question — "something about volcanoes."
why: That question is the **Query**. It never changes what's on the shelf; it just decides where your eyes go.
step: Each spine shows a short label — "COOKING", "SPACE", "VOLCANOES".
why: That label is the **Key**. It exists purely to be *matched against*, which is why it can be short and blunt.
step: You pull the matching book down and read its pages.
why: The pages are the **Value** — the real content. Notice the pages are nothing like the spine label: that's the whole reason Key and Value stay separate.
step: You spend more time on the matching books, less on the others.
why: That "more time here, less there" is attention itself. Everything else today is just this, done with numbers.
%%%

%%% insight
Here's the part worth pausing on: the **Key and the Value are deliberately different things.** A spine says "VOLCANOES" — three words wide — while the pages hold hundreds of paragraphs. If a book had to use its pages *as* its label, you'd have to read every book to find out which one to read. That's why a word advertises one thing (its Key) and hands over another (its Value).
%%%

So far: three jobs, three versions of the word. Now let's give them a sentence to live in.

#### The sentence we'll use all day
Our sentence is **"the animal crossed the river bank"**. The word "bank" is confused — river-bank or money-bank?

- Its **Query** is basically "what kind of place am I?"
- Its neighbour "river" advertises a **Key** that says "I'm watery, and I'm a place."
- "river" also carries a **Value** full of water-meaning.

When "bank"'s question matches "river"'s label, "bank" pulls in "river"'s value — and settles on river-bank. That's the whole day in one sentence, and you just read it.

!!! c-info 🗺️
<b>The one line to hold:</b> match your <b>Query</b> against everyone's <b>Key</b>, then take more of the <b>Values</b> whose keys matched best. Everything today is a slow-motion version of that one line.
!!!

@@@ concept id=c2 tag="Making the three" title="Where the three versions come from" gotit="Got the projections"
So a word needs three versions of itself. Fair question: where do they come from? A word only arrived with *one* card — the embedding from Day 1. How does one card become three?

Think of a [[photo booth||A little booth that takes one photo of you and prints three differently-styled copies.]] at a fair that takes *one* photo of you and prints three copies through three different filters: one in bright cartoon colours, one in black-and-white, one in sepia. Same you, same original photo — but each filter reshapes it for a different use (the cartoon for a birthday card, the black-and-white for an ID).

A word does exactly this. It passes its one embedding through three different *filters* to get its Query, its Key and its Value. In machine learning a filter is just a grid of numbers you multiply by, and the multiply-by-a-grid step is called a [[projection||Turning one list of numbers into another by multiplying it by a grid of numbers. Q, K, and V are three projections of the same embedding.]].

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="One embedding card passes through three different filters, drawn as three coloured lenses labelled W_Q, W_K and W_V, and comes out as three different cards: a Query card, a Key card, and a shorter Value card. The caption says same input, three filters, three outputs."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One card + three filters → three cards</text><rect x="14" y="86" width="100" height="48" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="64" y="104" text-anchor="middle" fill="#6B645E" font-size="10">"bank" card</text><text x="64" y="122" text-anchor="middle" fill="#2C2A28" font-size="10">[0, 1, 1, 0]</text><g><line x1="114" y1="102" x2="192" y2="56" stroke="#5E5191" stroke-dasharray="3,2"/><line x1="114" y1="110" x2="192" y2="110" stroke="#C99A12" stroke-dasharray="3,2"/><line x1="114" y1="118" x2="192" y2="164" stroke="#2D8B55" stroke-dasharray="3,2"/></g><ellipse cx="208" cy="52" rx="18" ry="14" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="208" y="56" text-anchor="middle" fill="#5E5191" font-size="9">W_Q</text><ellipse cx="208" cy="110" rx="18" ry="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="208" y="114" text-anchor="middle" fill="#8A6D3B" font-size="9">W_K</text><ellipse cx="208" cy="168" rx="18" ry="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="208" y="172" text-anchor="middle" fill="#276b45" font-size="9">W_V</text><g><line x1="226" y1="52" x2="286" y2="52" stroke="#B8AEA2"/><line x1="226" y1="110" x2="286" y2="110" stroke="#B8AEA2"/><line x1="226" y1="168" x2="286" y2="168" stroke="#B8AEA2"/></g><rect x="290" y="30" width="210" height="44" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="304" y="50" fill="#5E5191" font-weight="bold">Query</text><text x="490" y="50" text-anchor="end" fill="#2C2A28" font-size="10">[1.0, 0.5, 0, 0]</text><text x="304" y="66" fill="#6B645E" font-size="9">the question it asks</text><rect x="290" y="88" width="210" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="304" y="108" fill="#8A6D3B" font-weight="bold">Key</text><text x="490" y="108" text-anchor="end" fill="#2C2A28" font-size="10">[0, 2.0, 4.0, 0]</text><text x="304" y="124" fill="#6B645E" font-size="9">the label it advertises</text><rect x="290" y="146" width="210" height="44" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="304" y="166" fill="#276b45" font-weight="bold">Value</text><text x="490" y="166" text-anchor="end" fill="#2C2A28" font-size="10">[0.1, 1.0]</text><text x="304" y="182" fill="#6B645E" font-size="9">the content it shares (shorter — that's allowed)</text></g></svg>
%%%

**Gets right:** one input, three filters, three genuinely different outputs — exactly Q, K, V from one embedding.
**Breaks down:** a photo booth's filters are set at the factory. A word's three filters are **learned** — during training the model slowly tunes them so Queries and Keys line up in useful ways. Nobody types them in, just like Day 1's embeddings.

#### First, the three filters get names
The three filters are three grids of numbers called [[W_Q, W_K, W_V||The three learned filter-grids that turn one embedding into a Query, a Key, and a Value. Training tunes them.]] — one grid per role. "W" is just the usual letter for a grid of *weights* (numbers the model learns).

In one plain line: to get a word's Query, multiply its embedding by the Query-grid. Same for Key and Value.

%%% formula
expr: Q = x · W_Q     K = x · W_K     V = x · W_V
note: x = one word's embedding (its Day-1 card). Each W is a learned grid. The dot means "multiply the list by the grid." Three multiplies → three role-vectors.
%%%

Notice how plain this stage is: three multiplies, and **no bend** — no activation, no threshold, nothing curved. That's on purpose. This stage only *re-aims* each card; the curved, feature-building part of a transformer lives in a later sublayer of the same block.

%%% insight
Why three *different* grids? Because with one grid used three times, a word's question and its label would be the very same list — so the grid of scores would come out perfectly symmetric, and every word would mostly match whichever word looks most like it (often itself). Three different grids are what let a word ask for one thing while advertising another. That freedom is the whole reason attention works.
%%%

!!! c-warn 🎚️
<b>The grids also have to <i>start</i> at a sensible size.</b> Born too big and the scores are deafening before training even begins, so attention freezes onto one word from birth; born too small and every score is a shrug, so there is nothing to learn from. The standard cure is <b>Xavier (Glorot) initialization</b> for these grids (spread roughly 1/d_model) plus <b>LayerNorm</b> on the incoming cards, so the filters always see numbers of a controlled size. Names filed away — the training module opens that box.
!!!

So far: one card, three learned grids, three role-vectors. Now the part beginners genuinely lose track of — the shapes.

#### Then, watch the shape change
Back to the photo booth: a filter doesn't just recolour the photo, it can also print it at a *different size*. Same here. Watch the numbers on the boxes.

%%% svg
<svg viewBox="0 0 520 224" role="img" aria-label="A shape diagram matching the worked numbers. A card of shape 1 by 4 is multiplied by a grid W_Q of shape 4 by 4, producing a Query of shape 1 by 4. Below, the same card multiplied by W_V of shape 4 by 2 produces a Value of shape 1 by 2. The near-side numbers are highlighted in red as the ones that must match, and the far side sets the output length."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Multiply reshapes the card — watch the numbers</text><rect x="20" y="42" width="64" height="44" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="52" y="68" text-anchor="middle" fill="#2C2A28">x</text><text x="52" y="102" text-anchor="middle" fill="#6B645E" font-size="10">1 × 4</text><text x="52" y="115" text-anchor="middle" fill="#9A938A" font-size="9">the word</text><text x="96" y="70" fill="#B8AEA2" font-size="16">×</text><rect x="118" y="38" width="64" height="52" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="150" y="68" text-anchor="middle" fill="#8A6D3B">W_Q</text><text x="150" y="102" text-anchor="middle" fill="#6B645E" font-size="10">4 × 4</text><text x="194" y="70" fill="#B8AEA2" font-size="16">=</text><rect x="216" y="42" width="80" height="44" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="256" y="68" text-anchor="middle" fill="#5E5191">Q</text><text x="256" y="102" text-anchor="middle" fill="#6B645E" font-size="10">1 × 4</text><rect x="66" y="126" width="14" height="14" fill="#FDECEC" stroke="#C93B3B"/><rect x="126" y="126" width="14" height="14" fill="#FDECEC" stroke="#C93B3B"/><text x="150" y="137" fill="#C93B3B" font-size="9">these two must match — the only rule of the multiply</text><rect x="20" y="156" width="64" height="44" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="52" y="182" text-anchor="middle" fill="#2C2A28">x</text><text x="52" y="216" text-anchor="middle" fill="#6B645E" font-size="10">1 × 4</text><text x="96" y="184" fill="#B8AEA2" font-size="16">×</text><rect x="118" y="152" width="64" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="150" y="182" text-anchor="middle" fill="#276b45">W_V</text><text x="150" y="216" text-anchor="middle" fill="#6B645E" font-size="10">4 × 2</text><text x="194" y="184" fill="#B8AEA2" font-size="16">=</text><rect x="216" y="156" width="48" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="240" y="182" text-anchor="middle" fill="#276b45">V</text><text x="240" y="216" text-anchor="middle" fill="#6B645E" font-size="10">1 × 2</text><text x="404" y="52" text-anchor="middle" fill="#5E5191" font-size="10">d_model = 4 → the card's width</text><text x="404" y="70" text-anchor="middle" fill="#8A6D3B" font-size="10">d_k = 4 → Query &amp; Key length</text><text x="404" y="88" text-anchor="middle" fill="#276b45" font-size="10">d_v = 2 → Value length (free!)</text><text x="404" y="122" text-anchor="middle" fill="#9A938A" font-size="9">Q and K MUST match in length —</text><text x="404" y="134" text-anchor="middle" fill="#9A938A" font-size="9">they get compared to each other.</text><text x="404" y="152" text-anchor="middle" fill="#9A938A" font-size="9">V never gets compared, only poured,</text><text x="404" y="164" text-anchor="middle" fill="#9A938A" font-size="9">so it can be a different length.</text><text x="404" y="196" text-anchor="middle" fill="#276b45" font-size="9">do the multiply 3 times, once per grid</text></g></svg>
%%%

Read that figure one rung at a time and it stops being scary.

%%% steps
step: The card goes in as `1 × 4` — one word, four numbers.
why: `1` means "one word". The `4` is the card's width, which has a name: `d_model` — the width of the vector that carries one word through the whole model.
step: The grid's *near* side must equal the card's width — the two red boxes in the picture.
why: That's the only rule of the multiply. Break it and your code crashes — honestly the friendliest kind of bug, because it tells you loudly instead of quietly.
step: The grid's *far* side decides the output length: `d_k` for Q and K, `d_v` for V.
why: Which means the filter can shrink or stretch the card. Here `d_k = 4` but `d_v = 2` — and that's completely legal, because a Value never gets compared to anything.
step: Run the same multiply three times, once per grid.
why: Therefore one card leaves as three role-vectors — Query, Key, Value. Same input, three different aims.
%%%

One rule to carry forward: **the Query and the Key must come out the same length** (`d_k` for both). They get compared to each other in a moment, and you can't line a 4-number list up against a 2-number list. The Value has no such duty.

!!! c-warn 📐
<b>The two shape bugs everyone hits, and the cure.</b> First: a Query and a Key built at different lengths — the dot product then has nothing to line up. Second: a forgotten flip (a "transpose") when scoring every Query against every Key, because rows and columns get mixed up. Both are loud crashes, not silent ones. The cure is boring and effective: hold the rule `d_q = d_k`, and **print the shape after every single step** — exactly what today's `experiment.py` does.
!!!

!!! c-info 🪜
<b>Optional (skippable) — the arithmetic, for the curious.</b> Skipping this costs you nothing today.<br>
<b>The shortcut:</b> We picked "bank"'s card as `x = [0, 1, 1, 0]` on purpose: when a card is made of 0s and 1s, multiplying by a grid is just "**add up the grid rows where the card has a 1**".<br>
<b>Row by row:</b> `W_Q`'s rows are `[0.1,0.1,0,0]`, `[0.5,0.5,0,0]`, `[0.5,0,0,0]`, `[0,0,0.5,0.5]` — so `Q = row 2 + row 3 = [1.0, 0.5, 0, 0]`. Do the same with `W_K`'s rows and you get `K = [0, 2.0, 4.0, 0]`; with `W_V`'s two-column rows you get `V = [0.1, 1.0]`.<br>
<b>Sanity check:</b> Those are exactly the numbers in both pictures above, and exactly what `experiment.py` prints.
!!!

If matrix multiplication feels slippery right now, that's completely normal — everyone re-learns it about four times. You can run every idea in today's lesson while holding only "a filter reshapes the card."

%%% demo id=proj label="turn one word into Q, K, V"
predict: "bank" walks in as `[0, 1, 1, 0]` and passes through three DIFFERENT filters. Will its Query, Key and Value come out the same, or different? Guess before you reveal.
code: x = [0, 1, 1, 0];  Q = x @ Wq;  K = x @ Wk;  V = x @ Wv
out: Q -> [1.0, 0.5, 0.0, 0.0]      (length 4 = d_k)
out: K -> [0.0, 2.0, 4.0, 0.0]      (length 4 = d_k)
out: V -> [0.1, 1.0]                (length 2 = d_v — shorter, and that's fine)
take: <b>One input card became three different lists.</b> The three filters pulled the same card in three directions — a question, a label, and some content. Q and K came out the same length because they're about to be compared; V came out shorter because it only ever gets poured. If all three filters were the same grid, Q and K would be identical lists and a word could no longer ask for one thing while advertising another.
%%%

Victory lap: you can now say where Q, K and V come from — one card, three learned grids — which is the single sentence most people fumble.

@@@ concept id=c3 tag="Scoring a match" title="How well does my question match your label?" gotit="Got the score"
Now the fun begins. Every word has a Query (its question) and every word has a Key (its label). The next job is the heart of the library trick: for one word, go down the shelf and score *how well my question matches each label*.

Picture standing in the aisle holding a slip that says "**volcano, lava, eruption**". You glance at each spine and almost instantly feel how strong the match is. A spine reading "**volcanoes, lava, magma**" lights up — lots of overlap. "**cake recipes**" gets nothing. You're not reading whole books yet; you're giving each one a quick *match score*.

A word does the same. It takes its Query and, for every other word, measures how much its Query lines up with that word's Key. Big number = strong match = "pay attention to this one." The name for this line-up score is the [[dot product||A single number that says how much two lists agree. Multiply them slot by slot, then add it all up. Big = strong match.]] — just "multiply the two lists slot by slot, then add it all up."

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A query slip reading volcano lava eruption is compared against two spine labels. The first label volcanoes lava magma has heavy overlap and gets a high match score. The second label cake recipes has no overlap and gets a low match score. Arrows and a green highlight show the strong match."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Score each spine-label against my question</text><rect x="180" y="30" width="160" height="40" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="260" y="48" text-anchor="middle" fill="#5E5191" font-weight="bold">🔎 my Query</text><text x="260" y="63" text-anchor="middle" fill="#6B645E" font-size="9">"volcano, lava, eruption"</text><line x1="230" y1="70" x2="140" y2="108" stroke="#2D8B55" stroke-width="2"/><line x1="300" y1="70" x2="392" y2="108" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3,2"/><rect x="30" y="112" width="212" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="46" y="132" fill="#276b45" font-size="16">🏷️</text><text x="70" y="132" fill="#276b45" font-size="10">"volcanoes, lava, magma"</text><text x="136" y="152" text-anchor="middle" fill="#276b45" font-weight="bold">strong match → high score</text><rect x="280" y="112" width="212" height="56" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="296" y="132" fill="#9A938A" font-size="16">🏷️</text><text x="320" y="132" fill="#6B645E" font-size="10">"cake recipes"</text><text x="386" y="152" text-anchor="middle" fill="#9A938A" font-weight="bold">weak match → low score</text><text x="260" y="190" text-anchor="middle" fill="#6B645E" font-size="10">one glance, one number, per book — no pages read yet</text></g></svg>
%%%

**Gets right:** you compare your question to a label and get a single "how much do these agree?" number, fast, for every book. That single number *is* the dot product.
**Breaks down:** real overlap counts shared *words*, but the dot product works on the number-lists Q and K — so it can spot a match even when no letter is shared. It feels the match in meaning, not spelling. It also has a quirk a librarian doesn't: the score grows with how **many slots** the two lists have — the puzzle we solve in the very next concept.

#### The one-line rule
Multiply a word's Query and another word's Key slot by slot, add the results, and one number falls out. That number is the **attention score** between the two words.

%%% formula
expr: score(word i → word j) = Q_i · K_j
note: Q_i = word i's question, K_j = word j's label. The dot means multiply slot-by-slot and add. Big score = strong match; word i should pay attention to word j.
%%%

Do that for every pair and you get a whole grid of scores: who matches whom, and how strongly. For a sentence of `n` words the grid is `n × n` — every word rating every word, one glance each.

#### Now watch one score get built
Here's the slip-and-label comparison drawn as numbers, using our real sentence. The four slots of these cards happen to stand for "water", "place", "money", "action" — so you can read the arithmetic as meaning, not just digits.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="A dot product built slot by slot. Bank's Query of four numbers sits above river's Key of four numbers. Each pair is multiplied giving four products, which stack into one long bar labelled score 5.4, a strong match. On the right, crossed's Key is about the same overall size but points a different way, so its products stack into a tiny bar, score 0.4, a weak match."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One score = multiply slot by slot, then add the pile</text><text x="14" y="42" fill="#9A938A" font-size="8.5">slot:</text><text x="52" y="42" text-anchor="middle" fill="#9A938A" font-size="8.5">water</text><text x="100" y="42" text-anchor="middle" fill="#9A938A" font-size="8.5">place</text><text x="148" y="42" text-anchor="middle" fill="#9A938A" font-size="8.5">money</text><text x="196" y="42" text-anchor="middle" fill="#9A938A" font-size="8.5">action</text><text x="14" y="62" fill="#5E5191" font-weight="bold" font-size="9">Q</text><rect x="30" y="50" width="44" height="18" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="52" y="63" text-anchor="middle" fill="#5E5191">1.0</text><rect x="78" y="50" width="44" height="18" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="100" y="63" text-anchor="middle" fill="#5E5191">0.5</text><rect x="126" y="50" width="44" height="18" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="148" y="63" text-anchor="middle" fill="#5E5191">0</text><rect x="174" y="50" width="44" height="18" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="196" y="63" text-anchor="middle" fill="#5E5191">0</text><text x="228" y="63" fill="#6B645E" font-size="8.5">"bank" asks</text><text x="14" y="90" fill="#8A6D3B" font-weight="bold" font-size="9">K</text><rect x="30" y="78" width="44" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="52" y="91" text-anchor="middle" fill="#8A6D3B">4.4</text><rect x="78" y="78" width="44" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="100" y="91" text-anchor="middle" fill="#8A6D3B">2.0</text><rect x="126" y="78" width="44" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="148" y="91" text-anchor="middle" fill="#8A6D3B">0</text><rect x="174" y="78" width="44" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="196" y="91" text-anchor="middle" fill="#8A6D3B">0</text><text x="228" y="91" fill="#6B645E" font-size="8.5">"river" advertises</text><line x1="52" y1="98" x2="52" y2="110" stroke="#B8AEA2"/><line x1="100" y1="98" x2="100" y2="110" stroke="#B8AEA2"/><line x1="148" y1="98" x2="148" y2="110" stroke="#B8AEA2"/><line x1="196" y1="98" x2="196" y2="110" stroke="#B8AEA2"/><text x="14" y="126" fill="#276b45" font-size="10">×</text><rect x="30" y="114" width="44" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="52" y="127" text-anchor="middle" fill="#276b45">4.4</text><rect x="78" y="114" width="44" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="127" text-anchor="middle" fill="#276b45">1.0</text><rect x="126" y="114" width="44" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="148" y="127" text-anchor="middle" fill="#276b45">0</text><rect x="174" y="114" width="44" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="196" y="127" text-anchor="middle" fill="#276b45">0</text><text x="124" y="152" text-anchor="middle" fill="#6B645E" font-size="9">stack the four products into one pile ↓</text><rect x="30" y="160" width="188" height="20" rx="3" fill="#FDF9F3" stroke="#2D8B55" stroke-width="2"/><rect x="30" y="160" width="153" height="20" fill="#2D8B55" opacity="0.35"/><rect x="183" y="160" width="35" height="20" fill="#2D8B55" opacity="0.75"/><text x="124" y="198" text-anchor="middle" fill="#276b45" font-weight="bold" font-size="11">score = 4.4 + 1.0 = 5.4</text><text x="124" y="214" text-anchor="middle" fill="#276b45" font-size="9">strong match</text><line x1="252" y1="34" x2="252" y2="228" stroke="#E5DFD6"/><text x="392" y="46" text-anchor="middle" fill="#9A938A" font-size="9">same Query, and "crossed"'s Key is about the</text><text x="392" y="58" text-anchor="middle" fill="#9A938A" font-size="9">same overall size — but it points a different way</text><text x="266" y="91" fill="#8A6D3B" font-size="9">K</text><rect x="282" y="78" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="304" y="91" text-anchor="middle" fill="#9A938A">0</text><rect x="330" y="78" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="352" y="91" text-anchor="middle" fill="#9A938A">0.8</text><rect x="378" y="78" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="400" y="91" text-anchor="middle" fill="#9A938A">0</text><rect x="426" y="78" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="448" y="91" text-anchor="middle" fill="#9A938A">4.0</text><rect x="282" y="114" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="304" y="127" text-anchor="middle" fill="#9A938A">0</text><rect x="330" y="114" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="352" y="127" text-anchor="middle" fill="#9A938A">0.4</text><rect x="378" y="114" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="400" y="127" text-anchor="middle" fill="#9A938A">0</text><rect x="426" y="114" width="44" height="18" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="448" y="127" text-anchor="middle" fill="#9A938A">0</text><text x="376" y="152" text-anchor="middle" fill="#6B645E" font-size="9">its big 4.0 hits a slot "bank" isn't asking about → dies</text><rect x="282" y="160" width="188" height="20" rx="3" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="2"/><rect x="282" y="160" width="14" height="20" fill="#B8AEA2"/><text x="376" y="198" text-anchor="middle" fill="#9A938A" font-weight="bold" font-size="11">score = 0.4</text><text x="376" y="214" text-anchor="middle" fill="#9A938A" font-size="9">weak match</text><text x="260" y="242" text-anchor="middle" fill="#6B645E" font-size="9">both bars drawn on the same scale — the pile of products is what separates them</text></g></svg>
%%%

%%% steps
step: Line the two lists up slot against slot.
why: This only works because Q and K were built the same length — the rule you carried over from the last concept, now earning its keep.
step: Multiply each pair: `1.0×4.4`, `0.5×2.0`, `0×0`, `0×0`.
why: When both numbers in a slot are big, the product is big. When either is near zero, the product dies. So each slot votes on "do we agree here?"
step: Add the pile up: `4.4 + 1.0 + 0 + 0 = 5.4`.
why: Therefore agreement in a few strong slots adds up to a big total — which means one number now summarizes the whole comparison.
step: Do it again with "crossed", whose Key is about the same overall size but points elsewhere — the pile stays tiny, `0.4`.
why: That gap, 5.4 against 0.4, IS the verdict. And notice *why* crossed lost: its loudest number (4.0, "I'm an action") landed in a slot "bank" wasn't asking about, so it got multiplied by zero.
%%%

%%% insight
Notice what just happened: the score never looked at spelling, at word order, or at how far apart the two words sat. It only felt whether the two number-lists *agree, slot by slot*. That's why this one little sum can catch "bank ↔ river" in English, in Chinese, in code, or in a protein sequence. One dot product, and the same machinery works everywhere. That is why the whole field bet on it.
%%%

So far: one number per pair. Now stop reading and go **drive it yourself** — change "bank"'s question and watch who wins.

%%% viz src=../../viz/qkv-score-explorer.html title="Score explorer" caption="Drag the four sliders and watch which word wins the match."
%%%

**One thing to try:** drag *money* up to 1.0 and *water* down to 0. "bank" now asks about money instead of water — and the winner flips from "river" to "bank" itself. You just changed a word's meaning by changing its question.

There's also a sneaky asymmetry hiding here, and it's a feature. Because `W_Q` and `W_K` are *different* grids, "A looks at B" and "B looks at A" are two different numbers: in our sentence "bank → river" scores **5.4** while "river → bank" scores only **1.2**. An adjective can lean hard on its noun while the noun barely glances back. Tie the two grids together (`W_Q = W_K`) and the grid of scores turns perfectly symmetric — every "who points at whom" vanishes. Keeping them untied is what lets attention encode direction.

%%% demo id=score label="score bank against every word"
predict: in "the animal crossed the river bank", "bank" is asking "what kind of place am I?" Which word scores highest — "river", "the", or "crossed"? Commit to a guess, then reveal.
code: raw_scores = K @ Q[bank]          # bank's Query vs every Key
out: the=0.2   animal=0.2   crossed=0.4
out: the=0.2   river=5.4    bank=1.0
take: <b>"river" scores 5.4 — far above everyone else.</b> "bank"'s question ("what kind of place am I?") lines up strongest with "river"'s label. The filler words barely register (0.2). Just like your eyes jumping to the volcano book, "bank"'s attention is being pulled toward "river". No word has been read yet — these are only the match scores, and they are exactly the numbers `experiment.py` prints.
%%%

You just turned "how relevant are these two words?" into a single number, for every pair. But those raw numbers have one bad habit — and that's the next puzzle.

@@@ concept id=c4 tag="The volume knob" title="A puzzle: when the scores get too loud" gotit="Got the scaling fix"
Here's a puzzle that trips up real engineers, and the fix is one of the neatest tricks in the whole transformer. The next station on the belt will turn these scores into *shares* — "listen 70% to this word, 5% to that one." That share-out step (its name is **softmax**, and you build it in the very next concept) cares about the *gaps* between scores. So if the scores come out huge, the gaps come out huge too, and the biggest score takes essentially everything.

Think of the **volume knob** on a speaker. At a sensible level you hear the whole song — drums, bass, singer. Crank it to maximum and the sound *clips*: it turns harsh and one loud part drowns out everything else. Attention has the same danger. A dot product adds one piece per slot, so a card with 64 or 128 slots (typical in a real model) produces a sum far bigger than our tidy 5.4. Big scores, huge gaps, and the share-out step hands the whole pizza to one word. The attention has been turned up too loud — it can only hear one voice.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two speaker volume knobs with the sound wave each produces. On the left the knob is set to a sensible level and the wave is smooth and rounded, so every instrument is audible. On the right the knob is cranked to maximum and the wave is squashed flat at the top and bottom, which is clipping: one loud part drowns out the rest."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A volume knob: sensible vs cranked</text><circle cx="130" cy="52" r="20" fill="#FDF9F3" stroke="#2D8B55" stroke-width="2"/><line x1="130" y1="52" x2="115" y2="39" stroke="#2D8B55" stroke-width="2.5"/><circle cx="106" cy="66" r="1.8" fill="#B8AEA2"/><circle cx="106" cy="38" r="1.8" fill="#B8AEA2"/><circle cx="154" cy="38" r="1.8" fill="#B8AEA2"/><circle cx="154" cy="66" r="1.8" fill="#B8AEA2"/><text x="130" y="90" text-anchor="middle" fill="#276b45" font-size="10">volume: sensible</text><path d="M40 126 Q55 104 70 126 Q85 148 100 126 Q115 104 130 126 Q145 148 160 126 Q175 106 190 126 Q205 146 220 126" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="130" y="166" text-anchor="middle" fill="#276b45" font-size="9">the whole song survives — drums, bass, AND singer</text><line x1="260" y1="30" x2="260" y2="176" stroke="#E5DFD6"/><circle cx="390" cy="52" r="20" fill="#FDF9F3" stroke="#C93B3B" stroke-width="2"/><line x1="390" y1="52" x2="405" y2="39" stroke="#C93B3B" stroke-width="2.5"/><circle cx="366" cy="66" r="1.8" fill="#B8AEA2"/><circle cx="366" cy="38" r="1.8" fill="#B8AEA2"/><circle cx="414" cy="38" r="1.8" fill="#C93B3B"/><circle cx="414" cy="66" r="1.8" fill="#B8AEA2"/><text x="425" y="34" fill="#C93B3B" font-size="9">MAX</text><text x="390" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">volume: cranked</text><path d="M300 126 L300 100 L326 100 L326 152 L352 152 L352 100 L378 100 L378 152 L404 152 L404 100 L430 100 L430 152 L456 152 L456 126 L480 126" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="390" y="166" text-anchor="middle" fill="#C93B3B" font-size="9">it clips — one loud part drowns out the rest</text></g></svg>
%%%

**Gets right:** too much gain wrecks the mix — one part drowns the rest and the detail is lost. Exactly what oversized scores do downstream.
**Breaks down:** a volume knob turns *everything* up equally. Here the scores grew for one specific reason — the cards got longer — so the fix isn't "turn everything down", it's "divide by an amount that exactly cancels the card length."

Want to work it out yourself before I tell you? The ladder below nudges without spoiling.

%%% hint
t1: Don't worry about any formula yet. Just ask one thing: if a score is built by adding up MORE little pieces, does the total tend to get bigger or smaller?
t2: A dot product adds one piece per slot — but the pieces partly cancel each other out, so the total does NOT grow as fast as the slot count. A 4-slot card lands near 2. A 100-slot card lands near 10, not 100. So the typical total grows like the SQUARE ROOT of the number of slots.
t3: Long cards make big scores; big scores make the share-out step pick one winner and ignore everyone else; so we divide every score by √(card length) first to keep the scores calm.
%%%

%%% insight
The reason this bug is famous is that it's **silent**. Nothing crashes. No error appears. Your model trains, the loss drops a little, everything looks fine — while every word is stubbornly listening to exactly one neighbour and ignoring the rest of the sentence. You'd never find it by reading the code. You find it by *looking at the shares*, which is why we keep drawing them.
%%%

#### The named fix: divide by √(card length)
The remedy is delightfully simple, and it has a name: [[scaled dot-product attention||The standard attention: divide every score by the square root of the card length (√d_k) before the share-out step, so long cards don't blow the scores up.]].

Before sharing out, divide every score by the **square root of the card length**. The length of each Query and Key is written `d_k`, so the divisor is `√d_k`.

%%% formula
expr: score = (Q · K) / √d_k
note: Q·K = the raw match score. d_k = the length of each Query/Key card. √d_k = its square root, the exact divisor that cancels the growth. Divide FIRST, share out second.
%%%

Longer cards make bigger scores *and* a bigger divisor — therefore the two cancel, and the scores stay a sensible size no matter how long the cards get. That single division is the "scaled" in the official name of attention. It isn't a hack; it's the standard.

So far: we know the disease and the cure. Now let's turn the knob down on a realistic card and watch the mix come back.

#### The volume knob, turned back down
Take a real-sized card: `d_k = 64`, so the divisor is `√64 = 8`. Three words come in with raw scores `[24, 8, 0]` — the kind of spread a 64-slot card produces. Left panel: no scaling. Right panel: after dividing by 8. Each panel's three bars are one set of shares, so each panel adds up to 1.00.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two bar charts side by side showing the same three words. On the left, no scaling: the first bar is completely full at 1.00 and the other two bars are invisible at 0.00. An arrow labelled divide each score by eight points right. On the right, scaled: the first bar is 0.84, the second 0.11, the third 0.04 — a clear winner but all three visible. Each panel's three shares add up to 1.00."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">raw [24, 8, 0], d_k = 64 → divide by √64 = 8</text><text x="110" y="38" text-anchor="middle" fill="#C93B3B" font-size="10">no scaling → it clips</text><rect x="40" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="40" y="50" width="40" height="100" fill="#C93B3B"/><text x="60" y="164" text-anchor="middle" fill="#6B645E" font-size="9">1.00</text><rect x="95" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="95" y="149.8" width="40" height="0.2" fill="#C93B3B"/><text x="115" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.00</text><rect x="150" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="150" y="149.9" width="40" height="0.1" fill="#C93B3B"/><text x="170" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.00</text><text x="115" y="178" text-anchor="middle" fill="#9A938A" font-size="8.5">two words have been deleted</text><g><line x1="210" y1="100" x2="298" y2="100" stroke="#5E5191" stroke-width="2"/><polygon points="298,95 310,100 298,105" fill="#5E5191"/><text x="254" y="90" text-anchor="middle" fill="#5E5191" font-size="9">÷ 8</text></g><text x="405" y="38" text-anchor="middle" fill="#276b45" font-size="10">scaled → balanced</text><rect x="330" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="330" y="66" width="40" height="84" fill="#2D8B55"/><text x="350" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.84</text><rect x="385" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="385" y="139" width="40" height="11" fill="#2D8B55"/><text x="405" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.11</text><rect x="440" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="440" y="146" width="40" height="4" fill="#2D8B55"/><text x="460" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.04</text><text x="405" y="178" text-anchor="middle" fill="#276b45" font-size="8.5">0.84 + 0.11 + 0.04 ≈ 1.00 — all three audible</text></g></svg>
%%%

%%% steps
step: The problem — raw `[24, 8, 0]` goes straight to the share-out step.
why: A gap of 16 between the top two is enormous once it gets shared out, so the shares come back `[1.00, 0.00, 0.00]`. Two words have effectively been deleted.
step: The fix — divide every score by `√64 = 8`.
why: Which turns `[24, 8, 0]` into `[3, 1, 0]`. Same ranking, same winner — just quieter. That's the volume knob coming back down.
step: The payoff — the shares come back `[0.84, 0.11, 0.04]`.
why: The winner still clearly leads, but the other two are audible again. Therefore the word can blend in more than one neighbour.
step: The bonus nobody mentions — learning stays healthy.
why: Those crushed 0.00 shares also crush the learning signal that flows back through them, so the model would barely improve. One division protects both the blend AND the training.
%%%

%%% demo id=scale label="shares before vs after scaling (d_k = 64)"
predict: before scaling, the winner takes 1.00 and the other two vanish completely. After dividing every score by 8, do the other two get ANY meaningful share back — or is the damage permanent? Guess, then reveal.
code: raw = [24, 8, 0];  d_k = 64;  share(raw)  vs  share(raw / sqrt(d_k))
out: NO SCALE  ->  [1.00, 0.00, 0.00]
out: scaled scores [3, 1, 0]
out: SCALED    ->  [0.84, 0.11, 0.04]
take: <b>Without scaling it clips: the winner takes everything and the others vanish.</b> Dividing each raw score by √d_k = √64 = 8 turns [24, 8, 0] into [3, 1, 0], and the shares come back as [0.84, 0.11, 0.04] — the winner still leads, but the others are heard. Quietly, this also keeps learning healthy, because near-zero shares would have stalled training. One little division, big payoff. That's why every real transformer scales.
%%%

#### Our own sentence, scaled
Our toy cards are only `d_k = 4` long, so the divisor is just `√4 = 2` and the effect is gentle — but the division is always there, and we'll carry the scaled numbers from here on.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="The six raw scores of our sentence on the left — the 0.2, animal 0.2, crossed 0.4, the 0.2, river 5.4, bank 1.0 — with an arrow labelled divide by root 4 equals 2 leading to the scaled scores on the right: 0.1, 0.1, 0.2, 0.1, 2.7 and 0.5. The ranking is unchanged."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="11">our sentence: raw scores ÷ √4 = scaled scores (same ranking, calmer numbers)</text><text x="66" y="38" text-anchor="middle" fill="#8A6D3B" font-size="10">raw</text><text x="454" y="38" text-anchor="middle" fill="#276b45" font-size="10">scaled</text><text x="14" y="56" fill="#6B645E">the</text><text x="90" y="56" text-anchor="end" fill="#8A6D3B">0.2</text><text x="430" y="56" text-anchor="end" fill="#276b45">0.1</text><text x="14" y="74" fill="#6B645E">animal</text><text x="90" y="74" text-anchor="end" fill="#8A6D3B">0.2</text><text x="430" y="74" text-anchor="end" fill="#276b45">0.1</text><text x="14" y="92" fill="#6B645E">crossed</text><text x="90" y="92" text-anchor="end" fill="#8A6D3B">0.4</text><text x="430" y="92" text-anchor="end" fill="#276b45">0.2</text><text x="14" y="110" fill="#6B645E">the</text><text x="90" y="110" text-anchor="end" fill="#8A6D3B">0.2</text><text x="430" y="110" text-anchor="end" fill="#276b45">0.1</text><text x="14" y="128" fill="#276b45" font-weight="bold">river</text><text x="90" y="128" text-anchor="end" fill="#276b45" font-weight="bold">5.4</text><text x="430" y="128" text-anchor="end" fill="#276b45" font-weight="bold">2.7</text><text x="14" y="146" fill="#6B645E">bank</text><text x="90" y="146" text-anchor="end" fill="#8A6D3B">1.0</text><text x="430" y="146" text-anchor="end" fill="#276b45">0.5</text><rect x="104" y="44" width="312" height="106" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="130" y1="97" x2="368" y2="97" stroke="#5E5191" stroke-width="2"/><polygon points="368,91 382,97 368,103" fill="#5E5191"/><text x="252" y="88" text-anchor="middle" fill="#5E5191" font-size="11">÷ √d_k = ÷ √4 = ÷ 2</text><text x="252" y="118" text-anchor="middle" fill="#9A938A" font-size="9">every score, same divisor</text></g></svg>
%%%

!!! c-info 🪜
<b>Optional (skippable) — why √d_k exactly, for the curious.</b> You never need this to use the fix. When you add up `d_k` roughly-independent little products, some are positive and some negative, so they partly cancel — and the typical size of the total grows like `√d_k`, not like `d_k`. (That's a standard fact about adding up random numbers.) So raw scores swell by about `√d_k` as cards get longer, and dividing by `√d_k` cancels that growth precisely. That's why the paper picked that exact divisor and not, say, `d_k` itself. The full derivation is Day 3's job.
!!!

One honest note so nothing surprises you later: dividing by `√d_k` cancels the growth that comes from **card length**. It does *not* normalize how big the individual vectors are — a genuinely long Query still produces bigger scores. Nothing today removes that, and nothing needs to.

@@@ concept id=c5 tag="Sharing attention" title="Turning scores into a 'how much to listen' recipe" gotit="Got softmax weights"
We have calm, scaled scores now: river 2.7, bank 0.5, crossed 0.2, and 0.1 for the fillers. But what do you *do* with a 2.7? It isn't a percentage. It doesn't say "listen this much." We need to turn the pile of scores into a clean recipe of shares.

Think of having exactly **one pizza** to share at a table, and you must give *everyone* a slice: no slice can be negative (you can't take pizza away), and all the slices together must make exactly one whole pizza. The strongest match gets the biggest slice, the fillers get slivers, everyone gets *something*, and it all adds up to one.

That's the recipe we want for our scores. The tool that does it has a name — [[softmax||A recipe that turns any list of scores into shares that are all positive and add up to 1 — like slicing one pizza. A bigger score gets a bigger slice.]] — and its whole job is "make these scores into positive shares that sum to 1."

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="One pizza of attention cut into six slices, one per word in the sentence. River takes the huge slice at 0.71, bank 0.08, crossed 0.06, animal 0.05, and the two the words 0.05 each. Together the six shares add up to exactly 1.00, and a matching bar chart on the right repeats the same six shares to scale."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One pizza of attention → a slice for every word</text><path d="M130 110 L196.0 110.0 A66 66 0 1 1 113.6 46.1 Z" fill="#2D8B55" opacity="0.75" stroke="#FDF9F3" stroke-width="1.5"/><path d="M130 110 L113.6 46.1 A66 66 0 0 1 146.4 46.1 Z" fill="#C99A12" stroke="#FDF9F3" stroke-width="1.5"/><path d="M130 110 L146.4 46.1 A66 66 0 0 1 168.8 56.6 Z" fill="#7C6DAA" stroke="#FDF9F3" stroke-width="1.5"/><path d="M130 110 L168.8 56.6 A66 66 0 0 1 183.4 71.2 Z" fill="#B8AEA2" stroke="#FDF9F3" stroke-width="1.5"/><path d="M130 110 L183.4 71.2 A66 66 0 0 1 192.8 89.6 Z" fill="#D8D2C8" stroke="#FDF9F3" stroke-width="1.5"/><path d="M130 110 L192.8 89.6 A66 66 0 0 1 196.0 110.0 Z" fill="#EFE9DF" stroke="#FDF9F3" stroke-width="1.5"/><circle cx="130" cy="110" r="66" fill="none" stroke="#C99A12" stroke-width="1.5"/><text x="106" y="134" text-anchor="middle" fill="#FFFFFF" font-weight="bold" font-size="11">river</text><text x="106" y="149" text-anchor="middle" fill="#FFFFFF" font-size="10">0.71</text><text x="296" y="55" fill="#6B645E" font-size="10">river</text><rect x="352" y="44" width="120" height="12" fill="#2D8B55"/><text x="478" y="55" fill="#6B645E" font-size="9">.71</text><text x="296" y="75" fill="#6B645E" font-size="10">bank</text><rect x="352" y="64" width="14" height="12" fill="#C99A12"/><text x="478" y="75" fill="#6B645E" font-size="9">.08</text><text x="296" y="95" fill="#6B645E" font-size="10">crossed</text><rect x="352" y="84" width="10" height="12" fill="#7C6DAA"/><text x="478" y="95" fill="#6B645E" font-size="9">.06</text><text x="296" y="115" fill="#6B645E" font-size="10">animal</text><rect x="352" y="104" width="8" height="12" fill="#B8AEA2"/><text x="478" y="115" fill="#6B645E" font-size="9">.05</text><text x="296" y="135" fill="#6B645E" font-size="10">the</text><rect x="352" y="124" width="8" height="12" fill="#D8D2C8"/><text x="478" y="135" fill="#6B645E" font-size="9">.05</text><text x="296" y="155" fill="#6B645E" font-size="10">the</text><rect x="352" y="144" width="8" height="12" fill="#EFE9DF" stroke="#E5DFD6"/><text x="478" y="155" fill="#6B645E" font-size="9">.05</text><text x="388" y="180" text-anchor="middle" fill="#276b45" font-size="10">every slice ≥ 0, and the six add to 1.00</text><text x="260" y="200" text-anchor="middle" fill="#6B645E" font-size="9">0.71 + 0.08 + 0.06 + 0.05 + 0.05 + 0.05 = 1.00 — one whole pizza, nothing left over</text></g></svg>
%%%

**Gets right:** you split one whole thing into shares, none negative, adding to exactly one — precisely what softmax does to the scores.
**Breaks down:** pizza slices are cut by hand in any size you like. Softmax's slices are decided by a fixed rule — bigger scores get *exponentially* bigger slices. You don't choose the slices; the scores do.

#### The two promises softmax makes
- **Every share is positive.** No word gets negative attention — you can only listen *to* a word, never anti-listen.
- **The shares add up to 1.** The whole attention budget is exactly one pizza. So a share of `0.71` means "71% of my attention on this word."

You met softmax back in Module 2, as the thing that turns scores into a probability-like list. Same tool, same job, new place.

%%% insight
Why do those two promises matter so much? Because they turn a *comparison* into a *decision*. Raw scores can only be ranked. Shares that sum to 1 can be **spent** — poured into a blend, in exact proportion. That's the hinge of the whole day: scores tell you who matched, shares tell you what to do about it.
%%%

So far: scaled scores in, pizza slices out. Now go play — drag the sliders and try to break the promise.

%%% svg
<svg id="sm-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive softmax. Drag three scores for the words river, animal and the, and watch the attention shares, drawn as bars, resize. They always stay positive and add up to exactly one whole."><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="40" y="26" width="440" height="140" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="150" x2="460" y2="150" stroke="#EFE9DF"/><rect id="sm-b0" x="110" y="61" width="60" height="89" fill="#2D8B55"/><text id="sm-t0" x="140" y="53" fill="#1a5c38">.79</text><text x="140" y="164" fill="#6B645E">river</text><rect id="sm-b1" x="230" y="137" width="60" height="13" fill="#2A7B9B"/><text id="sm-t1" x="260" y="129" fill="#1F6280">.12</text><text x="260" y="164" fill="#6B645E">animal</text><rect id="sm-b2" x="350" y="141" width="60" height="9" fill="#C99A12"/><text id="sm-t2" x="380" y="133" fill="#9A7208">.09</text><text x="380" y="164" fill="#6B645E">the</text><text x="260" y="184" fill="#1a5c38" font-size="10">the three shares always add up to one whole pizza (1.00)</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>river <input id="sm-s0" type="range" min="-2" max="5" step="0.1" value="2.7" style="accent-color:#2D8B55;vertical-align:middle"></label>
<label>animal <input id="sm-s1" type="range" min="-2" max="5" step="0.1" value="0.8" style="accent-color:#2A7B9B;vertical-align:middle"></label>
<label>the <input id="sm-s2" type="range" min="-2" max="5" step="0.1" value="0.5" style="accent-color:#C99A12;vertical-align:middle"></label>
</div>
<div id="sm-out" style="margin-top:6px;color:#2C2A28">scores [2.7, 0.8, 0.5] → shares <b>0.79 / 0.12 / 0.09</b> (sum 1.00) — when one word pulls ahead, it grabs most of the pizza.</div>
</div>
<script>(function(){
  var s0=document.getElementById('sm-s0');if(!s0)return;
  var s1=document.getElementById('sm-s1'),s2=document.getElementById('sm-s2'),out=document.getElementById('sm-out');
  var BASE=150,MAXH=112;
  function paint(){
    var v=[+s0.value,+s1.value,+s2.value];
    var ex=v.map(function(x){return Math.exp(x);});var Z=ex[0]+ex[1]+ex[2];
    var sh=ex.map(function(e){return e/Z;});
    for(var i=0;i<3;i++){var h=Math.max(2,sh[i]*MAXH);var b=document.getElementById('sm-b'+i),t=document.getElementById('sm-t'+i);
      if(b){b.setAttribute('y',(BASE-h).toFixed(1));b.setAttribute('height',h.toFixed(1));}
      if(t){t.setAttribute('y',(BASE-h-8).toFixed(1));t.textContent=sh[i].toFixed(2);}}
    out.innerHTML='scores ['+v.map(function(x){return x.toFixed(1);}).join(', ')+'] → shares <b>'+sh.map(function(x){return x.toFixed(2);}).join(' / ')+'</b> (sum '+(sh[0]+sh[1]+sh[2]).toFixed(2)+')';
  }
  [s0,s1,s2].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

**Two things to try.** Set all three scores nearly equal — the pizza splits almost evenly and attention is basically shrugging. Then drag one score far ahead and watch it swallow the pie. Try as hard as you like: the sum stays glued to 1.00.

#### How the slicing actually works, one rung at a time
Three little moves, and softmax is fully explained. Nothing here is harder than "make positive, then share out."

%%% steps
step: Make every score positive.
why: Scores can be negative, and a negative slice makes no sense. Raising each score as a power of `e` (about 2.718) is the standard way to force positivity — and it keeps the ranking, because a bigger score always gives a bigger result.
step: Add those positive numbers into one total pile.
why: That total is your "whole pizza". It exists only so you have something to divide by.
step: Divide each word's positive number by the total.
why: Which means each word now holds a *fraction* of the pile — and fractions of one pile always add back to exactly 1. Both promises, satisfied by construction.
step: The winner's lead gets amplified, not just preserved.
why: That's the "exponentially bigger slice" from the pizza note — and it's exactly why a clear match becomes a confident decision instead of a mild preference.
%%%

The result has a name you'll hear constantly: these shares are the [[attention weights||The shares softmax produces — one per word, all positive, adding to 1. They say how much of each word to mix in.]]. If you can say "attention weights are the softmax of the scaled scores," you can read most transformer code.

%%% demo id=softmax label="scaled scores → attention shares"
predict: "river" scored 2.7 after scaling and everyone else came in at 1.0 or below. Will "river" end up with roughly half the pizza, about three-quarters, or nearly all of it? Guess, then reveal.
code: weights = softmax([0.1, 0.1, 0.2, 0.1, 2.7, 0.5])   # the scaled scores
out: the=0.05   animal=0.05   crossed=0.06
out: the=0.05   river=0.71    bank=0.08
out: sum = 1.00
take: <b>The scaled scores became clean shares that add to exactly 1.00.</b> "river"'s 2.7 turned into a fat 0.71 slice — 71% of "bank"'s attention lands on "river". Everyone still gets a sliver, but the winner is clear. These shares are the <b>attention weights</b>: the recipe for how much of each word to mix in next. (`experiment.py` prints 0.706 before rounding.)
%%%

@@@ concept id=c6 tag="Blending the room" title="Mix the Values by the recipe → a fresh word" gotit="Got the weighted sum"
We have the recipe — the shares that say how much to listen to each word. The last step is the payoff: actually *use* it to build a brand-new, context-aware version of the word. And here's the beautiful part: we mix the **Values**, not the words themselves.

Think of making a [[smoothie||A drink blended from several fruits in different amounts, following a recipe.]] from a recipe. The recipe says "71% banana, 8% strawberry, a splash of everything else." You don't dump in whole fruit — you pour each fruit in *by its share* and blend. Out comes one new drink that tastes mostly of banana, a little of strawberry, a hint of the rest.

Attention does exactly this. The **shares** are the recipe. The **Values** are the fruits — each word's content. Pour in each word's Value scaled by its share, add them up, and out comes one fresh vector: a new "bank" that is mostly river, a little itself, a hint of the rest. That new vector *knows it means river-bank.*

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A smoothie blender. Value vectors are poured in by their shares: river's Value at 0.71 is the big pour, bank's Value at 0.08 a small pour, others a splash. They blend into one new output vector on the right labelled the new context-aware bank, now river flavoured."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Pour each Value in by its share → blend → one new word</text><rect x="24" y="40" width="140" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="36" y="56" fill="#276b45" font-size="10">V(river) × 0.71</text><rect x="24" y="70" width="140" height="24" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="36" y="86" fill="#8A6D3B" font-size="10">V(bank) × 0.08</text><rect x="24" y="100" width="140" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="36" y="116" fill="#6B645E" font-size="10">V(crossed) × 0.06</text><rect x="24" y="130" width="140" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="36" y="146" fill="#9A938A" font-size="10">…the rest × splashes</text><line x1="164" y1="52" x2="250" y2="96" stroke="#2D8B55" stroke-width="3"/><line x1="164" y1="82" x2="250" y2="100" stroke="#C99A12" stroke-width="1.5"/><line x1="164" y1="112" x2="250" y2="104" stroke="#B8AEA2"/><line x1="164" y1="142" x2="250" y2="108" stroke="#B8AEA2"/><path d="M256 78 L316 78 L306 150 Q286 166 266 150 Z" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="286" y="112" text-anchor="middle" fill="#5E5191" font-size="9">blend</text><text x="352" y="90" fill="#B8AEA2" font-size="16">→</text><rect x="380" y="76" width="126" height="52" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="443" y="96" text-anchor="middle" fill="#5E5191" font-weight="bold" font-size="10">new "bank"</text><text x="443" y="111" text-anchor="middle" fill="#2C2A28" font-size="10">[0.74, 0.15]</text><text x="443" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">watery 0.74, money 0.15</text><text x="260" y="196" text-anchor="middle" fill="#276b45" font-size="10">output = 0.71·V(river) + 0.08·V(bank) + … — a weighted sum of Values</text></g></svg>
%%%

**Gets right:** you combine ingredients *in proportion* to a recipe and get one new thing that tastes mostly of the biggest ingredient — that's a weighted sum, exactly what attention outputs.
**Breaks down:** a smoothie loses its separate fruits forever. The model keeps every word's original Value around and re-blends fresh at every layer — so the mixing happens over and over, not just once.

%%% insight
This "compare, then take more of the best match" move is quietly everywhere. It's close to how your phone decides which faces in your camera roll belong to the same person, and how a music app picks the next song — compare a question to a bunch of labels, then lean toward what matched. Attention's twist is that it doesn't pick just one winner: it *blends* everybody, in exact proportion. Same instinct, better arithmetic.
%%%

#### The one-line rule
Each word's **Value** multiplied by its **share**, all added up. Bigger share → more of that word's Value in the mix. "Weighted" just means "each thing counts by its share."

%%% formula
expr: output = Σ (share_j × V_j)
note: share_j = how much to listen to word j (from softmax, all adding to 1). V_j = word j's Value (its content). Multiply each Value by its share, add them up → one new vector. "Σ" just means "add up over all words j."
%%%

#### Pour, pour, pour — watch the glass fill
Here's the same blend drawn as pours stacking up. Every block is drawn to the same scale, so a block's width IS its share.

%%% svg
<svg viewBox="0 0 520 224" role="img" aria-label="Four pours drawn to one scale. River at 0.71 is a wide block, bank at 0.08 is a small block, crossed at 0.06 slightly smaller, and the remaining three filler words at 0.16 together a block in between. To the right, the finished output vector is drawn as one filled glass whose colour is dominated by the river pour, labelled 0.74 and 0.15."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Each pour is share × Value — they stack into one output</text><text x="60" y="42" fill="#276b45">0.71 × V(river) = 0.71 × [1.0, 0.1]</text><rect x="60" y="48" width="185" height="30" rx="3" fill="#2D8B55" opacity="0.75"/><text x="152" y="68" text-anchor="middle" fill="#FFFFFF">the big pour</text><text x="60" y="98" fill="#8A6D3B">0.08 × V(bank) = 0.08 × [0.1, 1.0]</text><rect x="60" y="104" width="21" height="22" rx="3" fill="#C99A12" opacity="0.85"/><text x="60" y="146" fill="#5E5191">0.06 × V(crossed) = 0.06 × [0.24, 0.04]</text><rect x="60" y="152" width="16" height="22" rx="3" fill="#7C6DAA" opacity="0.7"/><text x="60" y="194" fill="#9A938A">0.16 × the other three words</text><rect x="60" y="200" width="42" height="16" rx="3" fill="#D8D2C8"/><text x="266" y="212" fill="#9A938A" font-size="8.5">same scale on every block</text><line x1="286" y1="34" x2="286" y2="216" stroke="#E5DFD6"/><text x="404" y="42" text-anchor="middle" fill="#5E5191">add every pour → one new vector</text><path d="M356 56 L452 56 L442 176 Q404 194 366 176 Z" fill="#FDF9F3" stroke="#5E5191" stroke-width="1.5"/><path d="M362 92 L446 92 L442 176 Q404 194 366 176 Z" fill="#2D8B55" opacity="0.6"/><path d="M360 78 L448 78 L446 92 L362 92 Z" fill="#C99A12" opacity="0.55"/><path d="M358 66 L450 66 L448 78 L360 78 Z" fill="#B8AEA2" opacity="0.6"/><text x="404" y="136" text-anchor="middle" fill="#1a5c38" font-weight="bold" font-size="11">new "bank"</text><text x="404" y="152" text-anchor="middle" fill="#276b45" font-size="9">[0.74, 0.15] — river-flavoured</text><text x="404" y="206" text-anchor="middle" fill="#6B645E" font-size="9">the tallest pour dominates the taste</text></g></svg>
%%%

%%% steps
step: Take word j's Value and shrink it by its share — `0.71 × [1.0, 0.1]` for river.
why: The share acts like a volume dial on that word's content. A 0.05 share barely whispers; a 0.71 share almost shouts.
step: Do that for every word, including the word itself.
why: A word keeps some of its own content — which means it doesn't lose its identity, it just gets *coloured* by its neighbours. That's why "bank"'s own 0.08 slice is in the picture too.
step: Add all the shrunken Values together, slot by slot.
why: Therefore you end with one vector the same length as a Value (`d_v = 2` here) — not a longer list. The room got summarized, not stapled together.
step: Read the result: `[0.74, 0.15]` — watery 0.74, money 0.15.
why: That's the moment "bank" stops being ambiguous. Its new vector sits close to river's Value `[1.0, 0.1]` and far from its own old `[0.1, 1.0]`, so the next layer reads a *river*-bank.
%%%

%%% insight
Here's the quiet genius, easy to skate past: we blend the **Values**, never the Keys. The Key's only job was to *win the match*; the Value's only job is to *carry the content*. Because they're separate vectors, a word can be found for one reason and contribute something completely different — "river" gets matched for being a watery place, then hands over its water-meaning. Fuse Key and Value into one vector and you lose that freedom instantly.
%%%

%%% demo id=blend label="blend the values by their shares"
predict: "bank"'s recipe is 71% river, 8% itself, small bits of the rest. Will the new "bank" land closer to river's Value `[1.0, 0.1]` or to its own old Value `[0.1, 1.0]`? Guess, then reveal.
code: output = sum(share_j * V_j for every word j)
out: V(the)=[0.02,0.02]  V(animal)=[0.22,0.02]  V(crossed)=[0.24,0.04]
out: V(the)=[0.02,0.02]  V(river)=[1.00,0.10]   V(bank)=[0.10,1.00]
out: shares  = [0.052, 0.052, 0.058, 0.052, 0.706, 0.078]
out: output  = [0.742, 0.154]
take: <b>The new "bank" is `[0.742, 0.154]` — right next to river's Value `[1.0, 0.1]`, and nowhere near its own old `[0.1, 1.0]`.</b> Every number you need is printed above, so you can check it by hand: `0.706×1.00 + 0.078×0.10 + 0.058×0.24 + 0.052×0.22 + 0.052×0.02 + 0.052×0.02 = 0.741`. (A thousandth off from 0.742 because the shares are rounded — that's rounding, not magic.) "bank" walked in confused and walked out <i>context-aware</i>: watery 0.74, money 0.15. This is what the next layer of the model reads.
%%%

Take a victory lap: you just built the entire attention move. Query asks, Key answers, the dot product scores, the divide keeps it calm, softmax shares, and a weighted sum of Values delivers a fresh word. Everything else in a transformer stacks on top of this.

@@@ concept id=c7 tag="The whole flow" title="Read the whole library visit end to end" gotit="Got the full pipeline"
You've built every piece. Now let's watch the whole thing run start to finish, so the shape of attention lives in your head as one picture. There's a lovely twist hiding in it that we'll name at the end.

Think of an **assembly line** where a word rides a conveyor belt through six stations and comes out transformed. Station 1: the word arrives as its Day-1 card. Station 2: it gets stamped with its three role-cards. Station 3: its Query is scored against every Key. Station 4: the scores get turned down to a sensible size. Station 5: they're sliced into shares. Station 6: the shares blend the Values into one fresh word. In, along the belt, out — a confused "bank" enters and a river-aware "bank" exits. And the same belt runs for *every* word at once.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="An assembly line of six stations connected by arrows: station one embeddings, station two make Q K V, station three Q dot K scores, station four scale by root d k, station five softmax into shares, station six weighted sum of V giving the output. A banner notes the belt runs for every word at the same time."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The whole attention flow, station by station</text><rect x="12" y="52" width="66" height="44" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="45" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 1</text><text x="45" y="80" text-anchor="middle" fill="#6B645E">embed</text><text x="45" y="92" text-anchor="middle" fill="#9A938A">x (Day 1)</text><text x="84" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="98" y="52" width="66" height="44" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="131" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 2</text><text x="131" y="80" text-anchor="middle" fill="#5E5191">project</text><text x="131" y="92" text-anchor="middle" fill="#5E5191">Q, K, V</text><text x="170" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="184" y="52" width="66" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="217" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 3</text><text x="217" y="80" text-anchor="middle" fill="#8A6D3B">score Q·K</text><text x="217" y="92" text-anchor="middle" fill="#8A6D3B">n × n grid</text><text x="256" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="270" y="52" width="66" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="303" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 4</text><text x="303" y="80" text-anchor="middle" fill="#8A6D3B">scale</text><text x="303" y="92" text-anchor="middle" fill="#8A6D3B">÷ √d_k</text><text x="342" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="356" y="52" width="66" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="389" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 5</text><text x="389" y="80" text-anchor="middle" fill="#276b45">softmax</text><text x="389" y="92" text-anchor="middle" fill="#276b45">→ shares</text><text x="428" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="442" y="52" width="70" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="477" y="66" text-anchor="middle" fill="#9A938A" font-size="8">STATION 6</text><text x="477" y="80" text-anchor="middle" fill="#5E5191">blend Σ·V</text><text x="477" y="92" text-anchor="middle" fill="#5E5191">= output</text><rect x="98" y="120" width="338" height="26" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text x="267" y="137" text-anchor="middle" fill="#6B645E" font-size="9">the belt runs for EVERY word at the same time — a whole sentence in one pass</text></g></svg>
%%%

**Gets right:** attention really is a fixed sequence of stations — same steps, same order, every time — and a word comes out changed at the end.
**Breaks down:** a factory belt runs one item after another. Attention runs *all* words through *all* stations at the same time, in one big batch of number-crunching.

#### Ride the belt once, naming each station
Six stations. If you can say this list out loud, you can explain attention to anyone.

%%% steps
step: Station 1 · embed — each word becomes its Day-1 card, `x`.
why: This is the only input the belt gets. Everything after it is arithmetic on these cards.
step: Station 2 · project — three learned grids (`W_Q`, `W_K`, `W_V`) make Q, K, V.
why: Because one card can't ask, advertise and share at once. Three grids in, three role-vectors out.
step: Station 3 · score — every Query meets every Key, `Q·K`.
why: Which gives an `n × n` grid of match scores for a sentence of `n` words. This is where "who is relevant to whom" gets decided.
step: Station 4 · scale — divide by `√d_k`.
why: So long cards don't make the scores too loud. This is the volume knob you installed.
step: Station 5 · share — softmax the scaled scores.
why: Therefore the scores become positive shares that add to 1 — a spendable recipe instead of a ranking.
step: Station 6 · blend — weighted sum of the Values.
why: And out comes each word's new, context-aware vector. That output is what the next layer reads.
%%%

So far: six stations, one belt. Now drag the slider and ride it yourself.

%%% svg
<svg id="pl-svg" viewBox="0 0 520 128" role="img" aria-label="Interactive attention pipeline. Six stations in a row — embed, project into Q K V, score, scale, softmax, blend the values — leading to the output word. Drag the slider to advance the active station and watch a sentence flow through in order."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" font-size="11" fill="#2C2A28">The attention pipeline — one sentence, six stations, in order</text><rect id="pl-r0" x="12" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="44" y="60" fill="#3A342E">embed</text><text x="44" y="73" font-size="7.5" fill="#6B645E">word→vec</text><text x="82" y="63" fill="#B8AEA2">›</text><rect id="pl-r1" x="92" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="124" y="60" fill="#3A342E">Q K V</text><text x="124" y="73" font-size="7.5" fill="#6B645E">3 filters</text><text x="162" y="63" fill="#B8AEA2">›</text><rect id="pl-r2" x="172" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="204" y="60" fill="#3A342E">score</text><text x="204" y="73" font-size="7.5" fill="#6B645E">Q·K</text><text x="242" y="63" fill="#B8AEA2">›</text><rect id="pl-r3" x="252" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="284" y="60" fill="#3A342E">÷√dₖ</text><text x="284" y="73" font-size="7.5" fill="#6B645E">scale</text><text x="322" y="63" fill="#B8AEA2">›</text><rect id="pl-r4" x="332" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="364" y="60" fill="#3A342E">softmax</text><text x="364" y="73" font-size="7.5" fill="#6B645E">→shares</text><text x="402" y="63" fill="#B8AEA2">›</text><rect id="pl-r5" x="412" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="444" y="60" fill="#3A342E">Σ·V</text><text x="444" y="73" font-size="7.5" fill="#6B645E">blend</text><text x="490" y="63" fill="#2D8B55">→out</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>step through <input id="pl-n" type="range" min="0" max="5" step="1" value="0" style="width:56%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="pl-out" style="margin-top:6px;color:#2C2A28"><b>Station 1 · embed:</b> each word becomes a vector — its profile card.</div>
</div>
<script>(function(){
  var ns=document.getElementById('pl-n');if(!ns)return;
  var out=document.getElementById('pl-out');
  var desc=[
    ['embed','each word becomes a vector — its profile card.'],
    ['project into Q, K, V','three learned grids (W_Q, W_K, W_V) turn that card into a Query, a Key and a Value.'],
    ['score','Q·K — each word scores how well its Query matches every Key. Our "bank" gets 5.4 for river.'],
    ['scale ÷√dₖ','divide the scores by √dₖ so long cards don’t make the numbers too loud. 5.4 → 2.7.'],
    ['softmax','turn the scaled scores into shares that always add up to 1. river takes 0.71.'],
    ['blend Σ·V','pour in everyone’s Value by those shares → the word’s new vector, [0.74, 0.15].']
  ];
  function paint(){
    var k=+ns.value;
    for(var i=0;i<6;i++){var r=document.getElementById('pl-r'+i);if(!r)continue;
      if(i<k){r.setAttribute('fill','#E8F5EE');r.setAttribute('stroke','#2D8B55');}
      else if(i===k){r.setAttribute('fill','#FDF3E8');r.setAttribute('stroke','#C99A12');}
      else{r.setAttribute('fill','#FDF9F3');r.setAttribute('stroke','#E5DFD6');}}
    out.innerHTML='<b>Station '+(k+1)+' · '+desc[k][0]+':</b> '+desc[k][1];
  }
  ns.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Here's the thing that made attention take over the field. There is **no left-to-right loop** anywhere on that belt. Every word makes its Q, K and V in the *same* multiply, and every score in the `n × n` grid is computed at the same moment. Older models had to read a sentence one word at a time and wait. Attention reads the whole sentence in one go — which is precisely why it trains fast enough to swallow the internet.
%%%

And the whole stage is cheap to own: the only things to learn here are the three grids, so roughly `d_model×d_k + d_model×d_k + d_model×d_v` numbers. No loop, no extra state.

#### Where the words "key" and "value" even came from
These names sound odd until you meet their ancestors. Three short stops, and the design stops looking arbitrary.

%%% cards
📇 | The hard lookup | A dictionary: give an EXACT key, get back exactly ONE value. Attention is the soft version — a fuzzy match that returns a blend. That's where the words come from.
🍾 | The squeezed summary | Older translators crushed a whole sentence into one final vector, then translated from that. Long sentences didn't fit through the bottleneck. Letting a word query EVERY position removed the squeeze.
📐 | The scoring network | The first alignment models scored with a tiny neural net and had no Value projection at all. Swapping that for a plain dot product plus three learned grids made it both cheaper AND more expressive — that's today's design.
%%%

#### Four honest limits — each one a door to a later day
None of these are bugs. They are the edges of what one set of Q/K/V can do, and each has a named fix waiting.

%%% table
:: what it can't do :: why :: the named fix
Feel word order or distance :: a score only ever sees what two cards *say*, never where they sit — shuffle the words and the outputs shuffle right along :: **positional encoding** (Day 5)
Follow two relationships at once :: one set of Q/K/V gives each word exactly one blend, so its subject and its adjective get smushed together :: **multi-head attention** (Day 4)
Ignore the filler slots in a batch :: short sentences get padded to equal length, and a pad gets a real Key and Value like any word, stealing share :: a **padding mask**, applied before the shares form
Build curved features, or refuse to peek ahead :: the stage is three plain multiplies, and nothing in it forbids looking at later words :: the block's **feed-forward** sublayer, and **causal masking** for generation
%%%

%%% insight
One extreme deserves a name because you'll hear it. Squeeze `d_k` too small, or let every Key drift to look alike, and the shares flatten toward a plain average — every word ends up blended into the same mush and loses its identity. That's **rank collapse** (also called *token uniformity*). The cures are the ones already on the table: enough width per head, several heads instead of one starved one, and the residual connection that carries each word's own vector forward untouched.
%%%

And one more name so it isn't a stranger later: Queries from one sentence reading *another* sentence's Keys is **cross-attention**. Nothing to learn today.

#### The twist worth naming — self-attention
Here's the quiet twist in today's whole library visit. All day, the Queries, the Keys and the Values came from the *same* sentence. "bank"'s Query looked at "river"'s Key — and "river" was sitting right there beside it.

When every word makes its Q, K and V from the same sequence, and every word attends to every other word in that same sequence, the mechanism has a name: [[self-attention||Attention where the Queries, Keys, and Values all come from the same sentence — every word looks at every other word in its own sequence.]]. "Self", because the sentence is looking at *itself*. That's the kind you built today, and it's the workhorse inside every layer of a transformer.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "Self-attention projects each token into a Query, Key and Value, scores every Query against every Key with a scaled dot product, softmaxes those scores into weights that sum to 1, and returns each token as a weighted sum of the Values — all drawn from the same sequence, so every token is rewritten in the context of the whole sentence." You can already say every piece of that. That is a real engineer's sentence, and it's yours now.
!!!

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the entire beating heart of a transformer today. Here's a fresh way to hold it all at once, plus cheat-sheets to come back to any time.

Think of the whole day as **one library visit written on a recipe card.** You walked in with a question, scanned the spine-labels, spent your afternoon in proportion to how well each matched, and walked out with one blended understanding. That's it. That's attention.

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="A recipe card titled attention in six steps. Step one embed. Step two three roles: Query asks, Key advertises, Value shares. Step three score: Query dot Key for every pair. Step four scale: divide by root d k. Step five share: softmax gives positive shares that add up to one. Step six blend: a weighted sum of the Values. A footer notes that in self-attention Q, K and V all come from the same sentence."><g font-family="monospace"><rect x="24" y="30" width="472" height="150" rx="10" fill="#FDF9F3" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="52" text-anchor="middle" fill="#8A6D3B" font-size="12" font-weight="bold">🍳 Attention, in six steps</text><text x="42" y="74" fill="#6B645E" font-size="9.5">1 · embed: each word arrives as its Day-1 card, x</text><text x="42" y="92" fill="#5E5191" font-size="9.5">2 · project: three learned grids → Query (ask), Key (advertise), Value (share)</text><text x="42" y="110" fill="#C99A12" font-size="9.5">3 · score: Query · Key for every pair — bigger = stronger match</text><text x="42" y="128" fill="#C99A12" font-size="9.5">4 · scale: divide by √d_k so long cards don't get too loud</text><text x="42" y="146" fill="#2D8B55" font-size="9.5">5 · share: softmax → positive shares that add up to 1</text><text x="42" y="164" fill="#7C6DAA" font-size="9.5">6 · blend: weighted sum of the Values → a fresh, context-aware word</text><text x="260" y="198" text-anchor="middle" fill="#6B645E" font-size="9">self-attention: Q, K, V all come from the SAME sentence — every word reads every other</text></g></svg>
%%%

**Gets right:** the whole mechanism really is a short, fixed recipe you can recite.
**Breaks down:** a recipe card gets followed once. A transformer runs this recipe dozens of times, stacked, each layer re-blending on top of the last.

#### The visit, one beat at a time
- **Three roles.** Each word makes a **Query** (what it's looking for), a **Key** (what it advertises) and a **Value** (what it shares) — three learned grids (**W_Q, W_K, W_V**) on its one embedding.
- **Score the match.** A word's Query · another word's Key = an **attention score** (a dot product). Because the grids differ, "A → B" is not "B → A".
- **Scale.** Divide by **√d_k** so long cards don't blow the scores up. This is the "scaled" in **scaled dot-product attention**.
- **Share.** **Softmax** turns the scaled scores into **attention weights** — all positive, adding to 1. One pizza of attention.
- **Blend.** The output is a **weighted sum of the Values**, each poured in by its share. Out comes a fresh, context-aware word.
- **The twist.** Q, K and V all come from the *same* sentence — so this is **self-attention**.

#### One word, all the way through — with its shapes
Let's re-derive today from scratch in six hops, following just the word "bank". Watch the shape at every hop; that's the habit that saves you hours of debugging.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="One word traced through six hops with its shape at each hop. The bank card of shape 1 by 4 becomes a Query of shape 1 by 4, a Key of shape 1 by 4 and a Value of shape 1 by 2. The Query scores against six Keys giving six numbers, which get divided by two, then softmaxed into six shares, then blended with the Values into one output of shape 1 by 2."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="11">"bank", hop by hop — the numbers AND the shapes</text><rect x="10" y="34" width="86" height="52" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="53" y="50" text-anchor="middle" fill="#6B645E">card x</text><text x="53" y="64" text-anchor="middle" fill="#2C2A28">[0,1,1,0]</text><text x="53" y="78" text-anchor="middle" fill="#9A938A">1 × 4</text><text x="102" y="62" fill="#B8AEA2" font-size="12">→</text><rect x="116" y="26" width="96" height="30" rx="5" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="164" y="38" text-anchor="middle" fill="#5E5191">Q [1,.5,0,0]</text><text x="164" y="50" text-anchor="middle" fill="#9A938A">1 × 4</text><rect x="116" y="60" width="96" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="164" y="71" text-anchor="middle" fill="#8A6D3B">K [0,2,4,0]</text><text x="164" y="82" text-anchor="middle" fill="#9A938A">1 × 4</text><rect x="116" y="90" width="96" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="164" y="101" text-anchor="middle" fill="#276b45">V [0.1, 1.0]</text><text x="164" y="112" text-anchor="middle" fill="#9A938A">1 × 2</text><text x="218" y="62" fill="#B8AEA2" font-size="12">→</text><rect x="232" y="34" width="88" height="52" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="276" y="48" text-anchor="middle" fill="#8A6D3B">scores</text><text x="276" y="62" text-anchor="middle" fill="#2C2A28" font-size="8">river 5.4 · bank 1.0</text><text x="276" y="78" text-anchor="middle" fill="#9A938A">1 × 6</text><text x="326" y="62" fill="#B8AEA2" font-size="12">→</text><rect x="340" y="34" width="82" height="52" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="381" y="48" text-anchor="middle" fill="#8A6D3B">÷ √4 = ÷2</text><text x="381" y="62" text-anchor="middle" fill="#2C2A28" font-size="8">river 2.7</text><text x="381" y="78" text-anchor="middle" fill="#9A938A">1 × 6</text><text x="428" y="62" fill="#B8AEA2" font-size="12">→</text><rect x="440" y="34" width="72" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="476" y="48" text-anchor="middle" fill="#276b45">softmax</text><text x="476" y="62" text-anchor="middle" fill="#2C2A28" font-size="8">river 0.71</text><text x="476" y="78" text-anchor="middle" fill="#9A938A">1 × 6</text><path d="M476 92 L476 118 L286 118 L286 132" fill="none" stroke="#7C6DAA" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="281,132 291,132 286,142" fill="#7C6DAA"/><rect x="176" y="146" width="220" height="46" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="286" y="162" text-anchor="middle" fill="#5E5191">blend: Σ share × Value</text><text x="286" y="176" text-anchor="middle" fill="#2C2A28">new "bank" = [0.74, 0.15]</text><text x="286" y="188" text-anchor="middle" fill="#9A938A">1 × 2 — same length as a Value</text><text x="88" y="170" text-anchor="middle" fill="#276b45" font-size="9">watery 0.74</text><text x="88" y="184" text-anchor="middle" fill="#9A938A" font-size="9">money 0.15 → river-bank</text></g></svg>
%%%

%%% steps
step: `x = [0, 1, 1, 0]` — shape `1 × 4`, so `d_model = 4`.
why: "bank"'s Day-1 card. Four slots standing for water / place / money / action.
step: Three grids → `Q = [1.0, 0.5, 0, 0]` (`1 × 4`), `K = [0, 2.0, 4.0, 0]` (`1 × 4`), `V = [0.1, 1.0]` (`1 × 2`).
why: `d_k = 4` for Q and K because they get compared; `d_v = 2` for V because it only gets poured. Three different grids, three different lists.
step: Score Q against all six Keys → `[0.2, 0.2, 0.4, 0.2, 5.4, 1.0]`, shape `1 × 6`.
why: One number per word in the sentence. River's 5.4 towers over everything, because its Key is loud in exactly the slots "bank" is asking about.
step: Divide by `√d_k = √4 = 2` → `[0.1, 0.1, 0.2, 0.1, 2.7, 0.5]`.
why: The volume knob. Ranking untouched, numbers calm — and at a real `d_k = 64` this step is the difference between hearing the sentence and hearing one word.
step: Softmax → `[0.05, 0.05, 0.06, 0.05, 0.71, 0.08]`, still `1 × 6`, adding to 1.00.
why: Which turns a ranking into a spendable recipe: 71% of "bank"'s attention goes to "river".
step: Blend the Values by those shares → `[0.74, 0.15]`, shape `1 × 2`.
why: Therefore "bank" leaves as a *watery* vector, right next to river's Value and far from its own old one. Six hops, one confused word made clear.
%%%

#### Cheat-sheet · the ideas at a glance
%%% table
:: idea :: in one line :: watch out
Query / Key / Value :: three roles of one word — ask / advertise / share :: they must come from three DIFFERENT grids
W_Q, W_K, W_V :: the three learned grids that make Q, K, V :: learned in training, not typed in
Shapes :: x is `1 × d_model`; Q, K are `1 × d_k`; V is `1 × d_v` :: d_q must equal d_k; d_v is free
Attention score :: Query · Key — how well two words match :: a dot product: multiply slot-by-slot, add
Scaling by √d_k :: shrink scores so long cards don't clip softmax :: divide, THEN softmax — the "scaled" in the name
Softmax :: turns scaled scores into positive shares summing to 1 :: a clear winner grabs most of the share
Weighted sum of V :: blend the Values by their shares → new word :: mixes the Values, never the Keys
Self-attention :: Q, K, V all from the same sentence :: order-blind until positional encoding (Day 5)
%%%

#### Cheat-sheet · the words you met today
%%% jargon
attention | letting each word gather info from the other words, weighted by how relevant they are
Query (Q) | the "what am I looking for?" version of a word — the question it asks
Key (K) | the "what do I advertise?" version — the label other words check against
Value (V) | the "what will I share?" version — the content a word hands over if listened to
projection | turning one list of numbers into another by multiplying by a learned grid
W_Q, W_K, W_V | the three learned grids that make Q, K, and V
d_model | the width of the card that carries one word through the model
d_k | the length of each Query / Key card; its square root is the scaling divisor
d_v | the length of each Value card — a free choice, it needn't match d_model
dot product | multiply two lists slot-by-slot and add — one number saying how well they agree
attention score | the dot product of one word's Query with another word's Key
softmax | turns a list of scores into positive shares that add up to 1
attention weights | the shares softmax produces — how much to listen to each word
weighted sum | add things up, each counted by its share
scaled dot-product attention | the standard attention: divide scores by √d_k before softmax
self-attention | attention where Q, K, V all come from the same sentence
rank collapse | when keys look alike or d_k is too small, the shares flatten and words lose identity
%%%

That's the whole day. Next you'll zoom in on **Attention Scores & Softmax** — the same middle stations, done for every word at once as one matrix, and where the honest limits start to bite.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A word makes three vectors — Query, Key, Value. What is the Query, in one line? | a:2 | The word's spelling | A copy of the embedding | The "what am I looking for?" version — the question the word asks the room | The final output vector | fb: Query = the question a word asks. It gets compared against every other word's Key to find matches.
q: How is a word's attention score with another word computed? | a:1 | By counting shared letters | By the dot product of its Query with the other word's Key | By averaging their embeddings | By softmax alone | fb: Score = Query · Key (a dot product): multiply slot-by-slot and add. Bigger = stronger match.
q: What does softmax guarantee about the attention weights? | a:2 | They are all whole numbers | They are sorted from high to low | They are all positive AND add up to exactly 1 | They are all equal | fb: Softmax makes positive shares that sum to 1 — one pizza of attention split among the words.
q: Why do we divide the scores by √d_k before softmax ("scaled" dot-product attention)? | a:1 | To make the model train faster on GPUs | Because a longer card means more slots in the sum, so scores grow like √d_k and clip softmax onto one winner; dividing by √d_k cancels exactly that growth | To convert the scores into letters | To add word-order information | fb: More slots → a bigger typical sum, growing like √d_k. Dividing by √d_k cancels that growth, so softmax can still hear more than one word.
%%%

@@@ produce id=produce tag="Produce" title="Predict which word 'bank' listens to, then run it" gotit="Done"
Here's the fun part — you get to watch today's whole mechanism prove itself in a few lines of output. **Before you run anything, predict:** in "the animal crossed the river bank", the word "bank" is asking "what kind of place am I?" When you score its Query against every Key, scale, and softmax those scores into shares — which word grabs the biggest slice, and will the new blended "bank" end up leaning toward "river"? Jot your guesses down, then **watch** it happen. That little "I was right / oh!" moment is what makes the day stick.

A finished reference version already sits in this folder as `sessions/m03-attention/day-02-qkv/experiment.py`, with a self-check at the bottom. **Run it first**, then pick a path below and make it your own.

```
python3 sessions/m03-attention/day-02-qkv/experiment.py
```

#### Option A · write it yourself
Start a fresh file (or copy the reference next to it and edit your copy, so the original keeps working). Make a tiny embedding matrix `x` — 6 word-rows of length 4, one row per word of our sentence. Make three *different* weight grids `Wq, Wk, Wv` and compute `Q = x@Wq`, `K = x@Wk`, `V = x@Wv`; print every shape, and assert that `Q` and `K` are *not* the same numbers. Take word 5's Query — that's "bank" — score it against every Key with a dot product, divide by `sqrt(d_k)`, run `softmax` to get shares (check they sum to 1), then compute the weighted sum of the Values. Print each step and **observe** which word wins the biggest share.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code write the code and run it for you. Read the result against your prediction.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 4 Day 2 artifact.

Write a script (keep the existing sessions/m03-attention/day-02-qkv/experiment.py intact — put mine at
sessions/m03-attention/day-02-qkv/my_experiment.py) that, with a comment on each step:
1. Defines a 6x4 embedding matrix x for the sentence "the animal crossed the river bank" — one row per
   word, d_model = 4. Say in a comment what each of the 4 slots stands for.
2. Defines three DIFFERENT weight grids: Wq (4x4), Wk (4x4) and Wv (4x2), so d_k = 4 and d_v = 2.
   Compute Q = x@Wq, K = x@Wk, V = x@Wv, print each with its shape, and assert Q and K are NOT
   identical — tying Wq = Wk would make the score grid symmetric and delete every directional relation.
3. Takes word 5's Query — that's "bank", the word asking "what kind of place am I?" — and scores it
   against every Key with a dot product (scores = K @ Q[5]); prints the raw scores next to the words.
4. Divides those scores by sqrt(d_k) — the scaled dot-product step — and prints the scaled scores,
   explaining in a comment why longer cards need this.
5. Runs softmax on the scaled scores to get attention weights; prints them and asserts they sum to ~1.0.
6. Computes the output as the weighted sum of the Values (weights @ V); prints it, prints each
   share x Value term so the sum is checkable by hand, and prints whether the result is closer to
   river's Value or to bank's own old Value.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- `Q` and `K` both print as `(6, 4)` and `V` as `(6, 2)` — **and `Q` is not identical to `K`**, because they came from two different grids. (If they ever match, the score grid goes symmetric and direction disappears.)
- The raw scores for "bank" come out `[0.2, 0.2, 0.4, 0.2, 5.4, 1.0]` — one dot product per word. **"river" is the 5.4.**
- After dividing by `√d_k = 2` they become `[0.1, 0.1, 0.2, 0.1, 2.7, 0.5]` — same ranking, calmer numbers. That's the scaling step.
- The softmax weights are all positive and **sum to 1.00** — one pizza of attention, with `0.706` of it on "river". Did you predict that word?
- The final output is `[0.742, 0.154]`: **notice** it sits at distance 0.26 from river's Value and 1.06 from bank's own old Value. The blend is river-flavoured. That vector is what the next layer reads.
- The thing most worth noticing: no word was ever compared by spelling. The whole decision — who listens to whom — came from Query·Key match scores turned into shares. That is attention.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-02-qkv/log.md`: (1) in your own words, what the Query, the Key and the Value each do; (2) which word won the biggest attention share, and whether that surprised you; (3) one thing you're still curious about (why the scores get divided by √d_k, or why Q and K must come from different grids, are both fair game).
!!!

@@@ fin
