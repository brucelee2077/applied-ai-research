---
quest_id: wf4-d03-attention-scores
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 3 — Attention Scores &amp; Softmax"
module_label: "Module 4 · Represent · Day 3"
title: "Attention Scores &amp; Softmax"
subtitle: "From Matches to a Weighted Blend"
brand_sub: "Foundations · M4 Day 3"
spine: "budget"
nav_prev_href: "../day-02-qkv/lesson.html"
nav_prev_label: "Query, Key, Value"
nav_next_href: "../day-04-multihead/lesson.html"
nav_next_label: "Multi-Head Attention"
fin_title: "Module 4 · Day 3 complete! 🏆"
fin_body: "Nice work — you've completed <b>Attention Scores &amp; Softmax</b>. You just wrote the heart of the transformer: score every pair, split a 100% attention <b>budget</b> with softmax, then blend the Values by that budget — <code>softmax(Q@Kᵀ/√d_k)@V</code>.<br>Next up: <b>Multi-Head Attention</b> — running several of these meetings at once."
notebook_yardstick: null
coverage_topics:
  - {topic: attention score one number, keywords: [attention score, one number, how much]}
  - {topic: query and key roles, keywords: [query, key, looking for, offers]}
  - {topic: dot product similarity, keywords: [dot product, slot by slot, bigger dot]}
  - {topic: score matrix every pair, keywords: [score matrix, every query, one row per]}
  - {topic: scaling by sqrt d_k, keywords: [sqrt, scale, d_k, divide]}
  - {topic: softmax weights sum to one, keywords: [softmax, sum to 1, positive]}
  - {topic: additive bahdanau attention, keywords: [additive, Bahdanau, small neural network]}
  - {topic: unscaled scores blow up, keywords: [scores swell, clips, gradients shrink, too loud]}
  - {topic: masking future and padding, keywords: [mask, negative infinity, padding, future]}
  - {topic: raw score is not a weight, keywords: [raw score, means nothing, until softmax]}
  - {topic: dot product has no position, keywords: [no word order, order-blind, word order]}
---

@@@ hero
@lede You know that moment at a busy dinner table when three people talk at once, and you quietly decide who to listen to? You lean toward the friend with the good story, give a little to the person beside you, almost none to the far end — and, without noticing, your listening always adds up to one whole you. That everyday balancing act — hand out shares of one fixed pot of attention — is exactly the move at the center of every transformer, the same move that lets ChatGPT figure out which words in your sentence matter most to each other. Today you build it by hand: for one word, score how well it matches every other word, then split a single 100% attention **budget** among them. By the end you'll have written the one line at the heart of the transformer — and it'll feel as familiar as choosing who to listen to at dinner.
@goal Together we'll turn "how well do these two words match?" into a single number with a friendly multiply-and-add, lay every pair's number into a tidy grid, keep those numbers from getting too loud as words grow long, and split one attention **budget** into shares with softmax. You'll meet the clumsy older way of scoring (and see why the modern way is faster), the two puzzles that break raw scores — and their two neat fixes — and the honest limits of a score. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.

%%% warmup
q: Yesterday each word split into THREE versions of itself. What were those three roles called? | a:2 | past, present, future | noun, verb, adjective | Query, Key, Value | red, green, blue | concept: query-key-value | fb: A word plays three roles at once — Query (what it asks), Key (what it advertises), Value (what it shares). Today the Query and Key finally meet to make a score.
q: Yesterday a word made its three versions by passing its one embedding through three learned filter-grids. What were those grids called? | a:1 | Q, K, V | W_Q, W_K, W_V | d_k and √d_k | softmax | concept: projection-filters | fb: The three learned filters are W_Q, W_K, W_V — one per role. Today we use the Query and Key they produced.
%%%

@@@ concept id=c1 tag="One little number" title="A score is just 'how much should I listen to you?'" gotit="Got the score idea"
Let's start with the one idea the whole day hangs on. When a word wants to understand itself, it doesn't listen to every other word equally. It gives each one a number — a little rating that says "how much should I listen to *you*?" That single rating is called an [[attention score||One number that says how much a word should listen to another word. Bigger = listen more.]]. One number, for one pair of words. Bigger number, listen more.

Think of a **dinner table** where three people are talking at once. You don't split your ears evenly. In your head, without noticing, you quietly rate each speaker: "the story about the trip — very interesting (an 8); the weather chat — meh (a 2); the mumbling at the far end — barely there (a 1)." Those ratings are your attention scores. A word does the exact same thing to the other words around it: it hands each one a number saying how interesting it finds them right now.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A dinner table scene. One listener in the middle rates three speakers with little score numbers: the storyteller gets an 8, the neighbor gets a 2, the far-end mumbler gets a 1. The listener leans toward the highest score."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">You rate each speaker: how much do I listen to you?</text><ellipse cx="260" cy="118" rx="120" ry="52" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="122" text-anchor="middle" fill="#8A6D3B" font-size="10">the table</text><circle cx="260" cy="70" r="16" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="260" y="74" text-anchor="middle" fill="#5E5191" font-size="10">you</text><circle cx="120" cy="118" r="16" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="120" y="122" text-anchor="middle" fill="#276b45" font-size="9">story</text><circle cx="400" cy="118" r="16" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="400" y="122" text-anchor="middle" fill="#9A938A" font-size="9">weather</text><circle cx="260" cy="176" r="15" fill="#FDF9F3" stroke="#B8AEA2"/><text x="260" y="180" text-anchor="middle" fill="#9A938A" font-size="8">far end</text><line x1="246" y1="80" x2="134" y2="110" stroke="#2D8B55" stroke-width="3"/><text x="188" y="86" text-anchor="middle" fill="#276b45" font-weight="bold">8</text><line x1="276" y1="78" x2="386" y2="110" stroke="#C99A12" stroke-width="1.5"/><text x="336" y="86" text-anchor="middle" fill="#8A6D3B">2</text><line x1="260" y1="86" x2="260" y2="161" stroke="#B8AEA2" stroke-dasharray="3,2"/><text x="272" y="130" fill="#9A938A">1</text></g></svg>
%%%

**What the dinner-table picture gets right:** you hand out one rating per speaker, and a bigger rating means you lean in more — that's exactly an attention score. **Where it breaks down:** at a real table you listen with two ears and can only really follow one voice at a time, but a word scores *every* other word at once and can blend many of them together, which no human ear can do.

#### The two words behind every score
A score always compares two things: what one word is *looking for*, and what another word *offers*. Day 2 gave these two things names.
- The [[Query||The "what am I looking for?" side of a word — the question it brings to the comparison.]] is the *asking* side — "what am I looking for?"
- The [[Key||The "what do I offer?" side of a word — the label it advertises to be matched against.]] is the *offering* side — "here's what I've got."

So the attention score is always: **how well does this word's Query match that word's Key?** One asks, one offers, and the score rates the fit. That's the whole idea — everything today is just how we turn that fit into a number, keep the number tidy, and split attention by it.

!!! c-info 🗺️
<b>The one line to hold all day:</b> score every Query against every Key, then split one attention <b>budget</b> among the words by those scores. Hold that sentence; the rest is slow motion.
!!!

@@@ concept id=c2 tag="The match number" title="The dot product — one number for 'do we point the same way?'" gotit="Got the dot product"
So a word needs a number for "how well does my Query match your Key?" Here's the happy surprise: there's a wonderfully simple recipe for it, and you already do something like it in your head. It's called the [[dot product||Multiply two lists slot by slot, then add it all up. One number that says how much the two lists point the same way.]] — multiply the two lists of numbers slot by slot, then add up the results. One number falls out. A big number means "these two point the same way — strong match." A small or negative number means "they don't line up — weak match."

Think of **two friends filling in the same taste survey** — each rates pizza, hiking, and horror movies from 0 to 5. To see how alike their tastes are, you'd go item by item: multiply their two pizza scores, then their two hiking scores, then their two movie scores, and add those up. If they both love the same things, the matching high numbers multiply into a big total. If one loves what the other hates, the products stay small and the total is low. That single "how alike are you?" total is the dot product — and it's exactly how a Query and a Key rate their match.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two taste surveys side by side. Friend A rates pizza 5, hiking 1, horror 0. Friend B rates pizza 4, hiking 2, horror 1. Arrows show multiplying each matching pair — 5 times 4, 1 times 2, 0 times 1 — and adding the products into one big match number, 22."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Multiply matching items, add them up → one match number</text><rect x="24" y="40" width="150" height="118" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="99" y="60" text-anchor="middle" fill="#5E5191" font-weight="bold">Query (friend A)</text><text x="40" y="84" fill="#3A342E">🍕 pizza .... 5</text><text x="40" y="106" fill="#3A342E">🥾 hiking ... 1</text><text x="40" y="128" fill="#3A342E">🎬 horror ... 0</text><rect x="346" y="40" width="150" height="118" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="421" y="60" text-anchor="middle" fill="#8A6D3B" font-weight="bold">Key (friend B)</text><text x="362" y="84" fill="#3A342E">🍕 pizza .... 4</text><text x="362" y="106" fill="#3A342E">🥾 hiking ... 2</text><text x="362" y="128" fill="#3A342E">🎬 horror ... 1</text><text x="260" y="82" text-anchor="middle" fill="#276b45" font-size="10">5 × 4 = 20</text><text x="260" y="104" text-anchor="middle" fill="#C99A12" font-size="10">1 × 2 = 2</text><text x="260" y="126" text-anchor="middle" fill="#9A938A" font-size="10">0 × 1 = 0</text><line x1="174" y1="82" x2="346" y2="82" stroke="#B8AEA2" stroke-dasharray="2,2"/><line x1="174" y1="104" x2="346" y2="104" stroke="#B8AEA2" stroke-dasharray="2,2"/><line x1="174" y1="126" x2="346" y2="126" stroke="#B8AEA2" stroke-dasharray="2,2"/><rect x="180" y="168" width="160" height="30" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="260" y="188" text-anchor="middle" fill="#276b45" font-weight="bold">add them: 20 + 2 + 0 = 22</text></g></svg>
%%%

**What the taste-survey picture gets right:** matching high numbers multiply into a big total, so a big dot product really does mean "you two are alike." **Where it breaks down:** a survey is easy to read because each row has a label like "pizza," but a Query and Key are just raw numbers with no labels — the model *learned* what each slot means during training, so the match is real even though nothing is spelled out.

#### The one-line rule (in plain words)
Take the word's Query and another word's Key. Multiply them slot by slot. Add up all the products. That total *is* the attention score between the two words.

%%% formula
expr: score = Q · K = (q₁·k₁) + (q₂·k₂) + … + (q_d·k_d)
note: Q = one word's Query list, K = another word's Key list. The dot "·" means multiply slot-by-slot and add. d = how many slots each list has. One number comes out — the match score.
%%%

Now let's *watch* the recipe run on real numbers. **Predict first:** here are two Keys being scored against the same Query `[2, 0, 1]`. Key-river is `[3, 0, 1]` (points a similar way) and Key-cake is `[0, 4, 0]` (points a different way). Which one do you think scores higher? Guess, then reveal.

%%% demo id=dot label="score two Keys against one Query"
code: Q = [2, 0, 1];  dot(Q, K_river=[3,0,1])  vs  dot(Q, K_cake=[0,4,0])
out: river -> 2*3 + 0*0 + 1*1 = 7        cake -> 2*0 + 0*4 + 1*0 = 0
take: <b>river scores 7, cake scores 0.</b> The Query and river's Key had big numbers in the same slots (slot 1: 2 and 3), so their products stacked up. The Query and cake's Key never had big numbers in the same slot, so every product was near zero. Same recipe both times — multiply slot-by-slot and add — but the match number tells them apart. That single number is the whole comparison.
%%%

Take a small victory lap: you just turned "how well do these two words match?" into one number you can compute by hand. That number is the atom of attention.

@@@ concept id=c3 tag="Every pair, a grid" title="The score matrix — every word scored against every word" gotit="Got the grid"
One score is nice, but a sentence has many words, and *each* word wants to rate *all* the others (including itself). So we don't compute one score — we compute a whole tidy grid of them. This grid is called the [[score matrix||A grid holding one score for every pair of words: one row per asking word, one column per offered word.]]. One **row per asking word** (the Queries), one **column per offered word** (the Keys). Cell in row 2, column 3 = "how much should word 2 listen to word 3?"

Think of a **round-robin sports schedule** pinned to the wall — a grid where every team plays every other team, and each little box holds the result of that one matchup. Reading across row "Lions" tells you how the Lions did against everybody. The score matrix is the same: reading across one row tells you how much *that one word* wants to listen to each of the others. Every word gets its own row; every word also shows up as a column.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A 3 by 3 score grid for the words the, river, bank. Rows are asking words (queries), columns are offered words (keys). Each cell holds a match score. The row for bank has a high score in the river column, showing bank listens most to river."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Score matrix: row = who's asking, column = who's offered</text><text x="150" y="52" fill="#9A938A" font-size="9">the</text><text x="240" y="52" fill="#9A938A" font-size="9">river</text><text x="330" y="52" fill="#9A938A" font-size="9">bank</text><text x="245" y="40" text-anchor="middle" fill="#8A6D3B" font-size="9">offered words (Keys) →</text><text x="70" y="86" fill="#5E5191" font-size="9">the</text><text x="66" y="126" fill="#5E5191" font-size="9">river</text><text x="66" y="166" fill="#5E5191" font-size="9">bank</text><text x="30" y="126" fill="#5E5191" font-size="9" transform="rotate(-90 30 126)">asking (Q)</text><rect x="130" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="91" text-anchor="middle" fill="#9A938A">3</text><rect x="220" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="255" y="91" text-anchor="middle" fill="#9A938A">1</text><rect x="310" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="345" y="91" text-anchor="middle" fill="#9A938A">1</text><rect x="130" y="110" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="131" text-anchor="middle" fill="#9A938A">1</text><rect x="220" y="110" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="255" y="131" text-anchor="middle" fill="#9A938A">4</text><rect x="310" y="110" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="345" y="131" text-anchor="middle" fill="#9A938A">2</text><rect x="130" y="150" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="171" text-anchor="middle" fill="#9A938A">1</text><rect x="220" y="150" width="70" height="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="255" y="171" text-anchor="middle" fill="#276b45" font-weight="bold">7</text><rect x="310" y="150" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="345" y="171" text-anchor="middle" fill="#9A938A">3</text><text x="255" y="206" text-anchor="middle" fill="#276b45" font-size="10">bank's row: it listens most to "river" (7)</text></g></svg>
%%%

**What the sports-grid picture gets right:** every player meets every other player, and each box is one matchup — so the whole grid captures all pairings at once, just like the score matrix. **Where it breaks down:** in a schedule a team never plays itself, but in attention a word *does* score itself (the diagonal cell), because a word often finds its own meaning useful.

#### One tidy move computes the whole grid
You could fill the grid one dot product at a time. But there's a single tidy step that does all of them at once: line up every Query as rows, every Key as columns, and multiply. That is `Q @ Kᵀ` — read it as "Queries times Keys, laid out so every row meets every column." Don't worry about the symbols; the point is one clean move produces the entire grid.

%%% formula
expr: Scores = Q @ Kᵀ        (shape: n_words × n_words)
note: Q = all the Queries stacked as rows. Kᵀ = all the Keys turned to stand as columns. "@" = do every row-meets-column dot product in one step. Out comes the n×n grid — one score per pair.
%%%

Here's that one move drawn as a shape story: the tall stack of Queries meets the wide row of Keys, and out pops the perfect square grid — one row and one column per word.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Shape transformation for the score matrix. A tall n by d block of Queries times a wide d by n block of Keys transposed gives a square n by n grid of scores. Boxes labelled with their shapes and an equals sign between them."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Queries (n×d) meet Keys (d×n) → a square n×n grid of scores</text><rect x="30" y="50" width="44" height="90" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="52" y="98" text-anchor="middle" fill="#5E5191" font-size="10">Q</text><text x="52" y="156" text-anchor="middle" fill="#5E5191" font-size="9">n × d</text><text x="95" y="100" text-anchor="middle" fill="#8A6D3B" font-size="16">@</text><rect x="118" y="72" width="120" height="46" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="178" y="100" text-anchor="middle" fill="#8A6D3B" font-size="10">Kᵀ</text><text x="178" y="134" text-anchor="middle" fill="#8A6D3B" font-size="9">d × n</text><text x="270" y="100" text-anchor="middle" fill="#8A6D3B" font-size="16">=</text><rect x="310" y="50" width="90" height="90" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><line x1="340" y1="50" x2="340" y2="140" stroke="#8Cc4a3"/><line x1="370" y1="50" x2="370" y2="140" stroke="#8Cc4a3"/><line x1="310" y1="80" x2="400" y2="80" stroke="#8Cc4a3"/><line x1="310" y1="110" x2="400" y2="110" stroke="#8Cc4a3"/><text x="355" y="46" text-anchor="middle" fill="#276b45" font-size="9">scores</text><text x="355" y="156" text-anchor="middle" fill="#276b45" font-size="9">n × n (a square!)</text><text x="460" y="90" text-anchor="middle" fill="#9A938A" font-size="9">one cell</text><text x="460" y="104" text-anchor="middle" fill="#9A938A" font-size="9">per pair</text></g></svg>
%%%

Notice the `d` (the slot count) disappears in the middle — it gets "used up" by the multiply-and-add — and what's left is a clean square, one row and one column per word.

!!! c-info 💡
<b>Optional (skippable) — the shape story in symbols.</b> If Q has shape n×d (n words, each a d-slot Query) and Kᵀ is d×n (each Key standing as a column), then multiplying them gives n×n — a perfect square, one row and one column per word. You never have to track this to use attention; it's here for the curious.
!!!

@@@ concept id=c4 tag="Turn it down" title="Scaling — keep the scores from getting too loud" gotit="Got the scaling"
Here's a quiet trouble that sneaks in, and the fix is a one-liner. The dot product adds up one product per slot. Short lists have few slots, so the total stays gentle. But real Query and Key lists are *long* — often hundreds of slots. More slots means more products piling on, so the totals can swell into big, shouty numbers. In a moment you'll see why big scores wreck the next step (softmax). For now, the fix is simple: **turn the volume down** by dividing every score by a fixed amount that grows with the list length.

Think of a **group of singers**. One singer is pleasant. Add fifty more all singing at once and the sound gets painfully loud — not because anyone sang louder, but because there are *more* voices stacking up. The gentle fix isn't to silence anyone; it's to turn the room's volume knob down by an amount that matches how many singers there are. Longer lists = more voices = turn the knob down more. That volume knob is the scaling step, and the amount we divide by is [[√d_k||The square root of d_k, the number of slots in each Key/Query list. Dividing scores by it cancels the growth that comes from having more slots.]].

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A volume knob metaphor. On the left a short list of 4 slots gives a gentle score bar. In the middle a long list of 64 slots gives a huge shouty score bar. On the right the same long list after dividing by root d_k gives a gentle bar again — the volume turned back down."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">More slots → louder score. Divide by √d_k → turned back down.</text><rect x="30" y="120" width="50" height="40" fill="#EAF5EE" stroke="#2D8B55"/><text x="55" y="176" text-anchor="middle" fill="#276b45" font-size="9">4 slots</text><text x="55" y="112" text-anchor="middle" fill="#276b45" font-size="9">gentle</text><rect x="200" y="40" width="50" height="120" fill="#FBE3E3" stroke="#C0392B" stroke-width="2"/><text x="225" y="176" text-anchor="middle" fill="#C0392B" font-size="9">64 slots</text><text x="225" y="32" text-anchor="middle" fill="#C0392B" font-size="9">too loud!</text><text x="300" y="100" fill="#8A6D3B" font-size="18">÷ √d_k</text><text x="300" y="120" fill="#8A6D3B" font-size="9">(÷ 8)</text><rect x="430" y="120" width="50" height="40" fill="#EAF5EE" stroke="#2D8B55"/><text x="455" y="176" text-anchor="middle" fill="#276b45" font-size="9">64 slots</text><text x="455" y="112" text-anchor="middle" fill="#276b45" font-size="9">gentle again</text><path d="M285 140 L420 140" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar4)"/><defs><marker id="ar4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#8A6D3B"/></marker></defs></g></svg>
%%%

**What the singers picture gets right:** the loudness grows just from adding more voices, and the cure is to turn a knob down by an amount that matches the crowd size — exactly what dividing by √d_k does. **Where it breaks down:** singers get literally louder in your ears, but scores don't make a sound — "too loud" here means "too big a number," which quietly breaks the *next* step rather than hurting anyone's ears.

#### Why √d_k, and not just any number — watch it work
`d_k` is simply how many slots each Key (and Query) has. We divide by its square root. **Predict first:** two Keys score 7 and 5 with a short 4-slot list. Grow the lists to 64 slots and the *same kind* of match now scores 56 and 40 — much bigger, just from more slots. What happens after we divide by √64 = 8? Guess, then reveal.

%%% demo id=scale label="score before and after dividing by √d_k"
code: raw @ d_k=64:  [56, 40]   then divide by  √64 = 8
out: scaled -> [56/8, 40/8] = [7.0, 5.0]   (back to the gentle 4-slot size)
take: <b>Dividing by √d_k puts the big scores right back to a gentle size.</b> The long list made everything 8× louder; √64 = 8 turns it down by exactly 8×. Notice what stayed the same: 7 is still bigger than 5, so river still beats the others — scaling changes the *loudness*, never *who wins*. That's the whole trick: tame the size, keep the order.
%%%

!!! c-warn ⚠️
<b>The puzzle this fixes (you'll meet it in full soon):</b> if scores get too loud, the next step — softmax — jams into a single spike and learning stalls. Scaling by √d_k is the cheap guard that keeps the scores in a range softmax can work with.
!!!

@@@ concept id=c5 tag="Split the budget" title="Softmax — turn scores into shares that add up to one" gotit="Got softmax"
Now the payoff — and the star of the day. A raw score like 7 doesn't yet tell a word *how much* of its attention to spend. We want to hand out attention like slices of one pizza: every slice is positive, and all the slices together make one whole pizza — no more, no less. The step that does this is [[softmax||A step that turns a row of scores into positive shares that add up to 1 — the biggest score gets the biggest share.]]. Feed it a row of scores; it hands back a row of shares, all positive, all adding up to exactly 1. The biggest score gets the biggest slice. This one budget-splitting move is the knob behind how ChatGPT decides where to "look."

Think of **splitting one pizza among friends by how hungry they are**. The hungriest friend gets the biggest slice, the least hungry gets a sliver — but you only have *one* pizza, so all the slices together still make exactly one whole. Nobody gets a negative slice, and you can't hand out more than one pizza. That fixed, must-add-up-to-one sharing is softmax: it turns "how hungry" (the scores) into "how big a slice" (the attention shares). We call that one whole pizza the attention **budget**.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="On the left, three raw scores as bars: river 2.0, table 1.0, cake 0.1. On the right, a pizza split into three slices by softmax: river gets a big 0.66 slice, table a 0.24 slice, cake a small 0.10 slice, and the three slices add up to one whole pizza."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Scores → one pizza of attention, split into shares that sum to 1</text><text x="90" y="40" text-anchor="middle" fill="#8A6D3B" font-size="10">raw scores</text><rect x="40" y="60" width="120" height="22" fill="#EAF5EE" stroke="#2D8B55"/><text x="170" y="76" fill="#276b45" font-size="9">river 2.0</text><rect x="40" y="92" width="60" height="22" fill="#FCF3DC" stroke="#C99A12"/><text x="110" y="108" fill="#8A6D3B" font-size="9">table 1.0</text><rect x="40" y="124" width="10" height="22" fill="#FDF9F3" stroke="#B8AEA2"/><text x="60" y="140" fill="#9A938A" font-size="9">cake 0.1</text><text x="235" y="110" fill="#5E5191" font-size="16">softmax →</text><circle cx="410" cy="120" r="70" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M410 120 L410 50 A70 70 0 0 1 468 158 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><path d="M410 120 L468 158 A70 70 0 0 1 352 158 Z" fill="#FDF7EC" stroke="#C99A12" stroke-width="1.5"/><path d="M410 120 L352 158 A70 70 0 0 1 410 50 Z" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="438" y="108" fill="#276b45" font-size="9" font-weight="bold">0.66</text><text x="404" y="176" fill="#8A6D3B" font-size="9">0.24</text><text x="368" y="96" fill="#9A938A" font-size="8">0.10</text><text x="410" y="205" text-anchor="middle" fill="#276b45" font-size="9">slices add to 1 whole pizza</text></g></svg>
%%%

**What the pizza picture gets right:** shares are all positive and always add to exactly one whole, and the hungriest friend (highest score) gets the biggest slice — that's precisely softmax. **Where it breaks down:** a pizza gives its biggest slice roughly in proportion to hunger, but softmax is *greedier* than that — it uses a growth curve that hands the top score an outsized slice, so the leader pulls ahead faster than plain proportions would.

#### Play with it — drag the scores and watch the slices move
Here's the live version of the pizza. Drag the score sliders and watch the shares (the slices) rebalance in real time. Try making one score much bigger than the rest — see how it grabs almost the whole budget. This is the exact behavior that makes a word "focus" on one other word.

%%% svg
<svg id="d3sm-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive softmax. Drag three raw scores — river, table, cake — and watch the attention shares, drawn as bars, resize. They always stay positive and add up to exactly one whole pizza."><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="40" y="26" width="440" height="140" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="150" x2="460" y2="150" stroke="#EFE9DF"/><rect id="d3sm-b0" x="110" y="78" width="60" height="72" fill="#2D8B55"/><text id="d3sm-t0" x="140" y="70" fill="#1a5c38">.66</text><text x="140" y="164" fill="#6B645E">river</text><rect id="d3sm-b1" x="230" y="123" width="60" height="27" fill="#C99A12"/><text id="d3sm-t1" x="260" y="115" fill="#9A7208">.24</text><text x="260" y="164" fill="#6B645E">table</text><rect id="d3sm-b2" x="350" y="139" width="60" height="11" fill="#B8AEA2"/><text id="d3sm-t2" x="380" y="131" fill="#6B645E">.10</text><text x="380" y="164" fill="#6B645E">cake</text><text x="260" y="184" fill="#1a5c38" font-size="10">the three shares always add up to one whole pizza (1.00)</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>river <input id="d3sm-s0" type="range" min="-2" max="5" step="0.1" value="2" style="accent-color:#2D8B55;vertical-align:middle"></label>
<label>table <input id="d3sm-s1" type="range" min="-2" max="5" step="0.1" value="1" style="accent-color:#C99A12;vertical-align:middle"></label>
<label>cake <input id="d3sm-s2" type="range" min="-2" max="5" step="0.1" value="0.1" style="accent-color:#B8AEA2;vertical-align:middle"></label>
</div>
<div id="d3sm-out" style="margin-top:6px;color:#2C2A28">scores [2.0, 1.0, 0.1] → shares <b>0.66 / 0.24 / 0.10</b> (sum 1.00) — crank one score up and it grabs most of the budget.</div>
</div>
<script>(function(){
  var s0=document.getElementById('d3sm-s0');if(!s0)return;
  var s1=document.getElementById('d3sm-s1'),s2=document.getElementById('d3sm-s2'),out=document.getElementById('d3sm-out');
  var BASE=150,MAXH=112;
  function paint(){
    var v=[+s0.value,+s1.value,+s2.value];
    var ex=v.map(function(x){return Math.exp(x);});var Z=ex[0]+ex[1]+ex[2];
    var sh=ex.map(function(e){return e/Z;});
    for(var i=0;i<3;i++){var h=Math.max(2,sh[i]*MAXH);var b=document.getElementById('d3sm-b'+i),t=document.getElementById('d3sm-t'+i);
      if(b){b.setAttribute('y',(BASE-h).toFixed(1));b.setAttribute('height',h.toFixed(1));}
      if(t){t.setAttribute('y',(BASE-h-8).toFixed(1));t.textContent=sh[i].toFixed(2);}}
    out.innerHTML='scores ['+v.map(function(x){return x.toFixed(1);}).join(', ')+'] → shares <b>'+sh.map(function(x){return x.toFixed(2);}).join(' / ')+'</b> (sum '+(sh[0]+sh[1]+sh[2]).toFixed(2)+')';
  }
  [s0,s1,s2].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

!!! c-info 💡
<b>Optional (skippable) — the exact recipe.</b> Softmax does three tiny steps to each score in the row: (1) raise a special growth number to the power of the score (this makes everything positive and stretches the leader ahead), (2) add up those grown values, (3) divide each one by that total so the row sums to 1. You never need the formula to use attention — the pizza intuition is enough. The full math gets its own day.
!!!

The victory lap: you now hold the whole spine — **score every pair, then split one attention budget by those scores.** That is attention. Everything left today is making it robust and honest.

@@@ concept id=c6 tag="The old way" title="How scoring used to work — and why the dot product won" gotit="Got the history"
Before the dot product became the standard, the first attention models scored matches a different, clumsier way. Knowing it makes you appreciate *why* today's version is so clean — and it's a lovely peek at how the field actually moved forward. This older method is called [[additive attention||Bahdanau's 2014 scoring: feed the Query and Key into a tiny neural network that outputs the match score. Works, but slower than a plain dot product.]] (or *Bahdanau attention*, after the researcher who introduced it in 2014).

Think of **two ways to check if a key fits a lock**. The clumsy way: hire a locksmith who studies the key and the lock and gives you a fitness rating — accurate, but you pay them and wait. The quick way: just push the key in and feel whether it turns — instant, free, good enough. Additive attention is the locksmith: it runs a **tiny neural network** on the Query and Key to produce a score. The dot product is just pushing the key in: multiply-and-add, done. The field kept the quick way because it's far cheaper to run millions of times.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two paths to a match score. Top path, additive attention: Query and Key feed into a small neural network box which outputs a score — labelled slow. Bottom path, dot product: Query and Key go straight into a multiply-and-add, outputting a score — labelled fast."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two ways to score a Query–Key match</text><rect x="20" y="40" width="60" height="24" fill="#EAF0FA" stroke="#5E5191"/><text x="50" y="56" text-anchor="middle" fill="#5E5191" font-size="9">Q</text><rect x="20" y="70" width="60" height="24" fill="#FCF3DC" stroke="#C99A12"/><text x="50" y="86" text-anchor="middle" fill="#8A6D3B" font-size="9">K</text><rect x="150" y="46" width="120" height="42" rx="6" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><text x="210" y="63" text-anchor="middle" fill="#C0392B" font-size="9">tiny neural net</text><text x="210" y="79" text-anchor="middle" fill="#C0392B" font-size="9">(learned weights)</text><line x1="80" y1="52" x2="150" y2="60" stroke="#8A6D3B"/><line x1="80" y1="82" x2="150" y2="74" stroke="#8A6D3B"/><rect x="330" y="52" width="60" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="360" y="71" text-anchor="middle" fill="#276b45" font-size="9">score</text><line x1="270" y1="67" x2="330" y2="67" stroke="#8A6D3B"/><text x="440" y="70" fill="#C0392B" font-size="10">slow 🐌</text><rect x="20" y="120" width="60" height="24" fill="#EAF0FA" stroke="#5E5191"/><text x="50" y="136" text-anchor="middle" fill="#5E5191" font-size="9">Q</text><rect x="20" y="150" width="60" height="24" fill="#FCF3DC" stroke="#C99A12"/><text x="50" y="166" text-anchor="middle" fill="#8A6D3B" font-size="9">K</text><rect x="150" y="128" width="120" height="38" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="210" y="151" text-anchor="middle" fill="#276b45" font-size="9">multiply &amp; add (dot)</text><line x1="80" y1="132" x2="150" y2="140" stroke="#8A6D3B"/><line x1="80" y1="162" x2="150" y2="154" stroke="#8A6D3B"/><rect x="330" y="132" width="60" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="360" y="151" text-anchor="middle" fill="#276b45" font-size="9">score</text><line x1="270" y1="147" x2="330" y2="147" stroke="#8A6D3B"/><text x="440" y="150" fill="#276b45" font-size="10">fast ⚡</text></g></svg>
%%%

**What the locksmith picture gets right:** both ways give you a fitness score, but one is a slow paid step and the other is instant — the same speed gap that made the field switch. **Where it breaks down:** a locksmith is genuinely more careful than a quick push, but here the fast dot product isn't sloppier — at scale it works just as well, so there was no real reason to keep the slow one.

#### Score the same pair both ways — watch the numbers
Let's score one Query–Key pair with each method and compare. Take Query `[2, 0, 1]` and Key `[3, 0, 1]`. **Predict first:** both should say "good match" — but which is fewer steps? Guess, then reveal.

%%% demo id=bahdanau label="additive vs dot product on the same pair"
code: Q=[2,0,1], K=[3,0,1]
out: ADDITIVE (old): glue Q,K -> feed 6 numbers into a tiny net -> multiply by learned weights, add a bias, bend, multiply again -> score ≈ 7   (many steps + extra learned weights)
     DOT (modern):   2*3 + 0*0 + 1*1 = 7   (one multiply-and-add, no extra weights)
take: <b>Same "7", far fewer steps.</b> Additive attention had to push the two lists through a little neural network — extra learned weights, several multiplies and a bend — just to land on a match number. The dot product reaches the same verdict with one multiply-and-add and *zero* extra weights. Run this a trillion times inside a big model and "one step vs many" is the whole ball game. That's why the modern form is a plain dot product.
%%%

!!! c-info 🔬
<b>Optional (skippable) — the one real trade-off.</b> Additive attention has its own small weight matrices, so in theory it can learn match patterns a bare dot product can't. In practice, once you give the Query and Key their own learned projections (Day 2), the plain dot product matches its quality at a fraction of the cost — so the field kept the cheap one. Cheaper *and* just as good is a rare win.
!!!

@@@ concept id=c7 tag="Two puzzles" title="Two ways raw scores misbehave — and their two neat fixes" gotit="Got both fixes"
Raw scores are almost the whole story, but two little puzzles can trip them up. The fun part: each has a one-move fix, and you've already met one of them. Let's solve both.

Think of **a smart speaker in a noisy kitchen**. Puzzle one: if everyone yells, the speaker can't tell voices apart — it just hears one wall of loud. Puzzle two: the speaker might obey the TV in the next room, or a voice it shouldn't be allowed to follow. The two fixes are just as homely: turn the volume down so voices stay distinct, and mute the sources it isn't allowed to hear. Attention's two fixes are exactly these — turn the scores down (scaling), and mute the words a query mustn't listen to (masking).

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two puzzles and their fixes. Left puzzle: scores too loud, softmax collapses to one spike; fix is divide by root d_k. Right puzzle: a query listens to a forbidden future or padding word; fix is masking, setting that score to minus infinity so its share becomes zero."><g font-family="monospace" font-size="11"><text x="130" y="16" text-anchor="middle" fill="#C0392B" font-size="11">Puzzle 1: too loud</text><rect x="40" y="34" width="14" height="70" fill="#FBE3E3" stroke="#C0392B"/><rect x="70" y="94" width="14" height="10" fill="#F3D9D9" stroke="#C0392B"/><rect x="100" y="98" width="14" height="6" fill="#F3D9D9" stroke="#C0392B"/><rect x="130" y="100" width="14" height="4" fill="#F3D9D9" stroke="#C0392B"/><text x="95" y="122" text-anchor="middle" fill="#C0392B" font-size="9">one spike, rest ~0</text><text x="95" y="140" text-anchor="middle" fill="#276b45" font-size="9">fix: ÷ √d_k</text><rect x="55" y="150" width="80" height="36" fill="#EAF5EE" stroke="#2D8B55"/><rect x="50" y="160" width="12" height="20" fill="#2D8B55"/><rect x="72" y="164" width="12" height="16" fill="#8Cc4a3"/><rect x="94" y="166" width="12" height="14" fill="#8Cc4a3"/><rect x="116" y="168" width="12" height="12" fill="#8Cc4a3"/><text x="95" y="199" text-anchor="middle" fill="#276b45" font-size="8">distinct again</text><line x1="260" y1="30" x2="260" y2="196" stroke="#E5DED2"/><text x="390" y="16" text-anchor="middle" fill="#C0392B" font-size="11">Puzzle 2: forbidden word</text><circle cx="330" cy="60" r="15" fill="#EAF0FA" stroke="#5E5191"/><text x="330" y="64" text-anchor="middle" fill="#5E5191" font-size="8">now</text><circle cx="450" cy="60" r="15" fill="#FBE3E3" stroke="#C0392B"/><text x="450" y="64" text-anchor="middle" fill="#C0392B" font-size="8">future</text><line x1="345" y1="60" x2="435" y2="60" stroke="#C0392B" stroke-width="2"/><text x="390" y="52" text-anchor="middle" fill="#C0392B" font-size="9">peeking! ✗</text><text x="390" y="110" text-anchor="middle" fill="#276b45" font-size="9">fix: mask → score = −∞</text><circle cx="330" cy="150" r="15" fill="#EAF0FA" stroke="#5E5191"/><text x="330" y="154" text-anchor="middle" fill="#5E5191" font-size="8">now</text><circle cx="450" cy="150" r="15" fill="#EEE" stroke="#B8AEA2" stroke-dasharray="3,2"/><text x="450" y="154" text-anchor="middle" fill="#9A938A" font-size="8">muted</text><line x1="345" y1="150" x2="435" y2="150" stroke="#B8AEA2" stroke-dasharray="4,3"/><text x="390" y="180" text-anchor="middle" fill="#276b45" font-size="9">share → 0</text></g></svg>
%%%

%%% hint
t1: Feeling stuck on why setting a score to negative infinity "mutes" a word? Go back to the pizza. Softmax hands out slices by how big each score is — so ask yourself what slice a score way down at the bottom would get.
t2: Try one tiny example. Scores are [river 3.0, FUTURE 2.0]. Now push FUTURE all the way down: [river 3.0, FUTURE −∞]. Softmax turns each score into a slice, and a score of −∞ turns into a slice of essentially 0 — so river keeps the whole pizza, and FUTURE gets nothing.
t3: In one sentence: masking sets a forbidden word's score to −∞ before softmax, so its slice comes out ~0 and the freed budget flows to the words that are allowed.
%%%

**What the noisy-kitchen picture gets right:** the two troubles really are "can't tell voices apart" and "listening to a source it shouldn't," fixed by lowering volume and muting — a clean map to scaling and masking. **Where it breaks down:** a speaker mutes a source by ignoring a microphone, but attention mutes a word with a number trick — it sets that word's score to negative infinity *before* softmax, so its slice comes out as essentially zero.

#### Puzzle 1 — scores get too loud (you met the fix already)
As lists grow long, scores swell. Feed swollen scores into softmax and it doesn't share nicely — it slams the whole budget onto the single biggest score (a **near one-hot spike**), and every other word gets almost nothing. Worse, when softmax is that lopsided, the learning signal (the gradient) shrinks toward zero, so the model stops improving. **The cause:** a dot product adds one product per slot, so more slots → bigger totals. **The fix you already know:** divide by √d_k. Turn the volume down, and softmax can share again.

#### Puzzle 2 — a word listens where it shouldn't
By default the grid scores *every* Query against *every* Key. But sometimes that's wrong: a word being generated must not peek at *future* words it hasn't produced yet, and no word should waste attention on empty [[padding||Blank filler slots added so every sentence in a batch is the same length. They hold no real word.]] slots (blank fillers added to make sentences the same length). **The cause:** the grid compares everything to everything. **The fix — [[masking||Setting forbidden scores to negative infinity before softmax, so their attention share comes out ~0.]]:** before softmax, set every forbidden score to negative infinity. Softmax turns a negative-infinity score into a share of about zero, so those words get muted — no peeking, no wasted attention.

**Predict first:** we score three words `[river=3.0, FUTURE=2.0, pad=1.0]`, but FUTURE and pad are forbidden. What should their shares be after we mask them? Guess, then reveal.

%%% demo id=mask label="softmax before vs after masking the forbidden words"
code: scores=[river 3.0, FUTURE 2.0, pad 1.0];  softmax(scores)  vs  softmax(mask(scores))
out: no mask -> river=0.66  FUTURE=0.24  pad=0.09    (FUTURE steals real attention!)
     masked  -> set FUTURE,pad = −∞ -> river=1.00  FUTURE=0.00  pad=0.00
take: <b>Masking drives the forbidden shares to ~0 and hands the freed budget to the allowed words.</b> Without it, "FUTURE" would have grabbed 0.24 of the budget — a full quarter of the word's attention spent on something it's not allowed to see. Set those scores to −∞ and softmax gives them essentially nothing, so river keeps the whole pizza. Same budget, honestly spent.
%%%

!!! c-ok ✅
<b>Both puzzles, both fixes, in one breath:</b> scores get too loud → divide by √d_k; a word peeks where it shouldn't → mask (set the score to −∞ before softmax). Two homely knobs, and raw scores become trustworthy.
!!!

@@@ concept id=c8 tag="Honest limits" title="What a score is — and what it quietly is not" gotit="Got the limits"
A great engineer knows exactly what a tool *can't* do. Scores are powerful, but they have two honest limits — knowing them now saves you real confusion later, and points straight at what the next days add.

Think of a **single thermometer reading**. A number like "22" tells you nothing on its own — 22 degrees? 22 out of 100? You only know what it means once you compare it to the whole scale and to the other readings. And a thermometer tells you *how warm*, never *what time* or *where in the room* — it simply doesn't carry that. A raw score is the same: meaningless until it's put next to the others, and blind to word order.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two honest limits of a score. Left: a lone score of 7 with a question mark, becoming meaningful only after softmax turns a whole row into shares. Right: the words river bank and bank river both produce the same set of scores, showing the score is blind to word order."><g font-family="monospace" font-size="11"><text x="130" y="16" text-anchor="middle" fill="#2C2A28" font-size="11">Limit 1: a lone score means nothing</text><text x="70" y="60" fill="#9A938A" font-size="20">7 ?</text><text x="130" y="60" fill="#5E5191" font-size="12">→ softmax the row →</text><text x="60" y="100" fill="#276b45" font-size="10">0.66 / 0.24 / 0.10</text><text x="130" y="126" text-anchor="middle" fill="#276b45" font-size="9">only now is it a share</text><line x1="260" y1="24" x2="260" y2="164" stroke="#E5DED2"/><text x="390" y="16" text-anchor="middle" fill="#2C2A28" font-size="11">Limit 2: blind to order</text><text x="300" y="56" fill="#3A342E" font-size="10">"river bank"</text><text x="410" y="56" fill="#3A342E" font-size="10">"bank river"</text><text x="355" y="86" text-anchor="middle" fill="#C0392B" font-size="10">same scores!</text><rect x="300" y="98" width="150" height="28" rx="4" fill="#FBE9E9" stroke="#C0392B"/><text x="375" y="116" text-anchor="middle" fill="#C0392B" font-size="9">dot product ignores position</text><text x="375" y="150" text-anchor="middle" fill="#5E5191" font-size="9">fixed later by positional encoding</text></g></svg>
%%%

**What the thermometer picture gets right:** a single reading is meaningless until you compare it to the scale and the others, and it carries no sense of time or place — exactly like a raw score. **Where it breaks down:** a thermometer *could* in principle be given a clock; a bare dot product truly has no way to sense position at all, which is why word order has to be added by a whole separate mechanism.

#### Limit 1 — a raw score is not a weight yet
Picture a single score popping up on screen: **7**. Is that a lot? You genuinely can't say. Now put it next to a neighbor's score, **40** — suddenly your 7 looks tiny, and after softmax its share collapses to almost nothing while the 40 grabs the budget. Flip it around: if the neighbor were **0.5** instead, the *same* 7 would now grab almost the whole pizza. Same number, opposite meaning — decided entirely by its company. That's the honest truth: a score means nothing **on its own**. It becomes a real share of attention only after softmax weighs the *whole row* together into slices that sum to 1. So never read one raw score as "how much attention" — it means nothing until the row is turned into a budget.

#### Limit 2 — scores carry no sense of order
Watch what the dot product actually does. Take the lists for **river** and **bank**, multiply them slot by slot, and add: one match number. Now swap the order — feed **bank** then **river**. Slot 1 times slot 1, slot 2 times slot 2, and so on — you multiply the exact same pairs of numbers, just visited in a different order, and adding them up doesn't care about order. So the total comes out *identical*. That means "river bank" and "bank river" score exactly the same. The dot product measures only *alignment* between a Query and a Key — it has no idea which word came first. Word order — which matters enormously — is injected by a separate trick called **positional encoding**, a forward-pointer to a later day. For today, just hold the honest truth: scores are **order-blind**.

!!! c-info 🗺️
<b>Forward-pointers (not today):</b> multiplying these shares into the <b>Value</b> vectors to get the final blended output → Day 4. Running several score-and-attend "meetings" at once (multi-head) → Day 4. Injecting word order (positional encoding) and the full softmax math → later days. Today you own the scoring heart; those add the body around it.
!!!

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Let's gather everything into one place you can come back to any time. And here's a fresh way to hold the whole day, nothing to do with dinner tables.

Think of **packing one lunchbox for the day**. The lunchbox is a fixed size — that's your attention **budget**, one whole and no more. First you decide how much you *want* each food (the snack rated high, the veggies rated low) — those wants are the **scores**. Then you actually pack: the box only holds so much, so you fit the foods in proportion to how much you wanted them, and when it's full, it's full — that packing step is **softmax**, turning wants into real, must-fit-in-one-box shares. If a food isn't allowed today (someone's allergic), you leave it out entirely — that's **masking**. **Where the lunchbox breaks down:** you pack a lunchbox once in the morning, but a transformer packs a fresh "budget" for every word, all at the same time — thousands of lunchboxes at once, which no morning routine could ever match.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A recap flow shown as packing a lunchbox. Step 1 score each Query against each Key with a dot product. Step 2 scale by dividing by root d_k. Step 3 mask forbidden words to minus infinity. Step 4 softmax to split one budget into shares that sum to one. Arrows connect the four steps left to right into a packed lunchbox."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The day in one line: score → scale → mask → split the budget</text><rect x="16" y="50" width="104" height="60" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="68" y="74" text-anchor="middle" fill="#5E5191">1. score</text><text x="68" y="92" text-anchor="middle" fill="#5E5191" font-size="8">Q · K (dot)</text><rect x="140" y="50" width="104" height="60" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="192" y="74" text-anchor="middle" fill="#8A6D3B">2. scale</text><text x="192" y="92" text-anchor="middle" fill="#8A6D3B" font-size="8">÷ √d_k</text><rect x="264" y="50" width="104" height="60" rx="8" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><text x="316" y="74" text-anchor="middle" fill="#C0392B">3. mask</text><text x="316" y="92" text-anchor="middle" fill="#C0392B" font-size="8">forbidden → −∞</text><rect x="388" y="50" width="116" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="446" y="74" text-anchor="middle" fill="#276b45">4. softmax</text><text x="446" y="92" text-anchor="middle" fill="#276b45" font-size="8">shares sum to 1</text><line x1="120" y1="80" x2="140" y2="80" stroke="#8A6D3B" stroke-width="1.5"/><line x1="244" y1="80" x2="264" y2="80" stroke="#8A6D3B" stroke-width="1.5"/><line x1="368" y1="80" x2="388" y2="80" stroke="#8A6D3B" stroke-width="1.5"/><text x="260" y="146" text-anchor="middle" fill="#5E5191" font-size="11">softmax(Q@Kᵀ / √d_k) → the attention budget, honestly split</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">next day: multiply these shares into the Values to get the blended output</text></g></svg>
%%%

#### The day in four beats
1. A **score** is one number: how much should this word listen to that word.
2. The **dot product** makes that number — multiply slot by slot, add up. Stack them into the **score matrix**, one row per word.
3. **Scale** by √d_k (turn the volume down) and **mask** forbidden words (set to −∞) so raw scores behave.
4. **Softmax** splits one attention **budget** into positive shares that sum to 1 — the biggest score gets the biggest slice.

#### Cheat-sheet — every word from today
%%% jargon
attention score | one number: how much a word should listen to another word
Query (Q) | the "what am I looking for?" side of a word
Key (K) | the "what do I offer?" side of a word
dot product | multiply two lists slot by slot, add up — one match number
score matrix | the grid of all pair scores; one row per asking word
√d_k | the amount we divide scores by to keep them from getting too loud
softmax | turns a row of scores into positive shares that sum to 1
budget | the one whole pot of attention softmax splits into shares
additive attention | the older, slower scorer that used a tiny neural network
masking | setting forbidden scores to −∞ so their share becomes ~0
%%%

#### The two fixes and two limits, side by side
%%% table
:: The move :: What it fixes / means
Divide by √d_k :: scores got too loud → softmax stops slamming into one spike
Mask (→ −∞) :: a word peeking at future/padding → its share becomes ~0
Limit: score ≠ weight :: a raw score means nothing until softmax normalizes the row
Limit: order-blind :: the dot product can't sense word order (fixed later by positional encoding)
%%%

You just wrote the heart of the transformer. Come back to this page whenever attention feels foggy — the four beats are all of it.

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: What is an attention score? | a:2 | a learned weight matrix | a whole sentence | one number: how much one word should listen to another | the final output vector | fb: One number per pair — how much to listen.
q: How do you get the score between a Query and a Key? | a:1 | run a big neural network | multiply them slot by slot and add (dot product) | take their average | count matching letters | fb: The dot product — multiply slot-by-slot, add.
q: Why divide scores by √d_k before softmax? | a:0 | to stop long lists making scores too loud (which jams softmax) | to add word order | to make scores negative | to speed up the GPU | fb: Longer lists → bigger scores; √d_k turns the volume down.
q: After softmax, the attention shares in a row… | a:3 | are all negative | can be any size | are the same as the raw scores | are positive and add up to 1 | fb: One budget, split into positive shares summing to 1.
%%%

@@@ produce id=produce tag="Produce" title="Build attention scoring by hand" gotit="Done"
Time to make it real with your own hands. In `experiment.py`, build the scoring pipeline on a tiny 3-word example and *watch* each step change the numbers.

Start with three little Query lists and three Key lists (just make up small numbers). **Predict before you run each step:**
1. Compute the score matrix `Q @ Kᵀ` by hand-style dot products. **Predict** which word will score highest against which — then print the grid and check.
2. Divide the whole grid by √d_k. **Notice** the scores get smaller but the *order* (who beats whom) never changes.
3. Run softmax on each row. **Observe** that every row now adds up to 1 — print the row sums to prove it.
4. Now mask one word: set its column to a very large negative number before softmax, and **watch** its share collapse to ~0 while the freed budget flows to the allowed words.

What you should see by the end: a row of raw match numbers turning into a clean budget of shares that sum to 1 — the exact move at the heart of every transformer. Write it in `experiment.py` and run it.

@@@ fin
