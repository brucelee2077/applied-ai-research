---
quest_id: wf4-d04-multihead
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 4 — Multi-Head Attention"
module_label: "Module 4 · Represent · Day 4"
title: "Multi-Head Attention"
subtitle: "Many Views at Once"
brand_sub: "Foundations · M4 Day 4"
spine: "panel"
nav_prev_href: "../day-03-attention-scores/lesson.html"
nav_prev_label: "Attention Scores &amp; Softmax"
nav_next_href: "../day-05-positional/lesson.html"
nav_next_label: "Positional Encoding &amp; RoPE"
fin_title: "Module 4 · Day 4 complete! 🏆"
fin_body: "Nice work — you've completed <b>Multi-Head Attention</b>. You saw why one meeting isn't enough: a <b>panel</b> of heads each listens for a different thing on its own slice, then their notes are stitched back together and mixed with one matrix.<br>Next up: <b>Positional Encoding</b> — teaching the model who spoke when."
notebook_yardstick: null
coverage_topics:
  - {topic: single head recap, keywords: [one head, single head, one meeting, one weighted blend]}
  - {topic: one blend averages relationships, keywords: [one average, blur together, only one thing, average away]}
  - {topic: multi head parallel own projections, keywords: [several heads, in parallel, own Query, own set]}
  - {topic: split d_model into h heads, keywords: [split, d_model, d_k, divide the vector, thinner slice]}
  - {topic: cost stays about the same, keywords: [same cost, not more work, roughly the same, no extra]}
  - {topic: per head specialization different views, keywords: [specialize, different relationship, one head tracks, subspace, different view]}
  - {topic: concatenate heads and output projection W_O, keywords: [concatenate, stitch, W_O, output projection, mix]}
  - {topic: scaled dot product reused inside every head, keywords: [same building block, inside every head, softmax, reuse]}
---

@@@ hero
@lede Imagine your whole family watches the same movie. Afterward you ask each person what it was about, and you get wonderfully different answers — your little sister only saw the funny dog, your dad followed the mystery, your mum caught the love story hiding underneath. Same movie, several sets of eyes, and together they tell you far more than any one person could alone. That everyday truth — many views of one thing beat a single view — is exactly the trick inside every transformer, and a big reason a model like ChatGPT can follow a long sentence so well. Yesterday you built ONE attention "meeting," where each word listens to the others and blends them into one answer. Today you'll feel why one meeting isn't enough — and how running a whole **panel** of them at once, each looking for something different, is the move that makes attention powerful.
@goal Together we'll see what one attention head can and can't do, then hand the job to a **panel** of heads working side by side — each with its own way of looking, each on its own slim slice of the vector, so the whole panel costs about the same as one head. We'll watch different heads quietly specialize on different relationships, then stitch their answers back together and mix them with one learned matrix. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.
%%% warmup
q: From yesterday — how do you get the attention score between one word's Query and another word's Key? | a:1 | run them through a big neural network | multiply them slot by slot and add it all up (the dot product) | count how many letters they share | take the average of the two lists | concept: dot-product | fb: The score is the dot product: multiply the two lists slot by slot, then add the products into one match number. Bigger total means a stronger match.
q: From yesterday — after softmax, what is true about a row of attention shares? | a:2 | they can be any size, even negative | they are the same as the raw scores | they are all positive and add up to exactly 1 | only the biggest one is kept, the rest are dropped | concept: softmax-budget | fb: Softmax splits one whole attention budget into positive shares that sum to 1 — like slicing a single pizza. The biggest score gets the biggest slice.
q: From yesterday — why do we divide the scores by √d_k before softmax? | a:0 | to stop long lists from making the scores too loud, which would jam softmax into one spike | to add a sense of word order | to make some scores negative | to speed up the GPU | concept: scaling | fb: Longer Query/Key lists pile on more products, so scores swell. Dividing by √d_k turns the volume back down so softmax can still share the budget nicely.
%%%

@@@ concept id=c1 tag="One meeting" title="Recap — what one attention head does" gotit="Got the one-head recap"
Let's warm up by remembering what you already built. Yesterday, one word looked around at all the other words, rated how much it wanted to listen to each one, and then blended them together into a single new answer. That whole one-shot process — score everyone, then mix them into one result — is called an attention [[head||One complete attention: one word rates all the others, then blends them into a single answer. Yesterday's whole "meeting" is one head.]]. One head, one meeting, one blended answer per word.

Think of **one meeting in a room**. Everyone talks, you decide how much to weigh each voice, and you walk out with a *single* summary in your head. That single summary is what one attention head hands back for each word. Everything today is about what happens when one meeting isn't enough — so we hold this whole meeting in our hand as the piece we're about to copy.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="One attention head shown as a single meeting. One word in the middle listens to three other words around a table, weighs them, and produces one blended answer coming out at the bottom."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One head = one meeting → one blended answer</text><ellipse cx="200" cy="96" rx="120" ry="50" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="200" y="100" text-anchor="middle" fill="#8A6D3B" font-size="9">one meeting</text><circle cx="200" cy="60" r="15" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="200" y="64" text-anchor="middle" fill="#5E5191" font-size="9">word</text><circle cx="110" cy="110" r="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="110" y="114" text-anchor="middle" fill="#276b45" font-size="8">river</text><circle cx="200" cy="126" r="14" fill="#FDF9F3" stroke="#B8AEA2"/><text x="200" y="130" text-anchor="middle" fill="#9A938A" font-size="8">the</text><circle cx="290" cy="110" r="14" fill="#FDF9F3" stroke="#B8AEA2"/><text x="290" y="114" text-anchor="middle" fill="#9A938A" font-size="8">bank</text><line x1="188" y1="70" x2="122" y2="102" stroke="#2D8B55" stroke-width="3"/><line x1="200" y1="75" x2="200" y2="112" stroke="#B8AEA2" stroke-width="1"/><line x1="212" y1="70" x2="278" y2="102" stroke="#C99A12" stroke-width="1.5"/><path d="M200 141 L200 168" stroke="#5E5191" stroke-width="1.5" marker-end="url(#a1)"/><defs><marker id="a1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#5E5191"/></marker></defs><rect x="360" y="78" width="140" height="36" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="430" y="100" text-anchor="middle" fill="#276b45" font-size="9">one blended answer</text><line x1="216" y1="168" x2="360" y2="100" stroke="#5E5191" stroke-dasharray="2,2"/></g></svg>
%%%

**What the meeting picture gets right:** you weigh many voices and leave with one summary — that is exactly what a head does for each word. **Where it breaks down:** a real meeting happens over time, one voice after another, but a head weighs *all* the words at the same instant, which no room full of people could do.

#### The one head, in one breath
Inside that head, three roles do the work — you named them on Day 2. A word's [[Query||The "what am I looking for?" side of a word.]] asks "what am I looking for?", each word's [[Key||The "what do I offer?" side of a word.]] answers "here's what I've got," and each word's [[Value||The actual content a word carries — the stuff that gets blended into the answer.]] is the actual content that gets blended in. The head scores each Query against each Key, softmaxes those scores into shares that add up to one, and uses those shares to blend the Values into a single answer. That's the whole meeting — and today we make copies of it.

!!! c-info 🗺️
<b>The one line to hold all day:</b> one head gives one blended answer per word; a <b>panel</b> of heads gives many answers at once, then stitches them into one. Hold that; the rest is slow motion.
!!!

@@@ concept id=c2 tag="One view isn't enough" title="Why one head has to blur things together" gotit="Got the limit"
Here's the puzzle that makes today exciting. A single head is only allowed to hand back *one* blended answer per word. But a real sentence carries several different relationships at the same time. Take "the tired **cat** sat on the mat." For the word "cat," a few different things matter at once: *which word describes it* (tired), *who did the sitting* (cat), *where it happened* (mat). One head, one blend — so it is forced to smush all of those into a single average. The important patterns get blurred together into one muddy answer.

Think of **one photo of a whole birthday party**. If you can only take a single wide shot, you catch everyone — but nobody is sharp. The kid blowing out candles is a little blurry, the gift pile in the corner is a little blurry, the dog under the table is a little blurry. To get every part crisp, you'd want *several* photos, each focused on one thing. One head is that single wide shot: it has to average everything into one frame, so no single relationship comes out sharp.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A birthday party captured in one blurry wide photo on the left versus three sharp close-up photos on the right. The single photo blurs the candle-blower, the gifts, and the dog together. The three photos each show one of them clearly."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One wide shot blurs everything · three shots each stay sharp</text><rect x="26" y="40" width="150" height="120" rx="8" fill="#F1ECE3" stroke="#B8AEA2" stroke-width="1.5"/><text x="101" y="34" text-anchor="middle" fill="#9A938A" font-size="9">1 photo (one head)</text><text x="101" y="80" text-anchor="middle" fill="#C0A9A0" font-size="10">candles…</text><text x="101" y="104" text-anchor="middle" fill="#C0A9A0" font-size="10">gifts…</text><text x="101" y="128" text-anchor="middle" fill="#C0A9A0" font-size="10">dog…</text><text x="101" y="152" text-anchor="middle" fill="#C0392B" font-size="9">all blurry</text><text x="205" y="104" fill="#8A6D3B" font-size="16">vs</text><rect x="250" y="40" width="80" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="290" y="70" text-anchor="middle" fill="#276b45" font-size="9">candles ✓</text><rect x="340" y="40" width="80" height="52" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="380" y="70" text-anchor="middle" fill="#5E5191" font-size="9">gifts ✓</text><rect x="430" y="40" width="70" height="52" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="465" y="70" text-anchor="middle" fill="#8A6D3B" font-size="9">dog ✓</text><text x="375" y="132" text-anchor="middle" fill="#276b45" font-size="9">3 photos (a panel): each one sharp</text><text x="375" y="154" text-anchor="middle" fill="#9A938A" font-size="9">same party, several focused views</text></g></svg>
%%%

**What the birthday-photo picture gets right:** one wide shot has to average the whole scene, so every part comes out soft — exactly how one head blurs the different relationships into one answer. **Where it breaks down:** a photo blurs because of a lens and light, while a head "blurs" by doing math — it literally averages several patterns into one set of numbers, no camera involved.

#### Watch one head average two relationships into mush
Let's make the blur concrete. Say the word "cat" cares about two things: the word that *describes* it ("tired") and the word for *where* it sat ("mat"). Two clean relationships. **Predict first:** if a single head must fold both into one blend, what happens to each one? Guess, then reveal.

%%% demo id=average label="one head folds two relationships into one blend"
code: head_1 must answer "cat" using BOTH: describe->tired  AND  place->mat
out: describe-only would give:  [tired 1.00, mat 0.00]   (crisp: it's the adjective)
     place-only would give:     [tired 0.00, mat 1.00]   (crisp: it's the location)
     ONE head must do both  ->  [tired 0.50, mat 0.50]   (muddy: half-and-half, neither is sharp)
take: <b>One head can only give one blend, so it splits the difference — 0.50 / 0.50.</b> Each relationship on its own was crisp, but folding both into a single answer averages them into mush: the model can no longer tell "which word describes cat" apart from "where cat sat." The fix isn't a cleverer single blend — it's to stop asking one head to do two jobs. Give each relationship its OWN head, and each stays sharp.
%%%

That is the whole reason multi-head attention exists. You just found the limit that the rest of the day repairs — a real victory lap, because now the fix will feel obvious.

@@@ concept id=c3 tag="A panel of heads" title="Run several heads at once — each with its own eyes" gotit="Got the panel"
Here's the fix, and it's beautifully simple: don't make one head do everything — run *several* heads at the same time, and let each one look for something different. This is called [[multi-head attention||Running several attention heads in parallel, each with its own Query/Key/Value projections, then combining their answers.]]. Each head is a full copy of yesterday's meeting, but with its OWN learned way of forming Queries, Keys, and Values — so each head can quietly choose a different thing to pay attention to. Because they run side by side, we call the group a **panel**.

Think of a **panel of judges at a talent show**. All three watch the same act, but one judge scores the singing, one scores the dancing, one scores the stage presence. They don't argue over a single number — each brings back their own score for their own thing. A panel of attention heads works the same way: every head sees the same sentence, but each has learned to notice a different relationship, and each returns its own answer.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="One sentence fans out to a panel of three attention heads. Each head has its own Query, Key, and Value, and each returns its own answer. Head one tracks describing words, head two tracks where-it-happened, head three tracks who-did-what."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One sentence → a panel of heads, each with its own eyes</text><rect x="200" y="30" width="120" height="26" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="260" y="47" text-anchor="middle" fill="#6B645E" font-size="9">the tired cat sat</text><line x1="230" y1="56" x2="110" y2="86" stroke="#8A6D3B"/><line x1="260" y1="56" x2="260" y2="86" stroke="#8A6D3B"/><line x1="290" y1="56" x2="410" y2="86" stroke="#8A6D3B"/><rect x="40" y="88" width="140" height="72" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="110" y="106" text-anchor="middle" fill="#276b45" font-size="10" font-weight="bold">head 1</text><text x="110" y="124" text-anchor="middle" fill="#276b45" font-size="8">own Q · K · V</text><text x="110" y="140" text-anchor="middle" fill="#276b45" font-size="8">describing words</text><rect x="190" y="88" width="140" height="72" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="260" y="106" text-anchor="middle" fill="#5E5191" font-size="10" font-weight="bold">head 2</text><text x="260" y="124" text-anchor="middle" fill="#5E5191" font-size="8">own Q · K · V</text><text x="260" y="140" text-anchor="middle" fill="#5E5191" font-size="8">where it happened</text><rect x="340" y="88" width="140" height="72" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="410" y="106" text-anchor="middle" fill="#8A6D3B" font-size="10" font-weight="bold">head 3</text><text x="410" y="124" text-anchor="middle" fill="#8A6D3B" font-size="8">own Q · K · V</text><text x="410" y="140" text-anchor="middle" fill="#8A6D3B" font-size="8">who did what</text><text x="110" y="182" text-anchor="middle" fill="#276b45" font-size="9">answer 1</text><text x="260" y="182" text-anchor="middle" fill="#5E5191" font-size="9">answer 2</text><text x="410" y="182" text-anchor="middle" fill="#8A6D3B" font-size="9">answer 3</text><text x="260" y="204" text-anchor="middle" fill="#9A938A" font-size="9">three answers, all at the same time</text></g></svg>
%%%

**What the judges picture gets right:** everyone watches the same act, yet each returns their own score for their own thing — exactly like heads each returning their own answer for their own relationship. **Where it breaks down:** the judges *know* their assigned category ahead of time, but nobody tells a head what to specialize in — it discovers its own job on its own while the model trains.

#### The key phrase: each head has its OWN Q, K, V
This is the one detail that makes heads different from each other. On Day 2 you learned a word becomes a Query, a Key, and a Value by passing through three little learned recipes (the projections `W_Q`, `W_K`, `W_V`). In multi-head attention, **every head gets its OWN set** of these three recipes. Same input word, different recipes, so head 1 might turn "cat" into a Query that looks for describing words, while head 2 turns the same "cat" into a Query that looks for the place word. Different eyes, same sentence. **Predict first:** send the same word "cat" through two heads' recipes — will they land on the same word or different ones? Guess, then reveal.

%%% demo id=ownqkv label="the SAME word through two different heads"
code: word "cat" -> head_1's own W_Q  vs  head_2's own W_Q
out: head_1 Query -> "I'm hunting for a word that DESCRIBES me"  -> lands on "tired"
     head_2 Query -> "I'm hunting for WHERE I happened"          -> lands on "mat"
take: <b>Same word, two heads, two totally different questions.</b> Because each head owns its projections, the identical input "cat" becomes two different Queries — so the two heads chase two different relationships and never blur them together. The muddy 0.50 / 0.50 from the last concept is gone: head 1 gives a crisp "tired," head 2 gives a crisp "mat." That is the panel doing what one head couldn't.
%%%

@@@ concept id=c4 tag="Same budget, sliced" title="Splitting the vector — many heads for the price of one" gotit="Got the split"
Now for the surprise that makes multi-head almost free. You might worry: if we run 8 heads instead of 1, doesn't that cost 8 times as much? Happily, no — and the reason is a neat trick. Instead of giving every head the *full* width of a word's vector, we **cut the width into equal slices** and hand one slice to each head. So the whole panel works on the same total amount of numbers as a single full-width head would. More heads, same budget — just sliced thinner.

Think of **one pizza cut into slices**. Whether you cut it into 4 big slices or 8 small ones, it's still the same one pizza — you never get more (or less) pizza by slicing. Cutting it into more pieces just means each piece is smaller. The word's vector is the pizza: its full width is called [[d_model||The full width of a word's vector — the total number of slots the model gives each word.]]. Slice it among `h` heads and each head gets a thinner piece of width [[d_k||The per-head width: the full width d_model split evenly across the heads, so d_k = d_model / h.]] `= d_model / h`. Same pizza, more slices, each slice smaller.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A word vector of width d_model equals 12 drawn as one long bar. Below, the same bar is cut into three equal slices of width d_k equals 4, each slice a different color for head 1, head 2, and head 3. The total width is unchanged."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same vector width, cut into equal head slices (nothing added)</text><text x="30" y="44" fill="#6B645E" font-size="9">one full vector:  d_model = 12</text><rect x="30" y="50" width="360" height="30" rx="4" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="210" y="70" text-anchor="middle" fill="#9A938A" font-size="9">12 slots, all to one head</text><text x="30" y="116" fill="#6B645E" font-size="9">cut into 3 heads:  d_k = 12 ÷ 3 = 4 each</text><rect x="30" y="122" width="120" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="90" y="142" text-anchor="middle" fill="#276b45" font-size="9">head 1 · 4</text><rect x="150" y="122" width="120" height="30" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="210" y="142" text-anchor="middle" fill="#5E5191" font-size="9">head 2 · 4</text><rect x="270" y="122" width="120" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="330" y="142" text-anchor="middle" fill="#8A6D3B" font-size="9">head 3 · 4</text><text x="430" y="70" fill="#9A938A" font-size="9">total: 12</text><text x="430" y="142" fill="#276b45" font-size="9">total: 12</text><text x="210" y="172" text-anchor="middle" fill="#9A938A" font-size="9">4 + 4 + 4 = 12 — the width never grew</text></g></svg>
%%%

%%% hint
t1: Don't reach for the formula yet — just picture the pizza. You start with ONE pizza (the full vector). Cutting it into more slices never gives you more pizza; each slice just gets smaller. So ask yourself: if the total pizza stays the same, and each head only gets a slice, what happens to the size of each slice as you add more heads?
t2: Put tiny numbers on it. Say the full width is d_model = 12 and you use h = 3 heads. Each head's slice is d_k = 12 ÷ 3 = 4. The work for the panel is (number of heads) × (each head's width) = 3 × 4 = 12. Now double the heads to h = 6: each slice halves to d_k = 12 ÷ 6 = 2, and the work is 6 × 2 = 12 again. The two numbers always multiply back to 12.
t3: Because d_k = d_model ÷ h, the product h × d_k always lands back on d_model — so adding heads just re-slices one fixed budget instead of buying more, and the total cost stays about the same.
%%%

**What the pizza picture gets right:** slicing changes how many pieces you have, never how much pizza there is — just like slicing the vector gives more heads without adding width. **Where it breaks down:** pizza slices are separate once cut, but the head slices are still part of one vector — they get stitched back into a full-width vector at the end, something you can't do with actual pizza.

#### Watch the cost stay flat as heads go up
The one-line rule: `d_k = d_model / h`. In words — the per-head width is the full width divided by the number of heads. **Predict first:** if a model has width `d_model = 64` and we go from 1 head to 8 heads, does the total amount of work roughly grow, shrink, or stay the same? Guess, then reveal.

%%% demo id=cost label="does adding heads cost more?"
code: d_model = 64;  compare 1 head vs 8 heads
out: 1 head :  d_k = 64/1 = 64   ->  work ∝ heads × d_k = 1 × 64 = 64
     8 heads:  d_k = 64/8 = 8    ->  work ∝ heads × d_k = 8 × 8  = 64
take: <b>Same total, 64 either way — more heads did NOT cost more.</b> Each extra head is paid for by making every head's slice thinner, so `heads × d_k` always lands back on `d_model`. The total work stays roughly the same no matter how many heads you use — you are re-slicing one fixed budget, not buying more. That is exactly why real models can afford 8, 16, even 128 heads without the bill exploding — the panel is (almost) as cheap as a single head.
%%%

!!! c-info 💡
<b>Optional (skippable) — the shape story.</b> One head on the full width would multiply by projection matrices of size `d_model × d_model`. Split into `h` heads and each head's projections are `d_model × d_k` with `d_k = d_model/h`; stack all `h` of them and the total size is `d_model × (h·d_k) = d_model × d_model` again. The arithmetic lands right back where it started — which is the whole point. You never need to track this to use attention; it's here for the curious.
!!!

@@@ concept id=c5 tag="Each head specializes" title="Different heads learn different views" gotit="Got specialization"
Now the delightful part — what heads actually *do* once they're trained. Because each head has its own eyes and its own thin slice to work in, each one drifts toward its own favorite kind of relationship. Nobody assigns the jobs. During training, one head quietly settles into tracking who-does-what, another leans toward the word right next door, another links describing words to what they describe. Each head becomes a little specialist looking at its own **view** of the same sentence — a different [[subspace||The head's own small slice of the vector — a little "room" of numbers where it does its looking.]], its own private room to think in.

Think of a **set of colored highlighters on one page**. You read the same paragraph three times: once with a yellow highlighter marking the names, once with green marking the places, once with pink marking the dates. Same page, three passes, three different things lit up. Put the marked-up pages side by side and you understand the paragraph far better than any single color could show you. Each head is one highlighter color — it lights up its own pattern in the sentence.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The same sentence read three times with three highlighter colors. Pass one, yellow, marks the name Ana. Pass two, green, marks the place Mars. Pass three, pink, marks the date 2035. Same sentence, three passes, three different patterns lit up — combined they give a richer read than any single color alone."><g font-family="monospace" font-size="12"><text x="260" y="16" text-anchor="middle" fill="#2C2A28">One page, three highlighter passes — each lights up its own pattern</text><rect x="40" y="36" width="46" height="20" rx="3" fill="#FBE7A1"/><text x="46" y="51" fill="#3A342E">Ana</text><text x="96" y="51" fill="#6B645E">reached</text><text x="176" y="51" fill="#6B645E">Mars</text><text x="216" y="51" fill="#6B645E">in</text><text x="244" y="51" fill="#6B645E">2035</text><text x="330" y="51" fill="#9A7208" font-size="10">→ head 1: names</text><rect x="170" y="72" width="52" height="20" rx="3" fill="#CDEBD6"/><text x="46" y="87" fill="#6B645E">Ana</text><text x="96" y="87" fill="#6B645E">reached</text><text x="176" y="87" fill="#276b45">Mars</text><text x="216" y="87" fill="#6B645E">in</text><text x="244" y="87" fill="#6B645E">2035</text><text x="330" y="87" fill="#276b45" font-size="10">→ head 2: places</text><rect x="240" y="108" width="48" height="20" rx="3" fill="#F6D6E0"/><text x="46" y="123" fill="#6B645E">Ana</text><text x="96" y="123" fill="#6B645E">reached</text><text x="176" y="123" fill="#6B645E">Mars</text><text x="216" y="123" fill="#6B645E">in</text><text x="244" y="123" fill="#A83D6B">2035</text><text x="330" y="123" fill="#A83D6B" font-size="10">→ head 3: dates</text><text x="260" y="152" text-anchor="middle" fill="#5E5191" font-size="10">same sentence · three views · combine for a far richer read than one color alone</text></g></svg>
%%%

%%% svg
<svg id="mh-svg" viewBox="0 0 520 208" role="img" aria-label="Interactive multi-head split. One model vector of width 8 splits into h equal colored slices, each of width d_k equals 8 divided by h. Below, one labelled row per head names the different relationship that head can specialize in. Drag the slider to change the number of heads between 1, 2, 4 and 8 and watch the slices recolor and the heads switch on."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One vector (d_model = 8) → split into h heads, each d_k = 8 ÷ h wide</text><rect id="mh-c0" x="44" y="40" width="48" height="30" fill="#2D8B55" stroke="#fff"/><rect id="mh-c1" x="96" y="40" width="48" height="30" fill="#2D8B55" stroke="#fff"/><rect id="mh-c2" x="148" y="40" width="48" height="30" fill="#2D8B55" stroke="#fff"/><rect id="mh-c3" x="200" y="40" width="48" height="30" fill="#2D8B55" stroke="#fff"/><rect id="mh-c4" x="252" y="40" width="48" height="30" fill="#2A7B9B" stroke="#fff"/><rect id="mh-c5" x="304" y="40" width="48" height="30" fill="#2A7B9B" stroke="#fff"/><rect id="mh-c6" x="356" y="40" width="48" height="30" fill="#2A7B9B" stroke="#fff"/><rect id="mh-c7" x="408" y="40" width="48" height="30" fill="#2A7B9B" stroke="#fff"/><text x="250" y="88" text-anchor="middle" fill="#9A938A" font-size="10">the 8 numbers of one word's vector, sliced into heads</text><rect id="mh-sw0" x="44" y="106" width="13" height="13" fill="#2D8B55"/><text id="mh-tx0" x="63" y="117" fill="#3A342E" font-size="10">head 1: looks at the word just before</text><rect id="mh-sw1" x="44" y="128" width="13" height="13" fill="#2A7B9B"/><text id="mh-tx1" x="63" y="139" fill="#3A342E" font-size="10">head 2: looks ahead to what comes next</text><rect id="mh-sw2" x="44" y="150" width="13" height="13" fill="#C99A12" opacity="0.15"/><text id="mh-tx2" x="63" y="161" fill="#3A342E" font-size="10" opacity="0.18">head 3: keeps its focus on itself</text><rect id="mh-sw3" x="44" y="172" width="13" height="13" fill="#7C6DAA" opacity="0.15"/><text id="mh-tx3" x="63" y="183" fill="#3A342E" font-size="10" opacity="0.18">head 4: links the subject to its verb</text><rect id="mh-sw4" x="284" y="106" width="13" height="13" fill="#C0392B" opacity="0.15"/><text id="mh-tx4" x="303" y="117" fill="#3A342E" font-size="10" opacity="0.18">head 5: tracks the overall topic</text><rect id="mh-sw5" x="284" y="128" width="13" height="13" fill="#8A6D3B" opacity="0.15"/><text id="mh-tx5" x="303" y="139" fill="#3A342E" font-size="10" opacity="0.18">head 6: matches quotes and brackets</text><rect id="mh-sw6" x="284" y="150" width="13" height="13" fill="#2f8f6f" opacity="0.15"/><text id="mh-tx6" x="303" y="161" fill="#3A342E" font-size="10" opacity="0.18">head 7: catches a long-range echo</text><rect id="mh-sw7" x="284" y="172" width="13" height="13" fill="#B87333" opacity="0.15"/><text id="mh-tx7" x="303" y="183" fill="#3A342E" font-size="10" opacity="0.18">head 8: notices rare-word detail</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>number of heads <input id="mh-h" type="range" min="0" max="3" step="1" value="1" style="width:46%;accent-color:#5E5191;vertical-align:middle"></label>
<div id="mh-out" style="margin-top:6px;color:#2C2A28"><b>h = 2 heads</b> · each head is d_k = 8 ÷ 2 = <b>4</b> dims wide · 2 × 4 = 8 → same total width, so about the <b>same cost</b> as one big head, but 2 different views.</div>
</div>
<script>(function(){
  var sl=document.getElementById('mh-h');if(!sl)return;
  var out=document.getElementById('mh-out');
  var HS=[1,2,4,8];
  var PAL=['#2D8B55','#2A7B9B','#C99A12','#7C6DAA','#C0392B','#8A6D3B','#2f8f6f','#B87333'];
  function paint(){
    var h=HS[+sl.value], dk=8/h;
    for(var i=0;i<8;i++){var c=document.getElementById('mh-c'+i);if(c){c.setAttribute('fill',PAL[Math.floor(i/dk)]);}}
    for(var k=0;k<8;k++){var sw=document.getElementById('mh-sw'+k),tx=document.getElementById('mh-tx'+k);var on=k<h;
      if(sw)sw.setAttribute('opacity',on?'1':'0.15');
      if(tx)tx.setAttribute('opacity',on?'1':'0.18');}
    if(h===1){out.innerHTML='<b>h = 1 head</b> · d_k = 8 · just <b>one view</b> — it has to blur every relationship into a single blend.';}
    else{out.innerHTML='<b>h = '+h+' heads</b> · each head is d_k = 8 ÷ '+h+' = <b>'+dk+'</b> dims wide · '+h+' × '+dk+' = 8 → same total width, so about the <b>same cost</b> as one big head, but '+h+' different views.';}
  }
  sl.addEventListener('input',paint);paint();
})();</script>
%%%

**What the highlighter picture gets right:** several passes over one page, each lighting up a different pattern, combine into a richer understanding — exactly what a panel of specialized heads does. **Where it breaks down:** you *choose* what each highlighter marks, but a head is never told its color — it discovers its specialty on its own while the model trains, and different training runs can hand the same job to different heads.

#### Play with it — drag the head count and watch the views multiply
Try the widget above. Set it to **1 head** first: the whole vector is one color and you get a single view — one relationship, just like our blurry photo. Now step up to **2, then 4, then 8** heads and watch two things happen at once: the one long vector splits into that many colored slices (each one thinner — that's d_k shrinking), and that many heads switch on, each free to specialize in a different relationship. That is "many views at once," and you're driving it by hand.

!!! c-ok ✅
<b>The payoff:</b> because heads specialize, a model with 8 heads can track 8 different relationships in a sentence at the same time — grammar, nearby words, long-range links, and more — instead of blurring them all into one average. That richness is a real reason transformers understand language so well.
!!!

@@@ concept id=c6 tag="Stitch it back" title="Concatenate the heads, then mix with one matrix" gotit="Got concat and W_O"
We have a panel of heads, each with its own answer. But the rest of the model expects *one* vector per word, not a pile of separate head answers. So we do two small, tidy moves to bring the panel back together. First we **join** the head answers into one long vector. Then we run that long vector through one more learned recipe that lets the heads' findings mix and talk to each other. That last mixing recipe is a matrix called [[W_O||The learned output projection: one matrix that mixes the joined head answers into the final single vector.]] (the "output" projection).

Think of a **group project**. Three teammates each write their own section — one does the intro, one the research, one the conclusion. First you **staple** the three sections together into one document (that's joining them). But a stapled stack still reads like three separate voices. So one editor reads the whole thing and rewrites it into a single smooth report where the parts refer to each other. Joining is the staple; the editor is `W_O` — the step that lets the separate pieces become one coherent answer.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Three head answers, colored green blue and gold, are concatenated into one long vector, which then passes through a box labelled W_O and comes out as one mixed final vector."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Join the head answers → mix them with W_O → one final vector</text><text x="70" y="46" text-anchor="middle" fill="#6B645E" font-size="9">3 head answers</text><rect x="30" y="54" width="34" height="30" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="47" y="74" text-anchor="middle" fill="#276b45" font-size="8">h1</text><rect x="66" y="54" width="34" height="30" rx="3" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="83" y="74" text-anchor="middle" fill="#5E5191" font-size="8">h2</text><rect x="102" y="54" width="34" height="30" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="119" y="74" text-anchor="middle" fill="#8A6D3B" font-size="8">h3</text><text x="180" y="74" fill="#8A6D3B" font-size="13">join →</text><text x="290" y="46" text-anchor="middle" fill="#6B645E" font-size="9">one long vector</text><rect x="228" y="54" width="34" height="30" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1"/><rect x="262" y="54" width="34" height="30" fill="#EAF0FA" stroke="#5E5191" stroke-width="1"/><rect x="296" y="54" width="34" height="30" fill="#FCF3DC" stroke="#C99A12" stroke-width="1"/><rect x="352" y="50" width="54" height="38" rx="6" fill="#F4EBFB" stroke="#7c3aed" stroke-width="1.5"/><text x="379" y="73" text-anchor="middle" fill="#7c3aed" font-size="11" font-weight="bold">W_O</text><line x1="330" y1="69" x2="352" y2="69" stroke="#8A6D3B" stroke-width="1.5"/><rect x="426" y="54" width="66" height="30" rx="4" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="1.5"/><text x="459" y="74" text-anchor="middle" fill="#1F6280" font-size="9">final</text><line x1="406" y1="69" x2="426" y2="69" stroke="#8A6D3B" stroke-width="1.5"/><text x="379" y="118" text-anchor="middle" fill="#7c3aed" font-size="9">the editor: lets the heads' findings mix</text><text x="260" y="150" text-anchor="middle" fill="#9A938A" font-size="9">join keeps the pieces separate · W_O blends them into one voice</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">width in = width out = d_model — one vector per word, as before</text></g></svg>
%%%

**What the group-project picture gets right:** stapling keeps the sections in one place but still separate, and the editor is what turns them into one coherent voice — exactly the join-then-`W_O` pair. **Where it breaks down:** an editor rewrites with judgment and taste, while `W_O` just multiplies numbers by learned weights — no opinions, only a learned mix that training tuned.

#### Follow the widths, end to end
Here's the whole pipeline in one predict-then-run. **Predict first:** start with `d_model = 6` and `h = 3` heads. What width does each head answer with, what width after joining them, and what width after `W_O`? Guess three numbers, then reveal.

%%% demo id=concat label="the width from split to join to W_O"
code: d_model = 6, h = 3;  split -> attend -> join -> W_O
out: split :  each head works in d_k = 6/3 = 2      -> 3 answers of width 2
     join  :  stitch them side by side              -> one vector of width 2+2+2 = 6
     W_O   :  multiply the width-6 vector by W_O     -> final width 6 (one vector per word)
take: <b>Width 6 in, width 6 out — the panel folds neatly back to the model width.</b> Each head shrank to width 2 so the panel stayed cheap; joining puts the 2's back together into 6; then `W_O` mixes those joined pieces so the heads can share what they found — and hands back exactly the width the next layer expects. Split to be many, join to be one. Did your three guesses match?
%%%

You now hold the entire mechanism: **split into a panel, let each head specialize, join their answers, mix with `W_O`.** That is multi-head attention, start to finish.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Let's gather the whole day into one place you can come back to any time. And here's a fresh way to hold it, nothing to do with movies or judges.

Think of **a band playing one song**. The song is the sentence. Each musician — drummer, bass, guitar — plays their own part on their own instrument at the same time; that's the **panel** of heads, each on its own slice, each hearing the song differently. One musician alone would flatten the song into a single line (that's the one-head blur). Because they split the parts, adding more players doesn't make the song longer — same three minutes, richer sound (that's `d_k = d_model / h`, cost stays flat). Finally the mixing desk blends all the tracks into one recording you can actually play — that's **join + `W_O`**. **Where the band breaks down:** musicians rehearse their parts on purpose, but heads are never told their parts — they find them on their own while the model trains.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A recap flow in four steps. Step 1 one head blurs relationships together. Step 2 split into a panel of heads each with its own Q K V on a thinner slice. Step 3 each head specializes on a different relationship. Step 4 join the heads and mix with W_O into one final vector. Arrows connect the four steps left to right."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The day in one line: one head blurs → a panel of heads → each specializes → join + W_O</text><rect x="12" y="52" width="108" height="60" rx="8" fill="#F1ECE3" stroke="#B8AEA2" stroke-width="1.5"/><text x="66" y="76" text-anchor="middle" fill="#9A938A">1 head</text><text x="66" y="94" text-anchor="middle" fill="#9A938A" font-size="8">one blend = blur</text><rect x="138" y="52" width="108" height="60" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="192" y="76" text-anchor="middle" fill="#5E5191">split</text><text x="192" y="94" text-anchor="middle" fill="#5E5191" font-size="8">h heads · d_k=d_model/h</text><rect x="264" y="52" width="108" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="318" y="76" text-anchor="middle" fill="#276b45">specialize</text><text x="318" y="94" text-anchor="middle" fill="#276b45" font-size="8">each own view</text><rect x="390" y="52" width="118" height="60" rx="8" fill="#F4EBFB" stroke="#7c3aed" stroke-width="1.5"/><text x="449" y="76" text-anchor="middle" fill="#7c3aed">join + W_O</text><text x="449" y="94" text-anchor="middle" fill="#7c3aed" font-size="8">mix → one vector</text><line x1="120" y1="82" x2="138" y2="82" stroke="#8A6D3B" stroke-width="1.5"/><line x1="246" y1="82" x2="264" y2="82" stroke="#8A6D3B" stroke-width="1.5"/><line x1="372" y1="82" x2="390" y2="82" stroke="#8A6D3B" stroke-width="1.5"/><text x="260" y="146" text-anchor="middle" fill="#5E5191" font-size="11">many views at once → stitched into one answer</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">inside every head is yesterday's exact attention: score → softmax → blend</text></g></svg>
%%%

#### The day in four beats
1. **One head blurs.** A single attention head must fold every relationship into one blend, so nothing stays sharp — the birthday-photo problem.
2. **A panel fixes it.** Run several heads in parallel, each with its OWN Query/Key/Value recipes, so each can chase a different relationship at once.
3. **Slicing keeps it cheap.** Split the width across heads — `d_k = d_model / h` — so more heads cost about the same as one. Inside each head runs yesterday's exact attention: score, softmax, blend the Values.
4. **Join + `W_O` bring it home.** Stitch the head answers into one long vector, then mix them with the learned matrix `W_O` — one final vector per word, ready for the next layer.

#### Cheat-sheet — every word from today
%%% jargon
head | one complete attention (yesterday's whole meeting): score → softmax → blend
multi-head attention | running several heads in parallel, each with its own Q/K/V
d_model | the full width of a word's vector — the model size
d_k | the per-head width: d_model split across heads, so d_k = d_model / h
h | the number of heads (real models use 8 to 128)
subspace | the head's own small slice of the vector — its private room to look in
concatenate | join the heads' answers side by side into one long vector
W_O | the learned output matrix that mixes the joined heads into one final vector
%%%

#### One head vs a panel, side by side
%%% table
:: One head :: A panel of heads (multi-head)
Relationships tracked :: one at a time (blurs the rest) :: several at once, each sharp
Width each works in :: full d_model :: a slice, d_k = d_model / h
Cost of more :: n/a :: about the same — same budget, sliced thinner
Combining :: nothing to combine :: join the answers, then mix with W_O
%%%

You just built the attention that lives inside every real transformer. Come back to this page whenever multi-head feels foggy — the four beats are all of it. Next day we teach the model something it still has no idea about: **word order**.

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: Why do transformers use more than one attention head? | a:2 | to run faster on the GPU | to save memory | so each head can track a different relationship instead of blurring them into one blend | to remove the softmax step | fb: One head blurs everything into one blend; a panel lets each head stay sharp on its own relationship.
q: If d_model = 512 and there are 8 heads, what is d_k (each head's width)? | a:1 | 512 | 64 | 8 | 4096 | fb: d_k = d_model / h = 512 / 8 = 64. The width is split across heads.
q: Does going from 1 head to 8 heads roughly multiply the compute by 8? | a:0 | no — each head works in a thinner slice, so total cost stays about the same | yes, 8 heads cost 8× | yes, but only during training | no, it actually costs less than 1 head | fb: d_k = d_model/h shrinks each head, so heads × d_k stays at d_model — same budget, sliced.
q: How are the heads combined back into one output? | a:3 | average all the heads | keep only the best head | add them together | concatenate their answers, then mix with the matrix W_O | fb: Join the head answers side by side, then W_O mixes them into one final vector.
%%%

@@@ produce id=produce tag="Produce" title="Build a two-head panel by hand" gotit="Done"
Time to make it real with your own hands. In `experiment.py`, build a tiny two-head panel and *watch* the widths split and rejoin.

**Predict before you write any code:** with input width `d_model = 4` and `h = 2` heads, (a) what width does each head's answer have, (b) what width after you join the two answers, and (c) what width after `W_O`? Write your three guesses down first.

Then build it, one step at a time:
1. Reuse your Day-3 attention (`softmax(Q @ Kᵀ / √d_k) @ V`). **Predict** that it still works unchanged — a head is just yesterday's attention on a slice.
2. Make **two** heads: two different random sets of `W_Q, W_K, W_V`, each projecting the same input into width `d_k = 2`. Run attention in each. **Print** each head's output shape and **notice** it is `(seq, 2)` — half the model width.
3. `np.concatenate` the two head outputs along the last axis. **Observe** the width is back to `d_model = 4`.
4. Multiply by an output matrix `W_O`. **Watch** the final shape land back at `(seq, 4)` — one vector per word.

What you should see by the end: two heads, each working in half the width, joined and mixed back into the full model width — the exact move inside every real transformer. Did your three shape guesses match what printed? Write it in `experiment.py` and run it.

@@@ fin
