---
quest_id: wf6-d05-causal
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 5 — Causal Masking"
module_label: "Module 05a · Represent · Day 5"
title: "Causal Masking"
subtitle: "Why a Model Can't Peek at the Future"
brand_sub: "Foundations · M5a Day 5"
spine: "blindfold"
nav_prev_href: "../day-04-full-block/lesson.html"
nav_prev_label: "The Full Block"
nav_next_href: "../day-06-encoder-vs-decoder/lesson.html"
nav_next_label: "Encoder vs Decoder"
fin_title: "Module 5a · Day 5 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Causal Masking</b>: the <b>blindfold</b> that stops each position from attending to words that come later, so the model must predict the next word using only the past. Set those future scores to −∞ before softmax and their attention share becomes zero.<br>Next up: <b>Encoder vs Decoder</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: causal masking each position may attend only to itself and earlier positions never the future, keywords: [causal mask, only look back, past and present, never the future, may attend only, look backward, no peeking]}
  - {topic: attention scores matrix the pre-softmax grid where every position scores every other position, keywords: [attention scores, score grid, every position scores, pre-softmax grid, the object the mask]}
  - {topic: the causal mask is a lower-triangular pattern keep on and below diagonal block above diagonal, keywords: [lower-triangular, on and below the diagonal, block the future, triangle of allowed, above the diagonal]}
  - {topic: masking adds negative infinity to future score entries before softmax so their weight becomes zero, keywords: [negative infinity, add minus infinity, before softmax, future entries, set to minus, blocked scores]}
  - {topic: softmax turns a minus infinity score into a near zero attention weight exp of minus infinity is zero, keywords: [softmax turns, exp of minus infinity, becomes zero, weight zero, contributes nothing]}
  - {topic: autoregressive next-token prediction is the reason the mask exists predict token t from tokens 1 to t, keywords: [autoregressive, next-token prediction, predict token t, from the past only, one word at a time, must not see]}
  - {topic: parallel training with the mask all positions trained at once in one forward pass mask keeps each honest, keywords: [parallel training, all positions at once, one forward pass, trained together, mask keeps honest, fast training]}
  - {topic: failure information leakage cheating a position attends to the future token it must predict remedy causal mask, keywords: [information leakage, cheating, sees the answer, attend to the future token, bidirectional would see, remedy the mask, trivial]}
---

@@@ hero
@lede Imagine you're doing a "guess the next word" game with a friend. They read you a sentence one word at a time — "the cat sat on the …" — and cover the rest of the page with their hand so you can't peek ahead. If you could see the next word, the game would be pointless: you'd just read the answer off the page instead of *guessing* it. Today you meet the exact same trick, built right into a transformer, called a **blindfold** — the real name is **causal masking**. It's the one rule that lets a model *generate* text: cover up the future so every word must be *predicted* from only the words that came before it. This tiny rule is the reason ChatGPT can write you a sentence one word at a time — and by the end you'll see how the model wears this blindfold while somehow still training on a whole page at once, blazingly fast.
@goal Together we'll build the blindfold one step at a time — first *why* a model that peeks at the future learns nothing useful, then the grid of scores the blindfold covers, then the neat triangle shape of the blindfold itself, then the clever "−∞ before softmax" trick that makes covered words truly disappear, then the surprising payoff (the blindfold lets us train on a whole page at once without cheating), and finally a one-page recap with a cheat-sheet. Every rule is a small puzzle with a satisfying click, and every bit of math is said out loud in plain words first.

@@@ concept id=c1 tag="Why not peek?" title="The word-guessing game (and why peeking ruins it)" gotit="Got why peeking is cheating"
Let's start with the whole point, before any grids or math. A transformer that writes text is really just playing one game over and over: **guess the next word.** Show it "the cat sat on the", and its job is to guess "mat". Do that again and again, and it writes you a whole sentence. Everything today exists to make that game *honest*.

Think of a **spelling bee where the answer card is face-down on the table**. The teacher asks you to spell "rhythm". You think hard, sound it out, and take your best shot. But imagine the card were face-*up*, right in front of you — you'd just read the letters off it. You'd score 100% and learn *nothing*, because you never actually had to spell. A model learns to write the same way you learn to spell: by *guessing* the next word and getting corrected. If it can *see* the next word while guessing, it just copies it — perfect score, zero learning. **Where the picture breaks down:** a real spelling bee has one card at a time, but a model reads a whole sentence at once, so "the answer" it must not peek at is *every* next word, all along the sentence.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Left: a face-down answer card with a child thinking hard and guessing, labeled card hidden, you must actually guess, real learning. Right: a face-up answer card with the child just copying it, labeled card showing, just copy the answer, zero learning."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Guess the next word: hide the answer or the game is pointless</text><text x="130" y="40" fill="#276b45" font-weight="bold" font-size="10">answer HIDDEN</text><rect x="95" y="60" width="70" height="46" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="130" y="87" fill="#276b45" font-size="10">? ? ?</text><text x="130" y="126" fill="#276b45" font-size="9">🤔 must guess</text><text x="130" y="150" fill="#276b45" font-size="9">→ real learning ✓</text><text x="390" y="40" fill="#C93B3B" font-weight="bold" font-size="10">answer SHOWING</text><rect x="355" y="60" width="70" height="46" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="87" fill="#C93B3B" font-size="10">m a t</text><text x="390" y="126" fill="#C93B3B" font-size="9">😐 just copies</text><text x="390" y="150" fill="#C93B3B" font-size="9">→ zero learning ✗</text></g></svg>
%%%

**What the spelling-bee picture gets right:** if the answer is visible while you're supposed to be producing it, you learn nothing — you just copy. That is *exactly* the trap a text model must avoid. **Where it breaks down:** a bee tests one word; a model is guessing the next word at *every* spot in the sentence at the same time, so it needs to hide *all* the future words, not just one.

#### The word we'll use all day: autoregressive
When a model writes text by guessing the next word from the words before it — then feeds its own guess back in to guess the *next* one — we call it **[[autoregressive||Writing one token at a time, each new token predicted only from the tokens already there. "Auto" = self, "regressive" = looking back: the model looks back at what it (and you) already wrote to decide what comes next.]]**. It's a fancy word for a simple habit: *look back at what's written so far, guess the next word, repeat.* A token, by the way, is just a chunk of text — roughly a word — the unit the model reads and writes.

!!! c-warn ⚠️
<b>The failure this whole lesson prevents — "cheating" (information leakage):</b> if, while guessing the word at some spot, the model is allowed to look *ahead* at the very word it's supposed to guess, training becomes trivial — it just copies the answer. It scores perfectly during training and then falls apart the moment you ask it to generate, because at generation time the future genuinely doesn't exist yet. <b>Cause:</b> the model can see future words. <b>Remedy:</b> the blindfold we build today — a rule that hides every future word.
!!!

**Victory lap:** you already know the *why* of the entire lesson — a text model must guess the next word without seeing it, or it learns nothing. Everything from here is *how* we enforce that. Next: the grid of scores where the blindfold will live.

@@@ concept id=c2 tag="The score grid" title="Where the blindfold goes: the grid of scores" gotit="Got the score grid"
Before we can cover up the future, we need to know *what* gets covered. So let's look at the thing the blindfold sits on top of: a grid of scores. You met this earlier in the module — every word gives every other word a score for "how much should I pay attention to you?" Today we don't recompute it; we just need to picture it clearly, because that grid is the surface the blindfold will cover.

Think of a **classroom seating chart where everyone rates everyone else**. Imagine each student fills in a little card: "how much do I want to listen to *this* classmate?" Line all the cards up into a square grid — one row per student (the one doing the listening), one column per student (the one being listened to). Cell (row *Amy*, column *Ben*) holds Amy's score for Ben. A high number means "I really want to hear Ben"; a low number means "meh". A transformer builds the exact same grid over the words in a sentence: row = the word doing the looking, column = the word being looked at, and the number is how strongly they connect. **Where the picture breaks down:** students rate each other in messy human ways, but these scores come from a precise dot-product between the words' vectors — a topic you already met — and here we just treat them as given numbers.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="A 4 by 4 grid of attention scores for the words the cat sat mat. Rows are the word doing the looking, columns are the word being looked at. Each cell holds a score number. Labeled every word scores every other word, higher means look harder."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The attention-score grid: row looks at column, number = how strongly</text><text x="260" y="36" fill="#6B645E" font-size="9">columns = word being looked at →</text><text x="150" y="60" fill="#5E5191" font-size="10">the</text><text x="230" y="60" fill="#5E5191" font-size="10">cat</text><text x="310" y="60" fill="#5E5191" font-size="10">sat</text><text x="390" y="60" fill="#5E5191" font-size="10">mat</text><text x="70" y="95" fill="#9A7A10" font-size="10" text-anchor="end">the</text><text x="70" y="140" fill="#9A7A10" font-size="10" text-anchor="end">cat</text><text x="70" y="185" fill="#9A7A10" font-size="10" text-anchor="end">sat</text><text x="70" y="230" fill="#9A7A10" font-size="10" text-anchor="end">mat</text><text x="20" y="145" fill="#6B645E" font-size="9" transform="rotate(-90 20 145)">rows = word looking ↓</text><g stroke="#B8AEA2" fill="#F7F5F0"><rect x="110" y="72" width="80" height="40"/><rect x="190" y="72" width="80" height="40"/><rect x="270" y="72" width="80" height="40"/><rect x="350" y="72" width="80" height="40"/><rect x="110" y="117" width="80" height="40"/><rect x="190" y="117" width="80" height="40"/><rect x="270" y="117" width="80" height="40"/><rect x="350" y="117" width="80" height="40"/><rect x="110" y="162" width="80" height="40"/><rect x="190" y="162" width="80" height="40"/><rect x="270" y="162" width="80" height="40"/><rect x="350" y="162" width="80" height="40"/><rect x="110" y="207" width="80" height="40"/><rect x="190" y="207" width="80" height="40"/><rect x="270" y="207" width="80" height="40"/><rect x="350" y="207" width="80" height="40"/></g><g fill="#2C2A28"><text x="150" y="97">2.1</text><text x="230" y="97">0.3</text><text x="310" y="97">1.0</text><text x="390" y="97">0.4</text><text x="150" y="142">0.5</text><text x="230" y="142">3.2</text><text x="310" y="142">0.9</text><text x="390" y="142">0.2</text><text x="150" y="187">1.1</text><text x="230" y="187">2.4</text><text x="310" y="187">2.0</text><text x="390" y="187">0.6</text><text x="150" y="232">0.7</text><text x="230" y="232">1.3</text><text x="310" y="232">2.8</text><text x="390" y="232">1.9</text></g></g></svg>
%%%

**What the seating-chart picture gets right:** a square grid where every row-person rates every column-person captures the shape perfectly — one score for each looker-lookee pair. **Where it breaks down:** the numbers aren't opinions; they come from the word vectors' math, and every word scores *every* word — including words that come *later* in the sentence.

#### The one thing to notice
Look at the grid and notice something a little dangerous. The word "cat" (row 2) has a score for "mat" (column 4) — but "mat" comes *later* in the sentence. Right now, "cat" is allowed to look *forward* at "mat". That's the peeking we banned in the last concept. So the upper-right part of this grid — every cell where a row-word looks at a *later* column-word — is exactly where the model could cheat.

We call this whole object the **[[attention-score grid||The pre-softmax square of numbers where entry (i, j) is how strongly word i wants to attend to word j. Softmax later turns each row into weights that sum to 1; the mask acts on this grid BEFORE softmax.]]** — the raw scores, *before* they get turned into final attention weights. That "before" matters a lot, and it's where the blindfold clips on. Hold that thought.

!!! c-info 💡
<b>Why we act here, not later:</b> these raw scores are the last honest place to intervene. In one more step (softmax, which you met earlier) they get blended into the final answer. If we're going to hide the future, we have to do it *while it's still just a grid of scores* — before they melt together. That's why the whole game today is "do something to this grid before softmax."
!!!

**Victory lap:** you can now point at the exact spot where cheating could happen — the cells where a word looks at a *later* word — and you know we must fix it here, on the raw grid, before softmax. Next: the beautifully simple *shape* of the fix.

@@@ concept id=c3 tag="The triangle" title="The blindfold has a triangle shape" gotit="Got the triangle"
Now the fun part: the blindfold has a *shape*, and it's a shape you already know — a triangle. Once you see it, you'll never un-see it. This is the moment the whole idea clicks into a picture.

Think of **reading a book with a bookmark card you slide down the page**. As you read, you keep a card just below the line you're on. Everything *above* the card — the words you've already read — is visible. Everything *below* the card — the words you haven't reached yet — is covered. As you move down line by line, more of the page is uncovered *above*, and the still-hidden part *below* shrinks. A causal mask does this to the score grid: for each word (each row), it uncovers the current word and all the *earlier* words, and covers every *later* word. **Where the picture breaks down:** a bookmark hides whole lines you read one after another, but the mask works on all rows at the same instant — every word gets its own "how far can I see" line, all at once.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A page of text with a bookmark card sliding down. Lines above the card are visible, lines below are covered by the card. Labeled you can read what is above the card the past, the card hides what is below the future."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Sliding bookmark: read the past (above), the future (below) is covered</text><rect x="150" y="34" width="220" height="170" rx="4" fill="#FFFDF8" stroke="#B8AEA2"/><text x="260" y="56" fill="#276b45" font-size="10">the ✓ (read)</text><text x="260" y="76" fill="#276b45" font-size="10">cat ✓ (read)</text><text x="260" y="96" fill="#276b45" font-size="10">sat ← you are here</text><rect x="150" y="108" width="220" height="96" rx="4" fill="#8C8378" opacity="0.85"/><text x="260" y="134" fill="#FFF" font-size="10">▓ covered ▓</text><text x="260" y="156" fill="#FFF" font-size="10">▓ the future ▓</text><text x="260" y="178" fill="#FFF" font-size="9">(can't peek)</text><path d="M150 108 L370 108" stroke="#C93B3B" stroke-width="2"/><text x="392" y="112" fill="#C93B3B" font-size="9" text-anchor="start">card</text></g></svg>
%%%

**What the bookmark picture gets right:** you can always see the current line and everything before it, never the lines still to come — precisely the "past and present, never the future" rule. **Where it breaks down:** you slide a real bookmark line by line over time, but the mask applies its own cut-off to *every* word simultaneously in one shot.

#### Draw it on the grid: a lower triangle of "allowed"
Now put that rule onto the score grid from the last concept and a gorgeous pattern appears. Number the words 1, 2, 3, 4. Word *i* (row *i*) may look at word *j* (column *j*) **only if j ≤ i** — the column is the same as or earlier than the row. Mark every allowed cell green and every blocked cell grey, and the green cells form a **[[lower triangle||The set of grid cells on or below the main diagonal — where column ≤ row. These are the "allowed" cells: a word looking at itself or at earlier words. The blocked future cells form the upper triangle, above the diagonal.]]**: a solid staircase of green filling the bottom-left, with the blocked future sitting as grey in the top-right.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="The 4 by 4 grid with allowed cells green and blocked cells grey. The green allowed cells fill the lower triangle on and below the diagonal, the grey blocked future cells fill the upper triangle above the diagonal. Labeled allowed equals column no later than row, a lower triangle."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Keep the lower triangle (past+present), block the upper triangle (future)</text><text x="150" y="52" fill="#5E5191" font-size="10">w1</text><text x="230" y="52" fill="#5E5191" font-size="10">w2</text><text x="310" y="52" fill="#5E5191" font-size="10">w3</text><text x="390" y="52" fill="#5E5191" font-size="10">w4</text><text x="70" y="90" fill="#9A7A10" font-size="10" text-anchor="end">w1</text><text x="70" y="135" fill="#9A7A10" font-size="10" text-anchor="end">w2</text><text x="70" y="180" fill="#9A7A10" font-size="10" text-anchor="end">w3</text><text x="70" y="225" fill="#9A7A10" font-size="10" text-anchor="end">w4</text><g stroke="#B8AEA2"><rect x="110" y="67" width="80" height="40" fill="#DCEEE2"/><rect x="190" y="67" width="80" height="40" fill="#EDEAE4"/><rect x="270" y="67" width="80" height="40" fill="#EDEAE4"/><rect x="350" y="67" width="80" height="40" fill="#EDEAE4"/><rect x="110" y="112" width="80" height="40" fill="#DCEEE2"/><rect x="190" y="112" width="80" height="40" fill="#DCEEE2"/><rect x="270" y="112" width="80" height="40" fill="#EDEAE4"/><rect x="350" y="112" width="80" height="40" fill="#EDEAE4"/><rect x="110" y="157" width="80" height="40" fill="#DCEEE2"/><rect x="190" y="157" width="80" height="40" fill="#DCEEE2"/><rect x="270" y="157" width="80" height="40" fill="#DCEEE2"/><rect x="350" y="157" width="80" height="40" fill="#EDEAE4"/><rect x="110" y="202" width="80" height="40" fill="#DCEEE2"/><rect x="190" y="202" width="80" height="40" fill="#DCEEE2"/><rect x="270" y="202" width="80" height="40" fill="#DCEEE2"/><rect x="350" y="202" width="80" height="40" fill="#DCEEE2"/></g><g font-size="13"><text x="150" y="92" fill="#276b45">✓</text><text x="230" y="92" fill="#C93B3B">✗</text><text x="310" y="92" fill="#C93B3B">✗</text><text x="390" y="92" fill="#C93B3B">✗</text><text x="150" y="137" fill="#276b45">✓</text><text x="230" y="137" fill="#276b45">✓</text><text x="310" y="137" fill="#C93B3B">✗</text><text x="390" y="137" fill="#C93B3B">✗</text><text x="150" y="182" fill="#276b45">✓</text><text x="230" y="182" fill="#276b45">✓</text><text x="310" y="182" fill="#276b45">✓</text><text x="390" y="182" fill="#C93B3B">✗</text><text x="150" y="227" fill="#276b45">✓</text><text x="230" y="227" fill="#276b45">✓</text><text x="310" y="227" fill="#276b45">✓</text><text x="390" y="227" fill="#276b45">✓</text></g></g></svg>
%%%

Trace one row to feel it. Row **w1** (the first word) may look only at itself — one green cell, three greys, because nothing came before it. Row **w3** may look at w1, w2, and itself — three greens, one grey (w4, the future). Row **w4** (the last word) may look at everyone — a full green row. See the staircase? That green **lower triangle** *is* the blindfold's shape. Engineers call this pattern **lower-triangular**, and it's the whole geometry of causal masking.

**Victory lap:** you can now draw the mask from memory — a green lower-triangle of "allowed" (a word and everything before it) and a grey upper-triangle of "blocked" (the future). Next: the clever trick that turns "grey" into "actually gone".

@@@ concept id=c4 tag="Make it vanish" title="The −∞ trick: how a covered word truly disappears" gotit="Got the minus-infinity trick"
Here's the cleverest move of the day. We know *which* cells to block (the grey upper triangle). But how do you actually make a word "invisible" to another word? You don't delete it — you do something sneakier and simpler. You give its score a value so low that, after the next step, it gets *zero* attention. It's still in the grid, but it counts for nothing. Let's see the trick.

Think of **voting where you can score each choice, and the winner takes a slice of a pizza based on its score**. Everyone gets a slice proportional to their score — a high score wins a big slice, a low score wins a sliver. Now suppose you want one option to get *no pizza at all*. You don't remove it from the ballot; you just give it a score of **negative infinity** — the most impossibly-low score there is. When the slices are handed out, that option's slice shrinks to *nothing*. It's still on the ballot, but it gets zero pizza. The blindfold does exactly this: it adds negative infinity to every blocked (future) score, so after the sharing step those words get a zero slice of attention. **Where the picture breaks down:** you can't really write "negative infinity" on a ballot, and in code we use a huge negative number instead — but the effect ("this gets zero share") is identical, so "−∞" is the perfect way to picture it.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A pizza divided into slices by score. Three options with normal scores get real slices. One option marked minus infinity gets a zero-width sliver, no pizza. Labeled give a blocked word a score of minus infinity and it wins zero slice of attention."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Attention is a pizza shared by score — give the future −∞ → zero slice</text><circle cx="180" cy="120" r="70" fill="#FCF3DC" stroke="#C99A12"/><path d="M180 120 L180 50 A70 70 0 0 1 245 95 Z" fill="#F2D98A" stroke="#C99A12"/><path d="M180 120 L245 95 A70 70 0 0 1 235 160 Z" fill="#EFEAF7" stroke="#7C6DAA"/><path d="M180 120 L235 160 A70 70 0 0 1 130 185 Z" fill="#DCEEE2" stroke="#2D8B55"/><path d="M180 120 L130 185 A70 70 0 0 1 180 50 Z" fill="#FBE7E7" stroke="#C93B3B"/><text x="205" y="78" fill="#9A7A10" font-size="9">w1</text><text x="222" y="128" fill="#5E5191" font-size="9">w2</text><text x="175" y="170" fill="#276b45" font-size="9">w3</text><text x="140" y="95" fill="#C93B3B" font-size="9">now</text><text x="360" y="90" fill="#C93B3B" font-size="10">future word → score = −∞</text><text x="360" y="112" fill="#C93B3B" font-size="10">→ slice shrinks to 0</text><text x="360" y="150" fill="#276b45" font-size="10">still on the "ballot",</text><text x="360" y="168" fill="#276b45" font-size="10">just wins no pizza</text></g></svg>
%%%

**What the pizza-vote picture gets right:** the share each word wins is proportional to its score, so an impossibly-low score wins an impossibly-small (zero) share — without deleting the option. **Where it breaks down:** real pizza can't be sliced to exactly zero by "infinity", but softmax *can* — a −∞ score gives *mathematically* zero weight, which is why the trick is exact.

#### Why −∞ becomes exactly zero: meet the sharing step
The "hand out pizza by score" step has a name you met earlier: **[[softmax||The step that turns a row of raw scores into attention weights that add up to 1. It raises e to the power of each score, then divides by the total — so bigger scores win bigger shares, and the shares always sum to 1.]]**. Here's the one fact that makes the whole trick work, said in plain words: softmax first takes each score and computes *e to that power*. And *e to the power of negative infinity is zero.* A number raised to a hugely negative power races to zero — like `e⁻¹⁰⁰⁰` being so tiny it *is* zero for all purposes. So a blocked cell contributes a big fat **0** to the sharing, and the leftover shares (from the allowed words) simply add up to 1 among themselves. The future word is present in the grid but wins exactly none of the attention.

Now let's *watch* it happen with real numbers — predict first. Here's one row of scores, `[2.0, 1.0, 5.0]`, where the third word is the *future* one we must block. We add −∞ to that third score, then run softmax. What do you think the third weight becomes?

%%% demo id=maskrow label="predict, then run it"
code: scores = [2.0, 1.0, 5.0];  scores[2] += -inf;  softmax(scores)   # block the future word (index 2)
out: [0.73, 0.27, 0.00]   # ← the blocked future word gets weight 0.00; the past two shares still sum to 1
take: <b>−∞ before softmax → weight 0.00.</b> Notice the huge 5.0 score would have DOMINATED (it's the biggest!) — but once we add −∞ its share collapses to exactly zero, and the two allowed words split all the attention (0.73 + 0.27 = 1). The future word is still in the row; it just wins none of the pizza. That is the entire mechanism of causal masking.
%%%

#### The rule, in one plain line
Here's the whole operation stated once, in words: *for every future cell in the grid, add −∞ to its score; then run softmax as usual.* That's it. No cell is deleted, no special case — you just push the future scores down to −∞ and let softmax zero them out.

!!! c-info ⚙️
<b>Optional (skippable) — the softmax fact in symbols.</b> Softmax turns a score `sⱼ` into a weight `wⱼ = e^{sⱼ} / Σₖ e^{sₖ}`. Masking replaces every future score with `sⱼ + (−∞) = −∞`. Since `e^{−∞} = 0`, the numerator for a blocked word is `0`, so `wⱼ = 0`, and the denominator's remaining terms (the allowed words) still divide to sum to 1. In real code, `−∞` is a big negative constant like `−1e9` to avoid `NaN`, but the intuition is exactly `e^{−∞}=0 → weight 0`. Skip this if the pizza already did it for you.
!!!

**Victory lap:** you now know the *mechanism*, not just the picture — add −∞ to future scores *before* softmax, and because `e^{−∞}=0` those words get exactly zero attention. That's the beating heart of causal masking, and you just watched it work on real numbers. Next: the surprising payoff that makes this worth all the fuss.

@@@ concept id=c5 tag="The payoff" title="Train the whole page at once — without cheating" gotit="Got the parallel payoff"
Now for the payoff I promised in the opening — the part that feels like magic. You'd think that to train the model on "guess the next word", you'd have to feed it the sentence one word at a time: show "the", ask for "cat"; show "the cat", ask for "sat"; and so on, slowly. That would be painfully slow. The blindfold lets us skip all that: we feed the *whole sentence at once*, in a single pass, and still get an honest next-word guess at *every* position — with no cheating. Here's how.

Think of **a class taking the same quiz at the same time, each in their own exam booth**. Instead of testing one student, waiting, then testing the next, you put the whole class in booths and everyone answers *simultaneously*. The booths matter: they stop anyone from copying a neighbour. You grade the entire class in one sitting — fast — and every answer is still honest, because nobody could see anyone else's. The blindfold is the booth: it lets *every position* in the sentence make its next-word guess at the same instant, in one forward pass, while its own triangle-cutoff stops it from seeing the answers it's not allowed to see. **Where the picture breaks down:** exam booths block *both* neighbours, but the causal mask blocks only the *future* neighbours — each word is still allowed to "copy from" the words behind it, because those are its legitimate context.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A row of exam booths, one per word, all answering at the same time. Each booth can see the booths to its left the past but a wall blocks the booths to its right the future. Labeled all positions guess at once one pass, each blindfolded from its own future."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One pass: every word guesses its next word at once, blindfolded from the future</text><g><rect x="40" y="60" width="90" height="90" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="85" y="52" fill="#276b45" font-size="9">"the" → cat?</text><text x="85" y="110" fill="#276b45" font-size="18">🧑‍🎓</text><rect x="150" y="60" width="90" height="90" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="195" y="52" fill="#276b45" font-size="9">"cat" → sat?</text><text x="195" y="110" fill="#276b45" font-size="18">🧑‍🎓</text><rect x="260" y="60" width="90" height="90" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="305" y="52" fill="#276b45" font-size="9">"sat" → on?</text><text x="305" y="110" fill="#276b45" font-size="18">🧑‍🎓</text><rect x="370" y="60" width="90" height="90" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="415" y="52" fill="#276b45" font-size="9">"on" → mat?</text><text x="415" y="110" fill="#276b45" font-size="18">🧑‍🎓</text></g><path d="M140 105 L152 105" stroke="#2D8B55"/><path d="M250 105 L262 105" stroke="#2D8B55"/><path d="M360 105 L372 105" stroke="#2D8B55"/><text x="145" y="175" fill="#276b45" font-size="8">can look left ✓ (past)</text><text x="415" y="175" fill="#C93B3B" font-size="8">wall blocks right ✗ (future)</text><path d="M462 60 L462 150" stroke="#C93B3B" stroke-width="3" stroke-dasharray="4,3"/></g></svg>
%%%

**What the exam-booth picture gets right:** everyone answers at once (fast), and the booth keeps each answer honest by hiding what they're not allowed to see — exactly parallel training with a mask. **Where it breaks down:** a booth hides everyone else, but the mask hides only the *future*; each word is meant to use the past as its clue.

#### The two ideas that just clicked together
- **[[next-token prediction||The training task: at every position, predict the token that actually comes next in the real text. The correct answer is just "the real next word", which the text already gives you for free.]]** is the task — at every spot, guess the real next word. Because the true next word is sitting right there in the text, you get a free "correct answer" at *every* position, all at once.
- **[[parallel training||Feeding the whole sequence through the model in one forward pass and computing a next-word guess at every position simultaneously, instead of one word at a time. The causal mask is what keeps this honest.]]** is what makes it fast — feed the whole sentence in one shot and grade every position together.

The blindfold is the bridge between them: it's what lets you do next-token prediction *in parallel* without any position peeking at the answer it's supposed to produce. Say it in one breath: *we train on the whole sentence at once, and the triangular mask keeps every position's guess honest.* It is **not** slow one-word-at-a-time processing that makes the mask necessary — it's the *opposite*: the mask is precisely what lets training be fast **and** honest at the same time.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it (in an interview):</b> "Causal masking makes a transformer autoregressive: before softmax, every future entry in the attention-score grid is set to −∞, so each position attends only to itself and earlier positions. Because `e^{−∞}=0`, blocked positions get zero weight. This is what lets us train next-token prediction on a whole sequence in one parallel forward pass — the lower-triangular mask keeps each position's prediction from seeing the token it must predict, so there's no information leakage." You can already say every word of that.
!!!

**Victory lap:** you just connected the two big ideas — the blindfold turns slow, one-word-at-a-time training into one fast parallel pass, *without* letting any word cheat. That's a genuine staff-level insight, and it's yours. Next: a one-page recap to lock it all in.

@@@ concept id=c6 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the whole blindfold. Before the cheat-sheets, one picture to hold the entire day in your head.

Think of **that sliding bookmark on the page again**. At every word, you can read what's above the bookmark (the past) and never what's below it (the future) — and the shape that rule makes on the score grid is a neat lower triangle: green (allowed) filling the bottom-left, grey (blocked) in the top-right. To *enforce* it, you add −∞ to every grey cell before softmax, and because `e^{−∞}=0`, those future words win zero attention. That one rule is what lets a model guess the next word honestly — and lets us train on a whole page at once, fast. **Where this breaks down:** a bookmark hides lines one at a time as you read, but the mask cuts off *every* word's view simultaneously, in a single pass.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A summary in three steps: a lower-triangular grid with green allowed cells and grey blocked cells, an arrow to adding minus infinity to the grey cells, an arrow to softmax giving the grey cells weight zero. Labeled triangle then minus infinity then softmax equals blindfold."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The blindfold in three steps: triangle → add −∞ → softmax → future = 0</text><g stroke="#B8AEA2"><rect x="40" y="50" width="24" height="24" fill="#DCEEE2"/><rect x="64" y="50" width="24" height="24" fill="#EDEAE4"/><rect x="88" y="50" width="24" height="24" fill="#EDEAE4"/><rect x="40" y="74" width="24" height="24" fill="#DCEEE2"/><rect x="64" y="74" width="24" height="24" fill="#DCEEE2"/><rect x="88" y="74" width="24" height="24" fill="#EDEAE4"/><rect x="40" y="98" width="24" height="24" fill="#DCEEE2"/><rect x="64" y="98" width="24" height="24" fill="#DCEEE2"/><rect x="88" y="98" width="24" height="24" fill="#DCEEE2"/></g><text x="76" y="140" fill="#276b45" font-size="9">1 · lower triangle</text><text x="76" y="153" fill="#6B645E" font-size="8">(allowed vs blocked)</text><text x="150" y="90" fill="#B8AEA2" font-size="16">→</text><rect x="180" y="58" width="90" height="56" rx="5" fill="#FBE7E7" stroke="#C93B3B"/><text x="225" y="82" fill="#C93B3B" font-size="10">grey cells</text><text x="225" y="100" fill="#C93B3B" font-size="10">+= −∞</text><text x="225" y="140" fill="#C93B3B" font-size="9">2 · add −∞</text><text x="225" y="153" fill="#6B645E" font-size="8">(before softmax)</text><text x="300" y="90" fill="#B8AEA2" font-size="16">→</text><rect x="330" y="58" width="150" height="56" rx="5" fill="#DCEEE2" stroke="#2D8B55"/><text x="405" y="80" fill="#276b45" font-size="10">softmax: e^(−∞)=0</text><text x="405" y="100" fill="#276b45" font-size="10">future weight = 0.00</text><text x="405" y="140" fill="#276b45" font-size="9">3 · future vanishes</text><text x="405" y="153" fill="#6B645E" font-size="8">(past shares sum to 1)</text><text x="260" y="188" fill="#5E5191" font-size="10">= the blindfold: each word predicts the next from the past only</text></g></svg>
%%%

**The day as signposts, in order** — each only makes sense because of the one before it:
- **1 · Why not peek?** A text model plays "guess the next word" (**autoregressive**). If it can see the next word while guessing, it just copies — perfect score, zero learning. That cheating is **information leakage**, and the blindfold prevents it.
- **2 · The score grid.** Attention builds a square grid: row = word looking, column = word looked at, cell = the raw score. The risky cells are where a word looks *forward* at a *later* word. We fix it here, *before* softmax.
- **3 · The triangle.** Allowed = column no later than row → a green **lower triangle** (past + present); blocked = the grey upper triangle (the future). That shape is **lower-triangular**.
- **4 · Make it vanish.** Add **−∞** to every blocked score *before* softmax. Since `e^{−∞}=0`, those words get weight **0** — present in the grid, but winning none of the attention.
- **5 · The payoff.** The blindfold lets us do **next-token prediction** on the whole sentence in one fast **parallel** pass, while keeping every position honest — fast *and* no cheating.

#### Cheat-sheet · the blindfold at a glance
%%% table
:: Step :: In plain words :: Why it matters
the score grid :: every word scores every word (row looks at column) :: the surface the mask acts on, before softmax
lower-triangular mask :: keep cells where column ≤ row (past+present), block the rest :: the shape that lets a word see backward but never forward
add −∞ before softmax :: push every future score to negative infinity :: makes blocked words win zero attention after softmax
e^(−∞) = 0 :: softmax turns a −∞ score into weight 0 :: why a blocked word contributes nothing, without being deleted
parallel + honest :: train the whole sentence in one pass, mask keeps each guess honest :: fast training with no information leakage
%%%

#### Cheat-sheet · the words you met today
%%% jargon
causal masking | the rule that each word may attend only to itself and earlier words, never the future
autoregressive | writing one word at a time, each predicted only from the words already there
attention-score grid | the pre-softmax square of raw scores: entry (i,j) = how much word i attends to word j
lower-triangular | the pattern of allowed cells (column ≤ row): a filled triangle in the bottom-left
softmax | turns a row of scores into attention weights that sum to 1 (via e-to-the-power then divide)
−∞ before softmax | the trick: adding negative infinity to future scores so e^(−∞)=0 zeroes their weight
next-token prediction | the task: guess the real next word at every position; the text gives the answer for free
parallel training | run the whole sequence in one pass; the mask keeps every position from peeking ahead
information leakage | "cheating": a word seeing the future token it must predict — the failure the mask prevents
%%%

That's the whole day. Next you'll compare two ways to wire attention: the **decoder** you just built (blindfolded, look-back-only, for *generating* text) versus the **encoder** (no blindfold, free to look both ways, for *understanding* a whole input) — that's **Encoder vs Decoder**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: Why must a text model NOT see the next word while guessing it? | a:2 | It would run out of memory | It would be too slow | It would just copy the answer — perfect score during training, but useless at generation time where the future doesn't exist yet | It would forget the earlier words | fb: Seeing the answer is cheating (information leakage): the model copies instead of learning, then fails when it has to actually generate.
q: What shape does the causal mask make on the attention-score grid? | a:1 | A full square, all cells allowed | A lower triangle — a word may attend to itself and earlier words (column ≤ row), and the future upper triangle is blocked | A single diagonal line only | A random scatter of allowed cells | fb: Allowed = column no later than row → a green lower triangle (past+present); the blocked future is the grey upper triangle.
q: How does the mask actually make a future word get zero attention? | a:3 | It deletes the word from the sentence | It sets the word's score to zero | It runs softmax twice | It adds −∞ to the future score BEFORE softmax, and since e^(−∞)=0 that word gets weight 0 | fb: Add −∞ before softmax; because e^(−∞)=0, the blocked word wins none of the attention while staying in the grid.
q: What does the causal mask let us do during training? | a:1 | Train one word at a time, very slowly | Train the whole sentence in one parallel forward pass while keeping every position's next-word guess honest (no peeking ahead) | Skip training entirely | Remove the need for softmax | fb: The mask is what makes training fast AND honest: all positions predict at once, and the triangle stops each from seeing the answer it must produce.
%%%

@@@ produce id=produce tag="Produce" title="Build the blindfold and watch the future vanish" gotit="Done"
Time to see today's big idea with your own eyes. You'll take a tiny row of attention scores, build the causal mask, apply the −∞ trick, run softmax, and **watch** the future words' weights collapse to exactly zero — then confirm the allowed (past) words still split all the attention. **Predict first:** for the third word in a 4-word sentence, which columns should end up with weight 0 after masking? (Hint: only the *future* one.) Then run it and **observe**. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-05-causal-masking/experiment.py`. **Notice** three things: (1) make a 4×4 grid of made-up attention scores, and build a `causal_mask` that is `0` on and below the diagonal and `-inf` above it — print it and **watch** the lower-triangle-of-zeros / upper-triangle-of-minus-infinity shape appear; (2) add the mask to the scores and run softmax on each row, then print the weight grid — **observe** that every above-diagonal (future) cell is `0.00` while each row's allowed weights sum to `1.0`; (3) as a control, run softmax *without* the mask and **notice** that now the future cells have real, non-zero weight — that's the "cheating" the mask prevents. Run with `python3 sessions/m05a-text-transformer/day-05-causal-masking/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5a Day 5 artifact.

Create sessions/m05a-text-transformer/day-05-causal-masking/experiment.py that, with a comment on each step:
1. Makes a 4x4 numpy array of arbitrary attention scores for a 4-word sentence.
2. Builds a causal_mask that is 0.0 on and below the main diagonal and -np.inf above it (hint: np.triu with k=1 to pick the future cells), and prints it so the lower-triangular shape is visible.
3. Defines a row-wise softmax, adds the mask to the scores, applies softmax per row, and prints the resulting weight grid — showing every future (above-diagonal) cell is 0.00 and each row sums to 1.0.
4. As a control, applies softmax to the UNmasked scores and prints that grid too, so you can see the future cells now carry non-zero weight — the information leakage the causal mask prevents.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the future collapses to zero)
- The `causal_mask` prints as a lower triangle of `0` (allowed) with `-inf` filling the cells above the diagonal (the future).
- After adding the mask and running softmax, every above-diagonal weight is `0.00`, and each row's remaining (allowed) weights add up to `1.0` — the future contributes nothing.
- The un-masked control shows those same future cells carrying real weight — proof of exactly what the blindfold is preventing.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-05-causal-masking/log.md`: (1) in one sentence, why a text model must not see the next word while guessing it; (2) the three steps of the blindfold (lower triangle → add −∞ → softmax gives future weight 0); (3) one thing the mask lets us do during training (train the whole sentence at once, fast, without cheating).
!!!

@@@ fin
