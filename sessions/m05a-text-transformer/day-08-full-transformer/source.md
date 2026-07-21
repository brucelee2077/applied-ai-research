---
quest_id: wf6-d08-assemble
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 8 — The Full Transformer"
module_label: "Module 05a · Represent · Day 8"
title: "The Full Transformer"
subtitle: "Every Piece, Wired Together"
brand_sub: "Foundations · M5a Day 8"
spine: "stack"
nav_prev_href: "../day-07-text-generation/lesson.html"
nav_prev_label: "Text Generation &amp; Sampling"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
fin_title: "Module 5a · Day 8 complete! 🏆"
fin_body: "You did it — you've assembled <b>The Full Transformer</b> end to end: tokens become embeddings, get their positions stamped in, flow up through a <b>stack</b> of identical blocks (attention + feed-forward, each with its own norm and shortcut), and a final projection turns the top vector into next-word scores. Every piece you built across this module now stands together in one tower.<br>Next up: the <b>Module 5a Review Gate</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: the full transformer as an encoder-decoder stack the whole picture from parts already learned, keywords: [full transformer, encoder-decoder, whole picture, all the parts, the whole machine, end to end, assemble the whole]}
  - {topic: the transformer block is the repeating unit attention sublayer then feed-forward sublayer each wrapped in residual and layer norm, keywords: [transformer block, repeating unit, attention sub-layer, feed-forward sub-layer, residual, layer norm, one block]}
  - {topic: stacking N identical blocks each layer refines the representation deeper stacks build more abstract features, keywords: [stack, n identical blocks, repeat the block, each layer refines, deeper, more abstract, six layers, tower of blocks]}
  - {topic: encoder stack reads the whole input at once bidirectional every position attends to every other producing context-rich source representations, keywords: [encoder stack, reads the whole input, bidirectional, every position attends, no mask, context-rich, understand the source]}
  - {topic: decoder stack generates one token at a time three sub-layers masked self-attention cross-attention and feed-forward, keywords: [decoder stack, one token at a time, three sub-layers, masked self-attention, cross-attention, feed-forward, generate the output]}
  - {topic: cross-attention encoder-decoder attention decoder queries attend over encoder output keys and values the bridge that lets the output look back at the input, keywords: [cross-attention, encoder-decoder attention, decoder queries, encoder keys and values, the bridge, look back at the input, read the source]}
  - {topic: masked causal self-attention in the decoder so it can only attend to earlier positions remedy causal mask hides future tokens, keywords: [masked self-attention, causal, only look back, earlier positions, remedy causal mask, hides future tokens, train matches generation]}
  - {topic: residual skip connection around each sub-layer adds the input back remedy for vanishing gradients lost signal keeps a clean gradient highway, keywords: [residual, skip connection, add the input back, remedy vanishing gradient, lost signal, clean gradient highway, express lane]}
  - {topic: layer normalization at each sub-layer normalizing each token feature vector so the stack trains stably remedy for shifting scaling activations, keywords: [layer normalization, layer norm, normalize each token, stable scale, remedy scale drift, shifting activations, steady]}
  - {topic: position-wise feed-forward network small two-layer MLP applied identically to each position per-token nonlinear processing, keywords: [position-wise feed-forward, two-layer, applied to each position, per-token, nonlinear, widen bend shrink, each word thinks alone]}
  - {topic: autoregressive generation at inference the decoder feeds its own previous outputs back to produce the next token until an end symbol, keywords: [autoregressive, feeds its own outputs back, next token, one step at a time, until an end symbol, generation loop, eos]}
  - {topic: the output head final linear projection over the vocabulary followed by softmax to pick the next token, keywords: [output head, final linear projection, over the vocabulary, softmax, next-token scores, logits, project to vocabulary]}
  - {topic: how the parts connect end to end tokens embeddings plus positional info encoder stack decoder stack with cross-attention output head next token, keywords: [end to end, tokens to embeddings, positional info, encoder stack, decoder stack, output head, the assembly, full data flow]}
  - {topic: positional information injected before the stack deferred to an earlier day, keywords: [positional, position information, order stamped in, injected before, added at the bottom]}
  - {topic: failure vanishing or lost signal in a deep stack remedy residual connections, keywords: [vanishing, lost signal, deep stack, signals shrink, remedy residual, skip past]}
  - {topic: failure activation scale drift across layers destabilizing training remedy layer normalization, keywords: [scale drift, activations shift, destabilize training, remedy layer norm, re-level]}
  - {topic: failure decoder peeking at future tokens train inference mismatch remedy causal mask, keywords: [peek at the future, cheating, train inference mismatch, remedy the causal mask, blindfold the future]}
  - {topic: encoder-decoder original translation transformer versus decoder-only GPT ChatGPT which drop the encoder and cross-attention, keywords: [decoder-only, original transformer, translation, gpt, chatgpt drops the encoder, no cross-attention, only the writer]}
---

@@@ hero
@lede You know how a tall LEGO tower is really just one little brick pressed onto another, onto another, all the way up — each brick doing the same small job, but together they make something huge? Over the last seven days you molded every brick you'll ever need: attention (words share), the feed-forward step (each word thinks alone), the two safety rails (a shortcut and a re-leveller), the blindfold that hides the future, and the loop that writes one word at a time. Today is the big reveal — we snap them all together into the whole machine: the **full transformer**. Think it'll be a tangle of new parts? Here's the lovely secret: there are *no* new parts today. Every piece is one you already built. All that's left is to see how they *stack* — how tokens go in the bottom and, floor by floor, come out the top as a next word. Build this tower once and you'll recognize its shape inside Google Translate, ChatGPT, and every large language model you've heard of.
@goal Together we'll assemble the whole tower one snap at a time — first the tokens-to-numbers on-ramp at the very bottom, then the block you already know as the repeating brick, then the two towers made of it (an encoder that *reads* and a decoder that *writes*), then the little bridge (cross-attention) that lets the writer peek at what the reader understood, then the output head that turns the top vector back into a word, then the generation loop that runs the whole thing one token at a time — and finally a one-page map of the entire machine. No new parts. Every failure we meet is a puzzle with a neat fix, and every formula is said out loud in plain words first.

@@@ concept id=c1 tag="The on-ramp" title="Getting words into the tower" gotit="Got the on-ramp"
Before any of the clever machinery can run, we have a small but essential job: the transformer only understands *numbers*, but you feed it *words*. So the very first thing at the bottom of the tower is an on-ramp that turns text into numbers the tower can work with. Let's build that on-ramp before we build the tower on top of it.

Think of **checking coats at a fancy theater**. You hand over your coat, and the attendant gives you a numbered ticket — the coat becomes a number. But a plain number isn't enough: the cloakroom also writes down *which peg* your coat hangs on, so it knows the *order* things came in. A transformer does both of these at the on-ramp: it turns each word into a list of numbers, then stamps on *where* that word sits in the sentence. **Where the picture breaks down:** a coat ticket is just an ID with no meaning, but a word's list of numbers actually *captures meaning* — words that mean similar things get similar numbers.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="The on-ramp. On the left the word cat is handed over like a coat. The attendant gives back a small list of numbers labeled embedding, a list of numbers that captures meaning. Then a peg number labeled position which one is it stamped on top. Arrow leads to the combined result ready to enter the tower.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">On-ramp: turn each word into numbers, then stamp WHERE it sits</text>
<rect x="30" y="80" width="70" height="34" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="65" y="101" fill="#5E5191" font-size="11">"cat"</text>
<text x="65" y="130" fill="#6B645E" font-size="8">a word</text>
<path d="M100 97 L134 97" stroke="#B8AEA2" stroke-width="2" marker-end="url(#on)"/>
<rect x="136" y="80" width="110" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="191" y="96" fill="#276b45" font-size="9">[0.2, -0.7, 0.9]</text>
<text x="191" y="109" fill="#276b45" font-size="8">embedding (meaning)</text>
<path d="M246 97 L280 97" stroke="#B8AEA2" stroke-width="2" marker-end="url(#on)"/>
<circle cx="315" cy="97" r="18" fill="#FCF3DC" stroke="#C99A12"/>
<text x="315" y="94" fill="#9A7A10" font-size="14">+</text>
<text x="315" y="106" fill="#9A7A10" font-size="8">pos 3</text>
<text x="315" y="132" fill="#6B645E" font-size="8">which spot?</text>
<path d="M336 97 L372 97" stroke="#B8AEA2" stroke-width="2" marker-end="url(#on)"/>
<rect x="374" y="80" width="120" height="34" rx="6" fill="#FBFAF7" stroke="#7C6DAA"/>
<text x="434" y="96" fill="#5E5191" font-size="9">meaning + place</text>
<text x="434" y="109" fill="#5E5191" font-size="8">ready for the tower ↑</text>
<text x="260" y="176" fill="#6B645E" font-size="9">every word goes through this on-ramp before the stack begins</text>
<defs><marker id="on" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#B8AEA2"/></marker></defs>
</g></svg>
%%%

**What the coat-check picture gets right:** a word becomes a compact code, and the system also records *the order* it arrived — meaning plus position, exactly what the on-ramp produces. **Where it breaks down:** a coat ticket is a meaningless ID; a word's number-list is packed with meaning, so similar words sit close together.

#### The two on-ramp steps, named
1. First, each word (really each **[[token||A chunk of text, roughly a word — the unit the model reads and writes. "cat" is one token; long words may split into a few.]]**) is turned into an **[[embedding||A list of numbers that stands for a word's meaning. Words with similar meanings get similar lists, so the model can do math on meaning.]]** — a list of numbers that captures its meaning. You met this early in the module.
2. Then, because a bare stack has *no idea* what order the words came in, we add **[[positional information||Extra numbers added to each embedding that say WHERE the word sits in the sentence — so "dog bites man" and "man bites dog" look different. Built on an earlier day of this module.]]** — a stamp saying *where* each word sits. You built this on the positional-encoding day; here we just remember it clips on at the very bottom.

That's the whole on-ramp: **word → embedding → add position.** Now the numbers are ready to climb the tower.

!!! c-info 💡
<b>Why position has to be added here, at the bottom.</b> Remember from the block day: a bare transformer block is *permutation-equivariant* — shuffle the words and it just shuffles the outputs, blind to order. Left alone it would read "dog bites man" and "man bites dog" as the *same bag of words*. Stamping position onto each embedding *before* the stack is how we fix that once, at the on-ramp, so every floor above already knows the order.

**Victory lap:** you've got the on-ramp — words in, meaning-plus-position numbers out. Every piece from here works on those numbers. Next: the single brick the whole tower is made of.

@@@ concept id=c2 tag="The one brick" title="The block — the brick the whole tower is made of" gotit="Got the brick"
Now the star of the show, and the whole reason today is easy: the entire tower is built from *one* repeating brick, the **transformer block** you assembled on the block day. Let's remind ourselves what's inside that brick, because once you can picture *one* block, you can picture the *whole* transformer — it's just this brick, stacked.

Think of a single **LEGO brick with a fixed shape**. Every copy of that brick is identical: same bumps on top, same holes underneath, so any brick snaps cleanly onto any other. That's the magic that lets you build a tower — one reliable, repeatable shape. The transformer block is exactly that: one fixed recipe you can copy and stack as many times as you like. **Where the picture breaks down:** every LEGO brick is truly identical, but each transformer block *learns its own* weights — same shape, but each one ends up doing a slightly different job.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="One transformer block drawn as a single brick. Words enter at the bottom, pass through the attention sub-layer labeled share, each wrapped by a shortcut and a norm, then through the feed-forward sub-layer labeled think, also wrapped by a shortcut and a norm, and refined words leave the top. A dashed outline marks it as one block, the same width in and out.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One block = attention (share) + feed-forward (think), each with rails</text>
<rect x="185" y="168" width="150" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="260" y="184" fill="#276b45">words in</text>
<line x1="130" y1="180" x2="130" y2="60" stroke="#2D8B55" stroke-width="4"/>
<text x="122" y="120" fill="#276b45" font-size="8" text-anchor="end">shortcut</text>
<text x="122" y="132" fill="#276b45" font-size="8" text-anchor="end">(residual)</text>
<rect x="200" y="128" width="120" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="260" y="145" fill="#5E5191" font-size="9">attention · SHARE (+ norm)</text>
<path d="M260 168 L260 154" stroke="#B8AEA2"/>
<circle cx="130" cy="118" r="10" fill="#FFF" stroke="#2C2A28"/><text x="130" y="122" fill="#2C2A28" font-size="11">+</text>
<path d="M260 128 L260 118 L141 118" stroke="#7C6DAA" stroke-width="1.5"/>
<rect x="200" y="80" width="120" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/>
<text x="260" y="97" fill="#9A7A10" font-size="9">feed-forward · THINK (+ norm)</text>
<path d="M130 108 L200 94" stroke="#B8AEA2"/>
<circle cx="130" cy="70" r="10" fill="#FFF" stroke="#2C2A28"/><text x="130" y="74" fill="#2C2A28" font-size="11">+</text>
<path d="M260 80 L260 70 L141 70" stroke="#7C6DAA" stroke-width="1.5"/>
<line x1="130" y1="60" x2="130" y2="44" stroke="#2D8B55" stroke-width="4"/>
<rect x="70" y="40" width="120" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="130" y="55" fill="#276b45">refined words out</text>
<rect x="345" y="70" width="12" height="122" rx="3" fill="none" stroke="#B8AEA2" stroke-dasharray="4,3"/>
<text x="385" y="126" fill="#6B645E" font-size="9">= ONE block</text>
<text x="385" y="140" fill="#6B645E" font-size="8">width in = width out</text>
</g></svg>
%%%

**What the LEGO-brick picture gets right:** one fixed, repeatable shape that snaps onto copies of itself — that's what makes a tall tower possible from a single design. **Where it breaks down:** LEGO bricks are identical inside; transformer blocks share only their shape, and each learns its own weights.

#### The brick, in one breath
A block runs two working parts, in order, and wraps each in two rails:
- The **[[attention||The sub-layer where every word looks at every other word and pulls in what matters — it MIXES information across words. Built earlier in this module.]]** sub-layer — the words *share* (mix information across positions).
- The **[[feed-forward||The position-wise feed-forward network: a small two-layer MLP that widens each word's numbers, bends them, and shrinks them back — run on each word on its own. It TRANSFORMS each word without mixing across words.]]** sub-layer — each word *thinks* on its own (a small two-layer network run per word).
- Around *each* part: a **[[residual||A shortcut that adds a sub-layer's input straight back onto its output — output = x + SubLayer(x) — so the signal survives and gradients get a clean highway through deep stacks.]]** shortcut (so the signal survives) and a **[[layer norm||Re-levels each word's numbers to a steady scale, so they don't drift too big or too small as blocks pile up.]]** re-leveller (so the numbers stay a steady size).

Say it out loud: *share, then think — and wrap each with a shortcut and a re-leveller.* That's one block. That's the brick.

!!! c-info ⚙️
<b>Optional (skippable) — one wrapped sub-layer in symbols.</b> If `x` is a word's vector and `SubLayer` is attention or the feed-forward step, one wrapped sub-layer (modern Pre-LN style) is `x + SubLayer(LayerNorm(x))` — re-level, run the part, then add the original back on the shortcut. A block is just this done twice: once with attention as the part, then once with the feed-forward step. You don't need the symbols — the brick picture says it all — but they're here if you like seeing the shapes line up.
!!!

Here's that one line drawn as a little flow, so you can *see* it instead of reading symbols — the input `x` flows straight up the shortcut *and* takes a detour through re-level → the part, then the two meet at a `+`:

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="One wrapped sub-layer drawn as a flow. The input x splits: a straight green shortcut lane carries x up untouched, while a side branch goes through a layer norm box then the sub-layer box. The two rejoin at a plus circle, whose output is x plus SubLayer of LayerNorm of x. Labeled one wrapped sub-layer, Pre-LN style.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One wrapped sub-layer: x + SubLayer(LayerNorm(x))</text>
<rect x="20" y="66" width="52" height="26" rx="4" fill="#FBFAF7" stroke="#7C6DAA"/><text x="46" y="83" fill="#5E5191">x in</text>
<circle cx="110" cy="79" r="7" fill="#B8AEA2"/>
<path d="M72 79 L103 79" stroke="#B8AEA2" stroke-width="2"/>
<path d="M110 72 C130 20, 400 20, 430 72" stroke="#2D8B55" stroke-width="3" fill="none"/>
<text x="270" y="34" fill="#276b45" font-size="9">shortcut — x flows up untouched</text>
<path d="M110 86 L110 116 L160 116" stroke="#B8AEA2" stroke-width="2" marker-end="url(#w)"/>
<rect x="162" y="103" width="90" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="207" y="120" fill="#9A7A10" font-size="9">layer norm</text>
<path d="M252 116 L280 116" stroke="#B8AEA2" stroke-width="2" marker-end="url(#w)"/>
<rect x="282" y="103" width="90" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="327" y="120" fill="#5E5191" font-size="9">the part (SubLayer)</text>
<path d="M372 116 L430 116 L430 89" stroke="#B8AEA2" stroke-width="2" marker-end="url(#w)"/>
<circle cx="430" cy="79" r="10" fill="#FFF" stroke="#2C2A28"/><text x="430" y="83" fill="#2C2A28" font-size="11">+</text>
<path d="M440 79 L472 79" stroke="#B8AEA2" stroke-width="2" marker-end="url(#w)"/>
<rect x="452" y="66" width="56" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="480" y="83" fill="#276b45" font-size="9">out</text>
<defs><marker id="w" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#B8AEA2"/></marker></defs>
</g></svg>
%%%

**Victory lap:** you can picture the one brick the whole tower is made of — share, think, each wrapped in rails. Now watch what happens when you stack it.

@@@ concept id=c3 tag="Stack the brick" title="Stack the block into a tower — and why the rails keep it standing" gotit="Got the stack"
Here's the quiet magic the whole module has pointed at. You have *one* block. To build a real transformer, you do almost nothing clever: you **stack the same block, over and over.** That tower of stacked blocks is the body of every transformer. And this concept also answers a question you might have been holding: *why* does each block carry those two rails? Because without them, a tall stack falls over — and now we get to see exactly how.

Think of **floors in a tall building**. Each floor follows the same blueprint, so once you've designed one, you just repeat it as high as you want — and each floor *refines* what the one below did (arrivals sorted downstairs, processed higher up, finished at the top). A transformer stacks identical blocks the same way: each block takes the words the block below refined, and refines them a little more — early blocks catch simple patterns, deep blocks catch meaning. **Where the picture breaks down:** building floors are truly identical, but each block *learns its own* weights, so a deep block does more abstract work than a shallow one — same blueprint, different learned job.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="A tower of identical stacked blocks. Words in at the bottom, then block 1, block 2, block 3, dots, block N, then refined words out at the top. A side note reads same shape stacked, each block refines the words a little more, early blocks simple deep blocks abstract.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Stack the SAME block N times → the body of a transformer</text>
<rect x="170" y="198" width="180" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="260" y="213" fill="#276b45">words in (+ position)</text>
<rect x="185" y="164" width="150" height="27" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="260" y="182" fill="#5E5191">block 1 · simple patterns</text>
<rect x="185" y="130" width="150" height="27" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="260" y="148" fill="#5E5191">block 2</text>
<rect x="185" y="96" width="150" height="27" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="260" y="114" fill="#5E5191">block 3</text>
<text x="260" y="82" fill="#9A938A" font-size="12">⋮</text>
<rect x="185" y="52" width="150" height="27" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="260" y="70" fill="#5E5191">block N · deep meaning</text>
<rect x="170" y="22" width="180" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="37" fill="#276b45">refined words out</text>
<path d="M260 198 L260 191" stroke="#B8AEA2"/><path d="M260 164 L260 157" stroke="#B8AEA2"/><path d="M260 130 L260 123" stroke="#B8AEA2"/><path d="M260 52 L260 44" stroke="#B8AEA2"/>
<text x="400" y="118" fill="#6B645E" font-size="9">same shape,</text>
<text x="400" y="132" fill="#6B645E" font-size="9">stacked;</text>
<text x="400" y="146" fill="#6B645E" font-size="9">each refines</text>
<text x="400" y="160" fill="#6B645E" font-size="9">a bit more</text>
</g></svg>
%%%

**What the tall-building picture gets right:** design one floor to a repeatable blueprint and you can stack it to any height, each floor refining the last — exactly how a transformer reuses one block. **Where it breaks down:** real floors are identical, but each block learns different weights, specializing up the tower.

#### Puzzle: a raw tall stack keeps falling over — two failures, two rails
Stacking sounds free, but if you stacked *raw* blocks (no rails), a deep tower would refuse to train. Here are the two failures — and the exact rail that fixes each.

**Failure 1 · the signal fades (or blows up).** As a learning signal travels back down a deep raw stack, each layer multiplies it by some factor. Multiply by less than 1 over and over — `0.5 × 0.5 × 0.5 …` — and it races toward zero (the **[[vanishing gradient||When the learning signal is multiplied down toward zero as it passes back through many layers, so the early layers barely learn.]]**); bigger than 1 and it explodes. Either way the deep layers stop learning. **Remedy — the residual shortcut:** the signal flows straight up the express lane, so the learning signal gets a clean path straight back down; it *skips past* each block instead of being multiplied down by it. Watch the fade, then watch the shortcut stop it:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two bar ladders. Left without a shortcut: bars start tall and shrink to almost nothing across layers, labeled multiplied by 0.5 each layer, signal vanishes. Right with a shortcut: bars stay the same tall height across all layers, labeled the express lane keeps it full-strength.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Signal down a deep stack: it vanishes · the shortcut keeps it</text>
<text x="130" y="38" fill="#C93B3B" font-weight="bold">raw stack (×0.5 each layer)</text>
<line x1="55" y1="170" x2="210" y2="170" stroke="#B8AEA2"/>
<rect x="60" y="70" width="20" height="100" fill="#C93B3B" opacity="0.8"/>
<rect x="92" y="120" width="20" height="50" fill="#C93B3B" opacity="0.7"/>
<rect x="124" y="145" width="20" height="25" fill="#C93B3B" opacity="0.6"/>
<rect x="156" y="158" width="20" height="12" fill="#C93B3B" opacity="0.5"/>
<rect x="188" y="164" width="20" height="6" fill="#C93B3B" opacity="0.4"/>
<text x="130" y="188" fill="#C93B3B" font-size="9">1.0 → 0.5 → 0.25 → … → ~0 ✗</text>
<text x="390" y="38" fill="#276b45" font-weight="bold">with the residual shortcut</text>
<line x1="315" y1="170" x2="470" y2="170" stroke="#B8AEA2"/>
<rect x="320" y="70" width="20" height="100" fill="#2D8B55" opacity="0.85"/>
<rect x="352" y="70" width="20" height="100" fill="#2D8B55" opacity="0.85"/>
<rect x="384" y="70" width="20" height="100" fill="#2D8B55" opacity="0.85"/>
<rect x="416" y="70" width="20" height="100" fill="#2D8B55" opacity="0.85"/>
<rect x="448" y="70" width="20" height="100" fill="#2D8B55" opacity="0.85"/>
<text x="390" y="188" fill="#276b45" font-size="9">stays full-strength every layer ✓</text>
</g></svg>
%%%

**Failure 2 · the scale drifts.** Even if the signal survives, each block *nudges* the numbers — average up here, spread wider there. Stack fifty blocks and a late block sees numbers a wildly different size from what the early ones saw: **feature-scale drift**, and it makes training wobble. **Remedy — layer norm:** wrap each part with a layer norm and it re-levels the numbers to a steady scale every step, so every block always receives numbers at a sensible size. No drift.

!!! c-warn ⚠️
<b>Failure mode (silent):</b> a raw deep stack doesn't crash — it just trains slower and slower, then stops improving, because its early layers get almost no signal (vanishing) and its numbers wander off-scale (drift). The two rails — <b>residual</b> to stop the fade, <b>layer norm</b> to stop the drift — are exactly what made very deep networks trainable. That's why you'll never see a serious block *without* both.
!!!

Real models pick a depth and stick with it: the original transformer stacked **6** blocks; big models like GPT use dozens. But the *design* never changes — same brick, stacked, each refining the words a bit more.

**Victory lap:** you can explain the whole body of a transformer in one sentence — *one block, stacked N times, each refining a little more* — and name the two failures the rails quietly fix. Next: what happens when you point a stack at *reading* versus *writing*.

@@@ concept id=c4 tag="Two towers" title="Two stacks: one reads, one writes" gotit="Got the two towers"
Now we point our stack of blocks at a job. It turns out the *original* transformer — the one built for translation back in 2017 — isn't *one* tower, it's *two*, side by side. One tower's job is to **read and understand** the input; the other's job is to **write** the output one word at a time. Same brick in both, but wired for two different jobs — the exact reader-vs-writer split you met on the encoder-vs-decoder day.

Think of the two things you do with a story. When you **read** a page, your eyes are free — you glance ahead, back, all over, soaking in the whole thing to *understand* it. But when you **write** a story, you can only put down one word at a time, in order, and you *can't* write a word that depends on a word you haven't written yet. Reading looks both ways; writing marches forward. The two towers are exactly this: the **encoder** reads both ways, the **decoder** writes forward only. **Where the picture breaks down:** you switch between reading and writing freely, but each tower is *committed* — the encoder always reads both ways, the decoder always marches forward.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="Two towers side by side. Left the encoder stack of blocks with a caption reads the whole input both ways no mask, job understand. Right the decoder stack of blocks with a caption writes one word at a time look back only causal mask, job generate. Each decoder block is noted to have three sub-layers.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Two towers of the same brick: one reads, one writes</text>
<rect x="30" y="30" width="200" height="200" rx="8" fill="#F1F8F3" stroke="#2D8B55"/>
<text x="130" y="50" fill="#276b45" font-weight="bold" font-size="11">ENCODER · reads</text>
<rect x="70" y="60" width="120" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="130" y="76" fill="#276b45" font-size="9">block ↔ both ways</text>
<rect x="70" y="90" width="120" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="130" y="106" fill="#276b45" font-size="9">block ↔ both ways</text>
<text x="130" y="130" fill="#9A938A" font-size="12">⋮</text>
<rect x="70" y="138" width="120" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="130" y="154" fill="#276b45" font-size="9">block ↔ both ways</text>
<text x="130" y="176" fill="#276b45" font-size="8">2 sub-layers · no mask</text>
<text x="130" y="192" fill="#276b45" font-size="9">bidirectional</text>
<text x="130" y="208" fill="#6B645E" font-size="9">JOB: understand the input</text>
<text x="130" y="222" fill="#6B645E" font-size="8">(the source, all at once)</text>
<rect x="290" y="30" width="200" height="200" rx="8" fill="#FDF1F1" stroke="#C93B3B"/>
<text x="390" y="50" fill="#C93B3B" font-weight="bold" font-size="11">DECODER · writes</text>
<rect x="330" y="60" width="120" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="76" fill="#C93B3B" font-size="9">block → back only</text>
<rect x="330" y="90" width="120" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="106" fill="#C93B3B" font-size="9">block → back only</text>
<text x="390" y="130" fill="#9A938A" font-size="12">⋮</text>
<rect x="330" y="138" width="120" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="154" fill="#C93B3B" font-size="9">block → back only</text>
<text x="390" y="176" fill="#C93B3B" font-size="8">3 sub-layers · causal mask</text>
<text x="390" y="192" fill="#C93B3B" font-size="9">look back only</text>
<text x="390" y="208" fill="#6B645E" font-size="9">JOB: generate the output</text>
<text x="390" y="222" fill="#6B645E" font-size="8">(one word at a time)</text>
</g></svg>
%%%

**What the read-vs-write picture gets right:** the same brain (same block) does two jobs, and the *rules for looking* separate them — free both ways when reading, forward-only when writing. **Where it breaks down:** you flip between modes; a real model commits each tower to one wiring.

#### The encoder tower · reads the whole input at once
The **[[encoder||A stack of blocks wired to read: every word attends to every other word, both earlier and later. Output = a context-rich understanding of the whole input.]]** stack reads the *whole* input at once. Because nothing is being generated yet — the input already exists — every word is free to look at every other word, both *before* and *after* it. We call this **[[bidirectional||"Bi" = two directions: each word can attend to words on both sides, earlier AND later. The encoder has no mask, so nothing stops a word from looking forward.]]** attention. To understand "I went to the *bank* to fish," a word can peek ahead at "fish" and realize it's the *river* bank. Out the top comes a rich, context-filled understanding of the input.

#### The decoder tower · writes one word at a time
The **[[decoder||A stack of blocks wired to write: each word may only look BACK at words already written (causal mask). Job = generate the output one token at a time.]]** stack *writes* the output. But a writer can't peek at words it hasn't written yet — so its self-attention wears a **[[causal mask||The blindfold: before softmax, every FUTURE score in the attention grid is set to −∞, so a word attends only to itself and earlier words. It's the switch that makes a decoder a decoder.]]**: each word may only look *back* at earlier words, never forward.

Here's the failure that mask prevents, and why it's the remedy:

!!! c-warn ⚠️
<b>Failure — the decoder peeking at the future (cause → remedy):</b> during training we show the decoder the whole target sentence at once for speed. Without a mask, position 3 could just *peek at* position 4 — the very word it's supposed to predict — and cheat. Then at real generation time, when the future genuinely doesn't exist yet, it collapses. <b>Cause:</b> unmasked self-attention lets a word see its own future, so training doesn't match generation. <b>Remedy:</b> the <b>causal mask</b> — set every future score to −∞ before softmax, so a word can only ever look back. Now training behaves exactly like generation: forward only.

So in their **self-attention**, the mask is the *first* of two differences between the towers: the encoder looks both ways, the decoder wears the blindfold. But the decoder also gains an *extra sub-layer* we'll meet next — **cross-attention** — which is why a decoder block has three sub-layers to the encoder's two. Same brick, but the decoder adds one blindfold and one bridge.

!!! c-info 💡
<b>Wait — does ChatGPT really have two towers?</b> No, and this is worth getting right. The *original* transformer above (encoder + decoder + the bridge between them) was built for **translation**, where you have a full input to read and a separate output to write. Some famous models keep this two-tower shape (like T5 for translation-style tasks). But **ChatGPT and the GPT family are <em>decoder-only</em>**: they throw away the encoder and the bridge entirely, keeping *just* the masked-decoder tower plus an output head. There's no separate input to "read" — the prompt and the reply are one growing stream the single tower writes forward. So the parts are all the ones you're learning today; a decoder-only model just uses a subset. We build the full two-tower version because once you see it, the decoder-only version is easy — it's this with two pieces removed.
!!!

**Victory lap:** you've got the two towers — a bidirectional reader (encoder) and a forward-only writer (decoder), differing by the causal mask *and* one extra sub-layer — plus the key caveat that GPT-style models keep only the writer. Next: the little bridge that lets the writer see what the reader understood.

@@@ concept id=c5 tag="The bridge" title="Cross-attention: the writer reads the reader" gotit="Got the bridge"
Two towers standing side by side raises an obvious question: if the encoder *understood* the input and the decoder is *writing* the output, **how does the writer see what the reader understood?** The answer is a small, beautiful bridge called **cross-attention**. It's the wire that connects the two towers — and it's also the reason a decoder block has *three* sub-layers instead of the encoder's two.

Think of **translating a paragraph with the original open on your desk** versus doing it from memory. The clumsy old way: a friend reads you the whole English paragraph, you cram it into your head as one fuzzy gist, they take the paper away, and *then* you translate from memory — and for a long paragraph you've forgotten the start by the time you reach the end. The obvious fix any sane person uses: *keep the paragraph open* and glance back at the exact words you need as you write each one. Cross-attention is "keep the source open and glance back". **Where the picture breaks down:** you glance at printed words, but the decoder glances at the encoder's *vectors* — its rich understanding of each source word, not the raw text.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Cross-attention wiring. On the left the decoder word emits a query Q what do I need. On the right the encoder source words emit keys K what is here and values V the meaning. An arrow shows the query reaching across to match the keys and pull the matching values back. Labeled queries from the decoder, keys and values from the encoder.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Cross-attention: Q from the decoder, K &amp; V from the encoder</text>
<rect x="30" y="66" width="120" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="90" y="86" fill="#C93B3B" font-size="9">decoder word</text>
<text x="90" y="100" fill="#C93B3B" font-size="9">(now writing)</text>
<rect x="52" y="108" width="76" height="16" rx="3" fill="#FFF" stroke="#C93B3B"/>
<text x="90" y="120" fill="#C93B3B" font-size="8">Q — "what do I need?"</text>
<rect x="370" y="56" width="130" height="86" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="435" y="76" fill="#276b45" font-size="9">encoder's understanding</text>
<rect x="385" y="84" width="100" height="16" rx="3" fill="#FFF" stroke="#2D8B55"/>
<text x="435" y="96" fill="#276b45" font-size="8">K — "what's here?"</text>
<rect x="385" y="106" width="100" height="16" rx="3" fill="#FFF" stroke="#2D8B55"/>
<text x="435" y="118" fill="#276b45" font-size="8">V — "the meaning"</text>
<path d="M150 100 L368 92" stroke="#7C6DAA" stroke-width="2" marker-end="url(#xa)"/>
<path d="M368 116 L152 116" stroke="#7C6DAA" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#xa)"/>
<text x="258" y="88" fill="#7C6DAA" font-size="8">Q matches the source K's →</text>
<text x="258" y="140" fill="#7C6DAA" font-size="8">← pulls back the matching V's (source meaning)</text>
<text x="260" y="170" fill="#5E5191" font-size="9">same attention you know — but Q and K/V come from different towers</text>
<defs><marker id="xa" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

**What the "source open on the desk" picture gets right:** cramming everything into one memory fails on long inputs; keeping the source open and glancing back is far better — that's why cross-attention exists. **Where it breaks down:** you glance at printed words; the decoder attends over the encoder's meaning-vectors.

#### Cross-attention in one plain line
It's ordinary attention (which you already know) with the pieces coming from *two different places*: the **queries** ("what am I looking for?") come from the *decoder* — the word it's writing now — while the **keys and values** ("what's available?") come from the *encoder* — its understanding of the source. The decoder word asks a question; the encoder's source words answer.

#### Why a decoder block has THREE sub-layers
An encoder block has two sub-layers (self-attention, feed-forward). A decoder block has *three*, because it needs the bridge in the middle. Here they are, in order:

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A decoder block with three sub-layers stacked bottom to top. One masked self-attention look back only. Two cross-attention with an arrow coming in from the encoder tower on the side. Three feed-forward. Each sub-layer noted as wrapped in a shortcut and a norm. Beside it an encoder block with only two sub-layers for comparison.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Decoder block = THREE sub-layers (encoder has two)</text>
<rect x="20" y="40" width="150" height="150" rx="8" fill="#F1F8F3" stroke="#2D8B55"/>
<text x="95" y="58" fill="#276b45" font-weight="bold" font-size="10">encoder block</text>
<rect x="40" y="70" width="110" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="95" y="89" fill="#276b45" font-size="8">1 · self-attention</text>
<rect x="40" y="110" width="110" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="95" y="129" fill="#276b45" font-size="8">2 · feed-forward</text>
<text x="95" y="168" fill="#6B645E" font-size="8">(each wrapped in</text>
<text x="95" y="180" fill="#6B645E" font-size="8">shortcut + norm)</text>
<rect x="230" y="30" width="180" height="180" rx="8" fill="#FDF1F1" stroke="#C93B3B"/>
<text x="320" y="48" fill="#C93B3B" font-weight="bold" font-size="10">decoder block</text>
<rect x="250" y="58" width="140" height="30" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="320" y="72" fill="#C93B3B" font-size="8">1 · masked self-attention</text><text x="320" y="83" fill="#C93B3B" font-size="7">(look back only)</text>
<rect x="250" y="98" width="140" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="320" y="112" fill="#5E5191" font-size="8">2 · cross-attention</text><text x="320" y="123" fill="#5E5191" font-size="7">(read the encoder)</text>
<rect x="250" y="138" width="140" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="320" y="157" fill="#9A7A10" font-size="8">3 · feed-forward</text>
<text x="320" y="192" fill="#6B645E" font-size="8">(each wrapped in shortcut + norm)</text>
<rect x="430" y="70" width="70" height="130" rx="6" fill="#F1F8F3" stroke="#2D8B55" stroke-dasharray="4,3"/>
<text x="465" y="120" fill="#276b45" font-size="8">encoder</text>
<text x="465" y="132" fill="#276b45" font-size="8">tower's</text>
<text x="465" y="144" fill="#276b45" font-size="8">K &amp; V</text>
<path d="M430 130 L392 113" stroke="#7C6DAA" stroke-width="2" marker-end="url(#br)"/>
<text x="410" y="104" fill="#7C6DAA" font-size="7">bridge</text>
<defs><marker id="br" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

Read the decoder block bottom to top: (1) **masked self-attention** — look back at what I've written so far; (2) **cross-attention** — glance over at the encoder's understanding of the source; (3) **feed-forward** — think it over per word. Three sub-layers, each still wrapped in its shortcut and norm. *This extra middle sub-layer is the second structural difference we promised in the two-towers section* — the encoder never has it.

!!! c-info ⚙️
<b>Optional (skippable) — the three attentions in a full encoder-decoder.</b> Attention runs in three spots: (1) the encoder's *self-attention* (both ways, no mask) to understand the source; (2) the decoder's *masked self-attention* (look-back-only) over what it has generated; (3) *cross-attention*, where decoder queries meet encoder keys/values — the bridge. Only the third mixes the two towers. (In a decoder-only model like GPT, there's no encoder, so spots 1 and 3 disappear — you're left with only the masked self-attention.) Skip this if the picture already did the job.
!!!

**Victory lap:** you've built the bridge — cross-attention lets the writer keep the source open, and it's why a decoder block has three sub-layers. Next: what turns the top of the decoder tower back into an actual word.

@@@ concept id=c6 tag="The output head" title="Turning the top vector back into a word" gotit="Got the head"
The decoder tower hands you a rich vector at the top — but a vector isn't a word. So the very last piece, sitting on the roof of the tower, is the **output head**: it turns that top vector into a *score for every word in the vocabulary*, so you can pick the next one. This is the mirror image of the on-ramp: the on-ramp turned words *into* numbers; the output head turns numbers back *into* a word.

Think of a **talent-show scoreboard**. Every contestant (every word the model knows) gets a score posted on the board. The top vector coming out of the tower is like the judges' whispered verdict; the output head is the scoreboard operator who turns that verdict into a public number *for every contestant at once*, so you can see who's winning. **Where the picture breaks down:** a scoreboard shows raw points, but the output head also converts the scores into clean *shares of 100%* (probabilities) so they add up to one whole.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="The output head. The top vector from the decoder enters a box labeled linear projection over the vocabulary, which outputs a raw score logit for every word. Then softmax turns those into a probability bar chart with mat tallest then floor rug sofa. Labeled top vector to a score for every word to probabilities pick the next word.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Output head: top vector → a score for every word → probabilities</text>
<rect x="20" y="70" width="80" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="60" y="86" fill="#5E5191" font-size="8">top vector</text>
<text x="60" y="97" fill="#5E5191" font-size="8">from tower</text>
<path d="M100 85 L128 85" stroke="#B8AEA2" stroke-width="2" marker-end="url(#oh)"/>
<rect x="130" y="66" width="96" height="38" rx="5" fill="#FCF3DC" stroke="#C99A12"/>
<text x="178" y="82" fill="#9A7A10" font-size="8">linear projection</text>
<text x="178" y="95" fill="#9A7A10" font-size="8">over the vocabulary</text>
<path d="M226 85 L254 85" stroke="#B8AEA2" stroke-width="2" marker-end="url(#oh)"/>
<text x="290" y="70" fill="#6B645E" font-size="8">raw scores (logits)</text>
<text x="290" y="94" fill="#5E5191" font-size="8">→ softmax →</text>
<path d="M330 85 L358 85" stroke="#B8AEA2" stroke-width="2" marker-end="url(#oh)"/>
<line x1="370" y1="150" x2="500" y2="150" stroke="#B8AEA2"/>
<rect x="378" y="70" width="20" height="80" fill="#7C6DAA"/><text x="388" y="163" fill="#2C2A28" font-size="7">mat</text>
<rect x="406" y="110" width="20" height="40" fill="#A99FC9"/><text x="416" y="163" fill="#2C2A28" font-size="7">floor</text>
<rect x="434" y="124" width="20" height="26" fill="#C4BDDA"/><text x="444" y="163" fill="#2C2A28" font-size="7">rug</text>
<rect x="462" y="134" width="20" height="16" fill="#DAD5E8"/><text x="472" y="163" fill="#2C2A28" font-size="7">sofa</text>
<text x="430" y="55" fill="#5E5191" font-size="8">next-word chart</text>
<defs><marker id="oh" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#B8AEA2"/></marker></defs>
</g></svg>
%%%

**What the scoreboard picture gets right:** one verdict becomes a public score for *every* contestant at once, so you can see who wins the next slot. **Where it breaks down:** the head also normalizes the scores into shares of 100%, which no simple scoreboard does.

#### The two steps on the roof
1. A **[[linear projection||A single matrix multiply that maps the top vector to one raw score per word in the whole vocabulary — the model's "how much do I like this word next?" for every word.]] over the vocabulary** turns the top vector into one raw score — a **[[logit||A raw, unbounded score for a word: any number, not yet a probability. Softmax turns logits into probabilities.]]** — for *every* word the model knows.
2. **[[softmax||The step that turns a row of raw scores into probabilities: bigger score → bigger share, and all shares add up to 1.]]** turns those raw scores into a tidy probability for every word, all adding up to 1 — the next-word chart from the generation day.

Said as one plain line: *project the top vector to a score per word, then softmax into probabilities.* Then you pick a word (greedy, or by sampling — that's the last day). Watch it happen on tiny numbers:

%%% demo id=head label="predict, then run it"
code: top_vector = [0.9, -0.2, 0.5]           # the vector out of the tower for one position
code: logits = linear_over_vocab(top_vector)  # → one raw score per vocabulary word
code: probs  = softmax(logits)                # → a probability for every word (sums to 1)
out: logits = [3.0, 1.5, 1.0, 0.2, -1.0]      # raw scores for: mat, floor, rug, sofa, banana
out: probs  = [0.63, 0.14, 0.09, 0.05, 0.02]  # (rest spread thin) — "mat" wins the next slot
out: sum(probs over ALL words) = 1.00         # every word's share adds up to one whole
take: <b>Vector in → a score for every word → probabilities out.</b> The linear head gives each word a raw score; softmax turns them into shares of one jar. Now you have exactly the next-word chart, ready to pick from.
%%%

!!! c-info 💡
<b>A neat trick you'll hear about: weight tying.</b> The on-ramp's embedding table (word → vector) and the output head's projection (vector → word scores) do mirror-image jobs, so many models *share the same weights* for both — saving parameters and often helping quality. You don't need this today; just know the bottom and the top of the tower are close cousins.

**Victory lap:** you've capped the tower — the output head turns the top vector into a next-word chart. Now you have *every* piece. Next: watch them all run together, one token at a time.

@@@ concept id=c7 tag="Run it" title="The whole machine, running one word at a time" gotit="Got the run"
Every piece is built. Now let's turn the key and watch the whole transformer *run* — from a sentence going in the bottom to words coming out the top. The trick that makes it produce a *whole* sentence (not just one word) is the loop you met on the generation day: the decoder writes one word, feeds it back into itself, and writes the next — over and over — until it decides to stop.

Think of **writing a reply text, one word at a time**. You read the message you got (that's the encoder reading the input). Then you type your first word. To choose the *next* word, you re-read your whole reply-so-far *plus* the original message, and pick again. Word by word your reply grows, and you stop when you feel it's *done*. The transformer does exactly this: read the input once, then generate the reply one word at a time, each new word feeding back in. **Where the picture breaks down:** you can delete and rewrite; the basic loop only ever *adds* to the end, one committed word at a time.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="The full transformer running end to end. At the bottom input tokens go through the on-ramp embeddings plus position into the encoder tower producing understanding. The decoder tower reads that via cross-attention and generates one word. The output head turns the top vector into a next word. An arrow curves back showing the new word appended and fed into the decoder again. It loops until the EOS stop token.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">The whole machine end-to-end — looping one word at a time</text>
<rect x="20" y="150" width="120" height="24" rx="4" fill="#FBFAF7" stroke="#7C6DAA"/><text x="80" y="166" fill="#5E5191" font-size="8">input tokens</text>
<path d="M80 150 L80 128" stroke="#B8AEA2" marker-end="url(#rn)"/>
<rect x="20" y="102" width="120" height="22" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="80" y="117" fill="#9A7A10" font-size="8">on-ramp: embed + position</text>
<path d="M80 102 L80 82" stroke="#B8AEA2" marker-end="url(#rn)"/>
<rect x="20" y="40" width="120" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="80" y="57" fill="#276b45" font-size="9">ENCODER stack</text><text x="80" y="70" fill="#276b45" font-size="7">(understand input)</text>
<path d="M140 60 L200 60" stroke="#7C6DAA" stroke-width="2" marker-end="url(#rn)"/><text x="170" y="52" fill="#7C6DAA" font-size="7">cross-attn</text>
<rect x="200" y="40" width="120" height="40" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="260" y="57" fill="#C93B3B" font-size="9">DECODER stack</text><text x="260" y="70" fill="#C93B3B" font-size="7">(write output)</text>
<path d="M320 60 L378 60" stroke="#B8AEA2" stroke-width="2" marker-end="url(#rn)"/>
<rect x="380" y="40" width="120" height="40" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="440" y="57" fill="#5E5191" font-size="9">output head</text><text x="440" y="70" fill="#5E5191" font-size="7">→ next word</text>
<path d="M440 80 C440 140, 260 150, 260 120 L260 82" stroke="#7C6DAA" stroke-width="2" fill="none" stroke-dasharray="5,3" marker-end="url(#rn)"/>
<text x="360" y="150" fill="#7C6DAA" font-size="8">append the new word, feed the decoder again ↺</text>
<path d="M440 80 L440 190" stroke="#B8AEA2" stroke-dasharray="3,3"/>
<path d="M440 190 L470 190" stroke="#C93B3B" stroke-width="2" marker-end="url(#rn)"/>
<rect x="360" y="200" width="150" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="435" y="216" fill="#C93B3B" font-size="8">stop at EOS / max length</text>
<text x="150" y="205" fill="#6B645E" font-size="8">read the input ONCE · then loop the decoder word by word</text>
<defs><marker id="rn" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

**What the reply-text picture gets right:** you read the incoming message once, then build your reply one word at a time, re-reading your own words-so-far each step, and stop when it feels done — exactly the transformer's loop. **Where it breaks down:** you can edit; the loop only appends, committing one word at a time.

#### The end-to-end path, in one breath
Trace the whole machine, bottom to top and around the loop:
1. **Input tokens** go up the **on-ramp** — embedded, and stamped with position.
2. The **encoder stack** reads them all at once and produces its understanding.
3. The **decoder stack** writes one word: it looks back at what it's written (masked self-attention), glances at the encoder's understanding (cross-attention), and thinks per word (feed-forward).
4. The **output head** turns the decoder's top vector into a next-word chart, and you pick a word.
5. **Append** that word and go back to step 3 — until the model draws its **[[EOS token||"End Of Sequence" — a special "I'm done" word the model learns to emit when a thought is finished. When the loop draws EOS, it stops.]]** or hits a length cap.

That "feed my own output back in, one word at a time" is what **[[autoregressive||"Auto" = self, "regressive" = going back over: the model feeds its own output back in as the next input. Each new word becomes part of the question for the next word.]]** means — and it's how *every* transformer generates. Watch three trips around the loop:

%%% demo id=runloop label="predict, then run it"
code: input = "chat: hi there"          # the encoder reads this ONCE
code: reply = ""
code: for step in range(3):
code:     word = transformer.next(input, reply)   # encoder understanding + reply-so-far → next word
code:     reply = reply + " " + word               # append it, loop with the LONGER reply
out: step 1 · reply=""            → picked "Hello"  → reply "Hello"
out: step 2 · reply="Hello"       → picked "how"    → reply "Hello how"
out: step 3 · reply="Hello how"   → picked "are"    → reply "Hello how are"
take: <b>Read the input once, then loop.</b> Each step the decoder re-reads its own growing reply and the encoder's understanding, and adds one word. That feed-back loop is autoregressive generation — the whole machine, running.
%%%

!!! c-info ⚙️
<b>Optional (skippable) — why re-reading is slow, and the fix you'll meet later.</b> Notice step 3 re-processes the whole reply-so-far every single time. For a long answer that's a lot of repeated work. A trick called the <b>KV-cache</b> stores the model's work on words it already saw, so each new word is cheap. You'll build it in the inference-systems module — skip it today; here we only care that the machine *runs*, not how fast.
!!!

**Victory lap:** you just ran the entire transformer end-to-end — input up the on-ramp, through the two towers and the bridge, out the head, and around the loop until it stops. That is the complete machine. One page left: the map that holds it all together.

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just assembled the whole transformer, and you did it with *no new parts*. Before the cheat-sheets, one picture to hold the entire machine in your head.

Think of the **LEGO tower again**: you molded one brick (the block — share, then think, each wrapped in rails), you stacked copies into two towers (a reader and a writer), you clipped a little bridge between them (cross-attention), you put an on-ramp at the bottom (words → embeddings + position) and a scoreboard on the roof (the output head → next-word chart), and you set the whole thing looping one word at a time. That is the original encoder-decoder transformer, built for translation — and the same building blocks power ChatGPT and every large language model (GPT-style models just keep the writer tower and drop the encoder + bridge). **Where this breaks down:** LEGO bricks are identical, but each block in the real tower learns its own weights — same shape, different learned job.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="The whole day as one map. At the bottom the on-ramp tokens to embeddings plus position. Above it two towers, the encoder stack that reads and the decoder stack that writes, joined by a cross-attention bridge. On top the output head turning the top vector into a next-word chart. A loop arrow shows the decoder feeding its own word back in until EOS.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">The full transformer in one map — every piece you built</text>
<rect x="150" y="210" width="220" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12"/>
<text x="260" y="223" fill="#9A7A10" font-size="9">ON-RAMP · tokens → embeddings + position</text>
<text x="260" y="233" fill="#9A7A10" font-size="7">(fixes order-blindness at the bottom)</text>
<path d="M260 210 L260 196" stroke="#B8AEA2" marker-end="url(#mp)"/>
<rect x="40" y="120" width="180" height="72" rx="8" fill="#F1F8F3" stroke="#2D8B55"/>
<text x="130" y="140" fill="#276b45" font-weight="bold" font-size="10">ENCODER stack</text>
<text x="130" y="157" fill="#276b45" font-size="8">N blocks · bidirectional</text>
<text x="130" y="171" fill="#276b45" font-size="8">reads → understands input</text>
<text x="130" y="185" fill="#6B645E" font-size="7">block = attn + FFN (+ rails)</text>
<rect x="300" y="120" width="180" height="72" rx="8" fill="#FDF1F1" stroke="#C93B3B"/>
<text x="390" y="140" fill="#C93B3B" font-weight="bold" font-size="10">DECODER stack</text>
<text x="390" y="157" fill="#C93B3B" font-size="8">N blocks · causal mask</text>
<text x="390" y="171" fill="#C93B3B" font-size="8">writes → generates output</text>
<text x="390" y="185" fill="#6B645E" font-size="7">block = masked-attn + cross-attn + FFN</text>
<path d="M220 156 L300 156" stroke="#7C6DAA" stroke-width="2" marker-end="url(#mp)"/>
<text x="260" y="150" fill="#7C6DAA" font-size="7">cross-attention (the bridge)</text>
<path d="M390 120 L390 100" stroke="#B8AEA2" marker-end="url(#mp)"/>
<rect x="300" y="60" width="180" height="34" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="390" y="76" fill="#5E5191" font-size="9">OUTPUT HEAD</text>
<text x="390" y="88" fill="#5E5191" font-size="7">top vector → next-word chart (softmax)</text>
<path d="M480 77 C512 77, 512 156, 482 156" stroke="#7C6DAA" stroke-width="2" fill="none" stroke-dasharray="5,3" marker-end="url(#mp)"/>
<text x="500" y="120" fill="#7C6DAA" font-size="7" transform="rotate(90 500 120)">loop until EOS</text>
<path d="M300 156 L484 156" stroke="none"/>
<defs><marker id="mp" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

**The day as signposts, in order** — each only makes sense because of the one before it:
- **1 · The on-ramp.** Words become **embeddings**, and **positional information** is stamped in (so a bare, order-blind stack knows word order).
- **2 · The one brick.** The **transformer block**: attention (share) then feed-forward (think), each wrapped in a **residual** shortcut and a **layer norm**.
- **3 · Stack the brick.** Repeat the block **N times** into a tower; each layer refines the words. The two rails fix the deep-stack failures — **vanishing signal → residual**, **scale drift → layer norm**.
- **4 · Two towers.** An **encoder** (bidirectional, no mask — *understands* the input) and a **decoder** (causal mask — *writes* the output). The decoder differs by **two** things: the causal mask on its self-attention (stops it peeking at the future) *and* an extra cross-attention sub-layer. (Decoder-only models like GPT keep only the decoder tower.)
- **5 · The bridge.** **Cross-attention** lets decoder queries read the encoder's keys/values, so the writer sees what the reader understood — and it's that second difference, the extra sub-layer that makes a decoder block have **three**.
- **6 · The output head.** A **linear projection over the vocabulary** + **softmax** turns the top vector into a next-word chart.
- **7 · Run it.** **Autoregressive** generation: read the input once, then loop the decoder one word at a time — appending each word — until **EOS** or a length cap.

#### Cheat-sheet · the pieces of the full transformer
%%% table
:: Piece :: In plain words :: Why it's there
on-ramp (embed + position) :: word → numbers, then stamp where it sits :: the tower needs numbers, and needs to know word order
transformer block :: attention (share) + feed-forward (think), each with rails :: the one repeatable brick the whole tower is built from
residual + layer norm :: a shortcut + a re-leveller around each sub-layer :: keep the signal alive (residual) and the scale steady (layer norm) in a deep stack
encoder stack :: N blocks, bidirectional, no mask, two sub-layers :: read and understand the whole input at once
decoder stack :: N blocks, causal mask, three sub-layers :: write the output one word at a time, looking back only
cross-attention :: decoder Q attends to encoder K/V :: the bridge — let the writer read what the reader understood
output head :: linear over vocabulary → softmax :: turn the top vector into a probability for every next word
autoregressive loop :: feed each new word back in until EOS :: build a whole sentence one committed word at a time
encoder-decoder vs decoder-only :: two towers (original, translation) vs writer-only (GPT/ChatGPT) :: same parts; GPT keeps just the masked decoder + head, drops the encoder and bridge
%%%

#### Cheat-sheet · the words you met (or re-met) today
%%% jargon
full transformer | the whole encoder-decoder machine: on-ramp → encoder stack → decoder stack (with cross-attention) → output head → next word
transformer block | one repeatable unit: attention sub-layer, then feed-forward sub-layer, each wrapped in a residual and a layer norm
stack | repeating the same block N times into a deep tower; each block refines the words a bit more
encoder | a stack wired to read both ways (no mask); understands the whole input; two sub-layers per block
decoder | a stack wired to write forward only (causal mask); generates the output one word at a time; three sub-layers per block
cross-attention | the bridge: decoder queries attend to the encoder's keys and values, so the writer can read the source; the extra decoder-only sub-layer
causal mask | the blindfold that hides future words, so the decoder can only look back — matching training to generation
residual connection | a shortcut adding a sub-layer's input back onto its output; keeps signal and gradients alive in deep stacks
layer normalization | re-levels each word's numbers to a steady scale, so they don't drift as blocks stack
output head | a linear projection over the vocabulary + softmax that turns the top vector into a next-word chart
autoregressive | the loop feeds the model's own output back in as the next input, one word at a time, until EOS
EOS token | the special "I'm done" word; when the loop draws it, generation stops
decoder-only | a transformer that keeps only the masked decoder tower + head (no encoder, no cross-attention) — the shape of GPT and ChatGPT
%%%

That's the whole machine — and the whole module. Every piece you built over eight days now stands together in one tower: the original encoder-decoder transformer, and the same building blocks that power ChatGPT and every large language model. Next up: the **Module 5a Review Gate**, where you'll put it all to the test.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: Trace the full transformer end-to-end. What is the correct order of the main pieces? | a:1 | Output head → encoder stack → on-ramp → decoder stack | On-ramp (embed + position) → encoder stack → decoder stack (with cross-attention) → output head → next word | Decoder stack → encoder stack → on-ramp → output head | Encoder stack → output head → decoder stack → on-ramp | fb: Words enter the on-ramp (embed + position), the encoder understands the input, the decoder writes the output while reading the encoder via cross-attention, and the output head turns the top vector into a next-word chart.
q: A decoder block has THREE sub-layers while an encoder block has two. What is the extra one, and what does it do? | a:2 | An extra feed-forward layer, to add more thinking | A second layer norm, to double the re-leveling | Cross-attention — decoder queries attend to the encoder's keys and values, so the writer can read the encoder's understanding of the input | A second causal mask, to hide more of the future | fb: The extra middle sub-layer is cross-attention: the decoder's queries meet the encoder's keys/values — the bridge that lets the writer read what the reader understood.
q: Why does the decoder use a causal (masked) self-attention, and what would go wrong without it? | a:0 | Without the mask, a word could peek at the future word it's supposed to predict — cheating during training, then collapsing at generation time when the future doesn't exist yet; the mask forces it to look back only | The mask makes the model faster | The mask adds positional information | Without the mask the encoder can't read the input | fb: The causal mask hides future tokens so training matches generation — each word predicts the next using only the past, never peeking ahead.
q: The two rails around each sub-layer — the residual shortcut and layer norm — fix which two failures of a deep stack? | a:2 | Slow training and high memory | Wrong word order and missing tokens | The residual fixes the vanishing/lost signal (gives it a clean highway); layer norm fixes feature-scale drift (re-levels the numbers to a steady size) | Overfitting and underfitting | fb: In a deep raw stack the signal fades (residual gives it an express lane) and the numbers drift in scale (layer norm re-levels them) — the two rails are exactly these two fixes.
%%%

@@@ produce id=produce tag="Produce" title="Wire the whole thing together and run it" gotit="Done"
Time to see today's big idea with your own eyes: *the full transformer is just the pieces you already built, wired in order.* You'll assemble a tiny end-to-end transformer from parts you know and push a sentence through it — on-ramp, encoder stack, decoder stack with cross-attention, output head — then run the autoregressive loop and **watch** it emit words one at a time until it hits a stop. **Predict first:** as the decoder loops, what happens to the length of the reply-so-far each step — and what single signal makes the loop stop? Then run it and **observe**. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-08-full-transformer/experiment.py`. **Notice** four things: (1) build the **on-ramp** — a tiny `embed(tokens)` (look up a small vector per word) plus `add_position(x)`, and print the shape going in; (2) build one `block(x, mask=None)` = a Pre-LN wrapped attention sub-layer then a wrapped feed-forward sub-layer, then make an **encoder** = a few blocks with *no* mask and a **decoder** = a few blocks *with* a causal mask, and add a `cross_attention(dec, enc_out)` step in the decoder — print the shape after each stack and **observe** it's preserved; (3) build the **output head** = a linear projection to a small vocabulary + softmax, and print that the next-word chart sums to 1.0; (4) run the **autoregressive loop**: read the input once, then generate up to N words, each time appending the picked word and feeding it back — **watch** the reply grow one word per step and stop at a chosen EOS token. Run with `python3 sessions/m05a-text-transformer/day-08-full-transformer/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5a Day 8 artifact.

Create sessions/m05a-text-transformer/day-08-full-transformer/experiment.py that, with a comment on each step, assembles a tiny end-to-end transformer from the parts already built and runs it:
1. ON-RAMP: a tiny embed(tokens) that maps each of ~6 vocabulary words to a d_model=8 vector, plus add_position(x) that adds a simple positional signal. Print the (seq_len, 8) shape going in.
2. THE BLOCK: layer_norm(x); a toy self_attention(x, mask=None) that mixes across positions (respect the mask by zeroing above-diagonal weights when a causal mask is given); a position-wise ffn(x) (widen to 32, ReLU, shrink to 8). Wrap each Pre-LN: sublayer(x, fn) = x + fn(layer_norm(x)). Encoder block = attn-sublayer then ffn-sublayer (no mask). Decoder block = masked-attn-sublayer, then a cross_attention(dec, enc_out) sublayer (decoder queries, encoder keys/values), then ffn-sublayer.
3. STACKS: encoder = 2 encoder blocks (no mask) over the input; decoder = 2 decoder blocks (causal mask) that also read enc_out via cross-attention. Print the shape after each stack to show it is preserved (seq_len, 8).
4. OUTPUT HEAD: a linear projection (8 → vocab) + softmax on the decoder's last-position vector; print the next-word chart and confirm it sums to 1.0.
5. AUTOREGRESSIVE LOOP: encode the input once, then generate up to 6 words — each step run the decoder over the reply-so-far, pick argmax from the head, append it, and stop if the picked token is the EOS symbol. Print the reply growing one word per step.
Then run it and paste the output at the bottom as a comment, noting how the reply length grows each step and what makes the loop stop.
%%%

#### What you should see (the whole machine, running)
- The **on-ramp** turns tokens into a `(seq_len, 8)` array of embeddings-plus-position — numbers the tower can use.
- Each **stack** (encoder, then decoder) hands back the *same* shape it received — proof the block preserves width, which is why you can stack it and chain the towers.
- The **output head** turns the decoder's top vector into a next-word chart that sums to `1.0` — a probability for every word.
- The **autoregressive loop** grows the reply *one word per step*, feeding each new word back in, and **stops** the moment it picks the EOS token — the whole transformer, running end to end.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-08-full-transformer/log.md`: (1) the end-to-end path of the full transformer, from input tokens to a next word, in order; (2) the one extra sub-layer a decoder block has that an encoder block doesn't, and what it does; (3) one deep-stack failure and the rail that fixes it (vanishing signal → residual, or scale drift → layer norm).

@@@ fin
