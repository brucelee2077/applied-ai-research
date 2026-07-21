---
quest_id: wf6-d06-encdec
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 6 — Encoder vs Decoder"
module_label: "Module 05a · Represent · Day 6"
title: "Encoder vs Decoder"
subtitle: "Two Ways to Wire the Same Block"
brand_sub: "Foundations · M5a Day 6"
spine: "job"
nav_prev_href: "../day-05-causal-masking/lesson.html"
nav_prev_label: "Causal Masking"
nav_next_href: "../day-07-text-generation/lesson.html"
nav_next_label: "Text Generation &amp; Sampling"
fin_title: "Module 5a · Day 6 complete! 🏆"
fin_body: "Nice work — you've untangled <b>Encoder vs Decoder</b>: two ways to wire the very same block for two different <b>job</b>s. An encoder reads the whole input at once (no mask) to understand it; a decoder writes one word at a time (causal mask) to generate. Same block, different blindfold.<br>Next up: <b>Text Generation &amp; Sampling</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: encoder stack bidirectional self-attention every token attends to every other token no masking full-context representations, keywords: [encoder, bidirectional, look both ways, no mask, whole input at once, every token attends, full context, understand]}
  - {topic: decoder stack masked causal self-attention each token attends only to itself and earlier tokens left to right generation, keywords: [decoder, causal mask, masked self-attention, only look back, left to right, generate, one word at a time, blindfold]}
  - {topic: the causal mask is the single mechanism that turns an encoder into a decoder by hiding future positions, keywords: [same block, only difference, the mask is the switch, turns an encoder into a decoder, add the blindfold, flip the switch, hide the future]}
  - {topic: three architecture shapes encoder-only decoder-only encoder-decoder and which task each fits, keywords: [encoder-only, decoder-only, encoder-decoder, three shapes, three wiring, which job, three ways to build]}
  - {topic: encoder-only property reads the whole sequence at once best for understanding tasks classification tagging embeddings, keywords: [reads the whole sequence, understanding tasks, classification, tagging, embeddings, best for understanding, no notion of next]}
  - {topic: decoder-only property predicts the next token given the past best for open-ended text generation, keywords: [predicts the next token, open-ended generation, best for generation, chatgpt, gpt, writes text]}
  - {topic: cross-attention the bridge in an encoder-decoder where decoder queries attend to encoder output keys and values so generation can read the source, keywords: [cross-attention, the bridge, decoder queries attend to encoder, read the source, keys and values from the encoder, look at the input]}
  - {topic: historical ancestor RNN seq2seq with a single fixed-length context vector and its information bottleneck problem that motivated attention, keywords: [seq2seq, single context vector, fixed-length, information bottleneck, one summary vector, squeezed into one vector, the old way]}
  - {topic: failure answer future leakage in a generator cause unmasked self-attention lets a token peek at future tokens it must predict remedy causal mask, keywords: [answer leakage, future leakage, peek at the future, cheating, unmasked generator, remedy the causal mask]}
  - {topic: failure fixed-context bottleneck of plain seq2seq cause whole source squeezed into one vector remedy cross-attention, keywords: [bottleneck, one vector for a long sentence, forgets the start, remedy cross-attention, squeezed into one vector]}
  - {topic: failure using a bidirectional encoder for open-ended generation cause an encoder has no notion of next it only fills in context remedy choose a decoder causal stack, keywords: [encoder cannot generate, no notion of next, only fills blanks, wrong tool, remedy pick a decoder, use a causal stack for generation]}
  - {topic: capability limit a pure encoder cannot generate open-ended text left to right a pure decoder has weaker whole-sequence bidirectional understanding because it never sees right context, keywords: [encoder cannot roll out, pure decoder weaker understanding, never sees the right context, right tool for the job, built to represent not roll out]}
  - {topic: why the blocks are otherwise identical same multi-head attention feed-forward residual norm masking and cross-attention are the only structural difference, keywords: [otherwise identical, same multi-head attention, same feed-forward, same residual, only structural difference, same lego brick]}
---

@@@ hero
@lede You've done this: you read a whole page of a comic first — eyes darting back and forth, front to back — to *understand* the story. But when you *write* your own story, you can only put down one word at a time, and you can't write a word that depends on a word you haven't written yet. Reading lets you look both ways; writing marches forward one step at a time. Today you'll see that a transformer has these same two modes — an **encoder** that reads the whole input at once to *understand* it, and a **decoder** that writes one word at a time to *generate*. The surprise: they are built from the *exact same block* you already know. The only thing that changes is one **job** — understand vs generate — and one tiny switch (a blindfold). This is the split behind the whole zoo of models you've heard of: the "BERT" family that reads and understands, and the "GPT" family (yes, the one behind ChatGPT) that writes.
@goal Together we'll sort out the two modes for the two **job**s — first *why* one block can do two jobs, then the encoder (reads both ways, for understanding), then the decoder (looks back only, for generating), then the three real shapes you can build (encoder-only, decoder-only, encoder-decoder) and which job each fits, then the clever *bridge* (cross-attention) that lets a writer read a source, and finally which tool is wrong for which job — plus a one-page recap with a cheat-sheet. Every step is a small "aha", and every bit of math is said out loud in plain words first.

@@@ concept id=c1 tag="One block, two jobs" title="Same LEGO brick, two very different jobs" gotit="Got the two jobs"
Let's start with the big picture, before any wiring. In the last few days you built one thing over and over: a **transformer block** — attention, then a small feed-forward step, wrapped in residual connections and normalization. It's a single, reusable LEGO brick. Today's whole surprise is that this *one* brick does *two* completely different **job**s, depending on how you point it. And the two jobs are things you already know from your own life: **understanding** something you're given, and **generating** something new.

Think of your own two modes with a comic book. When you **read** a comic, your eyes are free — you glance ahead at the last panel, back at the first, up at the title, all at once, soaking in the whole page to *understand* the story. That's one **job**. But when you sit down to **write** a comic of your own, you can only draw one panel at a time, in order, and you can't draw panel 5 before you've decided panels 1 through 4 — because panel 5 depends on them. That's the other **job**. Same you, same brain — but "reading to understand" and "writing to create" feel totally different. A transformer is exactly like this: the **encoder** is the block in "reading" mode, and the **decoder** is the block in "writing" mode. **Where the picture breaks down:** you're one person switching between modes, but a real model is usually built *committed* to one mode — the encoder is wired to always read both ways, the decoder is wired to always march forward.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Left: a child reading a comic with eyes darting both ways across the whole page, labeled understand, look both ways, all at once. Right: the same child writing a comic one panel at a time left to right, labeled generate, one word at a time, forward only. A single LEGO brick in the middle feeds both.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One block (brick), two jobs: read-to-understand vs write-to-generate</text>
<rect x="228" y="30" width="64" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="260" y="47" fill="#5E5191" font-size="10">the brick</text>
<path d="M232 58 L150 78" stroke="#B8AEA2"/><path d="M288 58 L372 78" stroke="#B8AEA2"/>
<rect x="40" y="80" width="200" height="110" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="140" y="100" fill="#276b45" font-weight="bold" font-size="10">ENCODER · reading mode</text>
<text x="140" y="122" fill="#276b45" font-size="10">👀 the whole page at once</text>
<text x="140" y="142" fill="#276b45" font-size="9">↔ eyes go both ways ↔</text>
<text x="140" y="164" fill="#276b45" font-size="9">JOB: understand the input</text>
<text x="140" y="182" fill="#6B645E" font-size="8">(no blindfold)</text>
<rect x="280" y="80" width="200" height="110" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="380" y="100" fill="#C93B3B" font-weight="bold" font-size="10">DECODER · writing mode</text>
<text x="380" y="122" fill="#C93B3B" font-size="10">✍️ one word at a time →</text>
<text x="380" y="142" fill="#C93B3B" font-size="9">can only look back ←</text>
<text x="380" y="164" fill="#C93B3B" font-size="9">JOB: generate new text</text>
<text x="380" y="182" fill="#6B645E" font-size="8">(wears the blindfold)</text>
</g></svg>
%%%

**What the comic picture gets right:** the same brain (the same block) does two jobs, and the *rules for looking* are what separate them — free to look anywhere when reading, forced to go forward when writing. **Where it breaks down:** you flip between modes moment to moment; a model usually picks one wiring and stays there.

#### The two words we'll use all day
- **[[encoder||The transformer block wired for understanding: it reads the whole input at once, letting every word look at every other word (both earlier and later). Output = a rich representation of the input. "En-code" = pack the input's meaning into vectors.]]** — the block in *reading* mode. Its job is to *understand* an input you hand it.
- **[[decoder||The transformer block wired for generating: it writes one word at a time and may only look back at words already written. "De-code" = unpack meaning into an actual output sequence, one token at a time.]]** — the block in *writing* mode. Its job is to *generate* new text, left to right.

A **token**, as before, is just a chunk of text — roughly a word — the unit the model reads and writes.

!!! c-ok 🎯
<b>The one idea that unlocks the whole day:</b> the encoder and the decoder are built from the *same* block — same multi-head attention, same feed-forward step, same residuals, same normalization. Nothing structural is added or removed. The *only* difference is a single rule about *what each word is allowed to look at.* Hold onto that: <b>same brick, one tiny switch.</b>

**Victory lap:** you already have the map for the whole lesson — one block, two jobs (understand vs generate). Everything from here just fills in *how* the same brick does each job. Next: the reading mode, the encoder.

@@@ concept id=c2 tag="The encoder" title="The encoder: read the whole page, both ways at once" gotit="Got the encoder"
Let's meet the *reading* mode first, because it's the simpler one — it has *no* blindfold at all. The encoder's **job** is to *understand* an input you give it: a sentence, a question, a review. To understand a word, it's allowed to look at *every* other word in the input — the ones before it *and* the ones after it. We call this looking-both-ways **[[bidirectional||"Bi" = two, "directional" = directions: each word can attend to words on both sides — earlier AND later. The encoder has no mask, so there's nothing stopping a word from looking forward.]]** attention.

Think of **figuring out a tricky word by reading the whole sentence around it**. Someone hands you: "I went to the *bank* to fish." To know which "bank" they mean — money or river — you don't just look at the words *before* "bank". You also peek *ahead* at "fish", and suddenly it's obvious: it's the river bank. You used the future words to understand the present one. That's exactly what the encoder does: to build a rich meaning for each word, it lets that word gather clues from *both* sides. Because the whole input already exists (nobody is generating anything), there's no reason to hide anything — looking ahead isn't cheating here, it's just *reading*. **Where the picture breaks down:** you read the sentence roughly left to right with your eyes; the encoder looks at *all* words *simultaneously*, in one pass, with no order to its glancing.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="The sentence I went to the bank to fish, with the word bank in the middle. Arrows fan out from bank to every other word, both to the left words and the right words including fish. Labeled the encoder lets each word look both ways, so fish disambiguates bank as the river bank.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Encoder: each word gathers clues from BOTH sides (no blindfold)</text>
<g fill="#EFEAF7" stroke="#7C6DAA"><rect x="20" y="70" width="40" height="26" rx="4"/><rect x="70" y="70" width="52" height="26" rx="4"/><rect x="132" y="70" width="36" height="26" rx="4"/><rect x="178" y="70" width="40" height="26" rx="4"/><rect x="290" y="70" width="42" height="26" rx="4"/><rect x="410" y="70" width="46" height="26" rx="4"/></g>
<rect x="228" y="70" width="52" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/>
<g fill="#2C2A28" font-size="10"><text x="40" y="87">I</text><text x="96" y="87">went</text><text x="150" y="87">to</text><text x="198" y="87">the</text><text x="254" y="87">bank</text><text x="311" y="87">to</text><text x="433" y="87">fish</text></g>
<g stroke="#2D8B55" fill="none"><path d="M240 68 C170 30, 90 30, 45 66" marker-end="url(#a)"/><path d="M244 68 C200 40, 160 40, 150 66" marker-end="url(#a)"/><path d="M258 68 C280 42, 300 42, 311 66" marker-end="url(#a)"/><path d="M266 68 C340 28, 400 28, 433 66" marker-end="url(#a)"/></g>
<defs><marker id="a" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#2D8B55"/></marker></defs>
<text x="260" y="130" fill="#276b45" font-size="10">"bank" looks LEFT (I, went, the) AND RIGHT (to, fish)</text>
<text x="260" y="150" fill="#276b45" font-size="10">→ "fish" reveals it's the RIVER bank ✓</text>
<text x="260" y="172" fill="#6B645E" font-size="9">nothing is hidden — the whole input already exists</text>
</g></svg>
%%%

**What the "bank … fish" picture gets right:** later words genuinely help you understand an earlier word, and the encoder is allowed to use them — no blindfold, both directions. **Where it breaks down:** you notice "fish" by re-reading; the encoder wires *every* word to *every* word in a single simultaneous step.

#### What the encoder is *for*
Because every word ends up "understanding" the whole input around it, the encoder is the right tool for **understanding tasks** — the ones where you're handed a complete input and asked a question *about* it:
- **Classification** — is this movie review positive or negative? (read the whole review, then decide)
- **Tagging** — which words are names, which are places? (label each word using its full context)
- **Embeddings** — turn a whole sentence into one meaning-vector for search. (used every time you search by meaning, not keywords)

Notice the shared theme: in all of these, the *whole input is given*, and the output is an *understanding* of it — not brand-new text. That's the encoder's lane.

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Masked-language-modeling training. The sentence the cat sat on the mat with the word sat hidden as a blank. Arrows come from both the left words and the right words into the blank. Labeled hide a word, guess it from BOTH sides — the encoder's fill-in-the-blank training.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">How an encoder trains: hide a word, guess it from BOTH sides</text>
<g fill="#EAF5EE" stroke="#2D8B55"><rect x="40" y="60" width="40" height="24" rx="4"/><rect x="86" y="60" width="42" height="24" rx="4"/><rect x="256" y="60" width="34" height="24" rx="4"/><rect x="296" y="60" width="40" height="24" rx="4"/><rect x="342" y="60" width="46" height="24" rx="4"/></g>
<rect x="160" y="60" width="60" height="24" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-dasharray="4,2"/>
<g fill="#2C2A28" font-size="10"><text x="60" y="76">the</text><text x="107" y="76">cat</text><text x="190" y="76" fill="#9A7A10">[ ??? ]</text><text x="273" y="76">on</text><text x="316" y="76">the</text><text x="365" y="76">mat</text></g>
<g stroke="#2D8B55" fill="none"><path d="M108 60 C130 40, 160 40, 180 58" marker-end="url(#m)"/><path d="M270 60 C240 40, 210 40, 200 58" marker-end="url(#m)"/><path d="M365 60 C300 30, 230 34, 205 58" marker-end="url(#m)"/></g>
<defs><marker id="m" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#2D8B55"/></marker></defs>
<text x="260" y="112" fill="#276b45" font-size="10">left clues (the cat) + right clues (on the mat) → guess "sat" ✓</text>
<text x="260" y="136" fill="#6B645E" font-size="9">it fills a BLANK inside given text — it never rolls out a NEW next word</text>
</g></svg>
%%%

!!! c-info 💡
<b>Optional (skippable) — how an encoder learns to read.</b> An encoder is usually trained by hiding a few random words and asking it to fill them back in ("the cat sat on the ___") — this is called *masked-language modeling*. Because it can look both ways, it's great at this "fill in the blank" game. But notice what it is NOT trained to do: continue a sentence off the end. It fills *gaps* inside a given text; it has no built-in habit of producing a *next* word. (We'll meet this exact limit in a moment.) You'll see the full training story on a later day — skip this for now if it's plenty already.

**Victory lap:** you've got the encoder — a both-ways reader whose job is to *understand* a whole given input. Notice we never once needed to hide anything. Now flip the job: what if the model has to *write*? Next: the decoder.

@@@ concept id=c3 tag="The decoder" title="The decoder: same block, one blindfold added" gotit="Got the decoder"
Here's the moment the whole day clicks. What do you change to turn the reading block into a writing block? Almost nothing. You keep the exact same attention, the same feed-forward step, the same everything. You add *one* rule: **each word may only look *back* at words already written, never *forward*.** That single rule — the [[causal mask||The blindfold from Day 5: before softmax, every future score in the attention grid is set to −∞, so a word can attend only to itself and earlier words. It's the ONE change that turns an encoder into a decoder.]] you met yesterday — is the *only* difference between an encoder and a decoder. Flip that one switch and a "reader" becomes a "writer".

Think of the difference between **reading a finished recipe and inventing a new one as you cook**. When you *read* a recipe, all the steps are printed — you can glance at step 8 while doing step 2. But when you *invent* a dish live, you can only decide the *next* step based on what you've *already* done — step 8 doesn't exist yet, so you obviously can't peek at it. The decoder's **job** is inventing, one step at a time, so it *must* work with a blindfold over the future: it decides word *t* using only words 1 through *t*, exactly the way you'd invent the next cooking step from the steps so far. **Where the picture breaks down:** you invent a recipe slowly, one real step at a time, but a decoder can be *trained* on a whole finished sentence at once (from Day 5) — the mask just guarantees each position still pretends the future isn't there.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Left panel encoder view of the score grid, all cells green allowed, both directions. Right panel decoder view of the same grid, lower triangle green allowed upper triangle grey blocked. An arrow between them labeled add the causal mask. Caption same grid, the mask is the only change.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Same grid — the causal mask is the ONE thing that changes</text>
<text x="90" y="38" fill="#276b45" font-size="10">ENCODER: look both ways</text>
<g stroke="#B8AEA2" fill="#DCEEE2"><rect x="30" y="48" width="28" height="28"/><rect x="58" y="48" width="28" height="28"/><rect x="86" y="48" width="28" height="28"/><rect x="114" y="48" width="28" height="28"/><rect x="30" y="76" width="28" height="28"/><rect x="58" y="76" width="28" height="28"/><rect x="86" y="76" width="28" height="28"/><rect x="114" y="76" width="28" height="28"/><rect x="30" y="104" width="28" height="28"/><rect x="58" y="104" width="28" height="28"/><rect x="86" y="104" width="28" height="28"/><rect x="114" y="104" width="28" height="28"/><rect x="30" y="132" width="28" height="28"/><rect x="58" y="132" width="28" height="28"/><rect x="86" y="132" width="28" height="28"/><rect x="114" y="132" width="28" height="28"/></g>
<text x="86" y="172" fill="#276b45" font-size="9">every cell allowed ✓</text>
<g><text x="230" y="95" fill="#C93B3B" font-size="10">add the</text><text x="230" y="110" fill="#C93B3B" font-size="10">causal mask</text><text x="255" y="130" fill="#C93B3B" font-size="18">→</text></g>
<text x="418" y="38" fill="#C93B3B" font-size="10">DECODER: look back only</text>
<g stroke="#B8AEA2"><rect x="358" y="48" width="28" height="28" fill="#DCEEE2"/><rect x="386" y="48" width="28" height="28" fill="#EDEAE4"/><rect x="414" y="48" width="28" height="28" fill="#EDEAE4"/><rect x="442" y="48" width="28" height="28" fill="#EDEAE4"/><rect x="358" y="76" width="28" height="28" fill="#DCEEE2"/><rect x="386" y="76" width="28" height="28" fill="#DCEEE2"/><rect x="414" y="76" width="28" height="28" fill="#EDEAE4"/><rect x="442" y="76" width="28" height="28" fill="#EDEAE4"/><rect x="358" y="104" width="28" height="28" fill="#DCEEE2"/><rect x="386" y="104" width="28" height="28" fill="#DCEEE2"/><rect x="414" y="104" width="28" height="28" fill="#DCEEE2"/><rect x="442" y="104" width="28" height="28" fill="#EDEAE4"/><rect x="358" y="132" width="28" height="28" fill="#DCEEE2"/><rect x="386" y="132" width="28" height="28" fill="#DCEEE2"/><rect x="414" y="132" width="28" height="28" fill="#DCEEE2"/><rect x="442" y="132" width="28" height="28" fill="#DCEEE2"/></g>
<text x="414" y="172" fill="#C93B3B" font-size="9">future (top-right) blocked ✗</text>
</g></svg>
%%%

**What the recipe picture gets right:** a finished recipe (all steps visible) is the encoder; inventing live (only past steps available) is the decoder — and *inventing* is exactly the job that needs the blindfold. **Where it breaks down:** inventing feels slow and one-step-at-a-time, but the mask lets a decoder *train* on a whole sentence in one fast pass while still acting as if the future is hidden.

#### Watch the switch flip — predict first
Same row of attention scores as an encoder would see, `[2.0, 1.0, 5.0]`, where index 2 is a *future* word. The encoder leaves it alone; the decoder adds −∞ to that future score before softmax. What happens to the future word's weight in each case?

%%% demo id=flip label="predict, then run it"
code: row = [2.0, 1.0, 5.0]   # index 2 is a FUTURE word
code: encoder_view = softmax(row)                 # no mask
code: decoder_view = softmax([2.0, 1.0, 5.0 + -inf])  # causal mask blocks index 2
out: encoder_view = [0.05, 0.02, 0.93]   # future word (index 2) DOMINATES — fine, it's just reading
out: decoder_view = [0.73, 0.27, 0.00]   # future word gets weight 0.00 — the blindfold worked
take: <b>Same block, same scores — the mask is the only change.</b> The encoder happily lets the future word win (understanding, no cheating). The decoder pushes that same future score to −∞, so it wins nothing (generating, no peeking). Flip one switch and a reader becomes a writer.
%%%

#### What the decoder is *for*
Because a decoder predicts *the next token given the past*, it is the right tool for **open-ended generation** — writing new text where you don't know the ending in advance:
- Continue a story, answer a question, hold a chat conversation.
- This is the **GPT** family — the wiring behind ChatGPT. It reads what's been written so far (your prompt plus its own words) and guesses the next word, again and again.

!!! c-warn ⚠️
<b>Failure this prevents — answer leakage (cause → remedy):</b> if you tried to generate text with an *unmasked* (encoder-style) block, every position could peek at the very next word it's supposed to predict — it would just copy the answer during training and then collapse at generation time, when the future genuinely doesn't exist yet. <b>Cause:</b> unmasked self-attention lets a token see its own future. <b>Remedy:</b> the causal mask — the blindfold that makes a decoder a decoder.

**Victory lap:** you just saw the whole encoder-vs-decoder split in one move — *add the causal mask, and a reader becomes a writer.* Same brick, one switch. Next: the three real shapes engineers actually build from these two modes.

@@@ concept id=c4 tag="Three shapes" title="Three real shapes you can build" gotit="Got the three shapes"
Now that you have two modes, you can snap them together in three ways — and each way is a real, famous family of models. This is the part where the names you've heard (BERT, GPT, T5) finally fall into place. The choice of shape always comes down to one question: *what's the **job**?*

Think of **tools in a kitchen drawer**. A *peeler* only peels (one job). A *knife* only cuts (one job). But a *food processor* has a peeling attachment feeding into a chopping blade — it does both, in sequence, because some dishes need reading-then-writing. Model shapes are the same three ideas: use just the reader, use just the writer, or bolt a reader onto a writer for jobs that need both. **Where the picture breaks down:** kitchen tools are physically different objects, but all three model shapes are made of the *same* block — only the *wiring and the mask* differ.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Three model shapes side by side. Encoder-only a single green reading block labeled BERT understand. Decoder-only a single red writing block labeled GPT generate. Encoder-decoder a green reading block feeding into a red writing block labeled T5 BART translate summarize, with an arrow labeled cross-attention bridging them.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Three shapes from the same block — pick by the job</text>
<rect x="20" y="40" width="150" height="150" rx="6" fill="#FBFAF7" stroke="#B8AEA2"/>
<text x="95" y="60" fill="#2C2A28" font-size="10" font-weight="bold">encoder-only</text>
<rect x="45" y="72" width="100" height="46" rx="5" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="95" y="92" fill="#276b45" font-size="9">read ↔ both ways</text>
<text x="95" y="108" fill="#276b45" font-size="9">(no mask)</text>
<text x="95" y="146" fill="#5E5191" font-size="9">e.g. BERT</text>
<text x="95" y="166" fill="#6B645E" font-size="8">job: UNDERSTAND</text>
<text x="95" y="180" fill="#6B645E" font-size="8">classify · tag · embed</text>
<rect x="185" y="40" width="150" height="150" rx="6" fill="#FBFAF7" stroke="#B8AEA2"/>
<text x="260" y="60" fill="#2C2A28" font-size="10" font-weight="bold">decoder-only</text>
<rect x="210" y="72" width="100" height="46" rx="5" fill="#FDECEC" stroke="#C93B3B"/>
<text x="260" y="92" fill="#C93B3B" font-size="9">write → back only</text>
<text x="260" y="108" fill="#C93B3B" font-size="9">(causal mask)</text>
<text x="260" y="146" fill="#5E5191" font-size="9">e.g. GPT / ChatGPT</text>
<text x="260" y="166" fill="#6B645E" font-size="8">job: GENERATE</text>
<text x="260" y="180" fill="#6B645E" font-size="8">chat · continue · answer</text>
<rect x="350" y="40" width="160" height="150" rx="6" fill="#FBFAF7" stroke="#B8AEA2"/>
<text x="430" y="60" fill="#2C2A28" font-size="10" font-weight="bold">encoder–decoder</text>
<rect x="362" y="72" width="62" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="393" y="96" fill="#276b45" font-size="8">read input</text>
<rect x="436" y="72" width="62" height="40" rx="5" fill="#FDECEC" stroke="#C93B3B"/>
<text x="467" y="96" fill="#C93B3B" font-size="8">write output</text>
<path d="M424 92 L436 92" stroke="#7C6DAA" stroke-width="2" marker-end="url(#b)"/>
<defs><marker id="b" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
<text x="430" y="128" fill="#7C6DAA" font-size="8">cross-attention</text>
<text x="430" y="146" fill="#5E5191" font-size="9">e.g. T5 / BART</text>
<text x="430" y="166" fill="#6B645E" font-size="8">job: INPUT → OUTPUT</text>
<text x="430" y="180" fill="#6B645E" font-size="8">translate · summarize</text>
</g></svg>
%%%

**What the kitchen-drawer picture gets right:** you pick the shape by the job, and some jobs (read *then* write) need both tools chained together. **Where it breaks down:** these aren't different objects — they're the same block wired three ways, differing only in the mask and whether a reader feeds a writer.

#### The three shapes, matched to their jobs
- **Encoder-only** (e.g. **BERT**) — just the reader. **Job: understand** a complete input. Use it to classify a review, tag words, or turn a sentence into a search embedding. It reads both ways and stops there — it never generates.
- **Decoder-only** (e.g. **GPT**, the ChatGPT family) — just the writer. **Job: generate** open-ended text. It reads the words so far (your prompt + its own output) and predicts the next word, over and over. Today, this is the most common shape by far.
- **Encoder-decoder** (e.g. **T5**, **BART**) — a reader *feeding* a writer. **Job: turn one sequence into another** — translate English → French, or squeeze a long article into a short summary. The encoder fully *understands* the source; the decoder *generates* the output while *reading* that understanding through a bridge we'll build next.

!!! c-info 💡
<b>Why encoder-decoder for translation but not for chat?</b> Translation has a clear, complete *source* to understand first (the English sentence), then a separate output to write (the French). Reading-then-writing fits perfectly. A chatbot, by contrast, just keeps extending one running conversation — there's no separate "source" to encode, so a single decoder that reads-and-writes the same growing text is simpler and works great. The job decides the shape.

**Victory lap:** the whole model zoo just got simple — three shapes, each picked by its job. Next: the clever bridge that lets the writer in an encoder-decoder actually *read* what the encoder understood.

@@@ concept id=c5 tag="The bridge" title="Cross-attention: how the writer reads the source" gotit="Got cross-attention"
In an encoder-decoder, the reader (encoder) understands the source and the writer (decoder) produces the output — but that raises an obvious question: *how does the writer see what the reader understood?* The answer is a beautiful little bridge called **cross-attention**. And to feel *why* it had to be invented, it helps to peek at the clumsy old way it replaced — because that old way had a famous, painful **job** it kept failing.

Think of **translating a long paragraph by memorizing it, versus keeping it open in front of you**. The old way (before transformers) was like this: a friend reads you a whole English paragraph, you cram *all* of it into your head as one fuzzy "gist", they take the paper away, and *then* you translate from memory alone. For a short sentence, fine. For a long paragraph, disaster — by the time you're translating the end, you've forgotten the beginning. The fix any sane person would use: *keep the paragraph open on the desk* and glance back at the exact words you need as you translate each one. Cross-attention is exactly "keep the source open and glance back at the relevant words". **Where the picture breaks down:** you glance at *whole words on paper*, but the decoder glances at the encoder's *vectors* — its rich understanding of each source word, not the raw text.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two ways to translate. Left the old seq2seq way, a long source sentence squeezed through a funnel into one small fixed context vector, then the decoder writes from that single vector, labeled bottleneck forgets the start. Right cross-attention, the decoder writing each word reaches back to attend to all the encoder word vectors, labeled keeps the whole source open.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Old way: one vector (bottleneck)   →   Fix: cross-attention (source stays open)</text>
<text x="130" y="40" fill="#C93B3B" font-size="10" font-weight="bold">seq2seq: ONE context vector</text>
<g fill="#EFEAF7" stroke="#7C6DAA"><rect x="20" y="52" width="30" height="18"/><rect x="52" y="52" width="30" height="18"/><rect x="84" y="52" width="30" height="18"/><rect x="116" y="52" width="30" height="18"/><rect x="148" y="52" width="30" height="18"/><rect x="180" y="52" width="30" height="18"/></g>
<path d="M20 72 L115 100 M210 72 L115 100" stroke="#C93B3B" fill="none"/>
<rect x="98" y="100" width="34" height="20" rx="4" fill="#FBE7E7" stroke="#C93B3B"/>
<text x="115" y="114" fill="#C93B3B" font-size="8">gist</text>
<path d="M115 120 L115 140" stroke="#C93B3B"/>
<rect x="70" y="140" width="90" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/>
<text x="115" y="156" fill="#C93B3B" font-size="9">decoder writes</text>
<text x="115" y="188" fill="#C93B3B" font-size="8">long input → forgets the start ✗</text>
<line x1="255" y1="30" x2="255" y2="195" stroke="#B8AEA2" stroke-dasharray="3,3"/>
<text x="390" y="40" fill="#276b45" font-size="10" font-weight="bold">cross-attention: source stays open</text>
<g fill="#EAF5EE" stroke="#2D8B55"><rect x="300" y="52" width="26" height="18"/><rect x="328" y="52" width="26" height="18"/><rect x="356" y="52" width="26" height="18"/><rect x="384" y="52" width="26" height="18"/><rect x="412" y="52" width="26" height="18"/><rect x="440" y="52" width="26" height="18"/></g>
<text x="383" y="86" fill="#276b45" font-size="8">encoder word-vectors (all kept)</text>
<rect x="345" y="140" width="76" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B"/>
<text x="383" y="156" fill="#C93B3B" font-size="9">decoder word</text>
<g stroke="#7C6DAA" fill="none"><path d="M383 140 L313 72" marker-end="url(#c)"/><path d="M383 140 L370 72" marker-end="url(#c)"/><path d="M383 140 L453 72" marker-end="url(#c)"/></g>
<defs><marker id="c" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
<text x="383" y="188" fill="#276b45" font-size="8">glances at the exact source words it needs ✓</text>
</g></svg>
%%%

**What the "translate a long paragraph" picture gets right:** cramming everything into one memory (one vector) fails on long inputs; keeping the source open and glancing back is far better — that's the whole reason cross-attention exists. **Where it breaks down:** you glance at printed words; the decoder attends over the encoder's *meaning-vectors*.

#### The historical ancestor, and the failure it caused
- **[[seq2seq||"Sequence to sequence": the pre-transformer design where one RNN reads the whole input into a single fixed-length context vector, and a second RNN generates the output from just that one vector.]]** was the old design. One network read the source into **a single fixed-length context vector** — one small summary of the whole input — and a second network wrote the output from *only* that vector.
- The **failure — the [[information bottleneck||When a whole variable-length input is forced through one fixed-size vector, long inputs lose detail: there simply isn't room to store everything, so the start gets forgotten by the time the end is generated.]]:** a whole paragraph, no matter how long, had to squeeze through one small vector. Long inputs lost their beginning. **Cause:** the entire source is crammed into one fixed-size summary. **Remedy:** cross-attention — don't summarize into one vector; keep *all* the encoder's word-vectors and let the decoder attend to whichever ones it needs, per output word.

#### What cross-attention actually is (in one plain line)
It's ordinary attention (which you already know) with the pieces coming from *two different places*: the **queries** ("what am I looking for?") come from the *decoder* — the word it's currently writing — while the **keys and values** ("what's available to look at?") come from the *encoder* — its understanding of the source. So the decoder word asks a question, and the encoder's source words answer.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Cross-attention wiring. On the left the decoder word emits a query Q. On the right the encoder source words emit keys K and values V. An arrow shows the query reaching across to match the keys and pull the matching values back. Labeled queries from the decoder, keys and values from the encoder.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Cross-attention: Q from the decoder, K &amp; V from the encoder</text>
<rect x="30" y="60" width="120" height="60" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="90" y="80" fill="#C93B3B" font-size="9">decoder word</text>
<text x="90" y="96" fill="#C93B3B" font-size="9">(now writing)</text>
<rect x="55" y="102" width="70" height="14" rx="3" fill="#FFF" stroke="#C93B3B"/>
<text x="90" y="112" fill="#C93B3B" font-size="8">Q — "what do I need?"</text>
<rect x="370" y="52" width="130" height="76" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="435" y="70" fill="#276b45" font-size="9">encoder source words</text>
<rect x="385" y="78" width="100" height="14" rx="3" fill="#FFF" stroke="#2D8B55"/>
<text x="435" y="88" fill="#276b45" font-size="8">K — "what's here?"</text>
<rect x="385" y="98" width="100" height="14" rx="3" fill="#FFF" stroke="#2D8B55"/>
<text x="435" y="108" fill="#276b45" font-size="8">V — "the meaning"</text>
<path d="M150 96 L368 90" stroke="#7C6DAA" stroke-width="2" marker-end="url(#x)"/>
<path d="M368 112 L152 108" stroke="#7C6DAA" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#x)"/>
<defs><marker id="x" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
<text x="258" y="82" fill="#7C6DAA" font-size="8">Q matches the source K's →</text>
<text x="258" y="132" fill="#7C6DAA" font-size="8">← pulls back the matching V's (the source meaning)</text>
<text x="260" y="160" fill="#5E5191" font-size="9">same attention you know — but Q and K/V come from different stacks</text>
</g></svg>
%%%

!!! c-info ⚙️
<b>Optional (skippable) — the three attentions in an encoder-decoder.</b> A full encoder-decoder actually runs attention in three spots: (1) the encoder's *self-attention* (both-ways, no mask) to understand the source; (2) the decoder's *masked self-attention* (look-back-only) over the words it has generated so far; (3) *cross-attention*, where decoder queries meet encoder keys/values — the bridge. Only the third one mixes the two stacks. Skip this if the picture already did the job.

**Victory lap:** you've built the bridge — cross-attention lets a writer keep the whole source open and glance at exactly the words it needs, fixing the old one-vector bottleneck. Next: the flip side — when each tool is the *wrong* one for the job.

@@@ concept id=c6 tag="Wrong tool" title="Right tool, wrong job: the limits" gotit="Got the limits"
Every tool is great at its own **job** and clumsy at the other. Knowing *which* tool fits *which* job is half the skill — and it's exactly what an interviewer probes. So let's make the two mismatches vivid, because they're easy to get backwards.

Think of a **hammer and a screwdriver**. A hammer is perfect for nails and useless for screws; a screwdriver is perfect for screws and useless for nails. Neither is "better" — each is the *right tool for its job* and the *wrong tool for the other*. Encoder and decoder are the same: pick by the job, and using the wrong one doesn't just work slightly worse — it fails at the task's core. **Where the picture breaks down:** a hammer *physically can't* turn a screw, but a mismatched model will often *run* and produce *something* — it just quietly does the job badly, which is more dangerous than an obvious failure.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two mismatches. Left an encoder tool pointed at a generation job, marked wrong, labeled encoder has no notion of next, only fills blanks, cannot roll out new text. Right a decoder tool pointed at a deep understanding job, marked weaker, labeled decoder never sees the right context, weaker whole-sequence understanding.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Right tool, wrong job — the two mismatches to remember</text>
<rect x="20" y="34" width="230" height="150" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="135" y="54" fill="#C93B3B" font-weight="bold" font-size="10">ENCODER → generation ✗</text>
<rect x="55" y="66" width="70" height="34" rx="5" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="90" y="87" fill="#276b45" font-size="9">encoder</text>
<text x="150" y="87" fill="#C93B3B" font-size="16">→</text>
<rect x="165" y="66" width="70" height="34" rx="5" fill="#FBE7E7" stroke="#C93B3B"/>
<text x="200" y="80" fill="#C93B3B" font-size="8">"write me</text>
<text x="200" y="94" fill="#C93B3B" font-size="8">a story"</text>
<text x="135" y="126" fill="#C93B3B" font-size="9">no notion of "next" —</text>
<text x="135" y="142" fill="#C93B3B" font-size="9">it fills blanks, can't roll out</text>
<text x="135" y="168" fill="#276b45" font-size="9">✓ fix: use a decoder</text>
<rect x="270" y="34" width="230" height="150" rx="6" fill="#FCF3DC" stroke="#C99A12"/>
<text x="385" y="54" fill="#9A7A10" font-weight="bold" font-size="10">DECODER → deep understanding ⚠</text>
<rect x="305" y="66" width="70" height="34" rx="5" fill="#FDECEC" stroke="#C93B3B"/>
<text x="340" y="87" fill="#C93B3B" font-size="9">decoder</text>
<text x="400" y="87" fill="#9A7A10" font-size="16">→</text>
<rect x="415" y="66" width="70" height="34" rx="5" fill="#FCF3DC" stroke="#C99A12"/>
<text x="450" y="80" fill="#9A7A10" font-size="8">classify a</text>
<text x="450" y="94" fill="#9A7A10" font-size="8">review</text>
<text x="385" y="126" fill="#9A7A10" font-size="9">never sees the RIGHT context —</text>
<text x="385" y="142" fill="#9A7A10" font-size="9">weaker whole-sentence read</text>
<text x="385" y="168" fill="#276b45" font-size="9">✓ often fine, encoder is stronger</text>
</g></svg>
%%%

**What the hammer-vs-screwdriver picture gets right:** each tool owns its job and struggles at the other; you choose by the task, not by which is "best". **Where it breaks down:** the wrong model usually still *runs* and outputs *something*, so the failure is quiet — you have to *know* the limit to catch it.

#### The two limits, spelled out
- **A pure encoder can't generate open-ended text.** Its job is to *represent* an input it's fully given — it has **no notion of "next"**. Ask it to write a story and there's nothing to do: it fills in *blanks* inside existing text, it doesn't *roll out* new words left to right. **Cause:** an encoder is built to understand, not to produce a next token. **Remedy:** for generation, pick a *decoder* (causal) stack.
- **A pure decoder has weaker whole-sequence understanding.** Because it only ever looks *back*, when it processes a word it has *never seen the words to its right*. For a task like "is this whole review positive?", a both-ways encoder — which lets every word use the full sentence — tends to build a richer understanding. **Cause:** the decoder never sees the right context. **Remedy:** for pure understanding tasks, an encoder is usually the stronger choice (though big decoders can still do a lot).

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it (in an interview):</b> "Encoder and decoder are the same block; the causal mask is the only structural difference. An encoder uses bidirectional self-attention — every token attends to every other token — so it's the right tool for understanding tasks like classification and embeddings, but it can't generate because it has no notion of a next token. A decoder uses masked self-attention so each token sees only the past, which makes it autoregressive and ideal for open-ended generation, at the cost of weaker whole-sequence understanding since it never sees right context. Encoder-decoder models bridge the two with cross-attention, where decoder queries attend to encoder keys and values — the fix for seq2seq's fixed-vector bottleneck." You can already say every word of that.

**Victory lap:** you can now match tool to job *and* name the failure of each mismatch — a genuinely staff-level way to reason about model choice. Next: a one-page recap to lock it all in.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you untangled the whole encoder-vs-decoder story. Before the cheat-sheets, one picture to hold the entire day in your head.

Think of that **comic-book you again**: *reading* the whole page both ways to understand it (the encoder, no mask) versus *writing* one panel at a time, forward only (the decoder, causal mask). Same brain, same block — the only difference is the blindfold. Chain a reader into a writer and you get an encoder-decoder, bridged by cross-attention so the writer can keep the source open. Pick the shape by the **job**: understand → encoder, generate → decoder, one-sequence-into-another → both. **Where this breaks down:** you switch modes freely, but a real model usually commits to one wiring — so choosing the right one *is* the design decision.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Summary. Same block splits two ways by the causal mask. No mask gives the encoder both ways understand shapes encoder-only BERT. Causal mask gives the decoder look back only generate shapes decoder-only GPT. Encoder plus decoder bridged by cross-attention gives encoder-decoder T5 for input to output.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The day in one map: same block, split by the mask, picked by the job</text>
<rect x="205" y="30" width="110" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="260" y="47" fill="#5E5191" font-size="10">the same block</text>
<path d="M225 56 L120 82" stroke="#2D8B55"/><text x="150" y="70" fill="#2D8B55" font-size="8">no mask</text>
<path d="M295 56 L400 82" stroke="#C93B3B"/><text x="372" y="70" fill="#C93B3B" font-size="8">+ causal mask</text>
<rect x="30" y="84" width="180" height="70" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="120" y="104" fill="#276b45" font-weight="bold" font-size="10">ENCODER (both ways)</text>
<text x="120" y="122" fill="#276b45" font-size="9">job: UNDERSTAND</text>
<text x="120" y="140" fill="#5E5191" font-size="9">→ encoder-only · BERT</text>
<rect x="310" y="84" width="180" height="70" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="400" y="104" fill="#C93B3B" font-weight="bold" font-size="10">DECODER (look back)</text>
<text x="400" y="122" fill="#C93B3B" font-size="9">job: GENERATE</text>
<text x="400" y="140" fill="#5E5191" font-size="9">→ decoder-only · GPT</text>
<path d="M210 119 L245 119" stroke="#7C6DAA" stroke-width="2" marker-end="url(#d)"/>
<path d="M310 119 L275 119" stroke="#7C6DAA" stroke-width="2" marker-end="url(#d)"/>
<defs><marker id="d" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#7C6DAA"/></marker></defs>
<rect x="175" y="168" width="170" height="34" rx="6" fill="#FBFAF7" stroke="#7C6DAA"/>
<text x="260" y="184" fill="#7C6DAA" font-size="9">encoder + decoder, bridged by</text>
<text x="260" y="197" fill="#7C6DAA" font-size="9">cross-attention → T5 · input→output</text>
</g></svg>
%%%

**The day as signposts, in order** — each only makes sense because of the one before it:
- **1 · One block, two jobs.** The transformer block does two jobs — *understand* (read) and *generate* (write). Same brick, one switch.
- **2 · The encoder.** Reads *both ways* (**bidirectional**, no mask) to understand a whole given input — best for classification, tagging, embeddings.
- **3 · The decoder.** Adds the **causal mask** so each word looks *back only* — the *single* change that turns a reader into a writer. Best for open-ended generation (GPT/ChatGPT).
- **4 · Three shapes.** Encoder-only (BERT · understand), decoder-only (GPT · generate), encoder-decoder (T5/BART · input→output). Pick by the job.
- **5 · Cross-attention.** The bridge: decoder queries attend to encoder keys/values, so a writer keeps the source open — fixing old **seq2seq**'s one-vector **bottleneck**.
- **6 · Wrong tool.** An encoder can't generate (no notion of "next"); a pure decoder understands whole sequences less well (never sees the right context). Right tool for the right job.

#### Cheat-sheet · encoder vs decoder at a glance
%%% table
:: :: Encoder :: Decoder
Attention :: bidirectional (no mask) :: masked / causal (look back only)
Each word sees :: every word, both directions :: itself and earlier words only
Job :: understand a given input :: generate new text, left to right
Famous family :: BERT :: GPT (ChatGPT)
Fits :: classify · tag · embed :: chat · continue · answer
Structural difference :: — same block — :: + the causal mask (the only change)
%%%

#### Cheat-sheet · the words you met today
%%% jargon
encoder | the block wired to read both ways (no mask); job = understand a whole input
decoder | the block wired to look back only (causal mask); job = generate text left to right
bidirectional | each word may attend to words on both sides — earlier and later
causal mask | the blindfold (Day 5) that hides the future; the ONE change from encoder to decoder
encoder-only | just a reader, e.g. BERT — classification, tagging, embeddings
decoder-only | just a writer, e.g. GPT/ChatGPT — open-ended generation
encoder-decoder | a reader feeding a writer, e.g. T5/BART — translate, summarize
cross-attention | the bridge: decoder queries attend to encoder keys/values, to read the source
seq2seq | the old design: one RNN reads the input into a single fixed context vector, another writes the output
information bottleneck | cramming a whole input into one fixed vector; long inputs lose the start — fixed by cross-attention
%%%

That's the whole day. Next you'll take the decoder you now understand and actually *use* it — turning its next-word guesses into real sentences, and choosing *how* to pick each word: that's **Text Generation & Sampling**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is the ONLY structural difference between an encoder and a decoder? | a:2 | The decoder has more layers | The decoder uses a different feed-forward step | The decoder adds a causal mask so each word looks back only, while the encoder looks both ways — everything else (attention, feed-forward, residuals, norm) is the same | The encoder has no attention at all | fb: Same block; the causal mask is the single switch. No mask → bidirectional encoder (understand); mask → causal decoder (generate).
q: You need to classify whether a whole product review is positive or negative. Which shape fits best? | a:1 | Decoder-only (GPT) | Encoder-only (BERT) — it reads the whole review both ways to understand it | Neither can do this | It must be encoder-decoder | fb: Understanding a complete given input is the encoder's job; it reads both ways. A decoder never sees the right context, so it understands whole sequences less well.
q: Why can't a pure encoder generate open-ended text? | a:3 | It's too slow | It uses too much memory | It has a causal mask that blocks everything | It has no notion of a "next" token — it fills blanks inside given text and can't roll out new words left to right | fb: An encoder is built to represent a given input, not to produce a next token. For generation you need a decoder (causal) stack.
q: In an encoder-decoder, what does cross-attention do, and what problem did it fix? | a:2 | It removes the causal mask from the decoder | It runs softmax twice on the encoder | The decoder's queries attend to the encoder's keys/values, letting the writer read the whole source — fixing old seq2seq's single fixed-vector bottleneck | It merges the encoder and decoder into one block | fb: Cross-attention keeps the source "open": decoder queries meet encoder keys/values, instead of cramming the whole input into one fixed vector that forgot the start.
%%%

@@@ produce id=produce tag="Produce" title="Flip the switch: turn an encoder into a decoder" gotit="Done"
Time to see today's big idea with your own eyes: *the causal mask is the only difference between an encoder and a decoder.* You'll take one grid of attention scores and run it two ways — once as an encoder (no mask) and once as a decoder (add the mask before softmax) — and **watch** the future weights stay alive in one and collapse to zero in the other, from the *same* starting scores. **Predict first:** for the second word in a 4-word sentence, which columns will have non-zero weight in the encoder view but zero weight in the decoder view? (Hint: only the *future* ones change.) Then run it and **observe**. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-06-encoder-vs-decoder/experiment.py`. **Notice** three things: (1) make one 4×4 grid of made-up attention scores and run row-wise softmax with *no* mask — this is the **encoder view**; print it and **observe** that every cell, including future (above-diagonal) ones, has real non-zero weight; (2) build a causal mask (`0` on and below the diagonal, `-inf` above it), add it to the *same* scores, and run softmax again — this is the **decoder view**; print it and **watch** every above-diagonal (future) weight become `0.00` while each row still sums to `1.0`; (3) print the two grids side by side and **notice** that the *only* difference is the future cells — same block, same scores, one switch. Run with `python3 sessions/m05a-text-transformer/day-06-encoder-vs-decoder/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5a Day 6 artifact.

Create sessions/m05a-text-transformer/day-06-encoder-vs-decoder/experiment.py that, with a comment on each step, shows the causal mask is the ONLY difference between an encoder and a decoder:
1. Makes one 4x4 numpy array of arbitrary attention scores for a 4-word sentence.
2. Defines a row-wise softmax. Runs it on the raw scores with NO mask and prints the result — this is the ENCODER view (bidirectional); point out that future (above-diagonal) cells carry real, non-zero weight.
3. Builds a causal_mask that is 0.0 on and below the main diagonal and -np.inf above it (hint: np.triu with k=1), adds it to the SAME scores, runs the same softmax, and prints the result — this is the DECODER view; show every above-diagonal (future) weight is 0.00 and each row sums to 1.0.
4. Prints both grids together and states in a comment that the only difference is the future cells — same block, same scores, the mask is the single switch that turns an encoder into a decoder.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (one switch, two behaviors)
- The **encoder view** (no mask): every cell has real weight, including future (above-diagonal) cells — a word freely looks both ways.
- The **decoder view** (masked): every above-diagonal (future) weight is `0.00`, and each row's allowed weights still sum to `1.0` — the same block, now looking back only.
- Side by side, the *only* difference is the future cells — proof that the causal mask is the single switch between "understand" and "generate".

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-06-encoder-vs-decoder/log.md`: (1) in one sentence, the single difference between an encoder and a decoder; (2) which shape (encoder-only / decoder-only / encoder-decoder) you'd pick for chatting, and why; (3) one thing cross-attention lets an encoder-decoder do that plain seq2seq did badly.

@@@ fin







