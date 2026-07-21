---
quest_id: wf6-d07-generate
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 7 — Text Generation and Sampling"
module_label: "Module 05a · Represent · Day 7"
title: "Text Generation and Sampling"
subtitle: "How a Model Chooses Its Next Word"
brand_sub: "Foundations · M5a Day 7"
spine: "dice"
nav_prev_href: "../day-06-encoder-vs-decoder/lesson.html"
nav_prev_label: "Encoder vs Decoder"
nav_next_href: "../day-08-full-transformer/lesson.html"
nav_next_label: "The Full Transformer"
fin_title: "Module 5a · Day 7 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Text Generation and Sampling</b>: how a model turns its next-word scores into an actual word. Softmax makes a probability for every word, then you pick — greedily (always the top), or by rolling weighted <b>dice</b> (sampling), with temperature and top-k/top-p deciding how adventurous the roll is.<br>Next up: <b>The Full Transformer</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: autoregressive decoding loop, keywords: [one word at a time, feed it back, append, loop, autoregressive, over and over, generation loop, repeat until you stop]}
  - {topic: next-token probability distribution, keywords: [probability for every word, softmax over the vocabulary, next-token distribution, a number for every word, scores into probabilities]}
  - {topic: greedy decoding, keywords: [greedy, always take the top, highest-probability, argmax, always pick the most likely, deterministic]}
  - {topic: temperature, keywords: [temperature, the knob, sharpen the distribution, flatten the distribution, more adventurous, creativity knob, cautious vs bold]}
  - {topic: random sampling, keywords: [sampling, roll weighted dice, draw according to probability, roll the dice, pick at random weighted, varied]}
  - {topic: top-k sampling, keywords: [top-k, keep the k most likely, cut off the tail, shortlist, keep only the top few]}
  - {topic: top-p nucleus sampling, keywords: [top-p, nucleus, add up to p, smallest set that adds up, adaptive cutoff, until the probabilities reach]}
  - {topic: beam search, keywords: [beam search, keep several drafts, several candidates alive, best overall path, several sentences at once]}
  - {topic: repetition penalty, keywords: [repetition penalty, lower the score of words already used, stop repeating, penalize repeats]}
  - {topic: EOS token and max-length cap, keywords: [eos, end-of-sequence, stop token, max length, length cap, knows when to stop, halt]}
  - {topic: greedy repetition failure and remedy, keywords: [stuck in a loop, repeating the same phrase, broken record, remedy sampling, remedy top-p, remedy repetition penalty]}
  - {topic: bland generic output failure and remedy, keywords: [bland, generic, boring, safe text, remedy raise temperature, remedy sampling]}
  - {topic: incoherent gibberish failure and remedy, keywords: [gibberish, incoherent, temperature too high, floods the tail, remedy lower temperature, remedy top-k, remedy top-p]}
  - {topic: never-stopping runaway output failure and remedy, keywords: [never stops, runs forever, runaway, remedy eos token, remedy max-length cap]}
  - {topic: capability limit hallucination extends only from context, keywords: [hallucination, confident but wrong, extends from context, cannot verify truth, cannot look ahead, made up]}
  - {topic: capability limit no memory beyond the context window, keywords: [context window, loses track, forgets earlier detail, no memory beyond, very long generations]}
---

@@@ hero
@lede You've done this on your phone: you type "I am going to the" and three little grey words pop up above the keyboard — "store", "movies", "gym" — and you tap one. Then three new suggestions appear, you tap again, and word by word a whole message builds itself. That little autocomplete guessing game *is* text generation. A trained transformer is just a much, much smarter version of it: it looks at everything written so far, and for its very next move it gives every word a score for "how likely you are to come next". Today's real question — the one behind every wild, funny, or boring thing ChatGPT has ever said — is what happens *after* the scores: **how does the model actually pick one word?** Sometimes you always grab the top suggestion. Sometimes you roll weighted **dice** and let luck in. That single choice is the knob behind a model sounding creative, or safe, or completely off the rails.
@goal Together we'll follow one word from score to page. First the loop that writes a whole passage one word at a time; then the little probability bar-chart the model builds at each step; then the simplest way to pick (greedy, always the top); then the temperature knob that makes the roll bold or cautious; then rolling the weighted dice for real (sampling), trimmed with top-k and top-p; then beam search, which keeps a few drafts alive at once; then the ways generation goes wrong — and its exact fixes — plus how it knows when to stop. We close with a one-page recap and a cheat-sheet. Every idea starts as a picture and an everyday thing you already know, before any math.

@@@ concept id=c1 tag="The loop" title="Writing one word at a time, then feeding it back" gotit="Got the loop"
Before we worry about *which* word to pick, let's see the machine that keeps picking. Yesterday you met the **decoder** — the block that reads what's been written so far and guesses the *next* word. But one guess is just one word. To write a whole sentence, you do that guess again and again, each time with your own last word added on. That repeating machine is called the **[[autoregressive||"Auto" = self, "regressive" = going back over: the model feeds its own output back into itself as the next input. Each new word becomes part of the question for the next word.]] decoding loop**, and it's the heart of every chatbot.

Think of the phone autocomplete from the hero. You type "I", the phone suggests "am", you tap it — now the line reads "I am". The phone looks at "I am" and suggests "going", you tap it — "I am going". Every tap, the phone re-reads *the whole line so far* (including the word you just added) and offers a fresh next word. You never jump ahead; you never edit the middle; you just keep extending the end. The model's loop is exactly this, but *it* does the tapping, using its own guess each time. **Where the picture breaks down:** the phone only ever shows you a handful of suggestions, but the model quietly scores *every* word it knows before it picks — and it never gets tired or bored of tapping.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A conveyor loop showing autoregressive generation. The text so far the cat goes into a box labeled model which outputs a next word sat. That word is appended to make the cat sat and an arrow curves back to feed it into the model again, which outputs on, then down, forming a loop that grows the sentence one word at a time.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The loop: read everything so far → pick ONE word → add it → repeat</text>
<rect x="30" y="48" width="150" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="105" y="68" fill="#5E5191" font-size="10">"The cat" (so far)</text>
<path d="M180 63 L212 63" stroke="#B8AEA2" stroke-width="2" marker-end="url(#l)"/>
<rect x="214" y="48" width="92" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/>
<text x="260" y="68" fill="#C93B3B" font-size="10">the model</text>
<path d="M306 63 L338 63" stroke="#B8AEA2" stroke-width="2" marker-end="url(#l)"/>
<rect x="340" y="48" width="150" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="415" y="68" fill="#276b45" font-size="10">next word: "sat"</text>
<path d="M415 78 C415 120, 260 120, 105 108 L105 82" stroke="#7C6DAA" stroke-width="2" fill="none" stroke-dasharray="5,3" marker-end="url(#l)"/>
<text x="260" y="132" fill="#7C6DAA" font-size="9">append it, feed the longer line back in →</text>
<rect x="90" y="150" width="340" height="30" rx="5" fill="#FBFAF7" stroke="#B8AEA2"/>
<text x="260" y="170" fill="#2C2A28" font-size="10">"The cat" → "The cat sat" → "The cat sat on" → "The cat sat on the mat"</text>
<text x="260" y="198" fill="#6B645E" font-size="9">one word per trip around the loop — this is autoregressive generation</text>
<defs><marker id="l" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

**What the autocomplete picture gets right:** you extend the end one word at a time, always re-reading the whole line, and each new word shapes the next guess. **Where it breaks down:** the phone shows a few options; the model scores its *entire* vocabulary each step, then commits to one.

#### The four steps of every loop
The loop is always the same four beats. Learn these once and you understand how *every* language model writes:
1. **Feed in** the text so far (your prompt, plus anything already generated).
2. **Read the scores** — the model hands back a "how likely to come next" number for every word (we'll draw this next).
3. **Pick one** word from those scores (the *whole* rest of today is about *how* you pick).
4. **Append** the chosen word to the text, and go back to step 1.

You keep spinning this loop until a stop signal fires (we'll meet those at the end). That's it — a whole essay, a whole chat reply, is just this tiny loop turning over and over.

Watch three trips around the loop grow the sentence — the *input* getting longer each time is the whole point of "auto-regressive":

%%% demo id=loopsteps label="predict, then run it"
code: text = "The cat"
code: for step in range(3):
code:     next_word = model.pick(text)     # read scores from text-so-far, pick ONE word
code:     text = text + " " + next_word    # append it, and loop with the LONGER text
out: step 1 · in="The cat"              → picked "sat"  → "The cat sat"
out: step 2 · in="The cat sat"          → picked "on"   → "The cat sat on"
out: step 3 · in="The cat sat on"       → picked "the"  → "The cat sat on the"
take: <b>Each trip, the input is the previous output.</b> The model re-reads the whole growing line and adds one word. That "feed my own output back in" is exactly what *autoregressive* means.
%%%

!!! c-info 💡
<b>Optional (skippable) — why this is slow, and the fix you'll meet later.</b> Notice that step 1 re-reads the *entire* line every single time. For a long reply, that means re-processing the same early words hundreds of times. A trick called the <b>KV-cache</b> stores the model's work on words it already saw, so each new word is cheap. You'll build that in the inference-systems module — skip it for now; today we only care about *how a word is chosen*, not how fast.

**Victory lap:** you've got the engine — a four-step loop that turns one next-word guess into a whole passage. Now let's open up step 2 and *look* at the scores the model hands back, because everything else today is about choosing from them.

@@@ concept id=c2 tag="The scores" title="A tiny bar-chart: a probability for every word" gotit="Got the scores"
Now let's open up step 2 of the loop and *look* at what the model hands back. It is not one word — it's an opinion about *every* word at once. For the vocabulary it knows (tens of thousands of words), it gives each one a raw score, then squeezes those scores into a **[[probability||A number between 0 and 1 saying how likely something is; all the words' probabilities add up to 1, so together they split up "100% of the next-word chance" among all the words.]] for every word**. All those probabilities add up to 1, like slicing a pizza so every slice is somebody's share. This little chart — one bar per word, taller = more likely — is the thing *every* picking strategy today works from. Master this chart and the rest is just "how do I choose from it?"

Think of a **jar of raffle tickets**. Everyone in your class put tickets in a jar for a prize. The kid who put in 50 tickets is very likely to win; the kid who put in 1 ticket almost certainly won't — but they *could*. The next-token chart is that jar: each word owns a pile of tickets, and its probability is just its share of the whole jar. "the" might own half the tickets, "banana" almost none. **Where the picture breaks down:** in a real raffle you draw one ticket blindly; with a model you get to *see* the whole jar first and *choose your rule* for drawing — always take the biggest pile, or actually draw at random, or throw out the tiny piles first. Those rules are the rest of the lesson.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A bar chart of the next-word probability distribution after the phrase the cat sat on the. Five bars of decreasing height labeled mat 0.55, floor 0.18, rug 0.12, sofa 0.08, banana 0.01, and a faded row of dots for thousands of tiny tail words. A note says every bar is one word, taller means more likely, and all bars add up to 1.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">After "The cat sat on the ___" — the model's score for every word</text>
<line x1="60" y1="150" x2="500" y2="150" stroke="#B8AEA2"/>
<g><rect x="80" y="40" width="52" height="110" fill="#7C6DAA"/><text x="106" y="166" fill="#2C2A28" font-size="9">mat</text><text x="106" y="34" fill="#5E5191" font-size="9">0.55</text></g>
<g><rect x="160" y="114" width="52" height="36" fill="#8E80B8"/><text x="186" y="166" fill="#2C2A28" font-size="9">floor</text><text x="186" y="108" fill="#5E5191" font-size="9">0.18</text></g>
<g><rect x="240" y="126" width="52" height="24" fill="#A99FC9"/><text x="266" y="166" fill="#2C2A28" font-size="9">rug</text><text x="266" y="120" fill="#5E5191" font-size="9">0.12</text></g>
<g><rect x="320" y="134" width="52" height="16" fill="#C4BDDA"/><text x="346" y="166" fill="#2C2A28" font-size="9">sofa</text><text x="346" y="128" fill="#5E5191" font-size="9">0.08</text></g>
<g><rect x="400" y="148" width="52" height="2" fill="#DAD5E8"/><text x="426" y="166" fill="#2C2A28" font-size="9">banana</text><text x="426" y="142" fill="#5E5191" font-size="9">0.01</text></g>
<text x="475" y="150" fill="#B8AEA2" font-size="16">· · ·</text>
<text x="260" y="192" fill="#6B645E" font-size="9">one bar per word · taller = more tickets in the jar · all bars add up to 1.00</text>
</g></svg>
%%%

**What the raffle-jar picture gets right:** every word owns a share of one whole "100%", big piles are likely and tiny piles are long-shots, and you get to *see the jar* before deciding how to draw. **Where it breaks down:** you're allowed to change the drawing rule, which no honest raffle would let you do — and that freedom is exactly the power (and danger) of generation.

#### From raw scores to the chart — one plain line
The model's raw outputs are called **[[logits||The raw, unbounded scores the model spits out for each word — can be any number, negative or positive. They are not yet probabilities; softmax turns them into probabilities.]]** — just numbers, big or small, that don't yet add up to anything. A step called **softmax** turns them into the tidy chart:

%%% formula
expr: probability(word) = softmax(logits) — bigger logit → taller bar; all bars sum to 1
note: you already met softmax on the loss day; here it just turns "raw scores" into "shares of the jar"
%%%

You don't need to re-derive softmax today — just hold the picture: *raw scores go in, a clean set of probabilities that sum to 1 comes out.* Let's watch it happen on real numbers so it's concrete, not abstract.

%%% demo id=softmax label="predict, then run it"
code: logits = [3.0, 1.5, 1.0, 0.2]      # raw scores for: mat, floor, rug, sofa
code: probs  = softmax(logits)            # turn them into shares of one jar
out: probs = [0.63, 0.14, 0.09, 0.05]     # (plus a little spread to other words) — biggest score → biggest share
out: sum(probs over ALL words) = 1.00     # every word's share adds up to one whole
take: <b>Raw scores → a probability for every word.</b> The word with the highest logit gets the tallest bar, but the others still keep real, non-zero shares. This chart — not a single word — is what the model actually produces each step. How you pick from it is everything.
%%%

**Victory lap:** you can now *see* what the model hands you every step — a full bar-chart of "how likely is each word", adding up to 1. From here on, the whole game is: given this chart, which word do you take? The simplest answer is next.

@@@ concept id=c3 tag="Greedy" title="Greedy: always grab the tallest bar" gotit="Got greedy"
You've got the chart. Here's the simplest possible rule for picking a word from it: **always take the tallest bar.** Every single step, look at the chart, grab the word with the highest probability, ignore everything else. This is called **[[greedy decoding||A picking rule that always chooses the single highest-probability word at each step. "Greedy" because it grabs the best-looking option right now without thinking about the whole sentence. Also called argmax — "the argument that gives the max".]]** — greedy because it grabs the best-looking option *right now*, without a thought for how the whole sentence turns out.

Think of ordering at your favorite restaurant. Every time you go, you order the *exact same dish* — your #1 favorite, no risk, no thinking. It's a perfectly safe choice: you know you'll like it. But you also never discover anything new, and if a friend asks "what's good here?" your answer is always the same one word. Greedy decoding is that diner: it always orders the top pick. **Where the picture breaks down:** at a restaurant your favorite dish is a real joy; greedy's "favorite word" is only *locally* best — it can march you into a boring or even stuck sentence because it never looks past the very next word.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A menu with the same top item circled every time. On the left a probability chart with the tallest bar mat highlighted and a hand pointing only at it, ignoring the shorter bars floor rug sofa. A caption says greedy always picks the single tallest bar, same input always gives the same output.">
<g font-family="monospace" font-size="11" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Greedy: only ever grab the TALLEST bar — ignore the rest</text>
<line x1="60" y1="130" x2="320" y2="130" stroke="#B8AEA2"/>
<rect x="76" y="46" width="46" height="84" fill="#2D8B55"/><text x="99" y="146" fill="#2C2A28" font-size="9">mat</text><text x="99" y="40" fill="#276b45" font-size="9">0.55 ✓</text>
<rect x="140" y="96" width="46" height="34" fill="#C4BDDA"/><text x="163" y="146" fill="#2C2A28" font-size="9">floor</text>
<rect x="204" y="106" width="46" height="24" fill="#C4BDDA"/><text x="227" y="146" fill="#2C2A28" font-size="9">rug</text>
<rect x="268" y="114" width="46" height="16" fill="#C4BDDA"/><text x="291" y="146" fill="#2C2A28" font-size="9">sofa</text>
<text x="99" y="86" fill="#FFFFFF" font-size="16">👆</text>
<rect x="360" y="46" width="140" height="84" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="430" y="70" fill="#276b45" font-size="10">"the usual"</text>
<text x="430" y="90" fill="#276b45" font-size="9">same input</text>
<text x="430" y="106" fill="#276b45" font-size="9">→ same output</text>
<text x="430" y="122" fill="#6B645E" font-size="8">(deterministic)</text>
<text x="260" y="168" fill="#6B645E" font-size="9">fast · never surprising · can get stuck (we'll see how)</text>
</g></svg>
%%%

**What the favorite-dish picture gets right:** always taking the top choice is fast, safe, and gives the *same* result every time — no gamble at all. **Where it breaks down:** the "best next word" isn't the "best sentence"; a chain of locally-safe words can lock you into repetitive, lifeless text.

#### What greedy gives you — and what it costs
- **Deterministic** — same input, same output, every time. Great when you want a *reproducible* answer (a math result, a fixed label).
- **Fast** — no rolling, no sorting, just "which bar is tallest?" (**argmax**, the "argument that gives the max").
- **The catch:** greedy is *shortsighted*. Picking the single most-likely word now can steer you toward bland, generic sentences — and, worse, into a repeating loop where the model gets stuck saying the same phrase over and over. We'll see that exact failure (and its fixes) later today. For now, remember: greedy is the safe, boring baseline everything else improves on.

Watch greedy's shortsightedness on a two-step toy. Step 1 has a tallest bar "nice"; but choosing the slightly-shorter "sunny" at step 1 unlocks a *much* taller bar at step 2 — a better whole sentence greedy will never see:

%%% demo id=greedyshort label="predict, then run it"
code: step1 = {"nice": 0.52, "sunny": 0.48}      # greedy grabs "nice" (taller)
code: after_nice  = {"day": 0.40}                # then the best next word is only 0.40
code: after_sunny = {"skies": 0.95}              # but "sunny" would have unlocked 0.95!
out: greedy path : "nice"  → "day"    → sentence score 0.52 × 0.40 = 0.208
out: better path : "sunny" → "skies"  → sentence score 0.48 × 0.95 = 0.456  ← higher!
take: <b>Best word now ≠ best sentence.</b> Greedy took the taller first bar and got a worse whole sentence. It can't see past one step — which is exactly the gap beam search (concept 6) tries to close.
%%%

!!! c-info 💡
<b>Optional (skippable) — "greedy" isn't the same as "best sentence".</b> The tallest bar each step doesn't add up to the highest-probability *whole* sentence. Example: picking a slightly-shorter bar now might open up a much taller one next word, giving a better sentence overall — but greedy can't see that far. This is exactly the gap <b>beam search</b> tries to close (concept 6). Skip this if the diner picture already made it click.

**Victory lap:** you've got the simplest picker — greedy, always the tallest bar. It's fast and repeatable but never adventurous. What if you *want* some adventure — a knob that dials the model from cautious to bold? That's temperature, next.

@@@ concept id=c4 tag="Temperature" title="Temperature: the boldness knob" gotit="Got temperature"
Before we let real luck into the picking, we need one knob — the single most famous dial in all of text generation. It's called **[[temperature||A single number that reshapes the probability chart before you pick. Low temperature (below 1) sharpens the chart toward the top word — cautious and repetitive. High temperature (above 1) flattens it — surprising and risky. At temperature 1 the chart is left as-is.]]**, and it decides *how adventurous* the model is before it even picks. It doesn't change *which* words are on the chart — it changes their *shape*: how tall the tallest bar is versus the rest. This is the knob you're actually turning when an app offers a "creative vs precise" slider — it's the dial behind ChatGPT sounding playful or buttoned-up.

Think of **two friends deciding where to eat**. Cautious Cara almost always says "the usual place" — she'll consider somewhere new maybe once a year. Bold Bruno is up for anything: the fancy spot, the food truck, the weird new place with one review — all feel roughly equally exciting to him. Same city, same restaurants, totally different *boldness*. Temperature is that boldness dial for the model: turn it *down* and the top word towers over everything (cautious Cara); turn it *up* and all the words flatten toward equal, so long-shots get a real chance (bold Bruno). **Where the picture breaks down:** your friends' moods drift on their own, but temperature is a number *you* set on purpose before generating.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three probability charts of the same four words at three temperatures. Low temperature 0.5 shows one very tall bar and tiny others cautious. Temperature 1.0 shows the original chart. High temperature 1.5 shows four bars near the same height bold and risky. A dial arrow points from low cautious to high bold.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Same words, one knob: temperature reshapes the chart before you pick</text>
<text x="90" y="36" fill="#276b45" font-size="10">low T = 0.5 (cautious)</text>
<line x1="30" y1="130" x2="160" y2="130" stroke="#B8AEA2"/>
<rect x="40" y="52" width="22" height="78" fill="#2D8B55"/><rect x="70" y="120" width="22" height="10" fill="#A9CDB6"/><rect x="100" y="125" width="22" height="5" fill="#A9CDB6"/><rect x="130" y="127" width="22" height="3" fill="#A9CDB6"/>
<text x="90" y="150" fill="#6B645E" font-size="8">top word towers over all</text>
<text x="260" y="36" fill="#5E5191" font-size="10">T = 1.0 (as-is)</text>
<line x1="200" y1="130" x2="330" y2="130" stroke="#B8AEA2"/>
<rect x="210" y="70" width="22" height="60" fill="#7C6DAA"/><rect x="240" y="104" width="22" height="26" fill="#A99FC9"/><rect x="270" y="112" width="22" height="18" fill="#A99FC9"/><rect x="300" y="118" width="22" height="12" fill="#A99FC9"/>
<text x="260" y="150" fill="#6B645E" font-size="8">the original chart</text>
<text x="430" y="36" fill="#C93B3B" font-size="10">high T = 1.5 (bold)</text>
<line x1="370" y1="130" x2="500" y2="130" stroke="#B8AEA2"/>
<rect x="380" y="92" width="22" height="38" fill="#C93B3B"/><rect x="410" y="98" width="22" height="32" fill="#E39A9A"/><rect x="440" y="102" width="22" height="28" fill="#E39A9A"/><rect x="470" y="106" width="22" height="24" fill="#E39A9A"/>
<text x="430" y="150" fill="#6B645E" font-size="8">bars flatten — long-shots wake up</text>
<path d="M60 178 L470 178" stroke="#B8AEA2" stroke-width="2" marker-end="url(#t)"/>
<text x="120" y="193" fill="#276b45" font-size="9">cautious / repetitive</text>
<text x="410" y="193" fill="#C93B3B" font-size="9">bold / risky</text>
<defs><marker id="t" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#B8AEA2"/></marker></defs>
</g></svg>
%%%

**What the two-friends picture gets right:** the *same* options can feel very peaked (one clear favorite) or very flat (everything roughly equal), and one dial slides you between the two moods. **Where it breaks down:** your friends' boldness wanders; temperature is a deliberate number you choose, and it only reshapes the chart — it never adds a new restaurant to town.

#### The one line, said out loud
Temperature divides every raw score before softmax. That's the whole trick:

%%% formula
expr: new_probs = softmax(logits ÷ T)
note: T < 1 → divide by a small number → scores spread apart → chart SHARPENS (cautious). T > 1 → divide by a big number → scores squeeze together → chart FLATTENS (bold). T = 1 → unchanged.
%%%

You don't need the algebra — you need the feel: **dividing by a small T makes the tall bar even taller; dividing by a big T pulls everyone toward the middle.** Let's watch it on the same four words so you *see* the chart change.

%%% demo id=temp label="predict, then run it"
code: logits = [3.0, 1.5, 1.0, 0.2]              # same raw scores as before
code: cool = softmax([x/0.5 for x in logits])    # T = 0.5  → sharpen (cautious)
code: warm = softmax([x/1.5 for x in logits])    # T = 1.5  → flatten (bold)
out: cool (T=0.5) = [0.85, 0.09, 0.04, 0.02]     # top word now DOMINATES — almost greedy
out: warm (T=1.5) = [0.47, 0.22, 0.17, 0.14]     # bars pulled together — long-shots get real weight
take: <b>One number reshapes the whole chart.</b> Cool it (T<1) and the top word towers up — safe and repetitive, drifting toward greedy. Warm it (T>1) and the bars flatten — surprising, but you're inviting risky long-shots in. Temperature is the boldness dial; it sets the *mood* before you pick.
%%%

!!! c-info 💡
<b>Optional (skippable) — the two extremes.</b> As T → 0, the tall bar swallows everything and sampling becomes *identical to greedy* (always the top word). As T grows very large, the chart flattens toward *totally random* — every word roughly equally likely, which is where gibberish comes from (concept 7). So greedy and pure-random are just the two ends of the temperature dial. Skip if the picture's enough.

**Victory lap:** you've got the model's boldness knob — cool for cautious, warm for bold. But a knob is only half the story: after you shape the chart, you still have to *draw a word from it by chance*. That draw — rolling the weighted **dice** — is next, and it's where generation finally stops being predictable.

@@@ concept id=c5 tag="Rolling dice" title="Sampling: roll weighted dice, then trim the tail" gotit="Got sampling"
Now for the move that makes generation come alive. Instead of always grabbing the tallest bar (greedy), you **actually draw a word by chance** — but a *fair* chance, weighted by the chart. A word with a tall bar is very likely to be drawn; a word with a tiny bar is a long-shot, but *possible*. This is **[[random sampling||Choosing the next word by chance, with each word's chance equal to its probability (its bar height). Unlike greedy, the same prompt can give different words on different runs — that's where variety comes from.]]** — and it's why asking ChatGPT the same question twice can give two different answers. This is the whole reason models feel creative instead of robotic.

Think of a pair of **weighted dice**, or a raffle jar where kids have different numbers of tickets. You shake and draw *for real* — so the outcome isn't fixed — but it's not *fair* in the everyday sense: the kid with 50 tickets wins far more often than the kid with 1. Sampling is exactly this weighted draw from the word-jar. Greedy would just hand the prize to whoever has the most tickets, every time; sampling actually *draws*, so the popular words usually win but a surprise is always on the table. **Where the picture breaks down:** real dice can't be re-shaped mid-game, but you've already got a knob (temperature) that reshapes the odds *before* you roll — and, as we're about to see, you can even *throw some kids out of the jar* first.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A raffle jar of word tickets being drawn. On the left greedy hands the prize straight to the biggest pile mat. On the right sampling shows a hand drawing one ticket from a shaken jar, with mat most likely but floor or rug or even banana possible. A caption says sampling draws by chance, weighted by bar height, so the same prompt can give different words.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Sampling: DRAW a word by chance, weighted by its bar height</text>
<text x="120" y="40" fill="#276b45" font-size="10">greedy: hand it to the biggest pile</text>
<rect x="70" y="52" width="100" height="70" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="120" y="82" fill="#276b45" font-size="10">"mat" (always)</text>
<text x="120" y="102" fill="#6B645E" font-size="8">no gamble</text>
<line x1="255" y1="35" x2="255" y2="150" stroke="#B8AEA2" stroke-dasharray="3,3"/>
<text x="390" y="40" fill="#C93B3B" font-size="10">sampling: shake the jar and DRAW</text>
<ellipse cx="390" cy="95" rx="60" ry="38" fill="#FDECEC" stroke="#C93B3B"/>
<text x="372" y="82" fill="#C93B3B" font-size="12">mat</text>
<text x="418" y="94" fill="#C93B3B" font-size="9">floor</text>
<text x="378" y="108" fill="#C93B3B" font-size="8">rug</text>
<text x="412" y="112" fill="#C93B3B" font-size="7">banana</text>
<text x="390" y="150" fill="#6B645E" font-size="8">usually "mat", sometimes a surprise 🎲</text>
<text x="260" y="172" fill="#6B645E" font-size="9">same prompt → possibly different word each run — this is where variety comes from</text>
</g></svg>
%%%

**What the raffle-jar draw gets right:** you draw *for real* (so results vary), but weighted — favorites win most of the time, long-shots occasionally sneak through. **Where it breaks down:** unlike a fixed jar, you can reshape the odds first (temperature) and even remove the no-hope tickets before drawing (top-k / top-p, coming right up).

#### The catch — and two clever fixes
Pure sampling has one danger: even after shaping, there's a huge *tail* of thousands of barely-possible words, and each has a tiny-but-nonzero chance. Add up thousands of tiny chances and *some* junk word gets drawn now and then — one bad roll and your sentence goes off the rails. The fix is simple and smart: **throw the no-hope words out of the jar before you draw.** Two ways to do it:

- **[[top-k sampling||Before drawing, keep only the k most-likely words (say k=40), throw the rest away, re-share the probabilities among the survivors, then draw. A fixed-size shortlist that kills the junk tail.]]** — keep only the *k* tallest bars (say the top 40 words), toss the rest, then draw from the survivors. A fixed-size shortlist. Simple, but *k* is rigid: 40 words is too many when the model is very sure (one word deserves 99%), and too few when it's genuinely torn among hundreds.
- **[[top-p sampling||Also called nucleus sampling. Before drawing, keep the smallest set of top words whose probabilities add up to p (say 0.9), throw the rest away, then draw. The shortlist grows or shrinks depending on how peaked the chart is.]]** (also called *nucleus* sampling) — keep the *smallest* group of top words whose bars *add up to p* (say 0.9 = 90%), toss the rest, then draw. The clever part: the shortlist **resizes itself**. When the model is sure, 90% might be just 1–2 words. When it's torn, 90% might be 50 words. It adapts to each step's chart.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="Two ways to trim the word chart before sampling. Left top-k with k equals 3 draws a fixed line after the third bar keeping mat floor rug and cutting sofa and the faded tail. Right top-p with p equals 0.9 draws a line where the running total of bars first reaches ninety percent, keeping mat 0.55 plus floor 0.18 plus rug 0.12 equals 0.85 then sofa 0.08 to reach 0.93, an adaptive cutoff. Both then re-share and draw.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Trim the junk tail BEFORE you draw — two ways</text>
<text x="130" y="36" fill="#5E5191" font-size="10">top-k (k=3): fixed shortlist</text>
<line x1="30" y1="130" x2="240" y2="130" stroke="#B8AEA2"/>
<rect x="40" y="60" width="30" height="70" fill="#7C6DAA"/><text x="55" y="143" fill="#2C2A28" font-size="8">mat</text>
<rect x="78" y="104" width="30" height="26" fill="#7C6DAA"/><text x="93" y="143" fill="#2C2A28" font-size="8">floor</text>
<rect x="116" y="112" width="30" height="18" fill="#7C6DAA"/><text x="131" y="143" fill="#2C2A28" font-size="8">rug</text>
<rect x="154" y="118" width="30" height="12" fill="#DAD5E8"/><text x="169" y="143" fill="#9A928A" font-size="8">sofa</text>
<rect x="192" y="125" width="30" height="5" fill="#DAD5E8"/><text x="207" y="143" fill="#9A928A" font-size="7">tail…</text>
<line x1="150" y1="48" x2="150" y2="136" stroke="#C93B3B" stroke-dasharray="4,3"/>
<text x="150" y="162" fill="#C93B3B" font-size="8">keep 3, cut the rest</text>
<text x="390" y="36" fill="#276b45" font-size="10">top-p (p=0.9): adaptive shortlist</text>
<line x1="290" y1="130" x2="500" y2="130" stroke="#B8AEA2"/>
<rect x="300" y="60" width="30" height="70" fill="#2D8B55"/><text x="315" y="143" fill="#2C2A28" font-size="7">mat.55</text>
<rect x="338" y="104" width="30" height="26" fill="#2D8B55"/><text x="353" y="143" fill="#2C2A28" font-size="7">flr.18</text>
<rect x="376" y="112" width="30" height="18" fill="#2D8B55"/><text x="391" y="143" fill="#2C2A28" font-size="7">rug.12</text>
<rect x="414" y="118" width="30" height="12" fill="#2D8B55"/><text x="429" y="143" fill="#2C2A28" font-size="7">sofa.08</text>
<rect x="452" y="125" width="30" height="5" fill="#DAD5E8"/><text x="467" y="143" fill="#9A928A" font-size="7">tail…</text>
<line x1="448" y1="48" x2="448" y2="136" stroke="#C93B3B" stroke-dasharray="4,3"/>
<text x="400" y="162" fill="#276b45" font-size="8">.55+.18+.12+.08 = .93 ≥ 0.9 → stop</text>
<text x="260" y="188" fill="#6B645E" font-size="9">then RE-SHARE the survivors to add up to 1, and draw the weighted dice 🎲</text>
<text x="260" y="200" fill="#9A928A" font-size="8">top-k: always the same count · top-p: count grows/shrinks with how sure the model is</text>
</g></svg>
%%%

%%% demo id=topp label="predict, then run it"
code: chart = {"mat":0.55, "floor":0.18, "rug":0.12, "sofa":0.08, "banana":0.01, "...tail...":0.06}
code: top_k(chart, k=3)        # keep the 3 tallest, drop the rest, re-share
code: top_p(chart, p=0.9)      # keep smallest set that sums to ≥ 0.9, re-share
out: top_k → {"mat":0.65, "floor":0.21, "rug":0.14}          # exactly 3 survivors, re-shared to 1.0
out: top_p → {"mat":0.51, "floor":0.17, "rug":0.11, "sofa":0.07, "banana":0.01, "tail":0.06→trimmed}
out: top_p kept mat+floor+rug+sofa (0.55+0.18+0.12+0.08 = 0.93 ≥ 0.9), tossed the long tail
take: <b>Both trim the no-hope tail, then draw.</b> top-k always keeps a fixed count (here 3). top-p keeps however many it takes to reach 90% — 4 words this step, maybe 1 the next. You get variety *without* the junk. In practice: sample with a modest temperature + top-p ≈ 0.9.
%%%

**Victory lap:** you now hold the modern generation toolkit — roll the weighted **dice** (sampling), shape the odds first (temperature), and cut the junk tail (top-k / top-p). This trio is what almost every real chatbot uses. But there's a totally different philosophy that gambles nothing and instead keeps several drafts alive — beam search, next.

@@@ concept id=c6 tag="Beam search" title="Beam search: keep several drafts alive" gotit="Got beam search"
Everything so far picks *one* word and never looks back. Greedy grabs the tallest bar; sampling rolls the dice. But remember greedy's blind spot: the best word *right now* might not lead to the best *sentence*. What if, instead of committing to one word, you kept a few of the most promising drafts going *at the same time*, and only later decided which one won? That's **[[beam search||A picking rule that keeps several candidate sequences (beams) alive at once, extends each by the top next words, keeps the few highest-scoring whole sequences, and repeats — trading more compute for a higher-probability full sentence.]]** — it trades extra computation for a higher-probability *whole* sentence.

Think of **planning a chess move by looking a few moves ahead**. A weak player grabs the move that looks best right now (that's greedy) and often walks into a trap. A stronger player keeps a *handful* of promising lines in mind — "if I do this, then that; or if I do that, then this" — carries the best few forward move by move, and only then picks the line that's strongest overall. Beam search is that "keep the best few lines alive" habit, applied to words. The number of drafts you keep is the **beam width** (say 4). **Where the picture breaks down:** a chess player reasons about the opponent's replies; beam search has no opponent — it's just keeping the top-scoring *own* continuations, not out-thinking anyone.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A tree of sentence drafts. From the start the sat two branches keep the top two beams the mat and the floor. Each branches again and only the two highest-scoring whole paths survive at each level, drawn as bold lines while lower-scoring branches fade. A caption says beam width equals 2, keep the best few whole drafts alive, greedy would keep only one.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Beam search (width 2): keep the best few WHOLE drafts alive</text>
<rect x="220" y="30" width="80" height="24" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="260" y="46" fill="#5E5191" font-size="9">"The cat"</text>
<path d="M250 54 L150 88" stroke="#2D8B55" stroke-width="2"/><path d="M270 54 L370 88" stroke="#2D8B55" stroke-width="2"/>
<path d="M245 54 L100 88" stroke="#DAD5E8"/><path d="M275 54 L430 88" stroke="#DAD5E8"/>
<rect x="105" y="90" width="90" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="150" y="105" fill="#276b45" font-size="8">…sat (0.6)</text>
<rect x="325" y="90" width="90" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="370" y="105" fill="#276b45" font-size="8">…ran (0.5)</text>
<text x="80" y="104" fill="#9A928A" font-size="8">✗ faded</text><text x="440" y="104" fill="#9A928A" font-size="8">✗ faded</text>
<path d="M140 112 L110 150" stroke="#2D8B55" stroke-width="2"/><path d="M160 112 L215 150" stroke="#2D8B55" stroke-width="2"/>
<path d="M360 112 L305 150" stroke="#DAD5E8"/><path d="M380 112 L420 150" stroke="#DAD5E8"/>
<rect x="65" y="152" width="100" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="115" y="167" fill="#276b45" font-size="8">…sat on (0.54)</text>
<rect x="170" y="152" width="100" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="220" y="167" fill="#276b45" font-size="8">…sat down (0.48)</text>
<text x="260" y="200" fill="#6B645E" font-size="9">at each step: extend every beam, keep the 2 best whole paths · greedy keeps only 1</text>
</g></svg>
%%%

**What the chess picture gets right:** looking a few lines ahead and keeping the best *few* alive beats grabbing the single best-looking move now — you end up with a stronger overall result. **Where it breaks down:** there's no opponent and no true "look-ahead into the future"; beam search only compares its own top-scoring continuations so far, and a wider beam costs proportionally more compute.

#### When beam search helps — and when it hurts
- **Great for:** tasks with one clearly "correct-ish" answer where you want the highest-probability *sentence*, not variety — translation, summarization, short factual answers. Widening the beam (more drafts) buys a better sentence at more compute.
- **Bad for:** open-ended creative writing. Here's the twist — the highest-probability sentence is often the *most boring* one ("I don't know. I don't know. I don't know."). Chasing "most probable" makes beam search bland and even repetitive, the opposite of what you want in a story or a chat.

Watch the same two-step toy from the greedy concept, now with a width-2 beam. Beam keeps *both* first words alive, so it *finds* the better whole sentence greedy threw away:

%%% demo id=beamvs label="predict, then run it"
code: # width-2 beam keeps BOTH "nice" (0.52) and "sunny" (0.48) alive at step 1
code: beam = extend_all(["nice", "sunny"])       # score every whole 2-word draft
out: draft "nice day"    → 0.52 × 0.40 = 0.208
out: draft "sunny skies" → 0.48 × 0.95 = 0.456   ← beam keeps this; greedy had killed "sunny"
out: beam winner = "sunny skies" (0.456)   vs   greedy = "nice day" (0.208)
take: <b>Keeping a few drafts alive finds the better whole sentence.</b> Beam beats greedy here because it didn't commit to the taller first bar. The cost: it carried 2 drafts instead of 1 — width B ≈ B× the work.
%%%

!!! c-warn ⚠️
<b>Failure — bland / generic output (cause → remedy):</b> both pure greedy and beam search chase the *highest-probability* text, and high-probability text is safe and *boring* — generic, repetitive, lifeless. <b>Cause:</b> "most likely" ≠ "most interesting"; the safest words are the dullest. <b>Remedy:</b> for anything creative, drop the "most probable" goal — <b>raise the temperature and sample</b> (with top-p) instead of chasing the top score. This is why chatbots sample rather than beam-search.

!!! c-info 💡
<b>Optional (skippable) — the compute cost.</b> A beam of width B roughly multiplies the work by B (you carry B drafts every step) and needs bookkeeping to track and prune them. That's why beam search shows up in translation/summarization systems but almost never in a live chatbot, where sampling is both cheaper and livelier. Skip if you've got the gist.

**Victory lap:** you've now seen *both* philosophies — gamble for variety (sampling) or hunt for the single best sentence (beam search) — and *when* each is the right or wrong tool. Now the last piece: what happens when generation goes wrong, the exact fix for each, and how the loop knows to *stop*.

@@@ concept id=c7 tag="When it breaks" title="When generation breaks — and how it stops" gotit="Got the failures"
You now have every knob. This last concept is the *maintenance manual*: the three ways generation famously goes wrong, the exact fix for each, and the two signals that tell the loop to *stop*. Each one is a little puzzle — a symptom you've probably seen a chatbot do, and a satisfying one-line cure. Knowing these by name is exactly what separates "I use ChatGPT" from "I understand how it generates."

Think of a **record player with a scratched record**. Three things can go wrong. (1) The needle catches a scratch and the same half-second plays over and over — *stuck in a loop*. (2) You spin the record way too fast and it's a screechy mess — *too much boldness → noise*. (3) The record just… never lifts, hissing on the run-out groove forever — *no stop signal*. Generation has the exact same three failures, and each has a clean fix. **Where the picture breaks down:** a record's problems are physical accidents; a model's are direct results of *your* picking settings, so you fix them by turning knobs, not by cleaning a disc.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three failure cards. One repetition, a broken record icon, text stuck saying the same phrase, cause always the top word greedy, remedy sample top-p or repetition penalty. Two gibberish, a scribble icon, text incoherent junk words, cause temperature too high, remedy lower temperature or top-k top-p. Three never stops, an infinity icon, text runs forever, cause no stop signal, remedy EOS token plus max-length cap.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Three failures — each with a one-line fix</text>
<rect x="14" y="30" width="160" height="150" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="94" y="50" fill="#C93B3B" font-size="20">🔁</text>
<text x="94" y="72" fill="#C93B3B" font-weight="bold" font-size="10">1 · repetition loop</text>
<text x="94" y="92" fill="#2C2A28">"the the the the…"</text>
<text x="94" y="112" fill="#6B645E">cause: always the top</text>
<text x="94" y="124" fill="#6B645E">word (greedy) → cycle</text>
<text x="94" y="150" fill="#276b45">fix: sample / top-p /</text>
<text x="94" y="162" fill="#276b45">repetition penalty</text>
<rect x="180" y="30" width="160" height="150" rx="6" fill="#FCF3DC" stroke="#C99A12"/>
<text x="260" y="50" fill="#9A7A10" font-size="20">🌀</text>
<text x="260" y="72" fill="#9A7A10" font-weight="bold" font-size="10">2 · gibberish</text>
<text x="260" y="92" fill="#2C2A28">"cat зд banana of"</text>
<text x="260" y="112" fill="#6B645E">cause: temperature</text>
<text x="260" y="124" fill="#6B645E">too high → tail junk</text>
<text x="260" y="150" fill="#276b45">fix: lower T /</text>
<text x="260" y="162" fill="#276b45">top-k / top-p</text>
<rect x="346" y="30" width="160" height="150" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/>
<text x="426" y="50" fill="#5E5191" font-size="20">♾️</text>
<text x="426" y="72" fill="#5E5191" font-weight="bold" font-size="10">3 · never stops</text>
<text x="426" y="92" fill="#2C2A28">…and…and…and…</text>
<text x="426" y="112" fill="#6B645E">cause: no stop</text>
<text x="426" y="124" fill="#6B645E">signal → runs forever</text>
<text x="426" y="150" fill="#276b45">fix: EOS token +</text>
<text x="426" y="162" fill="#276b45">max-length cap</text>
</g></svg>
%%%

**What the scratched-record picture gets right:** the same output device can loop, screech, or never stop — three distinct failures, three distinct fixes. **Where it breaks down:** you cure a model's failures by changing *settings you chose* (temperature, top-p, stop rules), not by swapping hardware.

#### Failure 1 · the repetition loop (and the repetition penalty)
Greedy's shortsightedness bites hardest here: if "the" is the top word after "the", greedy picks "the" again… and again… and the model gets **stuck repeating the same phrase**, like a needle on a scratch.
- **Cause:** always taking the argmax can enter a cycle it can't escape — the top word leads to a state whose top word leads back.
- **Remedy:** two options. (a) Stop being greedy — **sample** with top-p, so a different word can break the cycle. (b) Add a **[[repetition penalty||A rule that lowers the score of any word the model has already produced, before picking — so the more a word has been used, the less likely it is chosen again, breaking loops.]]** — actively *lower the score* of words already used, so each repeat gets less and less likely. Most real systems use both.

#### Failure 2 · gibberish (temperature too high)
Turn the boldness knob too far and the chart flattens so much that pure junk from the long tail gets picked — the output becomes **incoherent gibberish**.
- **Cause:** temperature set too high (or no tail-trimming) floods the draw with unlikely words.
- **Remedy:** **lower the temperature**, or **truncate the tail with top-k / top-p** before sampling so no-hope words can't be drawn. (This is the mirror image of failure 1: too-bold vs too-cautious — the sweet spot is in between, which is why "modest temperature + top-p ≈ 0.9" is the popular default.)

#### Failure 3 · never stopping (EOS + max-length)
Nothing in the loop *inherently* ends — left alone it would generate forever. Two stop signals fix this:
- The **[[EOS token||"End Of Sequence": a special token the model is trained to emit when a thought is complete. When the loop draws EOS, it stops. It's how a model chooses to end its own reply.]]** — a special "I'm done" word the model learns to produce when a thought is finished. When the loop picks EOS, it halts. This is how a chatbot decides *its reply is over*.
- A **max-length cap** — a hard limit ("never more than 500 words") so that even if EOS never comes, the loop is forced to stop. A safety net.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="A stop check inside the loop. After each word is appended, a diamond asks did we draw the EOS token or hit the max-length cap. If yes, stop and return. If no, loop back to generate another word. Two exits both lead to stop.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">How the loop knows to stop: two exit signals</text>
<rect x="30" y="40" width="110" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="85" y="59" fill="#276b45" font-size="9">pick + append word</text>
<path d="M140 55 L172 55" stroke="#B8AEA2" stroke-width="2" marker-end="url(#e)"/>
<path d="M240 40 L300 55 L240 70 L180 55 Z" fill="#FCF3DC" stroke="#C99A12"/>
<text x="240" y="52" fill="#9A7A10" font-size="8">drew EOS or hit</text>
<text x="240" y="63" fill="#9A7A10" font-size="8">max length?</text>
<path d="M300 55 L360 55" stroke="#C93B3B" stroke-width="2" marker-end="url(#e)"/>
<text x="330" y="48" fill="#C93B3B" font-size="8">yes</text>
<rect x="362" y="40" width="80" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/>
<text x="402" y="59" fill="#C93B3B" font-size="9">STOP · return</text>
<path d="M240 70 L240 98 L85 98 L85 72" stroke="#7C6DAA" stroke-width="2" fill="none" stroke-dasharray="5,3" marker-end="url(#e)"/>
<text x="160" y="112" fill="#7C6DAA" font-size="8">no → loop back for another word</text>
<defs><marker id="e" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

#### Two hard limits you can't turn a knob to fix
No picking strategy can fix these — they're built into *what generation is*:
- **It only extends from its context, and can be confidently wrong.** The model builds each next word from patterns, not from a fact-checker. It will happily produce fluent, sure-sounding text that is simply *false* — this is called **[[hallucination||When a model generates fluent, confident text that is factually wrong. It's not lying — it has no truth-check; it only continues plausible-sounding patterns from its context.]]**. It cannot look ahead or verify truth. No temperature setting fixes this; you fix it with *outside* tools (giving it real sources, checking its claims) on later days.
- **It has no memory beyond its context window.** The model can only "see" a fixed span of recent text (its **context window**). In a very long generation, the earliest details scroll out of view and are simply *gone* — so long outputs can quietly contradict how they started.

**Victory lap:** you can now *diagnose* generation like an engineer — name the failure, name its cause, reach for the exact fix, and know the two limits no knob can touch. That's genuinely how practitioners talk about it. One page left: the recap.

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you followed one word all the way from the model's scores onto the page, and you met every knob that shapes it. Before the cheat-sheets, one picture to hold the whole day at once.

Think of the **phone autocomplete from the hero, but now you can see the whole guessing game**: the model hands you a jar of word-tickets each step, and your only real decision is *how to draw*. Always grab the biggest pile (greedy), or shake and roll the weighted **dice** (sampling) — with temperature setting how bold the roll is and top-k/top-p tossing the junk tickets first. Or keep several drafts alive and pick the best whole sentence (beam search). Then it loops, until an EOS or a length cap says stop. **Where this breaks down:** autocomplete just suggests; the real loop also decides *how adventurous* to be and *when to stop* — and it can sound confident while being wrong.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="The day in one map. The loop feeds text to the model which outputs a probability chart. From the chart two philosophies branch. Left greedy or beam search chase the most probable, safe but bland. Right sampling draws by chance for variety, shaped by temperature and trimmed by top-k top-p. Both feed the chosen word back into the loop, which ends at EOS or the max-length cap.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">The day in one map: from the chart to the page, and how you pick</text>
<rect x="200" y="28" width="120" height="26" rx="5" fill="#7C6DAA"/>
<text x="260" y="45" fill="#FFF" font-size="10">next-word chart (probs)</text>
<path d="M235 54 L140 82" stroke="#2D8B55"/><text x="150" y="70" fill="#276b45" font-size="8">chase most-probable</text>
<path d="M285 54 L385 82" stroke="#C93B3B"/><text x="395" y="70" fill="#C93B3B" font-size="8">draw by chance</text>
<rect x="24" y="84" width="200" height="66" rx="6" fill="#EAF5EE" stroke="#2D8B55"/>
<text x="124" y="102" fill="#276b45" font-weight="bold" font-size="9">GREEDY · BEAM SEARCH</text>
<text x="124" y="118" fill="#2C2A28">tallest bar / best whole draft</text>
<text x="124" y="132" fill="#6B645E">safe, reproducible — but bland,</text>
<text x="124" y="144" fill="#6B645E">can repeat. good for translation</text>
<rect x="296" y="84" width="200" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B"/>
<text x="396" y="102" fill="#C93B3B" font-weight="bold" font-size="9">SAMPLING 🎲</text>
<text x="396" y="118" fill="#2C2A28">roll weighted dice · variety</text>
<text x="396" y="132" fill="#6B645E">temperature = boldness knob</text>
<text x="396" y="144" fill="#6B645E">top-k / top-p trim the junk tail</text>
<path d="M124 150 L240 178" stroke="#7C6DAA" stroke-dasharray="4,3" marker-end="url(#r)"/>
<path d="M396 150 L280 178" stroke="#7C6DAA" stroke-dasharray="4,3" marker-end="url(#r)"/>
<rect x="170" y="180" width="180" height="26" rx="5" fill="#FBFAF7" stroke="#7C6DAA"/>
<text x="260" y="197" fill="#7C6DAA" font-size="9">append word → loop → stop at EOS / max-length</text>
<defs><marker id="r" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#7C6DAA"/></marker></defs>
</g></svg>
%%%

**The day as signposts, in order** — each step only makes sense because of the one before it:
- **1 · The loop.** Generation is a four-step loop: feed in the text so far → read the scores → pick one word → append and repeat. This is **autoregressive** decoding.
- **2 · The scores.** Each step the model hands back a **probability for every word** (softmax over its vocabulary) — a bar-chart adding up to 1. Every picking rule works from this.
- **3 · Greedy.** Always take the tallest bar (**argmax**). Fast, deterministic — but shortsighted and can get stuck.
- **4 · Temperature.** The boldness knob: divide scores by T before softmax. Low T sharpens (cautious), high T flattens (bold).
- **5 · Sampling + top-k/top-p.** Roll the weighted **dice** for variety; **top-k** keeps a fixed shortlist, **top-p** keeps the smallest set summing to p (adaptive) — both trim the junk tail first.
- **6 · Beam search.** Keep several drafts alive, pick the best *whole* sentence. Good for translation; too bland for chat.
- **7 · When it breaks.** Repetition (→ sample / top-p / repetition penalty), gibberish (→ lower T / top-k / top-p), never stops (→ EOS + max-length). Hard limits: hallucination and the context window.

#### Cheat-sheet · which picker, when
%%% table
:: Strategy :: What it does :: Best for :: Watch out for
Greedy :: always the top word :: exact, reproducible answers :: bland, repetition loops
Sampling + temperature :: roll weighted dice, T = boldness :: creative / chat text :: too-high T → gibberish
Top-k :: keep k best, then draw :: killing the junk tail :: fixed k is rigid
Top-p (nucleus) :: keep smallest set summing to p :: adaptive trimming (default) :: p too high lets junk in
Beam search :: keep best few whole drafts :: translation, summaries :: bland/slow for open-ended
%%%

#### Cheat-sheet · the words you met today
%%% jargon
autoregressive | the loop feeds the model's own output back in as the next input, one word at a time
next-token distribution | the probability the model gives every word to come next; adds up to 1
logits | the raw scores before softmax turns them into probabilities
greedy (argmax) | always pick the single highest-probability word; fast, deterministic, can get stuck
temperature | the boldness knob: divide scores by T — low sharpens (cautious), high flattens (bold)
sampling | draw the next word by chance, weighted by its probability (rolling the dice)
top-k | keep only the k most-likely words, then draw; a fixed shortlist
top-p (nucleus) | keep the smallest set of top words summing to p, then draw; an adaptive shortlist
beam search | keep several candidate sequences alive; pick the best whole sentence
repetition penalty | lower the score of words already produced, to break repeating loops
EOS token | the special "I'm done" word; when drawn, the loop stops
max-length cap | a hard limit that forces the loop to stop even if EOS never comes
hallucination | fluent, confident text that is factually wrong; no knob fixes it
context window | the fixed span of recent text the model can see; earlier detail scrolls out of view
%%%

That's the whole day. Next you'll zoom all the way out and put every piece you've built this module — embeddings, attention, masking, encoder/decoder, and today's generation loop — together into one working machine: **The Full Transformer**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: In the generation loop, what does the model produce at each step, before any word is chosen? | a:2 | A single finished word | The whole sentence at once | A probability for every word in its vocabulary — a chart that adds up to 1 | Nothing until the very end | fb: Each step the model hands back a full next-token distribution (softmax over the vocabulary). Every picking rule — greedy, sampling, beam search — works from this same chart.
q: You turn the temperature way UP and the output becomes incoherent gibberish. What happened, and what's the fix? | a:1 | The model ran out of memory; add more RAM | A high temperature flattened the chart so junk tail words got picked; lower the temperature or trim with top-k / top-p | The EOS token fired too early; remove it | Beam search collapsed; widen the beam | fb: High temperature flattens the distribution, so unlikely tail words get real weight and can be drawn. Lower T, or truncate the tail with top-k / top-p before sampling.
q: How does top-p (nucleus) sampling differ from top-k? | a:3 | top-p is deterministic, top-k is random | top-p always keeps more words than top-k | They are the same thing with different names | top-k keeps a FIXED number of words; top-p keeps the SMALLEST set whose probabilities add up to p, so its size adapts to how peaked the chart is | fb: top-k is a fixed shortlist (say 40). top-p adapts: it keeps however many words it takes to reach p (e.g. 90%) — few when the model is sure, many when it's torn.
q: A greedy decoder gets stuck repeating "the the the". What is the cause, and one good remedy? | a:2 | Cause: temperature too low; remedy: set it to 0 | Cause: the context window is too big; remedy: shrink it | Cause: always taking the top word can enter a repeating cycle; remedy: sample (top-p) or add a repetition penalty | Cause: there is no EOS token; remedy: use beam search | fb: Greedy's argmax can lock into a loop. Break it by sampling instead (top-p), or by adding a repetition penalty that lowers the score of already-used words.
%%%

@@@ produce id=produce tag="Produce" title="Watch one chart become five different words" gotit="Done"
Time to see today's whole idea with your own eyes: *the same chart of scores can become totally different words, depending on how you pick.* You'll take one fixed set of next-word scores and run every strategy on it — greedy, then sampling at a cool and a warm temperature, then top-k, then top-p — and **watch** how the chosen word (and the variety across runs) changes even though the underlying chart never moves. **Predict first:** which strategy will give the *exact same* word every single run, and which one will sometimes surprise you? Then run it and **observe**. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-07-text-generation/experiment.py`. **Notice** five things: (1) make one fixed list of logits for a handful of made-up words (e.g. `[3.0, 1.5, 1.0, 0.2, -1.0]`) and a `softmax` — print the probability chart and **observe** it sums to 1; (2) write `greedy` (return the argmax word) and **notice** it prints the same word every run; (3) write `sample(probs, T)` that divides logits by a temperature, re-softmaxes, and draws with `np.random.choice` — run it several times at `T=0.7` and again at `T=1.5` and **watch** the warmer setting surprise you more often; (4) write `top_k(probs, k)` that keeps the k tallest, re-shares, and samples — print which words survive; (5) write `top_p(probs, p)` that keeps the smallest set summing to ≥ p, re-shares, and samples — **notice** its shortlist size changes if you edit the logits to be more or less peaked. Run with `python3 sessions/m05a-text-transformer/day-07-text-generation/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5a Day 7 artifact.

Create sessions/m05a-text-transformer/day-07-text-generation/experiment.py that, with a comment on each step, shows how the SAME next-token scores become different words under different decoding strategies:
1. Fix one list of logits for ~5 made-up words (e.g. [3.0, 1.5, 1.0, 0.2, -1.0]) and define a softmax. Print the probability chart and confirm it sums to 1.
2. greedy(probs): return the argmax word. Call it a few times and show it is always the same (deterministic).
3. sample(logits, T): divide logits by temperature T, softmax, then np.random.choice weighted by the probs. Run it 5 times at T=0.7 and 5 times at T=1.5; show the warmer temperature produces more variety.
4. top_k(logits, k): keep the k largest logits, set the rest to -inf, softmax, then sample. Print which words survive for k=3.
5. top_p(logits, p): sort by probability, keep the smallest prefix whose cumulative probability >= p, drop the rest, re-normalize, then sample. Print the surviving set for p=0.9, and show it changes size if the logits are made more vs less peaked.
Then run it and paste the output at the bottom as a comment, noting which strategy is deterministic and which surprised you.
%%%

#### What you should see (one chart, many words)
- **Greedy** prints the *same* word every run — it always grabs the tallest bar.
- **Sampling at T=0.7** mostly gives the top word but occasionally a runner-up; at **T=1.5** the surprises come far more often — the boldness knob at work.
- **Top-k** keeps a fixed shortlist (e.g. 3 words) and never draws outside it; **top-p** keeps a shortlist that *grows or shrinks* depending on how peaked you made the chart.
- Across all of them, the underlying chart never changed — only your *rule for picking* did. That's the whole lesson in one run.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-07-text-generation/log.md`: (1) in one sentence, the difference between greedy and sampling; (2) what the temperature knob does, and what happens if you turn it too high; (3) one failure of generation (repetition, gibberish, or never-stopping) and its fix.

@@@ fin
