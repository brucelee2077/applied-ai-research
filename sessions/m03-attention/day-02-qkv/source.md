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
@lede You know how, when you're looking for a book in the library, you walk down the aisle reading the little labels on each spine? You hold one question in your head — "I want something about volcanoes" — and your eyes jump from label to label, spending more time on the books whose labels match, and pulling those ones off the shelf to read. You do this without thinking. Here's the surprise: this exact library trick — hold a question, scan the labels, take more from the matching books — is the single move that makes ChatGPT understand a sentence. It's called **attention**, and today you build its beating heart. Yesterday you turned each word into a little profile card of numbers. But a card just sits there. Today each word learns to *look around the room* — because "bank" can only tell if it means a river-bank or a money-bank by peeking at its neighbors. To pull that off, every word quietly splits its one card into **three** versions of itself: a **Query** (what am I looking for?), a **Key** (what do I advertise?), and a **Value** (what will I share?). Three roles, one word. That little split is the engine of every transformer — and by the end of today it'll feel as natural as walking the aisle.
@goal Together we'll give one word three roles, let a word compare its question to everyone's label with one friendly little score, turn those scores into a "how much to listen" recipe that always adds up to 1, and blend the room into a fresh, context-aware version of the word. You'll also meet the one puzzle this raw scoring runs into at scale — and the neat, named fix that saves it. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.

%%% warmup
q: Yesterday you turned each word into its own short list of numbers. What is that list called? | a:1 | a token id | an embedding | a softmax | a one-hot | concept: embeddings | fb: An embedding is a word's report card — a fixed-length list of numbers that stands for its meaning. Today each word splits this one list into three roles.
q: On yesterday's "map of meaning", what did it mean when two words sat CLOSE together? | a:2 | they are spelled alike | they have the same length | they mean similar things | they came from the same sentence | concept: map-of-meaning | fb: Close = similar meaning. That "how alike are these two arrows?" idea comes back today as the match score between a Query and a Key.
%%%

@@@ concept id=c1 tag="Three roles" title="One word plays three roles at once" gotit="Got the three roles"
Let's start with the surprise that trips up almost everyone: today each single word turns into *three* little lists of numbers, not one. That sounds like numbers appearing from thin air. It isn't. It's because a word has three genuinely different jobs to do in the sentence — and three jobs need three tools.

Think about walking into a [[library||A room full of books; today's picture for a sentence full of words.]] with a question in your head. You do three separate things without noticing. First, you *ask* — you hold your question, "something about volcanoes." Second, every book on the shelf *advertises* itself with a little label on its spine — "COOKING", "SPACE", "VOLCANOES". Third, once you find a matching book, you *take its actual contents* off the shelf to read. Asking, advertising, and handing over content are three different jobs. A word in a sentence does all three at the same time. So it makes three versions of itself: a **Query** (its question), a **Key** (its spine-label), and a **Value** (its contents).

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="One word splits into three roles. On the left, a single word card. Three arrows fan out to three labelled cards: a Query card showing a magnifying glass and the words what am I looking for, a Key card showing a spine label and the words what do I advertise, and a Value card showing a book and the words what will I share."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One word → three roles it plays at the same time</text><rect x="24" y="80" width="96" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="72" y="100" text-anchor="middle" fill="#8A6D3B" font-weight="bold">word</text><text x="72" y="116" text-anchor="middle" fill="#6B645E" font-size="10">"it"</text><line x1="120" y1="96" x2="180" y2="60" stroke="#B8AEA2"/><line x1="120" y1="102" x2="180" y2="102" stroke="#B8AEA2"/><line x1="120" y1="108" x2="180" y2="150" stroke="#B8AEA2"/><rect x="184" y="38" width="316" height="44" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="200" y="58" fill="#5E5191" font-size="16">🔎</text><text x="224" y="58" fill="#5E5191" font-weight="bold">Query</text><text x="224" y="73" fill="#6B645E" font-size="9">"what am I looking for?"</text><rect x="184" y="88" width="316" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="200" y="108" fill="#8A6D3B" font-size="16">🏷️</text><text x="224" y="108" fill="#8A6D3B" font-weight="bold">Key</text><text x="224" y="123" fill="#6B645E" font-size="9">"what do I advertise about myself?"</text><rect x="184" y="138" width="316" height="44" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="200" y="158" fill="#276b45" font-size="16">📖</text><text x="224" y="158" fill="#276b45" font-weight="bold">Value</text><text x="224" y="173" fill="#6B645E" font-size="9">"what content will I share if listened to?"</text></g></svg>
%%%

**What the library picture gets right:** asking, advertising, and handing over content really are three separate jobs — so it makes sense that one word needs three separate versions of itself to do them. **Where it breaks down:** in a real library one person searches while the books sit still on the shelf, but in a sentence *every* word is a searcher, a spine-label, and a book — all at once, all looking at each other at the same time. Nobody waits their turn.

#### Say the three roles out loud
- **Query** = the question a word is asking the room. ("Which word decides my meaning?")
- **Key** = the label a word advertises, so others can check if it's a match. ("I'm a noun about rivers.")
- **Value** = the actual content a word will hand over if someone listens to it.

Here's the sentence we'll use all day: **"the animal crossed the river bank"**. The word "bank" is confused — river-bank or money-bank? Its *Query* is basically "what kind of place am I?" Its neighbor "river" advertises a *Key* that says "I'm about water," and carries a *Value* full of water-meaning. When "bank" 's question matches "river" 's label, "bank" pulls in "river" 's value — and settles on river-bank. That's the whole day in one sentence.

!!! c-info 🗺️
<b>The one line to hold:</b> match your <b>Query</b> against everyone's <b>Key</b>, then take more of the <b>Values</b> whose keys matched best. Everything today is a slow-motion version of that one line.
!!!

@@@ concept id=c2 tag="Making the three" title="Where the three versions come from" gotit="Got the projections"
So a word needs three versions of itself. Fair question: where do those three versions come from? A word only arrived with *one* card — the embedding from Day 1. How does one card become three?

Think of a [[photo booth||A little booth that takes one photo of you and prints three differently-styled copies.]] at a fair that takes *one* photo of you and prints three copies through three different filters. One copy comes out in bright cartoon colors, one in black-and-white, one in sepia. Same you, same original photo — but each filter reshapes it for a different use (the cartoon for a birthday card, the black-and-white for an ID, the sepia for a keepsake). A word does exactly this. It takes its one embedding and passes it through three different *filters* to get its Query, its Key, and its Value. In machine-learning the filter is just a grid of numbers you multiply by, and the multiply-by-a-grid step is called a [[projection||Turning one list of numbers into another by multiplying it by a grid of numbers. Q, K, and V are three projections of the same embedding.]].

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="One embedding card passes through three different filters, drawn as three colored lenses labelled W_Q, W_K and W_V, and comes out as three different cards: a Query card, a Key card, and a Value card. The caption says same input, three filters, three outputs."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One card + three filters → three cards</text><rect x="20" y="88" width="92" height="44" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="66" y="106" text-anchor="middle" fill="#6B645E" font-size="10">embedding</text><text x="66" y="122" text-anchor="middle" fill="#2C2A28">[1, 0]</text><g><line x1="112" y1="104" x2="196" y2="56" stroke="#5E5191" stroke-dasharray="3,2"/><line x1="112" y1="110" x2="196" y2="110" stroke="#C99A12" stroke-dasharray="3,2"/><line x1="112" y1="116" x2="196" y2="164" stroke="#2D8B55" stroke-dasharray="3,2"/></g><ellipse cx="212" cy="52" rx="18" ry="14" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="212" y="56" text-anchor="middle" fill="#5E5191" font-size="9">W_Q</text><ellipse cx="212" cy="110" rx="18" ry="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="212" y="114" text-anchor="middle" fill="#8A6D3B" font-size="9">W_K</text><ellipse cx="212" cy="168" rx="18" ry="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="212" y="172" text-anchor="middle" fill="#276b45" font-size="9">W_V</text><g><line x1="230" y1="52" x2="300" y2="52" stroke="#B8AEA2"/><line x1="230" y1="110" x2="300" y2="110" stroke="#B8AEA2"/><line x1="230" y1="168" x2="300" y2="168" stroke="#B8AEA2"/></g><rect x="304" y="30" width="196" height="44" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="320" y="50" fill="#5E5191" font-weight="bold">Query</text><text x="490" y="50" text-anchor="end" fill="#2C2A28">[0.9, 0.1]</text><text x="320" y="66" fill="#6B645E" font-size="9">the question it asks</text><rect x="304" y="88" width="196" height="44" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="320" y="108" fill="#8A6D3B" font-weight="bold">Key</text><text x="490" y="108" text-anchor="end" fill="#2C2A28">[0.2, 0.8]</text><text x="320" y="124" fill="#6B645E" font-size="9">the label it advertises</text><rect x="304" y="146" width="196" height="44" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="320" y="166" fill="#276b45" font-weight="bold">Value</text><text x="490" y="166" text-anchor="end" fill="#2C2A28">[0.5, 0.5]</text><text x="320" y="182" fill="#6B645E" font-size="9">the content it shares</text></g></svg>
%%%

**What the photo-booth picture gets right:** one input, three filters, three genuinely different outputs — that's exactly Q, K, V from one embedding. **Where it breaks down:** a photo booth's filters come pre-set at the factory, but a word's three filters are **learned** — during training the model slowly tunes them so the Queries and Keys line up in useful ways. Nobody types them in, just like Day 1's embeddings.

#### The three filters have names
The three filters are three grids of numbers called [[W_Q, W_K, W_V||The three learned filter-grids that turn one embedding into a Query, a Key, and a Value. Training tunes them.]] — one grid per role. "W" is just the usual letter for a grid of *weights* (numbers the model learns). In one plain line: to get a word's Query, multiply its embedding by the Query-grid; same for Key and Value. That's it — three multiplies, one per role.

%%% formula
expr: Q = x · W_Q     K = x · W_K     V = x · W_V
note: x = one word's embedding (its Day-1 card). Each W is a learned grid. The dot means "multiply the list by the grid." Three multiplies → three role-vectors.
%%%

Now let's *see the shapes* actually change as the multiply happens — this is the build-up, and shapes are the thing beginners lose track of. Watch the figure assemble left to right.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A shape diagram. A card of shape 1 by 2 is multiplied by a grid W of shape 2 by 3, producing an output of shape 1 by 3. The middle number 2 is highlighted as the number that must match for the multiply to work. The caption says the same multiply happens three times, once per grid."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Multiply reshapes the card — watch the numbers</text><rect x="34" y="70" width="70" height="50" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="69" y="92" text-anchor="middle" fill="#2C2A28">x</text><text x="69" y="140" text-anchor="middle" fill="#6B645E" font-size="10">1 × 2</text><text x="69" y="153" text-anchor="middle" fill="#9A938A" font-size="9">the word</text><text x="120" y="100" fill="#B8AEA2" font-size="16">×</text><rect x="140" y="60" width="70" height="70" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="175" y="98" text-anchor="middle" fill="#8A6D3B">W</text><text x="175" y="150" text-anchor="middle" fill="#6B645E" font-size="10">2 × 3</text><text x="175" y="163" text-anchor="middle" fill="#9A938A" font-size="9">the filter</text><text x="226" y="100" fill="#B8AEA2" font-size="16">=</text><rect x="250" y="70" width="100" height="50" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="300" y="98" text-anchor="middle" fill="#5E5191">Q (or K, or V)</text><text x="300" y="140" text-anchor="middle" fill="#6B645E" font-size="10">1 × 3</text><rect x="60" y="176" width="16" height="16" fill="#FDECEC" stroke="#C93B3B"/><rect x="158" y="176" width="16" height="16" fill="#FDECEC" stroke="#C93B3B"/><text x="200" y="188" fill="#C93B3B" font-size="10">these two "2"s must match — that's the rule of the multiply</text><text x="435" y="88" text-anchor="middle" fill="#276b45" font-size="10">do this 3 times,</text><text x="435" y="102" text-anchor="middle" fill="#276b45" font-size="10">once per grid</text><text x="435" y="116" text-anchor="middle" fill="#9A938A" font-size="9">→ Q, K, V</text></g></svg>
%%%

**What this build-up shows:** the word's card (`1 × 2`) times a filter (`2 × 3`) gives a `1 × 3` role-vector — the inner numbers must match, and the output length is whatever the filter's far side says. Run this three times, once per grid, and one card becomes Query, Key, and Value.

!!! c-info 🪜
<b>Optional (skippable) — the exact numbers, for the curious.</b> You don't need this to get the idea. Take a word with embedding `x = [1, 0]`. With filter `W_Q = [[0.9, 0.1], [0.3, 0.7]]`, the Query is `x·W_Q = [1·0.9 + 0·0.3,  1·0.1 + 0·0.7] = [0.9, 0.1]` — you multiply each column of the grid by the card and add. With a *different* grid `W_K = [[0.2, 0.8], [0.6, 0.4]]`, the Key is `[0.2, 0.8]`. Same input `x = [1, 0]`, different grids, different outputs — which is exactly the point: three roles, three filters, three answers. (These are the very numbers the picture above shows.)
!!!

Let's watch one word actually become three. **Predict first:** the word starts as `[1, 0]`. After three *different* filters, will its Query, Key, and Value come out the same or different? Guess, then reveal.

%%% demo id=proj label="turn one word into Q, K, V"
code: x = [1, 0];  Q = x @ Wq;  K = x @ Wk;  V = x @ Wv
out: Q -> [0.9, 0.1]    K -> [0.2, 0.8]    V -> [0.5, 0.5]
take: <b>One input word `[1, 0]` became three different lists.</b> The three filters pulled the same card in three directions — a question, a label, and some content — so the outputs genuinely differ. If all three filters were the same grid, Q, K, and V would come out identical and the whole trick would collapse. Three <i>different</i> learned filters is what gives a word its three separate hats.
%%%

@@@ concept id=c3 tag="Scoring a match" title="How well does my question match your label?" gotit="Got the score"
Now the fun begins. Each word has its Query (its question) and every word has a Key (its label). The next job is the heart of the library trick: for one word, go down the shelf and score *how well my question matches each label*.

Think of standing in the library aisle holding a slip that says "**volcano, lava, eruption**." You glance at each spine label and, almost instantly, feel how strong the match is. A book labeled "**volcanoes, lava, magma**" lights up — lots of overlap, strong match. A book labeled "**cake recipes**" gets nothing — no overlap, weak match. You're not reading whole books yet; you're just giving each one a quick *match score*. A word does the same: it takes its Query and, for every other word, measures how much its Query lines up with that word's Key. Big number = strong match = "pay attention to this one." The name for this line-up score is the [[dot product||A single number that says how much two lists point the same way. Multiply them slot by slot, add it all up. Big = strong match.]] — and it's just "multiply the two lists slot by slot, then add it all up."

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A query slip reading volcano lava eruption is compared against two spine labels. The first label volcanoes lava magma has heavy overlap and gets a high match score. The second label cake recipes has no overlap and gets a low match score. Arrows and glowing highlight show the strong match."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Score each label against my question</text><rect x="180" y="30" width="160" height="40" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="260" y="48" text-anchor="middle" fill="#5E5191" font-weight="bold">🔎 my Query</text><text x="260" y="63" text-anchor="middle" fill="#6B645E" font-size="9">"volcano, lava, eruption"</text><line x1="230" y1="70" x2="140" y2="108" stroke="#2D8B55" stroke-width="2"/><line x1="300" y1="70" x2="392" y2="108" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3,2"/><rect x="30" y="112" width="212" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="46" y="132" fill="#276b45" font-size="16">🏷️</text><text x="70" y="132" fill="#276b45" font-size="10">"volcanoes, lava, magma"</text><text x="136" y="152" text-anchor="middle" fill="#276b45" font-weight="bold">strong match → high score</text><rect x="280" y="112" width="212" height="56" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="296" y="132" fill="#9A938A" font-size="16">🏷️</text><text x="320" y="132" fill="#6B645E" font-size="10">"cake recipes"</text><text x="386" y="152" text-anchor="middle" fill="#9A938A" font-weight="bold">weak match → low score</text><text x="260" y="192" text-anchor="middle" fill="#6B645E" font-size="10">a bigger score just means "your question and my label point the same way"</text></g></svg>
%%%

**What the slip-and-labels picture gets right:** you compare your question to a label and get a single "how much do these overlap?" number, fast, for every book. That single overlap number is exactly the dot product. **Where it breaks down:** real overlap counts shared *words*, but the dot product works on the *number-lists* Q and K, so it can spot matches even when no literal word is shared — it feels the match in meaning, not spelling.

#### The one-line rule (in plain words)
For a word's Query and another word's Key: multiply them slot by slot, add up the results — one number falls out. That number is the **attention score** between the two words. Do it for every pair and you get a grid of scores: who matches whom, and how strongly.

%%% formula
expr: score(word i → word j) = Q_i · K_j
note: Q_i = word i's question, K_j = word j's label. The dot means multiply slot-by-slot and add. Big score = i and j are a strong match; i should pay attention to j.
%%%

Let's *watch* one word score the whole room. **Predict first:** in "the animal crossed the river bank", the word "bank" is asking "what kind of place am I?" Which word do you think it'll match most strongly — "river", "the", or "crossed"? Guess, then reveal.

%%% demo id=score label="score bank against every word"
code: scores = [ Q_bank · K_word  for word in sentence ]
out: the=0.1  animal=0.5  crossed=0.4  the=0.1  river=3.0  bank=0.9
take: <b>"river" scores far higher (3.0) than every other word.</b> "bank" 's question — "what kind of place am I?" — lines up strongest with "river" 's label. The filler words ("the") barely register (0.1). Just like your eyes jumping to the volcano book, "bank" 's attention is being pulled toward "river" — which is exactly how it'll learn it means the river-kind of bank, not the money kind. No word was read yet; these are just the match scores.
%%%

You just turned "how relevant are these two words?" into a single number, for every pair. But raw scores like 3.0 and 0.1 are awkward to *use* — next we turn them into a clean recipe.

@@@ concept id=c4 tag="Sharing attention" title="Turning scores into a 'how much to listen' recipe" gotit="Got softmax weights"
We have raw scores now: river 3.0, bank 0.9, animal 0.5, the 0.1, and so on. But what do you *do* with a 3.0? It's not a percentage, it's not "listen this much." We need to turn the pile of raw scores into a clean recipe that says, in plain shares, how much of my attention goes to each word.

Think of having exactly **one pizza** to share at a table, and you must give *everyone* a slice — no slice can be negative (you can't take pizza away), and all the slices together must make exactly one whole pizza. The word with the strongest match gets the biggest slice; the fillers get slivers; but everyone gets *something*, and it all adds up to one. That's the recipe we want: turn the raw match-scores into slices of a single pizza of attention. The tool that does this has a name — [[softmax||A recipe that turns any list of scores into shares that are all positive and add up to 1 — like slicing one pizza. Bigger score gets a bigger slice.]] — and its whole job is "make these scores into positive shares that sum to 1."

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="One pizza divided into slices, one slice per word. The word river gets a large slice labelled 0.72, animal and crossed get medium slices, and the two the words get tiny slivers. A note says every slice is positive and all slices add up to one whole pizza."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One pizza of attention → a slice for every word</text><circle cx="130" cy="110" r="66" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M130 110 L196 110 A66 66 0 0 0 92 49 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><line x1="130" y1="110" x2="196" y2="110" stroke="#C99A12"/><line x1="130" y1="110" x2="92" y2="49" stroke="#C99A12"/><line x1="130" y1="110" x2="70" y2="142" stroke="#C99A12"/><line x1="130" y1="110" x2="150" y2="174" stroke="#C99A12"/><text x="166" y="86" fill="#276b45" font-weight="bold" font-size="10">river</text><text x="166" y="98" fill="#276b45" font-size="9">0.72</text><text x="300" y="70" fill="#6B645E">river</text><rect x="350" y="58" width="140" height="14" fill="#2D8B55"/><text x="300" y="98" fill="#6B645E">animal</text><rect x="350" y="86" width="26" height="14" fill="#C99A12"/><text x="300" y="126" fill="#6B645E">crossed</text><rect x="350" y="114" width="22" height="14" fill="#C99A12"/><text x="300" y="154" fill="#6B645E">the ×2</text><rect x="350" y="142" width="10" height="14" fill="#B8AEA2"/><text x="410" y="182" text-anchor="middle" fill="#276b45" font-size="10">every slice ≥ 0, and all slices add to 1.00</text></g></svg>
%%%

**What the pizza picture gets right:** you split one whole thing into shares, no share is negative, and they add to exactly one — that's precisely what softmax does to the scores. **Where it breaks down:** pizza slices are cut by hand in any size you like, but softmax's slices are decided by a fixed rule — bigger scores get *exponentially* bigger slices, so a clear winner grabs a lot and near-ties split fairly evenly. You don't choose the slices; the scores do.

#### What softmax guarantees (two promises)
- **Every share is positive.** No word gets negative attention — you can only listen *to* words, never anti-listen.
- **The shares add up to 1.** The whole attention budget is exactly one pizza, split among the words. So a share of `0.72` means "72% of my attention on this word."

These two promises — you met softmax back in Module 2 as the thing that turns scores into a probability-like list — are exactly why we run the scores through it. It makes the numbers *usable* as "how much to listen."

Now let's *watch* raw scores become slices. Below is a live playground: drag the scores and see the slices resize (and always add to 1). Try making two scores nearly equal, then pull one far ahead.

%%% svg
<svg id="sm-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive softmax. Drag three raw scores and watch the attention shares, drawn as bars, resize. They always stay positive and add up to exactly one whole."><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="40" y="26" width="440" height="140" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="150" x2="460" y2="150" stroke="#EFE9DF"/><rect id="sm-b0" x="110" y="78" width="60" height="72" fill="#2D8B55"/><text id="sm-t0" x="140" y="70" fill="#1a5c38">.66</text><text x="140" y="164" fill="#6B645E">river</text><rect id="sm-b1" x="230" y="123" width="60" height="27" fill="#2A7B9B"/><text id="sm-t1" x="260" y="115" fill="#1F6280">.24</text><text x="260" y="164" fill="#6B645E">money</text><rect id="sm-b2" x="350" y="139" width="60" height="11" fill="#C99A12"/><text id="sm-t2" x="380" y="131" fill="#9A7208">.10</text><text x="380" y="164" fill="#6B645E">the</text><text x="260" y="184" fill="#1a5c38" font-size="10">the three shares always add up to one whole pizza (1.00)</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>river <input id="sm-s0" type="range" min="-2" max="5" step="0.1" value="2" style="accent-color:#2D8B55;vertical-align:middle"></label>
<label>money <input id="sm-s1" type="range" min="-2" max="5" step="0.1" value="1" style="accent-color:#2A7B9B;vertical-align:middle"></label>
<label>the <input id="sm-s2" type="range" min="-2" max="5" step="0.1" value="0.1" style="accent-color:#C99A12;vertical-align:middle"></label>
</div>
<div id="sm-out" style="margin-top:6px;color:#2C2A28">scores [2.0, 1.0, 0.1] → shares <b>0.66 / 0.24 / 0.10</b> (sum 1.00) — when one word pulls ahead, it grabs most of the pizza.</div>
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

**What to notice in the playground:** when scores are close, the pizza splits fairly evenly; when one score pulls far ahead, it grabs almost the whole pie. That "winner takes most" behavior is softmax turning a match into a decision.

%%% demo id=softmax label="scores → attention shares"
code: weights = softmax([river=3.0, bank=0.9, animal=0.5, crossed=0.4, the=0.1, the=0.1])
out: river=0.72  bank=0.09  animal=0.06  crossed=0.05  the=0.04  the=0.04   (sum = 1.00)
take: <b>The raw scores became clean shares that add to exactly 1.00.</b> "river" 's big 3.0 turned into a fat 0.72 slice — 72% of "bank" 's attention lands on "river". Everyone still gets a sliver, but the winner is clear. These shares are the <b>attention weights</b>: the recipe for how much of each word to mix in next.
%%%

@@@ concept id=c5 tag="Blending the room" title="Mix the values by the recipe → a fresh word" gotit="Got the weighted sum"
We have the recipe now — the shares that say how much to listen to each word. The last step is the payoff: actually *use* the recipe to build a brand-new, context-aware version of the word. And here's the beautiful part — we mix the **Values**, not the words themselves.

Think of making a [[smoothie||A drink blended from several fruits in different amounts, following a recipe.]] from a recipe. The recipe says "72% banana, 9% strawberry, a splash of everything else." You don't dump in whole fruit — you pour each fruit in *by its share* and blend. Out comes one new drink that tastes mostly of banana, a little of strawberry, a hint of the rest. Attention does exactly this. The **shares** (from softmax) are the recipe. The **Values** are the fruits — each word's content. Pour in each word's Value scaled by its share, add them all up, and out comes one fresh vector: a new version of "bank" that is mostly "river," a little of itself, a hint of the rest. That new vector *knows it means river-bank.*

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A smoothie blender. Three value vectors are poured in by their shares: river value at 0.72 is the big pour, bank value at 0.09 a small pour, others a splash. They blend into one new output vector at the bottom labelled the new context-aware bank."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Pour each Value in by its share → blend → one new word</text><rect x="30" y="40" width="130" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="42" y="56" fill="#276b45" font-size="10">V(river) × 0.72</text><rect x="30" y="70" width="130" height="24" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="42" y="86" fill="#8A6D3B" font-size="10">V(bank) × 0.09</text><rect x="30" y="100" width="130" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="42" y="116" fill="#6B645E" font-size="10">V(animal) × 0.06</text><rect x="30" y="130" width="130" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="42" y="146" fill="#9A938A" font-size="10">…the rest × splashes</text><line x1="160" y1="52" x2="250" y2="96" stroke="#2D8B55" stroke-width="3"/><line x1="160" y1="82" x2="250" y2="100" stroke="#C99A12" stroke-width="1.5"/><line x1="160" y1="112" x2="250" y2="104" stroke="#B8AEA2"/><line x1="160" y1="142" x2="250" y2="108" stroke="#B8AEA2"/><path d="M256 78 L316 78 L306 150 Q286 166 266 150 Z" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="286" y="112" text-anchor="middle" fill="#5E5191" font-size="9">blend</text><text x="360" y="90" fill="#B8AEA2" font-size="16">→</text><rect x="386" y="80" width="120" height="44" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="446" y="100" text-anchor="middle" fill="#5E5191" font-weight="bold" font-size="10">new "bank"</text><text x="446" y="116" text-anchor="middle" fill="#6B645E" font-size="9">now river-flavored</text><text x="260" y="196" text-anchor="middle" fill="#276b45" font-size="10">output = 0.72·V(river) + 0.09·V(bank) + … — a weighted sum of Values</text></g></svg>
%%%

**What the smoothie picture gets right:** you combine several ingredients *in proportion* to a recipe and get one new thing that tastes mostly of the biggest ingredient — that's a weighted sum, exactly what attention outputs. **Where it breaks down:** a smoothie loses the separate fruits forever, but the model keeps every word's original Value around too, and does this blending fresh at every layer — so the mixing happens over and over, not just once.

#### The one-line rule (in plain words)
The word's new vector is: each word's **Value** multiplied by its **share**, all added up. Bigger share → more of that word's Value in the mix. That's a **weighted sum** — "weighted" just means "each thing counts by its share."

%%% formula
expr: output = Σ (share_j × V_j)
note: share_j = how much to listen to word j (from softmax, all add to 1). V_j = word j's Value (its content). Multiply each Value by its share, add them up → one new vector. The word "Σ" just means "add up over all words j."
%%%

Let's *watch* the blend produce a new word. **Predict first:** "bank" 's shares are 72% river, 9% itself, small bits of the rest. Will the new "bank" vector end up looking more like river's Value or more like the old "bank" Value? Guess, then reveal.

%%% demo id=blend label="blend the values by their shares"
code: output = 0.72*V(river) + 0.09*V(bank) + 0.06*V(animal) + ...
out: V(river)=[0.9,0.1]  V(bank)=[0.4,0.5] ...  →  output = [0.72, 0.16]
take: <b>The new "bank" vector `[0.72, 0.16]` sits much closer to river's Value `[0.9, 0.1]` than to the old bank Value `[0.4, 0.5]`.</b> Because 72% of the recipe was river, the blend came out river-flavored. That's the magic moment: "bank" walked in confused and walked out as a <i>context-aware</i> vector that leans "river." This new vector is what the next layer of the model reads — a word that finally knows which "bank" it is.
%%%

Take a victory lap: you just built the entire attention move. Query asks, Key answers, the dot product scores, softmax shares, and a weighted sum of Values delivers a fresh word. Everything else in a transformer stacks on top of this.

@@@ concept id=c6 tag="The scaling fix" title="A puzzle: when the scores get too loud" gotit="Got the scaling fix"
Here's a puzzle that trips up real engineers, and the fix is one of the neatest tricks in the whole transformer. It only shows up when the word-cards get *long* — and modern models use very long cards (hundreds of numbers each). So this matters for every real model.

Think of a **volume knob** on a speaker. At a sensible volume you hear the whole song — the drums, the bass, the singer. Crank it all the way up and the sound *clips*: it turns harsh and one loud part drowns out everything else. Attention has the same danger. Remember the dot-product score is "multiply slot by slot and add up." If each word-card has hundreds of slots, that sum is adding up hundreds of pieces — so the scores get *huge*. Feed huge scores into softmax and the pizza gets sliced brutally: one word grabs almost everything and the rest get crumbs. The attention has been turned up too loud — it can only hear one voice.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A before and after of softmax shares. On the left, labelled scores too loud, the shares are extreme: one bar near 1.0 and the rest almost 0. On the right, labelled after dividing by root d k, the shares are balanced: one clear winner but the others still visible. An arrow labelled divide by root d k connects them."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Too-loud scores vs balanced scores</text><text x="120" y="42" text-anchor="middle" fill="#C93B3B" font-size="10">scores too loud → softmax clips</text><rect x="40" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="40" y="62" width="30" height="98" fill="#C93B3B"/><rect x="80" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="80" y="157" width="30" height="3" fill="#C93B3B"/><rect x="120" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="120" y="158" width="30" height="2" fill="#C93B3B"/><rect x="160" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="160" y="159" width="30" height="1" fill="#C93B3B"/><text x="115" y="176" text-anchor="middle" fill="#9A938A" font-size="9">one word hogs it all</text><g><line x1="210" y1="110" x2="300" y2="110" stroke="#5E5191" stroke-width="2"/><polygon points="300,105 312,110 300,115" fill="#5E5191"/><text x="256" y="100" text-anchor="middle" fill="#5E5191" font-size="9">÷ √(card length)</text></g><text x="400" y="42" text-anchor="middle" fill="#276b45" font-size="10">after scaling → balanced</text><rect x="330" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="330" y="88" width="30" height="72" fill="#2D8B55"/><rect x="370" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="370" y="120" width="30" height="40" fill="#2D8B55"/><rect x="410" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="410" y="134" width="30" height="26" fill="#2D8B55"/><rect x="450" y="60" width="30" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="450" y="140" width="30" height="20" fill="#2D8B55"/><text x="405" y="176" text-anchor="middle" fill="#276b45" font-size="9">clear winner, others still heard</text></g></svg>
%%%

%%% hint
t1: Don't worry about the formula yet. Just ask one thing: if a score is built by adding up MORE little pieces, does the total tend to get bigger or smaller?
t2: A dot product adds one piece per slot. A short card of 4 slots might add up to about 4. A long card of 100 slots adds 100 pieces, so it lands near 100 — much bigger, just because there were more pieces. Dividing by the square root of the card length shrinks that big number back down to a calm size.
t3: Long cards make big scores, big scores make softmax pick one winner and ignore everyone else, so we divide every score by √(card length) first to keep the scores calm.
%%%

**What the volume-knob picture gets right:** too much gain wrecks the mix — one part drowns out the rest, and you lose the detail. That's exactly what oversized scores do to softmax. **Where it breaks down:** a volume knob makes *everything* louder equally, but here the trouble is only that the scores grew with the card length — so the fix isn't "turn down everything," it's "divide by an amount that exactly cancels how long the card is."

#### The named fix: divide by √(card length)
The remedy is delightfully simple and it has a name: [[scaled dot-product attention||The standard attention: divide every score by the square root of the card length (√d_k) before softmax, so long cards don't blow the scores up.]]. Before you run softmax, divide every score by the **square root of the card length** (the length of each Key/Query is written `d_k`, so the divisor is `√d_k`). Longer cards make bigger scores *and* a bigger divisor, and the two cancel — so the scores stay a sensible size no matter how long the cards get. This one division is the "scaled" in the official name of attention, "scaled dot-product attention." It's not a hack; it's the standard.

%%% formula
expr: score = (Q · K) / √d_k
note: Q·K = the raw match score. d_k = the length of each Key (the card length). √d_k = its square root, the exact divisor that cancels the growth. Divide, THEN softmax.
%%%

Let's *watch* the fix save the day, with concrete numbers you can reproduce. Say the card length is `d_k = 9`, so the divisor is `√9 = 3`. The raw scores for three words came out `[6, 2, 0]` — a wide gap. **Predict first:** before scaling, softmax hands the winner almost everything. After dividing each score by 3, will the other two words get *any* share back? Guess, then reveal.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two bar charts side by side showing the same three words. On the left, no scaling: the first bar is nearly full at 0.98 and the other two bars are almost invisible at 0.02 and 0.00. An arrow labelled divide each score by root d k equals 3 points right. On the right, scaled: the first bar is 0.71, the second 0.19, the third 0.10 — a clear winner but all three visible."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">raw [6, 2, 0], d_k = 9 → divide by √9 = 3</text><text x="110" y="38" text-anchor="middle" fill="#C93B3B" font-size="10">no scaling → softmax clips</text><rect x="40" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="40" y="52" width="40" height="98" fill="#C93B3B"/><text x="60" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.98</text><rect x="95" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="95" y="148" width="40" height="2" fill="#C93B3B"/><text x="115" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.02</text><rect x="150" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="150" y="150" width="40" height="0.5" fill="#C93B3B"/><text x="170" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.00</text><g><line x1="210" y1="100" x2="298" y2="100" stroke="#5E5191" stroke-width="2"/><polygon points="298,95 310,100 298,105" fill="#5E5191"/><text x="254" y="90" text-anchor="middle" fill="#5E5191" font-size="9">÷ 3</text></g><text x="405" y="38" text-anchor="middle" fill="#276b45" font-size="10">scaled → balanced</text><rect x="330" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="330" y="79" width="40" height="71" fill="#2D8B55"/><text x="350" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.71</text><rect x="385" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="385" y="131" width="40" height="19" fill="#2D8B55"/><text x="405" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.19</text><rect x="440" y="50" width="40" height="100" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="440" y="140" width="40" height="10" fill="#2D8B55"/><text x="460" y="164" text-anchor="middle" fill="#6B645E" font-size="9">0.10</text></g></svg>
%%%

**What this build-up shows:** the *same* three words, before and after one division. Raw `[6, 2, 0]` through softmax clips to `[0.98, 0.02, 0.00]` — the winner hogs it. Divide each score by `√9 = 3` to get `[2.0, 0.67, 0.0]`, and softmax now gives `[0.71, 0.19, 0.10]` — the winner still leads, but the others come back to life. One division, and the room can be heard again.

%%% demo id=scale label="softmax before vs after scaling (d_k = 9)"
code: raw = [6, 2, 0];  d_k = 9;  softmax(raw)  vs  softmax(raw / sqrt(d_k))
out: NO SCALE -> [0.98, 0.02, 0.00]     scaled scores [2.0, 0.67, 0.0] -> SCALED -> [0.71, 0.19, 0.10]
take: <b>Without scaling, softmax nearly clips: the winner takes 0.98 and the others almost vanish.</b> Dividing each raw score by √d_k = √9 = 3 turns [6, 2, 0] into [2.0, 0.67, 0.0], and now softmax gives [0.71, 0.19, 0.10] — the winner still leads, but the others come back. And, quietly, this also keeps learning healthy, because those crushed near-zero shares would have stalled training. One little division, big payoff. That's why every real transformer scales.
%%%

!!! c-info 🪜
<b>Optional (skippable) — why √d_k exactly, for the curious.</b> You don't need this to use the fix. When you add up `d_k` roughly-independent little products, the typical size of the sum grows like `√d_k` (this is a standard fact about adding up random numbers). So the raw scores swell by about `√d_k` as cards get longer. Dividing by `√d_k` cancels that growth precisely, keeping the scores at a steady size whatever `d_k` is — which is why the paper picked that exact divisor and not, say, `d_k` itself.
!!!

@@@ concept id=c7 tag="The whole flow" title="Read the whole library visit end to end" gotit="Got the full pipeline"
You've built every piece. Now let's step back and watch the whole thing run start to finish — one clean flow — so the shape of attention lives in your head as a single picture. And there's a lovely twist hiding in it that we'll name at the end.

Think of an **assembly line** where a word rides a conveyor belt through a few stations, and comes out transformed. Station 1: the word gets stamped with its three role-cards (Query, Key, Value). Station 2: its Query is scored against every Key — the match scores. Station 3: the scores get scaled down to a sensible size. Station 4: softmax slices them into shares. Station 5: the shares blend the Values into one fresh word. In, along the belt, out — a confused "bank" enters and a river-aware "bank" exits. Same belt runs for *every* word at once.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="An assembly line of six stations connected by arrows. Station one embeddings, station two make Q K V, station three Q dot K scores, station four scale by root d k, station five softmax into shares, station six weighted sum of V giving the output. A word rides along the belt and comes out transformed."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The whole attention flow, station by station</text><rect x="12" y="52" width="66" height="44" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="45" y="72" text-anchor="middle" fill="#6B645E">embeddings</text><text x="45" y="86" text-anchor="middle" fill="#9A938A">x (Day 1)</text><text x="84" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="98" y="52" width="66" height="44" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="131" y="70" text-anchor="middle" fill="#5E5191">make</text><text x="131" y="84" text-anchor="middle" fill="#5E5191">Q, K, V</text><text x="170" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="184" y="52" width="66" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="217" y="70" text-anchor="middle" fill="#8A6D3B">Q · K</text><text x="217" y="84" text-anchor="middle" fill="#8A6D3B">scores</text><text x="256" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="270" y="52" width="66" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="303" y="70" text-anchor="middle" fill="#8A6D3B">÷ √d_k</text><text x="303" y="84" text-anchor="middle" fill="#8A6D3B">scale</text><text x="342" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="356" y="52" width="66" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="389" y="70" text-anchor="middle" fill="#276b45">softmax</text><text x="389" y="84" text-anchor="middle" fill="#276b45">→ shares</text><text x="428" y="78" fill="#B8AEA2" font-size="13">→</text><rect x="442" y="52" width="70" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="477" y="66" text-anchor="middle" fill="#5E5191">weighted</text><text x="477" y="78" text-anchor="middle" fill="#5E5191">sum of V</text><text x="477" y="90" text-anchor="middle" fill="#5E5191">= output</text><rect x="98" y="120" width="338" height="26" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text x="267" y="137" text-anchor="middle" fill="#6B645E" font-size="9">the belt runs for EVERY word at the same time — a whole sentence in one pass</text></g></svg>
%%%

**What the assembly-line picture gets right:** attention really is a fixed sequence of stations — same steps, same order, every time — and a word comes out changed at the end. **Where it breaks down:** a factory belt runs one item after another, but attention runs *all* words through *all* stations at the same time, in one big batch of number-crunching — which is exactly why it's so fast on modern hardware.

#### Read the pipeline out loud, in order
embeddings → make Q, K, V (three filters) → Q·K scores → scale by 1/√d_k → softmax into shares → weighted sum of V → the output word. Six stations. If you can say that list, you can explain attention.

Here's a live version of the whole pipeline — step through it and watch a real sentence flow from embeddings all the way to the output. Notice the same stations you just read.

%%% svg
<svg id="pl-svg" viewBox="0 0 520 128" role="img" aria-label="Interactive attention pipeline. Six stations in a row — embeddings, make Q K V, scores, scale, softmax, weighted sum of values — leading to the output word. Drag the slider to advance the active station and watch a sentence flow through in order."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" font-size="11" fill="#2C2A28">The attention pipeline — one sentence, six stations, in order</text><rect id="pl-r0" x="12" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="44" y="60" fill="#3A342E">embed</text><text x="44" y="73" font-size="7.5" fill="#6B645E">word→vec</text><text x="82" y="63" fill="#B8AEA2">›</text><rect id="pl-r1" x="92" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="124" y="60" fill="#3A342E">Q K V</text><text x="124" y="73" font-size="7.5" fill="#6B645E">3 filters</text><text x="162" y="63" fill="#B8AEA2">›</text><rect id="pl-r2" x="172" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="204" y="60" fill="#3A342E">scores</text><text x="204" y="73" font-size="7.5" fill="#6B645E">Q·K</text><text x="242" y="63" fill="#B8AEA2">›</text><rect id="pl-r3" x="252" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="284" y="60" fill="#3A342E">÷√dₖ</text><text x="284" y="73" font-size="7.5" fill="#6B645E">scale</text><text x="322" y="63" fill="#B8AEA2">›</text><rect id="pl-r4" x="332" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="364" y="60" fill="#3A342E">softmax</text><text x="364" y="73" font-size="7.5" fill="#6B645E">→shares</text><text x="402" y="63" fill="#B8AEA2">›</text><rect id="pl-r5" x="412" y="40" width="64" height="42" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2"/><text x="444" y="60" fill="#3A342E">Σ·V</text><text x="444" y="73" font-size="7.5" fill="#6B645E">blend</text><text x="490" y="63" fill="#2D8B55">→out</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>step through <input id="pl-n" type="range" min="0" max="5" step="1" value="0" style="width:56%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="pl-out" style="margin-top:6px;color:#2C2A28"><b>Station 1 · embed:</b> each word becomes a vector — its profile card.</div>
</div>
<script>(function(){
  var ns=document.getElementById('pl-n');if(!ns)return;
  var out=document.getElementById('pl-out');
  var desc=[
    ['embed','each word becomes a vector — its profile card.'],
    ['make Q, K, V','from that card, make three filters: a Query, a Key, a Value.'],
    ['scores','Q·K — each word scores how well its Query matches every Key.'],
    ['scale ÷√dₖ','divide the scores by √dₖ so big numbers don’t blow up softmax.'],
    ['softmax','turn the scores into shares that always add up to 1.'],
    ['weighted sum Σ·V','blend everyone’s Value by those shares → the word’s new, context-aware vector.']
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

#### The twist worth naming — self-attention
Here's the quiet twist. All day, the Queries, the Keys, and the Values came from the *same* sentence. "bank" 's Query looked at "river" 's Key — and "river" was sitting right there in the same sentence. When every word makes its Q, K, and V from the *same* sequence, and every word attends to every other word in that same sequence, the mechanism has a special name: [[self-attention||Attention where the Queries, Keys, and Values all come from the same sentence — every word looks at every other word in its own sequence.]]. "Self" because the sentence is looking at *itself*. That's the kind you built today, and it's the workhorse inside every layer of a transformer.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "Self-attention projects each token into a Query, Key, and Value, scores every Query against every Key with a scaled dot product, softmaxes those scores into weights that sum to 1, and returns each token as a weighted sum of the Values — all drawn from the same sequence, so every token is rewritten in the context of the whole sentence." You can already say every piece of that.
!!!

!!! c-info 🔗
<b>Coming attractions (names to file away, nothing to learn today):</b><br>• Running <i>several</i> of these attentions side by side, each looking for something different, is <b>multi-head attention</b> — next lesson.<br>• Attention today is <i>order-blind</i> (it never wrote down word order — remember Day 1's "dog bites man" puzzle); the fix is <b>positional encoding</b>, a couple of days out.<br>• Hiding future words during training is <b>masking</b>; queries from one sentence reading another's keys is <b>cross-attention</b>. All coming — today you built the one clean mechanism they all stand on.
!!!

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the entire beating heart of a transformer today. Here's a fresh way to hold it all at once, plus cheat-sheets to come back to any time.

Think of the whole day as **one library visit, written on a recipe card.** You walked in with a question (your **Query**), you scanned every book's spine-label (its **Key**) and felt how well each matched (the **dot product score**), you decided how much of your time to spend on each book so it all added to your one afternoon (**softmax shares**), and you walked out having blended what you read into one new understanding (a **weighted sum of Values**). And you kept the scores from getting too loud by turning the volume down for long cards (**scaling by √d_k**). Query, Key, Value; score, share, blend — that's the recipe.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A recipe card titled attention in six steps. Step one three roles Q K V. Step two score Q dot K. Step three scale by root d k. Step four softmax into shares. Step five blend Values by shares. Step six out comes a context-aware word. A footer notes self-attention means all three come from the same sentence."><g font-family="monospace" font-size="10"><rect x="24" y="30" width="472" height="118" rx="10" fill="#FDF9F3" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="50" text-anchor="middle" fill="#8A6D3B" font-size="12" font-weight="bold">🍳 Attention, in six steps</text><text x="44" y="74" fill="#5E5191">1 · three roles: Query (ask), Key (advertise), Value (share) — from one embedding</text><text x="44" y="92" fill="#C99A12">2 · score: Query · Key for every pair — bigger = stronger match</text><text x="44" y="110" fill="#C99A12">3 · scale: divide scores by √d_k so long cards don't get too loud</text><text x="44" y="128" fill="#2D8B55">4 · share: softmax → positive shares that add to 1 · 5 · blend: weighted sum of Values</text><text x="260" y="164" text-anchor="middle" fill="#6B645E" font-size="9">self-attention: Q, K, V all come from the SAME sentence — every word reads every other</text></g></svg>
%%%

**What this picture gets right:** the whole mechanism really is a short, fixed recipe you can recite. **Where it breaks down:** a recipe card is followed once, but a transformer runs this recipe dozens of times, stacked, each layer re-blending on top of the last — so the real thing is this card, repeated.

The whole visit, in order:
- **Step 1 · Three roles.** Each word makes a **Query** (what it's looking for), a **Key** (what it advertises), and a **Value** (what it shares) — three filters (**W_Q, W_K, W_V**) applied to its one embedding.
- **Step 2 · Score the match.** A word's Query · another word's Key = an **attention score** (a dot product). Bigger = stronger match.
- **Step 3 · Scale.** Divide the scores by **√d_k** so long word-cards don't blow the scores up and clip softmax. This is the "scaled" in **scaled dot-product attention**.
- **Step 4 · Share.** **Softmax** turns the scores into **attention weights** — all positive, all adding to 1 (one pizza of attention).
- **Step 5 · Blend.** The output is a **weighted sum of the Values** — each Value poured in by its share. Out comes a fresh, context-aware word.
- **The twist.** Because Q, K, V all come from the *same* sentence, this is **self-attention** — the sentence reading itself.

#### Cheat-sheet · the ideas at a glance
%%% table
:: idea :: in one line :: watch out
Query / Key / Value :: three roles of one word — ask / advertise / share :: they must come from three DIFFERENT filters
W_Q, W_K, W_V :: the three learned filter-grids that make Q, K, V :: they're learned in training, not typed in
Attention score :: Query · Key — how well two words match :: it's a dot product: multiply slot-by-slot, add
Softmax :: turns scores into positive shares that sum to 1 :: a clear winner grabs most of the share
Weighted sum of V :: blend the Values by their shares → new word :: mixes the Values, not the raw words
Scaling by √d_k :: shrink scores so long cards don't clip softmax :: divide, THEN softmax — the "scaled" in the name
Self-attention :: Q, K, V all from the same sentence :: it's order-blind until positional encoding (later)
%%%

#### Cheat-sheet · the words you met today
%%% jargon
attention | letting each word gather info from the other words, weighted by how relevant they are
Query (Q) | the "what am I looking for?" version of a word — the question it asks
Key (K) | the "what do I advertise?" version — the label other words check against
Value (V) | the "what will I share?" version — the content a word hands over if listened to
projection | turning one list of numbers into another by multiplying by a learned grid
W_Q, W_K, W_V | the three learned grids that make Q, K, and V
dot product | multiply two lists slot-by-slot and add — one number saying how well they match
attention score | the dot product of one word's Query with another word's Key
softmax | turns a list of scores into positive shares that add up to 1
attention weights | the shares softmax produces — how much to listen to each word
weighted sum | add things up, each counted by its share
scaled dot-product attention | the standard attention: divide scores by √d_k before softmax
d_k | the length of each Key / Query card; its square root is the scaling divisor
self-attention | attention where Q, K, V all come from the same sentence
%%%

That's the whole day. Next you'll zoom in on **Attention Scores & Softmax** — the exact machinery of Steps 2–4, where those match scores become a decision.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A word makes three vectors — Query, Key, Value. What is the Query, in one line? | a:2 | The word's spelling | A copy of the embedding | The "what am I looking for?" version — the question the word asks the room | The final output vector | fb: Query = the question a word asks. It gets compared against every other word's Key to find matches.
q: How is a word's attention score with another word computed? | a:1 | By counting shared letters | By the dot product of its Query with the other word's Key | By averaging their embeddings | By softmax alone | fb: Score = Query · Key (a dot product): multiply slot-by-slot and add. Bigger = stronger match.
q: What does softmax guarantee about the attention weights? | a:2 | They are all whole numbers | They are sorted from high to low | They are all positive AND add up to exactly 1 | They are all equal | fb: Softmax makes positive shares that sum to 1 — one pizza of attention split among the words.
q: Why do we divide the scores by √d_k before softmax ("scaled" dot-product attention)? | a:1 | To make the model train faster on GPUs | Long word-cards make the scores huge, which clips softmax to one winner; dividing by √d_k keeps scores a sensible size | To convert the scores into letters | To add word-order information | fb: Big d_k → big scores → softmax clips to almost 100% on one word. Dividing by √d_k cancels that growth so the model can hear more than one word.
%%%

@@@ produce id=produce tag="Produce" title="Predict which word 'bank' listens to, then run it" gotit="Done"
Here's the fun part — you get to watch today's whole mechanism prove itself in a few lines of output. **Before you write any code, predict:** in "the animal crossed the river bank", the word "bank" is asking "what kind of place am I?" When you score its Query against every Key, then softmax those scores into shares — which word will grab the biggest slice, and will the new blended "bank" end up leaning toward "river"? Jot your guesses down, then build tiny Q, K, V and **watch** it happen. That little "I was right / oh!" moment is what makes the day stick. Pick one path.

#### Option A · write it yourself
Create `sessions/m03-attention/day-02-qkv/experiment.py`. Make a tiny embedding matrix `x` (a few word-rows). Make three weight grids `Wq, Wk, Wv` and compute `Q = x@Wq`, `K = x@Wk`, `V = x@Wv`; print their shapes to confirm they're three different views. Take one word's query, score it against every key with a dot product, divide by `sqrt(d_k)`, run `softmax` to get shares (check they sum to 1), then compute the weighted sum of the Values. Print each step and **observe** which word wins the biggest share. Run with `python3 sessions/m03-attention/day-02-qkv/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 4 Day 2 artifact.

Create sessions/m03-attention/day-02-qkv/experiment.py that, with a comment on each step:
1. Defines a small embedding matrix x (e.g. 4 word-rows, each length 4) standing for a tiny sentence.
2. Defines three learned-style weight grids Wq, Wk, Wv (shape 4 x 4) and computes Q=x@Wq, K=x@Wk, V=x@Wv; prints each with its shape and notes they are three DIFFERENT views of the same words.
3. Takes word 0's query Q[0], scores it against every key with a dot product (scores = K @ Q[0]); prints the raw scores.
4. Divides the scores by sqrt(d_k) (the key length) — the scaled dot-product step — and prints the scaled scores, explaining in a comment why long cards need this.
5. Runs softmax on the scaled scores to get attention weights; prints them and asserts they sum to ~1.0 (positive shares of one pizza).
6. Computes the output as the weighted sum of the Values (weights @ V); prints it and notes it is a blend of the words word 0 chose to attend to.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- `Q`, `K`, `V` each print with the same shape and are *not* identical — three different filters gave one word three different vectors.
- The raw scores are one number per word (a dot product each). After dividing by `√d_k` they get smaller and calmer — that's the scaling step.
- The softmax weights are all positive and **sum to 1.00** — one pizza of attention. One word holds the biggest slice. Did you predict which?
- The final output is a weighted sum of the Values — a fresh vector that leans toward the word with the biggest share. That blend is a context-aware version of the word, and it's exactly what the next layer reads.
- The thing worth **noticing**: no word was ever compared by spelling. The whole decision — who listens to whom — came from Query·Key match scores turned into shares. That is attention.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-02-qkv/log.md`: (1) in your own words, what the Query, the Key, and the Value each do; (2) which word won the biggest attention share, and whether that surprised you; (3) one thing you're still curious about (why the scores need dividing by √d_k, or why Q and K must come from different filters, are both fair game).
!!!

@@@ fin
