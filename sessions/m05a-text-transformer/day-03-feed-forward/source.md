---
quest_id: wf6-d03-ffn
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 3 — The Feed-Forward Layer"
module_label: "Module 05a · Represent · Day 3"
title: "The Feed-Forward Layer"
subtitle: "Where Each Word Thinks on Its Own"
brand_sub: "Foundations · M5a Day 3"
spine: "workshop"
nav_prev_href: "../day-02-layer-norm/lesson.html"
nav_prev_label: "Layer Normalization"
nav_next_href: "../day-04-full-block/lesson.html"
nav_next_label: "The Full Block"
fin_title: "Module 5a · Day 3 complete! 🏆"
fin_body: "Nice work — you've unlocked the <b>Feed-Forward Layer</b>: the little per-token <b>workshop</b> that expands each word's vector into a wider space, bends it with an activation, then shrinks it back — giving every position room to think on its own between attention steps.<br>Next up: <b>The Full Block</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: position-wise feed-forward network applied independently per token after attention, keywords: [feed-forward, position-wise, each token, independently, after attention, its own workshop]}
  - {topic: two linear layer sandwich up-projection nonlinearity down-projection, keywords: [two linear layers, up-projection, down-projection, first linear layer, second linear layer, nonlinearity in the middle]}
  - {topic: expand then contract shape wider hidden dim 4x expansion, keywords: [expand then contract, wider, 4x, 512 to 2048, hidden layer is wider, expansion]}
  - {topic: why the FFN exists attention only averages linear mixing across positions, keywords: [attention only, weighted average, linear mixing, think on its own, per-token nonlinear, transform its own]}
  - {topic: middle nonlinearity relu original gelu modern, keywords: [relu, gelu, nonlinearity, the bend, more powerful than a single linear layer]}
  - {topic: FFN is a plain MLP applied token by token, keywords: [multi-layer perceptron, mlp, dense hidden layer, applied token by token, same idea]}
  - {topic: GELU modern default remedy for dying relu smooth for small negatives, keywords: [gelu, smooth, small negative, modern default, less likely to get stuck]}
  - {topic: dying dead relu failure outputs zero for negatives unit stuck, keywords: [dying relu, dead relu, outputs exactly 0, stuck, stop learning, negative pre-activation]}
  - {topic: residual skip connection wrapping the FFN sublayer, keywords: [residual, skip connection, adds the input back, flow straight through, gradients]}
  - {topic: layer normalization around the FFN sublayer keeps scale stable, keywords: [layer normalization, layer norm, stable range, around the sublayer, steady scale]}
  - {topic: training instability of deep stacked block remedy residual plus layer norm, keywords: [unstable, shrink or blow up, deep stack, many stacked sublayers, remedied by residual, layer norm]}
  - {topic: dropout inside or after the FFN randomly zeros hidden units, keywords: [dropout, randomly zeroing, hidden units, during training, reduce overfitting]}
  - {topic: overfitting parameter concentration FFN holds most parameters remedy dropout, keywords: [most of the parameters, bulk of the parameters, memorize, overfitting, two wide matrices, parameter concentration]}
  - {topic: capability limit position-wise cannot move information between token positions, keywords: [cannot move information between, in isolation, does not mix, needs both attention and, position-wise limit]}
---

@@@ hero
@lede Picture a busy classroom where every student has just finished **sharing notes** with their neighbours — everyone now knows a bit of what everyone else was thinking. But sharing isn't the same as *understanding*. So each student now goes back to their **own desk**, alone, to actually work through what they just heard: they spread their notes out wide, chew on them, and write down a tidier idea. That quiet solo-desk step is today's whole idea — the **feed-forward layer**, the little per-token *workshop* inside every transformer block. Yesterday you kept the numbers on an even keel with layer norm; the day before, attention let words *share* with each other. But attention only ever *mixes* what's already there. Today you meet the step where each word finally gets to *think on its own* — the same quiet workshop stacked inside every layer of ChatGPT. Sounds like just "two more layers"? Hold that thought — by the end you'll see why depth without this workshop buys a transformer almost nothing.
@goal Together we'll build this workshop one bench at a time — first the puzzle (why attention alone isn't enough and each word needs a solo step), then the shape of the workshop (a tiny network run on each word), why it *expands then shrinks*, the little *bend* in the middle that gives it power, a sneaky failure where benches go dark and the fix that saves them, the two safety rails that keep a deep stack of these workshops trainable, a trick that stops the workshop from just memorizing, and finally the honest limit — the one thing this workshop can never do. Every failure we meet is a small puzzle with a neat answer, and every formula gets said out loud in plain words first.

@@@ concept id=c1 tag="The puzzle" title="Sharing isn't thinking" gotit="Got the puzzle"
Let's pick up right where the last two days left off. You know a transformer is a tall stack of blocks. Inside each block, **attention** lets every word peek at the other words and pull in what matters. That's powerful — but there's a quiet gap in it, and spotting that gap is today's starting point.

Think of a group of friends **pooling their pocket money** to buy a gift. Everyone throws coins into one jar, and you work out the *average* each person chipped in. That's a fair, useful number. But notice what averaging can and can't do: it can only ever land you *somewhere between* the amounts people already put in. You can't average a pile of coins and suddenly get a *banknote* — averaging never invents a brand-new value, it only blends the ones you started with. Attention is exactly this kind of averaging: for each word it takes a weighted average of the other words' vectors. It's brilliant at *sharing*, but on its own it can only ever produce blends of what's already there.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Several coins of different sizes drop into a jar, and an arrow points to a single averaged coin whose size sits between the inputs. A note says averaging only lands between the inputs, never a new banknote."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Averaging coins → a value BETWEEN them, never something new</text><circle cx="60" cy="70" r="16" fill="#FCF3DC" stroke="#C99A12"/><text x="60" y="74" fill="#9A7A10" font-size="9">2</text><circle cx="60" cy="120" r="22" fill="#FCF3DC" stroke="#C99A12"/><text x="60" y="124" fill="#9A7A10" font-size="9">8</text><circle cx="60" cy="168" r="13" fill="#FCF3DC" stroke="#C99A12"/><text x="60" y="172" fill="#9A7A10" font-size="9">1</text><path d="M95 120 L175 120" stroke="#B8AEA2" stroke-width="1.5"/><text x="135" y="112" fill="#6B645E" font-size="9">average</text><rect x="185" y="92" width="90" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="230" cy="120" r="17" fill="#EAF5EE" stroke="#2D8B55"/><text x="230" y="124" fill="#276b45" font-size="9">≈3.7</text><text x="230" y="166" fill="#276b45" font-size="9">a blend</text><path d="M290 120 L330 120" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="4,3"/><rect x="340" y="98" width="150" height="44" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="415" y="118" fill="#C93B3B" font-size="9">a fresh banknote?</text><text x="415" y="134" fill="#C93B3B" font-size="9">✗ averaging can't</text></g></svg>
%%%

**What the coin-jar picture gets right:** averaging blends what's already in the jar and lands somewhere in the middle — that's exactly what attention does across words. **Where it breaks down:** coins are single numbers, but a word is a long list of numbers (its vector), and attention averages *all* of those at once, weighted differently for each word — still just a blend, but a much richer one than a single average.

#### Why a blend alone isn't enough
Here's why that gap matters. To *understand* a word — not just gather context, but reshape it into something new — the model needs a step that can *bend and transform* a word's numbers, not merely blend them with its neighbours'. Mathematicians have a name for "only ever blends, never bends": a [[linear||A straight-line relationship: the output is just a weighted sum of the inputs. Stacking linear steps still gives you one linear step — no new shapes.]] operation. Attention's averaging is linear. And you already learned the deep truth about linear steps: stack as many as you like, and they collapse into *one* linear step. So a tower made of nothing but attention, however tall, can only do linear things.

That's the puzzle. Attention shares information *across* words beautifully — but each word still needs a private moment to *transform* what it now holds. Below: attention hands each word a shared blend, but the "think on its own" bench is still empty. The step that fills it is today's whole lesson.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Top row: three word boxes connected by arrows showing attention sharing between them, labeled sharing. Below each word an empty desk with a question mark labeled think on its own, still empty."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Attention shares across words ✓ — but each word's solo bench is empty</text><rect x="60" y="34" width="90" height="34" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="105" y="55" fill="#5E5191" font-size="10">word 1</text><rect x="215" y="34" width="90" height="34" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="55" fill="#5E5191" font-size="10">word 2</text><rect x="370" y="34" width="90" height="34" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="415" y="55" fill="#5E5191" font-size="10">word 3</text><path d="M150 46 L215 46" stroke="#7C6DAA" stroke-width="1.5"/><path d="M305 60 L370 60" stroke="#7C6DAA" stroke-width="1.5"/><path d="M370 46 L305 46" stroke="#7C6DAA" stroke-width="1.5"/><path d="M215 60 L150 60" stroke="#7C6DAA" stroke-width="1.5"/><text x="260" y="86" fill="#276b45" font-size="9">↑ attention = sharing (linear blend)</text><rect x="60" y="112" width="90" height="50" rx="6" fill="#FBFBFA" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="105" y="140" fill="#C93B3B" font-size="15">?</text><rect x="215" y="112" width="90" height="50" rx="6" fill="#FBFBFA" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="260" y="140" fill="#C93B3B" font-size="15">?</text><rect x="370" y="112" width="90" height="50" rx="6" fill="#FBFBFA" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="415" y="140" fill="#C93B3B" font-size="15">?</text><text x="260" y="184" fill="#C93B3B" font-size="9">↓ each word's own workshop — still empty (that's today)</text></g></svg>
%%%

If that feels like a real gap — "wait, attention alone can't actually *think*?" — you're exactly where you should be. The fix is a small, familiar network you've met before, and it's waiting at the next bench.

@@@ concept id=c2 tag="The workshop" title="A tiny network run on each word" gotit="Got the workshop"
So what fills that empty bench? Something you already know well: a small neural network — the same kind you built back in the very first modules. The lovely twist is *how* the transformer uses it.

Think of an **airport security lane**. There isn't one giant machine that scans the whole crowd at once. Instead there's a single, identical scanner, and passengers step through it **one at a time** — same machine, same rules, applied to each person on their own. The person in front of you doesn't change how your bag gets scanned. The feed-forward layer is that scanner: it's one small network, and each word's vector steps through it **on its own**, separately from every other word. The transformer runs the *exact same* little network on every position — that's why we call it a [[position-wise||Applied to each token position separately and identically: the same little network runs on every word's vector, one at a time, with no word affecting another's pass.]] feed-forward network.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Three word vectors line up like passengers and each passes one at a time through one identical scanner box labeled the same small network, coming out transformed. A note says same machine, applied to each word on its own."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One scanner, each word steps through on its own (position-wise)</text><rect x="30" y="60" width="60" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="60" y="80" fill="#5E5191" font-size="9">word 1</text><rect x="30" y="100" width="60" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="60" y="120" fill="#5E5191" font-size="9">word 2</text><rect x="30" y="140" width="60" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="60" y="160" fill="#5E5191" font-size="9">word 3</text><path d="M95 115 L200 115" stroke="#B8AEA2" stroke-width="1.5"/><text x="150" y="108" fill="#6B645E" font-size="9">one at a time</text><rect x="205" y="70" width="110" height="90" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="260" y="108" fill="#9A7A10" font-size="10">the SAME</text><text x="260" y="124" fill="#9A7A10" font-size="10">small network</text><text x="260" y="176" fill="#6B645E" font-size="9">(the scanner)</text><path d="M320 115 L420 115" stroke="#B8AEA2" stroke-width="1.5"/><rect x="425" y="60" width="65" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="457" y="80" fill="#276b45" font-size="9">word 1′</text><rect x="425" y="100" width="65" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="457" y="120" fill="#276b45" font-size="9">word 2′</text><rect x="425" y="140" width="65" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="457" y="160" fill="#276b45" font-size="9">word 3′</text></g></svg>
%%%

**What the scanner picture gets right:** one identical machine, applied to each traveller on their own, no one affecting anyone else — that's position-wise exactly. **Where it breaks down:** an airport has *one* physical scanner and a real queue, but the transformer can run all the words through its network *at the same time* on the hardware — it only *acts* as if each word went alone, because the network never lets one word's numbers touch another's.

#### You've built this network before
Here's the reassuring part. That little network is nothing exotic — it's the plain [[multi-layer perceptron||An MLP: the classic neural network of a few "dense" layers, where every input number connects to every output number, with a bend in between. The first network you ever built.]] (MLP) you already met — a couple of "dense" layers with a bend between them. The feed-forward layer *is* an MLP; the only new idea is that the transformer applies it **token by token** instead of to one big input. So you're not learning a new machine today — you're learning where an old friend gets bolted into the block.

!!! c-info 💡
<b>Same network, new job.</b> When you first met the MLP, you fed it *one* input (say, the pixels of a digit). Here the transformer feeds it *each word's vector, one at a time*, and reuses the very same weights for all of them. Nothing about the MLP changed — only *what* gets pushed through it and *how many times*.
!!!

**Victory lap:** you now know the feed-forward layer isn't a mysterious new part — it's your familiar MLP, run once per word. Next we open it up and look at its unusual *shape*.

@@@ concept id=c3 tag="Expand then shrink" title="Spread it wide, then pack it back" gotit="Got the shape"
Now let's look inside the workshop. It has an odd, deliberate shape: it first makes each word's vector **much wider**, does its work in that roomy space, then squeezes it back to the original size. That "spread out, then pack back" shape is the heart of the feed-forward layer.

Think of **doing a jigsaw puzzle on a small coffee table**. If you tip the box out and try to solve it right there, the pieces are cramped — you can't spread them, can't see them all, can't sort edges from middles. So what do you do? You carry everything to the big **dining table**, spread all the pieces out where there's room to *see* and *arrange* them, solve the puzzle — and then, once it's done, you carry the finished picture back to the small coffee table where it belongs. The feed-forward layer does exactly this: it carries each word's numbers into a big, wide space where there's room to work, does the thinking there, then brings the result back to the original size.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A small coffee table with cramped puzzle pieces, an arrow to a big dining table where pieces are spread out and worked on, then an arrow back to the small table with the finished picture. Labeled narrow, wide workspace, narrow again."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Carry it to the big table to work, then bring it back</text><rect x="30" y="90" width="70" height="46" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="48" cy="105" r="4" fill="#7C6DAA"/><circle cx="62" cy="115" r="4" fill="#7C6DAA"/><circle cx="52" cy="126" r="4" fill="#7C6DAA"/><text x="65" y="158" fill="#5E5191" font-size="9">small table</text><text x="65" y="172" fill="#6B645E" font-size="9">cramped (narrow)</text><path d="M108 112 L165 112" stroke="#B8AEA2" stroke-width="1.5"/><text x="136" y="104" fill="#276b45" font-size="9">spread out</text><rect x="170" y="60" width="180" height="110" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><g fill="#C99A12"><circle cx="195" cy="85" r="4"/><circle cx="225" cy="80" r="4"/><circle cx="255" cy="92" r="4"/><circle cx="285" cy="82" r="4"/><circle cx="315" cy="90" r="4"/><circle cx="200" cy="120" r="4"/><circle cx="240" cy="128" r="4"/><circle cx="280" cy="118" r="4"/><circle cx="320" cy="126" r="4"/></g><text x="260" y="158" fill="#9A7A10" font-size="9">big table — room to work (WIDE)</text><path d="M358 112 L415 112" stroke="#B8AEA2" stroke-width="1.5"/><text x="386" y="104" fill="#276b45" font-size="9">pack back</text><rect x="420" y="90" width="70" height="46" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="455" y="117" fill="#276b45" font-size="14">🧩</text><text x="455" y="158" fill="#276b45" font-size="9">small table</text><text x="455" y="172" fill="#6B645E" font-size="9">finished (narrow)</text></g></svg>
%%%

**What the jigsaw picture gets right:** you need a *bigger workspace* to actually do the work, then you return the finished result to its normal size — that's expand-then-shrink exactly. **Where it breaks down:** on a table you move the *same* pieces to a bigger surface, but the feed-forward layer doesn't just relocate the numbers — the wide space is full of *brand-new* numbers computed from the originals, giving the model far more room to spot patterns than the narrow vector had.

#### The shape in three plain steps
Let's name the pieces gently. The width of a word's vector as it flows up the transformer has a name: the [[model dimension||The fixed width of every token's vector as it travels up the transformer — often written d_model, e.g. 512 numbers wide. Everything entering and leaving a block is this wide.]] (often written `d_model`). The feed-forward layer works in three steps:

1. **Up-projection** — the first [[linear layer||A dense layer: it multiplies the vector by a matrix of learned weights, turning a list of numbers into a new, differently-sized list. No bend on its own.]] widens the vector from `d_model` to a much bigger hidden width. This is where we spread the puzzle out.
2. **The bend** — a nonlinearity (you'll meet it next concept) works on the numbers in that wide space. This is the actual thinking.
3. **Down-projection** — a second linear layer packs the wide result back down to `d_model`, so the block's output is the same width it started. This is carrying the finished picture home.

How much wider is "wide"? The convention from the original transformer is a **4× expansion**: if `d_model` is `512`, the hidden width is `2048`. Big models keep this ratio — spread out to roughly four times the room, then pack back. Watch the width itself balloon and collapse in one picture:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A bar showing width. It starts at 512 narrow, then a tall wide bar at 2048 in the middle labeled 4x wider hidden space, then back down to 512. Arrows labeled up-projection and down-projection."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Width: 512 → 2048 (4× wider) → 512</text><line x1="40" y1="180" x2="490" y2="180" stroke="#B8AEA2"/><rect x="70" y="130" width="60" height="50" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="100" y="122" fill="#5E5191" font-size="10">512</text><text x="100" y="196" fill="#6B645E" font-size="9">d_model</text><path d="M138 100 L212 100" stroke="#C99A12" stroke-width="2"/><text x="175" y="92" fill="#9A7A10" font-size="9">up-project ↑</text><rect x="220" y="40" width="80" height="140" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="260" y="32" fill="#9A7A10" font-size="10">2048</text><text x="260" y="196" fill="#9A7A10" font-size="9">4× wider — room to think</text><path d="M308 100 L382 100" stroke="#2D8B55" stroke-width="2"/><text x="345" y="92" fill="#276b45" font-size="9">down-project ↓</text><rect x="390" y="130" width="60" height="50" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="420" y="122" fill="#276b45" font-size="10">512</text><text x="420" y="196" fill="#6B645E" font-size="9">back to d_model</text></g></svg>
%%%

Say the whole shape out loud: "widen each word, work in the wide space, then narrow it back." Let's watch the sizes on a real example — predict first: if a word arrives 4 numbers wide and we expand 4×, how wide is the hidden space? Then run it.

%%% demo id=shapedemo label="run it"
code: d_model = 4;  hidden = 4 * d_model;  x = [0.5, -1.0, 2.0, 0.0];  print("in:", len(x), "→ hidden:", hidden, "→ out:", d_model)
out: in: 4 → hidden: 16 → out: 4
take: <b>4 → 16 → 4.</b> The word enters 4 wide, gets spread to 16 (4× the room), then packs back to 4. Same width out as in — so the block's output slots right back onto the residual stream.
%%%

!!! c-info ⚙️
<b>Optional (skippable) — the two matrices.</b> Up-projection is a matrix `W1` of shape `[d_model, hidden]` and down-projection is a matrix `W2` of shape `[hidden, d_model]`. A word `x` (width `d_model`) becomes `x·W1` (width `hidden`), gets bent, then `·W2` brings it back to width `d_model`. You don't need this to understand the layer — the picture above says it all — but it's here if you like seeing the shapes line up.
!!!

**Victory lap:** you now know the feed-forward layer's signature move — *widen, work, narrow* — and the classic 4× ratio. Next: the little *bend* in the middle, without which all this widening would be pointless.

@@@ concept id=c4 tag="The bend" title="The little bend that gives it power" gotit="Got the bend"
Here's a fair question about that wide workshop: if it's just *widen with one matrix, then narrow with another matrix*, is the extra width really doing anything? The honest answer is **no — not by itself.** Two matrices back-to-back, with nothing between them, secretly collapse into a single matrix. The workshop needs one more thing: a tiny **bend** in the middle. That bend is what turns the whole thing from a fancy do-nothing into a real thinker.

Think of **folding a piece of paper**. Lay a flat sheet on the table — however you slide it or turn it, it stays a flat sheet; sliding a flat thing around never makes a shape. But *fold* it once, and suddenly you can make a paper boat, a plane, a crane. The single act of folding is what unlocks every shape. A straight-line (linear) step is like sliding the flat paper — no new shapes. The bend is the fold: it's the one move that lets the network build shapes it could never reach by sliding alone.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Left: a flat sheet of paper being slid around, still flat, labeled linear no new shapes. Right: the same paper folded into a peaked shape, labeled a bend unlocks new shapes."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Slide a flat sheet → still flat · fold it → new shapes</text><rect x="55" y="70" width="150" height="70" rx="2" fill="#EFEAF7" stroke="#7C6DAA"/><path d="M85 105 L175 105" stroke="#7C6DAA" stroke-width="1" stroke-dasharray="3,3"/><text x="130" y="60" fill="#5E5191" font-size="10">flat (linear)</text><text x="130" y="162" fill="#C93B3B" font-size="9">slide it → still flat, no new shape</text><text x="255" y="105" fill="#B8AEA2" font-size="14">vs</text><path d="M320 130 L385 80 L450 130 Z" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><path d="M385 80 L385 130" stroke="#C99A12" stroke-width="1" stroke-dasharray="3,3"/><text x="385" y="60" fill="#9A7A10" font-size="10">folded (a bend)</text><text x="385" y="162" fill="#276b45" font-size="9">a fold → a whole new shape ✓</text></g></svg>
%%%

**What the paper picture gets right:** sliding a flat sheet can never make a shape, but one fold unlocks endless shapes — that's precisely why a linear-only network is powerless and a bend sets it free. **Where it breaks down:** you fold paper once by hand into a fixed crease, but the network's bend is applied to *every number* and the network *learns* how to use it — so it's less one crease and more a bend it can exploit differently for every feature.

#### Meet the bend: ReLU
The simplest, most famous bend is [[ReLU||"Rectified Linear Unit": a bend that passes positive numbers straight through and turns every negative number into 0. Written ReLU(z) = max(0, z).]]. In plain words: **keep positive numbers as they are, and turn every negative number into 0.** It's like a one-way valve — positives flow through, negatives get shut to zero. The original 2017 transformer put a ReLU in the middle of its feed-forward layer.

%%% formula
expr: ReLU(z) = max(0, z)
note: if z is positive keep it; if z is negative, output 0. That kink at zero is the "bend".
%%%

Now watch *why* the bend matters, in one before→after picture. On the left: widen then narrow with **no** bend — the two matrices collapse, so the output is just a straight line, no better than a single layer. On the right: slip a ReLU in between, and the output can now *bend* — a shape the straight line could never make.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two small plots. Left labeled no bend: a single straight diagonal line, note says two matrices collapse to one line. Right labeled with ReLU bend: a line that is flat then kinks upward, note says a real bend, new shape."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">No bend → still a straight line · with ReLU → a real bend</text><text x="130" y="40" fill="#C93B3B" font-weight="bold">without a bend</text><line x1="55" y1="60" x2="55" y2="160" stroke="#B8AEA2"/><line x1="55" y1="160" x2="205" y2="160" stroke="#B8AEA2"/><line x1="55" y1="150" x2="205" y2="70" stroke="#C93B3B" stroke-width="2.5"/><text x="130" y="182" fill="#C93B3B" font-size="9">two matrices collapse → one line</text><text x="390" y="40" fill="#276b45" font-weight="bold">with a ReLU bend</text><line x1="315" y1="60" x2="315" y2="160" stroke="#B8AEA2"/><line x1="315" y1="160" x2="465" y2="160" stroke="#B8AEA2"/><polyline points="315,140 385,140 465,70" fill="none" stroke="#2D8B55" stroke-width="2.5"/><circle cx="385" cy="140" r="3" fill="#2D8B55"/><text x="390" y="182" fill="#276b45" font-size="9">flat, then kinks up → a new shape ✓</text></g></svg>
%%%

Let's make the collapse concrete. Predict first: if you multiply by 2, then multiply by 3, with nothing in between — is that different from just multiplying by 6? Then run it and watch the bend break the tie.

%%% demo id=benddemo label="run it"
code: x = 5;  no_bend = (x * 2) * 3;  one_matrix = x * 6;  with_bend = max(0, (x*2 - 20)) * 3;  print("no bend:", no_bend, " one matrix:", one_matrix, " with ReLU bend:", with_bend)
out: no bend: 30  one matrix: 30  with ReLU bend: 0
take: <b>Without a bend, ×2 then ×3 is just ×6 — the two steps collapse into one.</b> Add a ReLU (here it zeroed a negative), and the answer changes — the two steps can no longer be merged. The bend is what makes the second matrix earn its keep.
%%%

So the middle bend isn't a garnish — it's the whole reason the feed-forward layer has *power*. Without it, all that widening would fold flat into a single linear step.

**Victory lap:** you can now explain, in one line, why the feed-forward layer needs a nonlinearity: *two linear layers with nothing between them are just one linear layer; the bend is what makes depth pay off.* That's a classic interview beat, and it's yours.

@@@ concept id=c5 tag="Dark benches" title="When benches go dark — and the fix" gotit="Got GELU"
ReLU is simple and fast, but it hides a sneaky failure. Some of the workshop's benches can quietly **go dark** — stop doing any work at all, forever. Spotting why, and meeting the fix, is one of the most satisfying little puzzles in the whole transformer.

Think of a **light switch that can get stuck OFF**. A normal switch flips between on and off. But imagine one that, once it clicks to *off*, jams — no matter how you flick it, it stays dark. That room is now useless: no light, and no way to bring it back. ReLU has exactly this danger. Remember it turns every negative number into a flat 0. If one of the workshop's units keeps landing on the *negative* side, ReLU keeps outputting 0 — and here's the trap: when the output is a flat 0, there's no *slope* for the network to learn from, so it can never nudge that unit back to positive. The bench is stuck OFF, dark forever. Engineers call this a [[dying ReLU||A hidden unit whose ReLU output is stuck at 0 for every input, so it produces nothing and receives no learning signal — effectively a dead bench in the workshop.]] (or "dead ReLU").

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three light switches. Two are ON and glowing. One is stuck in the OFF position, dark, with a small lock icon showing it is jammed and cannot turn back on. Labeled a dead ReLU unit stuck outputting zero."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A bench stuck OFF — outputs 0, can't learn its way back</text><rect x="70" y="50" width="70" height="100" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="105" cy="80" r="14" fill="#FFF3C4" stroke="#C99A12"/><rect x="92" y="105" width="26" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="105" y="168" fill="#276b45" font-size="9">ON ✓</text><rect x="225" y="50" width="70" height="100" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="260" cy="80" r="14" fill="#FFF3C4" stroke="#C99A12"/><rect x="247" y="105" width="26" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="168" fill="#276b45" font-size="9">ON ✓</text><rect x="380" y="50" width="70" height="100" rx="8" fill="#F1F1F1" stroke="#C93B3B" stroke-width="2"/><circle cx="415" cy="80" r="14" fill="#E5E5E5" stroke="#9A938A"/><rect x="402" y="115" width="26" height="30" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="415" y="104" fill="#C93B3B" font-size="12">🔒</text><text x="415" y="168" fill="#C93B3B" font-size="9">stuck OFF (dead)</text></g></svg>
%%%

**What the light-switch picture gets right:** a switch jammed off gives no light and can't be flicked back — just like a dead unit that outputs 0 and gets no signal to recover. **Where it breaks down:** a real switch is stuck by a mechanical fault, but a dead ReLU is stuck by *maths* — the flat-zero region has no slope, so learning simply has nothing to grab onto there.

#### The fix: a gentler bend, GELU
The remedy is beautifully simple: use a bend that **doesn't go completely flat** on the negative side. The modern favourite is [[GELU||"Gaussian Error Linear Unit": a smooth bend used in GPT and BERT. Like ReLU for positives, but instead of a hard flat 0 it lets small negatives leak through gently, so units keep a bit of slope and rarely die.]]. GELU looks a lot like ReLU for positive numbers, but instead of slamming negatives to a hard flat 0, it lets small negatives *leak through gently* along a smooth curve. Because the curve is never perfectly flat, every unit keeps a little bit of slope — so there's always something for learning to grab, and benches almost never get stuck dark. Modern models like GPT and BERT use GELU as their default bend for exactly this reason.

Watch the two bends side by side in one before→after picture. ReLU (left) has a hard corner and a dead flat zero region; GELU (right) has a smooth dip that never fully flattens:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two curves. Left ReLU: flat along zero for negatives then a hard corner rising for positives, negative region marked flat, no slope, can die. Right GELU: a smooth curve that dips slightly below zero for small negatives then rises, marked smooth, always some slope."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">ReLU: hard flat zero (can die) · GELU: smooth, never fully flat</text><text x="130" y="40" fill="#C93B3B" font-weight="bold">ReLU</text><line x1="55" y1="60" x2="55" y2="165" stroke="#B8AEA2"/><line x1="45" y1="140" x2="205" y2="140" stroke="#B8AEA2"/><text x="40" y="144" fill="#6B645E" font-size="8">0</text><polyline points="60,140 128,140 195,72" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="92" y="156" fill="#C93B3B" font-size="8">flat 0 — no slope</text><text x="130" y="188" fill="#C93B3B" font-size="9">negatives → hard 0 → can die</text><text x="390" y="40" fill="#276b45" font-weight="bold">GELU</text><line x1="315" y1="60" x2="315" y2="165" stroke="#B8AEA2"/><line x1="305" y1="140" x2="465" y2="140" stroke="#B8AEA2"/><text x="300" y="144" fill="#6B645E" font-size="8">0</text><path d="M320 138 Q355 150 380 138 Q410 122 458 72" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="352" y="164" fill="#276b45" font-size="8">gentle dip — slope stays</text><text x="390" y="188" fill="#276b45" font-size="9">small negatives leak → rarely dies</text></g></svg>
%%%

!!! c-warn ⚠️
<b>Failure mode (silent):</b> a dead ReLU never crashes or throws an error — it just quietly outputs 0 forever, so a chunk of your workshop's capacity is wasted and you'd never know from the logs. Swapping to <b>GELU</b> (or its cousins) keeps a whisper of slope everywhere, so units stay alive. This is why "which activation?" is a real design choice, not a detail.
!!!

**Victory lap:** you now know a genuine production failure — the silently dying unit — *and* the named fix the whole field moved to. When someone asks "why GELU instead of ReLU in GPT?", you can answer: *GELU stays smooth for small negatives, so units keep a learning signal and don't die.*

@@@ concept id=c6 tag="Safety rails" title="Two rails that keep a deep stack trainable" gotit="Got the rails"
So far we have one workshop. But a real transformer stacks *dozens* of them, block after block. And stacking so many workshops on top of each other creates a fresh problem — one you already met the fixes for in the last two days. Let's see why the workshop needs two safety rails around it.

Think of a **whispered message passed down a long line of people** (the game "telephone"). Each person whispers to the next. By the time the message reaches the end of a long line, it has either faded into a mumble nobody can hear, or somebody shouted and it turned into noise. The more people in the line, the worse it gets. A deep stack of workshops is that long line: each one passes its signal to the next, and across dozens of them the numbers tend to **shrink toward nothing** (fade to a mumble) or **blow up too big** (turn to a shout). Either way, a very deep stack becomes hard to train.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A line of people passing a whisper. The speech bubble starts clear and full on the left and gets smaller and fainter each person, until it is a tiny faded mumble on the right. Labeled signal fades down a deep stack."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Whisper down a long line → fades to a mumble (or turns to a shout)</text><circle cx="70" cy="110" r="20" fill="#EAF5EE" stroke="#2D8B55"/><rect x="60" y="60" width="60" height="26" rx="10" fill="#EAF5EE" stroke="#2D8B55"/><text x="90" y="77" fill="#276b45" font-size="10">clear!</text><circle cx="190" cy="110" r="20" fill="#F1F8F3" stroke="#7FB394"/><rect x="185" y="66" width="46" height="20" rx="8" fill="#F1F8F3" stroke="#7FB394"/><text x="208" y="80" fill="#4f8666" font-size="8">ok</text><circle cx="310" cy="110" r="20" fill="#F7F9F7" stroke="#B8D0BF"/><rect x="308" y="72" width="30" height="14" rx="6" fill="#F7F9F7" stroke="#B8D0BF"/><text x="323" y="82" fill="#9ab0a1" font-size="7">..</text><circle cx="430" cy="110" r="20" fill="#FBFBFA" stroke="#D8D2C8"/><rect x="428" y="78" width="16" height="8" rx="4" fill="#FBFBFA" stroke="#D8D2C8"/><text x="70" y="150" fill="#276b45" font-size="9">block 1</text><text x="310" y="150" fill="#9A938A" font-size="9">deeper</text><text x="430" y="150" fill="#C93B3B" font-size="9">faded away</text><path d="M92 110 L168 110" stroke="#B8AEA2"/><path d="M212 110 L288 110" stroke="#B8AEA2"/><path d="M332 110 L408 110" stroke="#B8AEA2"/></g></svg>
%%%

**What the telephone picture gets right:** pass a signal through many hands and it degrades — fades or distorts — the longer the line, exactly like activations through a deep stack. **Where it breaks down:** in telephone the message *content* gets garbled, but here it's the *scale* of the numbers (too small or too big) that breaks training — the information could be fine if only the size stayed sensible.

#### The two rails you already know
You don't need a new idea to fix this — you built both fixes in the last two days:

1. **The residual (skip) connection.** Instead of forcing the signal to squeeze *through* the workshop, we let the original signal flow *straight past* on an express lane, and just **add** the workshop's change on top. So even if the workshop's part fades, the original signal is still there, full-strength. In one line: `output = x + FFN(x)`.
2. **Layer normalization.** Right around the workshop, layer norm re-levels the numbers to a steady scale, so nothing drifts too big or too small as blocks pile up.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A block diagram. The input flows up a straight express lane on the left labeled residual skip. It also branches through layer norm then the FFN workshop, and the workshop output is added back onto the express lane. Labeled two rails keep the deep stack steady."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two rails around the workshop: skip lane + layer norm</text><rect x="60" y="150" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="105" y="167" fill="#276b45">input x</text><path d="M105 150 L105 60" stroke="#2D8B55" stroke-width="3"/><text x="60" y="110" fill="#276b45" font-size="9" text-anchor="end">residual</text><text x="60" y="122" fill="#276b45" font-size="9" text-anchor="end">skip lane</text><path d="M150 163 L230 163" stroke="#B8AEA2"/><rect x="235" y="150" width="80" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="275" y="167" fill="#9A7A10">layer norm</text><path d="M315 163 L370 163" stroke="#B8AEA2"/><rect x="375" y="150" width="90" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="420" y="167" fill="#5E5191">FFN workshop</text><path d="M420 150 L420 60 L118 60" stroke="#7C6DAA" stroke-width="1.5"/><circle cx="105" cy="60" r="12" fill="#FFF" stroke="#2C2A28"/><text x="105" y="65" fill="#2C2A28" font-size="13">+</text><path d="M105 48 L105 30" stroke="#2D8B55" stroke-width="3"/><text x="105" y="24" fill="#276b45" font-size="9">output = x + FFN(norm(x))</text></g></svg>
%%%

Here's the payoff: **neither rail alone is enough — the pair is the magic.** The residual keeps a full-strength copy of the signal flowing straight up; layer norm keeps that signal's scale steady. Together they let you stack dozens of workshops without the signal fading to a mumble or blowing up to a shout. Watch the difference in one before→after picture — the same deep stack, wobbling without the rails, held steady with them:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A line chart across blocks. A dashed steady line marks the healthy scale. A red line without rails swings wildly and fades toward zero. A green line with rails stays flat and steady along the healthy scale."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Deep stack: without rails it wobbles/fades, with rails it holds</text><line x1="55" y1="40" x2="55" y2="160" stroke="#B8AEA2"/><line x1="55" y1="160" x2="490" y2="160" stroke="#B8AEA2"/><text x="42" y="46" fill="#6B645E" font-size="8">big</text><text x="42" y="164" fill="#6B645E" font-size="8">0</text><line x1="55" y1="100" x2="490" y2="100" stroke="#2D8B55" stroke-width="1.2" stroke-dasharray="5,4"/><text x="486" y="94" fill="#276b45" font-size="8" text-anchor="end">healthy scale</text><polyline points="60,100 130,55 200,140 270,150 340,156 410,158 480,159" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="300" y="140" fill="#C93B3B" font-size="9">without rails: swings, then fades ✗</text><polyline points="60,100 130,98 200,102 270,99 340,101 410,100 480,100" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="300" y="86" fill="#276b45" font-size="9">with rails: steady all the way ✓</text><text x="275" y="182" fill="#6B645E" font-size="9">block 1 → deeper → block N</text></g></svg>
%%%

**Victory lap:** you can now sketch a full transformer sublayer — norm, then the workshop, then a residual add — and explain *why* each rail is there. That's the exact shape you'll assemble into the full block next lesson.

@@@ concept id=c7 tag="Too much memory" title="Stop the workshop from just memorizing" gotit="Got dropout"
Here's a surprising fact that leads to the last fix. Those two wide matrices in the workshop are *big* — so big that the feed-forward layers hold **most of a transformer's weights**. That gives the workshop enormous room to learn... and also enormous room to simply **memorize** the training data instead of learning real patterns. Let's meet the trick that keeps it honest.

Think of **studying for a test**. Imagine a student with a photographic memory who just memorizes every answer in the practice book word-for-word. They ace the practice test — but on the *real* exam, with new questions, they're lost, because they memorized answers instead of understanding ideas. A model with a huge, powerful workshop can do the same: it can memorize the training examples and look brilliant on them, then flop on new text. Memorizing-instead-of-understanding has a name: [[overfitting||When a model learns the training examples so exactly (memorizes them) that it fails on new, unseen data — it fit the training set instead of the real pattern.]].

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Left: a bar chart showing the FFN slice is much larger than attention, labeled most parameters live in the FFN. Right: a student acing a memorized practice test but failing a new exam, showing overfitting."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The workshop holds most of the weights → room to just memorize</text><text x="130" y="40" fill="#2C2A28" font-size="10">where the parameters live</text><line x1="60" y1="160" x2="200" y2="160" stroke="#B8AEA2"/><rect x="80" y="120" width="34" height="40" fill="#EFEAF7" stroke="#7C6DAA"/><text x="97" y="176" fill="#5E5191" font-size="8">attention</text><rect x="140" y="66" width="34" height="94" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="157" y="176" fill="#9A7A10" font-size="8">FFN</text><text x="157" y="58" fill="#9A7A10" font-size="8">most!</text><rect x="300" y="60" width="80" height="46" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="340" y="80" fill="#276b45" font-size="9">practice</text><text x="340" y="94" fill="#276b45" font-size="9">100% ✓</text><rect x="410" y="60" width="80" height="46" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="450" y="80" fill="#C93B3B" font-size="9">real exam</text><text x="450" y="94" fill="#C93B3B" font-size="9">fails ✗</text><text x="395" y="140" fill="#C93B3B" font-size="9">memorized, didn't understand = overfitting</text></g></svg>
%%%

**What the studying picture gets right:** memorizing the practice book gives a perfect practice score but fails on new questions — exactly what an over-powerful workshop risks. **Where it breaks down:** a student memorizes on purpose, but a model overfits *by accident* — it's just following the training signal too literally, with no intent to cheat.

#### The fix: dropout
The clever remedy is [[dropout||A training trick that randomly switches off a fraction of the workshop's units on each step, so the model can't lean on any single unit and is forced to learn patterns that generalize. Switched off at test time.]]. On each training step, dropout **randomly switches off** a fraction of the workshop's hidden units — sets them to 0 for that step. Because any unit might vanish at any moment, the network can't lean on a few units to memorize specific examples; it's forced to spread its learning around and pick up patterns that actually **generalize** to new text. At test time dropout is turned off, so the full workshop is used. It's like making the study group randomly send a few members home each session — the rest can't rely on one know-it-all and everyone has to actually learn.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A row of workshop units. On each training step some units are randomly crossed out and greyed, showing dropout switching them off. Labeled randomly switch off some units each step so the model can't memorize."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Dropout: randomly switch off some units each training step</text><g><circle cx="80" cy="80" r="16" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="140" cy="80" r="16" fill="#F1F1F1" stroke="#C93B3B" stroke-dasharray="3,2"/><line x1="130" y1="70" x2="150" y2="90" stroke="#C93B3B"/><line x1="150" y1="70" x2="130" y2="90" stroke="#C93B3B"/><circle cx="200" cy="80" r="16" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="260" cy="80" r="16" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="320" cy="80" r="16" fill="#F1F1F1" stroke="#C93B3B" stroke-dasharray="3,2"/><line x1="310" y1="70" x2="330" y2="90" stroke="#C93B3B"/><line x1="330" y1="70" x2="310" y2="90" stroke="#C93B3B"/><circle cx="380" cy="80" r="16" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="440" cy="80" r="16" fill="#EFEAF7" stroke="#7C6DAA"/></g><text x="140" y="120" fill="#C93B3B" font-size="9">off this step</text><text x="320" y="120" fill="#C93B3B" font-size="9">off this step</text><text x="260" y="150" fill="#276b45" font-size="10">can't lean on any one unit → learns patterns that generalize</text><text x="260" y="170" fill="#6B645E" font-size="9">(switched off during training only; full workshop used at test time)</text></g></svg>
%%%

!!! c-info 💡
<b>Why this belongs in the FFN.</b> Because the workshop holds the bulk of the parameters, it's the main place a transformer can overfit — so dropout is commonly applied right inside or after the feed-forward layer. It's a cheap insurance policy: a little randomness during training, big gains in how well the model handles text it has never seen.
!!!

**Victory lap:** you now know *why* the feed-forward layer is where overfitting hides (it holds most of the weights) and the named fix that keeps it honest (dropout). Three failure modes met, three fixes in your pocket.

@@@ concept id=c8 tag="The honest limit" title="It thinks alone — it can't share" gotit="Got the limit"
Before the recap, one honest caveat — the kind a good engineer always keeps in mind. The workshop is powerful, but there's one thing it *fundamentally cannot do*, and knowing that limit is the sign you truly understand it.

Think of **soundproof study booths in a library**. Each booth is great for focused, solo work — you go in, spread out your books, and think hard, with no distractions. But the walls are soundproof, so you *cannot* pass a note or hear the person in the next booth. If you need to combine ideas with someone else, the booth simply can't do it — you'd have to leave and meet in the shared room. The feed-forward workshop is a soundproof booth: it's brilliant at letting one word think deeply *on its own*, but it has **no way to pass information between words**. Each word goes into its own booth and comes out transformed, but no word ever hears another.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three soundproof booths side by side, each with one word inside working alone. Thick walls between them are marked with a no-passing symbol, showing the FFN cannot move information between words."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Soundproof booths: each word thinks alone, no note passes between</text><rect x="50" y="50" width="120" height="110" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="110" y="105" fill="#5E5191" font-size="11">word 1</text><text x="110" y="124" fill="#6B645E" font-size="8">thinks alone</text><rect x="200" y="50" width="120" height="110" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="260" y="105" fill="#5E5191" font-size="11">word 2</text><text x="260" y="124" fill="#6B645E" font-size="8">thinks alone</text><rect x="350" y="50" width="120" height="110" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="410" y="105" fill="#5E5191" font-size="11">word 3</text><text x="410" y="124" fill="#6B645E" font-size="8">thinks alone</text><g stroke="#C93B3B" stroke-width="2"><line x1="185" y1="80" x2="185" y2="130"/><line x1="335" y1="80" x2="335" y2="130"/></g><text x="185" y="72" fill="#C93B3B" font-size="12">🚫</text><text x="335" y="72" fill="#C93B3B" font-size="12">🚫</text><text x="260" y="182" fill="#C93B3B" font-size="9">no information crosses between words — that's attention's job</text></g></svg>
%%%

**What the booth picture gets right:** each booth is perfect for solo focus but soundproof, so nothing passes between them — exactly the workshop's position-wise limit. **Where it breaks down:** you *could* walk out of a real booth to chat, but the feed-forward layer has literally no path between words at all — the only place words ever meet is attention.

#### Why the block needs BOTH parts
This is the deep reason a transformer block has *two* working parts, not one:

%%% cards
🔀 | Attention = share | Attention *mixes across words* — it lets each word gather what matters from the others. But it only blends (linear).
🧠 | FFN = think | The feed-forward workshop *transforms each word on its own* — it bends and reshapes, but never shares between words.
🤝 | Together = understand | Share, then think. Neither alone is enough: attention without the workshop can't reshape; the workshop without attention can't gather context.
%%%

That's the whole rhythm of a transformer block, and now you can see it: **first the words share (attention), then each word thinks alone (feed-forward)** — over and over, block after block. Remove either half and the block loses its power.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd explain it (say, in an interview):</b> "The feed-forward layer is a position-wise MLP: an up-projection to a wider hidden dim (classically 4×), a nonlinearity — ReLU originally, GELU in modern models — then a down-projection back to d_model, applied independently to every token. Attention only takes a linear weighted average across positions, so the FFN is where each token gets a per-token *nonlinear* transform. It's wrapped by a residual connection and layer norm so a deep stack stays trainable, and dropout guards it since it holds most of the model's parameters. Crucially it's position-wise — it never mixes information across tokens; that's attention's job — which is exactly why a block needs both." Notice you can already say every piece of that — that's how far you've come today.
!!!

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the whole workshop. Before the cheat-sheets, one picture to hold the whole day.

Think of a **potter's wheel**. Attention was the step where clay from all around the table got *gathered* onto the wheel — sharing. The feed-forward layer is the wheel itself: it takes that lump of clay (one word's numbers), *spreads it wide* with your hands, *shapes* it with a firm press (the bend), then brings it back to a neat form — and it does this to each lump *on its own*, one at a time. The whole transformer is a line of these wheels, each shaping every word a little more. **Where this breaks down:** a potter shares information *by hand* between lumps, but the wheel (the FFN) never mixes lumps — only the gathering step (attention) does that.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A potter's wheel. A lump of clay is placed on it, spread wide by hands, shaped with a press, then formed into a neat pot. Labeled gather, spread wide, shape with a bend, bring back — one lump at a time."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The workshop = a potter's wheel: spread wide, shape, bring back</text><ellipse cx="70" cy="120" rx="34" ry="10" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="70" cy="108" r="12" fill="#EFEAF7" stroke="#7C6DAA"/><text x="70" y="150" fill="#276b45" font-size="9">1 · a word (clay)</text><text x="120" y="112" fill="#B8AEA2" font-size="14">→</text><ellipse cx="190" cy="120" rx="34" ry="10" fill="#EAF5EE" stroke="#2D8B55"/><ellipse cx="190" cy="104" rx="28" ry="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="190" y="150" fill="#9A7A10" font-size="9">2 · spread wide</text><text x="240" y="112" fill="#B8AEA2" font-size="14">→</text><ellipse cx="310" cy="120" rx="34" ry="10" fill="#EAF5EE" stroke="#2D8B55"/><path d="M290 112 L310 90 L330 112 Z" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="310" y="150" fill="#9A7A10" font-size="9">3 · shape (the bend)</text><text x="360" y="112" fill="#B8AEA2" font-size="14">→</text><ellipse cx="430" cy="120" rx="34" ry="10" fill="#EAF5EE" stroke="#2D8B55"/><path d="M414 118 Q430 88 446 118" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="430" y="150" fill="#276b45" font-size="9">4 · bring back (a pot)</text></g></svg>
%%%

**The day as signposts, in order** — each one only makes sense because of the one before it:
- **1 · The puzzle.** Attention only takes a linear *weighted average* across words — it *shares*, but can't *transform*. Each word needs its own thinking step.
- **2 · The workshop.** That step is a small MLP — a *position-wise* feed-forward network — run on each word's vector on its own, reusing the same weights for every word.
- **3 · Expand then shrink.** It *up-projects* each word to a wider hidden space (classically **4×**), works there, then *down-projects* back to `d_model`.
- **4 · The bend.** A nonlinearity in the middle (**ReLU** originally, **GELU** in modern models) is what gives it power — without it, two linear layers collapse into one.
- **5 · Dark benches.** ReLU units can *die* (stuck at flat 0, no slope to learn from); **GELU** stays smooth for small negatives, so units rarely die.
- **6 · Two rails.** A **residual** skip lane plus **layer norm** wrap the workshop so a deep stack doesn't fade or blow up.
- **7 · Too much memory.** The workshop holds *most* of the parameters, so it can *overfit* (memorize); **dropout** randomly switches off units to force it to generalize.
- **The limit.** It's *position-wise* — it never mixes information across words. Sharing is attention's job; a block needs **both**.

#### Cheat-sheet · the feed-forward layer at a glance
%%% table
:: Piece :: In plain words :: Why it matters
up-projection :: first linear layer, widens `d_model` → hidden :: gives each word room to compute richer features (classically 4× wide)
the bend :: a nonlinearity (ReLU / GELU) in the middle :: without it, the two linear layers collapse into one — no power
down-projection :: second linear layer, hidden → `d_model` :: packs the result back so it fits the residual stream
position-wise :: same network run on each word on its own :: each word thinks alone; never mixes across words
residual + layer norm :: skip lane + re-level around the workshop :: keeps a deep stack of workshops trainable
dropout :: randomly switch off units in training :: stops the parameter-heavy workshop from memorizing (overfitting)
%%%

#### Cheat-sheet · the words you met today
%%% jargon
feed-forward network (FFN) | the per-token workshop: up-project, bend, down-project — run on each word on its own
position-wise | the same small network applied to each word's vector independently, one at a time
up-projection / down-projection | the two linear layers that widen the vector then pack it back to d_model
expansion ratio | how much wider the hidden space is than d_model — classically 4×
ReLU | a bend that keeps positives and turns negatives into a flat 0 (max(0, z))
GELU | a smooth bend used in GPT/BERT; lets small negatives leak so units rarely die
dying ReLU | a unit stuck outputting 0 forever, with no slope to learn its way back
residual connection | the skip lane that adds the workshop's change onto the untouched signal
dropout | randomly switching off units during training to prevent overfitting
overfitting | memorizing the training data instead of learning patterns that generalize
%%%

That's the whole day. Next you'll snap attention and this workshop together with their rails into **the full transformer block** — the complete repeating unit of every large language model.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What shape does the feed-forward layer give each word's vector? | a:2 | It keeps it the same width the whole way | It shrinks it, then never grows it back | It expands it to a wider hidden space (classically 4×), works there, then shrinks it back to d_model | It sorts the numbers from largest to smallest | fb: The FFN up-projects to a wider hidden dim (e.g. 512→2048), applies a nonlinearity, then down-projects back to d_model.
q: Why does the feed-forward layer need a nonlinearity in the middle? | a:1 | To make training faster | Because two linear layers with nothing between them collapse into a single linear layer — the bend is what gives depth real power | To reduce the number of parameters | To mix information between different words | fb: A bend (ReLU/GELU) is what lets stacked layers build shapes a single linear layer never could.
q: What is a "dying ReLU" and what fixes it? | a:2 | A unit that grows too large; fixed by layer norm | A slow layer; fixed by dropout | A unit stuck outputting a flat 0 with no slope to recover; GELU fixes it by staying smooth for small negatives | A memorized answer; fixed by attention | fb: ReLU flattens negatives to 0 (no slope), so a unit can get stuck dead; GELU's gentle curve keeps a learning signal alive.
q: Which is TRUE of the feed-forward layer? | a:3 | It mixes information across all the words in the sentence | It replaces the need for attention | It holds few parameters, so overfitting is not a concern | It's position-wise — it transforms each word on its own and never mixes across words (that's attention's job) | fb: The FFN is per-token: it thinks on each word alone. Sharing between words is attention's job — which is why a block needs both.
%%%

@@@ produce id=produce tag="Produce" title="Watch a word go through the workshop" gotit="Done"
Time to see today's big idea with your own eyes. You'll push a word's vector through a tiny feed-forward layer and **watch** it widen, bend, and shrink back — then prove to yourself that *removing the bend* makes the whole thing collapse. **Predict first:** if you widen a 4-number word to 16, then shrink it back, what width comes out? And with *no* bend in the middle, will two matrices behave any differently from one? Then run it and **observe**. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-03-feed-forward/experiment.py`. Start with one word `x = np.random.randn(4)` and set `hidden = 4 * 4`. **Notice** three things: (1) build an up-projection `W1` of shape `[4, 16]` and a down-projection `W2` of shape `[16, 4]`, compute `h = x @ W1`, apply ReLU (`np.maximum(0, h)`), then `out = relu_h @ W2`, and print the width at each step — **watch** it go `4 → 16 → 4`; (2) run the *same two matrices with NO bend* (`(x @ W1) @ W2`) and confirm it equals multiplying by the single collapsed matrix `x @ (W1 @ W2)` (use `np.allclose`) — proof that without the bend, depth buys nothing; (3) count how many negative values ReLU zeroed in `h`, then swap ReLU for a GELU-like smooth curve and **observe** that the small negatives are no longer a hard 0. Run with `python3 sessions/m05a-text-transformer/day-03-feed-forward/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5a Day 3 artifact.

Create sessions/m05a-text-transformer/day-03-feed-forward/experiment.py that, with a comment on each step:
1. Defines a position-wise feed_forward(x, W1, W2) with d_model=4 and hidden=16: h = x @ W1 (up-projection), a = np.maximum(0, h) (ReLU bend), out = a @ W2 (down-projection). Print the shape at each step to show 4 -> 16 -> 4.
2. Takes one word x = np.random.randn(4) and runs it through, printing input and output shapes (same width out as in).
3. Proves the bend matters: shows (x @ W1) @ W2 equals x @ (W1 @ W2) with np.allclose — i.e. WITHOUT the ReLU the two matrices collapse to one, so depth buys nothing. Then insert the ReLU and show the result now differs.
4. Counts how many entries ReLU set to exactly 0 (potential dead units), then defines a smooth GELU-like activation gelu(z)=0.5*z*(1+np.tanh(0.7978845608*(z+0.044715*z**3))) and shows those same small-negative entries are no longer a hard 0.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (widen, bend, shrink — then the collapse)
- The word goes in 4 wide, balloons to 16 in the hidden space, and comes back out 4 wide — the *widen-work-narrow* shape you drew today.
- With **no** bend, `(x @ W1) @ W2` exactly matches `x @ (W1 @ W2)` — two matrices really do collapse into one, so the extra layer was doing nothing. Slip the ReLU in and the answers split apart — that's the bend earning its keep.
- ReLU zeroed a handful of hidden values (those are the benches at risk of going dark); the GELU-like curve leaves those same small negatives as tiny non-zero numbers instead — the reason modern models prefer it.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-03-feed-forward/log.md`: (1) the workshop's shape in your own words (widen → bend → shrink) and why it's called *position-wise*; (2) why the bend matters — what happens to two linear layers with nothing between them; (3) one failure mode you met today (dying ReLU, unstable deep stack, or overfitting) and its named fix.
!!!

@@@ fin
