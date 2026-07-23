---
quest_id: wf5-d02-backward
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 2 — The Backward Pass, Wired by Hand"
module_label: "Module 5 · Train · Day 2"
title: "The Backward Pass"
subtitle: "Wired by Hand"
brand_sub: "Foundations · M5 Day 2"
spine: "practice: a machine learns by being corrected, one nudge at a time"
nav_prev_href: "../day-01-mlp-mnist/lesson.html"
nav_prev_label: "Your First Model"
nav_next_href: "../day-03-minibatch-loop/lesson.html"
nav_next_label: "The Mini-Batch Loop"
fin_title: "Module 5 · Day 2 complete! 🏆"
fin_body: "Nice work — you've completed <b>The Backward Pass, Wired by Hand</b>. You turned yesterday's guessing machine into a learning one: a single loss number, its gradient for every weight, the chain rule that carries blame backward, and the one reverse sweep that reuses the forward pass to compute it all. You even know how to catch a silent gradient bug.<br>Next up: <b>The Mini-Batch Loop</b> — where you run this update over batch after batch and watch the loss actually fall."
notebook_yardstick: 00-neural-networks/fundamentals/07_backpropagation.ipynb
coverage_topics:
  - {topic: loss as a single scalar signal, keywords: [loss, single number, how wrong, scalar]}
  - {topic: gradient is the partial derivative that points uphill, keywords: [gradient, partial derivative, steepest increase, points uphill, direction]}
  - {topic: chain rule multiplies local slopes along the path, keywords: [chain rule, multiply, local slope, along the path, blame]}
  - {topic: computational graph of forward operations, keywords: [computational graph, operations, matmul, add-bias, activation, edges]}
  - {topic: local gradient cached during forward pass, keywords: [local gradient, own derivative, cached, stored, forward pass]}
  - {topic: upstream gradient multiplied by local gradient, keywords: [upstream gradient, incoming, multiply, pass back, downstream]}
  - {topic: delta error signal propagated layer to layer, keywords: [delta, error signal, pre-activation, layer to layer]}
  - {topic: per-layer gradient formulas for a linear layer, keywords: [dL/dW, "delta · x", dL/db, "Wᵀ · delta", linear layer]}
  - {topic: activation-derivative element-wise multiply, keywords: [activation derivative, element-wise, relu passes, zeros it, input>0]}
  - {topic: backpropagation is one reverse sweep reusing cached forward values, keywords: [backpropagation, reverse sweep, reuse, cached, forward pass, one sweep]}
  - {topic: forward pass is the prerequisite that produced the loss, keywords: [forward pass first, produced a loss, day-01, prerequisite]}
  - {topic: numerical finite-difference gradient check, keywords: [numerical gradient, finite difference, "f(x+h)", gradient check, sanity]}
  - {topic: gradient shape must match parameter shape, keywords: [same shape, gradient shape, dL/dW same as W, debug invariant]}
  - {topic: vanishing gradients cause and ReLU remedy, keywords: [vanishing gradient, saturating, small slopes, relu, normalization, slope near 1]}
  - {topic: exploding gradients cause and gradient clipping remedy, keywords: [exploding gradient, large factors, gradient clipping, weight initialization]}
  - {topic: dead ReLU during backprop and leaky relu remedy, keywords: [dead relu, zero gradient forever, leaky relu, gelu, he initialization]}
require_artifact: true
---

@@@ hero
@lede Think about learning to shoot a basketball. Your first shot sails way over the hoop. Nobody just yells "wrong!" — a good coach says "too hard, and a little left." You adjust *exactly that much*, shoot again, and get closer. Practice alone teaches nothing; what teaches is **practice plus that aimed correction** after every miss. Yesterday you built a machine that looks at a messy "7" and guesses — but a fresh one guesses blindly, about one in ten right. Today you build its coach: the **backward pass**, the exact-correction step that turns a guessing machine into a learning one. It is the same idea, scaled up, that let ChatGPT learn to write.
@goal By the end you can say what the loss, the gradient, and the chain rule each are; walk a correction backward through a 2-layer network in your head; write the one-line rule that turns a forward pass into a gradient for every weight; and spot a silent gradient bug before it wastes a training run.

@@@ concept id=c1 tag="The score" title="One number that says how wrong we were" gotit="Got the loss"
Practice needs a scorekeeper. Before a machine can improve, it needs one honest number that says how badly *this* guess missed. Picture a round of **golf**: your whole game — every swing — is squeezed into a single score, and a *lower* score means a *better* round. You do not need ten numbers to know how you did; one is enough, and you play to shrink it. That single number is the [[loss||A single number measuring how wrong the model's output was on an example — small means close, large means far off. It is the one signal the whole backward pass tries to shrink.]]: one **scalar** (just one value, not a list) that gets smaller as the guess gets better.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A golf scorecard: many separate output scores on the left funnel into one single loss number, with lower meaning better and an arrow pointing toward zero"><g font-family="monospace" font-size="11" text-anchor="middle"><text x="90" y="20" fill="#6B645E">10 output scores (the guess)</text><g fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1"><rect x="30" y="32" width="24" height="18"/><rect x="56" y="32" width="24" height="18"/><rect x="82" y="32" width="24" height="18"/><rect x="108" y="32" width="24" height="18"/><rect x="134" y="32" width="24" height="18"/></g><text x="160" y="46" fill="#6B645E">…</text><text x="95" y="72" fill="#6B645E" font-size="10">compared to the true digit</text><path d="M170 44 L230 60" fill="none" stroke="#C99A12" stroke-width="2" marker-end="url(#a2a)"/><rect x="235" y="42" width="120" height="34" rx="8" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="295" y="64" fill="#C93B3B" font-weight="bold">loss = 2.7</text><text x="295" y="94" fill="#6B645E" font-size="10">one number · lower = better</text><path d="M360 59 L440 59" fill="none" stroke="#2D8B55" stroke-width="2" marker-end="url(#a2a)"/><text x="475" y="55" fill="#1a5c38" font-weight="bold">→ 0</text><text x="475" y="70" fill="#6B645E" font-size="9">perfect</text><defs><marker id="a2a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#6B645E"/></marker></defs></g></svg>
%%%

**What the golf-score picture gets right:** many separate things (each swing / each output score) collapse into one number you can push down, and lower is better. **Where it breaks down:** in golf you can only *count* your score after the round, but the loss is built from a formula we can do calculus on — that is what lets us compute *how to improve*, not just *that* we did badly.

#### Where this number comes from
Remember yesterday's **forward pass** from day-01 — pixels flowing left to right into 10 scores? The forward pass is the strict **prerequisite** for everything today: the loss can only exist once a forward pass has produced a loss to measure. The loss sits at the very end of that trip: after the forward pass produces the 10 scores, we compare them to the true digit and boil the difference down to this one number. So today's whole story has a strict order: **first a forward pass makes a guess, then we score it, then we correct.** Practice is the forward pass; the loss is the coach noticing the miss; the correction is everything below.

Let's *watch* that one number shrink as the guess gets closer to the target. Same target (5), two guesses — the squared miss is the bar's height:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Two bars showing squared loss: guessing 2 against a target of 5 gives a tall bar of height 9; guessing 4 gives a short bar of height 1 — closer guess, smaller loss"><g font-family="monospace" font-size="11" text-anchor="middle"><line x1="60" y1="20" x2="60" y2="130" stroke="#C9C2D8"/><line x1="60" y1="130" x2="470" y2="130" stroke="#C9C2D8"/><text x="40" y="80" fill="#6B645E" transform="rotate(-90 40 80)">loss</text><rect x="120" y="24" width="80" height="106" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="160" y="18" fill="#C93B3B" font-weight="bold">L = 9</text><text x="160" y="146" fill="#6B645E" font-size="10">guess 2 (far)</text><rect x="300" y="118" width="80" height="12" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="340" y="110" fill="#1a5c38" font-weight="bold">L = 1</text><text x="340" y="146" fill="#6B645E" font-size="10">guess 4 (close)</text><path d="M210 60 L292 108" stroke="#2D8B55" stroke-width="2" marker-end="url(#l1)"/><text x="252" y="70" fill="#1a5c38" font-size="9">closer → smaller</text><text x="420" y="128" fill="#6B645E" font-size="9">target = 5</text><defs><marker id="l1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker></defs></g></svg>
%%%

!!! c-info 🧮
<b>Optional (skippable) — a tiny loss.</b> A simple loss for one number is the squared miss: `L = (guess − target)²`. If the guess is `2` and the target is `5`, then `L = (2−5)² = 9`. Guess `4` instead and `L = (4−5)² = 1` — smaller, because you are closer. Squaring makes every miss positive and punishes big misses more. The exact loss formula (and why we use a fancier one for 10 classes) is Day 4 — today one honest number is all we need.
!!!

@@@ concept id=c2 tag="Which way" title="The gradient — which way, and how much" gotit="Got the gradient"
Good — we have one score to shrink. But "you're wrong" is useless on its own. The coach must also say *which way to move* and *how much*. Picture standing **blindfolded on a grassy hill**, wanting to reach the bottom. You cannot see the valley, but you can feel the ground under your feet: which way slopes down, and how steep it is. You take a small step in the steepest-downhill direction, feel again, and repeat. That "steepness and direction under your feet" is exactly the [[gradient||A vector of slopes — one number per knob — telling how much the loss changes if you nudge that knob up. It points in the direction of steepest *increase*, so you step the opposite way to go downhill.]]: for each knob in the network it says *how fast the loss changes* if you nudge that knob, and it points in the direction of steepest **increase**. To go downhill (shrink the loss), you step the *opposite* way.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A blindfolded figure standing on a hill of loss, an arrow along the slope showing the uphill gradient direction and a green arrow the opposite way toward the valley bottom"><g font-family="monospace" font-size="11"><path d="M20 130 Q160 30 300 90 Q400 130 500 60" fill="none" stroke="#C9C2D8" stroke-width="3"/><path d="M20 130 Q160 30 300 90 Q400 130 500 60 L500 145 L20 145 Z" fill="#F3EFFA" opacity="0.5"/><circle cx="120" cy="66" r="7" fill="#3A342E"/><text x="120" y="52" fill="#3A342E" text-anchor="middle" font-size="10">you (a knob's setting)</text><path d="M128 70 L170 92" stroke="#C93B3B" stroke-width="3" marker-end="url(#g2)"/><text x="205" y="92" fill="#C93B3B" font-size="10">gradient → uphill (loss grows)</text><path d="M112 62 L70 44" stroke="#2D8B55" stroke-width="3" marker-end="url(#g2)"/><text x="55" y="34" fill="#1a5c38" font-size="10" text-anchor="middle">step this way ↓ loss</text><text x="300" y="118" fill="#5E5191" font-size="10" text-anchor="middle">valley = smallest loss</text><defs><marker id="g2" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="context-stroke"/></marker></defs></g></svg>
%%%

**What the blindfolded-hill picture gets right:** you never see the whole landscape, you only feel the slope right where you stand — and that local slope is enough to know which way is down. **Where it breaks down:** a real hill has just two directions to lean (left–right, front–back); the loss "hill" has one direction *per knob* — for our model, about 100,000 of them at once — so the gradient is a long list of slopes, not a single arrow.

#### Slope means "how much the loss moves per nudge"
A slope is a plain idea you already know. If you walk from mile 10 to mile 14 in 2 hours, your speed is `(14−10)/2 = 2` miles per hour — that "2" is a slope: *how much one thing changes per change in another.* The gradient of a single knob is the same: **how much does the loss change per tiny nudge of this knob?** Let's *see* that on the tiny golf-loss curve from c1. Drag your eye across it: on the left the slope tilts down steeply (big correction), near the bottom it flattens (tiny correction).

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A U-shaped loss curve. At a point far up the left side a steep tangent line is drawn (large gradient, big step); near the flat bottom a nearly level tangent (small gradient, small step)"><g font-family="monospace" font-size="10" text-anchor="middle"><line x1="50" y1="20" x2="50" y2="150" stroke="#C9C2D8"/><line x1="50" y1="150" x2="500" y2="150" stroke="#C9C2D8"/><text x="30" y="90" fill="#6B645E" transform="rotate(-90 30 90)">loss</text><text x="275" y="172" fill="#6B645E">knob setting →</text><path d="M80 40 Q275 230 470 40" fill="none" stroke="#C93B3B" stroke-width="2.5"/><circle cx="120" cy="95" r="5" fill="#3A342E"/><line x1="90" y1="135" x2="160" y2="55" stroke="#9A5A12" stroke-width="2.5"/><text x="140" y="45" fill="#9A5A12">steep slope → big step</text><circle cx="275" cy="147" r="5" fill="#3A342E"/><line x1="235" y1="149" x2="315" y2="145" stroke="#2D8B55" stroke-width="2.5"/><text x="275" y="140" fill="#1a5c38">flat slope → tiny step</text></g></svg>
%%%

Far from the bottom the slope is steep, so the gradient is big and the correction is big; near the bottom the slope flattens, the gradient shrinks toward zero, and the corrections get gentle — the machine naturally eases off as it gets good, just like your basketball shots needed smaller and smaller adjustments.

!!! c-info 🧮
<b>Optional (skippable) — the word "partial."</b> When the loss depends on many knobs, the slope for *one* knob (holding the rest still) is called a [[partial derivative||The slope of the loss with respect to one knob while all other knobs are held fixed — written ∂L/∂w. Stack one of these per knob and you get the gradient vector.]], written `∂L/∂w`. Stack one partial derivative per knob into a list and that whole list *is* the gradient. So "compute the gradient" just means "find every knob's slope." How we actually do the *update* — `w ← w − learning_rate × gradient`, and how big a step to take — is Day 3; today we only compute the slopes.
!!!

@@@ concept id=c3 tag="Chain of blame" title="The chain rule — passing blame down a line" gotit="Got the chain rule"
Here is the deep puzzle. The loss lives at the *very end* of the network. But the knob we want to fix might sit way back at the *first* layer. How does the blame travel back that far? The answer is a rule so simple you already use it. Picture a **line of dominoes**: the first tips the second, the second tips the third, the third tips the last. If you want to know how hard your *first* push moves the *last* domino, you don't need to watch the whole chain — you just multiply the effects link by link. Push twice as hard → first domino falls twice as far → and if each domino moves the next one three times as much, the last one moves `2 × 3 × 3 = 18` times as far. That multiply-along-the-line rule is the [[chain rule||The calculus rule for a chain of steps: to get how the first input affects the final output, multiply the local slope of each step along the path. It is the engine that carries the loss's blame backward.]] — the engine of the whole backward pass.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A row of four dominoes falling left to right, each labeled with a local slope, and the arrows between them multiplied together to give the total effect of the first push on the last domino"><g font-family="monospace" font-size="10" text-anchor="middle"><g fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"><rect x="40" y="40" width="18" height="70" rx="2" transform="rotate(-12 49 75)"/><rect x="150" y="40" width="18" height="70" rx="2" transform="rotate(-8 159 75)"/><rect x="260" y="40" width="18" height="70" rx="2" transform="rotate(-6 269 75)"/><rect x="370" y="40" width="18" height="70" rx="2"/></g><text x="49" y="128" fill="#5E5191">push</text><text x="379" y="128" fill="#C93B3B" font-weight="bold">last</text><path d="M70 62 L142 62" stroke="#C99A12" stroke-width="2" marker-end="url(#d3)"/><text x="106" y="52" fill="#9A5A12">× 2</text><path d="M180 62 L252 62" stroke="#C99A12" stroke-width="2" marker-end="url(#d3)"/><text x="216" y="52" fill="#9A5A12">× 3</text><path d="M290 62 L362 62" stroke="#C99A12" stroke-width="2" marker-end="url(#d3)"/><text x="326" y="52" fill="#9A5A12">× 3</text><text x="260" y="145" fill="#C93B3B" font-weight="bold">first push → last domino: 2 × 3 × 3 = 18</text><defs><marker id="d3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#C99A12"/></marker></defs></g></svg>
%%%

**What the dominoes picture gets right:** you find a far-reaching effect by *multiplying* the small effect at each link — no need to re-simulate the whole chain. **Where it breaks down:** dominoes only fall one way, so the picture makes it look forward; in a network we run the multiply *backward*, from the loss end toward the knob, because that is the cheap direction (more on why in c5).

#### Each step keeps its own little slope
A network's forward pass is just such a chain: input → multiply by weights → add bias → bend (ReLU) → next layer → … → loss. Each little step has its **own** slope — how much its output moves when its input moves. That per-step slope is the [[local gradient||A single operation's own derivative: how much its output changes when its input changes. It is computed and stored during the forward pass, ready to be multiplied in on the way back.]]. Here is the elegant part: each step can figure out its local slope *while the forward pass is happening* and quietly **cache** (store) it. Then, coming back, the chain rule just multiplies those saved slopes together. Predict the answer, then run this three-link chain:

%%% demo id=chain label="multiply the local slopes"
code: local = [2.0, 3.0, 3.0]; total = 1; [total := total * s for s in local]; total
out: 18.0    # 2 · 3 · 3 — the first push moves the last domino 18× as far
take: <b>The chain rule is just multiplication.</b> Each step contributes one local slope; multiply them along the path and you get how the very first input affects the final loss. Change any single link and the whole product changes — that is how blame flows.
%%%

So the chain rule turns a scary "how does this early knob affect the far-away loss?" into a friendly "multiply the slopes I already saved." That is the trick the backward pass runs on — and next we'll draw the exact line the multiply travels down.

@@@ concept id=c4 tag="The wiring" title="The graph, and the multiply-and-pass-back" gotit="Got the pass-back"
To multiply slopes along a path, we first need to *see* the path. Picture the forward pass as a **plumbing diagram**: water (numbers) flows through a series of connected pipes — one pipe multiplies by the weights, the next adds the bias, the next is the ReLU bend — and out the end comes the loss. Drawing the network this way, as boxes (operations) joined by arrows (the numbers flowing between them), is called the [[computational graph||A picture of the forward pass as boxes (operations like matmul, add-bias, activation) joined by arrows (the values flowing between them). Gradients flow backward along the very same arrows.]]. Every arrow that carried a number *forward* will carry a slope *backward*.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A computational graph: input flows right through matmul, add-bias, and ReLU boxes to a loss box; below, red arrows flow left showing the gradient traveling backward along the same edges"><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#1a5c38">forward: numbers flow right →</text><g fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"><rect x="20" y="30" width="60" height="30" rx="5"/><rect x="120" y="30" width="70" height="30" rx="5"/><rect x="230" y="30" width="80" height="30" rx="5"/><rect x="350" y="30" width="60" height="30" rx="5"/><rect x="450" y="30" width="55" height="30" rx="5"/></g><text x="50" y="50" fill="#1a5c38">input x</text><text x="155" y="50" fill="#1a5c38">× W</text><text x="270" y="50" fill="#1a5c38">+ b</text><text x="380" y="50" fill="#1a5c38">ReLU</text><text x="477" y="50" fill="#C93B3B">loss</text><g stroke="#2D8B55" stroke-width="1.5"><line x1="80" y1="45" x2="118" y2="45" marker-end="url(#f4)"/><line x1="190" y1="45" x2="228" y2="45" marker-end="url(#f4)"/><line x1="310" y1="45" x2="348" y2="45" marker-end="url(#f4)"/><line x1="410" y1="45" x2="448" y2="45" marker-end="url(#f4)"/></g><text x="260" y="92" fill="#C93B3B">backward: slopes flow left ← (same arrows)</text><g stroke="#C93B3B" stroke-width="1.5"><line x1="448" y1="112" x2="410" y2="112" marker-end="url(#b4)"/><line x1="348" y1="112" x2="310" y2="112" marker-end="url(#b4)"/><line x1="228" y1="112" x2="190" y2="112" marker-end="url(#b4)"/><line x1="118" y1="112" x2="80" y2="112" marker-end="url(#b4)"/></g><text x="470" y="116" fill="#C93B3B">start</text><text x="50" y="116" fill="#5E5191">to knobs</text><defs><marker id="f4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker><marker id="b4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#C93B3B"/></marker></defs></g></svg>
%%%

**What the plumbing picture gets right:** the whole network is just connected boxes, and whatever flows forward through an arrow can flow back through the same arrow. **Where it breaks down:** water only flows downhill one way, but our arrows carry *numbers* forward and *slopes* backward along the identical wiring — one set of pipes, two directions.

#### At each box: two arrows in, one arrow out
Now zoom into one box. A box receives a slope coming *in from its right* — the [[upstream gradient||The slope arriving from the operation just after this one — "how much does the loss care about my output?" You multiply it by your own local gradient to get the slope you send further back.]] (from the layer closer to the loss). It already knows its **own** local slope from c3. The chain rule says: **multiply the upstream gradient by your local gradient, and pass the product further back.** That product is the box's downstream gradient — its message to the boxes behind it.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Zoom into one operation box: an upstream gradient arrow arrives from the right, it is multiplied by the box's stored local gradient, and the product is sent out to the left as the downstream gradient"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="200" y="45" width="120" height="60" rx="8" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="260" y="70" fill="#5E5191" font-weight="bold">one op</text><text x="260" y="90" fill="#5E5191" font-size="9">local slope = 0.5</text><path d="M470 75 L322 75" stroke="#C93B3B" stroke-width="2.5" marker-end="url(#u4)"/><text x="400" y="62" fill="#C93B3B" font-size="10">upstream = 4</text><text x="400" y="98" fill="#6B645E" font-size="9">(comes from the loss side)</text><path d="M198 75 L50 75" stroke="#2D8B55" stroke-width="2.5" marker-end="url(#u4b)"/><text x="120" y="62" fill="#1a5c38" font-size="10">downstream = 4 × 0.5 = 2</text><text x="120" y="98" fill="#6B645E" font-size="9">(sent further back)</text><text x="260" y="140" fill="#3A342E" font-weight="bold">multiply-and-pass-back: upstream × local → downstream</text><defs><marker id="u4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#C93B3B"/></marker><marker id="u4b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker></defs></g></svg>
%%%

Every box does the same tiny job: take the slope from the right, multiply by its own saved slope, hand the result to the left. Do this box after box and the chain rule *automatically* multiplies every local slope along the path — no giant formula, just the same little multiply repeated.

#### The signal that flows between layers has a name
When this slope reaches the point *just before* a layer's bend — the layer's raw pre-bend score — we give it a special name: the [[delta||The error signal at a layer: the gradient of the loss with respect to that layer's pre-activation score. Backprop carries delta from one layer to the previous one — it is the "blame" each layer receives.]] (written **δ**), the layer's **error signal**. Think of δ as the blame each layer is handed: "this much of the final miss is on you." Backprop's real job, layer by layer, is to pass δ from each layer to the one before it. Next we turn that δ into an actual correction for the weights.

@@@ concept id=c5 tag="One sweep back" title="Backprop — one reverse trip that reuses the forward pass" gotit="Got backprop"
Now the payoff — and it's a beautiful one. Picture retracing your steps through a **corn maze**. On the way *in* you dropped a breadcrumb at every turn. To find your way *out*, you don't re-explore the whole maze — you just walk the breadcrumb trail backward, one turn at a time. The forward pass left breadcrumbs (the cached local slopes from c3 and the values that flowed through), and [[backpropagation||The algorithm that computes the gradient for every weight and bias in one reverse sweep: start with δ at the loss, walk backward through the graph multiplying by each local gradient (chain rule), reusing the values cached during the forward pass.]] is one single walk backward along that trail, carrying δ and multiplying as it goes. One forward trip, one backward trip — and you come out holding a slope for *every* knob.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A two-layer network drawn twice: top row shows the forward pass left to right dropping cached values as breadcrumbs; bottom row shows one backward sweep right to left carrying delta and reading those breadcrumbs to produce a gradient for each weight"><g font-family="monospace" font-size="9" text-anchor="middle"><text x="90" y="14" fill="#1a5c38">FORWARD — leave breadcrumbs 🍞</text><g fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"><rect x="20" y="22" width="46" height="24" rx="4"/><rect x="110" y="22" width="60" height="24" rx="4"/><rect x="215" y="22" width="46" height="24" rx="4"/><rect x="305" y="22" width="60" height="24" rx="4"/><rect x="410" y="22" width="46" height="24" rx="4"/></g><text x="43" y="38" fill="#1a5c38">x</text><text x="140" y="38" fill="#1a5c38">layer1</text><text x="238" y="38" fill="#1a5c38">ReLU</text><text x="335" y="38" fill="#1a5c38">layer2</text><text x="433" y="38" fill="#C93B3B">loss</text><g stroke="#2D8B55"><line x1="66" y1="34" x2="108" y2="34" marker-end="url(#s5)"/><line x1="170" y1="34" x2="213" y2="34" marker-end="url(#s5)"/><line x1="261" y1="34" x2="303" y2="34" marker-end="url(#s5)"/><line x1="365" y1="34" x2="408" y2="34" marker-end="url(#s5)"/></g><text x="88" y="60" fill="#8A6D3B" font-size="8">save x</text><text x="285" y="60" fill="#8A6D3B" font-size="8">save which units &gt; 0</text><text x="430" y="90" fill="#C93B3B">BACKWARD — one sweep, read breadcrumbs</text><g fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.3"><rect x="20" y="100" width="46" height="24" rx="4"/><rect x="110" y="100" width="60" height="24" rx="4"/><rect x="215" y="100" width="46" height="24" rx="4"/><rect x="305" y="100" width="60" height="24" rx="4"/><rect x="410" y="100" width="46" height="24" rx="4"/></g><g stroke="#C93B3B"><line x1="408" y1="112" x2="366" y2="112" marker-end="url(#s5b)"/><line x1="303" y1="112" x2="261" y2="112" marker-end="url(#s5b)"/><line x1="213" y1="112" x2="170" y2="112" marker-end="url(#s5b)"/><line x1="108" y1="112" x2="66" y2="112" marker-end="url(#s5b)"/></g><text x="140" y="116" fill="#C93B3B" font-size="8">δ · x → dW1</text><text x="335" y="116" fill="#C93B3B" font-size="8">δ · h → dW2</text><text x="260" y="150" fill="#3A342E" font-weight="bold">1 forward + 1 backward → a gradient for every knob</text><defs><marker id="s5" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L5 3 L0 6 z" fill="#2D8B55"/></marker><marker id="s5b" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L5 3 L0 6 z" fill="#C93B3B"/></marker></defs></g></svg>
%%%

**What the corn-maze picture gets right:** you reuse the breadcrumbs from the way in instead of re-solving the maze, which is why one backward walk is cheap. **Where it breaks down:** in a maze you drop breadcrumbs to remember the *path*; here the "breadcrumbs" are actual numbers (each layer's input and which ReLU units were on) kept in memory — that memory cost is real, and it is why the forward values are held until the backward pass finishes.

#### Two jobs at each layer
Walking backward, each layer takes the δ handed to it and does exactly two things:
- **Make its correction.** It combines δ with the layer's *saved input* to get the slope for its own weights — the layer's `dW` (how to nudge W) and `db` (how to nudge b). This is where the breadcrumb from the forward pass gets used.
- **Pass blame further back.** It sends a new δ to the layer before it, so that layer can do the same.

There is one extra twist at a bend. When δ passes back *through* a ReLU, the ReLU acts like a **gate**: it only lets the slope through for the units that were *positive* on the way forward, and blocks it (multiplies by 0) for the units that were negative. That is the [[activation derivative||Each activation's own local slope. For ReLU it is 1 where the input was positive and 0 where it was negative — so backprop passes the gradient through "on" units and zeroes it for "off" units, an element-wise multiply.]] doing its job — an element-wise multiply of δ by "was this unit on? 1 or 0." Watch the gate open and close:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Before and after: an incoming gradient vector of four numbers passes back through a ReLU gate. Units that were positive on the forward pass keep their gradient; units that were negative have their gradient set to zero"><g font-family="monospace" font-size="10" text-anchor="middle"><text x="90" y="16" fill="#C93B3B">incoming δ (from the right)</text><g fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.2"><rect x="30" y="26" width="34" height="24"/><rect x="64" y="26" width="34" height="24"/><rect x="98" y="26" width="34" height="24"/><rect x="132" y="26" width="34" height="24"/></g><text x="47" y="43" fill="#C93B3B">0.4</text><text x="81" y="43" fill="#C93B3B">-0.7</text><text x="115" y="43" fill="#C93B3B">0.9</text><text x="149" y="43" fill="#C93B3B">-0.2</text><text x="255" y="16" fill="#2D8B55">ReLU gate: was unit on (&gt;0)?</text><g font-size="10"><rect x="220" y="26" width="34" height="24" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.2"/><text x="237" y="43" fill="#1a5c38">on</text><rect x="254" y="26" width="34" height="24" fill="#EFEBE4" stroke="#B7AFA3" stroke-width="1.2"/><text x="271" y="43" fill="#8A8378">off</text><rect x="288" y="26" width="34" height="24" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.2"/><text x="305" y="43" fill="#1a5c38">on</text><rect x="322" y="26" width="34" height="24" fill="#EFEBE4" stroke="#B7AFA3" stroke-width="1.2"/><text x="339" y="43" fill="#8A8378">off</text></g><text x="90" y="98" fill="#5E5191">outgoing δ (to the left)</text><g fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.2"><rect x="30" y="108" width="34" height="24"/><rect x="64" y="108" width="34" height="24"/><rect x="98" y="108" width="34" height="24"/><rect x="132" y="108" width="34" height="24"/></g><text x="47" y="125" fill="#5E5191">0.4</text><text x="81" y="125" fill="#8A8378">0</text><text x="115" y="125" fill="#5E5191">0.9</text><text x="149" y="125" fill="#8A8378">0</text><text x="330" y="120" fill="#3A342E" font-weight="bold">on → keep the slope · off → zero it</text><text x="330" y="138" fill="#6B645E" font-size="9">(element-wise multiply by 1 or 0)</text></g></svg>
%%%

See the before-and-after: the gradient for units that were "on" passes straight through; the gradient for units that were "off" is zeroed. That is the *whole* rule for sending δ back across a ReLU. Predict which entries survive, then run it:

%%% demo id=relugate label="push a gradient back through ReLU"
code: z_fwd = np.array([2.0, -1.0, 3.0, -0.5]); upstream = np.array([0.4, -0.7, 0.9, -0.2]); grad = upstream * (z_fwd > 0); grad
out: array([0.4, 0. , 0.9, 0. ])   # kept where z_fwd>0, zeroed where z_fwd<=0
take: <b>ReLU's backward step is a gate.</b> The slope survives only for units that were positive on the forward pass; the rest are zeroed. That is why backprop needs the forward pass's saved values — it must remember which units were on.
%%%

!!! c-info 🧮
<b>Optional (skippable) — the three per-layer formulas.</b> For one linear layer `z = W·x + b` with incoming error signal `δ`, the backward pass is exactly three lines:
<br>• <b>weight slope:</b> `dL/dW = δ · xᵀ` — δ times the layer's *saved input* (this is the breadcrumb).
<br>• <b>bias slope:</b> `dL/db = δ` — the bias slope is just δ itself.
<br>• <b>pass-back:</b> `dL/dx = Wᵀ · δ` — this becomes the δ for the layer before (after the ReLU gate multiplies it by 1/0).
<br>The `ᵀ` (transpose) just flips a matrix's rows and columns so the shapes line up. That is the entire backward pass for a layer — three short lines, reused at every layer.
!!!

@@@ concept id=c6 tag="Catch the bug" title="Two quick ways to trust your gradient" gotit="Got the checks"
You just wired a backward pass by hand — exciting, but easy to get subtly wrong (a flipped sign, a forgotten transpose). Here is the good news: you can *check* your work before wasting hours of training. Picture weighing gold on a fancy digital scale you just bought. Before you trust it, you put a **known 1-gram weight** on it: if it reads 1.00, you trust it; if it reads 3.7, something's off. We do the same for gradients with a slow-but-honest second opinion called the [[numerical gradient||A gradient estimated by brute force: nudge one knob up by a tiny h and down by h, see how much the loss moved, divide by 2h. Slow but simple — used only to check the fast hand-derived gradient.]]: nudge one knob up a tiny bit, nudge it down a tiny bit, and see how much the loss actually moved.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A loss curve with a point on it. Two nearby samples are taken at knob minus h and knob plus h; the straight line between them approximates the true slope, illustrating the finite-difference numerical gradient"><g font-family="monospace" font-size="10" text-anchor="middle"><line x1="50" y1="20" x2="50" y2="135" stroke="#C9C2D8"/><line x1="50" y1="135" x2="500" y2="135" stroke="#C9C2D8"/><text x="30" y="80" fill="#6B645E" transform="rotate(-90 30 80)">loss</text><text x="275" y="152" fill="#6B645E">knob w →</text><path d="M70 40 Q275 210 480 40" fill="none" stroke="#C93B3B" stroke-width="2.5"/><circle cx="200" cy="112" r="4" fill="#8A6D3B"/><circle cx="240" cy="130" r="4" fill="#8A6D3B"/><line x1="180" y1="102" x2="260" y2="140" stroke="#2D8B55" stroke-width="2.5" stroke-dasharray="5,3"/><text x="200" y="98" fill="#8A6D3B">w − h</text><text x="255" y="150" fill="#8A6D3B">w + h</text><text x="360" y="70" fill="#1a5c38">slope ≈ (L(w+h) − L(w−h)) / 2h</text><text x="360" y="88" fill="#6B645E" font-size="9">the line through two nearby points</text></g></svg>
%%%

**What the calibration-weight picture gets right:** you test a fast instrument against a slow, trusted reference, and if they disagree the instrument is wrong. **Where it breaks down:** a calibration weight is exact, but the numerical gradient is only an *approximation* (it uses a tiny step `h`, not an infinitely small one), so you check that the two *match closely*, not that they are bit-for-bit identical.

#### The check in one line
Compute the same slope two ways and compare. Your fast hand-derived (analytic) gradient should match the slow numerical one to several decimal places. Predict whether they'll match, then run it:

%%% demo id=gradcheck label="analytic vs numerical"
code: f = lambda w: (w - 5)**2; w = 2.0; analytic = 2*(w - 5); h = 1e-5; numerical = (f(w+h) - f(w-h)) / (2*h); (analytic, round(numerical, 4))
out: (-6.0, -6.0)    # they match → the hand-derived gradient is correct
take: <b>Gradient check = trust, but verify.</b> The analytic slope (fast, from calculus) and the numerical slope (slow, from nudging) agree, so your backward pass has no sign or transpose bug. If they disagreed, you'd hunt the bug before training, not after.
%%%

#### The free sanity check: shapes
There is an even faster check that costs nothing. A knob's gradient tells you how to nudge that knob, so it must have **exactly the same shape** as the knob. `dL/dW` must be the same shape as `W`; `dL/db` the same shape as `b`. If `W` is `(784, 128)` and your `dW` comes out `(128, 784)`, you flipped a transpose — caught in one glance, before you run anything.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A weight matrix W of shape 784 by 128 shown beside its gradient dW, which must be the identical shape; a matching pair is checkmarked green and a flipped 128 by 784 pair is X-marked red"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="30" y="30" width="70" height="60" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="65" y="64" fill="#5E5191">W</text><text x="65" y="104" fill="#6B645E" font-size="9">(784,128)</text><text x="130" y="64" fill="#2D8B55">=?</text><rect x="160" y="30" width="70" height="60" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="195" y="64" fill="#1a5c38">dW</text><text x="195" y="104" fill="#6B645E" font-size="9">(784,128)</text><text x="255" y="64" fill="#2D8B55" font-weight="bold">✓ match</text><rect x="360" y="42" width="60" height="36" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="390" y="64" fill="#C93B3B" font-size="9">dW (128,784)</text><text x="460" y="64" fill="#C93B3B" font-weight="bold">✗ flipped</text></g></svg>
%%%

Two checks, both cheap, both catch the classic bugs: **numerical gradient** for a wrong value, **shape match** for a wrong transpose. Run them once and you can trust your backward pass for the whole rest of training.

@@@ concept id=c7 tag="When it fades" title="Three ways the signal misbehaves — and the fixes" gotit="Got the failure modes"
Now a genuinely fun puzzle. Backprop multiplies a slope by a slope by a slope, layer after layer — and *repeated multiplication* is a wild thing. Picture a game of **telephone**, where a whispered message passes down a line of kids. If each kid whispers a little *quieter*, the message fades to silence before it reaches the end. If each kid *shouts louder*, it becomes a deafening, garbled roar. And if one kid just stays silent, everyone behind them hears nothing at all. The exact same three things happen to δ as it travels back through a deep network — three quiet failures, each with a clean, named fix.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A line of nodes passing a signal. Top: each multiply by 0.25 shrinks the bar until it vanishes. Middle: each multiply by 4 grows the bar until it explodes. Bottom: a dead unit passes zero so nothing flows"><g font-family="monospace" font-size="9" text-anchor="middle"><text x="20" y="20" fill="#C99A12" text-anchor="start">VANISH (×0.25 each step):</text><g fill="#C99A12"><rect x="150" y="8" width="40" height="16"/><rect x="220" y="12" width="20" height="12"/><rect x="290" y="15" width="9" height="9"/><rect x="360" y="18" width="4" height="6"/><rect x="430" y="20" width="2" height="4"/></g><text x="470" y="22" fill="#C99A12">→ 0 😶</text><text x="20" y="62" fill="#C93B3B" text-anchor="start">EXPLODE (×4 each step):</text><g fill="#C93B3B"><rect x="150" y="54" width="4" height="10"/><rect x="220" y="48" width="9" height="16"/><rect x="290" y="42" width="18" height="22"/><rect x="360" y="34" width="30" height="30"/><rect x="430" y="26" width="40" height="38"/></g><text x="475" y="52" fill="#C93B3B">→ 💥</text><text x="20" y="110" fill="#5E5191" text-anchor="start">DEAD unit (×0 forever):</text><g><rect x="150" y="98" width="30" height="16" fill="#5E5191"/><rect x="290" y="98" width="30" height="16" fill="#EFEBE4" stroke="#B7AFA3"/><text x="305" y="110" fill="#8A8378">0</text><rect x="430" y="98" width="30" height="16" fill="#EFEBE4" stroke="#B7AFA3"/><text x="445" y="110" fill="#8A8378">0</text></g><text x="230" y="110" fill="#5E5191">→ blocked →</text><text x="260" y="140" fill="#6B645E">repeated multiplication makes a signal fade, blow up, or die</text></g></svg>
%%%

**What the telephone picture gets right:** a message passed hand to hand can quietly shrink, blow up, or get blocked — and you often don't notice until the end. **Where it breaks down:** telephone garbles *words*, but here it is a *number* (the gradient) being multiplied, so the failure is precise and, better yet, fixable with the right activation and weight setup.

#### Watch a signal vanish, one multiply at a time
Sigmoid and tanh — older bends — have slopes far below 1 across most of their range (often near 0.25 or less). Multiply 0.25 by itself a few times and the signal collapses to nearly nothing. Predict the last number before you run it:

%%% demo id=vanish label="multiply 0.25 six times"
code: g = 1.0; [g := g * 0.25 for _ in range(6)]; round(g, 8)
out: 0.00024414    # 0.25^6 — after six layers the gradient is basically gone
take: <b>Vanishing gradient.</b> Small slopes multiplied over many layers shrink toward zero, so early layers get almost no correction and stop learning. The fix: use <b>ReLU</b> (slope exactly 1 for positive inputs, so nothing shrinks) and <b>normalization</b> to keep signals well-scaled.
%%%

#### The three stalls, and their one-move fixes
Each failure has a cause (what the multiply does) and a named remedy. This table is the whole story at a glance:

%%% table
:: Failure :: Cause (what repeated ×  does) :: Named fix
Vanishing gradient :: slopes < 1 multiply toward 0 → early layers barely learn :: ReLU (slope 1) + normalization
Exploding gradient :: slopes / weights > 1 multiply to a huge number → wild, unstable steps :: gradient clipping + careful weight init
Dead ReLU :: a unit stuck negative passes 0 gradient forever → its weights never update :: Leaky ReLU / GELU + He initialization
%%%

- **Vanishing** — the whisper fades. Slopes below 1 multiply toward zero, so the earliest layers get almost no signal. *Fix:* [[ReLU||A bend whose slope is exactly 1 for positive inputs, so it does not shrink the gradient the way sigmoid/tanh do — the standard remedy for vanishing gradients, alongside normalization.]] keeps the slope at 1 for "on" units, and normalization keeps values well-scaled.
- **Exploding** — the shout gets louder. Big weights or slopes multiply into an enormous number, and the update jumps wildly. *Fix:* [[gradient clipping||Capping the gradient's size before the update so one huge value can't throw the model off — the standard remedy for exploding gradients, together with careful weight initialization.]] caps the gradient's size, and careful weight initialization stops it growing in the first place.
- **Dead ReLU** — the silent kid. A unit shoved permanently negative outputs 0, so (from c5's gate) it passes **0** gradient forever and never recovers. *Fix:* [[Leaky ReLU||A ReLU variant with a tiny negative slope instead of a flat zero, so a stuck unit still passes a little gradient and can climb back to life; GELU is a smooth cousin. He initialization also helps units not start dead.]] (or GELU) keeps a small negative slope so blame still trickles through, and **He initialization** helps units not start dead.

!!! c-warn ⚠️
<b>The senior's trick (one line):</b> all three are <em>silent</em> — no crash, the loss just sits still or turns into <code>NaN</code>. You catch them by <em>watching the gradient sizes and how many ReLU units are stuck at 0</em> during training, not by staring at the final accuracy. Spotting a flat loss and knowing to look at gradient norms is exactly the instinct that separates a junior from a staff engineer.
!!!

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath and look what you built. Yesterday's machine only guessed. Today you gave it a coach, so its practice finally pays off: it can score its own miss, feel which way to move every knob, and carry that correction all the way back to the very first layer — in one clean reverse trip. Here is the whole day as one staircase, each rung resting on the one below, exactly the way the backward pass carries δ down through the layers.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A staircase of today's ideas, bottom to top: loss is one number, gradient is which-way-and-how-much, chain rule multiplies local slopes, the graph passes upstream times local backward, backprop is one reverse sweep, gradient-check and shapes catch bugs, and watch for vanishing exploding dead-ReLU"><g font-family="monospace" font-size="10"><rect x="20" y="176" width="150" height="24" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.4"/><text x="28" y="192" fill="#C93B3B">1 · loss = one number</text><rect x="55" y="150" width="185" height="24" rx="4" fill="#F3EFFA" stroke="#7C6DAA" stroke-width="1.4"/><text x="63" y="166" fill="#5E5191">2 · gradient = way + how much</text><rect x="90" y="124" width="205" height="24" rx="4" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.4"/><text x="98" y="140" fill="#8A6D3B">3 · chain rule = multiply slopes</text><rect x="125" y="98" width="215" height="24" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.4"/><text x="133" y="114" fill="#5E5191">4 · graph: upstream × local → δ back</text><rect x="160" y="72" width="220" height="24" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="168" y="88" fill="#1a5c38">5 · backprop = one reverse sweep</text><rect x="195" y="46" width="205" height="24" rx="4" fill="#F3EFFA" stroke="#7C6DAA" stroke-width="1.4"/><text x="203" y="62" fill="#5E5191">6 · gradient-check + shape match</text><rect x="230" y="20" width="230" height="24" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.4"/><text x="238" y="36" fill="#C93B3B">7 · watch vanish / explode / dead</text></g></svg>
%%%

#### The arc in seven beats — each with the picture that anchored it
1. **The loss (the golf score):** the forward pass makes a guess, then one honest number says how wrong it was — the thing the whole backward pass shrinks.
2. **The gradient (the blindfolded hill):** each knob's slope — which way the loss grows and how steeply — so you step the opposite way. Steep far from the bottom, gentle near it.
3. **The chain rule (the dominoes):** to know how an early knob affects the far-away loss, multiply the local slope of each step along the path.
4. **The graph + pass-back (the plumbing):** draw the forward pass as boxes and arrows; at each box multiply the upstream gradient by its own local slope and hand the product back. The signal between layers is δ, the error signal.
5. **Backprop (the corn-maze breadcrumbs):** one forward trip leaves cached values, one backward trip reuses them — combining δ with each layer's saved input gives a slope for every weight; a ReLU gate keeps the slope for "on" units and zeroes the rest.
6. **Trust it (the calibration weight):** a numerical gradient check catches a wrong value, and "gradient shape must equal knob shape" catches a flipped transpose — both cheap, both before you train.
7. **When it misbehaves (the game of telephone):** repeated multiplication makes gradients vanish, explode, or die — fixed by ReLU + normalization, clipping + good init, and Leaky ReLU / GELU + He init.

#### Cheat-sheet — every new word from today, in plain English
%%% jargon
loss | one number saying how wrong a guess was; the backward pass shrinks it
gradient | the slope for each knob — which way the loss grows and by how much; step the opposite way
partial derivative | the slope of the loss for ONE knob with the others held still; stack them → the gradient
chain rule | multiply the local slope of each step along a path to get the far-reaching effect
computational graph | the forward pass drawn as boxes (ops) and arrows (values); slopes flow back along the same arrows
local gradient | one operation's own slope, cached during the forward pass
upstream gradient | the slope arriving from the layer closer to the loss; multiply it by the local gradient
delta (δ) | the error signal at a layer — the loss's slope w.r.t. that layer's pre-bend score; backprop carries it layer to layer
backpropagation | one reverse sweep that reuses cached forward values to get a gradient for every knob
activation derivative | a bend's own slope; ReLU's is 1 where the input was positive, 0 where negative (a gate)
numerical gradient | a slow brute-force slope from nudging a knob ±h; used only to check the fast one
He initialization | a ReLU-friendly random weight start (spread ~2/n_in) so units aren't born dead
%%%

#### The three quiet stalls at a glance
%%% table
:: Failure :: Cause :: Named fix
Vanishing gradient :: slopes < 1 multiply toward 0 :: ReLU + normalization
Exploding gradient :: slopes / weights > 1 multiply huge :: gradient clipping + careful init
Dead ReLU :: unit stuck negative passes 0 gradient :: Leaky ReLU / GELU + He init
%%%

Keep this page. Day 3 is where you finally *use* these gradients: the update rule `w ← w − learning_rate × gradient`, run over batch after batch, and you watch the loss actually fall.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions with instant feedback. If one trips you up, that's a cue to scroll back — not a failing grade.
%%% quiz
q: What is the gradient of the loss with respect to a weight? | a:2 | The loss value itself | The model's final guess | A slope saying how much the loss changes if you nudge that weight, and which way | The number of layers | fb: The gradient is a slope per knob — how fast the loss changes per tiny nudge, pointing toward steepest increase. You step the opposite way to shrink the loss.
q: Backpropagation carries the error signal from the loss back to the first layer. What mathematical rule makes this work? | a:1 | The argmax rule | The chain rule — multiply the local slope of each step along the path | Matrix inversion | Random sampling | fb: The chain rule multiplies each step's local slope along the path, so blame from the far-away loss reaches an early knob by one product of cached slopes.
q: When a gradient passes backward through a ReLU, what happens to it? | a:2 | It is doubled everywhere | It is turned into a probability | It passes through for units that were positive on the forward pass and is zeroed for the rest | It is reversed in sign | fb: ReLU's activation derivative is 1 where the input was positive and 0 where negative — an element-wise gate. That's why backprop needs the forward pass's saved values (which units were on).
q: You wrote a backward pass by hand and want to check it before training. Which is a correct, cheap check? | a:1 | Train for 100 epochs and hope | Compare the analytic gradient to a numerical one (nudge ±h), and confirm dL/dW has the same shape as W | Count the number of layers | Set the learning rate to zero | fb: A numerical gradient catches a wrong value; the shape-match invariant (dL/dW same shape as W) catches a flipped transpose. Both are cheap and run before wasting a training run.
%%%

@@@ produce id=produce tag="Produce" title="Wire a backward pass and gradient-check it" gotit="Done"
Time to see the coach with your own eyes — a real backward pass, checked for bugs. **Before you write any code, predict two things:** (1) after you push a gradient back through a ReLU, how many of your `[0.4, -0.7, 0.9, -0.2]`-style entries will survive (stay non-zero)? and (2) will your hand-derived (analytic) gradient *match* the slow numerical one — yes or no? Write your two guesses down, then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py`. Build a tiny 2-layer network in NumPy on made-up data: forward pass `h = relu(x @ W1 + b1)`, `out = h @ W2 + b2`, and a squared loss `L = ((out - target)**2).mean()`. **Save** `x` and the ReLU mask (`z1 > 0`) on the way forward. Then write the backward pass by hand: seed δ at the output, use `dL/dW = δ · (saved input)ᵀ`, `dL/db = δ`, and push δ back through layer 2, through the ReLU gate (multiply by the saved mask), and into layer 1. Print the shape of each gradient and confirm `dW1.shape == W1.shape`. Finally, **gradient-check** one weight: recompute the loss at `w+h` and `w-h`, form `(L(w+h) - L(w-h)) / (2h)`, and print it next to your analytic gradient — they should match. Run with `python3 sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5 Day 2 artifact.

Create sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py that, with a comment on each step, wires a backward pass BY HAND for a tiny 2-layer MLP in NumPy (no autograd, no frameworks):
1. Make small random data x and a target. Init W1,b1,W2,b2 small and random.
2. Forward pass: z1 = x @ W1 + b1; h = relu(z1); out = h @ W2 + b2; loss = ((out - target)**2).mean(). SAVE x, z1 (or the mask z1>0), and h for the backward pass.
3. Backward pass by hand using the chain rule: seed delta at the output from the loss derivative; compute dW2 = h.T @ delta2, db2 = delta2.sum(0); propagate delta1 = (delta2 @ W2.T) * (z1 > 0)  # the ReLU gate; compute dW1 = x.T @ delta1, db1 = delta1.sum(0). Print the shape of every gradient and assert dW1.shape == W1.shape and dW2.shape == W2.shape.
4. Gradient-check ONE entry of W1: numerical = (loss(W1 with that entry +h) - loss(W1 with that entry -h)) / (2*h) for h=1e-5; print it next to the analytic dW1 entry and show they match to ~5 decimals.
5. Bonus: push the vector [0.4,-0.7,0.9,-0.2] back through a ReLU whose forward inputs were [2,-1,3,-0.5] and print how many entries survive.
Then run it and paste the analytic-vs-numerical match and the surviving-entries count at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Pushing the gradient back through the ReLU, exactly the entries whose forward input was positive survive — the rest become 0. Did you predict how many?
- Your analytic gradient and the numerical one **match** to several decimals — proof your hand-wired backward pass has no sign or transpose bug. Was your yes/no right?
- Every gradient has the **same shape** as the knob it corrects: `dW1.shape == W1.shape`. If it didn't, you'd have a flipped transpose to hunt.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m04-first-model-mlp/day-02-backward-pass/log.md`: (1) what the chain rule multiplies to carry blame backward; (2) why backprop must save values from the forward pass; (3) the two cheap checks that catch a gradient bug.

@@@ fin
