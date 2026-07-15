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
  - {topic: gradient is the partial derivative that points uphill, keywords: [gradient, partial derivative, steepest increase, step against, uphill]}
  - {topic: chain rule multiplies local slopes along the path, keywords: [chain rule, multiply, local slope, along the path]}
  - {topic: per-layer local derivatives compose, keywords: [local derivative, activation, linear step, compose]}
  - {topic: backpropagation is one reverse sweep reusing cached forward values, keywords: [backpropagation, reverse sweep, reuse, cached, forward pass]}
  - {topic: gradient-descent update rule and learning rate size, keywords: [update rule, learning rate, "param = param -", overshoot, crawls]}
  - {topic: naive numerical finite-difference gradient ancestor, keywords: [numerical gradient, finite difference, one forward pass per parameter, noisy]}
  - {topic: gradient checking against finite differences, keywords: [gradient check, verify, sign, transpose]}
  - {topic: vanishing gradients cause and ReLU remedy, keywords: [vanishing gradient, saturating, relu, slope of 1]}
  - {topic: exploding gradients cause and gradient clipping remedy, keywords: [exploding gradient, gradient clipping, cap, smaller learning rate]}
  - {topic: capability limit local minima saddle points and no gradient path, keywords: [local minimum, saddle point, non-convex, non-differentiable, no gradient path]}
require_artifact: true
---

@@@ hero
@lede Yesterday you built a machine that looks at a messy handwritten "7" and makes a guess. But a fresh machine guesses blindly — about one in ten right. Here's the puzzle: how does it get *better*? Practice alone teaches nothing. What teaches is practice **plus correction** — after every wrong guess, someone who says not just "wrong" but "you leaned too far left, ease off *this* much." Today you build that corrector, by hand.
@goal By the end you can explain what the loss, the gradient, the chain rule, and backpropagation each are; walk a correction backward through a 2-layer network in your head; write the one-line update rule; and say how to catch a gradient bug before it wastes a training run.

@@@ concept id=c1 tag="The score" title="One number that says how wrong we were" gotit="Got the loss"
Practice needs a scorekeeper. Before a machine can improve, it needs one honest number that says how badly this guess missed. That number is the [[loss||A single number measuring how wrong the model's output was on an example — small means close, large means far off. It is the one signal the whole backward pass tries to shrink.]] — a **single scalar** (just one value, not a list) that shrinks toward zero as the guess gets better.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="The network turns pixels into ten scores, then a loss box collapses those scores plus the true answer into one number that measures how wrong the guess was"><g font-family="monospace" font-size="12" text-anchor="middle"><rect x="20" y="50" width="70" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="55" y="65" fill="#1a5c38" font-weight="bold">image</text><text x="55" y="79" fill="#6B645E" font-size="10">(1,784)</text><text x="115" y="70" fill="#6B645E" font-size="10">→ network →</text><rect x="180" y="50" width="80" height="34" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="220" y="65" fill="#C93B3B" font-weight="bold">10 scores</text><text x="220" y="79" fill="#6B645E" font-size="10">the guess</text><text x="290" y="70" fill="#6B645E" font-size="10">+ true digit</text><rect x="350" y="50" width="86" height="34" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="2.5"/><text x="393" y="65" fill="#8A6D3B" font-weight="bold">loss</text><text x="393" y="79" fill="#6B645E" font-size="10">one number</text><text x="470" y="72" fill="#6B645E">shrink!</text><text x="260" y="112" fill="#6B645E" font-size="11">everything the network did collapses into a single "how wrong" score</text></g></svg>
%%%

**Think of the loss like a golf score.** After each swing, one number tells you how far the ball still is from the hole. You don't need the whole story of the swing — just that one distance, and you know whether you're doing better or worse. **What the golf-score picture gets right:** the loss squeezes a whole complicated output into a single "how far off" number you can try to shrink, swing after swing (guess after guess). **Where it breaks down:** in golf you only get the score; here the loss also secretly carries the *shape* of the mistake, and that shape is what tells each weight which way to move — which is the entire rest of this lesson.

#### Why one number is the whole trick
That single number is what makes correction possible. If the loss went *up* after a change, the change was bad; if it went *down*, good. Learning is just: keep changing the weights so this one number keeps going down. Practice with correction is the spine of the module — the loss is the correction's yardstick.

@@@ concept id=c2 tag="The nudge" title="The gradient: which way to nudge each weight" gotit="Got the gradient"
Now the key question. The loss is one number, but the network has about 100,000 weights. For *each* weight we want to know: **if I nudge you up a hair, does the loss go up or down, and how sharply?** That answer — one per weight — is the [[gradient||The set of partial derivatives of the loss with respect to every weight and bias. For one weight it says how fast the loss changes as that weight changes, and its sign says which way is uphill.]].

For a single weight, its piece of the gradient is a **partial derivative**: the slope of the loss as you slide *that one* weight while holding all the others still. A positive slope means "raising this weight raises the loss" — uphill. A negative slope means the loss falls when you raise it.

%%% viz src=../../viz/gradient-descent.html title="the loss as a hill; the gradient is the slope under your feet" caption="interactive — the arrow shows the gradient (steepest-increase) direction; learning steps the opposite way, downhill"
%%%

**Picture a hiker in fog on a hillside.** They can't see the valley, but they can feel the ground tilt under their boots. The gradient is exactly that felt tilt: the direction of **steepest increase** — the way *up*. **What the fog-hiker picture gets right:** you only need the local slope where you're standing, not a map of the whole mountain, to know which way is downhill. **Where it breaks down:** a real hiker feels one 2-D slope; here the "slope" lives in 100,000 dimensions at once — one number per weight — which no person can picture, but the math handles it the same way.

#### The one rule you must remember
The gradient points **uphill** — toward *more* loss. We want *less* loss. So we always **step against** the gradient. That single fact ("compute the uphill direction, then walk the other way") is what every training run on Earth is doing. Getting the gradient for every weight, cleanly, is the job of the rest of today.

@@@ concept id=c3 tag="The chain" title="The chain rule: multiply the slopes along the path" gotit="Got the chain rule"
A weight deep in the network doesn't touch the loss directly. It changes a hidden value, which changes the next value, which changes the score, which changes the loss — a **chain** of steps. So how does a nudge at the start ripple all the way to the end? By the [[chain rule||To find the slope of the whole chain, multiply the local slope of each step along the path. It is the engine that carries a weight's effect through every layer to the loss.]]: to get the total slope, **multiply the local slope of every step along the path**.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A chain of three gears labeled with local slopes 2, then 3, then one half; the total slope is the product 2 times 3 times one half equals 3"><g font-family="monospace" font-size="12" text-anchor="middle"><text x="60" y="30" fill="#6B645E">weight</text><circle cx="90" cy="70" r="26" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="90" y="68" fill="#5E5191" font-weight="bold">×2</text><text x="90" y="82" fill="#6B645E" font-size="9">step 1</text><text x="150" y="66" fill="#6B645E">→</text><circle cx="200" cy="70" r="26" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="200" y="68" fill="#5E5191" font-weight="bold">×3</text><text x="200" y="82" fill="#6B645E" font-size="9">step 2</text><text x="260" y="66" fill="#6B645E">→</text><circle cx="310" cy="70" r="26" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="310" y="68" fill="#5E5191" font-weight="bold">×½</text><text x="310" y="82" fill="#6B645E" font-size="9">step 3</text><text x="370" y="66" fill="#6B645E">→</text><rect x="405" y="52" width="94" height="34" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><text x="452" y="68" fill="#8A6D3B" font-weight="bold">loss</text><text x="452" y="82" fill="#6B645E" font-size="9">total ×3</text><text x="260" y="116" fill="#6B645E" font-size="11">total slope = 2 × 3 × ½ = 3  —  multiply the local slopes along the path</text></g></svg>
%%%

**Think of a chain of gears.** If the first gear turns twice for every turn of its handle, and the second turns three times per turn of the first, then one turn of the handle spins the second gear six times. You multiply the gear ratios. **What the gear-chain picture gets right:** each step has its own local ratio (slope), and the effect of the whole chain is those local ratios multiplied together. **Where it breaks down:** gears only multiply positive whole-number ratios; real slopes can be fractions or negative — a negative slope flips the direction of the push, and that sign matters enormously later.

#### Local slopes are the pieces you multiply
Each step in the network has a simple **local derivative** — its own little slope you can write down without thinking about the whole network:
- The **linear step** `z = x @ W + b`: nudging a weight changes `z` in proportion to the input `x` that it multiplies. So the local slope of the linear step is just the input.
- The **activation step** (ReLU): its local slope is `1` where the input was positive and `0` where it was negative — a gate that's either open or shut.

The chain rule says: to get a weight's gradient, **multiply these local slopes together, one per step, from the loss back to that weight.** That is the whole engine. Everything else today is bookkeeping for doing it efficiently.

@@@ concept id=c4 tag="One sweep" title="Backprop: one reverse pass that reuses the forward values" gotit="Got backprop"
Here's the clever part. You *could* apply the chain rule separately for each of the 100,000 weights — but that repeats the same multiplications a hundred thousand times. Instead we do [[backpropagation||Computing every weight's gradient in a single pass that walks from the loss back to the input, reusing the values already cached during the forward pass instead of recomputing a slope per weight.]]: **one reverse sweep** through the network, from the loss back to the input, that computes every gradient in a single trip by **reusing the values cached during the forward pass**.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Top row: forward pass left to right through image, hidden, logits, loss, with cached values saved. Bottom row: backward sweep right to left carrying blame back through the same layers, reusing the cached values"><g font-family="monospace" font-size="11" text-anchor="middle"><text x="20" y="28" fill="#2D8B55" font-weight="bold" text-anchor="start">FORWARD → (save z1, h, probs)</text><rect x="30" y="36" width="66" height="26" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="63" y="53" fill="#1a5c38">image</text><text x="112" y="52" fill="#2D8B55">→</text><rect x="128" y="36" width="66" height="26" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="161" y="53" fill="#1a5c38">hidden</text><text x="210" y="52" fill="#2D8B55">→</text><rect x="226" y="36" width="66" height="26" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="259" y="53" fill="#1a5c38">logits</text><text x="308" y="52" fill="#2D8B55">→</text><rect x="324" y="36" width="66" height="26" rx="4" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="357" y="53" fill="#8A6D3B">loss</text><text x="20" y="104" fill="#C93B3B" font-weight="bold" text-anchor="start">← BACKWARD (reuse the saved values)</text><rect x="30" y="112" width="66" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="63" y="129" fill="#C93B3B">dW1</text><text x="112" y="128" fill="#C93B3B">←</text><rect x="128" y="112" width="66" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="161" y="129" fill="#C93B3B">dh</text><text x="210" y="128" fill="#C93B3B">←</text><rect x="226" y="112" width="66" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="259" y="129" fill="#C93B3B">dW2</text><text x="308" y="128" fill="#C93B3B">←</text><rect x="324" y="112" width="66" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="357" y="129" fill="#C93B3B">dloss</text></g></svg>
%%%

**Picture unpacking groceries, then re-packing the bags.** On the way in (forward) you set each bag down in order and remember where it went. On the way back out (backward) you don't re-derive the whole trip — you just walk the bags back in reverse, one at a time, using the layout you already remember. **What the grocery picture gets right:** the backward trip reuses what the forward trip already worked out (the cached values), so it costs about the same as one forward pass, not a hundred thousand. **Where it breaks down:** groceries don't multiply on the way back; here each step multiplies the incoming blame by its local slope (the chain rule) as it passes it along.

#### Backward is the forward pass, mirrored
Walk it once, right to left. The "blame" flowing into a layer times that layer's local slope gives the blame for the layer before it — and the input the layer saw (cached going forward) turns that blame into the weight's gradient:
- blame at the output → gradient of the last layer's weights (uses the cached hidden values `h`)
- hand the blame back through `W2` → blame on the hidden units
- through the ReLU gate: blame passes only where the unit was on (the cached `z1 > 0`)
- that gated blame → gradient of the first layer's weights (uses the cached input `x`)

One sweep. Every gradient. That reuse — **not** recomputing a derivative per weight — is the entire reason backprop is fast enough to train real models.

@@@ concept id=c5 tag="The step" title="The update rule: walk downhill by the learning rate" gotit="Got the update"
You have the gradient. Now you *use* it. The correction is one line — the [[gradient-descent update rule||param = param − learning_rate × gradient. Each weight steps a little in the downhill direction; the learning rate sets the step size.]]:

%%% formula
expr: param = param − learning_rate × gradient
note: gradient points uphill, so the minus sign walks downhill; learning_rate sets how big each step is
%%%

The gradient gives the *direction* (well, the uphill direction — the minus sign flips it downhill). The [[learning rate||A small positive number that scales the step. Too large and the weights overshoot and the loss diverges; too small and training crawls.]] gives the *size* of the step. That's it: nudge every weight a little against its gradient, and the loss ticks down.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A U-shaped loss valley. Left: tiny steps crawl slowly toward the bottom. Middle: right-sized steps reach the bottom. Right: huge steps overshoot back and forth and climb out of the valley."><g font-family="monospace" font-size="10" text-anchor="middle"><path d="M15 30 Q60 130 105 30" fill="none" stroke="#B8B0C8" stroke-width="2"/><circle cx="35" cy="72" r="4" fill="#2D8B55"/><circle cx="45" cy="88" r="4" fill="#2D8B55"/><circle cx="54" cy="100" r="4" fill="#2D8B55"/><text x="60" y="145" fill="#8A6D3B">too small: crawls</text><path d="M185 30 Q230 130 275 30" fill="none" stroke="#B8B0C8" stroke-width="2"/><circle cx="200" cy="55" r="4" fill="#2D8B55"/><circle cx="222" cy="105" r="4" fill="#2D8B55"/><circle cx="230" cy="118" r="5" fill="#1a5c38"/><text x="230" y="145" fill="#1a5c38" font-weight="bold">just right</text><path d="M355 30 Q400 130 445 30" fill="none" stroke="#B8B0C8" stroke-width="2"/><circle cx="372" cy="60" r="4" fill="#C93B3B"/><circle cx="430" cy="55" r="4" fill="#C93B3B"/><circle cx="360" cy="35" r="4" fill="#C93B3B"/><path d="M372 60 L430 55 L360 35" fill="none" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3 2"/><text x="400" y="145" fill="#C93B3B">too large: overshoots</text></g></svg>
%%%

**Think of adjusting a shower's temperature.** Turn the knob too far and you scald yourself, then freeze, then scald — overshooting past comfortable every time. Turn it a hair at a time and you creep toward perfect but it takes forever. **What the shower-knob picture gets right:** the learning rate is the size of each turn, and there's a "just right" that lands you at the target fastest. **Where it breaks down:** a shower has one knob; here you turn 100,000 knobs at once, each by its own gradient — but they all share the *same* learning rate, which is why picking it well matters so much.

#### The learning rate is a knob you tune
Set the learning rate **too large** and the steps overshoot the valley and the loss diverges (grows without bound). Set it **too small** and training crawls — technically improving, but so slowly you'll give up first. Finding a good learning rate is one of the first knobs you'll ever tune. (It stays a *knob* on this update rule; fancier optimizers that reshape the step — momentum, Adam — come in a later training-loop day.)

@@@ concept id=c6 tag="The check" title="The slow, honest ancestor — and how to use it as a test" gotit="Got the gradient check"
Before backprop existed, how did people get gradients at all? The crude, obvious way: the [[numerical gradient||A gradient estimated by finite differences: nudge one weight by a tiny amount, re-run the forward pass, and see how much the loss changed. Slow (one forward pass per weight) and slightly noisy, but almost impossible to get wrong.]] — also called **finite differences**. To find how the loss depends on one weight, you literally nudge that weight by a tiny amount `ε`, re-run the whole forward pass, and measure how much the loss changed.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="A curve with two dots a tiny epsilon apart on the x-axis; the rise over run of the line between them approximates the slope. Formula: gradient is approximately loss at w plus epsilon minus loss at w minus epsilon, over two epsilon."><g font-family="monospace" font-size="11" text-anchor="middle"><path d="M40 110 Q160 20 300 90" fill="none" stroke="#7C6DAA" stroke-width="2"/><circle cx="150" cy="52" r="4" fill="#C93B3B"/><circle cx="190" cy="46" r="4" fill="#C93B3B"/><line x1="150" y1="52" x2="190" y2="46" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="3 2"/><text x="150" y="128" fill="#6B645E">w−ε</text><text x="190" y="128" fill="#6B645E">w+ε</text><text x="170" y="30" fill="#C93B3B">slope ≈ rise / run</text><text x="410" y="55" fill="#3A342E" font-size="10">grad ≈</text><text x="410" y="74" fill="#3A342E" font-size="10">loss(w+ε) − loss(w−ε)</text><line x1="360" y1="80" x2="475" y2="80" stroke="#3A342E" stroke-width="1"/><text x="410" y="96" fill="#3A342E" font-size="10">2ε</text></g></svg>
%%%

**Think of measuring how steep a ramp is with a ruler.** You step a tiny bit forward, measure how much you rose, and divide rise by run. Take a smaller step and your estimate gets sharper — but too tiny a step and your ruler's own wobble (rounding error) starts to dominate. **What the ruler picture gets right:** you can measure a slope from two nearby points without knowing any calculus at all. **Where it breaks down:** a ramp is one slope; to measure the gradient of a network this way you'd re-run the forward pass **once per weight** — 100,000 forward passes for one gradient — which is why backprop's single reverse sweep replaced it.

#### From ancestor to debugging tool: gradient checking
The numerical gradient is far too slow to *train* with. But it is almost impossible to get *wrong*, which makes it the perfect **test**. This is [[gradient checking||Verifying a hand-derived (analytic) backprop gradient against the slow numerical/finite-difference gradient on a tiny network, to catch sign or transpose bugs before a real training run.]]: compute a weight's gradient both ways — your fast hand-wired backprop, and the slow finite-difference estimate — and confirm they agree to several decimal places. If they don't, your backprop has a **sign or transpose bug**, and you just caught it on a tiny network in seconds instead of after a wasted overnight run.

!!! c-info ⚖️
<b>Trade-off — analytic vs. numerical gradient.</b> The hand-wired (analytic) gradient is fast (one backward sweep) but easy to get subtly wrong. The numerical gradient is trustworthy but painfully slow (one forward pass per weight). Standard practice: use the numerical one only as a one-time *check* on a small net, then trust the fast analytic one for real training.
!!!

@@@ concept id=c7 tag="When it breaks" title="Two silent gradient failures — each with its fix" gotit="Got the failure modes"
The backward pass can run without error and still fail — the gradients come out useless, and nothing crashes. Two failure modes are worth knowing today, and each has a clear **cause** and a named **remedy**.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Two failure boxes. One: vanishing gradients, chained saturating slopes multiply toward zero, fixed by ReLU which keeps a slope of one. Two: exploding gradients, repeated large slopes compound, fixed by gradient clipping plus a smaller learning rate."><g font-family="monospace" font-size="11"><rect x="20" y="18" width="480" height="52" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="32" y="37" fill="#C93B3B" font-weight="bold">Vanishing gradients</text><text x="32" y="53" fill="#6B645E">chained sigmoid/tanh slopes (each &lt; 1) multiply → 0 across layers</text><text x="32" y="66" fill="#2D8B55">FIX: ReLU — slope of 1 for positive inputs, so slopes stop collapsing</text><rect x="20" y="82" width="480" height="52" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="32" y="101" fill="#8A6D3B" font-weight="bold">Exploding gradients</text><text x="32" y="117" fill="#6B645E">repeated large slopes/weights multiply → blow up to huge numbers</text><text x="32" y="130" fill="#2D8B55">FIX: gradient clipping (cap the size) + a smaller learning rate</text></g></svg>
%%%

**Think of a message whispered down a long line of people.** If everyone whispers a little quieter than they heard, the message fades to silence by the end. If everyone shouts a little louder, it becomes a deafening roar. **What the whisper-line picture gets right:** the chain rule *multiplies* slopes step after step, so a consistent shrink drives the signal to nothing and a consistent growth blows it up. **Where it breaks down:** a whisper line has one message; here it's a whole vector of gradients, and the shrink or growth depends on the specific activation and weight sizes at each layer.

#### 1 · Vanishing gradients
**Cause:** older activations like sigmoid and tanh have slopes smaller than 1 almost everywhere (they *saturate* — flatten out — at the ends). Chain enough of them and the multiplied local slopes shrink toward **0**, so early layers get a gradient of nearly nothing and stop learning. **Remedy:** use [[ReLU||max(0, z): its slope is exactly 1 for positive inputs, so chaining many of them does not shrink the gradient toward zero the way saturating sigmoid/tanh slopes do.]] as the activation — its slope is a clean `1` for positive inputs, so chained slopes no longer collapse. (ReLU is the default bend you met on Day 1; this is *why* it became the default.)

#### 2 · Exploding gradients
**Cause:** the mirror image — when repeated local slopes or weight sizes are larger than 1, the multiplied chain **blows up** to enormous numbers, and a single update flings the weights off to nonsense. **Remedy:** [[gradient clipping||Before applying the update, cap the gradient's magnitude to a fixed ceiling, so one huge gradient cannot fling the weights off. Paired with a smaller learning rate for exploding-gradient problems.]] — cap the gradient's size before the update so no single step can be catastrophic — plus a **smaller learning rate**.

!!! c-warn ⚠️
<b>How a senior engineer catches these:</b> none of the three (vanishing, exploding, or a plain wrong learning rate) throws an error. You spot them by <b>watching the gradient magnitudes and the loss curve</b> — gradients drifting to ~0, or spiking to huge values, or a loss that climbs instead of falling — not by waiting for a crash. A silent failure is still a failure.

@@@ concept id=c8 tag="The ceiling" title="What backprop still can't promise you" gotit="Got the ceiling"
End with honesty about what you built. Backprop computes a perfect *local* answer: which way is downhill, right here. But downhill-from-here does not always lead to the lowest point on the whole map. The loss surface of a real network is **non-convex** — bumpy, with many valleys — so gradient descent can settle into a [[local minimum||A valley that is lower than everything nearby but not the lowest valley overall. Gradient descent can get stuck here because every direction out of it is uphill, even though a deeper valley exists elsewhere.]] or stall on a [[saddle point||A flat spot that goes downhill in some directions and uphill in others, so the gradient is near zero and progress stalls even though it is not a true minimum.]] and never find the global best.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="A wavy loss curve with two valleys. A ball rolls into the shallow left valley (a local minimum) and stops, while the deeper right valley (the global minimum) sits lower and unreached."><g font-family="monospace" font-size="10" text-anchor="middle"><path d="M20 40 Q90 120 160 70 Q230 20 300 110 Q370 150 440 60 Q470 40 500 55" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle cx="130" cy="92" r="7" fill="#C93B3B"/><text x="130" y="118" fill="#C93B3B" font-weight="bold">stuck: local min</text><circle cx="330" cy="128" r="7" fill="#2D8B55"/><text x="345" y="118" fill="#1a5c38" font-weight="bold">true global min ↓</text><text x="260" y="20" fill="#6B645E">the loss surface is bumpy (non-convex) — downhill ≠ lowest</text></g></svg>
%%%

**Think of rain falling on a hilly field.** Each drop rolls straight downhill — the locally correct move every time — yet drops land in all sorts of puddles, and only some reach the deepest pond. **What the rain picture gets right:** following the local slope is always locally sensible, but a bumpy landscape means "always go down" can trap you in a shallow puddle. **Where it breaks down:** in practice, huge networks have so many dimensions that most traps are shallow and good-enough solutions are easy to reach — so this limit matters less than beginners fear, but it is real, and it is why there is no guarantee of the global minimum.

#### The other hard wall
Backprop also **cannot learn through a step that has no slope.** If your forward graph contains a hard, non-differentiable operation — a true on/off step with a flat-then-vertical jump — there is **no gradient path** through it, so no correction can flow back past it. That is why the whole network is built from smooth-ish, differentiable pieces. Backprop hands you a descent *direction*, not a guarantee of the destination — and only where a gradient path actually exists.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions with instant feedback. If one trips you up, that's a cue to scroll back — not a failing grade.
%%% quiz
q: The gradient of the loss for a weight points in the direction of steepest *increase*. So to lower the loss, the update rule does what? | a:2 | Steps along the gradient (uphill) | Sets the weight to zero | Steps against the gradient (downhill) | Multiplies the weight by the loss | fb: The gradient points uphill, toward more loss. We want less, so we step *against* it: param = param − learning_rate × gradient.
q: Why is backpropagation one reverse sweep instead of applying the chain rule separately per weight? | a:1 | It uses a bigger learning rate | It reuses the values cached during the forward pass, so it costs about one pass instead of one-per-weight | It skips the activation layers | It only trains half the weights | fb: Backprop caches the forward values (z1, h, probs) and walks blame back once, reusing them — so all ~100k gradients come from a single reverse pass, not 100k separate chains.
q: Your hand-wired gradient disagrees with a finite-difference (numerical) gradient in the 1st decimal place. What is this telling you? | a:2 | The learning rate is too small | Training is finished | There is likely a sign or transpose bug in your backprop | The loss is a scalar | fb: That is exactly what gradient checking is for: the numerical gradient is slow but trustworthy, so a disagreement means your fast analytic backprop has a bug (often a sign flip or a misplaced transpose) — caught before a wasted run.
q: Chained sigmoid/tanh slopes multiply toward 0 across layers so early layers stop learning. What is this failure, and its usual remedy? | a:1 | Not a real problem | Vanishing gradients — remedy: use ReLU (slope of 1 for positive inputs) | Exploding gradients — remedy: a bigger learning rate | Overfitting — remedy: more data | fb: Slopes below 1 multiplied many times shrink to ~0: vanishing gradients. ReLU's slope is a clean 1 for positive inputs, so chained slopes stop collapsing — which is why ReLU became the default bend.
%%%

@@@ produce id=produce tag="Produce" title="Wire the backward pass, then check it against the ancestor" gotit="Done"
Time to build the corrector with your own hands — wiring it yourself is what makes it stick. **Before you write any code, predict two things:** (1) when you compare your hand-wired gradient for `W2` against a slow numerical (finite-difference) gradient, to how many decimal places will they agree — 1, 6, or none? And (2) if you *deliberately* break the backward pass (drop the ReLU gate, so blame leaks through units that were off), do you think the numerical check will still pass? Write your two guesses down, then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py`. Run one forward pass of the 784→128→10 MLP on a small batch, **saving** `z1, h, probs`. Then walk the backward pass: blame at the output, `dW2 = h.T @ dlogits`, hand blame back `dh = dlogits @ W2.T`, through the ReLU gate `dz1 = dh * (z1 > 0)`, and `dW1 = x.T @ dz1`. Print every gradient's shape and confirm each matches its weight's shape. Finally, write a **numerical-gradient check** on two entries of `W2` — nudge by a tiny `ε`, use `(loss(w+ε) − loss(w−ε)) / (2ε)` — and confirm it matches your analytic gradient. Run with `python3 sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5 Day 2 artifact.

Create sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py that, with a comment on each step, hand-wires the backward pass of a 784->128->10 MLP in NumPy:
1. Forward pass on one small batch, SAVING z1, h, probs; print all shapes.
2. Backward pass reusing those cached values: dlogits = probs - y_true; dW2 = h.T @ dlogits; dh = dlogits @ W2.T; dz1 = dh * (z1 > 0); dW1 = x.T @ dz1. Print each gradient shape and confirm it equals its weight's shape.
3. The update rule: show W2 = W2 - learning_rate * dW2 for a small learning rate, and print how the loss changes.
4. Numerical gradient check (the finite-difference ancestor): nudge two entries of W2 by eps and confirm (loss(w+eps)-loss(w-eps))/(2*eps) matches the analytic dW2 to ~6 decimals.
5. As a bonus, drop the ReLU gate (dz1 = dh) and show the numerical check now FAILS — a silent gradient bug the check caught.
Then run it and paste the max abs difference (good vs. broken) at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Every gradient has the *same shape* as its weight: `dW1` is `(784, 128)`, `dW2` is `(128, 10)`. A mismatch means a transpose is in the wrong place.
- Your hand-wired `dW2` agrees with the numerical gradient to roughly **6 decimal places** — proof the chain rule was applied correctly. Was your guess close?
- Drop the ReLU gate and the check *breaks*: the numbers no longer match, because blame leaked back through units that were off. Did you predict that?
- The thing worth noticing: nothing here fought back with an error. The only thing that told you the gradient was right was the numerical check — that is the habit to keep.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m04-first-model-mlp/day-02-backward-pass/log.md`: (1) what the loss, the gradient, and the chain rule each are, in one sentence each; (2) why backprop is one reverse sweep and not one derivative per weight; (3) the max difference between your analytic and numerical gradient, and what broke when you dropped the ReLU gate.

@@@ fin
