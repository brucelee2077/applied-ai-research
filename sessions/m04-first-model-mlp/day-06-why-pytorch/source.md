---
quest_id: wf5-d06-mirror
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 6 — Why PyTorch Mirrors NumPy"
module_label: "Module 5 · Train · Day 6"
title: "Why PyTorch"
subtitle: "Mirrors NumPy"
brand_sub: "Foundations · M5 Day 6"
spine: "practice: a machine learns to read by practicing on examples and being corrected"
nav_prev_href: "../day-05-pytorch-version/lesson.html"
nav_prev_label: "The Same Model in PyTorch"
nav_next_href: "../review.html"
nav_next_label: "Module 5 · Review Gate"
fin_title: "Module 5 · Day 6 complete! 🏆"
fin_body: "You did it — and you finished all of <b>Module 5</b>. You now know <i>why</i> a deep-learning framework like <b>PyTorch</b> exists at all: as models grow, hand-writing the backward pass for every new shape becomes slow and bug-prone, so the framework's whole job is to remove that repetitive, error-prone work. You understand its one central trick — <b>autograd</b>: PyTorch records every tensor operation into a <b>computation graph</b> as the forward pass runs, then walks that graph backward when you call <b>loss.backward()</b>, filling every <code>.grad</code> automatically — the exact hand-derived backward pass you wrote on Day 2, done for you. You can name the four layers you stand on: the <b>tensor + autograd</b> substrate underneath, <b>nn.Module</b> holding your parameters, a <b>loss function</b>, and an <b>optimizer</b> that owns the update rule — and you know the canonical loop <b>zero_grad → forward → loss → backward → step</b>. You know the four silent bugs and their one-line remedies: forgetting <b>zero_grad()</b> (gradients accumulate), calling <b>backward()</b> twice on a freed graph, evaluating without <b>torch.no_grad()</b>, and mixing CPU/GPU tensors. And you know the honest limit: autograd computes <i>correct</i> gradients for <i>whatever</i> graph you build — it never chooses your architecture, loss, or learning rate, and it will faithfully train a badly-designed model into a badly-trained one without a single complaint. That judgment stays with you. That's exactly why this module made you build it all by hand first.<br>Take a breath. You now own the tool every frontier lab trains with — and you own it without fear, because you can see straight through it to the numpy you wrote yourself. Next up: the <b>Module 5 Review Gate</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: why a deep-learning framework exists at all, keywords: [why a framework exists, hand-writing the backward pass, error-prone, unscalable, removes the repetitive, remove that repetitive]}
  - {topic: tensor is a numpy array that tracks operations and lives on a device, keywords: [tensor, numpy array, tracks operations, torch.tensor, dtype, shape, gpu, lives on a device]}
  - {topic: autograd is the core mechanism that records operations and walks them backward, keywords: [autograd, automatic differentiation, records every operation, walks it backward, computes gradients automatically, replaces the by-hand backward]}
  - {topic: the computation graph is built as the forward runs, keywords: [computation graph, the tape, built as the forward runs, records the operations, walks the graph backward]}
  - {topic: requires_grad marks a tensor as tracked and .grad stores its gradient after backward, keywords: [requires_grad, .grad, tracked, stored in .grad, maps the manual, dL/dW to the automatic version]}
  - {topic: loss.backward() is the single call that replaces the whole hand-written backward chain and accumulates into every leaf .grad, keywords: [loss.backward, single call, replaces the whole hand-written backward, accumulates gradients, every leaf tensor]}
  - {topic: the four PyTorch abstraction layers nn.Module loss optimizer and the tensor autograd substrate, keywords: [four abstraction layers, nn.Module, loss function, optimizer, tensor and autograd substrate, four layers]}
  - {topic: the canonical PyTorch training loop shape maps to the from-scratch loop, keywords: [same five moves, line by line, every line on the left has a twin, clear the scoreboard]}
  - {topic: nn.Module and nn.Linear package weights and biases as a reusable layer, keywords: [nn.Module, nn.linear, packages weights, parameter bookkeeping, auto-registers, no longer track a dict of arrays, .parameters(), feeds them to the optimizer]}
  - {topic: parameters are the learnable tensors the optimizer updates, keywords: [parameters, model.parameters, learnable tensors, the optimizer will update]}
  - {topic: torch.optim SGD is the update-rule abstraction, keywords: [torch.optim, SGD, update-rule abstraction, w := w - lr, applies the update, optimizer.step]}
  - {topic: optimizer.zero_grad clears accumulated gradients, keywords: [optimizer.zero_grad, zero_grad, clear, accumulated gradients, before the next backward]}
  - {topic: GPU acceleration via .to(device) runs the same tensor code on a GPU for large speedups, keywords: [GPU acceleration, .to(device), .cuda(), moving tensors to the device, speedups, matrix multiplies]}
  - {topic: what you gain by using the framework versus from-scratch, keywords: [less boilerplate, fewer gradient bugs, hardware portability, ecosystem, pretrained models, what you gain]}
  - {topic: the honest trade-off you give up some transparency about what the gradients are doing, keywords: [trade-off, give up, transparency, intuition about, why the module taught the from-scratch version first]}
  - {topic: failure mode forgetting optimizer.zero_grad remedy call zero_grad at the top of every step, keywords: [forgetting, gradients pile up, adds them up by default, at the top of each step, beginner bug]}
  - {topic: failure mode calling backward twice without retain_graph remedy recompute forward or pass retain_graph, keywords: [backward twice, graph is freed after the first backward, retain_graph, recompute the forward, retain_graph=True]}
  - {topic: failure mode evaluating with gradients still tracked remedy wrap in torch.no_grad and model.eval, keywords: [evaluation with gradients tracked, wasting memory, torch.no_grad, model.eval, keeps building the graph]}
  - {topic: failure mode tensor and model on different devices remedy move both with .to(device) using one device variable, keywords: [different devices, one is on CPU the other on GPU, runtime error, single device variable]}
  - {topic: capability limit the framework does not choose architecture loss learning rate or diagnose failures, keywords: [does not choose the architecture, the loss, the learning rate, diagnose why training fails, modeling judgment stays with you]}
  - {topic: how PyTorch maps onto the from-scratch numpy manual-backprop workflow, keywords: [from-scratch, numpy, manual backprop, the ancestor, maps onto, workflow it improves on]}
require_artifact: true
---

@@@ hero
@lede Imagine you learned to add long numbers on paper first — carrying the ones, lining up the columns, checking each digit by hand. It was slow, but now you truly *understand* how adding works. Then someone hands you a **calculator**. You press the buttons and the answer appears instantly. You did not lose your understanding — you just stopped doing the slow, mistake-prone part by hand. That is exactly where you stand today. Over the last five days you hand-built a little machine that learns to read handwritten digits — the same **practice** loop of guessing, being corrected, and trying again. You wrote the forward pass. You *derived* the backward pass yourself with the chain rule. You nudged every weight by hand. Yesterday you met **PyTorch**, the tool that rebuilds that same machine in a handful of lines. Today we answer the deeper question a real engineer always asks: *why does a tool like this exist at all, and what is it actually doing underneath?* This is not just about typing less. It's about one clever trick — called **autograd** — that every serious AI lab, including the teams behind ChatGPT, leans on to train huge models without ever deriving a gradient by hand. And here's your quiet superpower: you already wrote those gradients yourself, so you will see straight through the magic to the numpy underneath.
@goal Here's what we'll figure out together today, one small idea at a time:
- **why** a framework like PyTorch exists at all — what real pain it removes that you felt this week;
- what a **tensor** really is, and how it quietly writes down every step you do to it;
- the one central trick — **autograd** and the **computation graph** — that turns your whole hand-derived backward pass into a single line;
- the four layers you stand on — **tensor+autograd**, **nn.Module**, **loss**, **optimizer** — and the short loop that ties them together;
- the four silent bugs (and their one-line fixes), plus the honest limit: PyTorch computes *correct* gradients, but it never picks a *good* model for you.

@@@ concept id=c1 tag="Why a framework" title="Why a framework like PyTorch exists at all" gotit="Got the why"
Before any code, let's answer the real question: *why bother with a framework?* You already made a working model in numpy. So what pain does PyTorch actually take away? The answer is the one part of this week that hurt the most — writing the **backward pass** by hand. That is the step where you figure out how much to nudge every weight. You did it once, for one small network, and it took hours of careful chain rule. Now picture doing that again for every new model shape you ever want to try. A [[framework||a toolkit like PyTorch that does the tedious, repetitive parts of building and training a model for you — chiefly, computing all the gradients — so you don't hand-derive them for every new shape]] exists to do exactly that repetitive, mistake-prone work for you.

**Think of cutting gears by hand versus a gear-stamping machine.** Imagine you build clocks. At first you cut every little gear yourself with a tiny file — measuring each tooth, checking each one. Your first clock works, and you learn a lot. But when you want a *bigger* clock with a hundred gears, hand-filing each one takes forever, and one badly-cut tooth stops the whole clock. So a workshop buys a **gear-stamping machine**: you tell it the shape, and it stamps out perfect gears in seconds. You still design the clock — the machine only removes the tedious cutting. PyTorch is that machine. You still design the model; PyTorch stamps out the gradients.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two workshops side by side. On the left, a person files a single gear tooth by tooth with a small file, labelled by hand — slow, one gear, easy to mis-cut. On the right, a gear-stamping machine presses out many identical perfect gears at once, labelled framework — fast, many gears, no mistakes. A note reads: you still design the clock; the machine just cuts the gears.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<rect x="16" y="26" width="228" height="150" rx="10" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.3"/>
<text x="130" y="20" font-size="11" fill="#C93B3B">by hand (numpy)</text>
<circle cx="95" cy="95" r="26" fill="none" stroke="#C93B3B" stroke-width="2"/>
<g stroke="#C93B3B" stroke-width="2"><line x1="95" y1="63" x2="95" y2="71"/><line x1="95" y1="119" x2="95" y2="127"/><line x1="63" y1="95" x2="71" y2="95"/><line x1="119" y1="95" x2="127" y2="95"/></g>
<line x1="150" y1="75" x2="120" y2="90" stroke="#8A6D3B" stroke-width="3"/>
<text x="175" y="80" font-size="8" fill="#8A6D3B">file</text>
<text x="130" y="150" font-size="8" fill="#C93B3B">one gear, tooth by tooth</text>
<text x="130" y="166" font-size="8" fill="#8A6D3B">slow · easy to mis-cut</text>
<rect x="276" y="26" width="228" height="150" rx="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/>
<text x="390" y="20" font-size="11" fill="#2D8B55">framework (PyTorch)</text>
<rect x="300" y="55" width="70" height="46" rx="6" fill="#CDE9D9" stroke="#2D8B55" stroke-width="1.4"/>
<text x="335" y="82" font-size="8" fill="#2D8B55">stamp</text>
<g fill="none" stroke="#2D8B55" stroke-width="1.6">
<circle cx="400" cy="70" r="12"/><circle cx="440" cy="70" r="12"/><circle cx="470" cy="95" r="12"/><circle cx="430" cy="105" r="12"/>
</g>
<text x="390" y="150" font-size="8" fill="#2D8B55">many perfect gears, fast</text>
<text x="390" y="166" font-size="8" fill="#6B645E">you still design the clock</text>
</g>
</svg>
%%%

**What the gear-machine picture gets right:** the machine only removes the *tedious cutting* — you still decide what clock to build, and a bad design still makes a bad clock. **Where it breaks down:** a gear machine stamps a fixed shape, but PyTorch adapts on the fly — it will happily stamp gradients for a model shape it has never seen, because it figures out the shape from the steps you ran.

#### The build-up: watch the by-hand backward pass grow, but PyTorch stay flat
Here is the pain, drawn as a picture. On day 2 your one small network needed a handful of hand-derived gradient lines. But every time you add a layer or change a shape, that hand-written backward pass grows — more chain-rule lines, more chances for a sign error. This is the [[from-scratch workflow||the numpy-only way you worked this week: write the forward pass, then hand-derive and hand-code the backward pass for that exact network — the older, cruder way that PyTorch improves on]] — the ancestor PyTorch was built to replace. With PyTorch, that backward work stays *flat at one line* — `loss.backward()` — no matter how big the model gets.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A chart of hand-written backward-pass lines versus model size. A rising red staircase labelled by hand climbs steeply: a tiny net needs a few lines, a bigger net needs many more. A flat green line labelled PyTorch stays at exactly one line, loss.backward(), for every model size. The gap between them widens as the model grows.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<line x1="60" y1="30" x2="60" y2="165" stroke="#6B645E" stroke-width="1.2"/>
<line x1="60" y1="165" x2="480" y2="165" stroke="#6B645E" stroke-width="1.2"/>
<text x="30" y="98" font-size="9" fill="#6B645E" transform="rotate(-90 30 98)">backward lines you write</text>
<text x="270" y="188" font-size="9" fill="#6B645E">model size (layers) →</text>
<polyline points="90,150 160,150 160,120 240,120 240,80 330,80 330,45 440,45" fill="none" stroke="#C93B3B" stroke-width="2.2"/>
<circle cx="90" cy="150" r="3" fill="#C93B3B"/><circle cx="240" cy="120" r="3" fill="#C93B3B"/><circle cx="330" cy="80" r="3" fill="#C93B3B"/><circle cx="440" cy="45" r="3" fill="#C93B3B"/>
<text x="380" y="38" font-size="9" fill="#C93B3B">by hand: grows + grows</text>
<line x1="90" y1="158" x2="450" y2="158" stroke="#2D8B55" stroke-width="2.2"/>
<text x="300" y="153" font-size="9" fill="#2D8B55">PyTorch: always 1 line — loss.backward()</text>
</g>
</svg>
%%%

The red staircase is why the framework exists. Your understanding of *why* it climbs — because you wrote those lines yourself — is exactly what makes you dangerous with the tool. You are not hiding from the backward pass. You *know* it, so you get to skip it. That is the whole **practice** loop of this module, now handed a machine.

Next, let's open up the one object everything in PyTorch is made of — and see how it quietly writes down its own receipt.

@@@ concept id=c2 tag="Tensor" title="The tensor: a numpy array that writes down what you do to it" gotit="Got tensors"
Everything in PyTorch — every weight, every batch of pixels — is made of one kind of object. You already know its cousin: the numpy array, a grid of numbers you can add, multiply, and reshape. PyTorch's version is called a [[tensor||PyTorch's version of a numpy array — the same grid of numbers, but it can secretly write down every step done to it, and it can run on a fast GPU chip]]. It holds the same numbers, has a shape and a **dtype** (the kind of number, like `float32`), and does the same math. So the freeing news first: *a tensor is just a numpy array with two extra powers.* If you can picture a numpy array, you already picture a tensor.

**Think of a notebook that writes itself.** Picture a magic notebook. Every time you do a sum on its page — "add 3, then multiply by 2" — the notebook doesn't only show the answer; in the margin it *quietly writes down each step you took*. Later, if you ask "how did I get here?", you just read the margin backward. A plain numpy array is an ordinary notebook — it shows the answer, but keeps no margin notes. A tensor is the self-writing notebook: same answer on the page, plus a running list of every step in the margin. That margin is the secret behind the automatic backward pass you'll meet next.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="Two notebooks side by side. The left, labelled numpy array, shows only a final answer with a blank margin. The right, labelled tensor, shows the same answer plus a margin that lists each step: input, times W, plus b. The margin is what lets PyTorch read the steps backward.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<rect x="20" y="26" width="220" height="150" rx="8" fill="#F5F2EC" stroke="#B7AEA0" stroke-width="1.2"/>
<text x="130" y="20" font-size="11" fill="#3B6FA0">numpy array</text>
<line x1="70" y1="40" x2="70" y2="164" stroke="#B7AEA0" stroke-width="1" stroke-dasharray="3 2"/>
<text x="45" y="100" font-size="8" fill="#B7AEA0" transform="rotate(-90 45 100)">margin: (blank)</text>
<text x="155" y="95" font-size="11" fill="#2b2b2b">= 1.07</text>
<text x="155" y="150" font-size="8" fill="#C93B3B">no record of steps</text>
<rect x="280" y="26" width="220" height="150" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/>
<text x="390" y="20" font-size="11" fill="#2D8B55">tensor</text>
<line x1="330" y1="40" x2="330" y2="164" stroke="#2D8B55" stroke-width="1" stroke-dasharray="3 2"/>
<g font-size="7" fill="#2D8B55" text-anchor="start"><text x="292" y="55">step: input</text><text x="292" y="72">step: × W</text><text x="292" y="89">step: + b</text><text x="292" y="106">step: square</text></g>
<text x="415" y="100" font-size="11" fill="#2b2b2b">= 1.07</text>
<text x="415" y="150" font-size="8" fill="#2D8B55">margin records each step</text>
</g>
</svg>
%%%

**What the self-writing-notebook picture gets right:** the numbers on the page are *identical* on both sides — a tensor is not a fancier number, it is the same number with a memory. **Where it breaks down:** a notebook margin is a flat list, but a tensor's memory is a little branching map of every add and multiply — and PyTorch only bothers keeping it when you ask, by flipping one flag.

#### The build-up: flip the flag, and a gradient slot appears
That flag is the first extra power. Every tensor has a switch called [[requires_grad||a true/false switch on a tensor: when true, PyTorch keeps the margin notes — it records the steps so it can compute gradients for this tensor later]]. When `requires_grad=True`, PyTorch keeps the margin notes and later fills a slot on the tensor called [[.grad||the slot on a tracked tensor where PyTorch writes the computed gradient after you call backward — empty (None) until then]] — that is *where the gradient lands*. Your input pixels never change, so they need no notes; your **weights** do change, so they get the flag on. The second extra power: a tensor knows which [[device||where a tensor lives and does its math — a plain CPU, or a much faster GPU chip; today everything stays on the CPU]] it lives on. Today we stay on the CPU, so you can set that one aside — but now you know the word. Let's flip the flag and watch the empty `.grad` slot appear, ready to be filled. Predict: before any backward pass, what's in the slot?

%%% demo id=tensor label="predict what .grad holds before backward, then run"
code: import torch
code: x = torch.tensor([3.0], requires_grad=True)  # flag ON: keep the margin
code: print("shape:", x.shape, " dtype:", x.dtype, " device:", x.device)
code: print("grad slot before backward:", x.grad)
out: shape: torch.Size([1])  dtype: torch.float32  device: cpu
out: grad slot before backward: None
take: <b>A number, a shape, a home — and an empty grad slot.</b> The tensor holds <code>3.0</code>, knows its shape and dtype, and lives on the <code>cpu</code>. Because <code>requires_grad=True</code>, it now has a <code>.grad</code> slot — but it starts empty (<code>None</code>), because nothing has been computed backward yet. That empty slot is a promise: run a backward pass, and PyTorch will fill it for you.
%%%

You just met the raw material. A tensor is your familiar numpy array plus a self-writing margin (`requires_grad`) and a home (`device`). Hold onto that margin idea — in the very next concept it becomes the trick that erases your entire hand-written backward pass.

@@@ concept id=c3 tag="autograd" title="autograd: the computation graph does your backward pass" gotit="Got autograd"
This is the one idea that makes the whole framework worth it. On day 2 you spent hours deriving the backward pass by hand — the chain rule, layer by layer, working out how much each weight should change. It was the hardest thing you built all module. Today PyTorch does *all* of it, correctly, in one line, for *any* network you can dream up. The engine behind this is called [[autograd||PyTorch's automatic-gradient engine: as the forward pass runs, it records every operation into a graph, then walks that graph backward to compute the gradient for every weight — your hand-derived backward pass, done for you]].

**Think of leaving a breadcrumb trail on a hike.** You go on a long, winding hike. You drop a breadcrumb at every turn: left here, uphill there, right at the big rock. To get home you don't need a map — you just follow the trail *backward*, breadcrumb by breadcrumb, straight to the start. Autograd does this with math. As your numbers flow *forward* through the model, autograd drops a breadcrumb at every step — every multiply, every add, every ReLU — building a little map called the [[computation graph||the running record autograd builds while the forward pass runs: it lists every operation, in order, so it can be walked backward to compute gradients]]. Then, when you ask, it walks that graph *backward* to work out exactly how each weight affected the final loss. That backward walk *is* your backward pass. Remember the tensor's margin notes from a moment ago? These breadcrumbs are those notes.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A trail of breadcrumbs from input through operations times W, ReLU, times W, to the final loss. Green forward arrows drop a breadcrumb at each step as data flows forward. Then a red dashed arrow retraces the exact same trail backward from loss to input, labelled loss.backward computing every gradient.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<circle cx="50" cy="70" r="15" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.4"/><text x="50" y="73" font-size="8" fill="#3B6FA0">input</text>
<circle cx="165" cy="55" r="14" fill="#F5F2EC" stroke="#B7AEA0"/><text x="165" y="58" font-size="7">× W</text>
<circle cx="285" cy="80" r="14" fill="#F5F2EC" stroke="#B7AEA0"/><text x="285" y="83" font-size="7">ReLU</text>
<circle cx="405" cy="55" r="14" fill="#F5F2EC" stroke="#B7AEA0"/><text x="405" y="58" font-size="7">× W</text>
<circle cx="478" cy="92" r="16" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="478" y="95" font-size="8" fill="#C93B3B">loss</text>
<path d="M65 67 L151 57" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/>
<path d="M178 58 L272 77" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/>
<path d="M299 78 L392 58" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/>
<path d="M419 60 L463 86" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/>
<path d="M464 110 Q260 172 52 92" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="6 4" fill="none" marker-end="url(#ar)"/>
<text x="235" y="38" font-size="9" fill="#2D8B55">forward: drop a breadcrumb at each step (the graph)</text>
<text x="262" y="162" font-size="9" fill="#C93B3B">loss.backward(): retrace the trail → every gradient</text>
<defs><marker id="af" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker><marker id="ar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#C93B3B"/></marker></defs>
</g>
</svg>
%%%

**What the breadcrumb picture gets right:** you only ever walk *forward* on purpose; the way back is rebuilt automatically from the trail you already left. **Where it breaks down:** a hiking trail is a single line, but the computation graph can branch and merge — it applies the chain rule at every fork, which is exactly the calculus you did by hand.

#### The build-up: pages of chain rule collapse into one call
Here is the payoff, side by side. On the left, a sketch of the backward pass you wrote on day 2 — the gradient pushed back through layer 2, through the ReLU, through layer 1, each step its own careful formula. On the right, the PyTorch version. Watch the whole stack vanish into `loss.backward()`.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A before-and-after of the backward pass. On the left, a stack of five hand-derived chain-rule lines computing dZ2, dW2, dA1, dZ1, dW1. On the right, a single line: loss.backward(). An arrow shows the whole stack collapsing to one line, with a note that autograd fills every .grad automatically.">
<g font-family="monospace" font-size="8.5" text-anchor="start">
<text x="120" y="24" font-size="10" fill="#C93B3B" text-anchor="middle">by hand (day 2)</text>
<rect x="18" y="32" width="204" height="164" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/>
<text x="30" y="56" fill="#2b2b2b">dZ2 = (A2 - Y) / m</text>
<text x="30" y="76" fill="#2b2b2b">dW2 = A1.T @ dZ2</text>
<text x="30" y="96" fill="#2b2b2b">dA1 = dZ2 @ W2.T</text>
<text x="30" y="116" fill="#2b2b2b">dZ1 = dA1 * (Z1 &gt; 0)</text>
<text x="30" y="136" fill="#2b2b2b">dW1 = X.T @ dZ1</text>
<text x="30" y="166" fill="#8A6D3B">...and the biases,</text>
<text x="30" y="180" fill="#8A6D3B">every layer, by hand</text>
<text x="262" y="115" font-size="20" fill="#7A5BA0" text-anchor="middle">→</text>
<text x="410" y="24" font-size="10" fill="#2D8B55" text-anchor="middle">PyTorch</text>
<rect x="300" y="32" width="204" height="164" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/>
<text x="312" y="100" fill="#2b2b2b" font-size="11">loss.backward()</text>
<text x="312" y="130" fill="#2D8B55">fills every .grad</text>
<text x="312" y="146" fill="#2D8B55">for every weight,</text>
<text x="312" y="162" fill="#2D8B55">automatically</text>
</g>
</svg>
%%%

Every one of those hand-derived lines — plus the ones for the biases, plus every extra layer — becomes the single call [[loss.backward()||the one call that walks the computation graph backward from the loss and deposits the gradient into every tracked tensor's .grad slot — it replaces your entire hand-derived backward pass]]. It reads the loss tensor's breadcrumb trail and, for each weight, drops the gradient into that weight's `.grad` slot. Let's watch a gradient appear where there was nothing. Predict: after we call backward on `(3w+1)²` at `w = 2`, what number lands in `w.grad`?

%%% demo id=autograd label="predict w.grad after backward, then run"
code: w = torch.tensor([2.0], requires_grad=True)
code: print("before backward:", w.grad)
code: loss = (3 * w + 1) ** 2       # forward: autograd records each step
code: loss.backward()               # walk the graph backward → fill w.grad
code: print("after backward: ", w.grad)
out: before backward: None
out: after backward:  tensor([42.])
take: <b>Empty, then filled — with no chain rule from you.</b> Before <code>backward()</code>, the slot was <code>None</code> (nothing recorded yet). After one <code>loss.backward()</code>, autograd walked the graph and dropped the exact gradient into <code>w.grad</code>. Sanity check by hand: the slope of <code>(3w+1)²</code> at w=2 is <code>2·(3·2+1)·3 = 42</code> — the same number the chain rule would have given you. The machine and your hand agree.
%%%

!!! c-info 🧮
<b>Optional (skippable) — the one line that ties manual to automatic.</b> In your numpy code you computed a gradient by hand and stored it, say, in a variable `dW`. In PyTorch the *same* number lands automatically in `W.grad`. That is the whole mapping: your hand-written `dW` ≙ PyTorch's `W.grad`, and your hand-derived backward math ≙ the single call `loss.backward()`. Nothing conceptually new happened — the calculus is identical. PyTorch just did the bookkeeping.
!!!

Take a breath — you just watched the hardest thing you built get done for you, correctly, in one line. But a gradient sitting in `.grad` doesn't move a weight on its own. Next, let's see the four neat layers PyTorch stacks on top of this, and the short loop that drives them.

@@@ concept id=c4 tag="Four layers + loop" title="The four layers, and the loop that ties them together" gotit="Got the four layers"
So far you have the ground floor: tensors that remember their steps, and autograd that walks them backward. On top of that, PyTorch gives you three more tidy pieces, and together they map *one-to-one* onto the hand-built model you already made. Once you see the four layers, PyTorch stops being a pile of new words and becomes an old friend in a new coat.

**Think of a four-person kitchen brigade.** Picture a small restaurant kitchen. On the bottom is the **pantry** — all the raw ingredients (that's your tensors + autograd, the stuff everything is made of). Above it, a **prep cook** keeps all the ingredients organized in labeled bins so nothing gets lost (that's [[nn.Module||the box you put your layers inside: it auto-registers every weight and bias so you can hand the whole set to the optimizer with one call, instead of tracking loose arrays yourself]], which keeps every weight tidy). Then a **taster** decides how far off the dish is from the recipe (that's the [[loss function||a ready-made object that scores how wrong the model's guess is, like nn.CrossEntropyLoss — you create it once and call it, replacing the loss formula you wrote by hand]]). Finally a **head chef** takes the taster's verdict and adjusts the seasoning for next time (that's the [[optimizer||the piece that owns the update rule: it reads every weight's .grad and nudges the weight, so optimizer.step() replaces the W -= lr*dW lines you wrote by hand]]). Four roles, one meal — and every real PyTorch model is exactly this brigade.

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="Four stacked layers of a PyTorch model. Bottom layer: tensor plus autograd, the substrate that holds numbers and computes gradients. Above it: nn.Module, which holds the weights and biases. Above that: the loss function, which scores how wrong the guess is. Top: the optimizer, which owns the update rule and nudges the weights. Each layer rests on the one below it.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<rect x="90" y="180" width="340" height="40" rx="6" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.5"/>
<text x="260" y="199" font-size="10" fill="#3B6FA0">1 · tensor + autograd</text>
<text x="260" y="213" font-size="7.5" fill="#6B645E">the numbers + the automatic gradients (the pantry)</text>
<rect x="110" y="132" width="300" height="40" rx="6" fill="#EEF2F0" stroke="#3E7C63" stroke-width="1.5"/>
<text x="260" y="151" font-size="10" fill="#3E7C63">2 · nn.Module</text>
<text x="260" y="165" font-size="7.5" fill="#6B645E">holds + tracks every weight and bias (the prep cook)</text>
<rect x="130" y="84" width="260" height="40" rx="6" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.5"/>
<text x="260" y="103" font-size="10" fill="#7A5BA0">3 · loss function</text>
<text x="260" y="117" font-size="7.5" fill="#6B645E">scores how wrong the guess is (the taster)</text>
<rect x="150" y="36" width="220" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
<text x="260" y="55" font-size="10" fill="#2D8B55">4 · optimizer</text>
<text x="260" y="69" font-size="7.5" fill="#6B645E">owns the update rule (the head chef)</text>
</g>
</svg>
%%%

**What the kitchen-brigade picture gets right:** each role has one clear job, and they hand work up the chain — pantry to prep to taster to chef — just like data flows through the four layers. **Where it breaks down:** a real brigade has four separate people, but in PyTorch these layers are woven together by autograd's graph — the taster's score is what the chef traces back through the whole kitchen at once.

#### The parameters handle: the prep cook lists every ingredient
Here's why layer 2 matters. In your numpy code you kept loose arrays — `W1`, `b1`, `W2`, `b2` — and had to remember every one to update it. `nn.Module` fixes that: the learnable tensors inside it are called [[parameters||the learnable tensors inside a module — its weights and biases — which the optimizer will update; model.parameters() hands you the whole list at once]], and one call, `model.parameters()`, hands you the *whole list* so you can pass it straight to the optimizer. No more hunting for weights by hand.

#### The build-up: your manual loop and the PyTorch loop, line by line
Now the satisfying part — laying your hand-built training loop next to the PyTorch one and matching them up. They are the *same five moves*. Read the PyTorch order as one sentence: *clear the scoreboard, guess, score, retrace, step.*

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="A line-by-line map between the hand-written numpy loop and the PyTorch loop. Row by row: reset your own gradient sums maps to optimizer.zero_grad; forward pass X@W+b maps to out = model(x); your loss formula maps to loss = criterion(out, y); your hand-derived backward maps to loss.backward; and W -= lr*dW for each weight maps to optimizer.step. The two loops are the same five moves.">
<g font-family="monospace" font-size="8" text-anchor="start">
<text x="140" y="22" font-size="10" fill="#C93B3B" text-anchor="middle">by hand (numpy)</text>
<text x="390" y="22" font-size="10" fill="#2D8B55" text-anchor="middle">PyTorch</text>
<rect x="20" y="32" width="230" height="186" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/>
<rect x="280" y="32" width="220" height="186" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/>
<text x="32" y="56" fill="#2b2b2b">reset dW, db = 0</text>
<text x="292" y="56" fill="#2b2b2b">optimizer.zero_grad()</text>
<text x="32" y="92" fill="#2b2b2b">Z1 = X@W1+b1 ; ...</text>
<text x="292" y="92" fill="#2b2b2b">out = model(x)</text>
<text x="32" y="128" fill="#2b2b2b">L = -log p[y] (yours)</text>
<text x="292" y="128" fill="#2b2b2b">loss = criterion(out, y)</text>
<text x="32" y="164" fill="#2b2b2b">dW2=... dW1=... (hand)</text>
<text x="292" y="164" fill="#2b2b2b">loss.backward()</text>
<text x="32" y="200" fill="#2b2b2b">W1 -= lr*dW1 ; ...</text>
<text x="292" y="200" fill="#2b2b2b">optimizer.step()</text>
<g stroke="#B7AEA0" stroke-width="1.2" stroke-dasharray="3 2">
<line x1="252" y1="52" x2="278" y2="52"/><line x1="252" y1="88" x2="278" y2="88"/><line x1="252" y1="124" x2="278" y2="124"/><line x1="252" y1="160" x2="278" y2="160"/><line x1="252" y1="196" x2="278" y2="196"/>
</g>
</g>
</svg>
%%%

Every line on the left has a twin on the right. Your gradient reset became [[optimizer.zero_grad()||the call that clears every parameter's .grad back to zero before the next backward pass; needed because PyTorch adds new gradients on top of old ones instead of replacing them]]. Your forward math became `out = model(x)`. Your loss formula became `criterion(out, y)`. Your whole hand-derived backward became `loss.backward()`. And your five `W -= lr*dW` lines became one call, [[optimizer.step()||the call that applies one update to every parameter at once, using each one's current .grad — it replaces the W -= lr*dW lines you wrote by hand for every weight]]. The simplest optimizer, [[torch.optim.SGD||PyTorch's plainest optimizer: it does exactly the rule you wrote by hand, w := w - lr*grad, for every parameter — swap it for a fancier one later with a one-line change]], does the *exact* rule you coded. Let's run the whole loop small and watch the loss fall — predict which way the number moves.

%%% demo id=loop label="predict which way loss moves, then run the loop"
code: import torch.nn as nn, torch.optim as optim
code: model = nn.Linear(4, 1)                                  # layer 2: nn.Module
code: criterion = nn.MSELoss()                                 # layer 3: loss
code: optimizer = optim.SGD(model.parameters(), lr=0.1)        # layer 4: optimizer
code: x = torch.randn(8, 4); y = torch.randn(8, 1)
code: for epoch in range(4):
code:     optimizer.zero_grad()          # clear the scoreboard
code:     out = model(x)                 # guess
code:     loss = criterion(out, y)       # score
code:     loss.backward()                # retrace
code:     optimizer.step()               # step
code:     print(f"epoch {epoch}  loss {loss.item():.3f}")
out: epoch 0  loss 1.842
out: epoch 1  loss 1.517
out: epoch 2  loss 1.264
out: epoch 3  loss 1.068
take: <b>Down, down, down — it's learning.</b> The same five moves you wrote by hand, now five short names: <code>zero_grad → forward → loss → backward → step</code>. Notice <code>optimizer.step()</code> nudged every weight at once (no per-weight lines), and <code>model.parameters()</code> handed the optimizer the whole set. This is the beating heart of every PyTorch project on earth — including the ones training ChatGPT.
%%%

You now own the whole engine: the four layers and the five-line loop. But a few silent bugs live in that loop, and one honest limit lives above it. Let's meet them next — as puzzles, not warnings.

@@@ concept id=c5 tag="Four puzzles" title="Four silent bugs, four one-line fixes" gotit="Got the four fixes"
Here's a fun truth: almost *everyone* new to PyTorch hits the same four little traps. They are silent — no crash, or a confusing crash — so they feel like magic gone wrong. But each one has a simple cause and a one-line fix. Once you've seen them, they can never catch you. Treat this like a detective collecting four clues: each clue is a puzzle, and you already know enough to solve every one.

**Think of four squeaky doors in a house you just moved into.** Each door sticks or squeaks in its own way, and at first each is a mystery. But every one has a tiny, obvious fix — a drop of oil, a turned screw. Once someone shows you the fix, you never fear that door again. These four bugs are those doors: annoying until named, trivial once fixed.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="Four puzzle cards, each a common PyTorch bug with its one-line fix. Card one: forgot zero_grad, gradients pile up, fix is optimizer.zero_grad. Card two: backward called twice, the graph is freed, fix is recompute the forward or retain_graph. Card three: evaluating with gradients on, wastes memory, fix is torch.no_grad. Card four: tensor on CPU but model on GPU, mismatch error, fix is .to(device) on both.">
<g font-family="monospace" font-size="8" text-anchor="middle">
<rect x="16" y="26" width="238" height="82" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/>
<text x="135" y="44" font-size="9" fill="#C93B3B">1 · forgot zero_grad</text>
<text x="135" y="62" font-size="7.5" fill="#6B645E">cause: gradients pile up (add)</text>
<text x="135" y="82" font-size="8" fill="#2D8B55">fix: optimizer.zero_grad()</text>
<text x="135" y="98" font-size="7" fill="#6B645E">each step, at the top</text>
<rect x="266" y="26" width="238" height="82" rx="8" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.2"/>
<text x="385" y="44" font-size="9" fill="#8A6D3B">2 · backward twice</text>
<text x="385" y="62" font-size="7.5" fill="#6B645E">cause: graph freed after 1st</text>
<text x="385" y="82" font-size="8" fill="#2D8B55">fix: redo forward (or</text>
<text x="385" y="98" font-size="8" fill="#2D8B55">retain_graph=True)</text>
<rect x="16" y="120" width="238" height="82" rx="8" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.2"/>
<text x="135" y="138" font-size="9" fill="#7A5BA0">3 · eval with grads on</text>
<text x="135" y="156" font-size="7.5" fill="#6B645E">cause: still building graph</text>
<text x="135" y="176" font-size="8" fill="#2D8B55">fix: with torch.no_grad():</text>
<text x="135" y="192" font-size="7" fill="#6B645E">saves memory + time</text>
<rect x="266" y="120" width="238" height="82" rx="8" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.2"/>
<text x="385" y="138" font-size="9" fill="#3B6FA0">4 · device mismatch</text>
<text x="385" y="156" font-size="7.5" fill="#6B645E">cause: CPU tensor, GPU model</text>
<text x="385" y="176" font-size="8" fill="#2D8B55">fix: .to(device) on both</text>
<text x="385" y="192" font-size="7" fill="#6B645E">one device variable</text>
</g>
</svg>
%%%

**What the squeaky-doors picture gets right:** each fix is small and independent — oiling door 1 has nothing to do with door 2. **Where it breaks down:** two of these doors (1 and 3) don't crash loudly — they *silently* waste your training, which is why naming them matters more than a door that just squeaks.

#### Puzzle 1, watchable: the gradients that pile up
The most famous trap is forgetting `zero_grad()`. Why must you clear the gradients every round? Because PyTorch **adds them up by default** — each `loss.backward()` *piles* the new gradient on top of whatever was already in `.grad`, instead of replacing it. Picture a scoreboard nobody resets: round 3 shows rounds 1+2+3 stacked, a wrong, growing number. Predict: if we call backward twice on the same thing *without* clearing, does `.grad` show 3, or 6?

%%% demo id=zerograd label="predict 3 or 6, then run backward twice without clearing"
code: w = torch.tensor([2.0], requires_grad=True)
code: (3 * w).backward(); print("after 1st backward:", w.grad)
code: (3 * w).backward(); print("after 2nd backward:", w.grad)   # no zero_grad between!
out: after 1st backward: tensor([3.])
out: after 2nd backward: tensor([6.])
take: <b>3, then 6 — it added, it did not replace.</b> The true gradient is 3 both times, but the second <code>backward()</code> <i>piled</i> another 3 on top. Over a real loop this snowballs into nonsense. Calling <code>optimizer.zero_grad()</code> at the top of each step wipes the slot to 0 first, so every step uses this round's gradient only. That's the #1 beginner bug — now it will never catch you.
%%%

#### Puzzles 2, 3, 4: cause → one-line fix
The other three follow the same shape — a simple cause, a one-line fix:

- **Backward twice (freed graph).** By default PyTorch throws away the computation graph right after the first `loss.backward()` to save memory. Call `backward()` a second time on the *same* loss and you get an error: *"backward through the graph a second time."* **Fix:** run the forward pass again to build a fresh graph, or (rarely) pass `retain_graph=True` to keep the old one.
- **Evaluating with gradients still on.** When you're only *checking* your model — not training — you don't need gradients. But by default PyTorch keeps recording the graph, wasting memory and time. **Fix:** wrap the check in `with [[torch.no_grad()||a block that tells PyTorch to stop recording the graph — use it whenever you only want a forward pass and no gradients, such as evaluation, to save memory and time]]:`, so it does a plain forward pass and records nothing.
- **Device mismatch.** A tensor lives on the CPU; your model lives on the GPU. They cannot do math together, and PyTorch raises a runtime error. **Fix:** put both on the *same* [[device||the CPU or GPU a tensor lives on; a CPU tensor and a GPU model cannot interact, so move both to one place]] — pick one `device` variable and call `.to(device)` on the model and on every batch.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A small before-and-after for the device fix. On the left, a CPU tensor and a GPU model with a red cross between them, labelled mismatch error. On the right, both moved with .to(device) to the same GPU, with a green check, labelled they can now do math together.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="130" y="22" font-size="10" fill="#C93B3B">before: mismatch</text>
<rect x="30" y="36" width="86" height="40" rx="6" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.3"/><text x="73" y="60" font-size="8" fill="#3B6FA0">tensor · CPU</text>
<rect x="150" y="36" width="86" height="40" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.3"/><text x="193" y="60" font-size="8" fill="#8A6D3B">model · GPU</text>
<text x="133" y="102" font-size="16" fill="#C93B3B">✗ error</text>
<text x="390" y="22" font-size="10" fill="#2D8B55">after: .to(device)</text>
<rect x="300" y="36" width="86" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="343" y="60" font-size="8" fill="#2D8B55">tensor · GPU</text>
<rect x="410" y="36" width="86" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="453" y="60" font-size="8" fill="#2D8B55">model · GPU</text>
<text x="398" y="102" font-size="16" fill="#2D8B55">✓ same device</text>
<text x="398" y="128" font-size="8" fill="#6B645E">one device variable, .to(device) on both</text>
</g>
</svg>
%%%

That's also where [[.to(device)||the call that moves a tensor or model to a chosen device (CPU or GPU); the same tensor code then runs on a fast GPU, which is why big models train far faster than numpy-on-CPU]] earns its keep beyond fixing bugs: move everything to a GPU and the *exact same* tensor code runs far faster than numpy ever could on the CPU — because a GPU does thousands of the matrix multiplies at once. That speed is a big part of *why* the framework exists. Four doors, four drops of oil. You just made yourself hard to trip.

Now one honest limit — the thing PyTorch will *never* do for you — and then we gather the whole day onto one page.

@@@ concept id=c6 tag="The honest limit" title="What PyTorch will never do for you" gotit="Got the limit"
This is the concept that separates someone who *uses* PyTorch from someone who *understands* it. PyTorch computes gradients — perfectly, tirelessly, for any model you build. But here is the honest, important truth: *it never decides whether your model is a good idea.* Autograd is not clever. It faithfully computes correct gradients for *whatever* graph you hand it — even a badly-designed one — and it will train a bad model into a badly-trained model without a single word of complaint. The judgment stays with you.

**Think of a GPS that never picks your destination.** A GPS is brilliant at one thing: give it a destination and it finds the perfect route, turn by turn. But it will *never* decide *where you want to go*. Type in the wrong address and it will cheerfully, flawlessly, drive you to the wrong place. Autograd is that GPS. Tell it a model (a destination) and it computes the perfect gradients (the route). But *which* model, *which* loss, *how big* a learning rate — those are your address, and PyTorch never fills them in. Point it somewhere silly and it takes you there, perfectly.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A GPS metaphor. On the left, a box labelled you decide: the architecture, the loss, the learning rate — the destination. In the middle, a GPS labelled PyTorch autograd that computes the perfect route (gradients) to whatever destination it is given. On the right, two outcomes: a good destination gives a good model, a bad destination gives a badly-trained model, and the GPS never complains about either.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<rect x="16" y="40" width="130" height="110" rx="8" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.4"/>
<text x="81" y="34" font-size="9" fill="#7A5BA0">you decide (the address)</text>
<text x="81" y="66" font-size="8" fill="#2b2b2b">architecture</text>
<text x="81" y="88" font-size="8" fill="#2b2b2b">the loss</text>
<text x="81" y="110" font-size="8" fill="#2b2b2b">learning rate</text>
<text x="81" y="134" font-size="8" fill="#2b2b2b">is this a good idea?</text>
<rect x="186" y="60" width="130" height="70" rx="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/>
<text x="251" y="90" font-size="9" fill="#2D8B55">PyTorch / autograd</text>
<text x="251" y="108" font-size="7.5" fill="#6B645E">perfect route to</text>
<text x="251" y="120" font-size="7.5" fill="#6B645E">any destination</text>
<path d="M146 95 L184 95" stroke="#7A5BA0" stroke-width="1.5" marker-end="url(#ag)"/>
<path d="M316 82 L356 60" stroke="#2D8B55" stroke-width="1.5" marker-end="url(#ag)"/>
<path d="M316 108 L356 130" stroke="#2D8B55" stroke-width="1.5" marker-end="url(#ag)"/>
<text x="430" y="52" font-size="8" fill="#2D8B55">good address → good model</text>
<text x="430" y="140" font-size="8" fill="#C93B3B">bad address → bad model</text>
<text x="430" y="154" font-size="7" fill="#6B645E">(no complaint either way)</text>
<defs><marker id="ag" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#6B645E"/></marker></defs>
</g>
</svg>
%%%

**What the GPS picture gets right:** the tool is flawless at *execution* and silent about *judgment* — it never second-guesses your address. **Where it breaks down:** a GPS at least warns you a route is long; autograd gives *no* feedback on whether your model is wise — a falling loss can still mean you're learning the wrong thing well.

#### The build-up: the deal you're making — and why you built it by hand first
So there's a trade-off, and it's worth saying plainly. What you *gain* from PyTorch is huge; what you *give up* is a little transparency. Here is the deal in one table:

%%% table
:: PyTorch does for you :: You must still decide :: What you give up
Compute every gradient (`.backward()`) :: The model shape (how many layers, how wide) :: You see less of *what* the gradients are doing
Track all weights (`nn.Module`) :: Which loss fits the task :: A little day-to-day intuition
Apply the update (`optimizer.step()`) :: The learning rate (`lr`) :: —
Run fast on a GPU (`.to(device)`) :: Whether the whole idea is sound :: —
%%%

That right-hand column is exactly *why this module made you build everything by hand first.* Because you already wrote the backward pass yourself, you didn't trade away your intuition — you *kept* it, and now you can read straight through PyTorch to the numpy underneath. A beginner who skips the by-hand version treats autograd as magic and gets stuck when training goes wrong. You won't, because you know what `.grad` really holds. Predict what happens if we point autograd at a pointless model — will it complain, or just compute?

%%% demo id=limit label="predict: does autograd refuse a silly model, or just compute?"
code: w = torch.tensor([5.0], requires_grad=True)     # a lone weight, no real task
code: silly_loss = (w * 0)          # a "loss" that ignores w entirely — bad modeling
code: silly_loss.backward()
code: print("gradient for w:", w.grad)
out: gradient for w: tensor([0.])
take: <b>No complaint — it just computed.</b> The loss ignores <code>w</code>, so the correct gradient really is 0 — and autograd dutifully reports 0, never warning you that this "model" learns nothing. That's the honest limit in one line: PyTorch computes <i>correct</i> gradients for whatever you build, but it will never tell you the model was a bad idea. That judgment is yours — and this whole module gave it to you.
%%%

Victory lap: you now understand PyTorch the way a real engineer does — not as magic, but as a fast, faithful gradient machine sitting on top of the numpy you already know. Let's gather the whole day onto one page.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Here's the whole day in a few beats. The theme never changed: **practice** — a machine learns by guessing, being corrected, and repeating — and PyTorch is just the tool that does the tedious, error-prone parts of that loop for you.

- **Why it exists:** hand-writing the backward pass grows and breaks as models get bigger. A framework stamps out those gradients for you, so the by-hand backward work stays flat at one line.
- **The tensor:** a numpy array that writes down every step in its margin (`requires_grad`) and knows its home (`device`); its gradient lands in `.grad`.
- **autograd:** as the forward pass runs, it records every step into the **computation graph**; `loss.backward()` walks that graph backward and fills every `.grad` — your whole hand-derived backward pass, gone.
- **The four layers:** tensor+autograd (the numbers + gradients), **nn.Module** (holds the weights), a **loss** (scores the guess), an **optimizer** (owns the update rule). The loop is `zero_grad → forward → loss → backward → step`.
- **The four fixes:** `zero_grad()` (gradients add up), redo the forward for a second backward (the graph is freed), `torch.no_grad()` at eval, and `.to(device)` on both tensor and model.
- **The honest limit:** PyTorch computes *correct* gradients for *any* model — it never picks a *good* one. That judgment is yours, and this module gave it to you.

Here is the single most useful thing to keep — the map from what you built by hand to what PyTorch calls it:

%%% table
:: What you wrote by hand :: PyTorch version :: What it does
Loose `W`, `b` arrays :: `nn.Module` + `model.parameters()` :: holds + lists every weight for you
Hand-derived `dW`, `db` :: `loss.backward()` :: autograd walks the graph and fills every `.grad`
`W -= lr * dW` (each weight) :: `optimizer.step()` :: one call nudges every parameter
Reset your own grad sums :: `optimizer.zero_grad()` :: required — gradients add up by default
Your loss formula :: `nn.CrossEntropyLoss()` etc. :: a ready-made loss function
Slow numpy on CPU :: `.to(device)` :: same code, far faster on a GPU
%%%

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="The five-line PyTorch loop as a compact ring: zero_grad, forward, loss, backward, step, with an arrow curving back to zero_grad, labelled the heart of every PyTorch project, repeated over epochs.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="48" y="60" fill="#C93B3B">zero_grad</text>
<text x="140" y="60" fill="#3B6FA0">forward</text>
<text x="218" y="60" fill="#7A5BA0">loss</text>
<text x="300" y="60" fill="#C99A12">backward</text>
<text x="392" y="60" fill="#2D8B55">step</text>
<text x="94" y="60" fill="#B7AEA0">→</text>
<text x="181" y="60" fill="#B7AEA0">→</text>
<text x="260" y="60" fill="#B7AEA0">→</text>
<text x="347" y="60" fill="#B7AEA0">→</text>
<path d="M418 55 Q472 30 472 76 Q472 96 46 72" fill="none" stroke="#B7AEA0" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#a5)"/>
<text x="260" y="30" fill="#6B645E">the heart of every PyTorch project — repeated over epochs</text>
<defs><marker id="a5" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#B7AEA0"/></marker></defs>
</g>
</svg>
%%%

Keep this cheat-sheet close — every term you'll meet in a real codebase is right here:

%%% jargon
framework | a toolkit like PyTorch that does the tedious parts of training (chiefly, all the gradients) for you
tensor | a numpy array that also records the steps done to it (requires_grad) and knows its device
requires_grad | the true/false switch that turns on a tensor's step-recording so gradients can be computed
.grad | the slot on a tracked tensor where the computed gradient lands after backward (None until then)
autograd | the engine that records the forward steps and computes all gradients on request
computation graph | the running record of every operation, built as the forward runs, walked backward for gradients
loss.backward() | walks the graph backward and fills every weight's .grad — replaces the hand-derived backward pass
nn.Module | the box that holds your layers and auto-tracks every weight; model.parameters() lists them all
parameters | the learnable tensors (weights + biases) inside a module that the optimizer updates
loss function | a ready-made object (e.g. nn.CrossEntropyLoss) that scores how wrong the guess is
optimizer / SGD | owns the update rule; optimizer.step() nudges every weight (SGD does w -= lr·grad)
optimizer.zero_grad() | clears every .grad to zero each step; skip it and gradients pile up (the #1 bug)
torch.no_grad() | stops graph recording for a plain forward pass (use at evaluation to save memory + time)
.to(device) | moves a tensor or model to CPU or GPU; the same code then runs far faster on a GPU
retain_graph | keep the graph after a backward so you can call backward again (rare; usually redo the forward)
%%%

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
%%% quiz
q: What is autograd's central trick? | a:2 | it picks the best model for you | it makes tensors run on the GPU | it records the forward steps into a graph and walks it backward to compute gradients | it stores your data in a file | fb: As the forward pass runs, autograd records every operation into the computation graph, then loss.backward() walks it backward to fill every .grad — your hand-derived backward pass, done for you.
q: Why must you call optimizer.zero_grad() each step? | a:1 | to reset the learning rate | because PyTorch adds new gradients on top of the old ones by default | to move tensors to the GPU | to build the model | fb: Gradients accumulate by default, so without a wipe this step's gradient piles on last step's — the #1 beginner bug.
q: What will PyTorch NOT do for you? | a:3 | compute correct gradients | fill every .grad automatically | run the same code on a GPU | choose your architecture, loss, or learning rate | fb: Autograd computes correct gradients for whatever graph you build, but it never decides whether the model, loss, or learning rate is a good idea — that judgment stays with you.
q: You call loss.backward() a second time and get a freed-graph error. The usual fix is? | a:0 | run the forward pass again to build a fresh graph | delete the optimizer | move the tensor to the CPU | lower the learning rate | fb: By default the graph is freed after the first backward to save memory; run the forward again for a fresh graph (or, rarely, pass retain_graph=True).
%%%

@@@ produce id=produce tag="Produce" title="Prove the mapping, then trip a bug on purpose" gotit="Done"
Time to see the whole picture with your own hands. Before you run anything, **predict** two things on paper: (1) which way the loss will move across epochs, and (2) what will happen to the loss if you *remove* the `optimizer.zero_grad()` line. Then build `experiment.py`: define a tiny model (an `nn.Linear` is enough, or your full 784→128→10 MLP), pick `nn.MSELoss` (or `nn.CrossEntropyLoss`) and `optim.SGD(model.parameters(), lr=...)`, and run the five-line loop — `zero_grad → forward → loss → backward → step` — printing `loss.item()` each epoch.

**What you should see:** the loss falling epoch after epoch — the four layers working together. Then run one deliberate experiment: **delete the `optimizer.zero_grad()` line** and watch the loss go strange as the gradients pile up. That's the #1 beginner bug, and now you can spot it in one glance. Put it back. For a victory lap, wrap a forward pass in `with torch.no_grad():` and notice the output tensor no longer carries `grad_fn` — you've turned off the graph, exactly as you would at evaluation. Then you're done — with the day, and with all of Module 5.

@@@ fin
