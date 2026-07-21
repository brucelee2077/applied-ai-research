---
quest_id: wf5-d05-pytorch
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 5 — The Same Model in PyTorch"
module_label: "Module 5 · Train · Day 5"
title: "The Same Model"
subtitle: "in PyTorch"
brand_sub: "Foundations · M5 Day 5"
spine: "practice: a machine learns to read by practicing on examples and being corrected"
nav_prev_href: "../day-04-training-loss-dropout/lesson.html"
nav_prev_label: "Full Training: The Loss &amp; Dropout"
nav_next_href: "../day-06-why-pytorch/lesson.html"
nav_next_label: "Why PyTorch Mirrors NumPy"
fin_title: "Module 5 · Day 5 complete! 🏆"
fin_body: "You did it! You finished <b>The Same Model in PyTorch</b>. You rebuilt the exact same 784→128→10 MLP you hand-wrote in numpy, but this time it shrank to a handful of lines that reach the same accuracy. You know what each new piece <i>replaces</i>: <b>nn.Module</b> is the box that holds your model and finds its weights; <b>nn.Linear</b> is your hand-written first-layer math, with the weights created for you; the <b>forward</b> method is your forward pass and nothing more. You know a <b>tensor</b> is a numpy array that also remembers how it was computed, that <b>autograd</b> records every step so <b>loss.backward()</b> can fill every <code>.grad</code> — replacing your whole hand-derived backward pass — and that <b>optimizer.step()</b> applies the update you used to write by hand. You can recite the five-line loop cold — <b>zero_grad → forward → loss → backward → step</b> — and you know the one habit that catches almost everyone: forgetting <b>zero_grad()</b>, because PyTorch adds gradients up by default. Most important: you <i>proved</i> it is the same model — same predictions, same learning, just far less typing.<br>Take a breath — you now own the tool every real lab trains with, and you own it without fear, because you built its insides by hand first. Next up: <b>Why PyTorch Mirrors NumPy</b>."
notebook_yardstick: 00-neural-networks/fundamentals/10_pytorch_equivalent.ipynb
coverage_topics:
  - {topic: tensors vs numpy arrays carry requires_grad and a device, keywords: [tensor, numpy array, requires_grad, device, torch.tensor, remembers how it was computed]}
  - {topic: reading a scalar out of a tensor with .item(), keywords: [.item(), read a number, scalar, plain python number, out of the tensor]}
  - {topic: nn.Linear holds weight and bias and does y equals x W transpose plus b, keywords: [nn.linear, in_features, out_features, weight, bias, fully-connected, y = xW]}
  - {topic: nn.Module base class you subclass with __init__ and forward, keywords: [nn.module, subclass, __init__, forward, base class, define the model]}
  - {topic: composing an MLP by stacking nn.Linear plus nn.ReLU, keywords: [stack, nn.relu, two-layer, mlp, compose, activation, 784, 128, 10]}
  - {topic: nn.Sequential chains layers so the forward pass is implicit, keywords: [nn.sequential, container, chain, implicit forward, snap together]}
  - {topic: parameters are tracked tensors and model.parameters enumerates them, keywords: [model.parameters, layer.weight, layer.bias, requires_grad, tracked, enumerate]}
  - {topic: autograd records operations and loss.backward computes every gradient, keywords: [autograd, loss.backward, records, graph, computes every gradient, .grad, replaces the backward pass]}
  - {topic: torch.optim SGD optimizer.step applies the update rule, keywords: [torch.optim, sgd, optimizer.step, update rule, applies the nudge, uses the .grad]}
  - {topic: optimizer.zero_grad clears gradients that accumulate by default, keywords: [zero_grad, gradients accumulate, add up, clear, reset, canonical beginner bug, forgetting]}
  - {topic: the canonical training loop zero_grad forward loss backward step, keywords: [training loop, zero_grad, forward, loss, backward, step, epochs, five lines]}
  - {topic: built-in loss module replaces a hand-written loss, keywords: [nn.mseloss, nn.bcewithlogitsloss, built-in loss, loss module, hand-written loss]}
  - {topic: learning rate is the optimizer lr argument that scales each update, keywords: [learning rate, lr, scales each update, same knob, step size]}
  - {topic: verifying the framework MLP reproduces the from-scratch result, keywords: [reproduces, same result, loss goes down, same decision boundary, prove it is the same model, matches]}
require_artifact: true
---

@@@ hero
@lede Imagine you spent a whole week hand-copying a long letter, word by word, checking every letter yourself. It works — you end up with a perfect copy — but your hand aches and it took forever. Then someone shows you a **photocopier**: you press one button, and out comes the same letter, identical, in two seconds. You did not lose anything you learned about the letter. You just stopped doing the tedious part by hand. That is exactly what today is. Over the last four days you hand-built a little machine that learns to read handwritten digits — the same **practice** loop of guessing and being corrected — where you wrote the forward pass, you *derived* the backward pass by hand, and you nudged every weight yourself. It was about 500 lines, and it worked. Today you meet the photocopier: a tool called **PyTorch** that rebuilds the *very same* 784→128→10 network in a handful of lines and reaches the *same* accuracy. Every real lab — the ones training the models behind ChatGPT — presses this same button. And here is the quiet superpower you have that most beginners don't: you already built the insides by hand, so you will *recognize* every piece PyTorch hands you. Nothing here is magic. It is your own machine, wearing a lighter coat.
@goal Here's what we'll figure out together today, one small idea at a time:
- what a **tensor** is (a numpy array that also *remembers how it was made*), and how numbers go in and come back out;
- the layer pieces — **nn.Linear**, **nn.ReLU**, **nn.Module**, **nn.Sequential** — and which hand-written line each one replaces;
- how **autograd** and **loss.backward()** do your entire hand-derived backward pass for free;
- the **optimizer** — how **.step()** applies the nudge you used to write yourself, and the one line, **zero_grad()**, that almost everyone forgets;
- the five-line **training loop** every real project uses — and how we *prove* it is the same model you built by hand.

@@@ concept id=c1 tag="Tensors" title="A tensor: a numpy array that remembers how it was made" gotit="Got tensors"
Before we touch a single layer, we need the thing everything in PyTorch is made of. You already know numpy arrays — grids of numbers you can add, multiply, and reshape. PyTorch has almost the exact same thing, and it even has almost the same name: a **[[tensor||PyTorch's version of a numpy array — the same grid of numbers, but it can also remember every step used to compute it, and can run on a GPU]]**. So the first, most freeing thing to hear is: *a tensor is just a numpy array with two extra powers.* If you can picture a numpy array, you can already picture a tensor.

**Think of a shopping receipt.** When you buy things at a store, you leave with two things: the groceries themselves, and a little paper receipt that lists *where each item came from and what it cost*. The groceries are the numbers. The receipt is a memory of how you got them. A plain numpy array is groceries with **no** receipt — you have the numbers, but no record of where they came from. A tensor is groceries **with** the receipt stapled on. That little receipt is what lets PyTorch, later, retrace your whole shopping trip backward — which is the secret behind the automatic backward pass you'll meet soon.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two shopping bags side by side. The left bag, labelled numpy array, has groceries but no receipt. The right bag, labelled tensor, has the same groceries plus a paper receipt stapled on that records where each number came from. The receipt is what lets PyTorch retrace the steps backward."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="20" y="20" width="470" height="150" rx="12" fill="#F5F2EC" stroke="#B7AEA0" stroke-width="1.2"/><g><path d="M70 70 L190 70 L178 150 L82 150 Z" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.6"/><text x="130" y="60" font-size="11" fill="#3B6FA0">numpy array</text><text x="130" y="100" font-size="10" fill="#2b2b2b">[0.5, 0.9, 0.1]</text><text x="130" y="120" font-size="9" fill="#8A6D3B">just the numbers</text><text x="130" y="140" font-size="9" fill="#C93B3B">no receipt</text></g><g><path d="M330 70 L450 70 L438 150 L342 150 Z" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="390" y="60" font-size="11" fill="#2D8B55">tensor</text><text x="390" y="100" font-size="10" fill="#2b2b2b">[0.5, 0.9, 0.1]</text><text x="390" y="120" font-size="9" fill="#2D8B55">same numbers</text><rect x="404" y="128" width="46" height="30" rx="2" fill="#FFFDF3" stroke="#C99A12" stroke-width="1.4"/><text x="427" y="140" font-size="6" fill="#8A6D3B">RECEIPT</text><text x="427" y="150" font-size="5" fill="#8A6D3B">how it was made</text></g><text x="260" y="180" font-size="9" fill="#6B645E">the receipt is what lets PyTorch retrace your steps backward</text></g></svg>
%%%

**What the receipt picture gets right:** the numbers are the same on both sides — a tensor is *not* a fancier kind of number, it is the *same* numbers with a memory attached. **Where it breaks down:** a real receipt is a flat list, but a tensor's memory is a little branching map of every add and multiply that led to it — and PyTorch only bothers keeping that map when you ask, by turning on a flag.

#### The two extra powers, in plain words
That flag is the first extra power. Every tensor has a setting called **[[requires_grad||a true/false flag on a tensor: when true, PyTorch keeps the receipt — it records every operation so it can compute gradients later]]**. When `requires_grad=True`, PyTorch staples on the receipt. When it's `False`, you get plain groceries — fast, but forgetful. Your input data doesn't need a receipt (it never changes), but your *weights* do (they're what you'll nudge), so PyTorch turns the flag on for weights automatically. The second extra power is that a tensor knows which **[[device||where a tensor lives and computes — the CPU, or a much faster GPU chip; today everything stays on the CPU]]** it lives on — a plain CPU, or a fast GPU. Today we keep everything on the CPU, so you can safely ignore this one — but now you know the word.

#### Numbers in, numbers out
You already have your digit data as numpy arrays. Getting them into PyTorch is one line — you wrap them with `torch.tensor(...)`. And when training is done and you want to *read* a single loss value back out as a normal Python number to print, you call **`.item()`** on it. That's the whole doorway: `torch.tensor` to go in, `.item()` to come back out. Try predicting the output before you reveal it.

%%% demo id=inout label="predict, then run: in with torch.tensor, out with .item()"
code: x = torch.tensor([0.5, 0.9, 0.1]); print(x); loss = (x ** 2).sum(); print(loss); print(loss.item())
out: tensor([0.5000, 0.9000, 0.1000])
out: tensor(1.0700)
out: 1.07
take: <b>Going in and coming out.</b> <code>torch.tensor([...])</code> turns your numbers into a tensor (notice it prints <code>tensor(...)</code>, not a bare list). Do math and you still have a tensor — <code>tensor(1.0700)</code> — carrying its receipt. Call <code>.item()</code> and you get a plain <code>1.07</code> you can print or store. Same numbers the whole way; only the wrapper changes.
%%%

You just met the raw material. A tensor is your familiar numpy array, plus a receipt (`requires_grad`) and a home (`device`). Hold onto the receipt idea — in a few concepts it becomes the trick that erases your entire hand-written backward pass.

@@@ concept id=c2 tag="nn.Linear" title="nn.Linear: your first-layer math, wired for you" gotit="Got nn.Linear"
Now the fun part — building the actual network, one piece at a time. Remember the very first thing your hand-built model did to a batch of pixels? It multiplied them by a weight matrix `W1` and added a bias `b1`: the line `Z1 = X @ W1 + b1`. You also had to *create* `W1` and `b1` yourself with the right shapes and a careful random start. PyTorch gives you a single object that does all of that: **[[nn.Linear||a ready-made fully-connected layer: it holds a weight matrix and a bias vector for you, and multiplying is one call — it is your hand-written X @ W + b in a box]]**.

**Think of a pre-wired junction box.** Picture an electrician's junction box on a wall. It has a row of input terminals on the left, a row of output terminals on the right, and inside, the wires are *already connected* in a fixed pattern. You don't wire it yourself — you just push signals in the left side and read what comes out the right side. `nn.Linear(784, 128)` is exactly that box: 784 terminals in, 128 terminals out, and the wiring (the weights) comes pre-installed. You tell it the two numbers — how many wires in, how many out — and it builds the matrix for you.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A junction box labelled nn.Linear(784,128). On its left edge are input terminals for 784 numbers; on its right edge are 128 output terminals. Inside, a grid of pre-installed wires connects them, labelled weight matrix W and bias b, created for you. The output equals x times W transpose plus b."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="150" y="30" width="220" height="140" rx="10" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.8"/><text x="260" y="24" font-size="11" fill="#7A5BA0">nn.Linear(784, 128)</text><line x1="60" y1="70" x2="150" y2="70" stroke="#3B6FA0" stroke-width="1.4"/><line x1="60" y1="100" x2="150" y2="100" stroke="#3B6FA0" stroke-width="1.4"/><line x1="60" y1="130" x2="150" y2="130" stroke="#3B6FA0" stroke-width="1.4"/><text x="60" y="60" font-size="9" fill="#3B6FA0">784 in</text><text x="60" y="150" font-size="8" fill="#3B6FA0">(the pixels)</text><rect x="175" y="55" width="170" height="90" rx="4" fill="#FFFFFF" stroke="#7A5BA0" stroke-width="1" stroke-dasharray="3 2"/><text x="260" y="95" font-size="10" fill="#7A5BA0">weight W + bias b</text><text x="260" y="112" font-size="9" fill="#2D8B55">created for you</text><text x="260" y="128" font-size="9" fill="#6B645E">y = xWᵀ + b</text><line x1="370" y1="80" x2="460" y2="80" stroke="#2D8B55" stroke-width="1.4"/><line x1="370" y1="120" x2="460" y2="120" stroke="#2D8B55" stroke-width="1.4"/><text x="460" y="70" font-size="9" fill="#2D8B55">128 out</text><text x="460" y="140" font-size="8" fill="#2D8B55">(the hidden layer)</text></g></svg>
%%%

**What the junction-box picture gets right:** the shape is fixed by the two numbers you give it, and you push data through without ever touching the wiring by hand. **Where it breaks down:** a real junction box's wiring never changes, but a layer's weights are *meant* to change — every training step nudges them a little, which is the whole point of learning.

#### Before and after: the same line, in one call
This is the heart of today, so let's put the two versions side by side. On the left is what you wrote by hand; on the right is the PyTorch version. Watch the whole block shrink to one line — and notice that PyTorch even *makes the weights for you*, with a sensible random start, so you don't set the shapes or the scale yourself.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A before-and-after comparison. On the left, three hand-written numpy lines create a random weight matrix, create a bias, and compute Z1 = X @ W1 + b1. On the right, a single PyTorch line: fc1 = nn.Linear(784, 128), then out = fc1(x). An arrow shows the three lines collapsing into one, with a note that PyTorch creates and shapes the weights for you."><g font-family="monospace" font-size="9" text-anchor="start"><text x="60" y="26" font-size="11" fill="#C93B3B" text-anchor="middle">by hand (numpy)</text><rect x="20" y="36" width="200" height="150" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/><text x="32" y="62" fill="#2b2b2b">W1 = np.random.randn(</text><text x="46" y="76" fill="#2b2b2b">784,128)*np.sqrt(2/784)</text><text x="32" y="100" fill="#2b2b2b">b1 = np.zeros(128)</text><text x="32" y="130" fill="#2b2b2b">Z1 = X @ W1 + b1</text><text x="32" y="160" fill="#8A6D3B">you shape + init</text><text x="32" y="174" fill="#8A6D3B">the weights</text><text x="260" y="105" font-size="20" fill="#7A5BA0" text-anchor="middle">→</text><text x="440" y="26" font-size="11" fill="#2D8B55" text-anchor="middle">PyTorch</text><rect x="300" y="36" width="200" height="150" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="312" y="70" fill="#2b2b2b">fc1 = nn.Linear(</text><text x="326" y="84" fill="#2b2b2b">784, 128)</text><text x="312" y="118" fill="#2b2b2b">Z1 = fc1(X)</text><text x="312" y="150" fill="#2D8B55">weights created</text><text x="312" y="164" fill="#2D8B55">+ shaped for you</text></g></svg>
%%%

Same math (`y = xWᵀ + b`), same shapes (784 in, 128 out), same random-but-sensible start. The three careful lines you wrote became `nn.Linear(784, 128)`. Let's see the shape come out exactly as you'd expect — predict it first.

%%% demo id=linshape label="predict the output shape, then run"
code: fc1 = nn.Linear(784, 128); batch = torch.randn(32, 784); out = fc1(batch); print(out.shape)
out: torch.Size([32, 128])
take: <b>32 images in, 32 hidden vectors out.</b> You handed it a batch of 32 pixel-rows (each 784 long) and got back 32 rows of 128 numbers — exactly the shape your hand-written <code>X @ W1 + b1</code> produced. The layer did the multiply and the bias-add for you; you only named the two sizes.
%%%

One box, two numbers, same result as three careful lines. Next we snap a few of these boxes together into your whole network.

@@@ concept id=c3 tag="nn.Module" title="nn.Module & nn.Sequential: the box that holds your model" gotit="Got the model box"
You have one junction box. Your real network has two of them with a bend in between: 784 → `nn.Linear` → **[[nn.ReLU||the same bend you built by hand: it keeps positive numbers and zeroes out negative ones, so the network can learn shapes a straight line never could]]** → `nn.Linear` → 10. The question is: what holds all these pieces together as *one model* you can train and save? That job belongs to **[[nn.Module||the base class every PyTorch model inherits from: you put your layers inside it, and in return it can find every weight in your whole model for you]]**.

**Think of a labeled toolbox.** Imagine a toolbox with a big handle on top. Inside, you place each tool — a hammer, a screwdriver, a wrench. The magic of this particular toolbox is that when you grab the handle and shake, it hands you a neat list of *every* tool inside, even tools tucked inside smaller boxes within it. `nn.Module` is that toolbox. You drop your layers in during setup, and later you grab one handle — `model.parameters()` — and it hands you *every weight and bias in the whole network*, ready to give to the trainer. You never hunt for weights by hand again.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A toolbox labelled nn.Module with a handle on top labelled model.parameters(). Inside the box sit two layers: nn.Linear 784 to 128 and nn.Linear 128 to 10, each holding a weight and bias. Grabbing the handle lists every weight and bias in the whole model at once."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="140" y="55" width="240" height="120" rx="8" fill="#EEF2F0" stroke="#3E7C63" stroke-width="1.8"/><rect x="220" y="30" width="80" height="26" rx="8" fill="#3E7C63"/><text x="260" y="47" font-size="9" fill="#FFFFFF">.parameters()</text><text x="260" y="24" font-size="10" fill="#3E7C63">the handle</text><rect x="160" y="80" width="90" height="80" rx="6" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.4"/><text x="205" y="100" font-size="8" fill="#7A5BA0">nn.Linear</text><text x="205" y="113" font-size="8" fill="#7A5BA0">784→128</text><text x="205" y="132" font-size="7" fill="#2b2b2b">weight</text><text x="205" y="144" font-size="7" fill="#2b2b2b">bias</text><rect x="270" y="80" width="90" height="80" rx="6" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.4"/><text x="315" y="100" font-size="8" fill="#7A5BA0">nn.Linear</text><text x="315" y="113" font-size="8" fill="#7A5BA0">128→10</text><text x="315" y="132" font-size="7" fill="#2b2b2b">weight</text><text x="315" y="144" font-size="7" fill="#2b2b2b">bias</text><text x="260" y="190" font-size="9" fill="#3E7C63">one grab → every weight + bias in the whole model</text></g></svg>
%%%

**What the toolbox picture gets right:** you fill it once, then one handle gives you everything inside — you never track the individual weights yourself. **Where it breaks down:** a real toolbox just *stores* tools; `nn.Module` also *runs* them — it knows the order the data flows through, which a toolbox never does.

#### Two beats: declare the layers, then define the path
Building a model is two small acts. First, in a setup method called `__init__`, you *declare* the layers (drop the tools in the box). Second, in a method called **[[forward||the method where you say how data flows through your layers, in order — it is literally your hand-written forward pass, nothing more]]**, you *define the path* the data takes. That `forward` method is not new magic — it is the exact forward pass you already wrote by hand, just written as a method. Here is the whole model:

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="Two ways to write the same MLP. On the left, a class MLP that subclasses nn.Module: __init__ declares fc1, relu, fc2, and forward passes x through fc1, relu, fc2 in order. On the right, the shorter nn.Sequential version stacking the same three layers so the forward path is automatic. Both produce the identical 784 to 128 to 10 network."><g font-family="monospace" font-size="8.5" text-anchor="start"><text x="130" y="24" font-size="10" fill="#7A5BA0" text-anchor="middle">subclass nn.Module</text><rect x="18" y="32" width="224" height="168" rx="8" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.3"/><text x="30" y="54" fill="#2b2b2b">class MLP(nn.Module):</text><text x="42" y="70" fill="#6B645E"># declare the layers</text><text x="42" y="86" fill="#2b2b2b">self.fc1 = nn.Linear(784,128)</text><text x="42" y="102" fill="#2b2b2b">self.relu = nn.ReLU()</text><text x="42" y="118" fill="#2b2b2b">self.fc2 = nn.Linear(128,10)</text><text x="42" y="140" fill="#6B645E"># define the path</text><text x="42" y="156" fill="#2b2b2b">def forward(self, x):</text><text x="54" y="170" fill="#2b2b2b">x = self.relu(self.fc1(x))</text><text x="54" y="184" fill="#2b2b2b">return self.fc2(x)</text><text x="278" y="24" font-size="20" fill="#3E7C63" text-anchor="middle">≡</text><text x="400" y="24" font-size="10" fill="#2D8B55" text-anchor="middle">nn.Sequential</text><rect x="300" y="32" width="212" height="168" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="312" y="60" fill="#2b2b2b">model = nn.Sequential(</text><text x="326" y="80" fill="#2b2b2b">nn.Linear(784, 128),</text><text x="326" y="100" fill="#2b2b2b">nn.ReLU(),</text><text x="326" y="120" fill="#2b2b2b">nn.Linear(128, 10),</text><text x="312" y="140" fill="#2b2b2b">)</text><text x="312" y="168" fill="#6B645E"># the path is automatic:</text><text x="312" y="184" fill="#6B645E"># top to bottom, in order</text></g></svg>
%%%

Both boxes build the **identical** network. The left one, subclassing `nn.Module`, is what you'll see in every real codebase because you can write any path you like inside `forward`. The right one, **[[nn.Sequential||a container that chains layers in a straight line, so it runs them top-to-bottom for you and you never write a forward method]]**, is a shortcut for the common case where data just flows straight through, one layer after the next — like train cars linked in a line, each passing the passengers to the next.

#### The handle really works: count the weights
Because your layers live inside the module, `model.parameters()` hands you every weight and bias as tracked tensors (each one already has `requires_grad=True` — the receipt is on). Let's grab the handle and count what's inside — predict roughly how many groups of numbers a two-layer network has first.

%%% demo id=params label="predict how many parameter groups, then run"
code: model = MLP(); [print(name, tuple(p.shape)) for name, p in model.named_parameters()]
out: fc1.weight (128, 784)
out: fc1.bias (128,)
out: fc2.weight (10, 128)
out: fc2.bias (10,)
take: <b>Four groups — exactly your two W's and two b's.</b> The handle found every weight and bias in the whole model without you naming them one by one. These are the *same* four number-blocks you created and shaped by hand; PyTorch just keeps them tidy inside the box and hands the whole list to the trainer in one move.
%%%

You now have a complete model in one box, and one handle that surrenders all its weights. Next comes the piece that made all this worth it: how PyTorch fills in the gradients for those weights *without you deriving a single one*.

@@@ concept id=c4 tag="autograd" title="autograd: your whole backward pass, done for free" gotit="Got autograd"
This is the concept that makes grown engineers grin. On day 2 you spent hours deriving the **backward pass** by hand — the chain rule, layer by layer, computing how much each weight should change. It was the hardest thing you built. Today PyTorch does *all* of it, correctly, in one line, for *any* network you can dream up. The engine behind this is called **[[autograd||PyTorch's automatic gradient engine: it silently records every operation during the forward pass, then retraces them to compute the gradient of the loss for every weight — your hand-derived backward pass, done for you]]**.

**Think of a GPS breadcrumb tracker on a hike.** Imagine you go on a long, winding hike. You start a little GPS tracker that drops a breadcrumb at every turn: left here, uphill there, right at the big rock. When you want to get home, you don't need a map or to remember the way — the tracker simply *plays your trail backward*, breadcrumb by breadcrumb, and walks you straight back to the start. Autograd does this with math. As data flows *forward* through your layers, autograd drops a breadcrumb at every operation (every multiply, every add, every ReLU). Then, when you ask, it walks that trail *backward* to work out exactly how each weight affected the final loss. That backward walk *is* your backward pass. Remember the tensor's receipt from concept 1? These breadcrumbs are that receipt.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A winding trail of breadcrumbs from Start through several operation stops to the final loss. Green arrows go forward along the trail as data flows through multiply, add, and ReLU operations. Then a red dashed arrow retraces the exact same trail backward from loss to start, which is loss.backward computing every gradient."><g font-family="monospace" font-size="9" text-anchor="middle"><circle cx="50" cy="70" r="14" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.4"/><text x="50" y="73" font-size="8" fill="#3B6FA0">input</text><circle cx="170" cy="55" r="13" fill="#F5F2EC" stroke="#B7AEA0"/><text x="170" y="58" font-size="7">×W</text><circle cx="290" cy="80" r="13" fill="#F5F2EC" stroke="#B7AEA0"/><text x="290" y="83" font-size="7">ReLU</text><circle cx="410" cy="55" r="13" fill="#F5F2EC" stroke="#B7AEA0"/><text x="410" y="58" font-size="7">×W</text><circle cx="480" cy="90" r="15" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="480" y="93" font-size="8" fill="#C93B3B">loss</text><path d="M64 68 L157 57" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/><path d="M182 58 L278 77" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/><path d="M303 78 L398 58" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/><path d="M423 59 L466 85" stroke="#2D8B55" stroke-width="1.6" marker-end="url(#af)"/><path d="M466 108 Q260 165 55 92" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="6 4" fill="none" marker-end="url(#ar)"/><text x="230" y="40" font-size="9" fill="#2D8B55">forward: drop a breadcrumb at each step</text><text x="260" y="158" font-size="9" fill="#C93B3B">loss.backward(): retrace the trail → every gradient</text><defs><marker id="af" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker><marker id="ar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#C93B3B"/></marker></defs></g></svg>
%%%

**What the breadcrumb picture gets right:** you only ever walk *forward* on purpose; the way back is reconstructed automatically from the trail you already left. **Where it breaks down:** a hiking trail is a single line, but autograd's trail can branch and merge — it's really a little map of operations, and it applies the chain rule at every fork, which is the calculus you did by hand.

#### Before and after: pages of chain rule become one line
Here is the payoff, side by side. On the left, a sketch of the backward pass you wrote on day 2 — gradient of the loss, pushed back through layer 2, through the ReLU, through layer 1, each step its own careful formula. On the right, the PyTorch version. Watch the entire block vanish into `loss.backward()`.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Before and after of the backward pass. On the left, a stack of five hand-derived chain-rule lines computing dZ2, dW2, dA1, dZ1, dW1. On the right, a single line: loss.backward(). An arrow shows the whole stack collapsing to one line, with a note that autograd fills every .grad automatically."><g font-family="monospace" font-size="8.5" text-anchor="start"><text x="120" y="24" font-size="10" fill="#C93B3B" text-anchor="middle">by hand (day 2)</text><rect x="18" y="32" width="204" height="164" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/><text x="30" y="56" fill="#2b2b2b">dZ2 = (A2 - Y) / m</text><text x="30" y="76" fill="#2b2b2b">dW2 = A1.T @ dZ2</text><text x="30" y="96" fill="#2b2b2b">dA1 = dZ2 @ W2.T</text><text x="30" y="116" fill="#2b2b2b">dZ1 = dA1 * (Z1 &gt; 0)</text><text x="30" y="136" fill="#2b2b2b">dW1 = X.T @ dZ1</text><text x="30" y="166" fill="#8A6D3B">...and the biases,</text><text x="30" y="180" fill="#8A6D3B">every layer, by hand</text><text x="262" y="115" font-size="20" fill="#7A5BA0" text-anchor="middle">→</text><text x="410" y="24" font-size="10" fill="#2D8B55" text-anchor="middle">PyTorch</text><rect x="300" y="32" width="204" height="164" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="312" y="100" fill="#2b2b2b" font-size="11">loss.backward()</text><text x="312" y="130" fill="#2D8B55">fills every .grad</text><text x="312" y="146" fill="#2D8B55">for every weight,</text><text x="312" y="162" fill="#2D8B55">automatically</text></g></svg>
%%%

Every one of those hand-derived lines — and the ones for the biases, and every layer — becomes the single call `loss.backward()`. It reads the loss tensor's breadcrumb trail and, for each weight, deposits the gradient into a slot on that weight called `.grad`. Let's watch a gradient appear where there was nothing — this is the moment your hand-derived math becomes automatic. Predict: before we call backward, what is a fresh weight's `.grad`?

%%% demo id=autograd label="predict .grad before and after backward, then run"
code: w = torch.tensor([2.0], requires_grad=True); print("before:", w.grad); loss = (3 * w + 1) ** 2; loss.backward(); print("after: ", w.grad)
out: before: None
out: after:  tensor([42.])
take: <b>Empty, then filled — with no chain rule from you.</b> Before <code>backward()</code>, the gradient slot is <code>None</code> (nothing recorded yet). After one <code>loss.backward()</code>, autograd retraced the trail and dropped the exact gradient into <code>w.grad</code>. (Sanity check: the slope of <code>(3w+1)²</code> at w=2 is <code>2·(3·2+1)·3 = 42</code> — the same number the chain rule would have given you by hand.)
%%%

Take a breath — you just watched the hardest thing you built get done for you, correctly, in one line. But a gradient sitting in `.grad` doesn't move a weight on its own. Someone still has to *take the step*. That someone is the optimizer.

@@@ concept id=c5 tag="Optimizer & the loop" title="The optimizer, zero_grad, and the five-line loop" gotit="Got the loop"
Autograd fills every `.grad` with *which way to nudge* each weight. Now a helper reads those `.grad` slots and actually *applies* the nudge. In your hand-built code you wrote the nudge yourself: `W1 -= learning_rate * dW1`, one line per weight. PyTorch bundles that into an **[[optimizer||a helper that reads every weight's .grad and applies the update rule for you: calling optimizer.step() nudges every weight, replacing your hand-written W -= lr * dW lines]]**. The simplest one, `torch.optim.SGD`, does the *exact* rule you wrote by hand.

**Think of a golf coach standing behind you.** You take a swing (the forward pass). The coach sees where the ball landed and how far off it was (the loss and its gradients). Then the coach gently adjusts your grip and stance — a little this way, a little that way — before your next swing. `optimizer.step()` is that adjustment: it reads every gradient and nudges every weight a little in the better direction. And the **[[learning rate||the optimizer's lr setting: how big each nudge is — the same step-size knob you set by hand; too small and learning crawls, too big and it overshoots]]**, which you pass in as `lr=...`, is simply *how big* each nudge is — the same step-size knob you tuned in your hand-built loop.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A golf coach reading the gradients and nudging each weight. On the left, four .grad slots hold arrows pointing which way to nudge. In the middle, optimizer.step() reads them. On the right, the four weights each shift a little in the better direction, by an amount set by the learning rate lr."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="70" y="24" font-size="9" fill="#C93B3B">.grad slots (which way)</text><rect x="30" y="34" width="80" height="110" rx="6" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.2"/><text x="70" y="56" font-size="8">w1.grad ↗</text><text x="70" y="78" font-size="8">b1.grad ↘</text><text x="70" y="100" font-size="8">w2.grad ↗</text><text x="70" y="122" font-size="8">b2.grad ↘</text><rect x="185" y="60" width="150" height="55" rx="8" fill="#EEF2F0" stroke="#3E7C63" stroke-width="1.6"/><text x="260" y="84" font-size="10" fill="#3E7C63">optimizer.step()</text><text x="260" y="102" font-size="8" fill="#6B645E">nudge size = lr</text><path d="M112 89 L183 89" stroke="#3E7C63" stroke-width="1.5" marker-end="url(#a2)"/><path d="M337 89 L406 89" stroke="#2D8B55" stroke-width="1.5" marker-end="url(#a3)"/><rect x="410" y="34" width="80" height="110" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.2"/><text x="450" y="24" font-size="9" fill="#2D8B55">weights nudged</text><text x="450" y="70" font-size="8">w1 ← w1 − lr·g</text><text x="450" y="92" font-size="8">b1 ← ...</text><text x="450" y="114" font-size="8">every weight</text><defs><marker id="a2" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#3E7C63"/></marker><marker id="a3" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker></defs></g></svg>
%%%

**What the coach picture gets right:** the coach only adjusts *after* seeing the result, and by a small amount, then you try again — exactly the guess-correct-repeat rhythm of **practice** you've built all week. **Where it breaks down:** a coach uses judgment and feel; the optimizer just follows a fixed arithmetic rule (`weight = weight − lr × gradient`), no intuition involved.

#### The five-line loop every project uses
Now snap the pieces into the loop. You build the model once and pick an optimizer once (`optimizer = optim.SGD(model.parameters(), lr=0.1)` — notice `model.parameters()`, the toolbox handle from concept 3, handing every weight to the optimizer). Then each round repeats **five lines**, and the order matters. Read them as one sentence: *clear the scoreboard, guess, score, retrace, step.*

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="The five-line training loop drawn as a cycle: zero_grad clears the gradient slots, forward makes a guess, loss scores it, backward retraces the trail to fill gradients, step nudges the weights, then back to zero_grad for the next round."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="20" font-size="10" fill="#6B645E">one round, repeated over epochs</text><rect x="30" y="90" width="86" height="44" rx="8" fill="#FBECEC" stroke="#C93B3B" stroke-width="1.3"/><text x="73" y="110" font-size="8" fill="#C93B3B">1. zero_grad()</text><text x="73" y="124" font-size="7" fill="#6B645E">clear slots</text><rect x="146" y="40" width="86" height="44" rx="8" fill="#E8EEF5" stroke="#3B6FA0" stroke-width="1.3"/><text x="189" y="60" font-size="8" fill="#3B6FA0">2. forward</text><text x="189" y="74" font-size="7" fill="#6B645E">out = model(x)</text><rect x="262" y="40" width="86" height="44" rx="8" fill="#EFEAF5" stroke="#7A5BA0" stroke-width="1.3"/><text x="305" y="60" font-size="8" fill="#7A5BA0">3. loss</text><text x="305" y="74" font-size="7" fill="#6B645E">criterion(out,y)</text><rect x="378" y="90" width="100" height="44" rx="8" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.3"/><text x="428" y="110" font-size="8" fill="#8A6D3B">4. loss.backward()</text><text x="428" y="124" font-size="7" fill="#6B645E">fill .grad</text><rect x="262" y="150" width="100" height="44" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="312" y="170" font-size="8" fill="#2D8B55">5. optimizer.step()</text><text x="312" y="184" font-size="7" fill="#6B645E">nudge weights</text><path d="M116 100 Q120 60 144 58" stroke="#B7AEA0" stroke-width="1.3" fill="none" marker-end="url(#a4)"/><path d="M232 62 L260 62" stroke="#B7AEA0" stroke-width="1.3" marker-end="url(#a4)"/><path d="M348 66 Q385 70 400 88" stroke="#B7AEA0" stroke-width="1.3" fill="none" marker-end="url(#a4)"/><path d="M410 134 Q385 158 364 166" stroke="#B7AEA0" stroke-width="1.3" fill="none" marker-end="url(#a4)"/><path d="M262 176 Q150 178 90 136" stroke="#B7AEA0" stroke-width="1.3" fill="none" marker-end="url(#a4)"/><defs><marker id="a4" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#B7AEA0"/></marker></defs></g></svg>
%%%

That cycle — `zero_grad → forward → loss → backward → step` — is the beating heart of *every* PyTorch project on earth, from your digit model to the ones training ChatGPT. It is the same guess-correct-repeat **practice** loop you built by hand, wearing five short names.

#### The one line everyone forgets — and why
Look at line 1: `optimizer.zero_grad()`. Why must you *clear* the gradients every round? Because PyTorch **adds them up by default** — each `loss.backward()` *piles* new gradients on top of whatever was in `.grad` already, instead of replacing them.

**Think of a scoreboard that never resets.** Imagine a scoreboard that keeps adding each round's points to the last round's total, and there's no automatic reset. If you forget to wipe it, round 3 shows rounds 1+2+3 stacked together — a wrong, ever-growing number. Your gradients are that scoreboard. **[[zero_grad||the line that wipes every weight's .grad back to zero before backward; skip it and PyTorch piles this round's gradients on top of the last round's, so the nudges grow wrong — the single most common PyTorch beginner bug]]** is the wipe. Skip it and your nudges bloat every step and training goes haywire. Predict what happens to the gradient if we call backward twice without clearing.

%%% demo id=zerograd label="predict: call backward twice without zero_grad, then run"
code: w = torch.tensor([2.0], requires_grad=True); (3*w).backward(); print("after 1st:", w.grad); (3*w).backward(); print("after 2nd:", w.grad)
out: after 1st: tensor([3.])
out: after 2nd: tensor([6.])
take: <b>3, then 6 — it added, it did not replace.</b> The true gradient is 3 both times, but the second <code>backward()</code> <i>piled</i> another 3 on top, giving 6. Over a real training loop this snowballs into nonsense. Calling <code>optimizer.zero_grad()</code> first wipes the slot back to 0 each round so every step uses this round's gradient only. Forgetting it is the #1 PyTorch beginner bug — now it will never catch you.
%%%

!!! c-info 🧮
<b>Optional (skippable) — why accumulation is a feature, not a mistake.</b> PyTorch adds gradients on purpose so advanced users can split one big batch across several `backward()` calls and sum the gradients (useful when a batch is too big for memory). For everything you do now, you want fresh gradients each step, so you always call `zero_grad()` first. You never have to use the accumulation trick today — just know that's *why* the default is "add," and `zero_grad()` is how you opt out.
!!!

You now own the whole engine: model, autograd, optimizer, and the five-line loop. The last thing left is the most satisfying — proving it's genuinely the *same model* you built by hand.

@@@ concept id=c6 tag="Same model" title="Proving it: same loss, same learning, far less typing" gotit="Got the proof"
There's one small piece left to swap, and then a promise to keep. The piece: your hand-written loss. On day 4 you coded cross-entropy yourself; PyTorch hands you a ready-made **[[loss module||a built-in loss like nn.MSELoss (for numbers) or nn.BCEWithLogitsLoss (for yes/no) — you create it once and call it, replacing the loss formula you wrote by hand]]** — you pick one, create it once (e.g. `criterion = nn.MSELoss()`), and call it. That's the third and last hand-written block to disappear. The promise: after all this shrinking, is it truly the *same* network? We're going to check, not just claim.

**Think of two loaves of bread.** You bake one loaf by hand — measuring flour, kneading, waiting. Your friend makes the *same* loaf in a bread machine — same flour, same recipe, one button. When they come out of the oven, you taste both. If they taste identical, the machine didn't cheat; it just did your steps for you. That taste test is our proof: we run the from-scratch model and the PyTorch model on the same digits, and check that the loss falls the same way and the accuracy lands in the same place.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A loss-versus-epochs chart with two curves that fall together and nearly overlap. One solid curve is labelled by hand, numpy; one dashed curve is labelled PyTorch. Both start high and drop to the same low loss, showing the two models learn the same way and reach the same result."><g font-family="monospace" font-size="10" text-anchor="middle"><line x1="60" y1="30" x2="60" y2="150" stroke="#6B645E" stroke-width="1.2"/><line x1="60" y1="150" x2="470" y2="150" stroke="#6B645E" stroke-width="1.2"/><text x="30" y="90" font-size="9" fill="#6B645E" transform="rotate(-90 30 90)">loss</text><text x="265" y="172" font-size="9" fill="#6B645E">epochs →</text><path d="M60 45 C140 70 220 118 300 135 C360 145 420 147 465 148" fill="none" stroke="#C93B3B" stroke-width="2"/><path d="M60 50 C140 74 220 121 300 137 C360 146 420 148 465 149" fill="none" stroke="#2D8B55" stroke-width="2" stroke-dasharray="5 4"/><rect x="330" y="45" width="12" height="3" fill="#C93B3B"/><text x="400" y="51" font-size="9" fill="#C93B3B">by hand (numpy)</text><rect x="330" y="63" width="12" height="3" fill="#2D8B55"/><text x="392" y="69" font-size="9" fill="#2D8B55">PyTorch</text><text x="265" y="24" font-size="9" fill="#6B645E">same curve → same model</text></g></svg>
%%%

**What the two-loaves picture gets right:** identical recipe plus identical ingredients gives an identical result, no matter who (or what) does the steps. **Where it breaks down:** with a fixed random seed the two match almost perfectly, but tiny rounding differences in the arithmetic mean the curves may not be *byte-for-byte* identical — they'll sit right on top of each other, which is all we need.

#### The whole loop, assembled
Here is the entire PyTorch training run — the pieces from every concept today, snapped together. Read it and notice: you've met every single line already.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="The full PyTorch training script. It builds the MLP model, creates the loss criterion nn.CrossEntropyLoss and the optimizer SGD with a learning rate, then loops over epochs running zero_grad, forward, loss, backward, step, printing loss.item() each epoch to watch it fall."><g font-family="monospace" font-size="8.5" text-anchor="start"><rect x="20" y="20" width="480" height="172" rx="8" fill="#F5F2EC" stroke="#B7AEA0" stroke-width="1.2"/><text x="34" y="42" fill="#2b2b2b">model = MLP()</text><text x="34" y="60" fill="#2b2b2b">criterion = nn.CrossEntropyLoss()</text><text x="220" y="60" fill="#6B645E"># built-in loss</text><text x="34" y="78" fill="#2b2b2b">optimizer = optim.SGD(model.parameters(), lr=0.1)</text><text x="34" y="102" fill="#2b2b2b">for epoch in range(20):</text><text x="52" y="120" fill="#C93B3B">optimizer.zero_grad()</text><text x="230" y="120" fill="#6B645E"># 1 clear</text><text x="52" y="138" fill="#3B6FA0">out = model(X)</text><text x="230" y="138" fill="#6B645E"># 2 forward</text><text x="52" y="156" fill="#7A5BA0">loss = criterion(out, Y)</text><text x="230" y="156" fill="#6B645E"># 3 score</text><text x="52" y="174" fill="#C99A12">loss.backward()</text><text x="230" y="174" fill="#6B645E"># 4 retrace</text><text x="52" y="192" fill="#2D8B55">optimizer.step()</text><text x="230" y="192" fill="#6B645E"># 5 nudge</text></g></svg>
%%%

That's it — the ~500 lines you wrote by hand, now a couple dozen. Let's run a tiny version and watch the loss fall, the same way it fell in your hand-built model. Predict: over the epochs, which way does the number go?

%%% demo id=proof label="predict the direction, then run the loop"
code: model = nn.Linear(4, 1); crit = nn.MSELoss(); opt = optim.SGD(model.parameters(), lr=0.1)
code: x = torch.randn(8, 4); y = torch.randn(8, 1)
code: for epoch in range(4):
code:     opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
code:     print(f"epoch {epoch}  loss {loss.item():.3f}")
out: epoch 0  loss 1.842
out: epoch 1  loss 1.517
out: epoch 2  loss 1.264
out: epoch 3  loss 1.068
take: <b>Down, down, down — it's learning.</b> Same five lines, same falling loss you saw in your hand-built run. Notice <code>loss.item()</code> pulling the plain number out to print. This is the taste test passing: the framework model learns exactly the way your from-scratch model did — just with far less typing.
%%%

Victory lap: you didn't just *trust* that PyTorch is your model — you *proved* it, line by line. Every piece maps back to something you built by hand. That is a rare and real superpower.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Here's the whole day in a few beats. The theme never changed: **practice** — a machine learns by guessing, being corrected, and repeating — and PyTorch is just the tool that does the tedious parts of that loop for you.

- A **tensor** is a numpy array with a receipt (`requires_grad`) and a home (`device`). Numbers go in with `torch.tensor`, and come out as plain values with `.item()`.
- **nn.Linear(in, out)** is your `X @ W + b`, with the weights made and shaped for you. Stack two of them with an **nn.ReLU** bend between, and you have your MLP.
- **nn.Module** is the box that holds your layers; `model.parameters()` hands you every weight. **nn.Sequential** is the shortcut for a straight-through path.
- **autograd** records the forward trail, and **loss.backward()** retraces it to fill every `.grad` — your whole hand-derived backward pass, gone.
- The **optimizer** (`SGD`) reads those `.grad` slots and `optimizer.step()` nudges every weight, using the **lr** step-size knob you already know.
- The five-line loop is **zero_grad → forward → loss → backward → step**. Never forget `zero_grad()` — PyTorch *adds* gradients by default, so you must wipe the scoreboard each round.

Here is the single most useful thing to keep — the map from what you built by hand to what PyTorch calls it:

%%% table
:: What you wrote by hand :: PyTorch version :: What changed
Create + shape `W`, `b` :: `nn.Linear(in, out)` :: PyTorch makes the weights for you
`Z = X @ W + b` :: `layer(x)` :: one call does the multiply + bias
Your whole `backward()` :: `loss.backward()` :: autograd does all the calculus
`W -= lr * dW` (each weight) :: `optimizer.step()` :: one call nudges every weight
Reset your own gradient sums :: `optimizer.zero_grad()` :: required — gradients add up by default
Hand-written loss formula :: `nn.CrossEntropyLoss()` etc. :: a ready-made loss module
Read a value to print :: `loss.item()` :: pulls a plain number out of a tensor
%%%

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="The five-line loop as a compact ring: zero_grad, forward, loss, backward, step, arrow curving back to zero_grad, labelled the heart of every PyTorch project."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="45" y="60" fill="#C93B3B">zero_grad</text><text x="135" y="60" fill="#3B6FA0">forward</text><text x="215" y="60" fill="#7A5BA0">loss</text><text x="300" y="60" fill="#C99A12">backward</text><text x="390" y="60" fill="#2D8B55">step</text><text x="90" y="60" fill="#B7AEA0">→</text><text x="178" y="60" fill="#B7AEA0">→</text><text x="258" y="60" fill="#B7AEA0">→</text><text x="345" y="60" fill="#B7AEA0">→</text><path d="M415 55 Q470 30 470 75 Q470 95 45 72" fill="none" stroke="#B7AEA0" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#a5)"/><text x="260" y="30" fill="#6B645E">the heart of every PyTorch project — repeated over epochs</text><defs><marker id="a5" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#B7AEA0"/></marker></defs></g></svg>
%%%

Keep this cheat-sheet close — every term you'll meet in a real codebase is right here:

%%% jargon
tensor | a numpy array that also remembers how it was made (requires_grad) and where it lives (device)
requires_grad | the true/false flag that turns on a tensor's receipt so gradients can be computed
.item() | pulls a single number out of a tensor as a plain Python value, for printing or storing
nn.Linear(in, out) | a ready-made layer holding weight + bias; does y = xWᵀ + b for you
nn.ReLU | the bend: keeps positives, zeroes negatives — same activation you built by hand
nn.Module | the base class your model subclasses; holds layers and finds every weight
forward | the method where you define how data flows through your layers — your forward pass
nn.Sequential | a container that chains layers so the forward path is automatic
model.parameters() | hands you every weight and bias in the model, for the optimizer
autograd | the engine that records the forward trail and computes all gradients on request
loss.backward() | retraces the trail and fills every weight's .grad — replaces the hand-derived backward pass
optimizer / .step() | reads every .grad and nudges every weight (SGD does W -= lr·grad)
zero_grad() | wipes .grad to zero each round; skip it and gradients pile up (the #1 beginner bug)
learning rate (lr) | how big each nudge is — the step-size knob you already know
%%%

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
%%% quiz
q: What does loss.backward() replace from your hand-built model? | a:2 | the forward pass | the loss formula | your whole hand-derived backward pass | the data loading | fb: Autograd retraces the recorded trail and fills every .grad — the exact backward pass you derived by hand.
q: Why must you call optimizer.zero_grad() each round? | a:1 | to reset the learning rate | because PyTorch adds gradients up by default | to reload the data | to switch to eval mode | fb: Gradients accumulate by default, so without a wipe this round's gradients pile on last round's — the #1 beginner bug.
q: What is the correct order of the five-line loop? | a:3 | forward → step → loss → backward → zero_grad | step → backward → loss → forward → zero_grad | loss → forward → backward → zero_grad → step | zero_grad → forward → loss → backward → step | fb: Clear the scoreboard, guess, score, retrace, nudge — zero_grad → forward → loss → backward → step.
q: What is a tensor, compared to a numpy array? | a:0 | the same numbers, plus a memory of how it was made and where it lives | a completely new kind of number | only usable on a GPU | a picture of a network | fb: A tensor is a numpy array with two extra powers — requires_grad (its receipt) and a device (its home).
%%%

@@@ produce id=produce tag="Produce" title="Predict the shrink, then run it and watch the same model learn" gotit="Done"
Time to press the photocopier button yourself. Before you run anything, **predict** two things on paper: (1) roughly how many lines your PyTorch version will be compared to the ~500 hand-written ones, and (2) which way the loss number will move across epochs. Then build `experiment.py`: define the same 784→128→10 MLP with two `nn.Linear` layers and an `nn.ReLU`, wrap it in `nn.Module` (or `nn.Sequential`), pick `nn.CrossEntropyLoss` and `optim.SGD`, and run the five-line loop — `zero_grad → forward → loss → backward → step` — printing `loss.item()` each epoch.

**What you should see:** the loss falling epoch after epoch, landing near the same place your hand-built model reached — the taste test passing. Then run one deliberate experiment: **delete the `optimizer.zero_grad()` line** and watch what happens to the loss. Notice how it goes strange as the gradients pile up — that is the #1 beginner bug, and now you can spot it in one glance. Put it back, and you're done.

@@@ fin








