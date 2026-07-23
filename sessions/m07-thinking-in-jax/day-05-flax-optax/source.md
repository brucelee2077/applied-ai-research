---
quest_id: w01-d05-flax
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Day 5 — Flax & Optax: Neural Nets, Functionally"
module_label: "Thinking in JAX · Week 1 · Day 5"
title: "Flax and Optax"
subtitle: "Neural Networks, the Functional Way"
brand_sub: "Frontier Lab · Week 1 Day 5"
spine: "toolkit"
nav_prev_href: "../day-04-jit/lesson.html"
nav_prev_label: "jit: Compilation"
nav_next_href: "../day-06-vit-capstone/lesson.html"
nav_next_label: "ViT Capstone (bridge)"
fin_title: "Week 1 · Day 5 complete! 🏆"
fin_body: "Great work — you've met the JAX neural-net <b>toolkit</b>: <b>Flax</b> to define a model whose parameters live <i>outside</i> it (a pytree you pass in with <code>apply</code>), and <b>Optax</b> to update those parameters with a composable optimizer whose state you also carry explicitly. Put together, they give you a clean functional training step: compute gradients, update, repeat.<br>Next up: the <b>ViT Capstone</b>."
notebook_yardstick: null
---

@@@ hero
@lede Think about your favourite pizza place. The **recipe** — how much dough, how long in the oven — is written on a card. But the actual dough, cheese, and toppings sit in the fridge, *outside* the recipe card. The chef takes the ingredients out, follows the card, and makes a pizza. Change the ingredients, and the same card still works. That simple split — steps here, ingredients there — is the whole idea behind today's **toolkit** for building neural networks in **JAX** (the library behind many of the biggest AI models, including work at Google DeepMind). We'll meet two friends. **Flax** writes the recipe card — the shape of your neural network — while keeping the ingredients (the numbers the model learns, called *parameters*) in a separate box you hold yourself. And **Optax** is the little helper that, after each taste-test, tells you exactly how to tweak those ingredients to make the next pizza better. Yesterday you learned to make functions fast with `jit`; today you use that speed to build and improve a real, tiny neural network — the same two tools researchers reach for to train models at the frontier.
@goal Together we'll meet the <b>toolkit</b> — <b>Flax</b> for defining a neural network whose learnable numbers (parameters) live in a box <i>outside</i> the model, and <b>Optax</b> for nudging those numbers in the right direction. You'll see the two-step Flax life — <code>init</code> to create the parameter box, <code>apply</code> to run the model — why keeping parameters outside makes the model a clean function that <code>jit</code> and <code>grad</code> love, how Optax turns a gradient into an update you add to your parameters, and the handful of beginner traps (and their fixes) — like forgetting to catch the new box the update hands back. Every new word gets explained the moment it appears.

@@@ concept id=c1 tag="Recipe vs. ingredients" title="Why a toolkit — and why parameters live OUTSIDE the model" gotit="Got the big idea"
Let's start with the picture that makes everything today click, and it's home base for the whole day: a **recipe card and a fridge.** The recipe card says *what to do* — mix the dough, bake it. The fridge holds the *ingredients* — the flour, the cheese. The card never stores the flour inside itself; the ingredients sit outside, and you hand them to the recipe only when you cook.

Now the idea. A neural network has two very different kinds of stuff inside it. First, its **shape**: how many layers, what each layer does, how the numbers flow through. Second, its **numbers**: the actual weights the model *learns* — thousands or billions of them. These learnable numbers are called [[parameters||The numbers a neural network learns during training — its weights and biases. They start random and get nudged toward better values.]]. Most older toolkits glue these two together — shape and numbers tangled inside one object. Today's **toolkit** does the recipe-and-fridge thing instead: it keeps the *shape* in one place (a **Flax** model, the recipe card) and the *numbers* in a separate box you hold in your own hands (the [[params||The box — a pytree, a nested dictionary of arrays — that holds all of a model's learnable numbers, kept OUTSIDE the Flax model. You pass it in every time you run the model.]] box, the fridge).

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A recipe-card and fridge analogy. On the left, a recipe card labelled 'Flax model: the SHAPE (what to do), no numbers inside'. On the right, a fridge labelled 'params box: the NUMBERS the model learned, held OUTSIDE'. An arrow from the fridge into the recipe card, labelled 'you hand the ingredients to the recipe each time you cook'. A label at the bottom reads: shape and numbers are kept separate — that is the whole trick.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="18" fill="#2C2A28" font-size="12">Flax keeps the SHAPE and the NUMBERS in two separate places</text>
<text x="118" y="44" fill="#5E5191">the recipe card (Flax model)</text>
<g transform="translate(40,54)"><rect x="0" y="0" width="156" height="110" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="78" y="24" fill="#5E5191" font-size="9">SHAPE / steps only</text><line x1="18" y1="38" x2="138" y2="38" stroke="#8579b8" stroke-dasharray="3 3"/><text x="78" y="54" fill="#5E5191" font-size="8">1. multiply by weights</text><text x="78" y="70" fill="#5E5191" font-size="8">2. add bias</text><text x="78" y="86" fill="#5E5191" font-size="8">3. hand back result</text><text x="78" y="103" fill="#9A5A12" font-size="8">(no numbers stored here)</text></g>
<path d="M355 108 C 300 108, 270 100, 205 100" stroke="#2D8B55" stroke-width="2" fill="none" marker-end="url(#a1)"/>
<defs><marker id="a1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="405" y="44" fill="#276b45">the fridge (params box)</text>
<g transform="translate(350,54)"><rect x="0" y="0" width="120" height="110" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="60" y="22" fill="#1a5c38" font-size="9">the NUMBERS</text><g fill="#d7ecdf" stroke="#2D8B55"><rect x="16" y="32" width="88" height="18" rx="3"/><rect x="16" y="56" width="88" height="18" rx="3"/><rect x="16" y="80" width="88" height="18" rx="3"/></g><text x="60" y="45" fill="#1a5c38" font-size="8">weights…</text><text x="60" y="69" fill="#1a5c38" font-size="8">bias…</text><text x="60" y="93" fill="#1a5c38" font-size="8">held by YOU</text></g>
<text x="260" y="182" fill="#9A5A12" font-size="9">you hand the ingredients (params) to the recipe (model) each time you cook (run it)</text>
<text x="260" y="200" fill="#9A938A" font-size="9">shape and numbers kept apart — that separation is the whole trick of the day</text>
</g></svg>
%%%

**What the recipe-and-fridge picture gets right:** the recipe never owns the flour, so you can swap in fresh ingredients and the same card still works — just as a Flax model owns no numbers, so you can pass in different params and the same model runs. **Where it breaks down:** a fridge just stores food, but the params box is a *pytree* — a tidy nested dictionary of arrays — that JAX can transform, inspect, and even hand to `grad`, which no ordinary fridge can do.

#### Why go to the trouble? Because the model stays a pure function
This might feel like extra work — why not just hide the numbers inside the model, the easy way? Because yesterday you learned that JAX's superpowers (`jit`, `grad`, `vmap`) only work on **pure functions**: functions whose output depends *only* on their inputs, with no hidden state. If the model secretly stored its own numbers and quietly changed them, it would *not* be pure — and `jit`/`grad` couldn't safely transform it. By keeping the numbers *outside* and passing them *in*, the model becomes `output = model(params, inputs)` — a clean, pure function of exactly its inputs. That is the price of admission for everything JAX does well.

#### The two ancestors this toolkit replaces
Two older ways of doing this help you feel *why* Flax and Optax exist.

- **The stateful object way (PyTorch's `nn.Module`).** In many frameworks, the model is an object that *hides* its weights inside itself, and a line like `optimizer.step()` reaches in and *changes* them in place. Convenient — but that hidden, changing state is exactly what a pure JAX function must not have. Flax pulls the numbers out so the model stays pure.
- **The hand-rolled way (raw JAX).** Earlier this week you could have built a model by hand: a plain dictionary of weight arrays, and a manual update line `w = w - lr * grad` for every weight, written out yourself. It works, but it's a lot of bookkeeping and easy to get wrong. Flax builds the parameter box for you; Optax writes the update rule for you.

!!! c-info 💡
<b>The one sentence to hold onto:</b> a Flax model is the <i>recipe</i> (the shape), the <code>params</code> box is the <i>ingredients</i> (the learned numbers) kept outside, and you hand the ingredients to the recipe each time you run it — which keeps the model a pure function <code>jit</code> and <code>grad</code> can transform.
!!!

You've got the big idea: shape here, numbers there. The natural next question is *how* you actually write that recipe card. Let's define our first tiny Flax model — that's the next concept.

@@@ concept id=c2 tag="Writing the recipe" title="Flax nn.Module: writing the recipe card" gotit="Got how to define a model"
Think of writing a network's shape like filling out a **build-a-sandwich order slip.** You don't hand the shop the actual bread and lettuce. You write a slip: "two slices of bread, then lettuce, then cheese." The slip lists the *steps and pieces* in order — the sandwich shape — while the real ingredients wait in the kitchen. Writing a Flax model is exactly that: you fill out an order slip that lists the layers in order, and the real numbers wait outside in the params box.

In Flax, that order slip is a small Python class. You describe a network's shape by writing a class that inherits from a base called [[nn.Module||Flax's base class for any piece of a neural network. You subclass it to describe the shape of your model. It stores NO numbers itself.]] (`nn` is the nickname for `flax.linen`, Flax's neural-network part). Inside the class you say what layers the network has and how the input flows through them. The class describes only the *shape* — it holds no numbers, exactly like the order slip lists pieces but holds no food.

For a plain fully-connected layer — the workhorse of neural nets, where every input touches every output — Flax hands you a ready-made piece called [[nn.Dense||A ready-made Flax layer that connects every input to every output. On its first run it creates a weight matrix (called the kernel) and a bias vector. nn.Dense(3) gives 3 outputs.]]. You just say how many outputs you want: `nn.Dense(3)` makes a layer with 3 outputs. When it first runs, it quietly creates its weight matrix (Flax calls it the [[kernel||Flax's name for a layer's weight matrix — the grid of numbers that multiplies the input. nn.Dense also makes a bias vector alongside it.]]) and a bias vector — but those numbers live in the params box, not in the layer.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A sandwich order-slip analogy for a Flax module. On the left, an order slip listing the layers in order, labelled 'the module: lists the layers in order, no numbers'. On the right, arrows show data of shape (2,) flowing in, becoming shape (4,) after the first Dense, then shape (3,) after the second. A label reads: the module describes the shape; the numbers live in the params box.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">A Flax module is an order slip: it lists the layers, holds no numbers</text>
<text x="112" y="42" fill="#5E5191">the order slip (the module)</text>
<g transform="translate(28,52)"><rect x="0" y="0" width="168" height="120" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="84" y="24" fill="#5E5191" font-size="9">class MLP(nn.Module):</text><line x1="16" y1="34" x2="152" y2="34" stroke="#8579b8" stroke-dasharray="3 3"/><text x="84" y="52" fill="#5E5191" font-size="8">1. take the input</text><text x="84" y="72" fill="#5E5191" font-size="8">2. nn.Dense(4) — a layer</text><text x="84" y="92" fill="#5E5191" font-size="8">3. bend (relu)</text><text x="84" y="108" fill="#5E5191" font-size="8">4. nn.Dense(3) — a layer</text></g>
<text x="380" y="42" fill="#276b45">the data flowing through (shapes)</text>
<g transform="translate(300,66)"><rect x="0" y="0" width="52" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="19" fill="#1a5c38" font-size="8">in (2,)</text></g>
<text x="360" y="86" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(372,66)"><rect x="0" y="0" width="52" height="30" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="26" y="19" fill="#8A6D3B" font-size="8">(4,)</text></g>
<text x="432" y="86" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(444,66)"><rect x="0" y="0" width="52" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="19" fill="#1a5c38" font-size="8">out (3,)</text></g>
<text x="380" y="130" fill="#9A938A" font-size="8">Dense(4) turns 2 numbers into 4; Dense(3) turns 4 into 3</text>
<text x="380" y="146" fill="#9A938A" font-size="8">the actual weights that do this live in the params box</text>
<text x="260" y="200" fill="#9A5A12" font-size="9">the module lists the layers; the real numbers wait outside, in params</text>
</g></svg>
%%%

**What the order-slip picture gets right:** the slip lists the steps and pieces in order without holding the food itself — just as the module lists the layers without holding the numbers. **Where it breaks down:** a sandwich slip is read top to bottom once, but a Flax module's steps run every time you call the model, on fresh data each time, with the numbers supplied from outside.

#### Two ways to write the steps: `__call__` and `setup`
Flax gives you two writing styles, and they say the same thing two ways.

- **The `__call__` style (compact).** You write one method named `__call__(self, x)` — the forward pass — and create layers *inline* as you use them. `x = nn.Dense(4)(x)` means "make a 4-output Dense layer and run `x` through it." This reads top to bottom like the order slip. It's the friendliest place to start.
- **The `setup` style (explicit).** You write a `setup(self)` method that names each layer up front (`self.layer1 = nn.Dense(4)`), then a `__call__` that uses them. Handy when layers are reused or the model is bigger.

Either way, the crucial thing stays true: `__call__` describes only *the computation*. The parameters are handed in from **outside** at call time — never stored on the object.

%%% demo id=define label="define a tiny Flax model (compact __call__ style)"
code: import jax, jax.numpy as jnp
code: import flax.linen as nn        # nn = Flax's neural-net toolkit
code: class MLP(nn.Module):          # a recipe card: subclass nn.Module
code:     @nn.compact                # lets us create layers inline in __call__
code:     def __call__(self, x):     # the forward pass — the STEPS only
code:         x = nn.Dense(4)(x)     # step 1: a Dense layer with 4 outputs
code:         x = nn.relu(x)         # step 2: a bend (activation)
code:         x = nn.Dense(3)(x)     # step 3: a Dense layer with 3 outputs
code:         return x
code: model = MLP()                  # this is just the SHAPE — still no numbers!
code: print(type(model).__name__, "created — but it holds NO parameters yet")
out: MLP created — but it holds NO parameters yet
take: <b>The model is just the recipe.</b> Creating <code>MLP()</code> wrote down the shape (two Dense layers with a bend between them) but created <i>zero</i> numbers. The weights don't exist yet — they'll be born in the next step, <code>init</code>, and they'll live in a box you hold, not inside <code>model</code>.
%%%

You've written the recipe card. But a recipe with no ingredients can't cook. Where do the actual numbers come from? That's the first half of the Flax two-step life — `init` — and it's next.

@@@ concept id=c3 tag="init: fill the box" title="init: build the parameter box" gotit="Got init"
Picture **filling a brand-new kitchen with the right-sized containers.** You don't know what jars to buy until you know what you'll store. So you bring one *sample* of each ingredient — a handful of flour, a spoon of sugar — and buy a jar to fit each. You don't keep the samples; you just used them to pick the sizes. Building a model's numbers works the same way: you show the toolkit a sample of the input, it sizes a box to fit, and you keep the box.

That box-building step is the first half of the Flax two-step life. The model is just a shape, so before it can do anything it needs its numbers created. The step that creates them is [[init||A Flax model's first step: it builds the parameter box (params). You give it a random key and a dummy input of the right shape; it returns the fresh, random starting parameters.]]. You call `model.init(key, dummy_input)` and it hands you back the `params` box — a fresh set of starting numbers, all randomly chosen, in exactly the right shapes for your layers. From then on, *you* hold that box.

`init` needs two things, and both make sense once you see why.

- **A random key.** The weights must *start* as random numbers (a network of all-zeros can't learn). Remember from Day 2: JAX makes randomness with an explicit [[PRNG key||JAX's explicit "seed of randomness." You pass it in to get random numbers, and you must SPLIT it to get fresh, independent randomness — never reuse the same key twice for different draws.]] you pass in, so `init` needs a key to draw the random starting weights.
- **A dummy input.** How does `init` know the weight matrix should be, say, 2-wide? It looks at an *example* input of the right shape — that's the "sample ingredient." You hand it a fake input — right shape, contents don't matter — and Flax sizes every layer to fit. This is why it's called *shape inference*.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="An init analogy. Two inputs feed into a box labelled model.init: a random key labelled 'random key: draw the starting weights' and a dummy input of shape (1,2) labelled 'dummy input: sets the shapes'. Out of init comes the params box, a nested tree showing Dense_0 with a kernel of shape (2,4) and bias (4,), and Dense_1 with kernel (4,3) and bias (3,). A label reads: init creates the fresh random parameter box, sized to the dummy input.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">init: random key + dummy input → the fresh params box</text>
<g transform="translate(20,44)"><rect x="0" y="0" width="132" height="30" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="66" y="19" fill="#8A6D3B" font-size="8">random key (draws weights)</text></g>
<g transform="translate(20,84)"><rect x="0" y="0" width="132" height="30" rx="5" fill="#F4F1FB" stroke="#5E5191"/><text x="66" y="19" fill="#5E5191" font-size="8">dummy input, shape (1,2)</text></g>
<path d="M156 62 C 180 62, 185 78, 205 80" stroke="#9A938A" stroke-width="1.5" fill="none"/>
<path d="M156 98 C 180 98, 185 84, 205 82" stroke="#9A938A" stroke-width="1.5" fill="none"/>
<g transform="translate(206,62)"><rect x="0" y="0" width="70" height="40" rx="6" fill="#EDE9E0" stroke="#3A342E" stroke-width="1.5"/><text x="35" y="18" fill="#3A342E" font-size="8">model</text><text x="35" y="32" fill="#3A342E" font-size="8">.init(...)</text></g>
<path d="M280 82 h24" stroke="#2D8B55" stroke-width="2" marker-end="url(#a3)"/>
<defs><marker id="a3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(310,40)"><rect x="0" y="0" width="196" height="128" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="98" y="20" fill="#1a5c38" font-size="9">params (the box, a pytree)</text><g fill="#1a5c38" font-size="8" text-anchor="start"><text x="14" y="44">Dense_0:</text><text x="30" y="60">kernel  shape (2, 4)</text><text x="30" y="76">bias    shape (4,)</text><text x="14" y="98">Dense_1:</text><text x="30" y="114">kernel  shape (4, 3)</text></g></g>
<text x="260" y="200" fill="#9A938A" font-size="9">random starting numbers, sized to fit the dummy input — you hold this box from now on</text>
</g></svg>
%%%

**What the kitchen-containers picture gets right:** you use a sample to *pick sizes*, then keep the empty jars, not the sample — just as `init` uses the dummy input to size the box, then keeps the box, not the dummy. **Where it breaks down:** buying jars is a one-off, but you can re-run `init` any time with a different dummy shape to make a differently-sized model — and the numbers inside always start random, never empty.

#### Peek inside the box: it's just a pytree
The `params` box is not a mysterious model object — it's a plain nested dictionary of arrays, a **pytree**, which you can print, index, and inspect like any Python data. Predict what shapes you'll see: our model was `Dense(4)` then `Dense(3)`, fed a 2-number input — so the first kernel should be `(2, 4)` and the second `(4, 3)`.

%%% demo id=init label="run init and look inside the params box"
code: from jax import random
code: key = random.PRNGKey(0)             # a starting random seed
code: dummy = jnp.ones((1, 2))            # ONE example, shape (1, 2) — sets the sizes
code: params = model.init(key, dummy)     # build the box; contents are random
code: # jax.tree_util lets us see every leaf array's shape at a glance
code: shapes = jax.tree_util.tree_map(lambda a: a.shape, params)
code: print(shapes)
out: {'params': {'Dense_0': {'bias': (4,), 'kernel': (2, 4)}, 'Dense_1': {'bias': (3,), 'kernel': (4, 3)}}}
take: <b>The box is a plain nested dict of arrays.</b> Flax sized each layer from the dummy: <code>Dense_0</code> maps 2→4 (kernel <code>(2,4)</code>), <code>Dense_1</code> maps 4→3 (kernel <code>(4,3)</code>), each with a bias. Because it's an ordinary pytree, you can inspect it, save it, or hand it to <code>jax.grad</code>. This box is now <i>yours</i> — you carry it into every run.
%%%

#### Trap 1 — reusing the same key
Remember the golden rule from Day 2: never reuse a PRNG key. If you call `init` twice with the *same* key to make two models, they get the *identical* random weights — not two independent networks. The fix is `jax.random.split`, which turns one key into several fresh, independent keys.

!!! c-warn ⚠️
<b>Failure mode:</b> initializing two models (or two layers) with the <b>same</b> PRNG key, so their random weights come out identical instead of independent — training becomes non-reproducible or the two never diverge. <b>Cause:</b> a JAX key is deterministic; the same key always draws the same numbers. <b>Remedy:</b> <code>k1, k2 = jax.random.split(key)</code> to get fresh independent keys, one per model, before calling <code>init</code>.
!!!

#### Trap 2 — the wrong dummy shape
Because `init` *sizes* the network from the dummy input, giving it the wrong shape bakes the wrong sizes into your box — and the mismatch usually explodes later, when real data of a different shape flows in. The remedy is simple: make the dummy the exact shape of one real input, and *check the returned shapes* (like we just did) before you train.

!!! c-info 🔬
<b>Optional (skippable) — a shape sanity habit:</b> if building the real box is expensive, <code>jax.eval_shape(model.init, key, dummy)</code> reports the parameter shapes <i>without</i> actually allocating the numbers — a cheap way to confirm the sizes are what you expect before committing.
!!!

The box now exists and holds random numbers. But `init` doesn't *run* the model — it only builds the box. To actually push data through the network, you use the second step of the two-step life: `apply`. That's next.

@@@ concept id=c4 tag="apply: run the model" title="apply: run the model as a pure function" gotit="Got apply"
Picture a **vending machine.** You put in money *and* press a button (your two inputs), and out comes a snack. The machine doesn't secretly remember your last visit or change what button 4 means — same money, same button, same snack, always. Running a Flax model is just like that: you give it two things, it gives you one thing back, and it has no memory of before.

That's the second step of the Flax two-step life. To actually run your network — to push an input through and get a prediction — you call [[apply||A Flax model's second step: it RUNS the forward pass. You write model.apply(params, inputs) — passing the params box in from outside — and get the output.]]. Notice the shape of that call: `model.apply(params, inputs)`. You hand the model *both* the ingredients (params, the "money") *and* the input (the "button you press"), and it hands back the answer. The model never reaches into a hidden store — you supply the numbers every single time, exactly like feeding the machine fresh coins.

This is the payoff of keeping numbers outside. Written this way, running the model is exactly `output = model.apply(params, inputs)` — a **pure function** of `(params, inputs)`. Same params, same input → same output, every time, with nothing hidden. And a pure function is precisely what `jit`, `grad`, and `vmap` can transform. This one design choice is why Flax models slot straight into everything you learned this week.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A vending-machine analogy for apply. Two inputs go into a machine labelled model.apply: the params box labelled 'params (the numbers)' and an input labelled 'inputs (the data)'. Out the bottom comes the output labelled 'prediction'. A caption reads: output = model.apply(params, inputs) — a pure function of exactly its two inputs.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">apply: give it params + input, get the output — a pure function</text>
<g transform="translate(50,48)"><rect x="0" y="0" width="110" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="55" y="19" fill="#1a5c38" font-size="8">params (numbers)</text></g>
<g transform="translate(50,92)"><rect x="0" y="0" width="110" height="30" rx="5" fill="#F4F1FB" stroke="#5E5191"/><text x="55" y="19" fill="#5E5191" font-size="8">inputs (data)</text></g>
<path d="M164 63 C 190 63, 195 82, 214 85" stroke="#9A938A" stroke-width="1.5" fill="none"/>
<path d="M164 107 C 190 107, 195 90, 214 87" stroke="#9A938A" stroke-width="1.5" fill="none"/>
<g transform="translate(216,50)"><rect x="0" y="0" width="120" height="90" rx="8" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/><text x="60" y="22" fill="#3A342E" font-size="9">model.apply</text><line x1="16" y1="30" x2="104" y2="30" stroke="#B8AEA2"/><g fill="#c9c1b4" stroke="#8a8276"><rect x="20" y="40" width="34" height="20" rx="2"/><rect x="66" y="40" width="34" height="20" rx="2"/><rect x="20" y="66" width="34" height="20" rx="2"/><rect x="66" y="66" width="34" height="20" rx="2"/></g><text x="60" y="80" fill="#6B645E" font-size="6">the vending machine</text></g>
<path d="M276 142 v18" stroke="#C99A12" stroke-width="2" marker-end="url(#a4)"/>
<defs><marker id="a4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C99A12"/></marker></defs>
<g transform="translate(214,164)"><rect x="0" y="0" width="124" height="24" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="62" y="16" fill="#8A6D3B" font-size="8">output (prediction)</text></g>
<text x="415" y="70" fill="#9A938A" font-size="8">no hidden memory:</text>
<text x="415" y="84" fill="#9A938A" font-size="8">same in → same out</text>
<text x="415" y="98" fill="#9A938A" font-size="8">so jit / grad / vmap</text>
<text x="415" y="112" fill="#9A938A" font-size="8">all work on it</text>
</g></svg>
%%%

**What the vending-machine picture gets right:** it takes two things in and gives one thing back, with no memory of past visits — just like `apply` takes params and input and returns output, purely. **Where it breaks down:** a vending machine's snacks are fixed forever, but `apply`'s output changes as you *learn* better params — the machine's "menu" (the params) is exactly what training improves.

#### Run the model, twice, on the same params
Let's push a real input through. Predict: since `apply` is pure, running it twice with the same params and input gives the identical answer both times.

%%% demo id=apply label="run the forward pass with model.apply"
code: x = jnp.ones((1, 2))                      # one real input, shape (1, 2)
code: y = model.apply(params, x)                # RUN the model: params in, output out
code: y2 = model.apply(params, x)               # run it again, same params + input
code: print("output shape:", y.shape, "| same answer twice?", bool(jnp.allclose(y, y2)))
out: output shape: (1, 3) | same answer twice? True
take: <b>apply ran the network — purely.</b> We handed in the params box and the input; out came a shape-<code>(1,3)</code> prediction. Running it again with the same inputs gave the exact same answer, because <code>apply</code> is a pure function of <code>(params, inputs)</code> — nothing hidden, nothing remembered.
%%%

#### Because it's pure, jit and grad just work
Here's the whole week paying off at once. Because `model.apply(params, inputs)` is pure, you can wrap it in `jax.jit` to make it fast, and — crucially for training — you can ask `jax.grad` for the gradient *with respect to the params*. That gradient is the compass that tells you which way to nudge each number. Watch a tiny loss-and-gradient step: we measure how far the output is from a target, then get the gradient of that loss with respect to `params`.

%%% demo id=grad label="grad works because apply is pure"
code: def loss_fn(params, x, target):
code:     pred = model.apply(params, x)         # pure forward pass
code:     return jnp.mean((pred - target) ** 2) # a simple "how wrong?" number
code: target = jnp.zeros((1, 3))
code: loss = loss_fn(params, x, target)
code: grads = jax.grad(loss_fn)(params, x, target)  # gradient w.r.t. params
code: same_shape = jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)
code: print("loss:", round(float(loss), 4), "| grads is a pytree shaped like params:", same_shape)
out: loss: 0.9032 | grads is a pytree shaped like params: True
take: <b>grad handed us a compass shaped exactly like the params box.</b> Because <code>apply</code> is pure, <code>jax.grad</code> could differentiate the loss with respect to every weight — and it returns the gradients in the <i>same</i> pytree shape as <code>params</code> (one gradient per number). Next we'll use that compass to actually improve the numbers. Note: <code>grad</code> <b>computes</b> the gradient; it does not <i>apply</i> it — that's Optax's job.
%%%

You can now build a model, fill its box, run it, and get gradients. You have the compass. But a compass only tells you the *direction* — you still need something to take the step. That something is **Optax**, and it's next.

@@@ concept id=c5 tag="Optax: the nudger" title="Optax: turn a gradient into an update" gotit="Got Optax"
Picture the **thermostat** on a wall. You don't yank the temperature to where you want in one violent jump. The thermostat reads how far off the room is from your target, and nudges the heating a *sensible amount* toward it. A fancy thermostat also remembers whether the room has been trending cold and leans in a little harder. That "read how far off, take a sensible nudge" is exactly the job of the second friend in our toolkit.

A gradient tells you which way each weight should move to lower the loss — but it doesn't take the step for you. [[Optax||JAX's optimizer toolkit. You pick an optimizer (like SGD or Adam), it turns your gradients into concrete parameter updates, and you add those updates to your params.]] is the thermostat: it reads that raw gradient and turns it into a concrete **update** — the actual amount to change each number — and it keeps its own memory in a box *outside*, just like params.

You pick an optimizer by name. Two you'll meet everywhere:

- [[optax.sgd||"Stochastic gradient descent" — the simplest optimizer. Its update is just: step a little in the downhill direction. optax.sgd(0.1) uses a learning rate of 0.1.]] — the plainest one: move a little bit downhill each step, by a size you choose called the *learning rate*. This is the basic thermostat.
- [[optax.adam||A popular optimizer that keeps a running memory of past gradients to adapt each weight's step size — usually learns faster than plain SGD. It has extra state to carry.]] — a cleverer one that *remembers* recent gradients to adjust each weight's step, and usually learns faster. This is the thermostat that notices the room trending cold.

Whichever you pick, Optax hands you a [[GradientTransformation||What every Optax optimizer really is: a pair of functions — one to set up its memory (init), one to turn gradients into updates (update). Composable — you can chain several together.]] — a tidy pair of functions: one to *set up* its memory, one to *turn gradients into updates*. And here's the elegant part: an optimizer is really just a **chain of small gradient transformations** glued together. Plain SGD is one link; Adam is a few links stacked. You *compose* an optimizer the same way you'd stack Lego bricks.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A thermostat analogy for Optax. A dial shows current temperature far from a target. An arrow labelled 'gradient: how far off, which way' feeds a thermostat box labelled 'Optax optimizer'. The thermostat outputs a small nudge labelled 'update: a sensible step, not the whole jump'. A side note shows SGD as one link and Adam as several links chained, labelled 'an optimizer is a chain of transforms'.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Optax reads the gradient and decides a sensible nudge (not the whole jump)</text>
<g transform="translate(30,48)"><circle cx="46" cy="46" r="42" fill="#FBF8F2" stroke="#C99A12" stroke-width="2"/><path d="M46 46 L70 24" stroke="#C93B3B" stroke-width="3"/><circle cx="46" cy="46" r="4" fill="#3A342E"/><text x="46" y="104" fill="#8A6D3B" font-size="8">how far off?</text></g>
<path d="M126 92 h30" stroke="#5E5191" stroke-width="2" marker-end="url(#a5)"/>
<defs><marker id="a5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<text x="141" y="84" fill="#5E5191" font-size="7">gradient</text>
<g transform="translate(158,64)"><rect x="0" y="0" width="120" height="56" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="60" y="24" fill="#5E5191" font-size="9">Optax optimizer</text><text x="60" y="42" fill="#5E5191" font-size="8">(the thermostat)</text></g>
<path d="M280 92 h30" stroke="#2D8B55" stroke-width="2" marker-end="url(#a5b)"/>
<defs><marker id="a5b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(312,64)"><rect x="0" y="0" width="118" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="59" y="24" fill="#1a5c38" font-size="9">update</text><text x="59" y="42" fill="#276b45" font-size="8">a small, sensible step</text></g>
<text x="260" y="152" fill="#9A5A12" font-size="9">an optimizer is a CHAIN of transforms:</text>
<g transform="translate(150,160)" font-size="8"><rect x="0" y="0" width="58" height="20" rx="3" fill="#EDE9E0" stroke="#8a8276"/><text x="29" y="14" fill="#3A342E">sgd = 1 link</text><rect x="72" y="0" width="130" height="20" rx="3" fill="#EDE9E0" stroke="#8a8276"/><text x="137" y="14" fill="#3A342E">adam = a few links chained</text></g>
</g></svg>
%%%

**What the thermostat picture gets right:** it reads how far off you are and makes a measured nudge toward the goal, rather than slamming to the target — just as an optimizer scales the gradient into a careful step. **Where it breaks down:** a thermostat controls one number (temperature), but an optimizer nudges *all* your parameters at once, and a smart one like Adam keeps a separate running memory for each — far more moving parts than a home dial.

#### Optax carries its own box: opt_state
Just like params, the optimizer's memory lives *outside*, in a box you hold. You create it once with `optimizer.init(params)`, and it's called the [[opt_state||The optimizer's memory box, created by optimizer.init(params). You hold it outside and thread it through every step, exactly like params. For SGD it's tiny; for Adam it stores running gradient stats.]]. For plain SGD this box is nearly empty. For Adam it holds the running gradient statistics Adam uses to adapt. Either way, *you* carry it and pass it in each step — the same functional pattern as params.

#### The two-move update: get updates, then add them
Optax splits the update into two clear moves:

1. `updates, opt_state = optimizer.update(grads, opt_state)` — turn the raw gradient into the actual **updates** (and get back the optimizer's *new* memory).
2. `params = optax.apply_updates(params, updates)` — **add** those updates to your params to get the new params.

Notice something vital: **both moves are pure and RETURN new boxes** — they never secretly change the old ones. This is the single most important habit of the whole day, and it's where beginners trip. Let's watch one full step.

!!! c-info 🔬
<b>Optional (skippable) — the third argument:</b> some optimizers accept a third argument, <code>optimizer.update(grads, opt_state, params)</code>, which they use for tricks like weight decay. Plain <code>optax.sgd</code> ignores it, so throughout this lesson we use the two-argument form <code>optimizer.update(grads, opt_state)</code>. Reach for the third argument only when a specific optimizer's docs ask for it.
!!!

%%% demo id=optax label="one full Optax step: update, then apply_updates"
code: import optax
code: optimizer = optax.sgd(learning_rate=0.1)   # pick the simplest optimizer
code: opt_state = optimizer.init(params)          # its memory box (held OUTSIDE, like params)
code: grads = jax.grad(loss_fn)(params, x, target)             # the compass (from before)
code: updates, opt_state = optimizer.update(grads, opt_state)  # gradient → actual step
code: new_params = optax.apply_updates(params, updates)        # add the step to params
code: old_k = params['params']['Dense_0']['kernel']
code: new_k = new_params['params']['Dense_0']['kernel']
code: print("params changed?", bool(not jnp.allclose(old_k, new_k)))
out: params changed? True
take: <b>One learning step, done.</b> <code>update</code> turned the gradient into a concrete step (and returned the optimizer's new memory); <code>apply_updates</code> added that step to the params to make <code>new_params</code>. Both returned <i>new</i> boxes — we caught them in variables. That catch is the whole trick, and the next trap shows what happens when you forget it.
%%%

#### The #1 trap — forgetting to catch the returned boxes
Because `update` and `apply_updates` are **pure**, they return the new state instead of changing the old in place. If you call them but *don't reassign* the result — if you write `optax.apply_updates(params, updates)` on its own line and keep using the old `params` — your weights never change and training silently stalls. The fix is the reassignment pattern: always catch what they hand back.

!!! c-warn ⚠️
<b>Failure mode (silent):</b> calling <code>optimizer.update(...)</code> or <code>optax.apply_updates(...)</code> but not catching their return values — so <code>params</code> and <code>opt_state</code> keep their old values and the model never learns (loss stays flat). <b>Cause:</b> both functions are <b>pure</b> — they RETURN new state and never mutate the boxes you passed in (unlike PyTorch's in-place <code>optimizer.step()</code>). <b>Remedy:</b> always reassign — <code>updates, opt_state = optimizer.update(grads, opt_state)</code> and <code>params = optax.apply_updates(params, updates)</code> — thread the new boxes into the next step.
!!!

Now the two friends meet: gradients from `apply`, steps from Optax. Putting them in a loop is the training step — and one last speed trap — which is our next concept.

@@@ concept id=c6 tag="The training step" title="Put it together: the functional train step" gotit="Got the loop"
Picture a **relay race passing the baton.** Each runner (one step) takes the baton (the current params + opt_state), runs their leg (computes an update), and passes the *same* baton — now a little further along — to the next runner. If any runner drops the baton instead of passing it, the race stops. The training loop is that relay: each step hands the improved boxes forward to the next. That "hand it forward" idea is the last piece we need.

This is where the whole toolkit clicks into one clean move. A single **training step** does the same four things every time: run the model, measure how wrong it is, ask for the gradient, take an Optax step. Because everything is pure and lives in boxes *you* hold, the step is just a function that takes the current boxes in and hands the *new* boxes back:

`params, opt_state, loss = train_step(params, opt_state, x, target)`

Then you loop: feed the new boxes into the next call, over and over, and the loss slides downhill. The word for one such pass is a step; passing the boxes from one step into the next — the baton handoff — is called **threading** the state.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A relay-race analogy for threading state in a training loop. Three runners in a row, each labelled 'step'. A baton passes from one to the next, labelled params + opt_state. Each runner takes the baton in and hands a slightly-improved baton forward. A note reads: each step returns the NEW boxes; the next step uses them — drop the baton (forget to return) and the race stops.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The training loop threads the boxes forward, step to step</text>
<g transform="translate(30,60)"><circle cx="30" cy="30" r="24" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="30" y="34" fill="#5E5191" font-size="9">step 1</text></g>
<path d="M92 90 h44" stroke="#2D8B55" stroke-width="2" marker-end="url(#a6b)"/>
<text x="114" y="82" fill="#276b45" font-size="7">baton</text>
<defs><marker id="a6b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(160,60)"><circle cx="30" cy="30" r="24" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="30" y="34" fill="#5E5191" font-size="9">step 2</text></g>
<path d="M222 90 h44" stroke="#2D8B55" stroke-width="2" marker-end="url(#a6b)"/>
<text x="244" y="82" fill="#276b45" font-size="7">baton</text>
<g transform="translate(290,60)"><circle cx="30" cy="30" r="24" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="30" y="34" fill="#5E5191" font-size="9">step 3</text></g>
<path d="M352 90 h44" stroke="#2D8B55" stroke-width="2" marker-end="url(#a6b)"/>
<g transform="translate(400,74)"><rect x="0" y="0" width="98" height="32" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="49" y="14" fill="#1a5c38" font-size="8">lower loss</text><text x="49" y="27" fill="#276b45" font-size="7">each pass</text></g>
<text x="150" y="140" fill="#9A5A12" font-size="8">the baton = params + opt_state; each step hands the IMPROVED boxes forward</text>
<text x="150" y="158" fill="#C93B3B" font-size="8">drop the baton (forget to return the new boxes) → the race (training) stops</text>
</g></svg>
%%%

**What the relay picture gets right:** the baton must be *handed forward* for the race to continue — just as each step must return the new params + opt_state for the next step to build on. **Where it breaks down:** in a relay the runners are different people, but here it's the same little function called again and again, each time with the boxes the previous call produced.

#### Watch the loss go down
Let's run a handful of steps and watch the loss fall. Predict: with each step nudging the params downhill, the loss should shrink toward zero. We'll thread `params` and `opt_state` through the loop and print the loss each time.

%%% demo id=loop label="run a few training steps and watch the loss fall"
code: def train_step(params, opt_state, x, target):
code:     loss, grads = jax.value_and_grad(loss_fn)(params, x, target)  # loss + compass
code:     updates, opt_state = optimizer.update(grads, opt_state)       # gradient → step
code:     params = optax.apply_updates(params, updates)                 # add the step
code:     return params, opt_state, loss                                # hand boxes forward
code: params = model.init(random.PRNGKey(0), jnp.ones((1, 2)))          # fresh box
code: opt_state = optimizer.init(params)                                # fresh memory
code: for step in range(5):
code:     params, opt_state, loss = train_step(params, opt_state, x, target)  # THREAD them
code:     print(f"step {step}: loss = {float(loss):.4f}")
out: step 0: loss = 0.9032
     step 1: loss = 0.5781
     step 2: loss = 0.3700
     step 3: loss = 0.2368
     step 4: loss = 0.1516
take: <b>The loss slides downhill — the model is learning.</b> Each step returned the improved <code>params</code> and <code>opt_state</code>, and we threaded them into the next call. That is a complete functional training loop: run, measure, gradient, step, repeat. Nothing hidden, every box carried by hand.
%%%

Here is that falling loss as a picture — the same numbers you just printed, drawn as bars so you can *see* the model getting less wrong each step:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A bar chart of the loss falling over five training steps. Five bars from left to right shrink in height: step 0 loss 0.90, step 1 loss 0.58, step 2 loss 0.37, step 3 loss 0.24, step 4 loss 0.15. A downward arrow labelled 'loss going down = learning' runs across the tops.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The loss falls each step — the model is learning</text>
<line x1="60" y1="150" x2="470" y2="150" stroke="#B8AEA2" stroke-width="1.5"/>
<line x1="60" y1="40" x2="60" y2="150" stroke="#B8AEA2" stroke-width="1.5"/>
<text x="30" y="46" fill="#9A938A" font-size="8">loss</text>
<g fill="#8579b8" stroke="#5E5191">
<rect x="82" y="42" width="52" height="108" rx="2"/>
<rect x="162" y="80" width="52" height="70" rx="2"/>
<rect x="242" y="106" width="52" height="44" rx="2"/>
<rect x="322" y="122" width="52" height="28" rx="2"/>
<rect x="402" y="132" width="52" height="18" rx="2"/>
</g>
<g fill="#5E5191" font-size="8"><text x="108" y="38">0.90</text><text x="188" y="76">0.58</text><text x="268" y="102">0.37</text><text x="348" y="118">0.24</text><text x="428" y="128">0.15</text></g>
<g fill="#6B645E" font-size="8"><text x="108" y="164">step 0</text><text x="188" y="164">step 1</text><text x="268" y="164">step 2</text><text x="348" y="164">step 3</text><text x="428" y="164">step 4</text></g>
<path d="M108 40 C 200 60, 320 118, 428 130" stroke="#2D8B55" stroke-width="2" fill="none" stroke-dasharray="5 4" marker-end="url(#a6c)"/>
<defs><marker id="a6c" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="300" y="182" fill="#276b45" font-size="9">↓ loss going down = the model getting less wrong</text>
</g></svg>
%%%

#### Trap — forgetting to jit the step
Your `train_step` is a pure function — so it *begs* to be wrapped in `jax.jit` from yesterday. If you leave it un-jitted, JAX runs it eagerly, one operation at a time, in slow Python — fine for 5 steps, painful for the millions in real training. The fix is one decorator.

!!! c-warn ⚠️
<b>Failure mode (slow):</b> running the training loop with a plain, un-<code>jit</code>ted <code>train_step</code> — each step crawls through eager Python op-by-op, so a real training run is far slower than it needs to be. <b>Cause:</b> without <code>jit</code>, none of yesterday's tracing/fusion speed-up applies. <b>Remedy:</b> wrap it — <code>train_step = jax.jit(train_step)</code> (or <code>@jax.jit</code> above it). It's pure, so this is safe; keep shapes stable so it compiles once.
!!!

#### The honest limits of the toolkit
Before the recap, be clear about what this toolkit does *not* do — it's part of understanding it.

- **It won't train the model *for* you.** Flax and Optax give you the pieces (a model, an optimizer); *you* still write the loop, thread the boxes, and pick the settings (learning rate, how many layers). There's no hidden auto-pilot.
- **Optax computes *no* gradients.** Optax turns gradients into updates — but the gradient itself comes from `jax.grad`, and the loss (the "how wrong?" number) is one *you* define. Optax never sees your data or your loss.
- **A stack of `nn.Dense` is still just an MLP.** Building layers with Flax doesn't grant new power on its own — a plain Dense stack has the same ceiling you met earlier (it can't magically escape the limits of a simple network). Flax is the *plumbing*, not a smarter model.

You've now built, run, and trained a tiny network the JAX way. Time to gather the whole toolkit onto one page.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you now think the JAX way about building and training a network: **the shape lives in the model, the numbers live in a box you hold, and you thread that box through every step.** Here's one last everyday picture for the whole day: a **coffee machine and a jar of beans.** The machine (**Flax**) knows the *steps* — grind, brew, pour — but stores no beans inside it. The beans (**params**) sit in a jar you keep on the counter, outside the machine, and you pour them in each time you brew (`apply`). After each cup you taste it and top up the jar a little to make the next cup better — that topping-up is **Optax**, reading how far off the taste is (the gradient) and adding a sensible scoop (the update). **Where the coffee picture breaks down:** a real machine has one fixed recipe, but your model *learns* — the beans (params) get better every step, which is the whole point of training. Below is the day step by step, plus a cheat-sheet to keep on your desk.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A five step flow for the day. Step one: define a Flax model (the shape, no numbers). Step two: init builds the params box from a key plus a dummy input. Step three: apply runs the model as a pure function of params and inputs. Step four: jax.grad gives the gradient; Optax turns it into an update you add to params. Step five: loop the train step, threading params and opt_state forward. Below, a note reads: parameters and optimizer state both live OUTSIDE, carried by hand.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The whole day: define → init → apply → grad+Optax → loop (all state OUTSIDE)</text>
<g transform="translate(6,42)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="48" y="22" fill="#5E5191">1 define</text><text x="48" y="38" fill="#5E5191" font-size="7">nn.Module (shape)</text></g>
<text x="106" y="72" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(120,42)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/><text x="48" y="22" fill="#8A6D3B">2 init</text><text x="48" y="38" fill="#9A5A12" font-size="7">key+dummy → params</text></g>
<text x="220" y="72" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(234,42)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="48" y="22" fill="#1a5c38">3 apply</text><text x="48" y="38" fill="#276b45" font-size="7">(params, x) → out</text></g>
<text x="334" y="72" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(348,42)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/><text x="48" y="22" fill="#8A6D3B">4 grad+Optax</text><text x="48" y="38" fill="#9A5A12" font-size="7">gradient → update</text></g>
<g transform="translate(200,110)"><rect x="0" y="0" width="120" height="34" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="60" y="15" fill="#5E5191">5 loop the step</text><text x="60" y="29" fill="#5E5191" font-size="7">thread params + opt_state</text></g>
<path d="M300 96 C 340 108, 330 112, 322 118" stroke="#5E5191" stroke-width="1.2" fill="none"/>
<text x="260" y="172" fill="#9A938A" font-size="9">params AND opt_state both live outside the model — you carry them by hand, every step</text>
</g></svg>
%%%

The day in a few beats:
- **Recipe vs. ingredients.** A [[Flax||JAX's neural-net library. You define a model as an nn.Module whose parameters live OUTSIDE it, in a params box.]] model is the *recipe* (the shape); the [[params||the box (a pytree) of learned numbers, held OUTSIDE the model, passed in every run]] box is the *ingredients* (the numbers), held outside. Keeping numbers outside makes the model a **pure function** `jit` and `grad` love.
- **Write the recipe.** Subclass [[nn.Module||Flax's base class; you subclass it to describe a model's shape]] and list layers like [[nn.Dense||a fully-connected Flax layer; nn.Dense(3) gives 3 outputs]] in `__call__` (compact) or `setup` (explicit). The class holds no numbers.
- **init → apply, the two-step life.** [[init||builds the params box from a random key + a dummy input that sets the shapes]] creates the box (needs a random key and a dummy input to size it); [[apply||runs the forward pass as model.apply(params, inputs) — a pure function]] runs the model as `model.apply(params, inputs)`.
- **Optax nudges the numbers.** [[Optax||JAX's optimizer library: turns gradients into updates you add to params]] turns a gradient into an *update*. Pick `optax.sgd` or `optax.adam`; create its memory box [[opt_state||the optimizer's memory box, created by optimizer.init(params) and threaded like params]] with `optimizer.init(params)`.
- **The two-move update.** `updates, opt_state = optimizer.update(grads, opt_state)` then `params = optax.apply_updates(params, updates)`. Both are **pure and RETURN new boxes** — catch them or nothing learns.
- **The train step, looped.** One function in, new boxes out: run → measure → `grad` → Optax step → **thread** the boxes forward. Wrap it in `jax.jit` for speed.

The one rule to memorize: **params and opt_state live outside the model — `update` and `apply_updates` hand you NEW boxes, so always catch them and thread them into the next step.**

#### Cheat-sheet · the moves at a glance
%%% table
:: You want to… :: Do this :: Note
Define a model :: `class MLP(nn.Module): ...` :: the shape only — holds no numbers
Build the params box :: `params = model.init(key, dummy)` :: needs a random key + a dummy input
Run the model :: `y = model.apply(params, x)` :: pure fn of (params, inputs)
Get the gradient :: `grads = jax.grad(loss_fn)(params, …)` :: `grad` computes; Optax applies
Pick an optimizer :: `opt = optax.sgd(0.1)` / `optax.adam(1e-3)` :: adam keeps running gradient stats
Make its memory box :: `opt_state = opt.init(params)` :: held outside, like params
Take one step :: `upd, opt_state = opt.update(grads, opt_state)` :: then `params = optax.apply_updates(params, upd)`
Go fast :: `train_step = jax.jit(train_step)` :: it's pure, so this is safe
%%%

#### Cheat-sheet · the words you met today
%%% jargon
Flax | JAX's neural-net library; you define a model whose parameters live OUTSIDE it
nn.Module | Flax's base class you subclass to describe a model's shape (no numbers inside)
nn.Dense | a ready-made fully-connected layer; nn.Dense(3) gives 3 outputs
kernel | Flax's name for a layer's weight matrix; a bias vector sits alongside it
params | the box (a pytree — a nested dict of arrays) of learned numbers, held outside the model
init | builds the params box from a random PRNG key + a dummy input that sets the shapes
apply | runs the forward pass: model.apply(params, inputs), a pure function
Optax | JAX's optimizer library; turns gradients into updates you add to params
optax.sgd / optax.adam | ready-made optimizers; adam remembers past gradients, sgd does not
opt_state | the optimizer's memory box, created by optimizer.init(params), threaded like params
update / apply_updates | update: gradient → step (returns new opt_state); apply_updates: add step to params
threading | passing the new params + opt_state from one training step into the next
%%%

That's the whole day. You now write a Flax model as a recipe (shape only), call `init` to build the params box and `apply` to run it purely, then use `jax.grad` for the gradient and Optax to turn it into an update — catching and threading the new boxes every step. Next you'll bring all of Week 1 together in the **ViT Capstone**, where you build a real vision transformer with exactly this toolkit.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: In Flax, where do a model's learnable numbers (parameters) live? | a:2 | Hidden inside the model object | In a secret global variable | OUTSIDE the model, in a params box you hold and pass in | They don't exist until training ends | fb: Flax keeps the SHAPE in the model and the NUMBERS in a params box outside it. Passing params in each call is what keeps the model a pure function that jit and grad can transform.
q: What are the two steps of a Flax model's life, in order? | a:1 | apply then init | init (build the params box) then apply (run the model) | grad then update | setup then jit | fb: init builds the params box from a random key + a dummy input; apply runs the forward pass as model.apply(params, inputs). Init first, then apply.
q: You call optimizer.update(...) and optax.apply_updates(...) but your loss stays flat and nothing learns. Most likely cause? | a:0 | You didn't catch (reassign) the NEW params/opt_state they return | Optax computed the gradient wrong | You used adam instead of sgd | The dummy input was too big | fb: update and apply_updates are PURE — they RETURN new boxes and never change the old ones in place. You must reassign: updates, opt_state = update(...) and params = apply_updates(params, updates), then thread them forward.
q: Which statement about Optax is true? | a:3 | Optax computes the gradients for you | Optax defines your loss function | Optax trains the model automatically with no loop | Optax turns gradients into updates, but jax.grad computes the gradient and you define the loss | fb: Optax only transforms and applies gradients. The gradient comes from jax.grad; the loss (the "how wrong?" number) is one you write. You still write the training loop yourself.
%%%

@@@ produce id=produce tag="Produce" title="Build and train a tiny net with your own hands" gotit="Done"
Time to feel today's rule for yourself. You'll define a Flax model, `init` its params box, run `apply`, and train it with Optax for a few steps — catching and threading the boxes each step. **Predict first:** when you run the training loop, what should happen to the loss printed at each step — go up, stay flat, or slide down? Then run it and **watch** the numbers. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-05-flax-optax/experiment.py`. Start with `import jax`, `import jax.numpy as jnp`, `import flax.linen as nn`, `import optax`, and `from jax import random`. **Print what you observe at each step.** (1) Define a tiny `MLP(nn.Module)` with `@nn.compact` `__call__`: `nn.Dense(4)` → `nn.relu` → `nn.Dense(3)`. Make `model = MLP()`. (2) Build the box: `params = model.init(random.PRNGKey(0), jnp.ones((1, 2)))`, then **observe** the leaf shapes with `jax.tree_util.tree_map(lambda a: a.shape, params)` — check the kernels are `(2,4)` and `(4,3)`. (3) Run `y = model.apply(params, jnp.ones((1, 2)))` twice and **notice** the same input gives the same output (it's pure). (4) Write `loss_fn(params, x, target) = jnp.mean((model.apply(params, x) - target)**2)`. Make `optimizer = optax.sgd(0.1)` and `opt_state = optimizer.init(params)`. (5) Write a `train_step` that does `loss, grads = jax.value_and_grad(loss_fn)(...)`, then `updates, opt_state = optimizer.update(grads, opt_state)`, then `params = optax.apply_updates(params, updates)`, and **returns** the new `params, opt_state, loss`. Loop it ~10 times, threading the boxes, and **watch** the loss fall. (6) On purpose, break it: call `optax.apply_updates(params, updates)` WITHOUT catching the result, keep using the old `params`, and **notice** the loss stops moving — then fix it by reassigning. Run with `python3 sessions/m07-thinking-in-jax/day-05-flax-optax/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Week 1 Day 5 (Flax & Optax) artifact.

Create sessions/m07-thinking-in-jax/day-05-flax-optax/experiment.py that, with a comment on each step and printing what it observes:
1. import jax, jax.numpy as jnp, flax.linen as nn, optax, and from jax import random. Define class MLP(nn.Module) with @nn.compact __call__: x = nn.Dense(4)(x); x = nn.relu(x); x = nn.Dense(3)(x); return x. model = MLP().
2. params = model.init(random.PRNGKey(0), jnp.ones((1, 2))); print jax.tree_util.tree_map(lambda a: a.shape, params) to show the params box is a pytree with kernels (2,4) and (4,3).
3. Run y = model.apply(params, jnp.ones((1,2))) twice; print that the two outputs are identical (apply is a pure function of (params, inputs)).
4. Define loss_fn(params, x, target) = jnp.mean((model.apply(params, x) - target)**2) with target = jnp.zeros((1,3)). optimizer = optax.sgd(0.1); opt_state = optimizer.init(params).
5. Define train_step(params, opt_state, x, target): loss, grads = jax.value_and_grad(loss_fn)(params, x, target); updates, opt_state = optimizer.update(grads, opt_state); params = optax.apply_updates(params, updates); return params, opt_state, loss. Loop 10 steps threading params + opt_state, printing the loss each step — it should slide down.
6. Demonstrate the #1 trap: call optax.apply_updates(params, updates) WITHOUT reassigning, keep using old params for a couple of steps, print that the loss stays flat; then show the fix (reassign) makes it fall again.
Then run it and paste the output at the bottom as a comment. Explain in one line why params and opt_state must be threaded (returned and reused) each step.
%%%

#### What you should see (build → init → apply → train — thread the boxes)
- `model.init(...)` returns a **params pytree** whose kernels are shaped `(2,4)` and `(4,3)` — sized from the dummy input, held by you.
- `model.apply(params, x)` gives the **same output twice** for the same input — it's a pure function.
- `jax.grad` (or `value_and_grad`) hands back a gradient pytree **shaped like params** — the compass.
- The training-loop loss **slides down** each step as Optax nudges the params — the model is learning.
- Forgetting to **catch the returned boxes** makes the loss **stay flat** (nothing learns); reassigning `params` and `opt_state` fixes it.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-05-flax-optax/log.md`: (1) in one sentence, where a Flax model's parameters live and why keeping them outside makes the model a pure function; (2) the two steps of a Flax model's life (`init` and `apply`) and what each one does; (3) why `optimizer.update` and `optax.apply_updates` return new boxes you must catch, and what happens if you don't.
!!!

@@@ fin
