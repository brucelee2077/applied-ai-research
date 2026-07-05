#!/usr/bin/env python3
"""Builder for M19 — Graph Systems & Link Prediction (wraps the People-You-May-Know case study).
Four lessons + review, rendered via _lesson_gen. Run: python3 sessions/_build_m19.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m19")
os.makedirs(OUT, exist_ok=True)

def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — Graphs & Message Passing ----------------
L("day-01-graphs.html", dict(
  qid="m19-d01-graphs", title="Module 19 · Day 1 — Graphs & Message Passing", nav_title="Spiral · M19 Day 1",
  eyebrow="Module 19 · Systems · Day 1", h1="Graphs & Message Passing",
  lead="Yesterday's world was rows in a table — one item at a time. But friends, web pages, and molecules are not rows: they are <em>connected</em>. A graph is the shape of connected data, and message passing is how each thing learns from the things it touches. This is the base every graph model — and the People-You-May-Know system — is built on.",
  goal="<b>🎯 By the end, you'll be able to:</b> say what a graph's nodes and edges are, run one round of message passing by hand (aggregate a node's neighbours), and name over-smoothing as the danger of too many rounds.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-gnns.html", next_label="Graph Neural Networks",
  sections=[
    {"title":"Data that is connected","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> you can read a vector (a list of numbers) and add vectors element-by-element (from the foundations). Nothing about graphs — we start from zero.</div></div>
     <h4>Rows vs connections</h4>
     <p>A ranking model (Module 18) sees one item at a time — a row of features. But a lot of real data is <em>connected</em>: people have friends, pages link to pages, atoms bond to atoms. To model that, we need a shape that stores the connections, not just the items.</p>
     <h4>A graph</h4>
     <p>A <span class="term" data-tip="A data structure made of nodes (the things) and edges (the connections between them). It stores relationships, not just items.">graph</span> has two parts. A <span class="term" data-tip="A node is one thing in the graph — a person, a web page, an atom. Each node carries a feature vector.">node</span> is one thing (a person). An <span class="term" data-tip="An edge is a link joining two nodes — it says these two things are connected (two people are friends).">edge</span> joins two nodes (two people are friends). Each node carries a <span class="term" data-tip="A feature vector is the list of numbers describing one node — its age, interests, activity, and so on.">feature vector</span>: a list of numbers describing it.</p>
     <h4>The core idea: you are shaped by your neighbours</h4>
     <p>The whole trick of graph learning is one sentence: <b>a node updates itself by looking at the nodes it is connected to.</b> A person's likely interests are partly their friends' interests. Doing this update is called <span class="term" data-tip="Message passing: each node builds a new feature vector by gathering (aggregating) its neighbours' feature vectors and mixing them with its own.">message passing</span>.</p>''',
     "gotit":"Got graphs and message passing"},
    {"title":"A friendship circle","body":
     '''<div class="relate">
       <div class="card"><span class="big">🧑‍🤝‍🧑</span><h5>You and your friends</h5><p>Picture yourself as a dot, with lines drawn to each of your friends. You are a node; the lines are edges. To guess a hobby you might like, a good first guess is: look at what your friends like. You "gather" their tastes.</p></div>
       <div class="card"><span class="big">📣</span><h5>A rumour spreads one hop</h5><p>In one round, each person hears only from people they directly know — one "hop" away. After a second round, news from friends-of-friends reaches you, because your friends already gathered it. Each round reaches one hop further.</p></div>
     </div>
     <p><strong>In one line:</strong> each round, every node gathers its direct neighbours' feature vectors and mixes them into its own — after k rounds, information from k hops away has reached it.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a rumour keeps its content as it spreads. In message passing, every round <i>averages</i> vectors together, so after many rounds everyone's vector drifts toward the same blur — the news is lost, not preserved. That is over-smoothing, which we meet in section 4.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"One round, by hand","body":
     '''<p>Below is one round of message passing on a tiny graph: read a node's neighbours, gather (average) their feature vectors, then combine with the node's own. <strong>Click all three.</strong></p>'''},
    {"title":"How one round of aggregation works","body":
     '''<h4>Step 1 — in words</h4>
     <p>To update node <code>v</code>: collect the feature vectors of every neighbour of <code>v</code>, squash them into one vector (we will average), then mix that with <code>v</code>'s own vector. The result is <code>v</code>'s new vector for this round.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       h<sub>v</sub><sup>new</sup> = COMBINE( h<sub>v</sub> ,  AGG<sub>u ∈ N(v)</sub> h<sub>u</sub> )<br>
       <span class="dim"># N(v) = the neighbours of v. AGG = aggregate (here: mean). COMBINE = mix self + neighbours.</span>
     </div></div>
     <p>Symbols: <code>h<sub>v</sub></code> = node <code>v</code>'s current feature vector. <code>N(v)</code> = the set of <code>v</code>'s neighbours (the <span class="term" data-tip="The neighbours of a node are all the nodes directly joined to it by an edge — one hop away.">neighbourhood</span>). <code>AGG</code> = an <span class="term" data-tip="Aggregation combines many neighbour vectors into one — a mean, a sum, or a max. It must not depend on the order of neighbours.">aggregation</span> that does not care about order (mean, sum, or max). <code>COMBINE</code> mixes the node's own vector with the gathered one.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       node A, own vector h_A = [2, 0]<br>
       neighbours of A: B=[0, 4], C=[4, 2]<br><br>
       AGG (mean of neighbours) = ([0,4] + [4,2]) / 2 = [4,6]/2 = [2, 3]<br>
       COMBINE (average self + neighbours) = ([2,0] + [2,3]) / 2 = <span class="hl">[2, 1.5]</span>
     </div></div>
     <p>Node A started at <code>[2,0]</code> and, after one round, became <code>[2,1.5]</code> — pulled toward its neighbours. That pull is the whole point: A now carries a hint of B and C.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — over-smoothing.</b> Each round averages neighbours in, so vectors keep moving toward each other. After too many rounds <b>every node's vector converges to almost the same value</b> — the graph becomes one grey blur and the model can no longer tell nodes apart. Nothing errors out: the loss may even look fine, but accuracy quietly collapses because the embeddings lost their individuality. You catch it by measuring how similar all node vectors have become, not by watching the loss.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — receptive field vs over-smoothing.</b> More rounds means each node's <span class="term" data-tip="The receptive field is how many hops away a node can see. After k rounds it covers everything within k hops.">receptive field</span> reaches further (k rounds → k hops), so it can use information from far across the graph. But more rounds also smooth harder, pushing you toward over-smoothing. You must pick k large enough to reach useful context, small enough to keep nodes distinct — usually just 2 or 3 rounds.</div></div>''',
     "gotit":"Got one round of aggregation"},
    {"title":"Building one round of message passing","body":
     '''<p>Here is one message-passing round, assembled piece by piece: the node, its neighbours, the messages, the aggregate, and the combined new vector. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a tiny graph","body":
     '''<p>Understanding is not the same as writing it. Build one round of message passing yourself. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m19_message_passing.py</code>. Make a 4-node graph with an edge list and a 2-number feature vector per node. Run one round of mean-aggregation message passing, printing each node's neighbours, the aggregated vector, and the new vector. Then run 10 rounds and print how similar all four vectors become (over-smoothing).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 19 Day 1 artifact.
Create experiments/foundations/m19_message_passing.py: a 4-node graph (edge list) with 2-dim node features. Run one round of mean-aggregation message passing and print each node's neighbours, aggregated neighbour vector, and new vector. Then run 10 rounds and print the pairwise similarity of all node vectors each round to demonstrate over-smoothing (vectors converge).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m19/day-01-log.md</code>: after how many rounds did the four node vectors become nearly identical?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "neighbours":{"btn":"① read the neighbours","html":'''<span class="prompt">&gt;&gt;&gt;</span> h = {"A":[2,0], "B":[0,4], "C":[4,2]}
<span class="prompt">&gt;&gt;&gt;</span> neighbours["A"]
<span class="hl">["B", "C"]</span>   <span class="dim"># A is connected to B and C</span>''',
      "take":"<b>①  Find who A touches.</b> Node A has two neighbours, B and C. Message passing will pull their vectors into A."},
    "agg":{"btn":"② aggregate the neighbours","html":'''<span class="prompt">&gt;&gt;&gt;</span> agg = mean([h["B"], h["C"]])
<span class="dim"># ([0,4] + [4,2]) / 2</span>
<span class="hl">agg = [2, 3]</span>   <span class="dim"># one vector summing up the neighbours</span>''',
      "take":"<b>②  Squash neighbours into one vector.</b> The mean of B and C is [2,3]. Mean does not care about order — that is required for graphs."},
    "combine":{"btn":"③ combine with self","html":'''<span class="prompt">&gt;&gt;&gt;</span> h_A_new = mean([h["A"], agg])
<span class="dim"># ([2,0] + [2,3]) / 2</span>
<span class="hl">h_A_new = [2, 1.5]</span>   <span class="dim"># A, nudged toward its friends</span>''',
      "take":"<b>③  Mix self and neighbours.</b> A moved from [2,0] to [2,1.5] — it now carries a hint of B and C. That is one round of message passing."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='12' text-anchor='middle'><circle cx='260' cy='45' r='22' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='42' fill='#1F6280'>A</text><text x='260' y='56' fill='#5A544E' font-size='10'>[2,0]</text></g></svg>",
     "note":"<b>Start with the node.</b> Node A carries a feature vector [2,0] — the numbers describing it right now, before this round."},
    {"viz":"<svg viewBox='0 0 520 120'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='260' cy='30' r='20' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='34' fill='#1F6280'>A</text><line x1='245' y1='45' x2='140' y2='90' stroke='#6B645E' stroke-width='1.5'/><line x1='275' y1='45' x2='380' y2='90' stroke='#6B645E' stroke-width='1.5'/><circle cx='130' cy='98' r='18' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='130' y='96' fill='#5E5191'>B</text><text x='130' y='108' fill='#5A544E' font-size='9'>[0,4]</text><circle cx='390' cy='98' r='18' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='390' y='96' fill='#5E5191'>C</text><text x='390' y='108' fill='#5A544E' font-size='9'>[4,2]</text></g></svg>",
     "note":"<b>Draw the edges to the neighbours.</b> A is joined to B and C. These are the only nodes A will hear from in one round — its one-hop neighbourhood."},
    {"viz":"<svg viewBox='0 0 520 110'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='260' cy='28' r='20' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='32' fill='#1F6280'>A</text><path d='M140,88 L245,42' stroke='#2D8B55' stroke-width='2' marker-end='url(#a1)'/><path d='M380,88 L275,42' stroke='#2D8B55' stroke-width='2' marker-end='url(#a1)'/><defs><marker id='a1' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0,0 L6,3 L0,6 Z' fill='#2D8B55'/></marker></defs><text x='130' y='100' fill='#5E5191'>msg [0,4]</text><text x='390' y='100' fill='#5E5191'>msg [4,2]</text></g></svg>",
     "note":"<b>Each neighbour sends a message.</b> B sends its vector [0,4] and C sends [4,2] toward A. A message is just the neighbour's feature vector travelling along the edge."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='120' y='20' width='280' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='40' fill='#9A7208'>AGG(mean) = ([0,4]+[4,2])/2 = [2,3]</text></g></svg>",
     "note":"<b>Aggregate the messages into one.</b> Average the two neighbour vectors → [2,3]. Aggregation must ignore order, so a node with 2 or 200 neighbours is handled the same way."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='90' y='20' width='340' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='40' fill='#1a5c38'>COMBINE([2,0],[2,3]) → hₐ = [2,1.5]</text></g></svg>",
     "note":"<b>Combine self with the aggregate.</b> Mix A's own [2,0] with the neighbour aggregate [2,3] → [2,1.5]. A now carries its friends' signal. One round done — stack rounds to reach further, but not too many (over-smoothing)."},
  ],
  quiz=[
    {"q":"1. In a graph, what is an edge?","opts":["a feature vector of a node","a link joining two nodes (they are connected)","the number of rounds","the aggregation function"],"ans":1,"fb":"An edge joins two nodes — it records that they are connected. Nodes are the things; edges are the connections."},
    {"q":"2. What does one round of message passing do to a node?","opts":["deletes its neighbours","updates it by aggregating its neighbours' vectors and mixing with its own","sorts the graph","removes its feature vector"],"ans":1,"fb":"Each round, a node gathers its neighbours' feature vectors, aggregates them, and combines with its own vector to get a new vector."},
    {"q":"3. Node X = [6,0]. Its two neighbours are [2,2] and [0,4]. Using mean aggregation and then averaging with X's own vector, X's new vector is:","opts":["[3.5, 1.5]","[8, 6]","[1, 3]","[3, 2]"],"ans":0,"fb":"Neighbour mean = ([2,2]+[0,4])/2 = [1,3]. Combine (average with self) = ([6,0]+[1,3])/2 = [7,3]/2 = [3.5,1.5]. Node X is pulled toward its neighbours."},
    {"q":"4. You stack 20 message-passing rounds and all node vectors become nearly identical. This is:","opts":["perfect training","over-smoothing — too many rounds averaged the nodes into one blur","a broken edge list","a large receptive field with no downside"],"ans":1,"fb":"Over-smoothing: repeated averaging pulls every node toward the same value, so the model can no longer tell nodes apart. Usually 2–3 rounds is enough."},
  ],
  fin={"em":"🕸️","h3":"Day 1 complete — you can pass messages on a graph!",
       "p":"You now have the shape of connected data: nodes, edges, feature vectors, and the one move that powers all graph learning — <code>h<sub>v</sub><sup>new</sup> = COMBINE(h<sub>v</sub>, AGG neighbours)</code>. You ran a round by hand and met over-smoothing and the receptive-field trade-off. Next: <b>Day 2</b> — stack these rounds into a Graph Neural Network."},
))
# ---------------- Day 2 — Graph Neural Networks ----------------
L("day-02-gnns.html", dict(
  qid="m19-d02-gnns", title="Module 19 · Day 2 — Graph Neural Networks", nav_title="Spiral · M19 Day 2",
  eyebrow="Module 19 · Systems · Day 2", h1="Graph Neural Networks",
  lead="One round of message passing lets a node see its friends. Stack a few rounds, put a small learned transform in each, and you have a Graph Neural Network — a model that turns every node into a rich embedding shaped by its whole neighbourhood. This is the model that powers friend suggestion; the only new problem is scale.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how stacking message-passing layers makes a GNN, say what a node embedding is, and describe why neighbour sampling (GraphSAGE) is needed to train on huge graphs.",
  prev_href="day-01-graphs.html", prev_label="Graphs & Message Passing", next_href="day-03-link-prediction.html", next_label="Link Prediction",
  sections=[
    {"title":"Message passing, made learnable","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 — nodes, edges, and one round of message passing (aggregate neighbours, combine with self).</div></div>
     <h4>Add a learned transform to each round</h4>
     <p>Day 1's round just averaged vectors. A <span class="term" data-tip="Graph Neural Network: a model made by stacking message-passing layers, each with learned weights, so node vectors are transformed as they aggregate neighbours.">Graph Neural Network (GNN)</span> makes each round <em>learnable</em>: after aggregating neighbours, it multiplies by a weight matrix and applies a non-linearity — the same idea as one layer of a normal neural network, but the input is "self + neighbours." Stack L of these and you have an L-layer GNN.</p>
     <h4>Two names you will hear</h4>
     <p>A <span class="term" data-tip="Graph Convolutional Network: a GNN where each layer aggregates neighbours with a normalized mean, then applies a learned weight and non-linearity.">GCN</span> aggregates neighbours with a normalized average. <span class="term" data-tip="GraphSAGE: a GNN that samples a fixed number of neighbours per node instead of using all of them, so it scales to huge graphs.">GraphSAGE</span> is almost the same but adds one crucial idea for scale — it samples a fixed number of neighbours instead of using all of them.</p>
     <h4>The output: a node embedding</h4>
     <p>After the last layer, each node has a <span class="term" data-tip="A node embedding is the final learned feature vector for a node — a compact list of numbers that captures the node and its neighbourhood, ready for a task.">node embedding</span>: a compact vector that captures the node <em>and</em> its L-hop neighbourhood. Two nodes with similar embeddings are similar in the graph — which is exactly what we will use tomorrow to predict friendships.</p>''',
     "gotit":"Got how a GNN stacks rounds"},
    {"title":"Layers of understanding","body":
     '''<div class="relate">
       <div class="card"><span class="big">🥞</span><h5>Stacking layers</h5><p>Layer 1 lets you describe a person from their direct friends. Layer 2 takes those descriptions and blends friends-of-friends in. Each layer is a smarter round: it looks one hop further and learns <i>how</i> to mix, not just to average.</p></div>
       <div class="card"><span class="big">📋</span><h5>Sampling a few friends</h5><p>If someone has 5,000 friends, reading all of them every round is impossible. Instead, pick a small handful at random each time — like polling 20 friends instead of all 5,000. You still get the gist, far cheaper. That is neighbour sampling.</p></div>
     </div>
     <p><strong>In one line:</strong> a GNN stacks learnable message-passing layers to build a rich embedding per node, and on big graphs it samples a few neighbours per layer instead of all of them.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: pancakes are identical; GNN layers each have their <i>own</i> learned weights, so layer 1 and layer 2 do different jobs. And polling friends is optional for you — for a billion-node graph, sampling is not optional, it is the only way training fits in memory.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Two layers, one node","body":
     '''<p>Watch a 2-layer GNN build a node embedding: layer 1 mixes direct neighbours, layer 2 reaches two hops, and neighbour sampling caps the cost. <strong>Click all three.</strong></p>'''},
    {"title":"How a GNN layer works, and why it must sample","body":
     '''<h4>Step 1 — in words</h4>
     <p>One GNN layer does three things: aggregate the neighbour embeddings, concatenate or add the node's own embedding, then apply a learned weight matrix <code>W</code> and a non-linearity (like ReLU). Stack L layers and node <code>v</code> sees everything within L hops.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       h<sub>v</sub><sup>(l+1)</sup> = σ( W<sup>(l)</sup> · COMBINE( h<sub>v</sub><sup>(l)</sup> , AGG<sub>u ∈ N(v)</sub> h<sub>u</sub><sup>(l)</sup> ) )<br>
       <span class="dim"># l = layer index. W = learned weights for that layer. σ = non-linearity (ReLU).</span>
     </div></div>
     <p>Symbols: <code>h<sub>v</sub><sup>(l)</sup></code> = node <code>v</code>'s embedding at layer <code>l</code>. <code>W<sup>(l)</sup></code> = that layer's learned weight matrix. <code>σ</code> = a non-linearity so the model can learn more than straight lines. It is Day 1's round with a trainable brain bolted on.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       neighbour aggregate a = [2, 3],  self h_v = [2, 0]<br>
       combine (sum) = [4, 3]<br>
       W = [[1, 0],[0, 1]] (identity, for the example)  →  W·[4,3] = [4, 3]<br>
       σ = ReLU (keep positives) → <span class="hl">h_v' = [4, 3]</span>
     </div></div>
     <p>With real learned <code>W</code> the layer would reshape [4,3] into something task-useful; here identity keeps the arithmetic clear. Stack another layer and node <code>v</code> now blends two-hop information.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — full-graph training runs out of memory.</b> To update one node with all its neighbours, then <i>their</i> neighbours for the next layer, the number of nodes touched explodes — for L layers on a dense graph it can reach the whole graph per node. On a billion-node social graph this <b>does not fit in GPU memory</b>: the job crashes with an out-of-memory error, or, worse, silently thrashes to a crawl. The fix is neighbour sampling: cap each node to, say, 25 neighbours per layer, so the cost per node is bounded no matter how large the graph.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — sampling vs full neighbourhood.</b> Sampling a fixed few neighbours (GraphSAGE) makes training scale to billions of nodes and adds useful randomness — but each node sees only a noisy slice of its true neighbourhood, so a single pass is less accurate. Using the full neighbourhood (GCN) is more exact per node but cannot scale. Big production systems almost always sample; they trade a little per-step accuracy for the ability to train at all.</div></div>''',
     "gotit":"Got the GNN layer and sampling"},
    {"title":"Building a 2-layer GNN","body":
     '''<p>Here is a GNN assembled layer by layer: input features, layer-1 mixing, layer-2 reaching further, neighbour sampling, and the final node embedding. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a small GNN","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m19_gnn.py</code>. Build a 2-layer GNN (each layer: mean-aggregate neighbours, concat self, apply a small random weight + ReLU) on a ~30-node graph. Print each node's embedding shape after each layer. Then add neighbour sampling (cap 5 neighbours) and print how many nodes get touched per update with vs without sampling.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 19 Day 2 artifact.
Create experiments/foundations/m19_gnn.py: a 2-layer GraphSAGE-style GNN on a ~30-node graph. Each layer mean-aggregates neighbours, concatenates the node's own embedding, applies a learned weight + ReLU. Print embedding shapes per layer. Add neighbour sampling (cap 5 per node) and print the count of nodes touched per node-update with vs without sampling, showing the memory blow-up sampling avoids.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m19/day-02-log.md</code>: how many nodes were touched per update with vs without neighbour sampling?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "layer1":{"btn":"① layer 1: direct friends","html":'''<span class="prompt">&gt;&gt;&gt;</span> h1 = relu(W1 @ combine(h[v], agg(neighbours(v))))
<span class="dim"># aggregate direct neighbours, transform, non-linearity</span>
<span class="hl">h1[v] shape = (16,)</span>   <span class="dim"># a 16-dim embedding of v + 1-hop</span>''',
      "take":"<b>①  Layer 1 sees one hop.</b> It mixes v with its direct neighbours and learns a transform. Now v's vector encodes its immediate circle."},
    "layer2":{"btn":"② layer 2: friends of friends","html":'''<span class="prompt">&gt;&gt;&gt;</span> h2 = relu(W2 @ combine(h1[v], agg(neighbours(v))))
<span class="dim"># neighbours' vectors already hold THEIR neighbours</span>
<span class="hl">h2[v] reaches 2 hops away</span>''',
      "take":"<b>②  Layer 2 reaches two hops.</b> Because each neighbour already gathered its own neighbours in layer 1, v now indirectly sees friends-of-friends."},
    "sample":{"btn":"③ neighbour sampling caps cost","html":'''<span class="prompt">&gt;&gt;&gt;</span> nbrs = sample(neighbours(v), k=25)   <span class="dim"># v has 5000 friends</span>
<span class="hl">len(nbrs) = 25</span>   <span class="dim"># fixed cost per node, per layer</span>
<span class="dim"># without this, 2 layers could touch the whole graph</span>''',
      "take":"<b>③  Sample, don't read all.</b> Capping neighbours to 25 keeps memory bounded no matter how large the graph — the key that lets GNNs train at web scale."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>input: raw node features</text></g></svg>",
     "note":"<b>Start with raw features.</b> Every node begins with its own feature vector — age, interests, activity — before any layer runs."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>layer 1: σ(W₁·[self + agg 1-hop])</text></g></svg>",
     "note":"<b>Layer 1 mixes direct neighbours.</b> Aggregate one-hop neighbours, add self, apply learned weights W₁ and a non-linearity. Each node now encodes its immediate circle."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>layer 2: σ(W₂·[self + agg 1-hop])</text></g></svg>",
     "note":"<b>Layer 2 reaches two hops.</b> The neighbours' layer-1 vectors already hold their neighbours, so stacking a second layer pulls in friends-of-friends — the receptive field grows one hop per layer."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='260' cy='24' r='16' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='28' fill='#1F6280'>v</text><circle cx='150' cy='62' r='12' fill='#E8F5EE' stroke='#2D8B55'/><circle cx='210' cy='66' r='12' fill='#E8F5EE' stroke='#2D8B55'/><circle cx='310' cy='66' r='12' fill='#E8F5EE' stroke='#2D8B55'/><circle cx='370' cy='62' r='12' fill='#E5DFD6' stroke='#C99A12' stroke-dasharray='3,2'/><line x1='253' y1='38' x2='153' y2='52'/><line x1='257' y1='39' x2='211' y2='54'/><line x1='263' y1='39' x2='309' y2='54'/><text x='260' y='79' fill='#9A7208'>sample 3 of many neighbours (dashed = skipped)</text></g></svg>",
     "note":"<b>Neighbour sampling caps the fan-out.</b> Instead of reading all neighbours, pick a fixed few (green). The rest (dashed) are skipped this pass. Cost per node stays bounded — this is what makes GraphSAGE scale."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>output: node embedding zᵥ</text></g></svg>",
     "note":"<b>The node embedding.</b> After the last layer, each node has a compact vector capturing it and its L-hop neighbourhood. Similar embeddings = similar nodes — the raw material for link prediction tomorrow."},
  ],
  quiz=[
    {"q":"1. What turns a plain message-passing round into a GNN layer?","opts":["deleting edges","adding learned weights W and a non-linearity to each round","using more nodes","removing the neighbours"],"ans":1,"fb":"A GNN layer aggregates neighbours then applies a learned weight matrix and a non-linearity — so the model learns how to mix, not just to average."},
    {"q":"2. A 3-layer GNN lets each node see information from how far away?","opts":["1 hop","3 hops","the whole graph always","0 hops"],"ans":1,"fb":"Each layer reaches one hop further, so L layers cover the L-hop neighbourhood. A 3-layer GNN sees up to 3 hops."},
    {"q":"3. Why does GraphSAGE sample a fixed number of neighbours?","opts":["to make the graph smaller permanently","to bound the cost per node so training fits in memory on huge graphs","to remove the non-linearity","to avoid computing embeddings"],"ans":1,"fb":"Without sampling, the neighbours-of-neighbours per node can explode toward the whole graph. Capping neighbours keeps memory and compute bounded at web scale."},
    {"q":"4. Your GNN crashes with an out-of-memory error on a billion-node graph using full neighbourhoods. Best fix?","opts":["add more layers","use neighbour sampling to cap the fan-out per node","increase the learning rate","delete the node features"],"ans":1,"fb":"Full-neighbourhood training touches too many nodes per update. Neighbour sampling caps the fan-out, so each update fits in memory — the standard fix."},
  ],
  fin={"em":"🥞","h3":"Day 2 complete — you can build a GNN!",
       "p":"You stacked learnable message-passing layers into a GNN: <code>h<sup>(l+1)</sup> = σ(W·COMBINE(self, AGG neighbours))</code>, each layer reaching one hop further to produce a node embedding. You saw why full-graph training runs out of memory and how neighbour sampling (GraphSAGE) fixes it. Next: <b>Day 3</b> — use those embeddings to predict links."},
))
# ---------------- Day 3 — Link Prediction ----------------
L("day-03-link-prediction.html", dict(
  qid="m19-d03-linkpred", title="Module 19 · Day 3 — Link Prediction", nav_title="Spiral · M19 Day 3",
  eyebrow="Module 19 · Systems · Day 3", h1="Link Prediction",
  lead="Now we have a node embedding for every person. Link prediction asks a simple question with those embeddings: <b>should an edge exist between these two nodes that is not there yet?</b> If two people's embeddings are similar, they probably should be friends. That single idea is the engine of the People-You-May-Know system.",
  goal="<b>🎯 By the end, you'll be able to:</b> score a candidate edge from two node embeddings (sigmoid of a dot product), name the severe class imbalance of graphs, and explain why we use candidate generation instead of scoring all n² pairs.",
  prev_href="day-02-gnns.html", prev_label="Graph Neural Networks", next_href="day-04-negative-sampling.html", next_label="Negative Sampling for Graphs",
  sections=[
    {"title":"Guessing the missing edges","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 — node embeddings, and from the foundations: the dot product of two vectors and the sigmoid function that squashes a number into 0–1.</div></div>
     <h4>The task</h4>
     <p><span class="term" data-tip="Link prediction: given a graph, predict which edges are missing but should exist — for example, which two people are likely to become friends.">Link prediction</span> takes the graph you have and predicts the edges you do not. In a social network, the edges you do not have yet are exactly the friendships worth suggesting.</p>
     <h4>Score from similarity</h4>
     <p>You already turned every node into an embedding (Day 2), and similar embeddings mean similar nodes. So to score a possible edge between nodes <code>u</code> and <code>v</code>, measure how aligned their embeddings are. The simplest measure is the <span class="term" data-tip="The dot product multiplies two vectors element-by-element and sums the results — a single number that is large when the vectors point the same way.">dot product</span>, then squash it to a probability with a <span class="term" data-tip="The sigmoid function maps any real number to a value between 0 and 1, so it can be read as a probability.">sigmoid</span>.</p>
     <h4>Contrast with ranking</h4>
     <p>Module 18 ranked <em>items for a user</em> from features. Link prediction ranks <em>nodes for a node</em> from graph structure. The scoring shape is similar — a similarity turned into a probability — but the signal comes from who-is-connected-to-whom, not from a feature table.</p>''',
     "gotit":"Got the link-prediction task"},
    {"title":"Do you two know the same people?","body":
     '''<div class="relate">
       <div class="card"><span class="big">👥</span><h5>Shared circles</h5><p>If you and a stranger share fifteen mutual friends, go to the same school, and like the same teams, you two probably should know each other. A good embedding packs all of that into a vector, so two similar vectors flag a likely friendship.</p></div>
       <div class="card"><span class="big">🧲</span><h5>Aligned arrows</h5><p>Think of each embedding as an arrow in space. Two arrows pointing the same way have a big dot product — a high score. Arrows pointing different ways score near zero. The sigmoid then turns that score into a "probability they should connect."</p></div>
     </div>
     <p><strong>In one line:</strong> embed every node, then score a possible edge as sigmoid(u · v) — high when the two nodes' vectors point the same way.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: two arrows can point the same way for the wrong reason — everyone in a huge popular group looks similar. Raw similarity can suggest edges to celebrities everyone already resembles. Good systems correct for this; a plain dot product does not.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Score a candidate edge","body":
     '''<p>Watch two embeddings turn into an edge score: take the dot product, apply the sigmoid, and read it as a probability. <strong>Click all three.</strong></p>'''},
    {"title":"Scoring an edge, and why n² hurts","body":
     '''<h4>Step 1 — in words</h4>
     <p>To score the edge between <code>u</code> and <code>v</code>: line up their embeddings, take the dot product (one number, big when they align), then pass it through a sigmoid so it reads as a probability between 0 and 1.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       score(u, v) = σ( z<sub>u</sub> · z<sub>v</sub> )   where  σ(x) = 1 / (1 + e<sup>−x</sup>)<br>
       <span class="dim"># z_u, z_v = the two node embeddings. σ = sigmoid → a probability in (0,1).</span>
     </div></div>
     <p>Symbols: <code>z<sub>u</sub></code>, <code>z<sub>v</sub></code> = the embeddings of the two nodes. <code>z<sub>u</sub> · z<sub>v</sub></code> = their dot product. <code>σ</code> = sigmoid. A large positive dot product → score near 1 (likely edge); a negative one → score near 0 (unlikely).</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       z_u = [1, 2],  z_v = [2, 1]<br>
       dot = 1·2 + 2·1 = 2 + 2 = 4<br>
       σ(4) = 1 / (1 + e<sup>−4</sup>) = 1 / (1 + 0.0183) ≈ <span class="hl">0.98</span><br>
       <span class="dim"># 0.98 → the model is very confident this edge should exist</span>
     </div></div>
     <p>Compare a mismatched pair: <code>z_a=[1,2]</code>, <code>z_b=[−2,1]</code> → dot = −2+2 = 0 → σ(0)=0.5 (a coin flip). Aligned vectors score high; orthogonal ones score neutral.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — class imbalance.</b> In a real graph almost every possible edge does <b>not</b> exist: a network with a million people has ~10¹² possible pairs but each person has only a few hundred friends, so real edges are a <b>tiny fraction of a percent</b>. A model that predicts "no edge" for everything scores 99.99%+ accuracy while being useless. Accuracy looks amazing; the system finds no friends. You must measure ranking metrics (like precision@k or AUC), never plain accuracy, and train against sampled negatives (Day 4).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — scoring all pairs (O(n²)) vs candidate generation.</b> Scoring every pair is exact but costs n² — for a billion nodes that is 10¹⁸ scores, impossible. <span class="term" data-tip="Candidate generation narrows the huge set of possible edges down to a small promising shortlist (e.g. friends-of-friends) before the expensive scoring model runs.">Candidate generation</span> first narrows to a cheap shortlist (e.g. only friends-of-friends, or approximate nearest-neighbour lookup), then scores just those. You trade a little recall — a good pair might be missed by the shortlist — for making the whole thing feasible.</div></div>
     <div class="callout c-info"><span class="ic">🔗</span><div><b>Go deeper ↗</b> — the full production design (candidate generation, features, serving, monitoring) lives in the <a href="../../ML%20Design/11-people-you-may-know/README.md">People-You-May-Know case study</a>.</div></div>''',
     "gotit":"Got edge scoring and the two failure modes"},
    {"title":"Building the edge scorer","body":
     '''<p>Here is link prediction assembled: two node embeddings, their dot product, the sigmoid, the score, and the candidate-generation funnel that avoids n². <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a scorer","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m19_link_prediction.py</code>. Give ~50 nodes random 8-dim embeddings. Score a few candidate edges as sigmoid(dot). Count real edges vs total possible pairs to show the imbalance. Then implement a friends-of-friends candidate generator and print how many pairs it scores vs the full n² count.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 19 Day 3 artifact.
Create experiments/foundations/m19_link_prediction.py: ~50 nodes with 8-dim embeddings. Score candidate edges as sigmoid(dot(z_u, z_v)). Print the ratio of real edges to total possible pairs (n choose 2) to show class imbalance. Add a friends-of-friends candidate generator and print how many pairs it scores vs full n^2, illustrating the O(n^2) vs candidate-generation trade-off.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m19/day-03-log.md</code>: what fraction of possible pairs were real edges, and how many pairs did candidate generation score vs full n²?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "dot":{"btn":"① dot product of embeddings","html":'''<span class="prompt">&gt;&gt;&gt;</span> z_u, z_v = [1,2], [2,1]
<span class="prompt">&gt;&gt;&gt;</span> dot = 1*2 + 2*1
<span class="hl">dot = 4</span>   <span class="dim"># big → the two arrows point the same way</span>''',
      "take":"<b>①  Measure alignment.</b> The dot product of the two embeddings is 4 — a large positive number means the nodes are similar."},
    "sigmoid":{"btn":"② sigmoid → probability","html":'''<span class="prompt">&gt;&gt;&gt;</span> score = sigmoid(4)   <span class="dim"># 1/(1+e^-4)</span>
<span class="hl">score = 0.98</span>   <span class="dim"># confident this edge should exist</span>''',
      "take":"<b>②  Squash to a probability.</b> The sigmoid turns 4 into 0.98 — the model strongly predicts an edge between u and v."},
    "candidate":{"btn":"③ candidate generation","html":'''<span class="prompt">&gt;&gt;&gt;</span> all_pairs = n*(n-1)/2   <span class="dim"># n=1e6 → 5e11 pairs!</span>
<span class="prompt">&gt;&gt;&gt;</span> candidates = friends_of_friends(u)   <span class="hl"># ~a few hundred</span>
<span class="prompt">&gt;&gt;&gt;</span> [score(u, c) for c in candidates]   <span class="dim"># only score the shortlist</span>''',
      "take":"<b>③  Don't score everyone.</b> Scoring all n² pairs is impossible; generate a small candidate shortlist (friends-of-friends) first, then score only those."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='70' y='22' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='145' y='41' fill='#1F6280'>zᵤ = [1,2]</text><rect x='300' y='22' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='375' y='41' fill='#1F6280'>zᵥ = [2,1]</text></g></svg>",
     "note":"<b>Start with two node embeddings.</b> Each is the vector a GNN produced for a node (Day 2). We want a score for the possible edge between them."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>zᵤ · zᵥ = 1·2 + 2·1 = 4</text></g></svg>",
     "note":"<b>Take the dot product.</b> One number that is large when the two embeddings point the same way. Here it is 4 — the nodes are well aligned."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>σ(4) = 1/(1+e⁻⁴) ≈ 0.98</text></g></svg>",
     "note":"<b>Sigmoid → probability.</b> The sigmoid squashes 4 into 0.98, a probability in (0,1). Close to 1 means \"this edge very likely should exist.\""},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='10' text-anchor='middle'><text x='260' y='16' fill='#C93B3B'>possible pairs ≈ n² (astronomically many)</text><polygon points='60,26 460,26 340,50 180,50' fill='#FDE8E8' stroke='#C93B3B'/><polygon points='180,52 340,52 300,72 220,72' fill='#E8F5EE' stroke='#2D8B55'/><text x='260' y='66' fill='#1a5c38'>candidate shortlist</text></g></svg>",
     "note":"<b>Funnel down before scoring.</b> The full set of pairs is ~n² — impossible to score. Candidate generation (friends-of-friends, nearest-neighbour) narrows to a small shortlist first."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>score only the shortlist → rank → suggest top-k</text></g></svg>",
     "note":"<b>Score, rank, suggest.</b> Run sigmoid(dot) only on the candidates, sort by score, and surface the top few. Because real edges are so rare, judge quality by precision@k / AUC, not accuracy."},
  ],
  quiz=[
    {"q":"1. How does link prediction score a possible edge between nodes u and v?","opts":["by counting their nodes","by turning their embeddings' similarity (e.g. sigmoid of a dot product) into a probability","by deleting one of them","by their alphabetical order"],"ans":1,"fb":"Score = sigmoid(z_u · z_v). Similar embeddings → high dot product → high edge probability."},
    {"q":"2. Two embeddings are z_u=[3,0] and z_v=[0,3]. Their dot product is:","opts":["9","0","6","3"],"ans":1,"fb":"Dot = 3·0 + 0·3 = 0. Orthogonal vectors give 0, so sigmoid(0)=0.5 — a neutral, coin-flip score."},
    {"q":"3. A social graph has 1,000,000 people but each has only ~200 friends. Why is plain accuracy a bad metric for link prediction?","opts":["accuracy is always bad","almost no pairs are real edges, so 'predict no edge' scores ~100% accuracy yet finds no friends","there are too few people","edges are undirected"],"ans":1,"fb":"Severe class imbalance: real edges are a tiny fraction of all pairs, so a do-nothing model looks near-perfect. Use precision@k or AUC instead."},
    {"q":"4. Why do production systems use candidate generation instead of scoring all pairs?","opts":["to make embeddings larger","scoring all n² pairs is infeasible at scale, so a cheap shortlist is scored instead","to remove the sigmoid","candidate generation is always more accurate"],"ans":1,"fb":"All-pairs scoring is O(n²) — 10¹⁸ scores for a billion nodes. Candidate generation narrows to a promising shortlist first, trading a little recall for feasibility."},
  ],
  fin={"em":"🔗","h3":"Day 3 complete — you can predict a link!",
       "p":"You can now score a candidate edge as <code>σ(z<sub>u</sub> · z<sub>v</sub>)</code>, and you know the two things that make it hard at scale: severe class imbalance (use precision@k, not accuracy) and the O(n²) blow-up (use candidate generation). Next: <b>Day 4</b> — since you only ever see positive edges, how do you sample the negatives you need to train?"},
))
# ---------------- Day 4 — Negative Sampling for Graphs ----------------
L("day-04-negative-sampling.html", dict(
  qid="m19-d04-negs", title="Module 19 · Day 4 — Negative Sampling for Graphs", nav_title="Spiral · M19 Day 4",
  eyebrow="Module 19 · Systems · Day 4", h1="Negative Sampling for Graphs",
  lead="A graph only ever shows you the edges that exist — the friendships people <em>have</em>. But to train a link predictor you also need examples of pairs that should <em>not</em> connect. You never observe those directly, so you must make them up by sampling. How you pick these negatives decides whether the model learns anything at all — and it is exactly how People-You-May-Know is trained.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why link prediction needs sampled negatives, tell random from hard negatives, and describe how friend suggestion is framed as link prediction trained on positives + sampled negatives.",
  prev_href="day-03-link-prediction.html", prev_label="Link Prediction", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"You only see the 'yes' examples","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 3 — scoring an edge as sigmoid(dot), and the class-imbalance problem (real edges are rare).</div></div>
     <h4>Positives are given, negatives are not</h4>
     <p>The edges in the graph are your <span class="term" data-tip="A positive example for link prediction: a real edge that exists in the graph (two people who are friends).">positive examples</span> — pairs that <em>should</em> connect, because they do. But a classifier also needs <span class="term" data-tip="A negative example for link prediction: a pair of nodes with no edge, used to teach the model what a non-connection looks like. These are not observed and must be sampled.">negative examples</span> — pairs that should <em>not</em> connect. The graph does not label those. You have to create them.</p>
     <h4>The fix: sample negatives</h4>
     <p><span class="term" data-tip="Negative sampling: creating training negatives by picking node pairs that have no edge, since only positive edges are observed.">Negative sampling</span> makes negatives by picking pairs with no edge between them. The simplest way: for each real edge (u, v), also pick a random node w that u is not connected to, and label (u, w) a negative. Now the model has both "yes" and "no" examples to learn the difference.</p>
     <h4>Not all negatives are equal</h4>
     <p>A <b>random</b> negative is usually two people from totally different worlds — plainly unconnected. A <span class="term" data-tip="A hard negative: a non-edge that looks like it could be an edge — e.g. two people with many mutual friends who are still not connected. It forces the model to learn subtle differences.">hard negative</span> is a pair that <em>looks</em> connectable (many mutual friends) but is not. Hard negatives teach far more, because they sit near the decision boundary.</p>''',
     "gotit":"Got why we sample negatives"},
    {"title":"Studying with real distractors","body":
     '''<div class="relate">
       <div class="card"><span class="big">📝</span><h5>A quiz needs wrong answers</h5><p>If every quiz answer is "true," you learn nothing — you just say "true" always and score 100%. You need believable wrong options to prove you understand. Negatives are the wrong options; without them the model just says "edge" to everything.</p></div>
       <div class="card"><span class="big">🎯</span><h5>Easy vs tricky distractors</h5><p>A distractor like "2 + 2 = 900" teaches nothing. "2 + 2 = 5" makes you think. Random negatives are the silly distractor; hard negatives are the tricky one that forces real understanding.</p></div>
     </div>
     <p><strong>In one line:</strong> since only real edges are observed, invent negatives by sampling non-edges — and prefer hard negatives (look connectable but are not) because they carry the most learning signal.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a quiz-writer knows for certain which answers are wrong. Here a sampled "negative" might actually be a true friendship the graph just has not recorded yet (a <span class="term" data-tip="A false negative: a sampled non-edge that is really a missing true edge — the connection exists but was not recorded in the graph.">false negative</span>). You are punishing a correct prediction, which adds noise.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Sample a negative","body":
     '''<p>Watch a real edge become a positive, a random non-edge become an easy negative, and a mutual-friend non-edge become a hard negative. <strong>Click all three.</strong></p>'''},
    {"title":"Sampling negatives, and how PYMK is trained","body":
     '''<h4>Step 1 — in words</h4>
     <p>For each positive edge, sample one or more node pairs that have no edge and label them negative. Train the scorer so positives get a score near 1 and negatives near 0. The loss is the same binary cross-entropy you would use for any yes/no classifier.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       loss = − log σ(z<sub>u</sub>·z<sub>v</sub>)  −  Σ<sub>w</sub> log( 1 − σ(z<sub>u</sub>·z<sub>w</sub>) )<br>
       <span class="dim"># (u,v) = a real edge (positive). each w = a sampled non-neighbour (negative).</span>
     </div></div>
     <p>Symbols: the first term pushes the positive edge's score <em>up</em> (toward 1). The second term, over each sampled negative <code>w</code>, pushes those scores <em>down</em> (toward 0). Together they teach the model to separate real edges from non-edges.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       positive (u,v): score σ(z_u·z_v) = 0.9  → −log(0.9) = <span class="hl">0.105</span> (small loss, good)<br>
       negative (u,w): score σ(z_u·z_w) = 0.8  → −log(1−0.8) = −log(0.2) = <span class="hl">1.609</span> (big loss!)<br>
       <span class="dim"># the negative scored high → large penalty → the model pushes z_u, z_w apart</span>
     </div></div>
     <p>The hard negative (scored 0.8 when it should be 0) creates a big loss, so training focuses on fixing it. An easy negative already scoring 0.01 would add almost no loss — and almost no learning.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — easy negatives teach nothing.</b> If you only sample random negatives, they are nearly always obvious non-edges (two strangers from different countries). The model separates them with no effort, the loss drops to near zero, and training <b>looks</b> finished — but the model never learned the hard cases (people with mutual friends who still are not connected), which are the ones that matter for good suggestions. The loss curve looks great; the suggestions are bland. You fix it by mixing in hard negatives.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — negative quality vs sampling cost.</b> Random negatives are free to draw but weak. Hard negatives (mine mutual-friend non-edges, or the highest-scoring wrong pairs) teach far more but cost compute to find, and if pushed too far they overlap with false negatives (real-but-unrecorded edges) and inject noise. The usual recipe: mostly random negatives for coverage, a controlled dose of hard negatives for signal.</div></div>
     <h4>How People-You-May-Know uses all of this</h4>
     <p>Friend suggestion is <em>exactly</em> link prediction: build the friend graph, learn a node embedding per person (GNN, Day 2), score candidate friendships as sigmoid(dot) (Day 3), and train on the observed friendships as positives plus sampled non-friendships as negatives (today). Candidate generation (friends-of-friends) supplies both the pairs to score at serving time and a natural source of hard negatives.</p>
     <div class="callout c-info"><span class="ic">🔗</span><div><b>Go deeper ↗</b> — the end-to-end system, including how negatives, features, and serving fit together, is in the <a href="../../ML%20Design/11-people-you-may-know/README.md">People-You-May-Know case study</a>.</div></div>''',
     "gotit":"Got negative sampling and PYMK"},
    {"title":"Building the training signal","body":
     '''<p>Here it is assembled: the observed positives, a sampled random negative, a mined hard negative, the combined loss, and the friend-suggestion pipeline. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with negatives","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m19_negative_sampling.py</code>. Take your Day-3 graph. For each real edge, sample (a) a random negative and (b) a hard negative (a non-edge with the most mutual neighbours). Train the sigmoid(dot) scorer with each strategy and print final precision@k. Show the hard-negative model ranks true friends higher.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 19 Day 4 artifact.
Create experiments/foundations/m19_negative_sampling.py: train a sigmoid(dot) link predictor on a small friend graph. For each positive edge, sample (a) random negatives and (b) hard negatives (non-edges with the most mutual neighbours). Use loss = -log σ(pos) - Σ log(1-σ(neg)). Compare final precision@k for random-only vs hard-negative training, showing hard negatives rank true links higher. Note any false-negative risk.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m19/day-04-log.md</code>: did hard negatives improve precision@k over random-only, and by how much?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "pos":{"btn":"① a positive (real edge)","html":'''<span class="prompt">&gt;&gt;&gt;</span> edges  <span class="dim"># the only labels the graph gives you</span>
<span class="hl">(u, v) is an edge → label = 1</span>
<span class="dim"># u and v are friends: score should be near 1</span>''',
      "take":"<b>①  Positives are the real edges.</b> Every friendship in the graph is a 'yes' example. But the graph never lists the 'no' examples."},
    "rand":{"btn":"② a random negative","html":'''<span class="prompt">&gt;&gt;&gt;</span> w = random_node(); assert not edge(u, w)
<span class="hl">(u, w) → label = 0</span>   <span class="dim"># strangers, easy 'no'</span>
<span class="dim"># model separates this easily → little learning</span>''',
      "take":"<b>②  Random negatives are easy.</b> Two unrelated people — an obvious non-edge. Cheap to sample, but it teaches the model almost nothing."},
    "hard":{"btn":"③ a hard negative","html":'''<span class="prompt">&gt;&gt;&gt;</span> w = most_mutual_friends_but_not_connected(u)
<span class="hl">(u, w) → label = 0</span>   <span class="dim"># looks connectable, but isn't</span>
<span class="dim"># big loss if scored high → strong learning signal</span>''',
      "take":"<b>③  Hard negatives teach.</b> A pair with many mutual friends that still is not an edge sits near the boundary — fixing it forces real understanding."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='180' cy='40' r='16' fill='#E8F5EE' stroke='#2D8B55'/><text x='180' y='44' fill='#1a5c38'>u</text><circle cx='340' cy='40' r='16' fill='#E8F5EE' stroke='#2D8B55'/><text x='340' y='44' fill='#1a5c38'>v</text><line x1='196' y1='40' x2='324' y2='40' stroke='#2D8B55' stroke-width='2.5'/><text x='260' y='24' fill='#1a5c38'>positive (real edge) → label 1</text></g></svg>",
     "note":"<b>Observed positives.</b> Every real edge is a 'should connect' example. These are the only labels the graph hands you directly."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='180' cy='42' r='16' fill='#E4F2F7' stroke='#2A7B9B'/><text x='180' y='46' fill='#1F6280'>u</text><circle cx='340' cy='42' r='16' fill='#FDE8E8' stroke='#C93B3B'/><text x='340' y='46' fill='#C93B3B'>w</text><line x1='196' y1='42' x2='324' y2='42' stroke='#C93B3B' stroke-width='1.5' stroke-dasharray='5,4'/><text x='260' y='24' fill='#C93B3B'>random negative → label 0 (easy)</text></g></svg>",
     "note":"<b>Sample a random negative.</b> Pick a node u is not connected to. Cheap, but usually an obvious stranger — a weak training example."},
    {"viz":"<svg viewBox='0 0 520 92'><g font-family='monospace' font-size='10' text-anchor='middle'><circle cx='150' cy='55' r='15' fill='#E4F2F7' stroke='#2A7B9B'/><text x='150' y='59' fill='#1F6280'>u</text><circle cx='370' cy='55' r='15' fill='#FCF3DC' stroke='#C99A12'/><text x='370' y='59' fill='#9A7208'>w</text><circle cx='260' cy='22' r='11' fill='#EDE9F8' stroke='#7C6DAA'/><circle cx='260' cy='78' r='11' fill='#EDE9F8' stroke='#7C6DAA'/><line x1='163' y1='50' x2='250' y2='27'/><line x1='163' y1='60' x2='250' y2='74'/><line x1='357' y1='50' x2='270' y2='27'/><line x1='357' y1='60' x2='270' y2='74'/><line x1='165' y1='55' x2='355' y2='55' stroke='#C99A12' stroke-width='1.5' stroke-dasharray='5,4'/><text x='260' y='90' fill='#9A7208'>hard negative: 2 mutual friends, still no edge</text></g></svg>",
     "note":"<b>Mine a hard negative.</b> u and w share mutual friends yet are not connected. It looks like an edge but is not — the model must work to score it low. This is where the learning is."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='16' width='440' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>loss = −log σ(pos) − Σ log(1−σ(neg))</text></g></svg>",
     "note":"<b>Combine into one loss.</b> Push the positive's score up and every negative's score down. Hard negatives (scored wrongly high) dominate the loss and drive learning."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='20' y='18' width='118' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='79' y='35' fill='#1F6280'>friend graph</text><text x='150' y='35' fill='#6B645E'>→</text><rect x='168' y='18' width='108' height='26' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='222' y='35' fill='#5E5191'>GNN embeds</text><text x='288' y='35' fill='#6B645E'>→</text><rect x='306' y='18' width='170' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='391' y='35' fill='#1a5c38'>score + suggest top-k</text></g></svg>",
     "note":"<b>The full PYMK pipeline.</b> Build the graph → GNN node embeddings → score candidate edges → train on positives + sampled negatives → suggest the top-k friends. Every piece is a day of this module."},
  ],
  quiz=[
    {"q":"1. Why does link prediction need negative sampling?","opts":["to delete edges","because the graph only shows real edges (positives); you must invent 'no-edge' examples to train against","to make the graph bigger","to remove the sigmoid"],"ans":1,"fb":"Only positive edges are observed. Without sampled negatives, the model would just predict 'edge' for everything, learning nothing."},
    {"q":"2. What is a hard negative?","opts":["a pair from opposite sides of the graph","a non-edge that looks connectable (e.g. many mutual friends) but is not","any real edge","a node with no features"],"ans":1,"fb":"Hard negatives sit near the decision boundary — they look like they should connect but do not — so they carry the most learning signal."},
    {"q":"3. A sampled negative (u,w) is scored 0.8 by the model (it should be ~0). Its loss term −log(1−σ) ≈ −log(0.2) is:","opts":["about 0.1 (tiny)","about 1.6 (large) — a strong push to separate them","exactly 0","negative"],"ans":1,"fb":"−log(1−0.8) = −log(0.2) ≈ 1.609. A negative wrongly scored high creates a large loss, so training works hardest on exactly these cases."},
    {"q":"4. You train only with random negatives and the loss drops to near zero, but suggestions are bland. Most likely cause?","opts":["the learning rate is too low","easy negatives — random pairs are separated with no effort, so the model never learned the hard cases","the graph has no edges","the sigmoid is broken"],"ans":1,"fb":"Random negatives are usually obvious non-edges. The model separates them easily (low loss) but never learns the borderline cases that matter — mix in hard negatives."},
  ],
  fin={"em":"🎯","h3":"Day 4 complete — you can train a link predictor!",
       "p":"You now know that graphs give only positives, so you sample negatives — and that hard negatives (look connectable, are not) teach far more than random ones, via <code>loss = −log σ(pos) − Σ log(1−σ(neg))</code>. You saw the whole People-You-May-Know pipeline: graph → GNN embeddings → sigmoid(dot) scoring → positives + sampled negatives. Next: the <b>Review Gate</b>."},
))
print("M19 Days 1-4 built")

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m19-review", title="Module 19 · Review Gate", nav_title="Spiral · M19 Review",
  eyebrow="Module 19 · The Gate — You Understand Graph Systems", h1="Module 19 — Review Gate",
  lead="Checkpoint. The gate question: <b>can I take connected data, learn a node embedding, predict a link, and train it with sampled negatives — the way People-You-May-Know does?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. The chain — <code>graph + message passing</code> → <code>GNN embeddings</code> → <code>link scoring</code> → <code>negative sampling</code> — is exactly the friend-suggestion pipeline; it should feel solid before you move on.",
  prev_href="day-04-negative-sampling.html", prev_label="Day 4 · Negative Sampling for Graphs",
  checks=[
    ["Day 1","I can define a graph's nodes and edges, run one round of message passing (aggregate neighbours, combine with self), and name over-smoothing and the receptive-field trade-off."],
    ["Day 2","I can explain how stacking learnable message-passing layers makes a GNN, what a node embedding is, and why GraphSAGE samples neighbours to scale."],
    ["Day 3","I can score a candidate edge as <code>σ(z<sub>u</sub>·z<sub>v</sub>)</code>, explain the severe class imbalance, and say why candidate generation beats scoring all n² pairs."],
    ["Day 4","I can explain why we sample negatives, tell random from hard negatives, and describe how friend suggestion is framed as link prediction on positives + sampled negatives."],
  ],
  quiz=[
    {"q":"1. One round of message passing updates a node by:","opts":["deleting its edges","aggregating its neighbours' feature vectors and combining with its own","sorting all nodes","dividing by the number of layers"],"ans":1,"fb":"Each round gathers the neighbours' vectors, aggregates them, and mixes with the node's own vector (Day 1)."},
    {"q":"2. You stack 15 message-passing rounds and every node vector becomes nearly identical. This is:","opts":["a large, harmless receptive field","over-smoothing — repeated averaging blurred all nodes together","a broken sigmoid","perfect convergence"],"ans":1,"fb":"Over-smoothing: too many rounds average nodes into one blur, so the model can't tell them apart. 2–3 rounds is usually enough (Day 1)."},
    {"q":"3. Why does GraphSAGE sample a fixed number of neighbours per layer?","opts":["to shrink the graph forever","to bound cost per node so training fits in memory on huge graphs","to remove node features","to skip the non-linearity"],"ans":1,"fb":"Without sampling, the neighbours-of-neighbours per node explode; capping the fan-out keeps memory bounded at web scale (Day 2)."},
    {"q":"4. Embeddings z_u=[1,3] and z_v=[3,1] give a link score of σ(z_u·z_v). The dot product is:","opts":["4","6","3","0"],"ans":1,"fb":"z_u·z_v = 1·3 + 3·1 = 3 + 3 = 6. A large positive dot product → σ near 1 → likely edge (Day 3)."},
    {"q":"5. Training a link predictor with only random negatives gives near-zero loss but bland suggestions. The fix is:","opts":["remove all negatives","add hard negatives (non-edges that look connectable) so the model learns the borderline cases","use plain accuracy as the metric","stop using embeddings"],"ans":1,"fb":"Random negatives are separated with no effort, so the model never learns the hard cases. Hard negatives near the boundary supply the real learning signal (Day 4)."},
  ],
  verdict_pass="you can turn connected data into node embeddings, predict missing links from embedding similarity, and train the whole thing with sampled negatives — the exact recipe behind People-You-May-Know. Read the <a href=\"../../ML%20Design/11-people-you-may-know/README.md\">full case study</a> to see it deployed end-to-end.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The pipeline — <code>message passing</code> → <code>GNN</code> → <code>link scoring</code> → <code>negative sampling</code> — should feel like one connected story before you move on.",
  complete_label="Mark Module 19 complete",
  fin={"em":"🕸️","h3":"Module 19 — passed!",
       "p":"You built graph learning from scratch: message passing on connected data, GNNs that produce node embeddings, link prediction from embedding similarity, and negative sampling to train it — all coming together as the People-You-May-Know system. Connected data no longer looks like a mystery; it looks like a graph."},
  score_target=4,
))
print("M19 built: 4 lessons + review")
