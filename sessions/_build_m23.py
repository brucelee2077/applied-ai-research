#!/usr/bin/env python3
"""Builder for M23 — Image & Text Generation Systems.
Wraps the genAI design case studies (image captioning, RAG, faces, high-res,
text-to-image, headshots, image-to-video, temporal consistency, serving+safety).
Each lesson is self-contained (builds on M22 diffusion + M8 retrieval) and links
out to the named genAI design README with a "Go deeper" link.
9 lessons + review. Run: python3 sessions/_build_m23.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m23")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

GD = "../../genAI%20design"  # relative to sessions/week-m23/

# ---------------- Day 1 — Image Captioning ----------------
L("day-01-image-captioning.html", dict(
  qid="m23-d01-caption", title="Module 23 · Day 1 — Image Captioning", nav_title="Spiral · M23 Day 1",
  eyebrow="Module 23 · GenAI Systems · Day 1", h1="Image Captioning: Turning Pixels into Sentences",
  lead="Show a computer a photo of a dog catching a frisbee and ask it to write \"a brown dog jumping to catch a red frisbee.\" That is image captioning. It joins two worlds you already know — a vision encoder that reads pixels and a language decoder that writes words. Today you build the bridge between them and learn the one failure that haunts every caption model: making things up.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the encoder-decoder (prefix-LM) design, say how a CLIP-style image embedding conditions text generation, read CIDEr/BLEU scores, and name why captions hallucinate.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-rag.html", next_label="RAG End-to-End",
  sections=[
    {"title":"Two halves, one bridge","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a <span class="term" data-tip="A model that reads an image and turns it into a list of numbers (an embedding) capturing what is in it. You met this idea in M6 (vision encoders) and M8.">vision encoder</span> turns an image into numbers (M6), and a <span class="term" data-tip="A model that generates text one token at a time, each new token conditioned on the ones before. This is the transformer decoder from M5a.">language decoder</span> writes text one token at a time (M5a).</div></div>
     <h4>The task</h4>
     <p><span class="term" data-tip="Given an image, automatically produce a short sentence in natural language that describes what is in it.">Image captioning</span> takes a picture and writes a sentence about it. The input is pixels. The output is words. Nothing you have seen so far does both at once — so the trick is to <em>connect</em> two models you already understand.</p>
     <h4>The bridge</h4>
     <p>Step one: the vision encoder squeezes the image into a small list of numbers called an <span class="term" data-tip="A list of numbers that represents the meaning of something (here, an image) in a form a model can compute with.">image embedding</span>. Step two: that embedding is handed to the language decoder as its starting context, and the decoder writes words one at a time — each word looking back at the image and at the words already written.</p>
     <h4>Two common shapes</h4>
     <p>An <span class="term" data-tip="A model where one network (encoder) reads the input and a separate network (decoder) writes the output, with cross-attention letting the decoder look at the encoder's output.">encoder-decoder</span> keeps vision and language as two blocks joined by cross-attention. A <span class="term" data-tip="A single transformer that treats the image embedding as the first few 'tokens' (a prefix) and then generates the caption tokens after it — one stream, no separate cross-attention.">prefix-LM</span> is simpler: it glues the image embedding on the front as a prefix and lets a single transformer continue writing. Both do the same job: text conditioned on an image.</p>''',
     "gotit":"Got the two halves"},
    {"title":"A tour guide describing a photo","body":
     '''<div class="relate">
       <div class="card"><span class="big">🧑‍🏫</span><h5>The eyes read the scene</h5><p>A tour guide first <i>looks</i> at a painting and forms a mental impression — the colors, the people, the mood. That impression is the image embedding: not the raw painting, but a compressed sense of what matters.</p></div>
       <div class="card"><span class="big">🗣️</span><h5>The mouth tells the story</h5><p>Then the guide <i>speaks</i>, one word after another, glancing back at the painting to stay accurate. That is the decoder writing tokens, each one conditioned on the image and on what was already said.</p></div>
     </div>
     <p><strong>In one line:</strong> the encoder forms an impression of the image, and the decoder speaks that impression as a sentence, word by word.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real guide will say "I'm not sure what that object is." A caption model almost never hesitates — it always produces fluent, confident words, even when it is guessing. That over-confidence is exactly the danger you meet in section 4.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Watch a caption form","body":
     '''<p>Watch the image become an embedding, condition the decoder, and get scored. <strong>Click all three.</strong></p>'''},
    {"title":"How the image conditions the words","body":
     '''<h4>Step 1 — conditioning, in words</h4>
     <p>The decoder normally predicts the next word from the words so far. Here it also gets the image embedding, so its prediction is <em>conditioned</em> on the picture. We write the probability of the caption as a product of per-word probabilities, each one seeing the image.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       P(caption | image) = &prod;<sub>t</sub> P(word<sub>t</sub> | word<sub>&lt;t</sub>, <b>v</b>)<br>
       <span class="dim"># v = the image embedding from the vision encoder</span><br>
       <span class="dim"># word_&lt;t = all words written so far; the product runs over every word t</span>
     </div></div>
     <p>Symbols: <code>v</code> = the image embedding (a CLIP-style vector). <code>word<sub>t</sub></code> = the t-th word. The decoder multiplies the probability of each word given the image and the earlier words. Training maximizes this on human-written captions.</p>
     <h4>Step 3 — worked example (a caption score)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       caption "a dog runs", per-word P given the image:<br>
       P(a)=0.5, P(dog|a)=0.4, P(runs|a,dog)=0.5<br><br>
       P(caption|image) = 0.5 &times; 0.4 &times; 0.5 = <span class="hl">0.10</span><br>
       <span class="dim"># higher product = the model finds this caption more likely for this image</span>
     </div></div>
     <p>To grade a finished caption against human references we use overlap metrics. <span class="term" data-tip="A metric that counts how many word-groups (n-grams) in the generated caption also appear in human reference captions. Higher = more overlap.">BLEU</span> counts matching word-runs. <span class="term" data-tip="Consensus-based Image Description Evaluation: like BLEU but it up-weights words that are rare and informative and down-weights common words, and it compares against many human captions. The standard captioning metric.">CIDEr</span> improves on it by rewarding rare, informative words and averaging over many human captions — it is the standard captioning score.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — object hallucination.</b> The decoder is a fluent language model first and an image reader second. If "person" and "surfboard" often appear together in training, a beach photo with no surfboard can still make the model write "a person holding a surfboard." The sentence is grammatical and confident, so nothing looks wrong — but it describes an object that is not there. BLEU/CIDEr can even stay high while the caption is factually false, so you must measure hallucination separately (e.g. the CHAIR metric).</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — fluency vs faithfulness.</b> Push the decoder to sound natural (strong language prior, higher sampling temperature) and captions read beautifully but drift from the image — more hallucination. Anchor it hard to the image (heavier visual conditioning, low temperature) and captions stay faithful but sound stiff and repetitive. Every caption system picks a point on this line; a medical or accessibility caption leans hard toward faithfulness.</div></div>''',
     "gotit":"Got the conditioning"},
    {"title":"Building the captioner","body":
     '''<p>Here is the captioner assembled piece by piece: image → encoder → embedding → decoder → words → score. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a captioner","body":
     '''<p>Understanding is not building. Wire a tiny captioner. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_captioner.py</code>. Load a small pretrained vision encoder (e.g. CLIP image tower) and a small language decoder. Feed a few images, project the image embedding into the decoder as a prefix, generate a caption, and print the per-word probabilities and the final caption for each image.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 1 artifact.
Create experiments/foundations/m23_captioner.py: a prefix-LM image captioner. Use a small pretrained CLIP-style image encoder to get an image embedding, project it to the decoder's hidden size, prepend it as a prefix to a small language decoder, and greedily generate a caption. Run on 3 sample images, print the caption and per-token probabilities, and add a simple object-hallucination check (does the caption mention an object not in a given ground-truth object list?).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-01-log.md</code>: did any caption mention an object that was not in the image, and did higher temperature make it worse?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/05-image-captioning/README.md">genAI design · 05 Image Captioning</a> — the full system design (data, training, serving, eval).</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "encode":{"btn":"① image → embedding","html":'''<span class="prompt">&gt;&gt;&gt;</span> img = load("dog_frisbee.jpg")
<span class="prompt">&gt;&gt;&gt;</span> v = vision_encoder(img)   <span class="dim"># CLIP-style tower</span>
<span class="hl">v.shape = (512,)</span>   <span class="dim"># the image squeezed into 512 numbers</span>''',
      "take":"<b>①  The encoder makes an impression.</b> The picture becomes a 512-number embedding <code>v</code> — a compressed sense of what is in it, not the raw pixels."},
    "decode":{"btn":"② embedding conditions words","html":'''<span class="prompt">&gt;&gt;&gt;</span> tokens = decoder(prefix=v)   <span class="dim"># v prepended as a prefix</span>
<span class="hl">"a"</span> → <span class="hl">"brown"</span> → <span class="hl">"dog"</span> → <span class="hl">"catching"</span> → …
<span class="dim"># each word sees v AND the words before it</span>''',
      "take":"<b>②  The decoder speaks it.</b> With the image embedding as its prefix, the decoder writes one word at a time — every word conditioned on the picture."},
    "score":{"btn":"③ score with CIDEr/BLEU","html":'''<span class="prompt">&gt;&gt;&gt;</span> refs = ["a brown dog catches a frisbee", ...]
<span class="prompt">&gt;&gt;&gt;</span> cider(pred, refs), bleu(pred, refs)
<span class="hl">CIDEr = 1.12   BLEU = 0.34</span>
<span class="dim"># overlap with human captions — but blind to hallucination</span>''',
      "take":"<b>③  Grade against humans.</b> CIDEr and BLEU measure word overlap with human captions. They can stay high even if the caption names an object that is not there."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='190' y='16' width='140' height='32' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='37' fill='#5A544E'>🖼️ input image</text></g></svg>",
     "note":"<b>Start with the image.</b> Raw pixels — a photo the model has never described before."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='24' width='110' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='85' y='43' fill='#5A544E'>image</text><text x='150' y='43' fill='#6B645E'>→</text><polygon points='175,24 300,37 300,41 175,54' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='240' y='42' fill='#1F6280'>vision encoder</text><text x='315' y='43' fill='#6B645E'>→</text><rect x='340' y='24' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='415' y='43' fill='#1F6280'>embedding v (512)</text></g></svg>",
     "note":"<b>Encode to an embedding.</b> The vision encoder compresses the image into a 512-number vector <code>v</code> — the impression the decoder will speak."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='120' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='100' y='43' fill='#1F6280'>v (prefix)</text><text x='172' y='43' fill='#6B645E'>→</text><polygon points='195,20 340,37 340,41 195,58' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='268' y='42' fill='#5E5191'>language decoder</text><text x='356' y='43' fill='#6B645E'>→</text><rect x='380' y='24' width='110' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='435' y='43' fill='#9A7208'>word₁</text></g></svg>",
     "note":"<b>Condition the decoder.</b> The embedding is prepended as a prefix (prefix-LM). The decoder reads it and predicts the first word."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='18' width='70' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='65' y='35' fill='#9A7208'>a</text><rect x='115' y='18' width='80' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='155' y='35' fill='#9A7208'>brown</text><rect x='210' y='18' width='70' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='245' y='35' fill='#9A7208'>dog</text><rect x='295' y='18' width='95' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='342' y='35' fill='#9A7208'>catching</text><text x='420' y='35' fill='#6B645E'>… →</text></g></svg>",
     "note":"<b>Write word by word.</b> Each new word is conditioned on the image and the words already written. This is P(caption|image) = ∏ P(wordₜ|word₍&lt;t₎, v)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>CIDEr / BLEU  vs  human captions</text></g></svg>",
     "note":"<b>Score it.</b> Compare against human references with CIDEr and BLEU — but remember these measure overlap, not truth, so hallucination hides here."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='16' width='180' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='150' y='36' fill='#C93B3B'>caption says 'surfboard'</text><rect x='290' y='16' width='170' height='32' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='375' y='36' fill='#5A544E'>no surfboard in image</text></g></svg>",
     "note":"<b>The lurking failure.</b> A fluent caption can name an object that is not present (hallucination). Measure it separately — overlap metrics will not catch it."},
  ],
  quiz=[
    {"q":"1. In a prefix-LM captioner, how does the image reach the decoder?","opts":["it is drawn on the screen","its embedding is prepended as a prefix so the decoder generates words conditioned on it","the decoder ignores the image","it is turned into a reward"],"ans":1,"fb":"The vision encoder makes an image embedding; a prefix-LM prepends it so every generated word is conditioned on the image and the words so far."},
    {"q":"2. A caption confidently mentions a \"surfboard\" that is not in the photo. This is:","opts":["saturation","object hallucination — a fluent but false description","a BLEU bug","correct behaviour"],"ans":1,"fb":"The language prior invents plausible co-occurring objects. The sentence is fluent and confident but factually wrong — object hallucination."},
    {"q":"3. Per-word probabilities are P(a)=0.5, P(dog|a)=0.5, P(sat|a,dog)=0.4. P(caption|image) is:","opts":["1.4","0.10","0.40","0.25"],"ans":1,"fb":"Multiply: 0.5 × 0.5 × 0.4 = 0.10. The caption's probability is the product of its per-word probabilities."},
    {"q":"4. You raise the decoder's sampling temperature and captions read more naturally but describe wrong objects more often. This is the:","opts":["contamination trade-off","fluency vs faithfulness trade-off","latency vs cost trade-off","bias vs variance trade-off"],"ans":1,"fb":"More fluent, less anchored to the image = more hallucination. That is the fluency vs faithfulness trade-off every caption system must balance."},
  ],
  fin={"em":"🖼️","h3":"Day 1 complete — pixels became sentences!",
       "p":"You built the bridge: a vision encoder makes an image embedding, a prefix-LM or encoder-decoder conditions text on it, and CIDEr/BLEU grade the words against humans. You also met the caption killer — object hallucination — and the fluency vs faithfulness dial. Next: <b>Day 2</b> — RAG, grounding a text generator in retrieved facts."},
))
# ---------------- Day 2 — RAG End-to-End ----------------
L("day-02-rag.html", dict(
  qid="m23-d02-rag", title="Module 23 · Day 2 — RAG End-to-End", nav_title="Spiral · M23 Day 2",
  eyebrow="Module 23 · GenAI Systems · Day 2", h1="RAG: Answers Grounded in Real Documents",
  lead="A chatbot that answers from memory will confidently invent facts. RAG fixes this by looking things up first. Before answering, it retrieves the most relevant chunks of your documents and pastes them into the prompt — so the model answers from real text, not vague memory. Today you build the full loop: embed, retrieve, stuff, generate.",
  goal="<b>🎯 By the end, you'll be able to:</b> trace the RAG loop (embed → retrieve → stuff → generate), explain chunking and re-ranking, compute a cosine similarity for retrieval, and name the failure where a retrieval miss produces a confident wrong answer.",
  prev_href="day-01-image-captioning.html", prev_label="Image Captioning", next_href="day-03-face-generation.html", next_label="Realistic Face Generation",
  sections=[
    {"title":"Look it up, then answer","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> an <span class="term" data-tip="A list of numbers representing the meaning of a piece of text, so that similar meanings sit close together. You met embeddings in M8 (retrieval).">embedding</span> turns text into a vector where similar meanings sit close together (M8), and a language model generates text from a prompt (M5a).</div></div>
     <h4>The problem</h4>
     <p>A language model answers from what it memorized during training. That memory is fuzzy, out of date, and cannot include your private documents. Ask about your company's 2024 refund policy and it will guess — fluently and wrongly.</p>
     <h4>The fix: retrieve first</h4>
     <p><span class="term" data-tip="Retrieval-Augmented Generation: before answering, the system retrieves relevant document chunks and puts them in the prompt, so the model answers from real text instead of memory.">RAG</span> (Retrieval-Augmented Generation) adds a lookup step before generation. It searches your documents for the passages most relevant to the question, pastes those passages into the prompt, and asks the model to answer <em>using them</em>. The model reads the facts instead of recalling them.</p>
     <h4>The four steps</h4>
     <ul>
       <li><b>Embed</b> — turn the question (and every document chunk, done in advance) into a vector.</li>
       <li><b>Retrieve</b> — find the chunks whose vectors are closest to the question's vector.</li>
       <li><b>Stuff</b> — paste the top chunks into the prompt as context.</li>
       <li><b>Generate</b> — the model answers, grounded in that pasted context.</li>
     </ul>''',
     "gotit":"Got the RAG loop"},
    {"title":"An open-book exam","body":
     '''<div class="relate">
       <div class="card"><span class="big">📖</span><h5>Closed book = guessing</h5><p>A plain chatbot is a student taking an exam from memory. On familiar topics it does fine; on details it half-remembers, it makes things up with total confidence.</p></div>
       <div class="card"><span class="big">🔎</span><h5>Open book = look it up</h5><p>RAG lets the student open the textbook, flip to the right page, and read the answer off it. The skill shifts from "remember everything" to "find the right page fast." That page-finding is retrieval.</p></div>
     </div>
     <p><strong>In one line:</strong> RAG turns a closed-book guesser into an open-book reader by fetching the right pages before it writes.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real student notices when the book does not cover the question and says "it's not in here." A RAG model usually does not — if retrieval fetches the wrong page, it will still answer from that wrong page, confidently. Catching that is the whole game (section 4).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run one retrieval","body":
     '''<p>Watch a question get embedded, matched against chunks by cosine similarity, and the top chunk stuffed into the prompt. <strong>Click all three.</strong></p>'''},
    {"title":"How retrieval finds the right chunk","body":
     '''<h4>Step 1 — similarity, in words</h4>
     <p>Every chunk and the question become vectors. "Closeness in meaning" becomes "closeness of vectors." The standard measure is <span class="term" data-tip="A number from -1 to 1 measuring the angle between two vectors: 1 = same direction (same meaning), 0 = unrelated. It ignores length and looks only at direction.">cosine similarity</span> — the cosine of the angle between two vectors. We retrieve the chunks with the highest cosine similarity to the question.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       cos(q, c) = (q · c) / (‖q‖ · ‖c‖)<br>
       <span class="dim"># q = question vector, c = chunk vector</span><br>
       <span class="dim"># q · c = dot product; ‖q‖ = length of q. Result in [-1, 1], higher = more relevant</span>
     </div></div>
     <p>Symbols: <code>q</code> = the question embedding, <code>c</code> = a chunk embedding, <code>q · c</code> = the dot product (multiply matching entries and sum), <code>‖q‖</code> = the length of <code>q</code>. Divide by the lengths so only the <em>direction</em> (meaning) matters, not the size.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       q = [3, 4],  chunk_A = [3, 4],  chunk_B = [4, -3]<br><br>
       ‖q‖ = √(9+16) = 5<br>
       cos(q, A) = (3·3 + 4·4) / (5·5) = 25/25 = <span class="hl">1.0</span>  ← perfect match<br>
       cos(q, B) = (3·4 + 4·(−3)) / (5·5) = 0/25 = <span class="hl">0.0</span>  ← unrelated<br>
       <span class="dim"># retrieve chunk_A, ignore chunk_B</span>
     </div></div>
     <p>Two design choices shape quality. <span class="term" data-tip="Splitting long documents into smaller passages (chunks) before embedding, so each chunk is a focused, retrievable unit. Chunk size trades recall against precision.">Chunking</span> decides how you cut documents into passages — too big and the relevant fact is buried in noise, too small and it loses context. <span class="term" data-tip="A second, more accurate scoring pass that reorders the top retrieved chunks (often with a cross-encoder that reads the question and chunk together) before they go into the prompt.">Re-ranking</span> takes the top ~50 fast matches and re-scores them with a slower, more accurate model, keeping only the best few for the prompt.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — retrieval miss → confident wrong answer.</b> If retrieval fetches the wrong chunks (bad chunking, a rare phrasing, an out-of-vocabulary term), the model answers fluently <b>from the wrong context</b> and sounds just as sure as when it is right. There is no error and no "I don't know" — the generator trusts whatever you stuffed in. This is why RAG systems measure <i>retrieval recall</i> separately and add "answer only from the context, else say you don't know" instructions and citation checks.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — context size vs cost/latency.</b> Stuffing more chunks raises the chance the right fact is present (higher recall) but every extra token costs money and adds latency, and too much context can bury the answer (the "lost in the middle" effect, where models ignore the middle of a long prompt). Fewer chunks are cheap and fast but risk missing the fact. Re-ranking lets you keep the prompt small <i>and</i> relevant — for the price of a second scoring pass.</div></div>''',
     "gotit":"Got retrieval"},
    {"title":"Building the RAG pipeline","body":
     '''<p>Here is RAG assembled: chunk the docs, embed them, embed the query, retrieve + re-rank, stuff, generate. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a mini-RAG","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_rag.py</code>. Take ~20 short documents, chunk them, embed the chunks with a small embedding model, embed a query, retrieve the top-k by cosine similarity, stuff them into a prompt, and generate an answer. Then delete the one relevant chunk and show the answer becomes a confident hallucination.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 2 artifact.
Create experiments/foundations/m23_rag.py: a minimal RAG pipeline. Chunk ~20 short docs, embed chunks and a query with a small sentence-embedding model, retrieve top-k by cosine similarity, optionally re-rank, stuff into a prompt, and generate. Print the retrieved chunks with their cosine scores. Then run an ablation: remove the one relevant chunk and show the answer becomes a confident wrong answer (a retrieval-miss hallucination).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-02-log.md</code>: what cosine score did the correct chunk get, and what did the model answer when you removed it?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/06-rag/README.md">genAI design · 06 RAG</a> — the full system design (indexing, retrieval, re-ranking, serving, eval).</div></div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "embed":{"btn":"① embed the question","html":'''<span class="prompt">&gt;&gt;&gt;</span> q = embed("what is our refund window?")
<span class="hl">q = [0.12, -0.04, 0.31, ...]</span>   <span class="dim"># a 384-number vector</span>
<span class="dim"># every doc chunk was embedded ahead of time too</span>''',
      "take":"<b>①  Turn the question into a vector.</b> Meaning becomes numbers, so similar meanings land close together in vector space."},
    "retrieve":{"btn":"② retrieve by cosine","html":'''<span class="prompt">&gt;&gt;&gt;</span> scores = cosine(q, all_chunks)
<span class="hl">chunk_7: 0.88</span>   <span class="dim"># "Refunds accepted within 30 days…"</span>
chunk_3: 0.41   chunk_9: 0.12
<span class="dim"># keep the top-k highest scores</span>''',
      "take":"<b>②  Find the closest chunks.</b> Cosine similarity ranks chunks by meaning. Chunk 7 (0.88) is clearly the relevant one; keep the top few."},
    "generate":{"btn":"③ stuff + generate","html":'''<span class="prompt">&gt;&gt;&gt;</span> prompt = f"Context: {chunk_7}\\nQ: refund window? A:"
<span class="prompt">&gt;&gt;&gt;</span> model.generate(prompt)
<span class="hl">"You can request a refund within 30 days."</span>
<span class="dim"># grounded in the retrieved text, not memory</span>''',
      "take":"<b>③  Answer from the context.</b> The retrieved chunk is pasted into the prompt, so the model answers from real text — grounded, not guessed."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='120' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='100' y='37' fill='#5A544E'>long document</text><text x='175' y='37' fill='#6B645E'>→</text><rect x='200' y='18' width='55' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='227' y='37' fill='#5E5191'>c1</text><rect x='262' y='18' width='55' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='289' y='37' fill='#5E5191'>c2</text><rect x='324' y='18' width='55' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='351' y='37' fill='#5E5191'>c3</text><text x='405' y='37' fill='#6B645E'>…</text></g></svg>",
     "note":"<b>Chunk the documents.</b> Cut long docs into small, focused passages. Chunk size trades recall (big chunks) against precision (small chunks)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='120' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='120' y='37' fill='#5E5191'>chunks</text><text x='195' y='37' fill='#6B645E'>→ embed →</text><rect x='300' y='18' width='160' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='380' y='37' fill='#1F6280'>vector index</text></g></svg>",
     "note":"<b>Embed and index.</b> Every chunk becomes a vector, stored ahead of time in a vector index for fast search."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='18' width='140' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='160' y='37' fill='#9A7208'>❓ query</text><text x='245' y='37' fill='#6B645E'>→ embed →</text><rect x='350' y='18' width='120' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='410' y='37' fill='#9A7208'>query vector</text></g></svg>",
     "note":"<b>Embed the query.</b> The question becomes a vector in the same space, ready to compare against the chunk vectors."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='18' width='150' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='38' fill='#1F6280'>cosine → top-50</text><text x='195' y='38' fill='#6B645E'>→</text><rect x='220' y='18' width='150' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='295' y='38' fill='#1a5c38'>re-rank → top-3</text><text x='385' y='38' fill='#6B645E'>→ 🏆</text></g></svg>",
     "note":"<b>Retrieve, then re-rank.</b> A fast cosine search grabs the top ~50; a slower, more accurate re-ranker keeps the best 3. Small, relevant context."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='150' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='135' y='38' fill='#1a5c38'>context + question</text><text x='225' y='38' fill='#6B645E'>→</text><polygon points='250,15 380,32 380,36 250,53' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='320' y='37' fill='#5E5191'>LLM</text><text x='395' y='38' fill='#6B645E'>→ answer</text></g></svg>",
     "note":"<b>Stuff and generate.</b> Paste the top chunks into the prompt and let the model answer from them — grounded in real text, not memory."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='200' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='140' y='38' fill='#C93B3B'>wrong chunk retrieved</text><text x='255' y='38' fill='#6B645E'>→</text><rect x='285' y='18' width='195' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='382' y='38' fill='#C93B3B'>confident wrong answer</text></g></svg>",
     "note":"<b>The failure to watch.</b> A retrieval miss feeds the wrong context, and the model answers from it just as confidently. Measure retrieval recall separately."},
  ],
  quiz=[
    {"q":"1. What does the \"retrieval\" step in RAG do?","opts":["it trains the language model","it finds the document chunks most relevant to the question and adds them to the prompt","it deletes old answers","it scores fluency"],"ans":1,"fb":"Retrieval searches your documents for the chunks closest in meaning to the question and stuffs them into the prompt so the model answers from real text."},
    {"q":"2. q=[3,4], chunk=[6,8]. Their cosine similarity is:","opts":["0.0","0.5","1.0","50"],"ans":2,"fb":"cos = (3·6+4·8)/(5·10) = 50/50 = 1.0. The vectors point the same direction (same meaning), so cosine is 1 regardless of length."},
    {"q":"3. Retrieval fetches the wrong chunk. What does the generator typically do?","opts":["throw an error","refuse to answer","answer confidently from the wrong context (a retrieval-miss hallucination)","re-run retrieval automatically"],"ans":2,"fb":"The generator trusts whatever context it is given. A retrieval miss yields a fluent, confident, wrong answer with no warning — hence measuring retrieval recall separately."},
    {"q":"4. You add many more chunks to the prompt. The main downside is:","opts":["retrieval becomes impossible","higher cost/latency and the answer can get buried in a long context","the embeddings stop working","cosine similarity turns negative"],"ans":1,"fb":"More context raises recall but costs more tokens, adds latency, and can bury the answer (lost-in-the-middle). That is the context-size vs cost/latency trade-off."},
  ],
  fin={"em":"🔎","h3":"Day 2 complete — grounded answers!",
       "p":"You built RAG end-to-end: embed the query and chunks, retrieve the closest by cosine similarity, re-rank, stuff into the prompt, and generate a grounded answer. You saw why a retrieval miss becomes a confident wrong answer, and the context-size vs cost/latency dial. Next: <b>Day 3</b> — generating photoreal faces with diffusion, and the safety that must ride with it."},
))

# ---------------- Day 3 — Realistic Face Generation ----------------
L("day-03-face-generation.html", dict(
  qid="m23-d03-face", title="Module 23 · Day 3 — Realistic Face Generation", nav_title="Spiral · M23 Day 3",
  eyebrow="Module 23 · GenAI Systems · Day 3", h1="Realistic Faces: Power and Responsibility",
  lead="A diffusion model can now paint a human face so real you cannot tell it apart from a photo — of a person who does not exist. That is astonishing and dangerous in the same breath. Today you connect M22's diffusion to the face problem, learn the identity-vs-diversity tension, and face head-on the safety issues no serious system can skip.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how diffusion generates a face, describe the identity-vs-diversity trade-off, name key safety concerns (consent, deepfakes), and describe the training-face memorization (privacy) failure.",
  prev_href="day-02-rag.html", prev_label="RAG End-to-End", next_href="day-04-high-res.html", next_label="High-Res Image Synthesis",
  sections=[
    {"title":"Painting a face from noise","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> <span class="term" data-tip="A generative model that starts from pure random noise and removes the noise step by step, guided by a trained network, until a clean image appears. This is M22b.">diffusion</span> generates images by denoising step by step (M22b). Today: aim that machinery at human faces.</div></div>
     <h4>The task</h4>
     <p>Generate a photorealistic human face that looks completely real but belongs to no real person. The engine is the diffusion model you built in M22b: start from random noise, and a trained network removes a little noise at each step until a sharp face emerges.</p>
     <h4>Why faces are hard</h4>
     <p>Human eyes are expert face detectors. We instantly notice a crooked eye, mismatched earrings, or teeth that blur into a smear. A landscape can be slightly wrong and still look fine; a face cannot. So face models are pushed to extreme detail and consistency, which is exactly what makes them powerful — and risky.</p>
     <h4>Two things we want at once</h4>
     <p>We want <span class="term" data-tip="Whether the generated face is a single coherent person — the two eyes match, the lighting is consistent, the features belong together.">identity</span> (a single coherent person, not a Picasso) and <span class="term" data-tip="Whether the model can produce many different faces across ages, ethnicities, and looks, rather than repeating a narrow set of similar faces.">diversity</span> (many different-looking people, not the same face over and over). These two pull against each other, as you will see.</p>''',
     "gotit":"Got the task"},
    {"title":"A sculptor and a police artist","body":
     '''<div class="relate">
       <div class="card"><span class="big">🗿</span><h5>Carving away noise</h5><p>A sculptor starts with a rough block of marble and chips away until a face appears. Diffusion is the same: it starts with a block of random noise and removes a little each step until a face is revealed.</p></div>
       <div class="card"><span class="big">👮</span><h5>The likeness question</h5><p>A police sketch artist can draw a convincing face — but if it looks too much like a real bystander, that is a serious problem. A face generator that reproduces real training faces raises the same alarm: it can leak a real person's likeness.</p></div>
     </div>
     <p><strong>In one line:</strong> diffusion sculpts a face out of noise, but the more lifelike and specific it gets, the harder we must guard against it copying a real person.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a sculptor works from imagination. A diffusion model works from a training set of real faces, so "imagination" can quietly become "recall of a specific person it saw" — the memorization risk in section 4.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Denoise a face, step by step","body":
     '''<p>Watch pure noise, a few denoising steps, and the identity-vs-diversity dial. <strong>Click all three.</strong></p>'''},
    {"title":"Denoising, diversity, and the privacy risk","body":
     '''<h4>Step 1 — the denoising step, in words</h4>
     <p>Diffusion runs many steps. At each step the network predicts the noise currently in the image and subtracts a portion of it, so the image gets a little cleaner. Repeat and a face forms. We write one step as: new image = current image minus a slice of the predicted noise.</p>
     <h4>Step 2 — the formula (one step)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       x<sub>t-1</sub> = x<sub>t</sub> − α · ε<sub>θ</sub>(x<sub>t</sub>, t)<br>
       <span class="dim"># x_t = image at step t (t counts DOWN from noisy to clean)</span><br>
       <span class="dim"># ε_θ = the network's predicted noise; α = how big a slice we remove</span>
     </div></div>
     <p>Symbols: <code>x<sub>t</sub></code> = the image at step t, <code>ε<sub>θ</sub></code> = the noise the network predicts is present, <code>α</code> (alpha) = the step size. Subtracting predicted noise nudges the image toward a clean face. (This is the M22b sampler, aimed at faces.)</p>
     <h4>Step 3 — worked example (a tiny step)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       one pixel: x_t = 0.90 (mostly noise),  predicted noise ε = 0.40,  α = 0.5<br><br>
       x_(t-1) = 0.90 − 0.5·0.40 = 0.90 − 0.20 = <span class="hl">0.70</span><br>
       <span class="dim"># the pixel moved toward its true value; many steps → a clean face</span>
     </div></div>
     <p>To get diversity vs identity, we control how strongly generation is steered (guidance, from M22b). Steer hard and every face converges toward one "average attractive" look — high identity, low diversity. Steer softly and faces vary more but can lose coherence.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — training-face memorization (privacy leak).</b> If a real person's face appears many times in the training data (a celebrity, a duplicated photo), the model can <b>memorize and regenerate that exact face</b>. It looks like a fresh, invented person, but it is a real individual's likeness reproduced without consent — a privacy and legal violation that is invisible unless you actively test for it (e.g. nearest-neighbor search against the training set). De-duplicating training data and adding noise/regularization reduce it.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — realism vs safety controls.</b> Maximum realism (huge models, minimal filtering, no watermark) makes the best faces — and the best deepfakes. Every safety control (consent-checked training data, refusing real-person prompts, invisible watermarks, output filters) costs some realism, coverage, or speed. A responsible face system deliberately gives up some capability for safety; an irresponsible one does not, and that is exactly how harmful deepfakes get shipped.</div></div>''',
     "gotit":"Got denoising & the risks"},
    {"title":"Building the face generator","body":
     '''<p>Here is the face pipeline assembled: noise → denoise loop → face → the identity/diversity dial → the safety layer. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it (safely) with diffusion faces","body":
     '''<p>Pick one. Use a public, consented dataset (e.g. FFHQ) and never real people's private photos.</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_faces.py</code>. Load a small pretrained face diffusion model. Sample faces at several guidance strengths, and for each generated face run a nearest-neighbor search against a sample of the training set to check for memorization. Print the guidance value, a diversity measure, and the nearest-neighbor distance.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 3 artifact.
Create experiments/foundations/m23_faces.py using a small pretrained face diffusion model on a public consented dataset (e.g. FFHQ). Sample faces at guidance scales {1, 3, 7}, measure diversity (e.g. average pairwise embedding distance) at each scale, and for each face run a nearest-neighbor search against a sample of training faces to detect memorization. Print guidance, diversity, and min nearest-neighbor distance. Do NOT use any private/real-person photos.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-03-log.md</code>: how did diversity change with guidance, and did any sample come suspiciously close to a training face?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/07-realistic-face-generation/README.md">genAI design · 07 Realistic Face Generation</a> — the full system design, including safety and evaluation.</div></div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "noise":{"btn":"① start from noise","html":'''<span class="prompt">&gt;&gt;&gt;</span> x = randn(3, 512, 512)   <span class="dim"># pure random noise</span>
<span class="hl">x = ▓▒░▓▒░▓▒░</span>   <span class="dim"># no face yet — just static</span>''',
      "take":"<b>①  Begin with static.</b> Diffusion starts from a block of random noise — the marble before the sculptor touches it."},
    "denoise":{"btn":"② denoise step by step","html":'''<span class="prompt">&gt;&gt;&gt;</span> for t in reversed(range(T)):
<span class="prompt">...</span>     x = x - alpha * eps_theta(x, t)
<span class="hl">step 40/50: a blurry face</span>
<span class="hl">step 50/50: a sharp, realistic face</span>''',
      "take":"<b>②  Chip away the noise.</b> Each step subtracts predicted noise, so the face slowly appears — like carving marble down to a portrait."},
    "guide":{"btn":"③ identity vs diversity","html":'''<span class="prompt">&gt;&gt;&gt;</span> sample(guidance=7)   <span class="dim"># steer hard</span>
<span class="hl">→ crisp identity, but faces look samey</span>
<span class="prompt">&gt;&gt;&gt;</span> sample(guidance=1)   <span class="dim"># steer soft</span>
<span class="hl">→ diverse faces, but some incoherent</span>''',
      "take":"<b>③  The dial.</b> Strong guidance = coherent but repetitive faces (high identity, low diversity); weak guidance = varied but sometimes broken faces."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='190' y='16' width='140' height='32' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='37' fill='#5A544E'>random noise xₜ</text></g></svg>",
     "note":"<b>Start with noise.</b> A block of random static — the same starting point as any M22b diffusion model."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='130' y='16' width='260' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>xₜ₋₁ = xₜ − α·εθ(xₜ, t)</text></g></svg>",
     "note":"<b>One denoising step.</b> The network predicts the noise and we subtract a slice of it, nudging the image toward a clean face."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='22' width='60' height='26' rx='4' fill='#F5F0E8' stroke='#6B645E'/><text x='60' y='39' fill='#5A544E'>noise</text><text x='100' y='39' fill='#6B645E'>→</text><rect x='120' y='22' width='60' height='26' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='150' y='39' fill='#5E5191'>blur</text><text x='190' y='39' fill='#6B645E'>→</text><rect x='210' y='22' width='70' height='26' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='245' y='39' fill='#1F6280'>shapes</text><text x='290' y='39' fill='#6B645E'>→</text><rect x='310' y='22' width='80' height='26' rx='4' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='350' y='39' fill='#1a5c38'>🙂 face</text></g></svg>",
     "note":"<b>Repeat many steps.</b> Over ~50 steps the noise becomes a blur, then shapes, then a sharp, realistic face."},
    {"viz":"<svg viewBox='0 0 520 78'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='95' y='28' fill='#5E5191'>soft guidance</text><text x='95' y='44' fill='#6B645E'>diverse, less coherent</text><line x1='170' y1='38' x2='360' y2='38' stroke='#6B645E' stroke-width='2'/><circle cx='330' cy='38' r='5' fill='#2D8B55'/><text x='430' y='28' fill='#5E5191'>hard guidance</text><text x='430' y='44' fill='#6B645E'>coherent, samey</text></g></svg>",
     "note":"<b>Identity vs diversity dial.</b> Guidance strength trades coherent-but-repetitive faces (right) against varied-but-fragile faces (left)."},
    {"viz":"<svg viewBox='0 0 520 64'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='18' width='140' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='100' y='38' fill='#1a5c38'>consent-checked data</text><rect x='185' y='18' width='120' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='245' y='38' fill='#1F6280'>watermark</text><rect x='320' y='18' width='170' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='405' y='38' fill='#9A7208'>NN memorization check</text></g></svg>",
     "note":"<b>The safety layer.</b> Consent-checked data, output watermarking, and nearest-neighbor memorization checks are not optional — they are part of the system."},
  ],
  quiz=[
    {"q":"1. How does a diffusion model create a face?","opts":["it copies a photo directly","it starts from random noise and removes predicted noise step by step until a face emerges","it retrieves a face from a database","it averages two input faces"],"ans":1,"fb":"Diffusion (M22b) denoises: from pure noise, the network subtracts predicted noise each step until a sharp face appears — like carving marble."},
    {"q":"2. A model outputs faces that all look like the same 'average attractive' person. This is:","opts":["too much diversity","low diversity from too-strong guidance","a retrieval miss","correct behaviour"],"ans":1,"fb":"Strong guidance steers every sample toward the same coherent mode — high identity, low diversity. Softening guidance restores variety."},
    {"q":"3. One denoising step: xₜ=0.8, predicted noise ε=0.6, α=0.5. The new pixel value is:","opts":["0.5","0.2","1.1","0.6"],"ans":0,"fb":"x = 0.8 − 0.5·0.6 = 0.8 − 0.3 = 0.5. We subtract a slice (α) of the predicted noise to move toward the clean value."},
    {"q":"4. A generated face turns out to be nearly identical to a real person in the training set. This is:","opts":["good generalization","training-face memorization — a privacy/consent leak","expected and safe","a watermark"],"ans":1,"fb":"The model memorized and reproduced a real individual's likeness without consent. De-duplicate training data and run nearest-neighbor checks to catch it."},
  ],
  fin={"em":"🗿","h3":"Day 3 complete — faces, with responsibility!",
       "p":"You aimed M22b diffusion at faces: denoise from noise, and use guidance as the identity-vs-diversity dial. You met the privacy failure — training-face memorization — and the realism-vs-safety trade-off that no serious system can skip. Next: <b>Day 4</b> — pushing images to megapixel resolution with latent diffusion and cascades."},
))
# ---------------- Day 4 — High-Res Image Synthesis ----------------
L("day-04-high-res.html", dict(
  qid="m23-d04-highres", title="Module 23 · Day 4 — High-Res Image Synthesis", nav_title="Spiral · M23 Day 4",
  eyebrow="Module 23 · GenAI Systems · Day 4", h1="High-Res Images: Diffusion Without the Bankruptcy",
  lead="A 1024×1024 image is a million pixels. Running diffusion directly on a million pixels, for 50 steps, is brutally expensive. The trick that made Stable Diffusion possible: don't denoise pixels — denoise a tiny compressed version, then decode. Today you learn latent diffusion and the upscaling cascades that reach megapixel images without melting your GPUs.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why latent space saves compute, describe upscaling cascades, estimate the compute saving from a compression factor, and name the fine-detail artifact failure.",
  prev_href="day-03-face-generation.html", prev_label="Realistic Face Generation", next_href="day-05-text-to-image.html", next_label="Text-to-Image",
  sections=[
    {"title":"Why full-resolution diffusion hurts","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> diffusion denoises step by step (M22b), and an <span class="term" data-tip="An encoder-decoder that compresses an image into a small latent code and reconstructs it. You built this in M22a — the VAE.">autoencoder</span> can compress and rebuild an image (M22a VAE).</div></div>
     <h4>The cost problem</h4>
     <p>Diffusion runs the denoising network dozens of times. Each run touches every pixel. A 1024×1024 image has ~1,000,000 pixels; doing 50 steps over a million pixels, for every image, is enormously expensive in compute and memory.</p>
     <h4>The idea: work small</h4>
     <p><span class="term" data-tip="Diffusion run in a compressed latent space instead of on raw pixels. An autoencoder shrinks the image, diffusion denoises the small code, then the decoder rebuilds the full image. This is what Stable Diffusion does.">Latent diffusion</span> does the whole denoising loop on a compressed version of the image — a small <span class="term" data-tip="A compact set of numbers that encodes an image. Much smaller than the pixel grid but keeps the important structure so a decoder can rebuild the image.">latent code</span> — instead of the raw pixels. An autoencoder shrinks the image (say 8× smaller on each side), diffusion cleans up the tiny code, and a decoder blows it back up to full resolution.</p>
     <h4>Cascades for the biggest images</h4>
     <p>For very large images, systems also <span class="term" data-tip="Generating a small image first, then passing it through one or more upscaling diffusion models that add detail and resolution in stages, rather than generating the full size at once.">cascade</span>: generate a small image, then run a second (and third) diffusion model that upscales it and adds fine detail. Each stage handles a manageable size.</p>''',
     "gotit":"Got latent diffusion"},
    {"title":"Editing a thumbnail, not the poster","body":
     '''<div class="relate">
       <div class="card"><span class="big">🖼️</span><h5>Sketch small, print big</h5><p>An artist plans a huge mural by sketching a small thumbnail first. Fixing the composition on a postcard is fast; redrawing the whole wall for every change is not. Latent diffusion works on the thumbnail.</p></div>
       <div class="card"><span class="big">🔍</span><h5>Then enlarge in stages</h5><p>Once the thumbnail is right, a printer enlarges it and a second pass adds fine texture. That staged enlarge-and-refine is exactly an upscaling cascade.</p></div>
     </div>
     <p><strong>In one line:</strong> do the hard creative work on a tiny compressed image, then decode and upscale it to full resolution — far cheaper than working big the whole way.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: enlarging a real thumbnail just adds blur — no new detail. The decoder and upscaler are <i>trained</i> to invent plausible detail, which is powerful but can also invent detail that is subtly wrong (section 4).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Compress, denoise, decode","body":
     '''<p>Watch an image compress to a latent, diffusion run on the small code, and the decoder rebuild it. <strong>Click all three.</strong></p>'''},
    {"title":"The compute saving, in numbers","body":
     '''<h4>Step 1 — where the cost lives</h4>
     <p>The denoising network's cost scales with the number of elements it processes. Halve the height and width and you quarter the elements. Compress by a factor f on each side and you shrink the element count by f² — and the whole 50-step loop shrinks with it.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       pixels = H × W<br>
       latent elements ≈ (H/f) × (W/f) = (H × W) / f²<br>
       <span class="dim"># f = compression factor per side (e.g. 8). Cost drops by ≈ f².</span>
     </div></div>
     <p>Symbols: <code>H</code>, <code>W</code> = image height and width in pixels, <code>f</code> = how much smaller each side becomes in latent space. Because area scales with the square, an 8× shrink per side is a 64× reduction in elements.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       image 512 × 512 = 262,144 pixels,  compression f = 8<br><br>
       latent = (512/8) × (512/8) = 64 × 64 = 4,096 elements<br>
       saving = 262,144 / 4,096 = <span class="hl">64×</span> fewer elements to denoise<br>
       <span class="dim"># 64² = 4096, and 262144/4096 = 64 = f²</span>
     </div></div>
     <p>That ~64× is why latent diffusion made consumer-GPU image generation possible: the expensive loop runs on 4,096 numbers, not 262,144.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — fine-detail artifacts.</b> Because diffusion works in a compressed space, the tiniest details — text on a sign, individual teeth, watch hands, distant faces — are reconstructed by the decoder rather than generated at full resolution. They come out garbled, warped, or nonsensical while the overall image looks great. Nothing errors; the picture is beautiful at a glance and broken under a magnifier. Higher-quality autoencoders and cascade refiners reduce it, but it is the signature weakness of latent methods.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — resolution vs compute/memory.</b> A larger latent (less compression) keeps more detail and fewer artifacts but costs quadratically more compute and memory. A smaller latent (more compression) is fast and cheap but loses fine detail. Cascades sidestep some of this — small base + upscalers — but add extra models to train, run, and keep consistent. You are always trading picture fidelity against GPU budget.</div></div>''',
     "gotit":"Got the compute saving"},
    {"title":"Building the latent pipeline","body":
     '''<p>Here is high-res synthesis assembled: encode → diffuse in latent → decode → upscale cascade → full-res image. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the saving with latent diffusion","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_latent.py</code>. Load a pretrained latent diffusion model. Generate an image and print the latent shape vs the pixel shape, and compute the element-count ratio (the compute saving). Then zoom into a region with fine detail (text or a small face) and show the artifact.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 4 artifact.
Create experiments/foundations/m23_latent.py using a pretrained latent diffusion model. Print the latent tensor shape and the decoded image shape, compute the element-count ratio (pixels / latent elements = the ~f^2 compute saving), and time one denoising step in latent space vs an estimate of pixel-space cost. Generate an image containing small text and crop/zoom the text region to reveal the fine-detail artifact.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-04-log.md</code>: what was your latent-vs-pixel element ratio (the f²), and where did fine-detail artifacts show up?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/08-high-res-image-synthesis/README.md">genAI design · 08 High-Res Image Synthesis</a> — the full system design (autoencoder, cascades, serving).</div></div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "encode":{"btn":"① compress to latent","html":'''<span class="prompt">&gt;&gt;&gt;</span> z = autoencoder.encode(image)   <span class="dim"># 512×512 → 64×64</span>
<span class="hl">image: 512×512×3 = 786,432 numbers</span>
<span class="hl">latent z: 64×64×4 = 16,384 numbers</span>   <span class="dim"># ~48× smaller</span>''',
      "take":"<b>①  Shrink first.</b> The autoencoder compresses the image into a small latent code — the thumbnail diffusion will work on."},
    "diffuse":{"btn":"② denoise the small code","html":'''<span class="prompt">&gt;&gt;&gt;</span> for t in reversed(range(50)):
<span class="prompt">...</span>     z = z - alpha * eps_theta(z, t)   <span class="dim"># on 64×64, not 512×512</span>
<span class="hl">50 steps over 4,096 latent positions</span>
<span class="dim"># vs 262,144 pixels — ~64× cheaper per step</span>''',
      "take":"<b>②  Do the hard work small.</b> The 50-step denoising loop runs on the tiny latent, so each step is ~f² cheaper. This is the whole saving."},
    "decode":{"btn":"③ decode to full-res","html":'''<span class="prompt">&gt;&gt;&gt;</span> image = autoencoder.decode(z)   <span class="dim"># 64×64 → 512×512</span>
<span class="hl">→ sharp 512×512 image</span>
<span class="dim"># zoom into tiny text: often garbled (fine-detail artifact)</span>''',
      "take":"<b>③  Blow it back up.</b> The decoder rebuilds full resolution. Overall it looks great, but tiny details (text, teeth) can come out garbled."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='16' width='180' height='32' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='260' y='36' fill='#5A544E'>512×512 image (262k px)</text></g></svg>",
     "note":"<b>Full-resolution image.</b> A million-ish numbers. Denoising directly on these, 50 times, is the expensive path we want to avoid."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='20' width='120' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='100' y='39' fill='#5A544E'>512×512</text><text x='172' y='39' fill='#6B645E'>→ encode →</text><rect x='300' y='24' width='70' height='22' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='335' y='39' fill='#1F6280'>64×64</text></g></svg>",
     "note":"<b>Compress to a latent.</b> The autoencoder shrinks each side by f=8, so the element count drops by f²=64×. Diffusion will run here."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>50-step denoise on 64×64 latent</text></g></svg>",
     "note":"<b>Denoise the small code.</b> The full diffusion loop runs on 4,096 positions instead of 262,144 — the ~64× compute saving."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='24' width='70' height='22' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='95' y='39' fill='#5E5191'>64×64</text><text x='145' y='39' fill='#6B645E'>→ decode →</text><rect x='270' y='20' width='120' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='330' y='39' fill='#1a5c38'>512×512</text></g></svg>",
     "note":"<b>Decode back up.</b> The decoder rebuilds the full-resolution image from the cleaned latent — inventing plausible pixel-level detail."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='18' width='90' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='75' y='38' fill='#1a5c38'>512²</text><text x='128' y='38' fill='#6B645E'>→↑→</text><rect x='165' y='18' width='90' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='210' y='38' fill='#1F6280'>1024²</text><text x='263' y='38' fill='#6B645E'>→↑→</text><rect x='300' y='18' width='100' height='30' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='350' y='38' fill='#9A7208'>2048² 🖼️</text></g></svg>",
     "note":"<b>Cascade to megapixel.</b> Optional upscaling diffusion stages enlarge and add detail step by step — each stage a manageable size."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='18' width='330' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='255' y='38' fill='#C93B3B'>zoom in: tiny text / teeth are garbled</text></g></svg>",
     "note":"<b>The signature failure.</b> Fine detail reconstructed from a compressed code comes out warped — beautiful overall, broken under a magnifier."},
  ],
  quiz=[
    {"q":"1. Why does latent diffusion save so much compute?","opts":["it uses fewer denoising steps","it runs the denoising loop on a small compressed latent instead of on full-resolution pixels","it skips the decoder","it uses no neural network"],"ans":1,"fb":"The expensive 50-step loop runs on a compressed latent (e.g. 64×64) rather than the full pixel grid (512×512), cutting elements by ~f²."},
    {"q":"2. A 1024×1024 image with per-side compression f=8 gives a latent of how many positions per side, and roughly what compute saving?","opts":["256 per side, ~4× saving","128 per side, ~64× saving","512 per side, ~2× saving","64 per side, ~8× saving"],"ans":1,"fb":"1024/8 = 128 per side, and area scales with the square, so the saving is f² = 8² = 64×."},
    {"q":"3. Your generated image looks great but the text on a sign is garbled. This is:","opts":["a retrieval miss","the fine-detail artifact of latent methods","object hallucination","saturation"],"ans":1,"fb":"Tiny details are reconstructed from a compressed code by the decoder, so they warp. It is the signature fine-detail artifact of latent diffusion."},
    {"q":"4. You reduce compression (a larger latent) to fix artifacts. The cost is:","opts":["nothing changes","quadratically more compute and memory","the decoder stops working","lower resolution"],"ans":1,"fb":"A larger latent keeps more detail but element count — and thus compute/memory — grows with the square. That is the resolution vs compute trade-off."},
  ],
  fin={"em":"🔍","h3":"Day 4 complete — megapixels on a budget!",
       "p":"You learned latent diffusion: compress with an autoencoder, denoise the tiny latent (a ~f² compute saving), decode, and cascade-upscale to full resolution. You met the fine-detail artifact and the resolution-vs-compute trade-off. Next: <b>Day 5</b> — steering all this with text prompts and classifier-free guidance."},
))

# ---------------- Day 5 — Text-to-Image ----------------
L("day-05-text-to-image.html", dict(
  qid="m23-d05-t2i", title="Module 23 · Day 5 — Text-to-Image", nav_title="Spiral · M23 Day 5",
  eyebrow="Module 23 · GenAI Systems · Day 5", h1="Text-to-Image: Painting From Words",
  lead="Type \"a red cube on a blue sphere\" and get exactly that picture. That is text-to-image: diffusion, but steered by a text prompt. The steering wheel is classifier-free guidance, the same knob from M22b — turn it up and the image obeys the words more tightly. Today you make prompts actually control the picture, and meet the bug where the colors land on the wrong objects.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how text conditions diffusion, write the classifier-free guidance formula, compute a guided prediction, and name the attribute-binding failure.",
  prev_href="day-04-high-res.html", prev_label="High-Res Image Synthesis", next_href="day-06-headshots.html", next_label="Personalized Headshots",
  sections=[
    {"title":"Steering diffusion with words","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> diffusion denoises step by step (M22b), <span class="term" data-tip="Classifier-Free Guidance: run the model with and without the prompt and push the prediction in the direction the prompt adds, scaled by a guidance strength. From M22b.">classifier-free guidance</span> steers it (M22b), and a text encoder turns words into a vector (M3/M8).</div></div>
     <h4>The task</h4>
     <p><span class="term" data-tip="Generating an image that matches a text description. The prompt conditions a diffusion model so the output shows what the words describe.">Text-to-image</span> takes a written prompt and produces a matching picture. The engine is diffusion, but now every denoising step is <em>conditioned</em> on the prompt, so the image grows toward what the words say.</p>
     <h4>How the text gets in</h4>
     <p>A text encoder turns the prompt into a set of vectors. The denoising network reads them (via cross-attention) at every step, so the emerging image is pulled toward "red cube on blue sphere" and away from unrelated pictures.</p>
     <h4>The steering knob</h4>
     <p>Conditioning alone is often too weak — the image drifts off-prompt. <span class="term" data-tip="How strongly classifier-free guidance pushes the image toward the prompt. Higher = more prompt adherence but less variety and possible over-saturation.">Guidance strength</span> (the CFG scale) is a dial that amplifies how hard the model steers toward the prompt. Turn it up and the picture obeys the words more tightly; turn it down and it wanders but varies more.</p>''',
     "gotit":"Got text conditioning"},
    {"title":"A GPS for the image","body":
     '''<div class="relate">
       <div class="card"><span class="big">🧭</span><h5>Two directions</h5><p>Imagine driving with a map that shows where you'd go with no destination (drift) and where the destination pulls you. The <i>difference</i> between them is "toward the goal." Guidance follows that difference.</p></div>
       <div class="card"><span class="big">🎚️</span><h5>How hard to steer</h5><p>The guidance scale is how hard you yank the wheel toward the destination. A gentle nudge lets you explore side streets (variety); a hard yank drives straight there (strict prompt adherence).</p></div>
     </div>
     <p><strong>In one line:</strong> guidance compares "what the model would draw with no prompt" to "with the prompt," and steers along the difference — the scale sets how hard.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a GPS destination is a single point. A prompt has many parts ("red", "cube", "blue", "sphere") that the model can mix up — steering hard toward the whole prompt does not guarantee each word lands on the right object (section 4).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Guide toward the prompt","body":
     '''<p>Watch the unconditioned prediction, the conditioned one, and the guided blend. <strong>Click all three.</strong></p>'''},
    {"title":"Classifier-free guidance, with numbers","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>At each step the model makes two predictions: one ignoring the prompt (unconditional) and one using it (conditional). The prompt's <em>effect</em> is the difference between them. Guidance takes the unconditional prediction and pushes it along that difference, scaled up by a strength s.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ε̂ = ε<sub>uncond</sub> + s · ( ε<sub>cond</sub> − ε<sub>uncond</sub> )<br>
       <span class="dim"># ε_cond = prediction using the prompt; ε_uncond = ignoring it</span><br>
       <span class="dim"># s = guidance scale. s=1 → plain conditioning; s&gt;1 → stronger adherence</span>
     </div></div>
     <p>Symbols: <code>ε<sub>cond</sub></code> = the noise the model predicts with the prompt, <code>ε<sub>uncond</sub></code> = without it, <code>s</code> = the guidance scale. The term <code>(ε<sub>cond</sub> − ε<sub>uncond</sub>)</code> is "the direction the prompt adds"; multiplying by s amplifies it.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       ε_uncond = 0.2,  ε_cond = 0.5,  guidance s = 7<br><br>
       ε̂ = 0.2 + 7·(0.5 − 0.2) = 0.2 + 7·0.3 = 0.2 + 2.1 = <span class="hl">2.3</span><br>
       <span class="dim"># the prediction is pushed far past ε_cond, toward the prompt — strong adherence</span>
     </div></div>
     <p>Typical scales are s ≈ 5–8. At s=1 the prompt barely bites; at s=15+ the image obeys the words hard but can look over-saturated and lose variety.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — attribute binding errors.</b> Prompt "a red cube and a blue sphere" and the model may paint a <b>blue cube and a red sphere</b> — right objects, right colors, wrong pairing. The model captured the words but not which attribute belongs to which object. The image is crisp and confident, so it passes a glance; only reading it carefully reveals the swap. Cranking guidance does not fix binding — it can even worsen it. Better text encoders and attention-control methods help.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — guidance strength vs diversity.</b> High guidance = tight prompt adherence but nearly identical, sometimes over-saturated, low-variety images (and slower, since CFG runs the model twice per step). Low guidance = varied, natural images that may ignore parts of the prompt. There is no single best s — a product picks a default (≈7) and often lets users tune it per image.</div></div>''',
     "gotit":"Got CFG"},
    {"title":"Building text-to-image","body":
     '''<p>Here it is assembled: prompt → text encoder → conditioned denoise → CFG blend → decode → image. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with prompt control","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_t2i.py</code>. Load a pretrained text-to-image model. Generate the same prompt at guidance scales {1, 3, 7, 15} and compare prompt adherence and diversity. Then test a binding prompt ("a red cube and a blue sphere") and check how often the colors land on the wrong objects.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 5 artifact.
Create experiments/foundations/m23_t2i.py with a pretrained text-to-image diffusion model. Implement classifier-free guidance eps_hat = eps_uncond + s*(eps_cond - eps_uncond). Generate one prompt at guidance s in {1, 3, 7, 15}, saving images and noting adherence vs diversity. Then run a binding test with "a red cube and a blue sphere" across several seeds and report how often the attributes bind to the wrong object.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-05-log.md</code>: which guidance scale looked best, and how often did the red/blue attributes bind to the wrong object?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/09-text-to-image/README.md">genAI design · 09 Text-to-Image</a> — the full system design (text encoder, guidance, serving, eval).</div></div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "uncond":{"btn":"① unconditioned prediction","html":'''<span class="prompt">&gt;&gt;&gt;</span> eps_uncond = model(x, t, prompt="")   <span class="dim"># ignore the prompt</span>
<span class="hl">eps_uncond = 0.2</span>   <span class="dim"># where the image drifts with no words</span>''',
      "take":"<b>①  What it draws with no prompt.</b> The model predicts noise ignoring the text — a generic, unsteered direction."},
    "cond":{"btn":"② conditioned prediction","html":'''<span class="prompt">&gt;&gt;&gt;</span> eps_cond = model(x, t, prompt="a red cube")
<span class="hl">eps_cond = 0.5</span>   <span class="dim"># where the prompt pulls it</span>
<span class="dim"># the difference (0.5 - 0.2) is the prompt's effect</span>''',
      "take":"<b>②  What the prompt adds.</b> With the text, the prediction shifts. The gap between conditioned and unconditioned is the prompt's pull."},
    "guide":{"btn":"③ guided blend (CFG)","html":'''<span class="prompt">&gt;&gt;&gt;</span> s = 7
<span class="prompt">&gt;&gt;&gt;</span> eps_hat = 0.2 + s*(0.5 - 0.2)
<span class="hl">eps_hat = 2.3</span>   <span class="dim"># pushed hard toward the prompt</span>''',
      "take":"<b>③  Amplify the pull.</b> CFG scales the prompt direction by s and steers the step there. Higher s = tighter adherence, less variety."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='37' fill='#9A7208'>prompt: \"a red cube on a blue sphere\"</text></g></svg>",
     "note":"<b>Start with the prompt.</b> The words the image must match — including which attribute belongs to which object."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='22' width='90' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='105' y='40' fill='#9A7208'>prompt</text><text x='162' y='40' fill='#6B645E'>→</text><polygon points='185,20 320,37 320,41 185,54' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='255' y='42' fill='#1F6280'>text encoder</text><text x='336' y='40' fill='#6B645E'>→</text><rect x='360' y='22' width='120' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='420' y='40' fill='#1F6280'>text vectors</text></g></svg>",
     "note":"<b>Encode the text.</b> A text encoder turns the prompt into vectors the denoising network can read via cross-attention."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='18' width='150' height='32' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='105' y='38' fill='#5E5191'>ε_uncond (no prompt)</text><rect x='210' y='18' width='150' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='285' y='38' fill='#1a5c38'>ε_cond (with prompt)</text></g></svg>",
     "note":"<b>Two predictions per step.</b> One ignoring the prompt, one using it. Their difference is the direction the prompt adds."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>ε̂ = ε_uncond + s·(ε_cond − ε_uncond)</text></g></svg>",
     "note":"<b>Classifier-free guidance.</b> Push the prediction along the prompt direction, scaled by s. This is the steering wheel."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='20' width='150' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='115' y='39' fill='#5E5191'>guided denoise ×50</text><text x='205' y='39' fill='#6B645E'>→</text><rect x='235' y='20' width='90' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='280' y='39' fill='#1F6280'>decode</text><text x='335' y='39' fill='#6B645E'>→</text><rect x='365' y='20' width='120' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='425' y='39' fill='#1a5c38'>🖼️ image</text></g></svg>",
     "note":"<b>Denoise and decode.</b> Repeat the guided step, then decode the latent to the final image — matching the prompt."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='180' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='150' y='38' fill='#C93B3B'>blue cube, red sphere</text><text x='260' y='38' fill='#6B645E'>≠</text><rect x='285' y='18' width='195' height='32' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='382' y='38' fill='#5A544E'>prompt: red cube, blue sphere</text></g></svg>",
     "note":"<b>The binding bug.</b> Right objects, right colors, wrong pairing. Guidance strength does not fix attribute binding — sometimes it worsens it."},
  ],
  quiz=[
    {"q":"1. In text-to-image diffusion, how does the prompt influence the image?","opts":["it is drawn as text on the image","a text encoder turns it into vectors the denoising network reads at every step, steering the image toward the words","it sets the number of steps","it replaces the decoder"],"ans":1,"fb":"The prompt is encoded to vectors and read via cross-attention at each denoising step, so the image is pulled toward what the words describe."},
    {"q":"2. With ε_uncond=0.1, ε_cond=0.3, guidance s=5, the guided prediction ε̂ is:","opts":["0.4","1.1","0.3","2.0"],"ans":1,"fb":"ε̂ = 0.1 + 5·(0.3 − 0.1) = 0.1 + 5·0.2 = 0.1 + 1.0 = 1.1. CFG pushes past ε_cond by amplifying the prompt direction."},
    {"q":"3. Prompt says \"red cube, blue sphere\" but you get a blue cube and red sphere. This is:","opts":["a retrieval miss","an attribute-binding error","a fine-detail artifact","memorization"],"ans":1,"fb":"The right objects and colors appear but paired wrongly — an attribute-binding error. Raising guidance does not reliably fix it."},
    {"q":"4. You raise guidance to 15. The most likely effect is:","opts":["more diverse, natural images","tighter prompt adherence but lower variety and possible over-saturation","the prompt is ignored","faster generation"],"ans":1,"fb":"High guidance forces strong adherence but reduces diversity, can over-saturate, and runs the model twice per step (slower). That is the guidance vs diversity trade-off."},
  ],
  fin={"em":"🎨","h3":"Day 5 complete — words became pictures!",
       "p":"You can steer diffusion with text: encode the prompt, predict with and without it, and blend with classifier-free guidance <code>ε̂ = ε_uncond + s·(ε_cond − ε_uncond)</code>. You met attribute-binding errors and the guidance-vs-diversity dial. Next: <b>Day 6</b> — personalizing the model to one specific subject with LoRA/DreamBooth."},
))
# ---------------- Day 6 — Personalized Headshots (LoRA/DreamBooth) ----------------
L("day-06-headshots.html", dict(
  qid="m23-d06-headshots", title="Module 23 · Day 6 — Personalized Headshots", nav_title="Spiral · M23 Day 6",
  eyebrow="Module 23 · GenAI Systems · Day 6", h1="Personalized Headshots: Teaching a Model One Face",
  lead="You upload five selfies and get back a hundred studio-quality headshots of <em>you</em> — as an astronaut, a CEO, a painting. How does a giant model that never saw your face learn it from five photos in minutes? Two tricks: LoRA teaches it cheaply, and DreamBooth binds your identity to a rare word. Today you learn both, and the failure everyone hits: the model overfits and gives you the same pose every time.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain LoRA and DreamBooth personalization, describe the overfitting-vs-likeness tension, estimate LoRA's parameter saving, and name the same-pose overfitting failure.",
  prev_href="day-05-text-to-image.html", prev_label="Text-to-Image", next_href="day-07-image-to-video.html", next_label="Image-to-Video",
  sections=[
    {"title":"Fine-tuning without retraining","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a text-to-image diffusion model (Day 5) and the idea of fine-tuning a pretrained model on new data (M21a).</div></div>
     <h4>The task</h4>
     <p>Take a general text-to-image model and teach it one specific subject — your face, a product, a pet — from just a few photos, so you can then generate that subject in any scene. This is <span class="term" data-tip="Adapting a general generative model so it can reliably produce one specific subject (a person, object, or style) from a handful of example images.">personalization</span>.</p>
     <h4>Trick 1 — LoRA (train cheap)</h4>
     <p>Retraining a billion-parameter model on five photos is wasteful and would overwrite everything it knows. <span class="term" data-tip="Low-Rank Adaptation: freeze the big model and train only a small pair of low-rank matrices added alongside its weights. Far fewer parameters to train and store.">LoRA</span> (Low-Rank Adaptation) freezes the whole model and trains only a tiny set of extra low-rank weights that ride alongside it. You adjust a few million parameters, not a few billion.</p>
     <h4>Trick 2 — DreamBooth (bind an identity)</h4>
     <p><span class="term" data-tip="A personalization method that ties a subject to a rare token (like 'sks') and fine-tunes so the model links that token to the subject, while a class prior ('a person') keeps it from forgetting the general concept.">DreamBooth</span> gives your subject a rare, unused word (like <code>sks</code>) and fine-tunes so the model links "a photo of sks person" to your face. A class-prior term ("a photo of a person") keeps it from forgetting what people look like in general.</p>''',
     "gotit":"Got personalization"},
    {"title":"A caricature artist learning your face","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎨</span><h5>Already knows how to draw</h5><p>A skilled caricature artist already knows faces, lighting, and style. To draw <i>you</i>, they only need a few glances — they are not relearning art, just your specific features. LoRA is those few glances layered on an expert.</p></div>
       <div class="card"><span class="big">🏷️</span><h5>A name tag for your face</h5><p>The artist scribbles "regular #4 → that person by the window" so they can summon your look on command. DreamBooth's rare token <code>sks</code> is that private name tag for your identity.</p></div>
     </div>
     <p><strong>In one line:</strong> LoRA adds a thin layer of "your features" onto an already-expert model, and DreamBooth gives those features a name you can call in a prompt.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a human artist naturally varies the pose. A model trained on five similar selfies can lock onto those exact poses and refuse to vary — the overfitting failure in section 4.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Personalize in three moves","body":
     '''<p>Watch LoRA add a small adapter, DreamBooth bind a token, and the model generate you in a new scene. <strong>Click all three.</strong></p>'''},
    {"title":"LoRA math and the overfitting trap","body":
     '''<h4>Step 1 — the LoRA idea in words</h4>
     <p>A weight matrix W is huge. Instead of changing W, LoRA adds a small correction made of two skinny matrices multiplied together. Two skinny matrices have far fewer numbers than one big one, so training is cheap and the change is easy to store and share.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       W' = W + B · A<br>
       <span class="dim"># W is frozen (d×d). A is r×d, B is d×r, with rank r ≪ d.</span><br>
       <span class="dim"># trainable numbers: 2·d·r instead of d·d</span>
     </div></div>
     <p>Symbols: <code>W</code> = the original frozen weight (size d×d), <code>A</code> and <code>B</code> = the small trainable matrices, <code>r</code> = the <span class="term" data-tip="The inner size of the two LoRA matrices. Small r means a small, cheap adapter; larger r can capture more but risks overfitting.">rank</span> (much smaller than d). Only <code>B·A</code> is learned; <code>W</code> never moves.</p>
     <h4>Step 3 — worked example (the saving)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       d = 1000,  rank r = 4<br><br>
       full matrix: d·d = 1000·1000 = 1,000,000 trainable numbers<br>
       LoRA: 2·d·r = 2·1000·4 = 8,000 trainable numbers<br>
       saving = 1,000,000 / 8,000 = <span class="hl">125×</span> fewer parameters<br>
       <span class="dim"># train and store a tiny adapter, not a whole new model</span>
     </div></div>
     <p>That is why a personalization runs in minutes and the adapter is a few megabytes you can email.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — overfitting to the same pose.</b> With only a few training images, all shot similarly, the model can memorize those exact photos: every generated headshot comes out in the <b>same pose, same angle, same expression</b>, no matter the prompt. It looks like a great likeness — because it is literally reproducing your training selfies — so the metric "does it look like me?" says yes while the model has actually collapsed. Fixes: fewer training steps, lower rank, more varied training photos, and the DreamBooth class-prior term.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — likeness vs flexibility.</b> Train harder / higher rank / more steps → stronger likeness but the model rigidly copies the training photos (poor flexibility, same pose, ignores parts of the prompt). Train lighter → the model stays flexible and follows prompts well but the face drifts and looks less like you. The sweet spot is "recognizably you, but freely posable" — found by tuning steps and rank, not by pushing either extreme.</div></div>''',
     "gotit":"Got LoRA & overfitting"},
    {"title":"Building the personalizer","body":
     '''<p>Here it is assembled: a few photos → LoRA adapter + DreamBooth token → the token in a new prompt → your face in a new scene. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a LoRA personalization","body":
     '''<p>Pick one. Use your own photos or a public consented subject only.</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_lora_dreambooth.py</code>. Fine-tune a small text-to-image model with LoRA + DreamBooth on ~5 images of one subject bound to a rare token. Print the trainable-parameter count vs the full model. Then vary training steps and show that too many steps causes the same-pose overfitting.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 6 artifact.
Create experiments/foundations/m23_lora_dreambooth.py: LoRA + DreamBooth personalization of a small text-to-image model on ~5 images of one consented subject bound to a rare token (e.g. "sks"). Print trainable params (2*d*r per adapted layer) vs full model params and the ratio. Sweep training steps {200, 800, 2000} and generate the subject in varied prompts; report when outputs collapse to the same pose (overfitting) and when likeness is lost (undertrained).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-06-log.md</code>: what was your LoRA parameter ratio, and at how many steps did the same-pose overfitting appear?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/10-personalized-headshots/README.md">genAI design · 10 Personalized Headshots</a> — the full system design (LoRA/DreamBooth, serving, safety).</div></div></div>''',
     "gotit":"Done — on to Day 7"},
  ],
  demos={
    "lora":{"btn":"① add a LoRA adapter","html":'''<span class="prompt">&gt;&gt;&gt;</span> model.freeze()   <span class="dim"># the big model stays fixed</span>
<span class="prompt">&gt;&gt;&gt;</span> add_lora(rank=4)   <span class="dim"># tiny trainable B·A alongside W</span>
<span class="hl">trainable: 8,000 vs 1,000,000 (125× fewer)</span>''',
      "take":"<b>①  Train cheap.</b> Freeze the whole model and learn only a tiny low-rank adapter. Minutes to train, megabytes to store."},
    "token":{"btn":"② bind a rare token","html":'''<span class="prompt">&gt;&gt;&gt;</span> train("a photo of sks person", your_selfies)
<span class="prompt">&gt;&gt;&gt;</span> train("a photo of a person", generic_people)   <span class="dim"># class prior</span>
<span class="hl">sks ↔ your face; 'person' stays general</span>''',
      "take":"<b>②  Name your face.</b> DreamBooth ties the rare token 'sks' to you, while the class prior keeps the model's general idea of a person intact."},
    "gen":{"btn":"③ generate you anywhere","html":'''<span class="prompt">&gt;&gt;&gt;</span> generate("sks person as an astronaut, studio light")
<span class="hl">→ you, in a spacesuit, new pose</span>
<span class="dim"># overfit warning: too many steps → always the same pose</span>''',
      "take":"<b>③  Summon your identity.</b> Put the token in any prompt to place your face in new scenes — unless it overfit, in which case every image is the same pose."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='18' width='55' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='87' y='37' fill='#5A544E'>📷</text><rect x='125' y='18' width='55' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='152' y='37' fill='#5A544E'>📷</text><rect x='190' y='18' width='55' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='217' y='37' fill='#5A544E'>📷</text><text x='275' y='37' fill='#6B645E'>~5 selfies</text></g></svg>",
     "note":"<b>Start with a few photos.</b> Just a handful of images of one subject — far too few to train a big model from scratch."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='22' width='150' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E' stroke-width='2'/><text x='135' y='41' fill='#5A544E'>frozen big model W</text><text x='222' y='41' fill='#6B645E'>+</text><rect x='245' y='24' width='120' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='305' y='41' fill='#1F6280'>LoRA B·A (r=4)</text></g></svg>",
     "note":"<b>Add a LoRA adapter.</b> Freeze W and train only the tiny B·A correction — 2·d·r numbers, ~125× fewer than the full matrix."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='150' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='165' y='36' fill='#9A7208'>token 'sks' ↔ face</text><rect x='260' y='16' width='200' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='360' y='36' fill='#1a5c38'>class prior: 'a person'</text></g></svg>",
     "note":"<b>Bind the identity (DreamBooth).</b> Link a rare token to the subject, while a class-prior term stops the model forgetting people in general."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='16' width='260' height='32' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='190' y='36' fill='#5E5191'>\"sks person as an astronaut\"</text><text x='335' y='36' fill='#6B645E'>→ 🧑‍🚀</text></g></svg>",
     "note":"<b>Prompt with the token.</b> Now the token can appear in any prompt to place the subject in new scenes, styles, and poses."},
    {"viz":"<svg viewBox='0 0 520 78'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='95' y='28' fill='#5E5191'>too few steps</text><text x='95' y='44' fill='#6B645E'>flexible, weak likeness</text><line x1='185' y1='38' x2='355' y2='38' stroke='#6B645E' stroke-width='2'/><circle cx='270' cy='38' r='5' fill='#2D8B55'/><text x='435' y='28' fill='#5E5191'>too many steps</text><text x='435' y='44' fill='#6B645E'>strong likeness, same pose</text></g></svg>",
     "note":"<b>Likeness vs flexibility dial.</b> Undertrain and the face drifts; overtrain and every image is the same memorized pose. The sweet spot is in the middle."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='18' width='330' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='255' y='38' fill='#C93B3B'>overfit: every output = the same pose</text></g></svg>",
     "note":"<b>The failure to watch.</b> Overfitting makes the model reproduce the training selfies — perfect 'likeness', zero variety, ignores the prompt."},
  ],
  quiz=[
    {"q":"1. What does LoRA do to make personalization cheap?","opts":["it retrains the whole model on the new photos","it freezes the big model and trains only a small pair of low-rank matrices added alongside the weights","it deletes most of the model","it uses no training at all"],"ans":1,"fb":"LoRA freezes W and learns only a small B·A correction (2·d·r params ≪ d·d), so training is fast and the adapter is tiny to store."},
    {"q":"2. With d=500 and rank r=2, LoRA's trainable count (2·d·r) vs the full d·d, the saving is:","opts":["2×","125×","250×","500×"],"ans":1,"fb":"Full = 500·500 = 250,000; LoRA = 2·500·2 = 2,000; ratio = 250,000 / 2,000 = 125×. A tiny adapter replaces a whole weight matrix."},
    {"q":"3. Every headshot comes out in the exact same pose regardless of the prompt. This is:","opts":["good likeness, no problem","overfitting — the model memorized the few training photos","an attribute-binding error","a retrieval miss"],"ans":1,"fb":"With few, similar training images the model reproduces them exactly — high 'likeness' but collapsed variety. Fewer steps, lower rank, and the class prior help."},
    {"q":"4. You train for far more steps to boost likeness. The main risk is:","opts":["the face looks less like the subject","the model rigidly copies the training photos and ignores the prompt (poor flexibility)","LoRA stops saving parameters","the token is forgotten"],"ans":1,"fb":"Overtraining maximizes likeness but destroys flexibility — the model reproduces training poses and ignores prompts. That is the likeness-vs-flexibility trade-off."},
  ],
  fin={"em":"🎨","h3":"Day 6 complete — one face, taught in minutes!",
       "p":"You learned personalization: LoRA trains a tiny low-rank adapter (~125× fewer params) on a frozen model, and DreamBooth binds your identity to a rare token with a class prior. You met the same-pose overfitting failure and the likeness-vs-flexibility dial. Next: <b>Day 7</b> — extending image diffusion into moving pictures: image-to-video."},
))

# ---------------- Day 7 — Image-to-Video ----------------
L("day-07-image-to-video.html", dict(
  qid="m23-d07-i2v", title="Module 23 · Day 7 — Image-to-Video", nav_title="Spiral · M23 Day 7",
  eyebrow="Module 23 · GenAI Systems · Day 7", h1="Image-to-Video: Making a Picture Move",
  lead="A video is just many images shown fast — 24 frames a second. So can we take our image diffusion model and make it generate a whole sequence of frames? Almost. The hard part is not drawing each frame; it is making frame 2 a believable <em>continuation</em> of frame 1. Today you meet the motion problem and the flicker that ruins naive attempts.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how video extends image diffusion to many frames, describe the motion problem, estimate the compute of a clip, and name the inter-frame flicker failure.",
  prev_href="day-06-headshots.html", prev_label="Personalized Headshots", next_href="day-08-temporal-consistency.html", next_label="Temporal Consistency",
  sections=[
    {"title":"A video is a stack of frames","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> image diffusion and latent diffusion (Days 4–5). Today: generate not one image but a sequence.</div></div>
     <h4>The task</h4>
     <p><span class="term" data-tip="Generating a short video from a starting image (or a text prompt), by producing a sequence of frames that move believably over time.">Image-to-video</span> starts from one image (or a prompt) and generates a run of frames that, played fast, look like motion. A short 2-second clip at 24 <span class="term" data-tip="One still image in a video. Video is many frames shown quickly; 24 frames per second looks like smooth motion.">frames</span> per second is ~48 images the model must produce.</p>
     <h4>The naive idea (and why it fails)</h4>
     <p>You could just run the image model 48 times independently. But each run invents its own details, so the person's shirt changes color, the background shifts, and objects pop in and out. Playing them back looks like violent flicker, not motion.</p>
     <h4>The fix: generate frames together</h4>
     <p>Video models generate the frames <em>jointly</em>, with <span class="term" data-tip="Attention that lets each frame look at other frames in the clip, so the model keeps details consistent across time instead of reinventing them per frame.">temporal attention</span> so each frame can look at its neighbors and stay consistent. And they must solve the <span class="term" data-tip="Producing believable movement between frames — where things go and how fast — not just static frames that happen to differ.">motion problem</span>: deciding how things should move, not just what they look like.</p>''',
     "gotit":"Got the setup"},
    {"title":"A flip-book","body":
     '''<div class="relate">
       <div class="card"><span class="big">📓</span><h5>Pages that continue</h5><p>A flip-book animates because each page is <i>almost</i> the previous page, nudged a little. Draw each page from scratch, ignoring the last, and flipping it looks like chaos, not movement.</p></div>
       <div class="card"><span class="big">🎬</span><h5>Motion is the story</h5><p>A good animator decides how the character <i>moves</i> — the arc of a jump, the sway of hair — before drawing any single frame. The model must do the same: plan motion, then render frames that follow it.</p></div>
     </div>
     <p><strong>In one line:</strong> video generation is drawing a flip-book where every page continues the last, so the frames must be planned together with the motion in mind.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a flip-book artist sees every page they already drew. A naive video model may only weakly connect frames, so without explicit temporal attention it "forgets" the previous frame and flicker creeps back.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"From one frame to many","body":
     '''<p>Watch a single frame, naive per-frame generation (flicker), and jointly-generated frames. <strong>Click all three.</strong></p>'''},
    {"title":"The cost of frames and the flicker failure","body":
     '''<h4>Step 1 — the cost, in words</h4>
     <p>Each frame is roughly the cost of one image. A clip's cost is the per-frame cost times the number of frames — plus extra for the temporal attention that links them. So doubling the number of frames roughly doubles the compute.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       frames = fps × seconds<br>
       clip_cost ≈ frames × cost_per_frame  (+ temporal-attention overhead)<br>
       <span class="dim"># fps = frames per second; more frames = smoother but costlier</span>
     </div></div>
     <p>Symbols: <code>fps</code> = frames per second, <code>seconds</code> = clip length, <code>cost_per_frame</code> ≈ one image's diffusion cost. The temporal attention adds an overhead that grows with how many frames attend to each other.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       clip = 24 fps × 2 seconds = 48 frames<br>
       if one frame costs 1 unit → clip ≈ <span class="hl">48 units</span> (before temporal overhead)<br><br>
       drop to 12 fps × 2 s = 24 frames → ≈ <span class="hl">24 units</span> (half the cost, choppier motion)<br>
       <span class="dim"># fewer frames = cheaper but less smooth</span>
     </div></div>
     <p>This is why generated video clips are short and often low-fps: the cost grows straight with frame count.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — inter-frame flicker.</b> If frames are generated too independently, small details the model is free to choose — texture, exact color, background clutter, tiny objects — differ frame to frame. Each single frame looks perfect, so a still screenshot passes review, but <b>played back the video shimmers and boils</b>. The defect only appears in motion, so it hides from any per-frame check. Temporal attention and latent propagation (tomorrow) are the cure; measuring it needs a temporal-consistency metric, not a per-frame one.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — frame count vs compute.</b> More frames (higher fps or longer clips) give smoother, more lifelike motion but cost proportionally more compute and memory, and stress temporal consistency over a longer span. Fewer frames are cheap and easier to keep consistent but look choppy. Systems pick a short clip at moderate fps and often interpolate extra frames afterward as a cheaper way to smooth motion.</div></div>''',
     "gotit":"Got cost & flicker"},
    {"title":"Building image-to-video","body":
     '''<p>Here it is assembled: start frame → frame stack → temporal attention links them → motion → a clip. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the flicker with a mini video model","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_i2v.py</code>. Take a starting image and generate a short clip two ways: (a) run the image model per-frame independently, (b) generate frames jointly (or with a pretrained video model). Compute a simple temporal-consistency metric (mean absolute difference between consecutive frames in still regions) and show (a) flickers while (b) is stable.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 7 artifact.
Create experiments/foundations/m23_i2v.py: from a start image, generate a short clip (a) per-frame independently with an image diffusion model and (b) with a pretrained image-to-video model. Compute frames = fps*seconds and estimate clip cost. Measure inter-frame flicker as the mean absolute pixel difference between consecutive frames in a background/still region, and show (a) has high flicker while (b) is temporally stable.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-07-log.md</code>: how much higher was the flicker metric for the per-frame version vs the joint one?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/11-text-to-video/README.md">genAI design · 11 Text-to-Video</a> — the full video generation system design.</div></div></div>''',
     "gotit":"Done — on to Day 8"},
  ],
  demos={
    "frame":{"btn":"① one frame (an image)","html":'''<span class="prompt">&gt;&gt;&gt;</span> frame0 = image_model(prompt)   <span class="dim"># just Day 5's model</span>
<span class="hl">a single sharp image</span>
<span class="dim"># a video needs ~48 of these for a 2-second clip</span>''',
      "take":"<b>①  Start with one image.</b> Each frame is the image-diffusion output you already know. The question is how to make many that flow."},
    "naive":{"btn":"② naive per-frame (flicker)","html":'''<span class="prompt">&gt;&gt;&gt;</span> frames = [image_model(prompt) for _ in range(48)]
<span class="hl">each frame invents its own details</span>
<span class="bad">→ shirt color, background jump every frame = FLICKER</span>''',
      "take":"<b>②  Generate independently = flicker.</b> With no link between frames, details change every frame. Each still looks fine; the video shimmers."},
    "joint":{"btn":"③ joint frames (smooth)","html":'''<span class="prompt">&gt;&gt;&gt;</span> frames = video_model(frame0, temporal_attn=True)
<span class="hl">each frame looks at its neighbors</span>
<span class="ok">→ consistent details, believable motion</span>''',
      "take":"<b>③  Generate together = motion.</b> Temporal attention lets frames see each other, so details stay consistent and the motion looks real."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='200' y='16' width='120' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='37' fill='#1a5c38'>🖼️ start frame</text></g></svg>",
     "note":"<b>Start with one frame.</b> A single image — the same kind of output as Day 5's text-to-image model."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='40' height='30' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='60' y='37' fill='#1a5c38'>f1</text><rect x='95' y='18' width='40' height='30' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='115' y='37' fill='#1F6280'>f2</text><rect x='150' y='18' width='40' height='30' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='170' y='37' fill='#5E5191'>f3</text><rect x='205' y='18' width='40' height='30' rx='4' fill='#FCF3DC' stroke='#C99A12'/><text x='225' y='37' fill='#9A7208'>f4</text><text x='290' y='37' fill='#6B645E'>… 48 frames</text></g></svg>",
     "note":"<b>Extend to a stack of frames.</b> A 2-second, 24-fps clip is ~48 frames. Cost ≈ frames × per-frame cost."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='22' width='40' height='28' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='80' y='40' fill='#1F6280'>f1</text><rect x='130' y='22' width='40' height='28' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='150' y='40' fill='#1F6280'>f2</text><rect x='200' y='22' width='40' height='28' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='220' y='40' fill='#1F6280'>f3</text><path d='M80,22 Q115,6 150,22' fill='none' stroke='#7C6DAA' stroke-width='1.5'/><path d='M150,22 Q185,6 220,22' fill='none' stroke='#7C6DAA' stroke-width='1.5'/><text x='340' y='40' fill='#5E5191'>temporal attention</text></g></svg>",
     "note":"<b>Link frames with temporal attention.</b> Each frame looks at its neighbors so shared details stay consistent instead of being reinvented."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>solve the motion problem: where things move</text></g></svg>",
     "note":"<b>Plan the motion.</b> The model must decide how things move over time — the arc of a jump, a pan — not just render static frames."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='37' fill='#1a5c38'>▶️ smooth 2-second clip</text></g></svg>",
     "note":"<b>Play it back.</b> Jointly generated, motion-aware frames play as believable movement — a real clip, not a flickering slideshow."},
    {"viz":"<svg viewBox='0 0 520 62'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='18' width='330' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='255' y='38' fill='#C93B3B'>naive per-frame → flicker (only visible in motion)</text></g></svg>",
     "note":"<b>The failure to watch.</b> Independent frames shimmer when played. Each still passes review; only the moving video reveals the flicker."},
  ],
  quiz=[
    {"q":"1. Why does generating each video frame independently look bad?","opts":["it is too slow","each frame invents its own details, so they change frame-to-frame and the video flickers","the frames are too sharp","it uses no diffusion"],"ans":1,"fb":"With no link between frames, free details (color, background, small objects) differ each frame — the clip shimmers even though each still looks fine."},
    {"q":"2. A 24-fps, 3-second clip needs how many frames, and what happens to cost if you halve the fps?","opts":["72 frames; cost roughly halves","24 frames; cost doubles","72 frames; cost is unchanged","12 frames; cost quadruples"],"ans":0,"fb":"frames = 24×3 = 72. Cost ≈ frames × per-frame cost, so halving fps to 12 gives 36 frames — roughly half the cost (and choppier motion)."},
    {"q":"3. Each individual frame of your clip looks perfect, but playback shimmers. This is:","opts":["an attribute-binding error","inter-frame flicker — a temporal failure invisible to per-frame checks","a fine-detail artifact","object hallucination"],"ans":1,"fb":"The defect appears only in motion, so per-frame review misses it. It is inter-frame flicker, fixed by temporal attention and measured with a temporal metric."},
    {"q":"4. What lets frames stay consistent instead of being reinvented each time?","opts":["higher guidance","temporal attention, so each frame can look at its neighbors","a bigger text encoder","more denoising steps per frame"],"ans":1,"fb":"Temporal attention connects frames across time so shared details are kept consistent — the core mechanism that turns a flickering stack into smooth motion."},
  ],
  fin={"em":"🎬","h3":"Day 7 complete — pictures that move!",
       "p":"You extended image diffusion to video: a clip is many frames (frames = fps × seconds), generated jointly with temporal attention to solve the motion problem. You met inter-frame flicker — invisible to per-frame checks — and the frame-count-vs-compute trade-off. Next: <b>Day 8</b> — keeping identity and objects stable across a whole clip: temporal consistency."},
))
# ---------------- Day 8 — Temporal Consistency ----------------
L("day-08-temporal-consistency.html", dict(
  qid="m23-d08-temporal", title="Module 23 · Day 8 — Temporal Consistency", nav_title="Spiral · M23 Day 8",
  eyebrow="Module 23 · GenAI Systems · Day 8", h1="Temporal Consistency: Keeping Things Stable in Motion",
  lead="Yesterday's flicker was small details wobbling. Today's problem is bigger: over a longer clip, the <em>identity</em> of things drifts. The hero's jacket slowly turns from red to brown; a character's face morphs into a stranger by the end. That is a temporal-consistency failure. Today you learn the two tools that fight it — temporal attention and latent propagation — and why long clips are so hard.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain temporal attention and latent propagation, describe a consistency score, compute a simple drift measure, and name the identity-drift failure over long clips.",
  prev_href="day-07-image-to-video.html", prev_label="Image-to-Video", next_href="day-09-serving-safety.html", next_label="Serving & Safety Trade-offs",
  sections=[
    {"title":"Staying the same thing over time","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 7 — video is a stack of frames and <span class="term" data-tip="Attention that lets each frame look at other frames in the clip, keeping shared details consistent across time.">temporal attention</span> links them.</div></div>
     <h4>The task</h4>
     <p><span class="term" data-tip="Keeping the same objects, colors, and identities stable across all the frames of a clip, so a thing looks like the same thing from start to finish.">Temporal consistency</span> means the same object stays the same object across every frame: the same face, the same shirt color, the same background layout. Flicker (Day 7) is fast small wobble; <span class="term" data-tip="When an object's identity slowly changes over a clip — a face morphs, a color shifts — so the end looks like a different thing than the start.">identity drift</span> is a slow, steady change over many frames that ends somewhere wrong.</p>
     <h4>Tool 1 — temporal attention (look across frames)</h4>
     <p>As in Day 7, each frame attends to other frames. For consistency we especially want later frames to attend to <em>early anchor</em> frames, so the model keeps checking back against how things started rather than slowly wandering.</p>
     <h4>Tool 2 — latent propagation (carry the state forward)</h4>
     <p><span class="term" data-tip="Reusing part of a frame's latent code as the starting point for the next frame, so shared structure is carried forward instead of regenerated from scratch.">Latent propagation</span> carries a frame's latent code forward into the next frame's generation. Because the next frame starts from the previous one's state, shared structure (the face, the layout) is inherited, not reinvented — so it stays put.</p>''',
     "gotit":"Got the two tools"},
    {"title":"A relay race with a baton","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏃</span><h5>Pass the baton</h5><p>In a relay, each runner hands the same baton to the next. The baton is what stays constant across the whole race. Latent propagation hands the "state baton" from frame to frame so the content persists.</p></div>
       <div class="card"><span class="big">📌</span><h5>Check the anchor</h5><p>Sailors take bearings off a fixed lighthouse so they don't drift off course. Later frames attending to an early anchor frame is the same idea — a fixed reference that stops slow drift.</p></div>
     </div>
     <p><strong>In one line:</strong> carry a state baton forward (latent propagation) and keep checking a fixed anchor (temporal attention) so things stay the same thing over the whole clip.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a baton is identical every handoff. A propagated latent is slightly changed each frame, so tiny errors can still accumulate — which is exactly why long clips remain hard even with these tools.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Measure and fix drift","body":
     '''<p>Watch identity drift over frames, latent propagation carry state forward, and a consistency score fall then rise. <strong>Click all three.</strong></p>'''},
    {"title":"Measuring drift and why long clips break","body":
     '''<h4>Step 1 — measuring consistency, in words</h4>
     <p>To catch drift, compare a stable feature of each frame — say the subject's identity embedding — against the first frame. If the difference grows steadily as the clip goes on, the identity is drifting.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       drift(t) = ‖ emb(frame<sub>t</sub>) − emb(frame<sub>0</sub>) ‖<br>
       <span class="dim"># emb = an identity/feature embedding of a frame</span><br>
       <span class="dim"># drift near 0 = consistent; drift growing with t = identity drift</span>
     </div></div>
     <p>Symbols: <code>emb(frame<sub>t</sub>)</code> = a feature vector of frame t (e.g. a face embedding), <code>frame<sub>0</sub></code> = the first frame, <code>‖·‖</code> = the length of the difference. A flat drift curve means consistent; a rising one means drift.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       identity embedding at frames 0, 24, 48 (2-second clip):<br>
       emb₀ = [1.0, 0.0],  emb₂₄ = [0.8, 0.2],  emb₄₈ = [0.4, 0.5]<br><br>
       drift(24) = √((1.0−0.8)² + (0.0−0.2)²) = √(0.04+0.04) = √0.08 ≈ <span class="hl">0.28</span><br>
       drift(48) = √((1.0−0.4)² + (0.0−0.5)²) = √(0.36+0.25) = √0.61 ≈ <span class="hl">0.78</span><br>
       <span class="dim"># drift grows with time → the face is morphing away from frame 0</span>
     </div></div>
     <p>A rising drift curve is the alarm; a good clip keeps it near flat with temporal attention and latent propagation.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — identity drift over long clips.</b> Each frame is only slightly different from the last, so <b>no single frame or transition looks wrong</b> — the change is below the per-frame noticing threshold. But small changes accumulate: over 100+ frames the red jacket has become brown and the face is a different person, all while every consecutive pair looks fine. Because it hides from short reviews and per-frame checks, you must measure drift against an early anchor frame, not just neighbor-to-neighbor.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — clip length vs consistency.</b> Longer clips are more useful but drift compounds with every frame, and attending to <i>all</i> frames to stay consistent costs compute that grows fast with length. Shorter clips (or clips split into chunks with shared anchors) stay consistent cheaply but limit what you can show. Systems cap clip length, use anchor/keyframe conditioning, and stitch chunks — trading unlimited length for reliable consistency.</div></div>''',
     "gotit":"Got drift & the trade-off"},
    {"title":"Building consistency","body":
     '''<p>Here it is assembled: frames → temporal attention to an anchor → latent propagation baton → flat drift curve → a stable clip. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it by measuring drift","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_temporal.py</code>. Generate a longer clip (say 96 frames) two ways: with and without latent propagation / anchor conditioning. Compute drift(t) = distance of each frame's identity embedding from frame 0, and plot the two drift curves to show propagation keeps it flat.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 8 artifact.
Create experiments/foundations/m23_temporal.py: generate a ~96-frame clip (a) without and (b) with latent propagation / early-anchor conditioning. Extract an identity/feature embedding per frame, compute drift(t) = ||emb(frame_t) - emb(frame_0)||, and plot both drift curves. Show that without propagation drift rises steadily (identity drift) while with it drift stays near flat.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-08-log.md</code>: how much did drift(t) grow by the last frame with vs without latent propagation?</div>
     <div class="callout c-info" style="margin-top:.6rem"><span class="ic">↗</span><div><b>Go deeper:</b> <a href="'''+GD+'''/11-text-to-video/README.md">genAI design · 11 Text-to-Video</a> — temporal consistency in the full video system design.</div></div></div>''',
     "gotit":"Done — on to Day 9"},
  ],
  demos={
    "drift":{"btn":"① identity drift over frames","html":'''<span class="prompt">&gt;&gt;&gt;</span> [emb(f) for f in clip]   <span class="dim"># face embedding per frame</span>
<span class="hl">frame 0: the hero</span>
<span class="hl">frame 48: subtly different</span>
<span class="bad">frame 96: a different person — drift!</span>''',
      "take":"<b>①  Slow drift.</b> Each frame is close to the last, but over 96 frames the identity has quietly morphed into someone else."},
    "prop":{"btn":"② latent propagation","html":'''<span class="prompt">&gt;&gt;&gt;</span> z_next = generate(prompt, init_from=z_prev)
<span class="hl">the previous frame's latent seeds the next</span>
<span class="dim"># shared structure is inherited, not reinvented</span>''',
      "take":"<b>②  Pass the baton.</b> Starting each frame from the previous latent carries the face and layout forward, so they stay put."},
    "score":{"btn":"③ consistency score","html":'''<span class="prompt">&gt;&gt;&gt;</span> drift(t) = norm(emb(f_t) - emb(f_0))
<span class="bad">no propagation: drift 0.0 → 0.78 (rises)</span>
<span class="ok">with propagation: drift 0.0 → 0.09 (flat)</span>''',
      "take":"<b>③  Measure against the anchor.</b> Drift from frame 0 rises without help and stays flat with propagation + anchor attention. That flat curve is the goal."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='40' height='30' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='60' y='37' fill='#1a5c38'>f0</text><rect x='120' y='18' width='40' height='30' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='140' y='37' fill='#1F6280'>f48</text><rect x='200' y='18' width='40' height='30' rx='4' fill='#FDE8E8' stroke='#C93B3B'/><text x='220' y='37' fill='#C93B3B'>f96</text><text x='330' y='37' fill='#6B645E'>identity drifting →</text></g></svg>",
     "note":"<b>The problem.</b> Over a long clip the subject's identity slowly changes — frame 96 no longer matches frame 0."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='40' height='28' rx='4' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='60' y='42' fill='#1a5c38'>anchor f0</text><rect x='150' y='24' width='40' height='28' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='170' y='42' fill='#1F6280'>fₜ</text><path d='M60,24 Q115,4 170,24' fill='none' stroke='#7C6DAA' stroke-width='1.5'/><text x='115' y='14' fill='#5E5191'>attend</text></g></svg>",
     "note":"<b>Temporal attention to an anchor.</b> Later frames look back at an early anchor frame — a fixed reference that stops slow wandering."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='16' width='70' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='95' y='35' fill='#5E5191'>z_prev</text><text x='150' y='35' fill='#6B645E'>→ seeds →</text><rect x='250' y='16' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='285' y='35' fill='#1F6280'>z_next</text><text x='390' y='35' fill='#6B645E'>baton passed</text></g></svg>",
     "note":"<b>Latent propagation.</b> The previous frame's latent seeds the next, so shared structure is inherited like a passed baton, not regenerated."},
    {"viz":"<svg viewBox='0 0 520 74'><g font-family='monospace' font-size='11' text-anchor='middle'><polyline points='40,55 130,42 220,28 310,16 400,8' fill='none' stroke='#C93B3B' stroke-width='2'/><polyline points='40,55 130,54 220,53 310,52 400,51' fill='none' stroke='#2D8B55' stroke-width='2'/><text x='260' y='70' fill='#6B645E'>red = drift rising (bad) · green = flat (consistent)</text></g></svg>",
     "note":"<b>Drift curve.</b> drift(t) = ‖emb(fₜ) − emb(f0)‖. Rising (red) means identity drift; near-flat (green) means the clip stayed consistent."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='37' fill='#1a5c38'>▶️ stable clip, same subject</text></g></svg>",
     "note":"<b>The result.</b> With anchor attention and latent propagation, the same subject stays the same from first frame to last."},
  ],
  quiz=[
    {"q":"1. What is identity drift in a generated clip?","opts":["fast flicker between neighboring frames","a slow, steady change of an object's identity over many frames so the end differs from the start","the clip playing too fast","the prompt being ignored"],"ans":1,"fb":"Drift is the slow accumulation of small per-frame changes until the subject (face, color) has morphed — distinct from Day 7's fast flicker."},
    {"q":"2. How does latent propagation help consistency?","opts":["it deletes early frames","it seeds each frame's generation from the previous frame's latent, so shared structure is carried forward","it raises the guidance scale","it shortens the prompt"],"ans":1,"fb":"By starting each frame from the previous latent (a passed baton), the face and layout are inherited rather than reinvented, so they stay put."},
    {"q":"3. emb(f0)=[1,0], emb(f_t)=[0,1]. The drift(t) = ‖emb(f_t)−emb(f0)‖ is:","opts":["0","1","√2 ≈ 1.41","2"],"ans":2,"fb":"Difference = [−1, 1]; norm = √((−1)²+1²) = √2 ≈ 1.41. A large drift means the identity moved far from frame 0."},
    {"q":"4. Every consecutive pair of frames looks fine, yet the 100-frame clip ends with a different-looking person. Why did per-frame review miss it?","opts":["the frames were too sharp","drift is below the per-frame noticing threshold but accumulates — you must measure against an early anchor, not just neighbors","the prompt was wrong","it used no diffusion"],"ans":1,"fb":"Each step is a tiny change that no single review flags, but they compound. Measuring drift against an anchor frame (not neighbor-to-neighbor) exposes it."},
  ],
  fin={"em":"📌","h3":"Day 8 complete — the subject stays itself!",
       "p":"You can now keep clips consistent: temporal attention to an early anchor plus latent propagation carry identity forward, and drift(t) = ‖emb(fₜ) − emb(f0)‖ measures it. You met identity drift — invisible per-frame but real over a clip — and the clip-length-vs-consistency trade-off. Next: <b>Day 9</b> — the module synthesis: serving generative media fast, cheaply, and safely."},
))

# ---------------- Day 9 — Serving & Safety Trade-offs ----------------
L("day-09-serving-safety.html", dict(
  qid="m23-d09-serving", title="Module 23 · Day 9 — Serving & Safety Trade-offs", nav_title="Spiral · M23 Day 9",
  eyebrow="Module 23 · GenAI Systems · Day 9", h1="Serving & Safety: Shipping Generative Media",
  lead="A model in a notebook helps no one. To serve millions of users you must generate media fast, cheaply, and — crucially — safely. Today ties the whole module together: the latency/batching/cost levers that make serving affordable, and the safety layer (NSFW filters, watermarking, provenance) that keeps it responsible. And the scariest failure of all: a filter that lets harmful content slip through.",
  goal="<b>🎯 By the end, you'll be able to:</b> reason about latency, batching, and cost for generative serving, describe NSFW filtering, watermarking, and provenance, compute a filter's false-negative rate, and name the failure where a leaky filter ships harmful content.",
  prev_href="day-08-temporal-consistency.html", prev_label="Temporal Consistency", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Fast, cheap, and safe — pick all three","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> the whole module (diffusion, latent methods, video) and the idea from M16a that serving cost = latency + throughput + hardware.</div></div>
     <h4>The serving problem</h4>
     <p>Generating an image takes ~50 diffusion steps; a video takes 50× many frames. Serving means doing this for many users at once, within a latency budget, without spending a fortune. Three levers: <span class="term" data-tip="How long a user waits for their result. A hard budget for interactive products.">latency</span>, <span class="term" data-tip="Running many users' requests through the model together at once, so the expensive hardware is used efficiently. Raises throughput but can add waiting time.">batching</span>, and <span class="term" data-tip="The money per generated image or video — driven by how many GPU-seconds it takes.">cost</span> (GPU-seconds per output).</p>
     <h4>The safety layer</h4>
     <p>Generative media can produce harmful content, so a serving system adds guards: an <span class="term" data-tip="A classifier that checks generated output (and prompts) for unsafe content — nudity, violence, hate — and blocks it before it reaches the user.">NSFW filter</span> blocks unsafe output, <span class="term" data-tip="An invisible signal embedded in generated media so it can later be detected as AI-made, even after editing.">watermarking</span> marks output as AI-made, and <span class="term" data-tip="A record of how a piece of media was made (which model, when), often via signed metadata standards like C2PA, so its origin can be verified.">provenance</span> records its origin so it can be traced.</p>
     <h4>Why this is the synthesis</h4>
     <p>Every system in this module — captions, RAG, faces, images, video — must be served and made safe. The engineering trade-offs here decide whether a great model becomes a great, responsible product.</p>''',
     "gotit":"Got serving & safety"},
    {"title":"A restaurant kitchen","body":
     '''<div class="relate">
       <div class="card"><span class="big">🍳</span><h5>Batch the orders</h5><p>A kitchen cooks many of the same dish together to use the stove efficiently — that is batching. But if it waits to fill a full tray, the first customer waits longer. Speed per plate trades against how long anyone waits.</p></div>
       <div class="card"><span class="big">🧑‍⚖️</span><h5>The health inspector</h5><p>Before food leaves the kitchen, an inspector checks it is safe. A fast inspector who waves everything through is cheap but dangerous; a thorough one is slower but catches bad plates. The NSFW filter is that inspector.</p></div>
     </div>
     <p><strong>In one line:</strong> batching cooks efficiently but adds waiting, and the safety filter is an inspector whose thoroughness trades against speed and cost.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a health inspector can taste and be sure. A safety classifier is itself a model that makes mistakes — it will sometimes pass unsafe content (a false negative) with full confidence, which is the failure in section 4.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Trade latency, cost, and safety","body":
     '''<p>Watch batching raise throughput, a cost estimate, and a safety filter's error rates. <strong>Click all three.</strong></p>'''},
    {"title":"Filter error rates and the leak","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>A safety filter is a classifier. It can make two kinds of mistake: a <span class="term" data-tip="When the filter wrongly blocks safe content. Annoying but not dangerous.">false positive</span> (block safe content) and a <span class="term" data-tip="When the filter wrongly passes unsafe content. This is the dangerous mistake — harmful media reaches users.">false negative</span> (pass unsafe content). The false-negative rate is the fraction of unsafe items that slip through — the number that matters most for safety.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       false-negative rate = (unsafe items passed) / (all unsafe items)<br>
       <span class="dim"># the fraction of harmful content the filter FAILS to catch</span><br>
       harmful shipped ≈ false-negative rate × (# unsafe attempts)
     </div></div>
     <p>Symbols: an <em>unsafe item</em> is content that should be blocked; <em>passed</em> means the filter let it through. A lower false-negative rate is safer.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       filter tested on 200 truly-unsafe images<br>
       it blocked 180, but passed 20<br><br>
       false-negative rate = 20 / 200 = <span class="hl">0.10</span> (10%)<br>
       if 10,000 unsafe attempts arrive → ~<span class="hl">1,000</span> harmful images ship<br>
       <span class="dim"># a "90% accurate" filter still leaks 1 in 10 unsafe items</span>
     </div></div>
     <p>At scale, even a small false-negative rate ships a lot of harmful content — which is why safety filters are layered (prompt filter + output filter + human review) rather than trusted alone.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — a leaky filter (high false-negative rate).</b> Teams often report a filter's <i>overall accuracy</i>, which is dominated by the easy safe majority and looks great (99%!). But the number that matters — the <b>false-negative rate on unsafe content</b> — can be terrible while accuracy looks fine, and the leak is silent: harmful outputs reach users with no error, no alert. You only see it if you measure the false-negative rate specifically, red-team the filter, and layer defenses. Optimizing overall accuracy actively hides this.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — latency/cost vs safety-check thoroughness.</b> A thorough safety pipeline (multiple classifiers, higher-resolution checks, human review of edge cases) catches more harmful content but adds latency and cost to every request. A light, fast check is cheap and snappy but leaks more. Because a single harmful output can cause real harm and reputational damage far exceeding the compute saved, responsible systems spend on safety even when it hurts latency and cost — and set the filter threshold to favor false positives (block safe) over false negatives (pass unsafe).</div></div>''',
     "gotit":"Got the safety trade-off"},
    {"title":"Building the serving + safety stack","body":
     '''<p>Here it is assembled: request → batching → generation → safety filter → watermark + provenance → user. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a serving+safety sim","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m23_serving_safety.py</code>. Simulate serving: vary batch size and measure throughput vs per-request latency and a cost estimate. Then simulate a safety filter over a labeled set and compute its overall accuracy AND its false-negative rate on unsafe items — show that accuracy can look great while the false-negative rate is dangerous.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 23 Day 9 artifact.
Create experiments/foundations/m23_serving_safety.py with two parts. (1) Serving: simulate batching a generative workload, plotting throughput and per-request latency vs batch size and a GPU-second cost estimate. (2) Safety: given a labeled set that is mostly safe with a small unsafe minority, evaluate a filter and print BOTH overall accuracy and the false-negative rate = (unsafe passed)/(all unsafe). Show a filter with 99% accuracy can still have a 10%+ false-negative rate, and estimate harmful items shipped at scale.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m23/day-09-log.md</code>: what batch size gave the best throughput/latency balance, and how did overall accuracy compare to the false-negative rate?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "batch":{"btn":"① batching raises throughput","html":'''<span class="prompt">&gt;&gt;&gt;</span> serve(batch_size=1)   <span class="hl">→ 2 images/sec, low latency</span>
<span class="prompt">&gt;&gt;&gt;</span> serve(batch_size=8)   <span class="hl">→ 12 images/sec, higher latency</span>
<span class="dim"># bigger batch = better GPU use, but each user waits longer</span>''',
      "take":"<b>①  Batch to use the GPU.</b> Grouping requests raises throughput (images/sec) but makes each user wait a little longer to fill the batch."},
    "cost":{"btn":"② estimate the cost","html":'''<span class="prompt">&gt;&gt;&gt;</span> gpu_seconds = steps * per_step_cost
<span class="prompt">&gt;&gt;&gt;</span> cost = gpu_seconds * price_per_sec
<span class="hl">image ≈ $0.002   ·   2-sec video ≈ $0.1+</span>
<span class="dim"># video is far pricier — many frames × 50 steps</span>''',
      "take":"<b>②  Cost = GPU-seconds.</b> More steps and more frames mean more GPU time and more money. Video costs far more than a single image."},
    "filter":{"btn":"③ safety filter error rates","html":'''<span class="prompt">&gt;&gt;&gt;</span> evaluate(filter, unsafe_set=200)
<span class="hl">overall accuracy = 99%</span>   <span class="dim"># looks great…</span>
<span class="bad">false-negative rate = 20/200 = 10%</span>   <span class="dim"># …but 1 in 10 unsafe items slips</span>''',
      "take":"<b>③  Watch the false-negative rate.</b> Overall accuracy is dominated by the safe majority. The number that matters for safety is how many unsafe items slip through."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='18' width='60' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='70' y='37' fill='#5A544E'>req</text><rect x='110' y='18' width='60' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='140' y='37' fill='#5A544E'>req</text><rect x='180' y='18' width='60' height='30' rx='5' fill='#F5F0E8' stroke='#6B645E'/><text x='210' y='37' fill='#5A544E'>req</text><text x='320' y='37' fill='#6B645E'>many users at once</text></g></svg>",
     "note":"<b>Requests arrive.</b> Millions of users want images or videos. The serving system must handle them within a latency budget."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>batch requests → efficient GPU use</text></g></svg>",
     "note":"<b>Batch them.</b> Grouping requests uses the expensive GPU efficiently (higher throughput) at the cost of a little extra latency per user."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='16' width='300' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>generate: 50 steps (× frames for video)</text></g></svg>",
     "note":"<b>Generate.</b> The diffusion loop runs — 50 steps for an image, times many frames for video. This dominates the GPU-second cost."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='150' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='165' y='36' fill='#9A7208'>NSFW filter</text><rect x='260' y='16' width='200' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='360' y='36' fill='#C93B3B'>watch false-negative rate</text></g></svg>",
     "note":"<b>Safety filter.</b> Check output before it ships. The key metric is the false-negative rate — how much unsafe content slips through."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='16' width='150' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='135' y='36' fill='#1F6280'>watermark</text><rect x='230' y='16' width='150' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='305' y='36' fill='#1a5c38'>provenance (C2PA)</text><text x='430' y='36' fill='#6B645E'>→ 👤</text></g></svg>",
     "note":"<b>Mark and trace.</b> Watermark the output as AI-made and record provenance, then deliver to the user. Fast, cheap, and safe together."},
  ],
  quiz=[
    {"q":"1. What does batching do in a generative serving system?","opts":["it makes the model smaller","it runs many requests together to use the GPU efficiently, raising throughput but adding some latency","it removes the safety filter","it lowers image quality"],"ans":1,"fb":"Batching groups requests so the expensive GPU is used well (higher throughput), at the cost of each user waiting a bit longer to fill the batch."},
    {"q":"2. A filter is tested on 500 unsafe images and passes 25 of them. Its false-negative rate is:","opts":["25%","5%","50%","0.5%"],"ans":1,"fb":"false-negative rate = 25/500 = 0.05 = 5%. That is the fraction of unsafe content the filter fails to block."},
    {"q":"3. Why is reporting only a filter's overall accuracy dangerous?","opts":["accuracy is always wrong","accuracy is dominated by the safe majority, so it can look great (99%) while the false-negative rate on unsafe content is high","accuracy ignores latency","accuracy blocks safe content"],"ans":1,"fb":"With mostly-safe traffic, high overall accuracy hides a bad false-negative rate. You must measure the false-negative rate on unsafe items specifically."},
    {"q":"4. You make the safety pipeline more thorough (extra classifiers + human review). The direct cost is:","opts":["lower image quality","higher latency and cost per request","the watermark disappears","throughput becomes infinite"],"ans":1,"fb":"Thorough safety checks catch more harm but add latency and cost to every request. That is the latency/cost vs safety-thoroughness trade-off — usually worth paying."},
  ],
  fin={"em":"🚀","h3":"Day 9 complete — you can ship it responsibly!",
       "p":"You tied the module together: serve generative media with batching (throughput vs latency), reason about GPU-second cost, and add the safety layer — NSFW filters (watch the false-negative rate!), watermarking, and provenance. You saw why a leaky filter is a silent disaster and why safety is worth its latency/cost. Next: the <b>review gate</b> — prove you own all nine days."},
))
print("M23 built: 9 lessons")

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m23-review", title="Module 23 · Review Gate", nav_title="Spiral · M23 Review",
  eyebrow="Module 23 · The Gate — GenAI Systems", h1="Module 23 — Review Gate",
  lead="Checkpoint. The gate question: <b>can I take the generative building blocks (diffusion, retrieval, transformers) and assemble them into real image and text generation systems — and serve them safely?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. These nine systems are exactly the genAI design case studies interviewers probe — the full designs are one click away in each lesson's \"Go deeper\" link.",
  prev_href="day-09-serving-safety.html", prev_label="Day 9 · Serving & Safety Trade-offs",
  checks=[
    ["Day 1","I can explain the encoder-decoder / prefix-LM captioner, how an image embedding conditions text, and why captions hallucinate objects."],
    ["Day 2","I can trace the RAG loop (embed → retrieve → stuff → generate), compute a cosine similarity, and explain the retrieval-miss failure."],
    ["Day 3","I can explain diffusion faces, the identity-vs-diversity dial, and the training-face memorization (privacy) risk."],
    ["Day 4","I can explain why latent diffusion saves ~f² compute, describe cascades, and name the fine-detail artifact."],
    ["Day 5","I can write classifier-free guidance <code>ε̂ = ε_uncond + s·(ε_cond − ε_uncond)</code> and name the attribute-binding error."],
    ["Day 6","I can explain LoRA (2·d·r params) and DreamBooth, and the same-pose overfitting (likeness vs flexibility)."],
    ["Day 7","I can explain video as jointly-generated frames (frames = fps × seconds) and the inter-frame flicker failure."],
    ["Day 8","I can explain temporal attention + latent propagation, compute drift(t), and name identity drift over long clips."],
    ["Day 9","I can reason about latency/batching/cost and safety (NSFW filter, watermark, provenance), and compute a false-negative rate."],
  ],
  quiz=[
    {"q":"1. In a prefix-LM captioner, the image reaches the decoder as:","opts":["a reward signal","an image embedding prepended as a prefix, conditioning every generated word","a separate video stream","a retrieved document"],"ans":1,"fb":"The vision encoder makes an embedding; the prefix-LM prepends it so each word is conditioned on the image (Day 1)."},
    {"q":"2. q=[1,0], chunk=[0,1]. Their cosine similarity is:","opts":["1.0","0.0","-1.0","0.5"],"ans":1,"fb":"cos = (1·0 + 0·1)/(1·1) = 0. The vectors are perpendicular — unrelated in meaning (Day 2)."},
    {"q":"3. A 512×512 image with per-side compression f=8 gives a latent with how many positions per side, and what compute saving?","opts":["64 per side, ~64× saving","256 per side, ~4× saving","128 per side, ~16× saving","8 per side, ~8× saving"],"ans":0,"fb":"512/8 = 64 per side; area scales with the square so the saving is f² = 8² = 64× (Day 4)."},
    {"q":"4. With ε_uncond=0.2, ε_cond=0.4, guidance s=3, the CFG prediction ε̂ is:","opts":["0.6","0.8","1.4","0.2"],"ans":1,"fb":"ε̂ = 0.2 + 3·(0.4 − 0.2) = 0.2 + 3·0.2 = 0.2 + 0.6 = 0.8 (Day 5)."},
    {"q":"5. A safety filter passes 30 of 300 truly-unsafe images. Its false-negative rate is:","opts":["30%","10%","3%","90%"],"ans":1,"fb":"false-negative rate = 30/300 = 0.10 = 10% — the fraction of unsafe content that slips through, the number that matters for safety (Day 9)."},
  ],
  verdict_pass="you can assemble the generative building blocks into real systems — captioning, RAG, face and image generation, personalization, and video — and serve them fast, cheaply, and safely. Every lesson's \"Go deeper\" link takes you to the full genAI design case study for interview depth.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The through-line — <code>condition a generator on something</code> (an image, retrieved text, a prompt, a subject, a previous frame) then <code>serve it safely</code> — should feel familiar before you study the full design docs.",
  complete_label="Mark Module 23 complete",
  fin={"em":"🏆","h3":"Module 23 — passed!",
       "p":"You wrapped nine generative systems: image captioning, RAG, realistic faces, high-res synthesis, text-to-image, personalized headshots, image-to-video, temporal consistency, and serving + safety. You saw the recurring pattern — condition a generator, then serve it responsibly — and met every system's signature failure mode. The full genAI design case studies are now yours to explore in depth."},
  score_target=4,
))
print("M23 review built")
