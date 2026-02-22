# Chunking Techniques

## What Is Chunking?

Imagine you have a 300-page textbook and a student asks: "What causes earthquakes?"
You wouldn't hand them the entire book and say "the answer is in here somewhere."
You'd flip to the right **chapter** or **paragraph** and point to it.

**Chunking** is exactly that -- splitting long documents into smaller, manageable
pieces so your RAG system can find and return the most relevant parts.

```
+-------------------------------------------------------------------+
|                    Why Chunking Matters                            |
|                                                                   |
|  Without chunking:                                                 |
|    Query: "What causes earthquakes?"                              |
|    Retrieved: [entire 300-page geology textbook]                  |
|    Problem: Too much text! The LLM can't process it all,          |
|    and most of it isn't relevant.                                  |
|                                                                   |
|  With chunking:                                                    |
|    Query: "What causes earthquakes?"                              |
|    Retrieved: [2 paragraphs about tectonic plate movement]        |
|    Result: Focused, relevant, fits in the LLM's context window.  |
+-------------------------------------------------------------------+
```

---

## The Chunking Tradeoff

Choosing chunk size is a balancing act:

```
+-------------------------------------------------------------------+
|              Chunk Size: The Goldilocks Problem                   |
|                                                                   |
|  TOO SMALL (e.g., single sentences):                              |
|    + Very precise retrieval                                       |
|    - Loses context ("it" refers to what?)                         |
|    - Needs many chunks for a full answer                          |
|                                                                   |
|  TOO BIG (e.g., entire chapters):                                 |
|    + Keeps full context                                           |
|    - Includes lots of irrelevant text                             |
|    - May not fit in LLM's context window                          |
|    - Embedding quality drops (too much meaning in one vector)     |
|                                                                   |
|  JUST RIGHT (e.g., 200-500 tokens):                               |
|    + Good balance of context and precision                        |
|    + Fits well in most LLM context windows                        |
|    + Embeddings capture a focused topic                           |
+-------------------------------------------------------------------+
```

---

## Chunking Strategies

### 1. Fixed-Size Chunking

The simplest approach: split text every N characters (or N tokens).

```
Original text (800 characters):
"The Earth's crust is made up of several large tectonic plates.
These plates float on the semi-fluid mantle below. When plates
collide, one may slide under the other in a process called
subduction. This can cause earthquakes and volcanic eruptions.
The Ring of Fire around the Pacific Ocean is where most of
the world's earthquakes occur. Scientists use seismographs
to measure earthquake intensity. The Richter scale, developed
in 1935, is the most famous measurement system..."

Fixed chunks of ~200 characters:
┌────────────────────────────────┐
│ Chunk 1: "The Earth's crust   │
│ is made up of several large   │
│ tectonic plates. These plates │
│ float on the semi-fluid..."   │
├────────────────────────────────┤
│ Chunk 2: "...mantle below.    │
│ When plates collide, one may  │
│ slide under the other in a    │
│ process called subduction..." │
├────────────────────────────────┤
│ Chunk 3: "...This can cause   │
│ earthquakes and volcanic      │
│ eruptions. The Ring of Fire..."│
└────────────────────────────────┘
```

**Pros:** Simple, predictable chunk sizes
**Cons:** Can cut sentences and ideas in half!

### 2. Fixed-Size with Overlap

Same as above, but chunks **overlap** by some amount. This ensures ideas
that span a chunk boundary aren't lost.

```
Chunk size: 200 chars, Overlap: 50 chars

  Chunk 1: [===========200 chars===========]
  Chunk 2:          [====50===][===========200 chars===========]
  Chunk 3:                              [====50===][===========200 chars=======]

The overlapping region (50 chars) appears in BOTH chunks.
If an important idea spans the boundary, at least one chunk
will have the full idea.
```

**Pros:** Doesn't lose ideas at boundaries
**Cons:** Some redundancy (same text in multiple chunks), slightly more storage

**Typical overlap:** 10-20% of chunk size (e.g., 200-char chunks with 40-char overlap)

### 3. Sentence-Based Chunking

Split on **sentence boundaries** instead of character counts. Group sentences
until you reach a target size.

```
Split by sentences, target ~3 sentences per chunk:

Chunk 1: "The Earth's crust is made up of several large tectonic
         plates. These plates float on the semi-fluid mantle below.
         When plates collide, one may slide under the other."

Chunk 2: "This can cause earthquakes and volcanic eruptions. The Ring
         of Fire around the Pacific Ocean is where most earthquakes
         occur. Scientists use seismographs to measure intensity."
```

**Pros:** Never cuts mid-sentence, more natural chunks
**Cons:** Variable chunk sizes, may still split related ideas

### 4. Paragraph-Based Chunking

Split on **paragraph boundaries** (double newlines). Each paragraph is a chunk.

**Pros:** Preserves the author's natural topic divisions
**Cons:** Paragraphs vary wildly in size (some are 1 sentence, others fill a page)

### 5. Semantic Chunking

The smartest approach: split based on **topic changes**. Use embeddings to
detect when the text shifts to a new topic.

```
+-------------------------------------------------------------------+
|                  How Semantic Chunking Works                      |
|                                                                   |
|  1. Split text into sentences                                     |
|  2. Compute embedding for each sentence                           |
|  3. Compare similarity of consecutive sentences                   |
|  4. When similarity drops significantly --> new chunk!            |
|                                                                   |
|  Sentence 1: "Earthquakes occur at plate boundaries"   ──┐       |
|  Sentence 2: "The plates move due to convection"        ──┤ High  |
|  Sentence 3: "This movement causes pressure to build"   ──┘ sim.  |
|  --- SIMILARITY DROP --- (topic change detected!)                 |
|  Sentence 4: "Scientists measure quakes with the Richter ──┐     |
|               scale"                                        │ New  |
|  Sentence 5: "The scale was developed in 1935 by           ──┘chunk|
|               Charles Richter"                                    |
+-------------------------------------------------------------------+
```

**Pros:** Best quality chunks, each chunk covers one coherent topic
**Cons:** More complex, requires running an embedding model, slower

### 6. Recursive Chunking

Try splitting by the largest boundary first (section headers), then paragraphs,
then sentences, then characters -- only going smaller if chunks are too big.

```
Try splitting by:     If chunk too big:
  1. Headers (##)  -->  2. Paragraphs (\n\n)  -->  3. Sentences (.)  -->  4. Characters
```

**Pros:** Preserves document structure, adapts to content
**Cons:** More complex to implement

This is the default strategy in LangChain's `RecursiveCharacterTextSplitter`.

---

## Chunking Strategy Comparison

| Strategy | Quality | Complexity | Best For |
|----------|---------|-----------|----------|
| Fixed-size | Low | Very simple | Quick prototypes |
| Fixed + overlap | Medium | Simple | Most use cases (good default) |
| Sentence-based | Medium | Simple | Clean, well-written text |
| Paragraph-based | Medium | Simple | Text with clear paragraphs |
| Semantic | High | Complex | Production RAG systems |
| Recursive | High | Medium | Structured documents (markdown, HTML) |

---

## Practical Tips

```
+-------------------------------------------------------------------+
|              Chunking Best Practices                              |
|                                                                   |
|  1. START SIMPLE: Fixed-size with overlap (200-500 tokens,        |
|     10-20% overlap) works for most cases                          |
|                                                                   |
|  2. MATCH YOUR EMBEDDING MODEL: If your embedding model was       |
|     trained on 512-token inputs, don't create 2000-token chunks   |
|                                                                   |
|  3. INCLUDE METADATA: Store the source document, page number,     |
|     section title, etc. with each chunk                           |
|                                                                   |
|  4. EXPERIMENT: Try different chunk sizes on your actual data     |
|     and measure retrieval quality                                 |
|                                                                   |
|  5. CONSIDER YOUR CONTENT: Code needs different chunking than     |
|     prose. Tables need different chunking than paragraphs         |
+-------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|               Chunking Cheat Sheet                               |
|                                                                  |
|  What:     Splitting documents into smaller pieces for RAG       |
|  Why:      LLMs have limited context, need focused content       |
|  Default:  Fixed-size with overlap (200-500 tokens)              |
|  Best:     Semantic chunking (topic-aware splitting)             |
|                                                                  |
|  Key rule: Chunk size should match your embedding model's        |
|  optimal input length (usually 256-512 tokens)                   |
+------------------------------------------------------------------+
```

---

[Back to RAG](../README.md)
