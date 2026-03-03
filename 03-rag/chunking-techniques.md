# Chunking Techniques

Here is something most people get wrong when building a RAG system: they focus all their energy on the language model and barely think about how they split their documents. But the way you chunk your text is one of the biggest factors in whether your system returns good answers or garbage. Get the chunking wrong, and even the best LLM in the world cannot help you.

**Before you start, you need to know:**
- What RAG is and why it needs external documents — covered in [what-is-rag.md](./what-is-rag.md)
- What an embedding is (a list of numbers that captures meaning) — covered in [what-is-rag.md](./what-is-rag.md)

## The Analogy

Imagine you have a 300-page textbook and a student asks: "What causes earthquakes?" You would not hand them the entire book and say "the answer is in here somewhere." You would flip to the right chapter or paragraph and point to it.

**Chunking** is exactly that — splitting long documents into smaller, searchable pieces so your RAG system can find and return the most relevant parts.

### What the analogy gets right

Just like you would pick the most relevant section of a textbook for a student, a RAG system picks the most relevant chunks to give the language model. The goal is the same: give focused, relevant information instead of everything at once.

### The concept in plain words

When a document is too long, you split it into smaller pieces called **chunks**. Each chunk gets turned into an embedding (a list of numbers). When a user asks a question, the system compares the question's embedding to all the chunk embeddings and finds the closest matches. Those matching chunks become the context the language model uses to answer.

The tricky part is deciding *where* to split. Split in the wrong place and you cut an idea in half. Split too small and each piece loses its context. Split too big and you include too much irrelevant text.

### Where the analogy breaks down

A human can skim and understand a whole chapter before picking the right paragraph. A chunking algorithm does not read or understand anything — it follows rules about where to cut, and some rules are smarter than others.

## The Goldilocks Problem

Choosing chunk size is a balancing act:

- **Too small** (single sentences): Very precise retrieval, but each piece loses context. The word "it" refers to what? You need many chunks to get a full answer.
- **Too big** (entire chapters): Keeps full context, but includes lots of irrelevant text. May not fit in the LLM's context window. The embedding tries to capture too many ideas at once and becomes blurry.
- **Just right** (200–500 tokens): Good balance of context and precision. Fits well in most LLM context windows. Each embedding captures a focused topic.

## Six Chunking Strategies

### 1. Fixed-Size Chunking

The simplest approach: split text every N characters (or N tokens).

**How it works:** Walk through the text and cut every 200 characters, regardless of what is there.

**Good:** Simple, predictable chunk sizes.
**Bad:** Can cut sentences and ideas in half.

### 2. Fixed-Size with Overlap

Same as above, but chunks overlap by some amount. If an important idea sits right at a boundary, at least one chunk will have the full idea.

**Typical overlap:** 10–20% of chunk size (e.g., 200-character chunks with 40-character overlap).

**Good:** Does not lose ideas at boundaries.
**Bad:** Some text appears in multiple chunks, which means slightly more storage.

### 3. Sentence-Based Chunking

Split on sentence boundaries instead of character counts. Group sentences until you reach a target size.

**Good:** Never cuts mid-sentence, more natural chunks.
**Bad:** Variable chunk sizes, may still split related ideas.

### 4. Paragraph-Based Chunking

Split on paragraph boundaries (double newlines). Each paragraph is a chunk.

**Good:** Preserves the author's natural topic divisions.
**Bad:** Paragraphs vary wildly in size — some are one sentence, others fill a page.

### 5. Semantic Chunking

The smartest approach: split based on topic changes. Use embeddings to detect when the text shifts to a new topic.

**How it works:**
1. Split text into sentences
2. Compute an embedding for each sentence
3. Compare similarity of consecutive sentences
4. When similarity drops a lot — that is a topic change — start a new chunk

**Good:** Best quality chunks, each chunk covers one coherent topic.
**Bad:** More complex, requires running an embedding model, slower.

### 6. Recursive Chunking

Try splitting by the largest boundary first (section headers), then paragraphs, then sentences, then characters — only going smaller if chunks are too big.

**Good:** Preserves document structure, adapts to content.
**Bad:** More complex to implement.

This is the default strategy in LangChain's `RecursiveCharacterTextSplitter`.

## Strategy Comparison

| Strategy | Quality | Complexity | Best For |
|----------|---------|-----------|----------|
| Fixed-size | Low | Very simple | Quick prototypes |
| Fixed + overlap | Medium | Simple | Most use cases (good default) |
| Sentence-based | Medium | Simple | Clean, well-written text |
| Paragraph-based | Medium | Simple | Text with clear paragraphs |
| Semantic | High | Complex | Production RAG systems |
| Recursive | High | Medium | Structured documents (markdown, HTML) |

## Quick Check — Can You Answer These?

- Why is chunk size a trade-off? What goes wrong if chunks are too small? Too big?
- What is the difference between fixed-size chunking and sentence-based chunking?
- How does semantic chunking detect where to split?

If you cannot answer one, go back and re-read that part. That is completely normal.

## Victory Lap

You just learned the step that separates good RAG systems from bad ones. Most tutorials skip straight to "put your documents in a vector database" without explaining how the splitting happens. Now you know six different strategies, their trade-offs, and when to use each. The next time you build a RAG system, you will make a deliberate choice about chunking instead of using the default and hoping for the best.

Ready to go deeper? The interview deep-dive covers the math behind embedding quality vs chunk size, failure modes in production, and complexity analysis of each strategy. See [chunking-techniques-interview.md](./chunking-techniques-interview.md).

---

[Back to RAG module](./README.md)
