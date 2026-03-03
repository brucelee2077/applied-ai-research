# What is RAG?

Have you ever asked a chatbot a question and gotten a confident but completely wrong answer? Large language models are powerful, but they have a big weakness: they can only use what they learned during training. Ask about your company's latest product, yesterday's news, or a private document — and they guess. Sometimes they guess wrong. RAG fixes this.

**Before you start, you need to know:**
- Basic Python (variables, functions, loops)
- No ML knowledge required — everything is explained from scratch

## The Analogy

Imagine you are a student about to take a big exam. There are two scenarios:

- **Closed-book test:** You can only use what you memorized before the test. If a question is about something you never studied, you have to guess. You might feel confident about a wrong answer because you misremember a fact.

- **Open-book test:** You have your notes, textbooks, and references right in front of you. You can look up specific facts to make sure you are right. You can reference the latest information.

Large language models are closed-book students. **RAG gives them an open book.**

### What the analogy gets right

Just like an open-book student can look up facts they do not remember, a RAG system looks up relevant documents before answering. The answers are grounded in real sources instead of guesses.

### The concept in plain words

RAG stands for **Retrieval-Augmented Generation**. Break it down:

- **Retrieval** — find relevant information from a collection of documents
- **Augmented** — add that information to the language model's input
- **Generation** — the language model writes an answer using both the question and the retrieved information

Without RAG, the model answers from memory alone. With RAG, it reads relevant documents first, then answers based on what it found. The model does not get smarter — it gets better reference material.

### Where the analogy breaks down

A real student can skim a textbook and understand it deeply. A RAG system does not understand anything in the human sense. It converts text into numbers, compares those numbers, and returns whatever is mathematically closest. If the numbers are bad, the retrieval is bad, even if the right answer is sitting in the database.

## How Computers Understand Text: Embeddings

Before a RAG system can find relevant documents, it needs a way to compare text. Computers do not understand words — they need numbers. An **embedding** is a list of numbers that captures the meaning of a piece of text.

Think of embeddings like GPS coordinates for meaning. Every place on Earth has coordinates (latitude, longitude). Places that are close together have similar coordinates. Embeddings work the same way, but for meaning instead of location.

- "King" and "Queen" end up close together in meaning-space because they are related concepts.
- "King" and "Banana" end up far apart because they have nothing in common.

Real embeddings use 384 to 1,536 numbers (dimensions) instead of just two. More dimensions means more nuance in capturing meaning.

## Measuring Similarity: Cosine Similarity

Once you have embeddings, you need a way to measure how close two of them are. The most common method is **cosine similarity**. It measures the angle between two vectors.

- A cosine similarity of **1.0** means identical meaning
- A cosine similarity of **0.0** means unrelated
- A cosine similarity of **-1.0** means opposite meaning

Cosine similarity only cares about direction, not length. A long document and a short document about the same topic will have high cosine similarity. This is why it is the most popular choice for RAG systems.

## The RAG Pipeline

RAG has two phases:

**Phase 1 — Indexing (happens once):**
1. Collect your documents (PDFs, web pages, notes)
2. Split them into smaller pieces called chunks
3. Convert each chunk into an embedding
4. Store all embeddings in a vector database

**Phase 2 — Querying (happens every time someone asks a question):**
1. Convert the question into an embedding
2. Search the vector database for the chunks with the most similar embeddings
3. Pull out the top 3-5 most relevant text chunks
4. Send the question plus the retrieved chunks to the language model
5. The language model reads everything and writes an answer

The language model does not need to have memorized your documents. It just needs to be good at reading and summarizing — which it already is.

## RAG vs Fine-Tuning

A common question: "Why not just fine-tune the model on my data instead?"

- **RAG** gives the model information at query time. It is cheaper, easier to update, and more transparent because you can show source documents. Start here.
- **Fine-tuning** changes the model's internal weights. It is better when you need to change the model's behavior or style, not just give it more information.

Start with RAG. Only fine-tune if RAG is not enough.

## Quick Check — Can You Answer These?

- In your own words: what problem does RAG solve?
- What is an embedding, and why do we need it?
- What are the two phases of a RAG pipeline, and when does each one run?

If you cannot answer one, go back and re-read that part. That is completely normal.

## Victory Lap

You just learned the core idea behind every modern AI search system. When you use ChatGPT with web search, when Google finds results for vague queries, when a company chatbot answers questions about internal documents — RAG is behind all of it. You now understand the pipeline: embed, store, search, retrieve, generate. That is the foundation everything else in this module builds on.

Ready to go deeper? The next step is learning how to split documents into searchable pieces. See [chunking-techniques.md](./chunking-techniques.md).

---

[Back to RAG module](./README.md)
