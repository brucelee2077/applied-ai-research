# Building a RAG Pipeline

You know what embeddings are. You know how to chunk documents. You know how vector databases store and search them. You know the retrieval strategies. Now it is time to bolt everything together into a working system that actually answers questions.

**Before you start, you need to know:**
- What embeddings are and how cosine similarity works — covered in [what-is-rag.md](./what-is-rag.md)
- How to split documents into chunks — covered in [chunking-techniques.md](./chunking-techniques.md)
- How vector databases store and search embeddings — covered in [vector-databases.md](./vector-databases.md)

## The Analogy

Imagine you are a librarian. When someone walks in with a question:

1. You know which shelves to check (document loading and indexing)
2. You quickly find the right pages (vector search and retrieval)
3. You read the relevant paragraphs and craft a helpful answer (prompt engineering and generation)

A RAG pipeline does exactly this, step by step. Each component you learned in previous topics is one piece of the machine. This topic is about assembling the pieces.

### What the analogy gets right

Each step in the pipeline matches a real skill a librarian uses. The librarian's knowledge of the library layout is the index. Their ability to scan pages quickly is the vector search. Their ability to synthesize an answer from multiple sources is what the language model does.

### The concept in plain words

A RAG pipeline has two phases. The indexing phase runs once (or whenever documents change). The query phase runs every time someone asks a question.

**Indexing:** Load documents. Clean and preprocess them. Split into chunks. Convert each chunk into an embedding. Store everything in a vector database.

**Querying:** Take the user's question. Convert it to an embedding. Search the vector database for the closest matches. Pull out the top chunks. Build a prompt that combines the question with the retrieved chunks. Send it to the language model. Return the answer.

The pipeline is only as strong as its weakest link. Bad chunking means bad retrieval. Bad retrieval means bad answers.

### Where the analogy breaks down

A real librarian understands context deeply. They know when a question is ambiguous and can ask for clarification. A RAG pipeline follows a fixed sequence of steps and has no judgment about whether the retrieved chunks actually answer the question — unless you add evaluation and self-correction on top.

## The Five Components

### 1. Document Loading and Preprocessing

Raw documents come in many formats: PDFs, web pages, markdown files, databases. Before anything else, you need to extract the text and clean it up. This means normalizing whitespace, lowercasing, and removing characters that add noise without adding meaning.

### 2. Chunking

Split the cleaned text into smaller pieces. The chunking strategy you choose (fixed-size, overlapping, sentence-based, semantic, or recursive) determines the quality of everything downstream. Attach metadata to each chunk — document ID, source filename, chunk index, page number if available. This metadata is nearly free to store and extremely expensive to reconstruct later.

### 3. Embedding

Convert each chunk into a vector. For learning and prototyping, TF-IDF works. For production, use a dense embedding model like Sentence-BERT or OpenAI's text-embedding models. Dense embeddings understand meaning, not just word overlap — they know that "space rock" and "asteroid" are the same concept.

### 4. Vector Storage and Search

Store the embeddings in a vector database. When a query comes in, embed the query using the same model and search for the nearest neighbors. The search returns the chunks whose embeddings are most similar to the query embedding.

### 5. Prompt Engineering and Generation

Build a prompt that combines the user's question with the retrieved chunks. The prompt is the instruction manual you give to the language model. A good prompt makes the difference between a helpful answer and a hallucinated mess.

Different question types need different prompt structures:
- **Factual Q&A:** "Answer based ONLY on the provided context."
- **Summarization:** "Summarize the following information, citing sources."
- **Comparison:** "Compare and contrast these two items using the context."

## Evaluation

You cannot improve what you do not measure. The two most important retrieval metrics are:

- **Precision@k** — of the k chunks you retrieved, how many were actually relevant?
- **Recall@k** — of all the relevant chunks in the database, how many did you find?

Build a small evaluation set: 20-50 questions with known answers and the documents that contain those answers. Run your pipeline on these questions and measure precision and recall. This tells you where your pipeline is failing and what to fix first.

## Common Failure Modes

| Failure | What Happens | Fix |
|---------|-------------|-----|
| Vocabulary mismatch | User says "space rock," docs say "asteroid" | Better embeddings, query expansion with synonyms |
| Chunk too large | Answer buried in irrelevant text | Reduce chunk size |
| Chunk too small | Context split across chunks | Increase chunk size or overlap |
| Wrong document retrieved | Surface word match, not meaning match | Use dense embeddings instead of TF-IDF |
| Insufficient context | Top-k too low, missing information | Increase top-k, add re-ranking |

## Quick Check — Can You Answer These?

- What are the five components of a RAG pipeline?
- Why does the prompt template matter for answer quality?
- How would you measure whether your RAG system is retrieving the right chunks?

If you cannot answer one, go back and re-read that part. That is completely normal.

## Victory Lap

You now understand every component of a production RAG system and how they fit together. Document loading, chunking, embedding, vector storage, retrieval, prompt engineering, evaluation — you can trace a question from the moment a user types it to the moment they get an answer. That is not a toy understanding. That is the architecture behind every document Q&A system shipped by companies like Google, Anthropic, and OpenAI.

---

[Back to RAG module](./README.md)
