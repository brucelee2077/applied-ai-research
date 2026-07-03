# Day 25: Explain Attention in 60 Seconds

> You've built attention from scratch. Now practice explaining it — because in interviews, clear communication matters as much as knowledge.

## What You Need to Know First

- You've completed [Day 24: Your Transformer Writes](./day-24-milestone-your-transformer-writes.md)

---

## The Exercise

An interviewer asks: **"Explain how attention works in transformers."**

You have 60 seconds. Here's a framework:

### Level 1: The One-Sentence Version

> "Attention lets each word look at all other words in a sentence and decide which ones are relevant to it, producing a context-aware representation."

### Level 2: The 30-Second Version

> "Each word is projected into three representations — Query, Key, and Value. The Query asks 'what am I looking for?' The Keys say 'here's what I'm about.' We compute similarity between each Query and all Keys using a dot product, scale it, and apply softmax to get attention weights. Then we take a weighted sum of the Values. The result: each word's output carries information from every word it found relevant."

### Level 3: The 60-Second Version (Staff Level)

Add to Level 2:

> "Multi-head attention runs this in parallel across multiple heads, each learning different types of relationships — one might capture syntax, another semantics. We scale by sqrt(d_k) to prevent softmax saturation in high dimensions. The time complexity is O(n^2 d) where n is sequence length — which is why long documents are expensive. For generation, we apply a causal mask so words can only attend to past positions."

---

## Practice It

Say your explanation out loud. Time yourself. If you can do Level 2 comfortably, you're ahead of most candidates. Level 3 is staff-level.

Common follow-up questions an interviewer might ask:
1. "Why do you need three separate projections (Q, K, V) instead of just comparing embeddings directly?"
2. "Why divide by sqrt(d_k)?"
3. "What's the computational complexity of attention?"

You can answer all three — because you built each piece yourself.

---

## Check In

```bash
python days/check.py 25
```

---

**Tomorrow: Day 26 — The Cost of Attention.** You'll benchmark the O(n^2) complexity and understand why it's the main bottleneck for long documents.
