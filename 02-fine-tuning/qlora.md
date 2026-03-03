# QLoRA (Quantized LoRA)

## Fine-Tuning a 65B Model on a Single GPU

LoRA made fine-tuning cheap. But "cheap" is relative. To run LoRA on a 7-billion parameter model, you still need the full model in GPU memory — that is 14 GB just for the weights. For a 65B model, that is 130 GB. Still too much for most researchers.

Then a team at the University of Washington asked: what if we could shrink the frozen model itself? Not change it — just store it more efficiently, the way JPEG compresses photos without losing what matters.

That question led to **QLoRA**, and it changed who can fine-tune large models. A 65B model that normally needs 130 GB of GPU memory can now fit in 48 GB — fine-tunable on a single GPU.

---

**Before you start, you need to know:**
- What LoRA is and how it works — covered in [lora.md](./lora.md)
- Why LoRA freezes the base model — covered in [03_lora.ipynb](./03_lora.ipynb)
- What "parameters" are (the numbers inside a model)

---

## The Analogy: A Box of Crayons

Imagine you have a painting made with 16 million colors. It is beautiful and detailed. But storing all those colors takes a lot of space.

Now imagine replacing the 16 million colors with just 16 crayons. You pick the 16 crayons that best represent the original colors. Each patch of the painting gets assigned to its closest crayon. The painting looks almost identical — maybe 99% as good — but it takes up much less space.

**That is quantization.** You take a number that uses 16 bits of space and store it using only 4 bits. The number is not exactly the same, but it is close enough.

**QLoRA combines this with LoRA:**
1. Take the frozen model and compress it from 16 bits to 4 bits (quantization)
2. Add small LoRA adapters on top (these stay in full precision)
3. Train only the adapters, just like normal LoRA

```
  Standard LoRA:                    QLoRA:
  ┌──────────────────┐             ┌──────────────────┐
  │  Frozen Model    │             │  Frozen Model    │
  │  (16-bit)        │             │  (4-bit!)        │
  │  14 GB for 7B    │             │  3.5 GB for 7B   │
  │                  │             │                  │
  │  + LoRA adapters │             │  + LoRA adapters  │
  │  (16-bit, tiny)  │             │  (16-bit, tiny)  │
  └──────────────────┘             └──────────────────┘
     Total: ~16 GB                    Total: ~6 GB
```

**What this analogy gets right:**
- You are trading a tiny amount of quality for a big reduction in storage
- The important details (LoRA adapters) stay in full quality
- The compressed version still has the same knowledge — it is just stored more efficiently

**Where the analogy breaks down:** Real quantization does not just pick the nearest crayon. QLoRA uses a special data type called NF4 (4-bit NormalFloat) that is designed specifically for the distribution of neural network weights. It also uses "double quantization" — quantizing the quantization constants themselves — to squeeze out even more savings.

---

## Three Key Ideas in QLoRA

### 1. NF4 (4-bit NormalFloat)

Normal neural network weights follow a bell curve (normal distribution). NF4 is a 4-bit format designed specifically for this distribution. It spaces its 16 possible values so that each one represents an equal chunk of the bell curve. This gives better precision than naive 4-bit rounding.

### 2. Double Quantization

When you quantize weights, you need to store some small helper numbers (quantization constants) for each block of weights. Double quantization compresses these constants too, saving an extra 0.4 GB on a 65B model.

### 3. Paged Optimizers

When GPU memory gets tight, QLoRA can temporarily move optimizer states to CPU memory, like a computer's swap file. This prevents out-of-memory crashes during training.

---

## When to Use QLoRA

| Situation | Use LoRA or QLoRA? |
|-----------|-------------------|
| You have plenty of GPU memory (A100 80 GB) | LoRA is simpler |
| You have a consumer GPU (RTX 3090, 24 GB) | QLoRA — 4-bit model fits |
| You want to fine-tune a 65B+ model | QLoRA — only way it fits on one node |
| Maximum quality matters more than anything | LoRA (no quantization error) |
| You are experimenting and iterating quickly | QLoRA — faster to load, less memory |

---

**Quick check — can you answer these?**
- What does "quantization" mean in simple terms?
- How is QLoRA different from standard LoRA?
- Why does the base model get compressed but the LoRA adapters do not?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## What You Just Learned

QLoRA takes the LoRA idea one step further: compress the frozen model from 16 bits to 4 bits. The LoRA adapters still train in full precision. This cuts memory by another 3-4x on top of LoRA's savings, making it possible to fine-tune 65B models on a single GPU.

The QLoRA paper showed that despite the 4-bit compression, the fine-tuned model matches the quality of full 16-bit fine-tuning on most benchmarks. The quantization error is small, and the full-precision adapters compensate for any lost detail.
