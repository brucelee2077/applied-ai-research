"""Quiz questions for Transformers architecture module"""

QUESTIONS = [
    {
        "concept_id": "attn_scaled_dot_product",
        "module": "transformers",
        "question": "Why is the dot product in attention scaled by 1/√d_k?",
        "choices": [
            "A. To normalize the output to [0, 1]",
            "B. For large d_k, dot products grow large in magnitude, pushing softmax into regions with near-zero gradients. Scaling by 1/√d_k keeps the variance of the dot product at 1, maintaining healthy gradients.",
            "C. To match the learning rate",
            "D. To reduce memory usage"
        ],
        "correct": "B",
        "hint": "If Q and K have unit variance, what is the variance of their dot product for d_k dimensions?",
        "explanation": "If Q and K entries are drawn from N(0,1), their dot product has variance d_k (sum of d_k terms, each with variance 1). For large d_k (e.g., 64), dot products are large, softmax saturates near 1 for the max and near 0 for others → near-zero gradients. Dividing by √d_k restores unit variance. This is the 'scaled' in Scaled Dot-Product Attention.",
        "difficulty": 4,
        "tags": ["attention", "scaled_dot_product", "gradients"]
    },
    {
        "concept_id": "attn_qkv",
        "module": "transformers",
        "question": "In self-attention, Q, K, V all come from the same input X. What is the role of each?",
        "choices": [
            "A. Q = what I'm looking for, K = what I have to offer, V = the actual content I contribute if selected",
            "B. Q = query word, K = key word, V = value word",
            "C. All three are identical — the projections are redundant",
            "D. Q controls the output dimension, K controls the hidden dimension, V controls the input dimension"
        ],
        "correct": "A",
        "hint": "Think of a database lookup: you send a query, match against keys, and retrieve values.",
        "explanation": "Attention as soft lookup: Q (query) — 'what information am I looking for?', K (key) — 'what information do I contain?', V (value) — 'what do I actually send if selected?'. Compatibility score = Q·K^T / √d_k, softmax gives weights, output = weighted sum of V. Q, K, V are learned linear projections of X — allowing the model to separate 'what to look for' from 'what to share'.",
        "difficulty": 2,
        "tags": ["attention", "qkv", "self_attention"]
    },
    {
        "concept_id": "attn_multi_head",
        "module": "transformers",
        "question": "Why use multiple attention heads instead of one large attention head?",
        "choices": [
            "A. Multiple heads are faster to compute",
            "B. Each head learns to attend to different aspects simultaneously (syntactic, semantic, positional). A single head must represent all relationship types in one attention pattern — multiple heads partition the representation space.",
            "C. Multiple heads reduce memory usage",
            "D. The original paper found it slightly better by chance"
        ],
        "correct": "B",
        "hint": "Think about what different heads might specialize in — not all relationships between words are the same type.",
        "explanation": "Multi-head attention runs h parallel attention operations with projected subspaces (d_k = d_model/h per head). Each head can specialize: one head captures syntactic dependencies (subject-verb), another captures coreference (pronoun-antecedent), another attends to positional proximity. Single-head attention must represent all these relationship types simultaneously in one distribution — much harder. Concatenating h head outputs and projecting back gives richer combined representations.",
        "difficulty": 3,
        "tags": ["multi_head_attention", "specialization", "architecture"]
    },
    {
        "concept_id": "attn_complexity",
        "module": "transformers",
        "question": "What is the time and memory complexity of self-attention for a sequence of length n?",
        "choices": [
            "A. O(n) time, O(1) memory",
            "B. O(n²) time and memory — the attention matrix is n×n, requiring computing all pairwise token interactions",
            "C. O(n log n) time, O(n) memory",
            "D. O(n³) time, O(n²) memory"
        ],
        "correct": "B",
        "hint": "How big is the attention score matrix for a sequence of n tokens?",
        "explanation": "Self-attention computes Q·K^T: an n×n matrix where entry (i,j) is the attention score between token i and token j. Computing this requires O(n²·d) operations and O(n²) memory to store the attention matrix. For n=512 this is manageable; for n=100,000 (long documents) it becomes prohibitive. This is why efficient attention variants exist: Longformer (O(n·k) local), Linformer (O(n·r) low-rank), FlashAttention (O(n²) compute but O(n) memory via tiling).",
        "difficulty": 3,
        "tags": ["complexity", "self_attention", "efficiency"]
    },
    {
        "concept_id": "pos_encoding_why",
        "module": "transformers",
        "question": "Why does the Transformer need positional encoding when RNNs don't?",
        "choices": [
            "A. Transformers are larger models that need more features",
            "B. Self-attention is permutation-invariant — it treats all tokens equally regardless of position. Without positional encoding, 'dog bites man' and 'man bites dog' produce identical representations.",
            "C. Positional encoding improves gradient flow",
            "D. It reduces the need for large embedding dimensions"
        ],
        "correct": "B",
        "hint": "What does an RNN process inherently (by architecture) that a Transformer doesn't?",
        "explanation": "RNNs process tokens sequentially — position is implicitly encoded by the order of processing. The Transformer processes all tokens in parallel: self-attention between any two tokens is independent of their relative positions (it only depends on their content embeddings). Without positional encoding, shuffling the input sequence produces the same output. Sinusoidal positional encodings or learned position embeddings inject position information into the token representations.",
        "difficulty": 2,
        "tags": ["positional_encoding", "permutation_invariance", "transformer"]
    },
    {
        "concept_id": "pos_encoding_sinusoidal",
        "module": "transformers",
        "question": "Why did the original Transformer paper use sinusoidal positional encodings instead of learned embeddings?",
        "choices": [
            "A. Sinusoidal is faster to compute",
            "B. Sinusoidal allows extrapolation to sequence lengths not seen during training — the formula PE(pos, 2i) = sin(pos/10000^{2i/d}) generates a unique encoding for any position, even beyond max training length",
            "C. Learned embeddings don't work for transformers",
            "D. Sinusoidal requires fewer parameters"
        ],
        "correct": "B",
        "hint": "What happens to a learned position embedding table if you give it a position ID it has never seen?",
        "explanation": "Sinusoidal advantage: the formula generates deterministic encodings for any position — you never need to have seen that position at training. Learned embeddings: you have a lookup table of size max_seq_len — inputs longer than max_seq_len have no embedding. In practice, modern models (BERT, GPT) use learned positional embeddings and set a fixed max sequence length. Relative positional encodings (Shaw et al., RoPE, ALiBi) offer more flexibility.",
        "difficulty": 4,
        "tags": ["positional_encoding", "sinusoidal", "extrapolation"]
    },
    {
        "concept_id": "transformer_ffn",
        "module": "transformers",
        "question": "The Transformer block has both attention and a feed-forward network (FFN). What does the FFN contribute that attention doesn't?",
        "choices": [
            "A. The FFN handles position information",
            "B. Attention aggregates information across positions (mixing). FFN processes each position independently with a large MLP — applying non-linear transformations to create new features. This is where much of the model's 'knowledge' is stored.",
            "C. The FFN is redundant — attention is sufficient",
            "D. The FFN controls the attention weights"
        ],
        "correct": "B",
        "hint": "Attention mixes information across tokens. What operates on each token individually?",
        "explanation": "Attention = 'where to look' (cross-position information mixing). FFN = 'what to do with the info' (per-position non-linear transformation). The FFN has 4× expansion: d_model → 4·d_model → d_model. Research shows FFN layers store factual associations (key-value memories). Removing FFN layers degrades factual recall more than reasoning. The alternating attention+FFN pattern is core to how Transformers process and store information.",
        "difficulty": 3,
        "tags": ["ffn", "transformer_block", "architecture"]
    },
    {
        "concept_id": "transformer_layer_norm",
        "module": "transformers",
        "question": "Transformers use Layer Normalization, not Batch Normalization. Why?",
        "choices": [
            "A. LayerNorm is faster to compute",
            "B. BatchNorm requires a large batch to estimate population statistics and can't normalize variable-length sequences consistently. LayerNorm normalizes each token independently across its feature dimension — works with any batch size and variable-length input.",
            "C. BatchNorm causes gradient explosion in transformers",
            "D. LayerNorm requires fewer parameters"
        ],
        "correct": "B",
        "hint": "What does BatchNorm compute statistics across, and why is that a problem for sequences?",
        "explanation": "BatchNorm statistics are computed across the batch dimension — requires large batches for stable estimates and breaks with batch_size=1. For sequences, batch statistics mix different positions/lengths, which is semantically wrong. LayerNorm normalizes each token's d-dimensional vector independently: μ and σ computed over the d features of a single token. This is batch-size independent and position-independent — perfect for variable-length sequence processing.",
        "difficulty": 3,
        "tags": ["layer_norm", "batch_norm", "normalization"]
    },
    {
        "concept_id": "transformer_pre_post_norm",
        "module": "transformers",
        "question": "Modern transformers (GPT-3, LLaMA) use Pre-LN (normalize before attention) instead of original Post-LN. Why?",
        "choices": [
            "A. Pre-LN is more accurate on all benchmarks",
            "B. Pre-LN has more stable gradients during training — gradients flow through the residual stream without passing through the normalizer's scaling. Post-LN can cause unstable training requiring careful learning rate warmup.",
            "C. Post-LN requires more memory",
            "D. Pre-LN was chosen for aesthetic reasons"
        ],
        "correct": "B",
        "hint": "Where does the gradient flow differ between Pre-LN and Post-LN?",
        "explanation": "Original Transformer: Post-LN (sublayer → add residual → normalize). Modern: Pre-LN (normalize input → sublayer → add residual). Pre-LN gradient analysis: the residual path is clean — gradient flows directly backward through residual connections without going through LayerNorm scaling. This eliminates the need for careful warmup scheduling and enables training without warmup. Tradeoff: Pre-LN can have slightly lower peak performance, but training stability makes it the standard choice.",
        "difficulty": 4,
        "tags": ["layer_norm", "pre_norm", "training_stability"]
    },
    {
        "concept_id": "attention_masking",
        "module": "transformers",
        "question": "What is the difference between the padding mask and the causal (look-ahead) mask in Transformer training?",
        "choices": [
            "A. They are the same mask",
            "B. Padding mask: prevents attention to [PAD] tokens (variable-length batching). Causal mask: upper-triangular mask ensuring position i can only attend to positions ≤ i, enforcing autoregressive generation order.",
            "C. Padding mask is for encoders, causal mask is for convolutional layers",
            "D. Both masks are applied to the value matrix"
        ],
        "correct": "B",
        "hint": "Two different problems: (1) ignoring padding, (2) preventing future token leakage.",
        "explanation": "Padding mask: when batching sequences of different lengths, shorter sequences are padded to match the longest. Padding tokens should not influence attention. Implemented by setting attention logits for padding positions to -∞ before softmax. Causal mask: for decoder/autoregressive models, position t must not attend to positions t+1, t+2, ... (future tokens). Implemented as an upper-triangular -∞ mask. Both are applied simultaneously in causal decoder training.",
        "difficulty": 3,
        "tags": ["masking", "padding_mask", "causal_mask", "autoregressive"]
    },
]
