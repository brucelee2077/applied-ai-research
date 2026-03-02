"""Quiz questions for RNN fundamentals module"""

QUESTIONS = [
    {
        "concept_id": "rnn_vanishing_gradient",
        "module": "rnn",
        "question": "Why does the vanilla RNN struggle to learn long-range dependencies?",
        "choices": [
            "A. RNNs are too slow to train",
            "B. Vanishing gradient: backpropagating through many time steps multiplies the same weight matrix repeatedly — if eigenvalues < 1, gradients shrink exponentially to zero, preventing learning of distant dependencies",
            "C. RNNs can only process fixed-length sequences",
            "D. The hidden state is too small"
        ],
        "correct": "B",
        "hint": "Think about what happens when you multiply a number < 1 by itself 100 times.",
        "explanation": "BPTT (Backpropagation Through Time) requires multiplying the recurrent weight matrix W at each step. If the spectral radius of W < 1, gradients shrink exponentially as they propagate back. After 50+ steps, the gradient from the early sequence is essentially zero — the model can't adjust early-time-step weights based on late-time-step errors. LSTM/GRU solve this with additive updates through gating.",
        "difficulty": 3,
        "tags": ["vanishing_gradient", "bptt", "rnn"]
    },
    {
        "concept_id": "rnn_hidden_state",
        "module": "rnn",
        "question": "In a vanilla RNN, what does the hidden state h_t represent?",
        "choices": [
            "A. The raw input at time step t",
            "B. A compressed summary of all inputs seen up to time t, computed as h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)",
            "C. The output prediction at time t",
            "D. The gradient at time t"
        ],
        "correct": "B",
        "hint": "The hidden state is the 'memory' of the network.",
        "explanation": "The hidden state h_t is a learned, fixed-size summary of the input sequence up to time t. It is the recurrent 'memory' that carries information forward. The tanh nonlinearity squashes values to [-1,1], providing bounded activations. The key limitation: this fixed-size vector must compress arbitrarily long context — impossible for very long sequences.",
        "difficulty": 1,
        "tags": ["hidden_state", "rnn_basics"]
    },
    {
        "concept_id": "rnn_bptt_truncation",
        "module": "rnn",
        "question": "Truncated BPTT limits backpropagation to k steps. What is the tradeoff?",
        "choices": [
            "A. Truncated BPTT has no tradeoff — it's always better",
            "B. Tradeoff: reduces memory/compute cost (no full sequence unrolling) but prevents the model from learning dependencies longer than k steps — a practical approximation for very long sequences",
            "C. Truncated BPTT increases gradient explosion risk",
            "D. It requires twice as much memory"
        ],
        "correct": "B",
        "hint": "What happens to dependencies longer than k steps if you stop backprop at k?",
        "explanation": "Full BPTT over a 10,000-step sequence requires storing all activations (huge memory) and propagating gradients through 10,000 matrix multiplications (slow + vanishing). Truncated BPTT unrolls only k steps (typically 20-200), making training feasible but sacrificing the ability to learn dependencies spanning > k steps. The hidden state still carries forward information, but the gradient signal for long-range dependencies is cut off.",
        "difficulty": 3,
        "tags": ["bptt", "truncated_bptt", "training"]
    },
    {
        "concept_id": "rnn_teacher_forcing",
        "module": "rnn",
        "question": "What is teacher forcing in RNN training and what problem does it cause?",
        "choices": [
            "A. Using a teacher model to initialize weights",
            "B. At training time, feeding the ground-truth previous token as input (instead of the model's own prediction). Causes train-inference discrepancy: at inference, the model receives its own (potentially wrong) predictions, creating error accumulation.",
            "C. Forcing the model to attend to all positions",
            "D. A regularization technique to prevent overfitting"
        ],
        "correct": "B",
        "hint": "During training you give the model the correct answer. During inference, who gives it the input?",
        "explanation": "Teacher forcing: at training step t, feed ground truth token y_{t-1} as input to predict y_t. Faster, more stable training. Problem: exposure bias — the model never sees its own wrong predictions at training time, but at inference, errors compound (a wrong prediction at step 5 becomes the input for step 6). Solution: scheduled sampling (gradually replace ground truth with model predictions during training).",
        "difficulty": 3,
        "tags": ["teacher_forcing", "exposure_bias", "sequence_generation"]
    },
    {
        "concept_id": "rnn_exploding_gradient",
        "module": "rnn",
        "question": "How do you handle exploding gradients in RNN training?",
        "choices": [
            "A. Use a smaller learning rate only",
            "B. Gradient clipping: if the gradient norm exceeds a threshold θ, rescale the gradient to have norm θ. Simple, effective, standard practice for RNN training.",
            "C. Use dropout on the recurrent connections",
            "D. Reduce the hidden state size"
        ],
        "correct": "B",
        "hint": "Vanishing gradients → LSTM. Exploding gradients → ?",
        "explanation": "Gradient clipping by norm: compute ||g||, if > θ then g = θ * g / ||g||. This prevents destabilizing parameter updates while preserving gradient direction. Threshold θ is typically 1.0 or 5.0. Unlike vanishing gradients (which require architectural solutions like LSTM), exploding gradients can be fixed with this simple trick. PyTorch: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0).",
        "difficulty": 2,
        "tags": ["exploding_gradient", "gradient_clipping", "training"]
    },
    {
        "concept_id": "lstm_cell_state",
        "module": "rnn",
        "question": "What is the key architectural innovation in LSTM that solves the vanishing gradient problem?",
        "choices": [
            "A. Using larger hidden states",
            "B. The cell state C_t: a separate memory highway that flows through time with only additive updates (no repeated matrix multiplication). Gradients flow back through this highway without shrinking.",
            "C. Using sigmoid instead of tanh",
            "D. Adding residual connections"
        ],
        "correct": "B",
        "hint": "What flows through an LSTM without being repeatedly multiplied by a weight matrix?",
        "explanation": "The LSTM cell state is updated additively: C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t. The forget gate f_t can be set close to 1 (keep memory), allowing the cell state to pass information across many time steps without the gradient shrinking. Contrast with vanilla RNN: h_t = tanh(W * h_{t-1} + ...) — the repeated W multiplication causes exponential gradient shrinkage.",
        "difficulty": 3,
        "tags": ["lstm", "cell_state", "vanishing_gradient"]
    },
    {
        "concept_id": "lstm_gates",
        "module": "rnn",
        "question": "What are the three gates in an LSTM and what does each control?",
        "choices": [
            "A. Input, output, recurrent — control data flow in three directions",
            "B. Forget gate (how much of C_{t-1} to keep), input gate (how much new info to write to C_t), output gate (how much of C_t to expose as h_t)",
            "C. Read, write, erase — like a differentiable memory",
            "D. Encode, decode, attend — for sequence-to-sequence tasks"
        ],
        "correct": "B",
        "hint": "Three gates, three jobs: what to forget, what to remember new, what to output.",
        "explanation": "LSTM gates (all sigmoid, output 0-1): Forget gate f_t = σ(W_f·[h_{t-1}, x_t] + b_f) — how much of old cell state to retain. Input gate i_t = σ(W_i·[h_{t-1}, x_t] + b_i) — how much of new candidate to write. Output gate o_t = σ(W_o·[h_{t-1}, x_t] + b_o) — how much of cell state to output as hidden state. Cell update: C_t = f_t⊙C_{t-1} + i_t⊙tanh(W_c·[h_{t-1}, x_t]).",
        "difficulty": 3,
        "tags": ["lstm", "gates", "architecture"]
    },
    {
        "concept_id": "gru_vs_lstm",
        "module": "rnn",
        "question": "When would you choose GRU over LSTM?",
        "choices": [
            "A. GRU is always better",
            "B. GRU has fewer parameters (2 gates vs 3, no separate cell state), trains faster on small-medium datasets. LSTM's extra capacity helps on tasks requiring nuanced memory control. GRU wins when data is limited or speed matters.",
            "C. GRU handles longer sequences better",
            "D. GRU is better for image data"
        ],
        "correct": "B",
        "hint": "GRU = simplified LSTM. When does simpler win?",
        "explanation": "GRU merges the forget and input gates into a single 'update gate', and merges cell state and hidden state — fewer parameters. On small datasets, GRU's lower capacity reduces overfitting. On tasks not requiring fine-grained memory control, GRU matches LSTM performance at lower compute cost. Empirically: for language modeling and NLP, performance is similar; LSTM slightly edges out GRU on very long-range dependencies.",
        "difficulty": 3,
        "tags": ["gru", "lstm", "model_selection"]
    },
    {
        "concept_id": "bidirectional_rnn",
        "module": "rnn",
        "question": "Why can't you use a bidirectional RNN for autoregressive text generation?",
        "choices": [
            "A. Bidirectional RNNs are too slow",
            "B. Autoregressive generation produces tokens one at a time — at token t, future tokens don't exist yet. A bidirectional RNN requires seeing the full sequence first, making it fundamentally incompatible with left-to-right generation.",
            "C. Bidirectional RNNs can't process text",
            "D. They require too much memory"
        ],
        "correct": "B",
        "hint": "What does the backward RNN in a BiRNN need as input?",
        "explanation": "Bidirectional RNNs run one RNN forward (left→right) and one backward (right→left), concatenating hidden states. This requires the complete sequence — impossible for generation where the future is unknown. BiRNNs excel for classification/labeling tasks (sentiment, NER, POS tagging) where the full sequence is available. For generation, use unidirectional or autoregressive transformer (causal masking).",
        "difficulty": 2,
        "tags": ["bidirectional_rnn", "generation", "architecture"]
    },
    {
        "concept_id": "sequence_to_sequence",
        "module": "rnn",
        "question": "In a seq2seq RNN (encoder-decoder), what is the information bottleneck problem and how does attention solve it?",
        "choices": [
            "A. The encoder is too slow",
            "B. The encoder must compress the entire source sentence into a single fixed-size context vector — long sentences lose information. Attention solves this by allowing the decoder to directly query all encoder hidden states at each decoding step.",
            "C. The decoder generates too many tokens",
            "D. The vocabulary is too large"
        ],
        "correct": "B",
        "hint": "A 100-word sentence compressed into a 256-dim vector — what gets lost?",
        "explanation": "Classic seq2seq bottleneck: the encoder's final hidden state must summarize the entire input. For long inputs, early tokens are 'forgotten' by the time the encoder finishes. Bahdanau attention (2015) fixes this: at each decoder step, compute attention weights over all encoder hidden states, creating a dynamic context vector. The decoder 'looks back' at relevant encoder positions — the foundation of the Transformer's self-attention.",
        "difficulty": 4,
        "tags": ["seq2seq", "attention", "encoder_decoder", "bottleneck"]
    },
]
