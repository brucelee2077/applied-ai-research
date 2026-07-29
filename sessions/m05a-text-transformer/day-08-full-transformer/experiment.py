# day-08-full-transformer — experiment
#
# Today's big idea in two lines of output:
#   The full transformer is only the parts you already built, wired in order:
#   on-ramp -> encoder stack -> decoder stack (+ cross-attention bridge) -> head -> loop.
#
# It builds a tiny end-to-end transformer (d_model=8, 2 encoder blocks, 2 decoder blocks,
# a 6-word vocabulary), pushes "hi there" through it, and loops the decoder until it draws
# <eos>. Every sub-layer is Pre-LN wrapped, as the lesson says: x + SubLayer(LayerNorm(x)).
# Run it:  python3 sessions/m05a-text-transformer/day-08-full-transformer/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@), and sin/cos

VOCAB = ["<start>", "hi", "there", "hello", "how", "<eos>"]   # the 6 words this model knows
EOS_ID = 5              # the "I'm done" word: when the loop picks it, generation stops
D_MODEL, D_FF, V = 8, 32, len(VOCAB)   # tower width, widened FFN width, vocabulary size
# The weights are RANDOM (this model was never trained), so the words it writes are babble.
# The seed is fixed so the pinned numbers in the self-check stay stable, and this seed was
# picked because the untrained model happens to draw <eos> at step 4 — the stop we watch.
SEED = 21


def softmax(scores):
    # Subtract each row's largest score first so exp() never blows up.
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)         # every row adds up to 1

def layer_norm(x, eps=1e-5):
    # Same rule as Day 2, with gamma = 1 and beta = 0 (their starting values, so they
    # change nothing) left out of the signature.
    # Re-level EACH word's own 8 numbers to mean 0, spread 1 (axis=-1 = one word at a time).
    mu, var = x.mean(axis=-1, keepdims=True), x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)             # eps only guards a divide by zero

def position_signal(max_len, d):
    # The seat stamp from the positional-encoding day (Module 4, Day 5): even sine, odd cosine.
    pe, seats = np.zeros((max_len, d)), np.arange(max_len)[:, None]
    div = np.power(10000.0, (2 * np.arange(d // 2)) / d)    # front pairs fast, back pairs slow
    pe[:, 0::2], pe[:, 1::2] = np.sin(seats / div), np.cos(seats / div)
    return pe

PE = position_signal(10, D_MODEL)        # enough seats for the longest reply we build

def build_weights(seed, gain=0.5):
    # Untrained weights. Each block gets its OWN draw: same shape, its own learned-in job.
    rng = np.random.default_rng(seed)
    def mat(fan_in, fan_out):   # 1/sqrt(fan_in) keeps a layer's output the size of its input
        return rng.normal(0.0, gain / np.sqrt(fan_in), (fan_in, fan_out))
    def qkv():        # three separate recipes, so a word's question is not its own key
        return {k: mat(D_MODEL, D_MODEL) for k in "QKV"}
    def feed():       # widen 8 -> 32, bend, shrink 32 -> 8; biases nonzero on purpose
        return {"W1": mat(D_MODEL, D_FF), "b1": rng.normal(0.0, 0.1, D_FF),
                "W2": mat(D_FF, D_MODEL), "b2": rng.normal(0.0, 0.1, D_MODEL)}
    return {"embed": rng.normal(0.0, 0.6, (V, D_MODEL)),
            "enc": [{"attn": qkv(), "ffn": feed()} for _ in range(2)],
            "dec": [{"attn": qkv(), "cross": qkv(), "ffn": feed()} for _ in range(2)],
            "Wout": mat(D_MODEL, V), "bout": rng.normal(0.0, 0.1, V)}

# ---- Part 1 pieces: the on-ramp ------------------------------------------------------
def embed(tokens, P):  return P["embed"][tokens]     # one row of 8 numbers per token
def add_position(x):   return x + PE[:len(x)]        # ADDED, so the width stays d_model

# ---- Part 2 pieces: the one brick (attention, feed-forward, and the two rails) -------
def attention(q_from, kv_from, W, mask=None):
    # Q asks, K advertises, V is handed over. q_from and kv_from are the SAME table for
    # self-attention and DIFFERENT tables for cross-attention — that is the only change.
    Q, K, Val = q_from @ W["Q"], kv_from @ W["K"], kv_from @ W["V"]
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])         # row i = how word i rates every word
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)     # future scores -> -inf, so share -> 0
    weights = softmax(scores)
    return weights @ Val, weights

def self_attention(x, W, mask=None):   return attention(x, x, W, mask)        # words share
def cross_attention(dec, enc_out, W):  return attention(dec, enc_out, W)      # the bridge
def sublayer(x, part):  return x + part(layer_norm(x))   # Pre-LN: re-level, run, add x back
def keep_mask(n):       return np.tril(np.ones((n, n), dtype=bool))  # True = self + earlier
# Days 5 and 6 built the causal mask as an ADDITIVE float grid (0 allowed, -inf blocked)
# and wrote `scores + mask`. This is the same rule in the other spelling: a boolean
# keep-list that np.where applies. Different name, because the polarity is inverted.
def ffn(x, W):          # each word thinks alone: widen, bend with ReLU, shrink back
    return np.maximum(0.0, x @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]

def encoder_block(x, P):                             # TWO sub-layers, no mask
    x = sublayer(x, lambda h: self_attention(h, P["attn"])[0])
    return sublayer(x, lambda h: ffn(h, P["ffn"]))

def decoder_block(x, enc_out, P):                    # THREE sub-layers: mask, bridge, think
    x = sublayer(x, lambda h: self_attention(h, P["attn"], keep_mask(len(h)))[0])
    x = sublayer(x, lambda h: cross_attention(h, enc_out, P["cross"])[0])
    return sublayer(x, lambda h: ffn(h, P["ffn"]))

# ---- Part 3 pieces: the two stacks · Part 4 piece: the output head -------------------
ENCODE_CALLS = [0]      # counts how many times the encoder is asked to read the input

def encode(tokens, P):
    ENCODE_CALLS[0] += 1                             # so Part 5 can PROVE it reads once
    x = add_position(embed(tokens, P))
    for block in P["enc"]:                           # the SAME brick, stacked
        x = encoder_block(x, block)
    return x

def decode(tokens, enc_out, P):
    x = add_position(embed(tokens, P))
    for block in P["dec"]:                           # same brick again, wired to write
        x = decoder_block(x, enc_out, block)
    return x

def output_head(vector, P):
    logits = vector @ P["Wout"] + P["bout"]          # one raw score per vocabulary word
    return softmax(logits), logits                   # softmax turns scores into shares of 1

def show(labels, matrix):
    for label, row in zip(labels, matrix):
        print("  %-8s %s" % (label, np.round(row, 4)))

def cosine(a, b):  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    P = build_weights(SEED)

    # --- Part 1: the on-ramp — words become numbers, then get stamped -----------------
    src_words = ["hi", "there"]
    src = [VOCAB.index(w) for w in src_words]                  # "hi there" -> token ids
    print("vocabulary:", VOCAB, " EOS is", VOCAB[EOS_ID])
    print("input tokens", src_words, "-> ids", src)
    src_emb = embed(src, P)                                    # word -> meaning numbers
    src_in = add_position(src_emb)                             # ...then stamp WHERE it sits
    print("embed(tokens)", src_emb.shape, "-> add_position(x)", src_in.shape,
          "= (seq_len, %d): one row per word" % D_MODEL)
    show(src_words, src_in)

    # --- Part 2: the one brick — re-level, run the part, add the input back -----------
    normed = layer_norm(src_in)
    print("\nlayer_norm, each word on its own: biggest |mean| %.1e" %
          float(np.abs(normed.mean(axis=-1)).max()), " spreads",
          np.round(normed.std(axis=-1), 4), "-> re-levelled, so scale cannot drift")
    one_block = encoder_block(src_in, P["enc"][0])
    print("one encoder block:", src_in.shape, "->", one_block.shape, "(width in = width out)")
    # The shortcut makes the output x PLUS a correction, so x's direction survives the block.
    # A bare number proves little on its own, so measure the SAME thing with the shortcut
    # cut out: run only the correction, and see how much of the input's direction is left.
    trace = min(cosine(src_in[i], one_block[i]) for i in range(len(src_in)))
    no_shortcut = ffn(self_attention(layer_norm(src_in), P["enc"][0]["attn"])[0],
                      P["enc"][0]["ffn"])            # the same two sub-layers, no "+ x"
    trace_without = min(cosine(src_in[i], no_shortcut[i]) for i in range(len(src_in)))
    print("closest cosine(block input, block output):", round(trace, 4),
          "· the same block with the shortcut cut out:", round(trace_without, 4),
          "-> the residual shortcut kept the input's trace")

    # --- Part 3: two stacks — the reader, then the writer with its bridge -------------
    enc_out = encode(src, P)
    print("\nENCODER stack (2 blocks, no mask):", src_in.shape, "->", enc_out.shape)
    reply_words = ["<start>", "how", "there"]                  # a reply-so-far, 3 words in
    reply = [VOCAB.index(w) for w in reply_words]
    dec_out = decode(reply, enc_out, P)
    print("DECODER stack (2 blocks, causal mask + bridge): (%d, %d) -> %s" %
          (len(reply), D_MODEL, dec_out.shape), "-> both stacks preserve the width")
    dec_x = layer_norm(add_position(embed(reply, P)))           # what block 1 sees inside
    _, self_w = self_attention(dec_x, P["dec"][0]["attn"], keep_mask(len(reply)))
    print("decoder self-attention shares, row i = what word i is allowed to look at:")
    show(reply_words, self_w)
    print("  above-diagonal total:", round(float(np.triu(self_w, 1).sum()), 12),
          "-> the causal mask gave every FUTURE word exactly 0 share")
    _, cross_w = cross_attention(dec_x, enc_out, P["dec"][0]["cross"])
    print("cross-attention shares, %d decoder rows x %d source columns (Q here, K/V there):"
          % cross_w.shape)
    show(reply_words, cross_w)
    # Predict, then observe (1): under the mask a row cannot depend on a word written after
    # it, so re-decoding a SHORTER prefix must reproduce the earlier rows exactly.
    mask_gap = float(np.abs(dec_out[:2] - decode(reply[:2], enc_out, P)).max())
    print("\nfirst 2 rows re-decoded from the shorter prefix -> gap", round(mask_gap, 12),
          "-> masked rows ignore the future word entirely")
    # Observe (2): the encoder wears no mask, so the same prediction must FAIL there.
    enc_gap = float(np.abs(enc_out - encode(src + [VOCAB.index("hello")], P)[:2]).max())
    print("encoder rows after adding a third source word -> gap", round(enc_gap, 4),
          "-> no mask, so old rows DID move: the encoder reads both ways")
    # Observe (3): the bridge is real — change the source and the writer's answer moves.
    other_out = encode([VOCAB.index("hello"), VOCAB.index("how")], P)
    bridge_gap = float(np.abs(dec_out - decode(reply, other_out, P)).max())
    print("same reply-so-far against a DIFFERENT source -> gap", round(bridge_gap, 4),
          "-> cross-attention really reads the encoder")

    # --- Part 4: the output head — top vector to a next-word chart --------------------
    top_vector = dec_out[-1]                                   # the LAST position's vector
    probs, logits = output_head(top_vector, P)
    print("\ntop vector (last decoder row):", np.round(top_vector, 4))
    print("logits (one raw score per word):", np.round(logits, 4))
    print("next-word chart, shape", probs.shape, "= one probability per vocabulary word:")
    for word, p in zip(VOCAB, probs):
        print("  %-8s %.4f %s" % (word, p, "#" * int(round(p * 40))))
    print("sum over ALL words:", round(float(probs.sum()), 6), "-> exactly one whole")
    print("winner:", VOCAB[int(np.argmax(probs))], "(greedy pick = the biggest bar)")

    # --- Part 5: the autoregressive loop — read once, then write word by word ---------
    max_new = 6
    predicted_lengths = [1 + k for k in range(max_new)]     # <start>, then one word per step
    print("\npredict: the decoder gets fed", predicted_lengths,
          "words as it loops, and stops early only when it picks", VOCAB[EOS_ID])
    enc_once = encode(src, P)                    # the encoder reads the input a single time
    encodes_before_loop = ENCODE_CALLS[0]        # a counter, so "reads once" is TESTED
    grown, generated, fed_lengths, rows_seen = [VOCAB.index("<start>")], [], [], []
    stop_reason = "hit the %d-word cap" % max_new
    for step in range(1, max_new + 1):
        fed_lengths.append(len(grown))           # how long the reply-so-far is this step
        state = decode(grown, enc_once, P)       # re-read the WHOLE reply-so-far
        rows_seen.append(len(state))             # one output row per word actually fed in
        chart, _ = output_head(state[-1], P)
        picked = int(np.argmax(chart))           # greedy: take the biggest bar
        generated.append(picked)
        grown.append(picked)                     # append it, and loop with a LONGER reply
        print("  step %d · fed %d word(s) %-34s -> picked %-8s p=%.4f" %
              (step, fed_lengths[-1], str([VOCAB[t] for t in grown[:-1]]),
               VOCAB[picked], chart[picked]))
        if picked == EOS_ID:
            stop_reason = "picked the EOS token"
            break
    print("  reply:", " ".join(VOCAB[t] for t in generated), "· stopped because it",
          stop_reason, "after", len(generated), "steps")
    encodes_in_loop = ENCODE_CALLS[0] - encodes_before_loop
    print("  the encoder ran", encodes_in_loop, "more time(s) inside the loop, while the",
          "decoder ran", len(fed_lengths), "-> read once, write many")

    # --- Self-check: one boolean per claim -------------------------------------------
    # Every expected number below was READ OFF a run and written down here, so a broken
    # change in the code above cannot quietly agree with itself.
    expected_self_row2 = np.array([0.2926, 0.2572, 0.4502])    # decoder row 2, printed above
    expected_cross_row0 = np.array([0.5304, 0.4696])           # cross-attention row 0
    expected_chart = np.array([0.0648, 0.3009, 0.1671, 0.0490, 0.3311, 0.0871])

    claims = {
        "on-ramp, encoder and decoder must all be (seq_len, 8), cross grid (3, 2)":
            src_in.shape == (2, D_MODEL) and enc_out.shape == (2, D_MODEL)
            and dec_out.shape == (3, D_MODEL) and cross_w.shape == (3, 2),
        "layer_norm must leave each word at mean 0 and spread 1":
            np.round(np.abs(normed.mean(axis=-1)).max(), 10) == 0.0
            and np.array_equal(np.round(normed.std(axis=-1), 4), np.ones(2)),
        "the residual shortcut should keep the input's trace: cosine 0.9589, and only "
        "0.0012 of it survives when the shortcut is cut out":
            round(trace, 4) == 0.9589 and round(trace_without, 4) == 0.0012
            and trace > trace_without,
        "every above-diagonal share must be exactly 0 (that is the causal mask)":
            float(np.triu(self_w, 1).sum()) == 0.0,
        "decoder self-attention row 2 should be [0.2926 0.2572 0.4502]":
            np.array_equal(np.round(self_w[2], 4), expected_self_row2),
        "cross-attention row 0 should be [0.5304 0.4696] over the 2 source words":
            np.array_equal(np.round(cross_w[0], 4), expected_cross_row0),
        "masked rows must be identical when re-decoded from the shorter prefix":
            mask_gap == 0.0,
        "the unmasked encoder's rows should move by 0.4055 in that same test":
            round(enc_gap, 4) == 0.4055,
        "a different source must move the decoder's answer by 1.2298":
            round(bridge_gap, 4) == 1.2298,
        "the chart should be [0.0648 0.3009 0.1671 0.049 0.3311 0.0871] and sum to 1.0":
            probs.shape == (V,) and round(float(probs.sum()), 6) == 1.0
            and np.array_equal(np.round(probs, 4), expected_chart),
        "the decoder must be fed one more word each step (and process every one)":
            fed_lengths == rows_seen == predicted_lengths[:len(generated)],
        # `stop_reason` is a label this script sets itself, so asserting it would only
        # restate the reply pin below (4 words under a cap of 6 already means it stopped
        # early). The pin is the claim; the label is only printed.
        "expected the reply 'how there how <eos>', ending on EOS before the 6-word cap":
            [VOCAB[t] for t in generated] == ["how", "there", "how", "<eos>"]
            and len(generated) < max_new,
        "the encoder must read the input once, OUTSIDE the loop: 0 more reads inside it":
            encodes_in_loop == 0 and len(fed_lengths) == 4,
    }

    if all(claims.values()):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected every stack to hand back (seq_len, 8), layer_norm rows "
              "at mean 0 / spread 1, block-trace cosine 0.9589 against 0.0012 with the "
              "shortcut cut out, zero share above the decoder "
              "diagonal with row 2 = [0.2926 0.2572 0.4502], cross-attention row 0 = "
              "[0.5304 0.4696] in a (3, 2) grid, a 0.0 gap when re-decoding the shorter "
              "prefix, a 0.4055 gap for the unmasked encoder, a 1.2298 gap when the source "
              "changes, the chart [0.0648 0.3009 0.1671 0.049 0.3311 0.0871] summing to 1.0, "
              "and the reply 'how there how <eos>' grown one word per step")
    for why, ok in claims.items():
        assert ok, why           # one assert per claim, so a wrong run stops the program
