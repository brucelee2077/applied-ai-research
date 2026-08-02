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
D_MODEL, V = 8, len(VOCAB)             # tower width, vocabulary size
D_FF = 4 * D_MODEL      # the widened FFN width: the SAME 4x rule days 3 and 4 called HIDDEN
# Two words this file uses that earlier days spelled differently, so nothing goes missing:
# the softmax output is called a SHARE here and a "weight" on days 5 and 6 (one number, two
# names), and the repeated unit is called a BRICK here and a "block" on day 4 — the
# identifiers stay `encoder_block` / `decoder_block` on purpose.
# The weights are RANDOM (this model was never trained), so the words it writes are babble.
# The seed is fixed so the pinned numbers in the self-check stay stable, and this seed was
# picked because the untrained model happens to draw <eos> at step 4 — the stop we watch.
SEED = 21


def softmax(scores):
    # Subtract each row's largest score first so exp() never blows up.
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)         # every row adds up to 1

def layer_norm(x, *, eps=1e-5):
    # Same rule as Day 2, with gamma = 1 and beta = 0 (their starting values, so they
    # change nothing) left out of the signature. Day 2's SECOND positional slot was gamma,
    # so `eps` here is keyword-only on purpose: a Day-2-style `layer_norm(x, 1.0)` has to
    # raise instead of silently handing a learned gain to the numerical-safety crumb.
    # Re-level EACH word's own 8 numbers to mean 0, spread 1 (axis=-1 = one word at a time).
    # "Spread 1" means 1 to the decimals printed; Day 2 pinned the exact hair below 1
    # (0.999999925) that eps inside the root leaves behind.
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
        # NEW today: days 3 and 4 taught this feed-forward with matrices only, no biases.
        # b1 and b2 are added here because a real transformer has them; nothing about the
        # widen-bend-shrink story changes, the two shifts just ride along.
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
def attention_with_shares(q_from, kv_from, W, mask=None):
    # Q asks, K advertises, V is handed over. q_from and kv_from are the SAME table for
    # self-attention and DIFFERENT tables for cross-attention — that is the only change.
    # NAMED IN FULL on purpose: Day 4's `attention(x)` took ONE table and returned ONE
    # array, so it could be dropped straight into `sublayer`. This one takes two tables
    # plus weights and hands back a PAIR (output, shares), which is why every call site
    # below ends in `[0]`. Different object, different name.
    Q, K, Val = q_from @ W["Q"], kv_from @ W["K"], kv_from @ W["V"]
    # The 1/sqrt(d_k) scaling module m03 introduced (Q.shape[1] is the head width). Days 5
    # and 6 worked from hand-written score grids, so this division never appeared there;
    # here the scores are computed, so it has to.
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])         # row i = how word i rates every word
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)     # future scores -> -inf, so share -> 0
    weights = softmax(scores)                        # days 5 and 6 called these "weights"
    return weights @ Val, weights

def self_attention(x, W, mask=None):   return attention_with_shares(x, x, W, mask)   # share
def cross_attention(dec, enc_out, W):  return attention_with_shares(dec, enc_out, W)  # bridge
def sublayer(x, part):  return x + part(layer_norm(x))   # Pre-LN: re-level, run, add x back
def keep_mask(n):       return np.tril(np.ones((n, n), dtype=bool))  # True = self + earlier
# Days 5 and 6 built the causal mask as an ADDITIVE float grid (0 allowed, -inf blocked)
# and wrote `scores + mask`. This is the same rule in the other spelling: a boolean
# keep-list that np.where applies. Different name, because the polarity is inverted.
def ffn(x, W):          # each word thinks alone: widen, bend with ReLU, shrink back
    # Day 3 called this `feed_forward(x, W1, W2)` and Day 4 called it `ffn(x)` with the
    # matrices at module level; here the same three steps read their weights out of a dict.
    return np.maximum(0.0, x @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]

def encoder_block(x, P):                             # TWO sub-layers, no mask
    x = sublayer(x, lambda h: self_attention(h, P["attn"])[0])
    return sublayer(x, lambda h: ffn(h, P["ffn"]))

# Day 6 showed the causal mask is the whole switch INSIDE one self-attention layer. A full
# decoder differs from an encoder in a second way as well, and it is right here: a THIRD
# sub-layer, the cross-attention bridge, which an encoder has no use for.
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
    # One labelled row per line. The CALLER rounds first and keeps the rounded grid in a
    # name, so the numbers printed here are the same values the self-check tests. The
    # lines are returned too: this one helper draws three different grids, so a slip
    # inside it would quietly change all three at once.
    lines = ["  %-8s %s" % (label, row) for label, row in zip(labels, matrix)]
    for line in lines:
        print(line)
    return lines

def cosine(a, b):  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    P = build_weights(SEED)

    # --- Part 1: the on-ramp — words become numbers, then get stamped -----------------
    # Every printed number below is bound to a name FIRST and then read again by the
    # self-check, so the line you read and the line that is tested are one single value.
    src_words = ["hi", "there"]
    src = [VOCAB.index(w) for w in src_words]                  # "hi there" -> token ids
    shown_eos_word = VOCAB[EOS_ID]
    print("vocabulary:", VOCAB, " EOS is", shown_eos_word)
    print("input tokens", src_words, "-> ids", src)
    src_emb = embed(src, P)                                    # word -> meaning numbers
    src_in = add_position(src_emb)                             # ...then stamp WHERE it sits
    shown_emb_shape, shown_in_shape = src_emb.shape, src_in.shape
    shown_src_in = np.round(src_in, 4)
    print("embed(tokens)", shown_emb_shape, "-> add_position(x)", shown_in_shape,
          "= (seq_len, %d): one row per word" % D_MODEL)
    src_in_lines = show(src_words, shown_src_in)

    # --- Part 2: the one brick — re-level, run the part, add the input back -----------
    normed = layer_norm(src_in)
    shown_norm_mean = float(np.abs(normed.mean(axis=-1)).max())
    shown_spreads = np.round(normed.std(axis=-1), 4)
    print("\nlayer_norm, each word on its own: biggest |mean| %.1e" % shown_norm_mean,
          " spreads", shown_spreads, "-> re-levelled, so scale cannot drift")
    one_block = encoder_block(src_in, P["enc"][0])
    shown_block_shape = one_block.shape
    print("one encoder block:", shown_in_shape, "->", shown_block_shape,
          "(width in = width out)")
    # The shortcut makes the output x PLUS a correction, so x's direction survives the block.
    # That correction is the same object Day 1 called F(x) — "the residual" — and Day 4
    # called "one block's change"; the lane it is added to is the residual stream.
    # A bare number proves little on its own, so measure the SAME thing with the shortcut
    # cut out: run only the correction, and see how much of the input's direction is left.
    trace = min(cosine(src_in[i], one_block[i]) for i in range(len(src_in)))
    no_shortcut = ffn(self_attention(layer_norm(src_in), P["enc"][0]["attn"])[0],
                      P["enc"][0]["ffn"])            # the same two sub-layers, no "+ x"
    trace_without = min(cosine(src_in[i], no_shortcut[i]) for i in range(len(src_in)))
    shown_trace, shown_trace_without = round(trace, 4), round(trace_without, 4)
    print("closest cosine(block input, block output):", shown_trace,
          "· the same block with the shortcut cut out:", shown_trace_without,
          "-> the residual shortcut kept the input's trace")

    # --- Part 3: two stacks — the reader, then the writer with its bridge -------------
    enc_out = encode(src, P)
    shown_enc_out_shape = enc_out.shape
    print("\nENCODER stack (2 blocks, no mask):", shown_in_shape, "->", shown_enc_out_shape)
    reply_words = ["<start>", "how", "there"]                  # a reply-so-far, 3 words in
    reply = [VOCAB.index(w) for w in reply_words]
    dec_out = decode(reply, enc_out, P)
    shown_dec_in_shape, shown_dec_out_shape = (len(reply), D_MODEL), dec_out.shape
    print("DECODER stack (2 blocks, causal mask + bridge): (%d, %d) -> %s" %
          (shown_dec_in_shape + (shown_dec_out_shape,)), "-> both stacks preserve the width")
    dec_x = layer_norm(add_position(embed(reply, P)))           # what block 1 sees inside
    _, self_w = self_attention(dec_x, P["dec"][0]["attn"], keep_mask(len(reply)))
    shown_self_w = np.round(self_w, 4)
    print("decoder self-attention shares, row i = what word i is allowed to look at:")
    self_w_lines = show(reply_words, shown_self_w)
    future_share = float(np.triu(self_w, 1).sum())
    shown_future_share = round(future_share, 12)
    print("  above-diagonal total:", shown_future_share,
          "-> the causal mask gave every FUTURE word exactly 0 share")
    _, cross_w = cross_attention(dec_x, enc_out, P["dec"][0]["cross"])
    shown_cross_shape, shown_cross_w = cross_w.shape, np.round(cross_w, 4)
    print("cross-attention shares, %d decoder rows x %d source columns (Q here, K/V there):"
          % shown_cross_shape)
    cross_w_lines = show(reply_words, shown_cross_w)
    # Predict, then observe (1): under the mask a row cannot depend on a word written after
    # it, so re-decoding a SHORTER prefix must reproduce the earlier rows exactly.
    mask_gap = float(np.abs(dec_out[:2] - decode(reply[:2], enc_out, P)).max())
    shown_mask_gap = round(mask_gap, 12)
    print("\nfirst 2 rows re-decoded from the shorter prefix -> gap", shown_mask_gap,
          "-> masked rows ignore the future word entirely")
    # Observe (2): the encoder wears no mask, so the same prediction must FAIL there.
    enc_gap = float(np.abs(enc_out - encode(src + [VOCAB.index("hello")], P)[:2]).max())
    shown_enc_gap = round(enc_gap, 4)
    print("encoder rows after adding a third source word -> gap", shown_enc_gap,
          "-> no mask, so old rows DID move: the encoder reads both ways")
    # Observe (3): the bridge is real — change the source and the writer's answer moves.
    other_out = encode([VOCAB.index("hello"), VOCAB.index("how")], P)
    bridge_gap = float(np.abs(dec_out - decode(reply, other_out, P)).max())
    shown_bridge_gap = round(bridge_gap, 4)
    print("same reply-so-far against a DIFFERENT source -> gap", shown_bridge_gap,
          "-> cross-attention really reads the encoder")

    # --- Part 4: the output head — top vector to a next-word chart --------------------
    top_vector = dec_out[-1]                                   # the LAST position's vector
    # CONVENTION, stated because Day 4 proved a Pre-LN lane exits un-normalised and growing:
    # this toy hands that raw lane vector straight to the head, with no final layer norm.
    # Production Pre-LN towers (GPT-2 onward) add one last layer_norm here for exactly the
    # reason Day 4 measured. It is left out so the wiring stays the four parts you built.
    probs, logits = output_head(top_vector, P)
    shown_top_vector, shown_logits = np.round(top_vector, 4), np.round(logits, 4)
    shown_probs_shape = probs.shape
    print("\ntop vector (last decoder row):", shown_top_vector)
    print("logits (one raw score per word):", shown_logits)
    print("next-word chart, shape", shown_probs_shape,
          "= one probability per vocabulary word:")
    # The bar and the number are rendered ONCE, into a list the self-check reads back.
    chart_lines = ["  %-8s %.4f %s" % (word, p, "#" * int(round(p * 40)))
                   for word, p in zip(VOCAB, probs)]
    for line in chart_lines:
        print(line)
    shown_prob_sum = round(float(probs.sum()), 6)
    shown_winner = VOCAB[int(np.argmax(probs))]
    print("sum over ALL words:", shown_prob_sum, "-> exactly one whole")
    print("winner:", shown_winner, "(greedy pick = the biggest bar)")

    # --- Part 5: the autoregressive loop — read once, then write word by word ---------
    max_new = 6
    predicted_lengths = [1 + k for k in range(max_new)]     # <start>, then one word per step
    print("\npredict: the decoder gets fed", predicted_lengths,
          "words as it loops, and stops early only when it picks", shown_eos_word)
    enc_once = encode(src, P)                    # the encoder reads the input a single time
    encodes_before_loop = ENCODE_CALLS[0]        # a counter, so "reads once" is TESTED
    grown, generated, fed_lengths, rows_seen = [VOCAB.index("<start>")], [], [], []
    step_lines = []
    stop_reason = "hit the %d-word cap" % max_new
    for step in range(1, max_new + 1):
        fed_lengths.append(len(grown))           # how long the reply-so-far is this step
        state = decode(grown, enc_once, P)       # re-read the WHOLE reply-so-far
        rows_seen.append(len(state))             # one output row per word actually fed in
        chart, _ = output_head(state[-1], P)
        picked = int(np.argmax(chart))           # greedy: take the biggest bar
        generated.append(picked)
        grown.append(picked)                     # append it, and loop with a LONGER reply
        # Rendered once, into a list, so the step line the reader sees is the one tested.
        step_lines.append("  step %d · fed %d word(s) %-34s -> picked %-8s p=%.4f" %
                          (step, fed_lengths[-1], str([VOCAB[t] for t in grown[:-1]]),
                           VOCAB[picked], chart[picked]))
        print(step_lines[-1])
        if picked == EOS_ID:
            stop_reason = "picked the EOS token"
            break
    shown_reply_words = [VOCAB[t] for t in generated]
    shown_reply, shown_steps = " ".join(shown_reply_words), len(generated)
    print("  reply:", shown_reply, "· stopped because it", stop_reason,
          "after", shown_steps, "steps")
    encodes_in_loop = ENCODE_CALLS[0] - encodes_before_loop
    shown_decoder_runs = len(fed_lengths)
    print("  the encoder ran", encodes_in_loop, "more time(s) inside the loop, while the",
          "decoder ran", shown_decoder_runs, "-> read once, write many")

    # --- Self-check: one boolean per claim -------------------------------------------
    # Every expected number below was READ OFF a run and written down here, so a broken
    # change in the code above cannot quietly agree with itself. Each claim reads the
    # SHOWN value, so corrupting a printed number fails the run instead of misleading you.
    expected_self_row2 = np.array([0.2926, 0.2572, 0.4502])    # decoder row 2, printed above
    expected_cross_row0 = np.array([0.5304, 0.4696])           # cross-attention row 0
    expected_chart = np.array([0.0648, 0.3009, 0.1671, 0.0490, 0.3311, 0.0871])

    claims = {
        "the 6-word vocabulary, 'hi there' as ids [1, 2], and <eos> as the stop word":
            VOCAB == ["<start>", "hi", "there", "hello", "how", "<eos>"]
            and src == [1, 2] and shown_eos_word == "<eos>" and V == 6,
        "on-ramp, encoder and decoder must all be (seq_len, 8), cross grid (3, 2)":
            shown_emb_shape == (2, D_MODEL) and shown_in_shape == (2, D_MODEL)
            and shown_block_shape == (2, D_MODEL) and shown_enc_out_shape == (2, D_MODEL)
            and shown_dec_in_shape == (3, D_MODEL) and shown_dec_out_shape == (3, D_MODEL)
            and shown_cross_shape == (3, 2)
            and src_in_lines[0] == "  hi       [-0.1342  1.5003  0.3504  1.383  -1.0169  "
                                   "0.0574  0.9323  1.5813]",
        "layer_norm must leave each word at mean 0 and spread 1":
            round(shown_norm_mean, 10) == 0.0 and abs(shown_norm_mean) < 1e-15
            and np.array_equal(shown_spreads, np.ones(2)),
        "the residual shortcut should keep the input's trace: cosine 0.9589, and only "
        "0.0012 of it survives when the shortcut is cut out":
            shown_trace == 0.9589 and shown_trace_without == 0.0012
            and shown_trace > shown_trace_without,
        "every above-diagonal share must be exactly 0 (that is the causal mask)":
            future_share == 0.0 and shown_future_share == 0.0,
        "decoder self-attention row 2 should be [0.2926 0.2572 0.4502]":
            np.array_equal(shown_self_w[2], expected_self_row2)
            and self_w_lines == ["  <start>  [1. 0. 0.]",
                                 "  how      [0.5684 0.4316 0.    ]",
                                 "  there    [0.2926 0.2572 0.4502]"],
        "cross-attention row 0 should be [0.5304 0.4696] over the 2 source words":
            np.array_equal(shown_cross_w[0], expected_cross_row0)
            and cross_w_lines[0] == "  <start>  [0.5304 0.4696]",
        "masked rows must be identical when re-decoded from the shorter prefix":
            mask_gap == 0.0 and shown_mask_gap == 0.0,
        "the unmasked encoder's rows should move by 0.4055 in that same test":
            shown_enc_gap == 0.4055,
        "a different source must move the decoder's answer by 1.2298":
            shown_bridge_gap == 1.2298,
        "the chart should be [0.0648 0.3009 0.1671 0.049 0.3311 0.0871] and sum to 1.0":
            shown_probs_shape == (V,) and shown_prob_sum == 1.0
            and np.array_equal(np.round(probs, 4), expected_chart)
            and chart_lines == ["  <start>  0.0648 ###", "  hi       0.3009 ############",
                                "  there    0.1671 #######", "  hello    0.0490 ##",
                                "  how      0.3311 #############", "  <eos>    0.0871 ###"],
        "the top vector and its logits, whose biggest score is the winning word 'how'":
            np.array_equal(shown_top_vector, [2.7424, -0.0738, -0.4124, 1.2916,
                                              -0.9444, 0.7385, 0.7867, 1.1952])
            and np.array_equal(shown_logits, [-0.5136, 1.0225, 0.4343,
                                              -0.7923, 1.1182, -0.2175])
            and shown_winner == "how" and int(np.argmax(shown_logits)) == int(np.argmax(probs)),
        "the decoder must be fed one more word each step (and process every one)":
            fed_lengths == rows_seen == predicted_lengths[:shown_steps],
        "each step line must show the reply-so-far growing by one word, with the "
        "probability the greedy pick actually had":
            step_lines == [
                "  step 1 · fed 1 word(s) ['<start>']                        "
                "-> picked how      p=0.2685",
                "  step 2 · fed 2 word(s) ['<start>', 'how']                 "
                "-> picked there    p=0.2290",
                "  step 3 · fed 3 word(s) ['<start>', 'how', 'there']        "
                "-> picked how      p=0.3311",
                "  step 4 · fed 4 word(s) ['<start>', 'how', 'there', 'how'] "
                "-> picked <eos>    p=0.2184"],
        # `stop_reason` is a label this script sets itself, so it is checked AGAINST the
        # mechanism it claims (the last drawn token really is EOS) rather than on its own.
        "expected the reply 'how there how <eos>', ending on EOS before the 6-word cap":
            shown_reply_words == ["how", "there", "how", "<eos>"]
            and shown_reply == "how there how <eos>" and shown_steps < max_new
            and generated[-1] == EOS_ID and stop_reason == "picked the EOS token",
        # Step 3 feeds exactly the reply Part 4 scored by hand, so the head's winner and
        # the loop's third pick have to be the same word.
        "Part 4's greedy winner must be the same word the loop picks at step 3":
            shown_winner == VOCAB[generated[2]],
        "the encoder must read the input once, OUTSIDE the loop: 0 more reads inside it":
            encodes_in_loop == 0 and shown_decoder_runs == 4 and len(fed_lengths) == 4,
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
