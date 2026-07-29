# day-07-text-generation — experiment
#
# Today's big idea in two lines of output:
#   One fixed chart of next-word scores turns into five different words —
#   the chart never moves, only your RULE for picking does.
#
# It builds the chart with softmax, then runs greedy, sampling at a cool and a
# warm temperature, top-k and top-p over the SAME five logits.
# Run it:  python3 sessions/m05a-text-transformer/day-07-text-generation/experiment.py

import numpy as np  # arrays, exp(), argsort(), and the weighted draw

# One seeded generator for the whole run. The lesson says np.random.choice; a seeded
# generator is the same weighted draw, but it makes the "random" words below identical
# on every run, so the pinned numbers in the self-check stay stable.
RNG = np.random.default_rng(0)

# Five made-up next words with one fixed set of raw scores (logits). The words are NOT
# listed in score order — a real vocabulary never is — so a picker that forgets to sort
# by probability gets caught instead of looking right by luck.
WORDS = ["mat", "banana", "floor", "sofa", "rug"]
LOGITS = np.array([3.0, -1.0, 1.5, 0.2, 1.0])
COOL, WARM = 0.7, 1.5     # the two temperatures we compare
N_DRAWS = 2000            # draws used to measure how varied a picking rule is


def softmax(scores):
    # Subtract the largest score first so exp() never blows up on a big number.
    fresh = np.exp(scores - scores.max())
    # Divide by the total, so the bars add up to 1 and become probabilities.
    return fresh / fresh.sum()

def greedy(probs):
    # Always the tallest bar. argmax = "the index that gives the max".
    return WORDS[int(np.argmax(probs))]

def draw_from(probs):
    # Roll the weighted dice once: a taller bar is drawn more often.
    return WORDS[int(RNG.choice(len(WORDS), p=probs))]

def sample(logits, T):
    # Temperature: divide every raw score by T BEFORE softmax. T < 1 spreads the
    # scores apart (sharper chart); T > 1 squeezes them together (flatter chart).
    return draw_from(softmax(logits / T))

def tally(probs, n_draws):
    # Draw n_draws words and count how many times each word came out.
    return np.bincount(RNG.choice(len(WORDS), size=n_draws, p=probs), minlength=len(WORDS))

def top_k_probs(logits, k):
    # Keep the k tallest logits; push every other word to -inf so softmax gives it
    # exactly 0. Softmax then re-shares the survivors back up to 1 for us.
    keep = np.argsort(logits)[-k:]                # the k LARGEST scores
    masked = np.full_like(logits, -np.inf)
    masked[keep] = logits[keep]
    return softmax(masked)

def top_p_probs(probs, p):
    # Walk the bars tallest-first, adding them up, and stop as soon as the running
    # total reaches p. Everything after that point is thrown away.
    order = np.argsort(probs)[::-1]                # tallest bar first
    running = np.cumsum(probs[order])
    # searchsorted gives the first place the total is >= p; +1 turns it into a count.
    n_keep = min(int(np.searchsorted(running, p)) + 1, len(probs))
    kept = np.zeros_like(probs)
    kept[order[:n_keep]] = probs[order[:n_keep]]
    return kept / kept.sum(), order[:n_keep]       # re-shared to 1, plus who survived

def words_of(values):
    # The words a chart can still produce, biggest first. Trimmed words sit at 0.
    return [WORDS[i] for i in np.argsort(values)[::-1] if values[i] > 0]

def show_chart(title, values):
    print("%s - shape %s - sums to %s" % (title, values.shape, round(float(values.sum()), 4)))
    for word, logit, prob in zip(WORDS, LOGITS, values):
        print("   %-7s logit %5.2f  prob %.4f  %s" % (word, logit, prob, "#" * int(prob * 40)))


if __name__ == "__main__":
    # --- Part 1: one fixed chart, built by softmax -------------------------
    probs = softmax(LOGITS)
    print("LOGITS shape:", LOGITS.shape, " values:", LOGITS)
    show_chart("the next-word chart (T = 1)", probs)
    top_index = int(np.argmax(probs))

    # --- Part 2: greedy — always grab the tallest bar ----------------------
    five_greedy = [greedy(probs) for _ in range(5)]
    print("\ngreedy, 5 calls:", five_greedy, "-> distinct words:", len(set(five_greedy)),
          "| no dice in it, so this chart always gives", greedy(probs))

    # --- Part 3: sampling at a cool and a warm temperature -----------------
    cool_probs, warm_probs = softmax(LOGITS / COOL), softmax(LOGITS / WARM)
    show_chart("\nT = 0.7 (cool: the top bar towers)", cool_probs)
    show_chart("T = 1.5 (warm: the bars flatten)", warm_probs)
    # A prediction we COMPUTE, not one we type: the chance of NOT drawing the top
    # word is 1 minus its own bar height, so warm should surprise us more often.
    pred_cool, pred_warm = 1 - cool_probs[top_index], 1 - warm_probs[top_index]
    print("predicted surprise rate (1 - top bar): cool %.4f  warm %.4f" % (pred_cool, pred_warm))
    cool_five = [sample(LOGITS, COOL) for _ in range(5)]
    warm_five = [sample(LOGITS, WARM) for _ in range(5)]
    print("sample 5x cool:", cool_five, "distinct", len(set(cool_five)),
          "| 5x warm:", warm_five, "distinct", len(set(warm_five)))
    cool_counts, warm_counts = tally(cool_probs, N_DRAWS), tally(warm_probs, N_DRAWS)
    cool_rate = (N_DRAWS - cool_counts[top_index]) / N_DRAWS
    warm_rate = (N_DRAWS - warm_counts[top_index]) / N_DRAWS
    print("over %d draws, cool %s -> surprise %.4f | warm %s -> surprise %.4f | warmer wins: %s"
          % (N_DRAWS, cool_counts, cool_rate, warm_counts, warm_rate, warm_rate > cool_rate))
    # Does the dice really follow the chart? Then the busiest words must come out in
    # the tallest-bar order. Checked at both temperatures.
    rank_ok = (words_of(cool_counts) == words_of(cool_probs)
               and words_of(warm_counts) == words_of(warm_probs))
    print("draw order matches bar order?", rank_ok, words_of(warm_counts))

    # --- Part 4: top-k — a fixed shortlist --------------------------------
    plain_counts = tally(probs, N_DRAWS)
    k3 = top_k_probs(LOGITS, 3)
    show_chart("\ntop-k (k = 3): the trimmed chart", k3)
    dropped = [i for i in range(len(WORDS)) if k3[i] == 0]
    k3_counts = tally(k3, N_DRAWS)
    escaped = int(k3_counts[dropped].sum())
    print("survivors", words_of(k3), "| thrown out", [WORDS[i] for i in dropped])
    print("plain draws %s -> the thrown-out words DO come out" % plain_counts)
    print("top-k draws %s -> they escaped the shortlist %d times out of %d"
          % (k3_counts, escaped, N_DRAWS))

    # --- Part 5: top-p — a shortlist that resizes itself -------------------
    p_base, kept_base = top_p_probs(probs, 0.9)
    base_mass = round(float(probs[kept_base].sum()), 4)
    print("\ntop-p (p = 0.9) chart:", np.round(p_base, 4), "- shape", p_base.shape)
    print("survivors", words_of(p_base), "| their bars add up to", base_mass,
          ">= 0.9 | one draw:", draw_from(p_base))
    # Sharpen and flatten the SAME five scores: top-p's shortlist resizes itself,
    # while a top-k shortlist would have stayed frozen at 3 words.
    sizes, tops = {}, {}
    for name, scale in [("peaked", 2.5), ("flat", 0.4)]:
        variant = softmax(LOGITS * scale)
        v_probs, v_kept = top_p_probs(variant, 0.9)
        sizes[name], tops[name] = len(v_kept), round(float(variant.max()), 4)
        print("%-6s (tallest bar %.4f) -> %d survivor(s) %-30s re-shared %s  drew %s"
              % (name, tops[name], sizes[name], str(words_of(v_probs)),
                 np.round(v_probs[v_kept], 4), draw_from(v_probs)))

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and written down here, so the code
    # above cannot quietly agree with itself.
    chart_ok = np.array_equal(np.round(probs, 4), [0.6956, 0.0127, 0.1552, 0.0423, 0.0941])
    greedy_ok = five_greedy == ["mat"] * 5
    temp_charts_ok = (
        np.array_equal(np.round(cool_probs, 4), [0.8359, 0.0028, 0.0981, 0.0153, 0.048])
        and np.array_equal(np.round(warm_probs, 4), [0.5389, 0.0374, 0.1983, 0.0833, 0.1421]))
    draws_ok = (cool_five == ["mat"] * 5
                and warm_five == ["rug", "floor", "floor", "banana", "rug"])
    counts_ok = (np.array_equal(cool_counts, [1667, 5, 202, 27, 99])
                 and np.array_equal(warm_counts, [1106, 74, 383, 176, 261])
                 and round(cool_rate, 4) == 0.1665 and round(warm_rate, 4) == 0.447)
    # relational, so it still bites even if a pin above were written down wrongly
    warmer_wins = (pred_warm > pred_cool and warm_rate > cool_rate
                   and len(set(warm_five)) > len(set(cool_five)))
    topk_ok = (words_of(k3) == ["mat", "floor", "rug"] and dropped == [1, 3]
               and np.array_equal(np.round(k3, 4), [0.7361, 0.0, 0.1643, 0.0, 0.0996]))
    # the trim must bite: 0 escapes, yet those same words ARE drawn without it
    trim_bites = escaped == 0 and int(plain_counts[dropped].min()) > 0
    topp_ok = words_of(p_base) == ["mat", "floor", "rug"] and base_mass == 0.945
    resizes = sizes == {"peaked": 1, "flat": 4} and tops == {"peaked": 0.9697, "flat": 0.3958}

    if (chart_ok and greedy_ok and temp_charts_ok and draws_ok and counts_ok and warmer_wins
            and rank_ok and topk_ok and trim_bites and topp_ok and resizes):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected the chart [0.6956 0.0127 0.1552 0.0423 0.0941], greedy "
              "'mat' 5 times, a cool chart topping 0.8359 and a warm one 0.5389, warm 5 draws "
              "['rug','floor','floor','banana','rug'], counts [1667 5 202 27 99] and "
              "[1106 74 383 176 261], surprise rates 0.1665 then 0.447, top-k=3 keeping "
              "mat/floor/rug at [0.7361 0.1643 0.0996] with 0 escapes, top-p=0.9 keeping the "
              "same three with mass 0.945, and top-p sizes 1 < 3 < 4 (peaked tallest bar "
              "0.9697, flat 0.3958)")

    assert chart_ok, "softmax(LOGITS) should be [0.6956 0.0127 0.1552 0.0423 0.0941]"
    assert greedy_ok, "greedy should return 'mat' on all 5 calls — the tallest bar, no dice"
    assert temp_charts_ok, "T=0.7 should sharpen 'mat' to 0.8359 and T=1.5 flatten it to 0.5389"
    assert draws_ok, "the seeded draws should be 5x 'mat' cool, rug/floor/floor/banana/rug warm"
    assert counts_ok, "2000 draws: [1667 5 202 27 99] / [1106 74 383 176 261], rates .1665/.447"
    assert warmer_wins, "the warm chart must be predicted AND measured to surprise more often"
    assert rank_ok, "the busiest drawn words must line up with the tallest bars at both T"
    assert topk_ok, "top-k=3 should keep mat/floor/rug re-shared to [0.7361 0.1643 0.0996]"
    assert trim_bites, "top-k must draw banana/sofa 0 times, though plain sampling draws them"
    assert topp_ok, "top-p=0.9 should keep mat/floor/rug, whose bars sum to 0.945"
    assert resizes, "the top-p shortlist should be 1 word when peaked and 4 when flat, not 3"
