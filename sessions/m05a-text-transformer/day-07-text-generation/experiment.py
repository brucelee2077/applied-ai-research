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
    # axis=-1 = "along each row", the same spelling days 5, 6 and 8 use. Every chart on
    # this day is a single row, so the axis changes nothing here — it is written in so the
    # one softmax in the module that could normalise a WHOLE grid does not.
    fresh = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # Divide by the row's total, so the bars add up to 1 and become probabilities.
    return fresh / fresh.sum(axis=-1, keepdims=True)

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
    # "Reaches p" must mean >= p and stop THERE — Part 6(a) feeds it a total that lands
    # exactly on p. The count can also run one PAST the last bar when p is at or above
    # the total mass (a softmax chart sums to 0.9999999999999999, not 1), so it is
    # clamped; n_keep is RETURNED so Part 6(b) can print the clamp doing its job.
    n_keep = min(int(np.searchsorted(running, p)) + 1, len(probs))
    kept = np.zeros_like(probs)
    kept[order[:n_keep]] = probs[order[:n_keep]]
    # re-shared to 1, plus who survived and how many survived
    return kept / kept.sum(), order[:n_keep], n_keep

def words_of(values):
    # The words a chart can still produce, biggest first. Trimmed words sit at 0.
    return [WORDS[i] for i in np.argsort(values)[::-1] if values[i] > 0]

def show_chart(title, values):
    # Build every line FIRST, print it, then hand the lines back. This one helper draws
    # five different charts, so a slip inside it would change all five at once — the
    # self-check tests the exact text that was printed, not a second copy of the numbers.
    lines = ["%s - shape %s - sums to %s" % (title, values.shape, round(float(values.sum()), 4))]
    lines += ["   %-7s logit %5.2f  prob %.4f  %s" % (word, logit, prob, "#" * int(prob * 40))
              for word, logit, prob in zip(WORDS, LOGITS, values)]
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    # Every printed number below is bound to a name FIRST and then read again by the
    # self-check, so the line you read and the line that is tested are one single value.
    # --- Part 1: one fixed chart, built by softmax -------------------------
    probs = softmax(LOGITS)
    shown_logits_shape, shown_logits = LOGITS.shape, LOGITS
    shown_probs = np.round(probs, 4)
    print("LOGITS shape:", shown_logits_shape, " values:", shown_logits)
    chart_lines = show_chart("the next-word chart (T = 1)", probs)
    top_index = int(np.argmax(probs))

    # --- Part 2: greedy — always grab the tallest bar ----------------------
    five_greedy = [greedy(probs) for _ in range(5)]
    shown_greedy_distinct = len(set(five_greedy))
    shown_greedy_word = greedy(probs)
    print("\ngreedy, 5 calls:", five_greedy, "-> distinct words:", shown_greedy_distinct,
          "| no dice in it, so this chart always gives", shown_greedy_word)

    # --- Part 3: sampling at a cool and a warm temperature -----------------
    cool_probs, warm_probs = softmax(LOGITS / COOL), softmax(LOGITS / WARM)
    cool_chart_lines = show_chart("\nT = 0.7 (cool: the top bar towers)", cool_probs)
    warm_chart_lines = show_chart("T = 1.5 (warm: the bars flatten)", warm_probs)
    # A prediction we COMPUTE, not one we type: the chance of NOT drawing the top
    # word is 1 minus its own bar height, so warm should surprise us more often.
    pred_cool, pred_warm = 1 - cool_probs[top_index], 1 - warm_probs[top_index]
    shown_pred = "cool %.4f  warm %.4f" % (pred_cool, pred_warm)
    print("predicted surprise rate (1 - top bar):", shown_pred)
    cool_five = [sample(LOGITS, COOL) for _ in range(5)]
    warm_five = [sample(LOGITS, WARM) for _ in range(5)]
    shown_cool_distinct, shown_warm_distinct = len(set(cool_five)), len(set(warm_five))
    print("sample 5x cool:", cool_five, "distinct", shown_cool_distinct,
          "| 5x warm:", warm_five, "distinct", shown_warm_distinct)
    cool_counts, warm_counts = tally(cool_probs, N_DRAWS), tally(warm_probs, N_DRAWS)
    shown_cool_rate = round(float((N_DRAWS - cool_counts[top_index]) / N_DRAWS), 4)
    shown_warm_rate = round(float((N_DRAWS - warm_counts[top_index]) / N_DRAWS), 4)
    shown_warmer_wins = shown_warm_rate > shown_cool_rate
    print("over %d draws, cool %s -> surprise %.4f | warm %s -> surprise %.4f | warmer wins: %s"
          % (N_DRAWS, cool_counts, shown_cool_rate, warm_counts, shown_warm_rate,
             shown_warmer_wins))
    # Does the dice really follow the chart? Then the busiest words must come out in
    # the tallest-bar order. Checked at both temperatures.
    shown_warm_order = words_of(warm_counts)
    rank_ok = (words_of(cool_counts) == words_of(cool_probs)
               and shown_warm_order == words_of(warm_probs)
               and shown_warm_order == ["mat", "floor", "rug", "sofa", "banana"])
    print("draw order matches bar order?", rank_ok, shown_warm_order)

    # --- Part 4: top-k — a fixed shortlist --------------------------------
    plain_counts = tally(probs, N_DRAWS)
    k3 = top_k_probs(LOGITS, 3)
    k3_chart_lines = show_chart("\ntop-k (k = 3): the trimmed chart", k3)
    # WHO should be thrown out is decided by the raw scores — the two smallest logits —
    # and never by reading zeros back out of k3. Reading them out of k3 would make the
    # escape count below impossible to fail: a bar of height 0 can never be drawn.
    dropped = sorted(int(i) for i in np.argsort(LOGITS)[:len(WORDS) - 3])
    k3_counts = tally(k3, N_DRAWS)
    escaped = int(k3_counts[dropped].sum())
    shown_survivors, shown_thrown = words_of(k3), [WORDS[i] for i in dropped]
    shown_k3 = np.round(k3, 4)
    print("survivors", shown_survivors, "| thrown out", shown_thrown)
    # The plain tally is the CONTROL for the whole top-k claim, so it is rendered once and
    # pinned as text below. `plain_counts[dropped].min() > 0` alone would let the two
    # busiest words swap on screen — the sampling evidence printed upside down under a ✅.
    plain_line = "plain draws %s -> the thrown-out words DO come out" % plain_counts
    print(plain_line)
    print("top-k draws %s -> they escaped the shortlist %d times out of %d"
          % (k3_counts, escaped, N_DRAWS))

    # --- Part 5: top-p — a shortlist that resizes itself -------------------
    p_base, kept_base, keep_base = top_p_probs(probs, 0.9)
    base_mass = round(float(probs[kept_base].sum()), 4)
    shown_p_base, shown_p_base_shape = np.round(p_base, 4), p_base.shape
    print("\ntop-p (p = 0.9) chart:", shown_p_base, "- shape", shown_p_base_shape)
    shown_topp_survivors = words_of(p_base)
    shown_topp_draw = draw_from(p_base)
    print("survivors", shown_topp_survivors, "| their bars add up to", base_mass,
          ">= 0.9 | one draw:", shown_topp_draw)
    # Sharpen and flatten the SAME five scores: top-p's shortlist resizes itself,
    # while a top-k shortlist would have stayed frozen at 3 words.
    sizes, tops, keeps, orders, reshared, draws = {}, {}, {}, {}, {}, {}
    for name, scale in [("peaked", 2.5), ("flat", 0.4)]:
        variant = softmax(LOGITS * scale)
        v_probs, v_kept, v_keep = top_p_probs(variant, 0.9)
        sizes[name], tops[name], keeps[name] = len(v_kept), round(float(variant.max()), 4), v_keep
        orders[name], reshared[name] = words_of(v_probs), np.round(v_probs[v_kept], 4)
        draws[name] = draw_from(v_probs)
        print("%-6s (tallest bar %.4f) -> %d survivor(s) %-30s re-shared %s  drew %s"
              % (name, tops[name], sizes[name], str(orders[name]), reshared[name], draws[name]))

    # --- Part 6: the edge cases the guards above exist for -----------------
    # Three lines of defence sit in the helpers above, and nothing printed so far can
    # tell you whether they work. Here is the input that makes each one earn its place.
    # (a) A cut-off that lands EXACTLY on p. These five bars are exact binary fractions,
    # so the running total hits 0.75 dead on. "Reaches p" must mean >= p and stop there:
    # a looser test would keep a third bar the reader was never promised.
    edge_bars = np.array([0.5, 0.25, 0.125, 0.09375, 0.03125])
    shown_edge_running = np.cumsum(edge_bars)
    edge_keeps = {q: top_p_probs(edge_bars, q)[2] for q in (0.7499, 0.75, 0.7501, 0.875)}
    shown_edge_mass = round(float(shown_edge_running[edge_keeps[0.75] - 1]), 4)
    print("\nedge (a) exact cut-off: bars", edge_bars, "running total", shown_edge_running)
    print("  p = 0.75 lands EXACTLY on the 2nd total -> keeps %d bar(s), mass %s"
          % (edge_keeps[0.75], shown_edge_mass),
          "| 0.7499 -> %d | 0.7501 -> %d | 0.875 -> %d"
          % (edge_keeps[0.7499], edge_keeps[0.7501], edge_keeps[0.875]))
    # (b) p = 1.0 on the real chart. Its bars add up to a hair under 1, so the cut-off
    # search runs one place PAST the last bar; the clamp is what keeps the count at 5.
    chart_total = float(probs.sum())
    # Printed at full precision so the learner SEES the hair; asserted as a PROPERTY, never as
    # 17 digits — one ulp of difference in a numpy build would make correct code print ❌.
    shown_chart_total = repr(chart_total)
    _, kept_all, keep_all = top_p_probs(probs, 1.0)
    all_order = np.argsort(probs)[::-1]
    unclamped_keep = int(np.searchsorted(np.cumsum(probs[all_order]), 1.0)) + 1
    print("edge (b) p = 1.0: the chart really sums to", shown_chart_total,
          "-> keeps %d of %d bars (with no clamp the count would read %d)"
          % (keep_all, len(WORDS), unclamped_keep))
    # (c) exp() overflow. Multiply every score by 400 and a raw exp() gives inf, whose
    # ratio is nan; subtracting the biggest score first keeps the chart finite.
    huge_chart = softmax(LOGITS * 400)
    shown_huge, shown_huge_sum = np.round(huge_chart, 4), round(float(huge_chart.sum()), 4)
    shown_huge_finite = bool(np.isfinite(huge_chart).all())
    print("edge (c) scores x400 -> chart", shown_huge, "sums to", shown_huge_sum,
          "| every bar a real number:", shown_huge_finite)
    # (d) A one-word chart. Only word 0 can ever be drawn, so counting the draws
    # without minlength would hand back a 1-long row instead of a 5-word row.
    edge_counts = tally(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 50)
    shown_edge_counts, shown_edge_counts_shape = edge_counts, edge_counts.shape
    print("edge (d) one-word chart, 50 draws -> counts", shown_edge_counts,
          "shape", shown_edge_counts_shape)

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and written down here, so the code
    # above cannot quietly agree with itself. The checks read the SHOWN values, so
    # corrupting a printed number fails the run instead of misleading the reader.
    chart_ok = (np.array_equal(shown_probs, [0.6956, 0.0127, 0.1552, 0.0423, 0.0941])
                and shown_logits_shape == (5,)
                and np.array_equal(shown_logits, [3.0, -1.0, 1.5, 0.2, 1.0])
                # the chart as it was RENDERED, bars and all
                # the chart as RENDERED, minus the bar: `.rstrip("#")` keeps every label and
                # number pinned while leaving the bar SCALE free, so widening it is not a failure
                and [line.rstrip("#") for line in chart_lines] == [
                    "the next-word chart (T = 1) - shape (5,) - sums to 1.0",
                    "   mat     logit  3.00  prob 0.6956  ",
                    "   banana  logit -1.00  prob 0.0127  ",
                    "   floor   logit  1.50  prob 0.1552  ",
                    "   sofa    logit  0.20  prob 0.0423  ",
                    "   rug     logit  1.00  prob 0.0941  "])
    greedy_ok = (five_greedy == ["mat"] * 5 and shown_greedy_distinct == 1
                 and shown_greedy_word == "mat")
    temp_charts_ok = (
        np.array_equal(np.round(cool_probs, 4), [0.8359, 0.0028, 0.0981, 0.0153, 0.048])
        and np.array_equal(np.round(warm_probs, 4), [0.5389, 0.0374, 0.1983, 0.0833, 0.1421])
        and cool_chart_lines[0] == "\nT = 0.7 (cool: the top bar towers) - shape (5,) - sums to 1.0"
        and cool_chart_lines[1].rstrip("#") == "   mat     logit  3.00  prob 0.8359  "
        and warm_chart_lines[0] == "T = 1.5 (warm: the bars flatten) - shape (5,) - sums to 1.0"
        and warm_chart_lines[1].rstrip("#") == "   mat     logit  3.00  prob 0.5389  ")
    draws_ok = (cool_five == ["mat"] * 5
                and warm_five == ["rug", "floor", "floor", "banana", "rug"]
                and shown_cool_distinct == 1 and shown_warm_distinct == 3)
    counts_ok = (np.array_equal(cool_counts, [1667, 5, 202, 27, 99])
                 and np.array_equal(warm_counts, [1106, 74, 383, 176, 261])
                 and shown_cool_rate == 0.1665 and shown_warm_rate == 0.447)
    # relational, so it still bites even if a pin above were written down wrongly
    warmer_wins = (pred_warm > pred_cool and shown_warm_rate > shown_cool_rate
                   and shown_warmer_wins and shown_warm_distinct > shown_cool_distinct
                   and shown_pred == "cool 0.1641  warm 0.4611")
    topk_ok = (shown_survivors == ["mat", "floor", "rug"] and dropped == [1, 3]
               and shown_thrown == ["banana", "sofa"]
               and np.array_equal(shown_k3, [0.7361, 0.0, 0.1643, 0.0, 0.0996])
               and k3_chart_lines[1].rstrip("#") == "   mat     logit  3.00  prob 0.7361  "
               # the words the raw scores dropped are exactly the bars top-k zeroed
               and all(k3[i] == 0.0 for i in dropped))
    # the trim must bite: 0 escapes, yet those same words ARE drawn without it
    trim_bites = escaped == 0 and int(plain_counts[dropped].min()) > 0
    # ...and the control tally exactly as it was PRINTED, so the untrimmed evidence the
    # reader compares against cannot be reordered or rescaled behind the inequality.
    plain_line_ok = (plain_line == "plain draws [1383   31  309   91  186] -> "
                                   "the thrown-out words DO come out"
                     and np.array_equal(plain_counts, [1383, 31, 309, 91, 186]))
    topp_ok = (shown_topp_survivors == ["mat", "floor", "rug"] and base_mass == 0.945
               and np.array_equal(shown_p_base, [0.7361, 0.0, 0.1643, 0.0, 0.0996])
               and shown_p_base_shape == (5,) and keep_base == 3 and shown_topp_draw == "mat")
    resizes = (sizes == {"peaked": 1, "flat": 4} and tops == {"peaked": 0.9697, "flat": 0.3958}
               and keeps == sizes and draws == {"peaked": "mat", "flat": "floor"}
               and orders == {"peaked": ["mat"], "flat": ["mat", "floor", "rug", "sofa"]}
               and np.array_equal(reshared["peaked"], [1.0])
               and np.array_equal(reshared["flat"], [0.4302, 0.2361, 0.1933, 0.1404]))
    # The guards, each judged on the input that reaches it.
    exact_cutoff_ok = (edge_keeps == {0.7499: 2, 0.75: 2, 0.7501: 3, 0.875: 3}
                       and shown_edge_mass == 0.75
                       and np.array_equal(shown_edge_running, [0.5, 0.75, 0.875, 0.96875, 1.0]))
    clamp_ok = (keep_all == 5 and len(kept_all) == 5 and unclamped_keep == 6
                and chart_total != 1.0 and abs(chart_total - 1.0) < 1e-15)
    overflow_ok = (shown_huge_finite and shown_huge_sum == 1.0
                   and np.array_equal(shown_huge, [1.0, 0.0, 0.0, 0.0, 0.0]))
    full_row_ok = (shown_edge_counts_shape == (5,)
                   and np.array_equal(shown_edge_counts, [50, 0, 0, 0, 0]))

    if (chart_ok and greedy_ok and temp_charts_ok and draws_ok and counts_ok and warmer_wins
            and rank_ok and topk_ok and trim_bites and plain_line_ok and topp_ok and resizes
            and exact_cutoff_ok
            and clamp_ok and overflow_ok and full_row_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected the chart [0.6956 0.0127 0.1552 0.0423 0.0941], greedy "
              "'mat' 5 times, a cool chart topping 0.8359 and a warm one 0.5389, warm 5 draws "
              "['rug','floor','floor','banana','rug'], counts [1667 5 202 27 99] and "
              "[1106 74 383 176 261], surprise rates 0.1665 then 0.447, top-k=3 keeping "
              "mat/floor/rug at [0.7361 0.1643 0.0996] with 0 escapes, top-p=0.9 keeping the "
              "same three with mass 0.945, top-p sizes 1 < 3 < 4 (peaked tallest bar "
              "0.9697, flat 0.3958), and the four edge cases: p=0.75 landing exactly on a "
              "total keeps 2 bars, p=1.0 keeps 5 (not 6), scores x400 stay finite, and a "
              "one-word chart still counts 5 words [50 0 0 0 0]")

    assert chart_ok, "softmax(LOGITS) should be [0.6956 0.0127 0.1552 0.0423 0.0941]"
    assert greedy_ok, "greedy should return 'mat' on all 5 calls — the tallest bar, no dice"
    assert temp_charts_ok, "T=0.7 should sharpen 'mat' to 0.8359 and T=1.5 flatten it to 0.5389"
    assert draws_ok, "the seeded draws should be 5x 'mat' cool, rug/floor/floor/banana/rug warm"
    assert counts_ok, "2000 draws: [1667 5 202 27 99] / [1106 74 383 176 261], rates .1665/.447"
    assert warmer_wins, "the warm chart must be predicted AND measured to surprise more often"
    assert rank_ok, "the busiest drawn words must line up with the tallest bars at both T"
    assert topk_ok, "top-k=3 should keep mat/floor/rug re-shared to [0.7361 0.1643 0.0996]"
    assert trim_bites, "top-k must draw banana/sofa 0 times, though plain sampling draws them"
    assert plain_line_ok, "the untrimmed control must print 'plain draws [1383 31 309 91 186]'"
    assert topp_ok, "top-p=0.9 should keep mat/floor/rug, whose bars sum to 0.945"
    assert resizes, "the top-p shortlist should be 1 word when peaked and 4 when flat, not 3"
    assert exact_cutoff_ok, "a total landing exactly on p=0.75 must keep 2 bars, and 0.7501 keep 3"
    assert clamp_ok, "p=1.0 on a chart summing to 0.9999999999999999 must keep 5 bars, not 6"
    assert overflow_ok, "scores x400 must still give a finite chart summing to 1.0"
    assert full_row_ok, "a one-word chart must still count all 5 words: [50 0 0 0 0]"
