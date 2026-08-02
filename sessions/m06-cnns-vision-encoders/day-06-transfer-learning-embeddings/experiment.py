# day-06-transfer-learning-embeddings — experiment
#
# Today's big idea in two lines of output:
#   A borrowed backbone turns each image into a short list of numbers — its embedding —
#   so "find similar images" becomes "find nearby numbers", with no training at all.
#
# The borrowed backbone is stood in for by fixed embeddings, exactly as the lesson's produce
# step asks (nothing is downloaded, and no image dataset is on this machine). Parts: three
# embeddings, the two rulers (cosine and L2), nearest-neighbor search, then the preprocessing
# mismatch that turns the ranking into noise. No random numbers are used — every vector is a
# written-down literal, so two runs print the same bytes.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-06-transfer-learning-embeddings/experiment.py

import numpy as np  # numpy gives us arrays, the dot product (@) and vector lengths


# ---- The two rulers for "close", straight from the lesson -------------------
def cosine(a, b):
    # Do the two number-lists POINT the same way? Near 1 = same direction = similar.
    # Dividing by BOTH lengths is what makes this about direction only, not size.
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def l2(a, b):
    # Straight-line (Euclidean) distance between the two pins on the map. Small = close.
    return float(np.linalg.norm(a - b))

def rank_by_cosine(query, db):
    # Score every stored embedding against the query, then put the best one first.
    scored = [(name, cosine(query, emb)) for name, emb in db.items()]
    return sorted(scored, key=lambda pair: -pair[1])   # minus sign = highest first


if __name__ == "__main__":
    # --- Part 1: the borrowed eye's output — three fixed embeddings --------
    # A real frozen backbone hands back 512 numbers per image; four numbers is the same idea,
    # small enough to read on one line. These are the lesson's stand-in values.
    cat    = np.array([0.8, 0.1, 0.7, 0.2])
    kitten = np.array([0.7, 0.15, 0.79, 0.11])
    car    = np.array([0.1, 0.9, 0.0, 0.8])
    db = {"cat": cat, "kitten": kitten, "car": car}   # the stored image database
    # Every number the learner reads below is bound to a name first and then checked in the
    # self-check, so the printed line and the checked line are never two computations.
    emb_lines = ["%-6s -> %s  shape %s" % (name, emb, emb.shape) for name, emb in db.items()]
    for line in emb_lines:
        print(line)
    print("(a real backbone gives shape (512,); kitten is slot-by-slot close to cat, car is not)")
    # Predict before measuring — but a COMPUTED prediction, not a typed one. Mean |a - b|
    # per slot uses no lengths and no dot product, so it is an independent third vote.
    pairs = [("cat-kitten", cat, kitten), ("cat-car", cat, car), ("kitten-car", kitten, car)]
    l1 = {name: float(np.abs(a - b).mean()) for name, a, b in pairs}
    shown_l1 = {k: round(v, 4) for k, v in l1.items()}
    print("mean |difference| per slot:", shown_l1)
    predicted_closest = min(l1, key=l1.get)
    print("-> prediction from that crude ruler: the closest pair is", predicted_closest)

    # --- Part 2: cosine similarity ----------------------------------------
    shown_dot_ck = round(float(cat @ kitten), 4)
    shown_len_cat = round(float(np.linalg.norm(cat)), 4)
    shown_len_kitten = round(float(np.linalg.norm(kitten)), 4)
    print("\ndot(cat, kitten) =", shown_dot_ck, " length of cat =",
          shown_len_cat, " of kitten =", shown_len_kitten)
    cos_ck, cos_cc, cos_kc = cosine(cat, kitten), cosine(cat, car), cosine(kitten, car)
    shown_cos_ck, shown_cos_cc, shown_cos_kc = (round(cos_ck, 4), round(cos_cc, 4),
                                                round(cos_kc, 4))
    print("cosine(cat, kitten) =", shown_cos_ck, "-> near 1: same direction, similar")
    print("cosine(cat, car)    =", shown_cos_cc, "-> far nearer 0 than 1: unrelated")
    print("cosine(kitten, car) =", shown_cos_kc, "-> also unrelated")
    # Honest note: every slot in these stand-ins is positive, so cosine can never reach 0 here.
    # Real 512-number embeddings have negative slots too — that is how the lesson reaches 0.11.

    # --- Part 3: Euclidean / L2 distance ----------------------------------
    l2_ck, l2_cc, l2_kc = l2(cat, kitten), l2(cat, car), l2(kitten, car)
    l2_pairs = {"cat-kitten": l2_ck, "cat-car": l2_cc, "kitten-car": l2_kc}
    shown_l2_ck = round(l2_ck, 4)
    print("")
    # The line reads the SAME l2 numbers the self-check pins — it does not re-measure them.
    diff_lines = ["%-11s difference = %s -> l2 = %.4f" % (name, np.round(a - b, 4),
                                                          l2_pairs[name])
                  for name, a, b in pairs]
    for line in diff_lines:   # the difference vector, then its length
        print(line)
    cos_pairs = {"cat-kitten": cos_ck, "cat-car": cos_cc, "kitten-car": cos_kc}
    cos_order = tuple(sorted(cos_pairs, key=lambda n: -cos_pairs[n]))  # most similar first
    l2_order = tuple(sorted(l2_pairs, key=lambda n: l2_pairs[n]))      # closest first
    l1_order = tuple(sorted(l1, key=lambda n: l1[n]))                  # the Part-1 crude ruler
    print("pairs ordered most-similar-first — cosine:", cos_order, "\n  L2:", l2_order,
          "\n  mean-|diff|:", l1_order, "-> all three rulers agree ON THESE THREE PAIRS")

    # --- Part 3b: where "small L2 = high cosine" stops being true ----------
    # That is NOT a general rule, so do not carry it away as one. Cosine reads DIRECTION
    # only; L2 reads direction AND length. The two agree only when the lengths are equal —
    # which is exactly what unit-normalizing (dividing each vector by its own length) does.
    # Counterexample 1: same direction, one tenth the length. Perfect cosine, big L2.
    cat_dim = 0.1 * cat
    cos_dim, l2_dim = cosine(cat, cat_dim), l2(cat, cat_dim)
    shown_cos_dim, shown_l2_dim = round(cos_dim, 4), round(l2_dim, 4)
    # Counterexample 2: two tiny vectors at right angles. Tiny L2, zero cosine.
    tiny_a, tiny_b = np.array([0.01, 0., 0., 0.]), np.array([0., 0.01, 0., 0.])
    cos_tiny, l2_tiny = cosine(tiny_a, tiny_b), l2(tiny_a, tiny_b)
    shown_cos_tiny, shown_l2_tiny = round(cos_tiny, 4), round(l2_tiny, 4)
    print("\ncounterexample 1 · cat vs 0.1*cat (same direction, shorter): cosine =",
          shown_cos_dim, "but l2 =", shown_l2_dim, ">", shown_l2_ck, "= l2(cat, kitten),")
    print("  whose cosine is only", shown_cos_ck,
          "-> HIGHER cosine here comes with a BIGGER l2: the two rulers disagree")
    print("counterexample 2 · [0.01 0 0 0] vs [0 0.01 0 0] (right angles, both tiny): l2 =",
          shown_l2_tiny, "(tiny) but cosine =", shown_cos_tiny, "(no similarity at all)")
    # The honest rule, with its condition attached. For UNIT-length vectors (length 1 each),
    # l2^2 = 2 - 2*cosine exactly, so there small l2 really does mean high cosine.
    unit_cat, unit_kitten = cat / np.linalg.norm(cat), kitten / np.linalg.norm(kitten)
    unit_l2_sq = l2(unit_cat, unit_kitten) ** 2
    identity_gap = abs(unit_l2_sq - (2 - 2 * cos_ck))
    shown_unit_l2_sq = round(float(unit_l2_sq), 6)
    shown_two_minus_2cos = round(float(2 - 2 * cos_ck), 6)
    shown_identity_holds = bool(identity_gap < 1e-12)
    print("after UNIT-normalizing both (length 1 each): l2^2 =", shown_unit_l2_sq,
          "and 2 - 2*cosine =", shown_two_minus_2cos, " equal?", shown_identity_holds)
    print("-> the rule to carry away: small L2 = high cosine ONLY for equal-length (e.g."
          " unit-normalized) vectors; on raw embeddings of different lengths it can fail.")
    # Forward pointer: day 8 (CLIP) opens by trimming EVERY photo-arrow and caption-arrow
    # to length 1 for exactly this reason — that step is what makes the condition above
    # hold, so one number, the cosine, is allowed to stand in for "how close".

    # --- Part 4: nearest-neighbor search, with no training at all ---------
    query = np.array([0.78, 0.12, 0.74, 0.15])   # a NEW cat photo, read by the same borrowed eye
    shown_query_shape = query.shape
    print("\nquery embedding:", query, "shape", shown_query_shape, "(never stored in the db)")
    ranked = rank_by_cosine(query, db)
    good_lines = ["  #%d %-6s cosine = %.4f" % (place, name, score)
                  for place, (name, score) in enumerate(ranked, start=1)]
    for line in good_lines:
        print(line)
    ranked_names = tuple(name for name, _ in ranked)
    good_scores = tuple(round(score, 4) for _, score in ranked)
    good_spread = ranked[0][1] - ranked[-1][1]
    shown_good_spread = round(good_spread, 4)
    print("ranking:", ranked_names, "-> both cats first, the car last: image search with no",
          "training.\nbest score minus worst:", shown_good_spread, "(a wide, meaningful gap)")

    # --- Part 5: the gotcha — the WRONG preprocessing ---------------------
    # "Normalize" here is the STANDARDIZING sense of days 4-6: subtract an average, divide by
    # a spread. (Not the unit-length sense used in Part 3b and in the grid below — that one
    # divides a vector by its own length. Same word, two different operations.)
    # The pretrained model expects its OWN recipe: its resize, its per-color average and spread.
    # Here the query is centered on 5.0 and divided by 0.1 — a recipe the eye never saw. Same
    # photo, same pixels; only the numbers arrive on the wrong scale.
    query_bad = (query - 5.0) / 0.1
    # Cosine ignores vector LENGTH, so dividing by 0.1 on its own could not change any ranking.
    # It is the -5.0 shift that swings the query away from every stored embedding.
    shown_query_bad = np.round(query_bad, 4)
    shown_query_bad_shape = query_bad.shape
    print("\nquery with the WRONG normalization:", shown_query_bad,
          "shape", shown_query_bad_shape)
    print("  every number is now large and negative — that scale never reached the eye")
    ranked_bad = rank_by_cosine(query_bad, db)
    bad_lines = ["  #%d %-6s cosine = %.4f" % (place, name, score)
                 for place, (name, score) in enumerate(ranked_bad, start=1)]
    for line in bad_lines:
        print(line)
    bad_names = tuple(name for name, _ in ranked_bad)
    bad_scores = tuple(round(score, 4) for _, score in ranked_bad)
    bad, bad_spread = dict(ranked_bad), ranked_bad[0][1] - ranked_bad[-1][1]
    shown_cat_place = bad_names.index("cat") + 1
    shown_bad_spread = round(bad_spread, 4)
    print("ranking:", bad_names, "-> 'cat', the true match, fell from #1 to #%d."
          % shown_cat_place, "\nbest score minus worst:", shown_bad_spread,
          "(was", str(shown_good_spread) + ") -> squashed together, and all negative")
    shown_gap_line = ("the gap deciding cat vs car: %.7f" % (bad["cat"] - bad["car"]))
    print(shown_gap_line,
          "-> noise, not meaning: the ranking has stopped carrying information")

    # Both queries in ONE matrix multiply — how a real index scores many images at once.
    E = np.stack([cat, kitten, car])                        # (3 stored images, 4 numbers)
    Q = np.stack([query, query_bad])                        # (2 queries, 4 numbers)
    # This is the UNIT-LENGTH sense of "normalize": divide each row by its own length so the
    # dot product IS the cosine. It is not the subtract-a-mean recipe from Part 5 above.
    # Day 8 wraps exactly these two lines in a function it NAMES `normalize`, so watch the
    # bookkeeping: this operation carries two names across the module (E_unit / Q_unit here,
    # normalize() there), and the name "normalize" carries two operations (the subtract-a-
    # mean recipe of days 4-6, and this length-only one). Note there is no eps in the
    # divide either — day 4's safety crumb. It is safe here only because no row of E or Q
    # is all zeros: every one of those five vectors is pinned cell by cell in the
    # self-check (embeddings_ok and bad_vector_ok), and two of the lengths are printed
    # outright above (1.0863 and 1.0718). A zero row would come back nan, which is the
    # case day 4 demonstrated.
    E_unit = E / np.linalg.norm(E, axis=1, keepdims=True)    # divide each ROW by its length
    Q_unit = Q / np.linalg.norm(Q, axis=1, keepdims=True)    # axis=1 = across the 4 numbers
    # `S` is the SIMILARITY grid here. In days 1-3's tape-measure ⌊(N − K + 2P)/S⌋ + 1, S
    # was the STRIDE — one scalar saying how far the stencil jumps. Same letter, unrelated
    # objects; no stride appears in this file.
    S = Q_unit @ E_unit.T                                   # (queries, stored) cosine grid
    shown_E_shape, shown_Q_shape, shown_S_shape = E.shape, Q.shape, S.shape
    shown_S0, shown_S1 = np.round(S[0], 4), np.round(S[1], 4)
    print("\nE shape", shown_E_shape, " Q shape", shown_Q_shape,
          " cosine grid shape", shown_S_shape)
    print("  row 0, good query, db order (cat, kitten, car):", shown_S0)
    print("  row 1, bad query,  db order (cat, kitten, car):", shown_S1)
    grid_matches_loop = (np.allclose(S[0], [cosine(query, e) for e in E])
                         and np.allclose(S[1], [cosine(query_bad, e) for e in E]))
    print("  same numbers as the one-pair-at-a-time loop?", grid_matches_loop)

    # --- Part 6: the one-line takeaway ------------------------------------
    print("\ntakeaway: transfer learning borrows a trained backbone; its penultimate features"
          "\nare an embedding; similar images have close embeddings (high cosine / small L2), so"
          "\nnearest-neighbor search needs no training — but only with the model's exact preprocessing.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down here, so a broken
    # change above cannot quietly re-derive its own "expected" value. Each claim reads the
    # SAME `shown_*` value (or the same rendered line) that was printed, so corrupting a
    # printed number or sentence also breaks its claim.
    embeddings_ok = (emb_lines == ["cat    -> [0.8 0.1 0.7 0.2]  shape (4,)",
                                   "kitten -> [0.7  0.15 0.79 0.11]  shape (4,)",
                                   "car    -> [0.1 0.9 0.  0.8]  shape (4,)"])
    l1_ok = (shown_l1 == {"cat-kitten": 0.0825, "cat-car": 0.7, "kitten-car": 0.7075}
             and predicted_closest == "cat-kitten")
    dot_ok = (shown_dot_ck == 1.15 and shown_len_cat == 1.0863
              and shown_len_kitten == 1.0718)
    cos_ck_ok = shown_cos_ck == 0.9878              # cat and kitten point almost the same way
    cos_cc_ok = shown_cos_cc == 0.2514              # cat and car point in unrelated directions
    cos_kc_ok = shown_cos_kc == 0.2262              # kitten and car are unrelated too
    l2_ck_ok = shown_l2_ck == 0.1694                # ...and sit close in straight-line distance
    l2_cc_ok = round(l2_cc, 4) == 1.4071            # ...while cat and car sit far apart
    diff_lines_ok = (diff_lines
                     == ["cat-kitten  difference = [ 0.1  -0.05 -0.09  0.09] -> l2 = 0.1694",
                         "cat-car     difference = [ 0.7 -0.8  0.7 -0.6] -> l2 = 1.4071",
                         "kitten-car  difference = [ 0.6  -0.75  0.79 -0.69] -> l2 = 1.4222"])
    # The computed Part-1 prediction and BOTH real rulers must order all three pairs alike.
    rulers_agree = (cos_order == l2_order == l1_order == ("cat-kitten", "cat-car", "kitten-car"))
    # ...but only ON THESE PAIRS. The counterexamples must keep contradicting the general
    # "small l2 = high cosine" rule, or the printed warning would be teaching nothing.
    counterexample_ok = (shown_cos_dim == 1.0 and shown_l2_dim == 0.9777
                         and cos_dim > cos_ck and l2_dim > l2_ck   # higher cosine, BIGGER l2
                         and shown_cos_tiny == 0.0 and shown_l2_tiny == 0.0141
                         and l2_tiny < l2_ck and cos_tiny < cos_ck)  # tiny l2, no similarity
    # The rule with its condition attached: for unit-length vectors l2^2 == 2 - 2*cosine.
    unit_identity_ok = (shown_unit_l2_sq == 0.02447 and shown_two_minus_2cos == 0.02447
                        and shown_identity_holds)
    search_order_ok = ranked_names == ("cat", "kitten", "car")     # the two cats first
    search_scores_ok = (good_scores == (0.9979, 0.9953, 0.2319)    # in that ranked order
                        and good_lines == ["  #1 cat    cosine = 0.9979",
                                           "  #2 kitten cosine = 0.9953",
                                           "  #3 car    cosine = 0.2319"]
                        and shown_good_spread == 0.7661            # the wide, meaningful gap
                        and shown_query_shape == (4,))
    grid_ok = (shown_S_shape == (2, 3) and grid_matches_loop   # rows are queries, cols stored
               and shown_E_shape == (3, 4) and shown_Q_shape == (2, 4)
               and tuple(shown_S0) == (0.9979, 0.9953, 0.2319)
               and tuple(shown_S1) == (-0.7884, -0.7754, -0.7884))
    bad_vector_ok = (tuple(shown_query_bad) == (-42.2, -48.8, -42.6, -48.5)
                     and shown_query_bad_shape == (4,))
    bad_top_ok = bad_names[0] == "kitten" and shown_cat_place == 3  # true match no longer #1
    # Sorted worst-to-best the bad scores are all negative, and cat and car tie to 4 places.
    bad_scores_ok = (bad_scores == (-0.7754, -0.7884, -0.7884)
                     and bad_lines == ["  #1 kitten cosine = -0.7754",
                                       "  #2 car    cosine = -0.7884",
                                       "  #3 cat    cosine = -0.7884"]
                     and shown_bad_spread == 0.013                 # collapsed from 0.7661
                     and shown_gap_line == "the gap deciding cat vs car: -0.0000006")

    if (embeddings_ok and l1_ok and dot_ok and cos_ck_ok and cos_cc_ok and cos_kc_ok
            and l2_ck_ok and l2_cc_ok and diff_lines_ok and rulers_agree
            and counterexample_ok and unit_identity_ok
            and search_order_ok and search_scores_ok and grid_ok and bad_vector_ok
            and bad_top_ok and bad_scores_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected cosine(cat,kitten) == 0.9878, cosine(cat,car) == 0.2514, "
              "l2(cat,kitten) == 0.1694, l2(cat,car) == 1.4071, all three rulers to order the "
              "pairs ('cat-kitten', 'cat-car', 'kitten-car'), cat vs 0.1*cat to break the "
              "'small l2 = high cosine' rule with cosine 1.0 at l2 0.9777, l2^2 == 2 - 2*cosine "
              "== 0.02447 after unit-normalizing, the ranking ('cat', 'kitten', "
              "'car') scoring (0.9979, 0.9953, 0.2319), a (2, 3) cosine grid matching the loop, "
              "and the wrongly-normalized query [-42.2, -48.8, -42.6, -48.5] to rank 'kitten' "
              "first with the scores collapsed to (-0.7754, -0.7884, -0.7884)")

    # These asserts make the check hard: a wrong fact stops the program.
    assert embeddings_ok, "the three stand-in embeddings should print with shape (4,)"
    assert l1_ok, "mean |difference| should be 0.0825 / 0.7 / 0.7075 and predict cat-kitten"
    assert dot_ok, "dot(cat, kitten) should be 1.15, with lengths 1.0863 and 1.0718"
    assert cos_ck_ok, "cosine(cat, kitten) should be 0.9878 — near 1, the same direction"
    assert cos_cc_ok, "cosine(cat, car) should be 0.2514 — far nearer 0 than 1"
    assert cos_kc_ok, "cosine(kitten, car) should be 0.2262 — unrelated as well"
    assert l2_ck_ok, "l2(cat, kitten) should be 0.1694 — a small straight-line distance"
    assert l2_cc_ok, "l2(cat, car) should be 1.4071 — a large straight-line distance"
    assert diff_lines_ok, "each difference vector should print with its own l2 beside it"
    assert rulers_agree, "mean-|diff|, cosine and L2 must all order these three pairs alike"
    assert counterexample_ok, ("the counterexamples must still break the general rule: "
                              "cat vs 0.1*cat is cosine 1.0 at l2 0.9777 (higher cosine, "
                              "bigger l2), and two tiny right-angle vectors are l2 0.0141 "
                              "at cosine 0.0")
    assert unit_identity_ok, "after unit-normalizing, l2^2 must equal 2 - 2*cosine (0.02447)"
    assert search_order_ok, "the query should rank cat, then kitten, then car"
    assert search_scores_ok, "those three cosines should be 0.9979, 0.9953, 0.2319"
    assert grid_ok, "the cosine grid must be (2 queries, 3 stored) and match the loop"
    assert bad_vector_ok, "the wrong recipe should give [-42.2, -48.8, -42.6, -48.5]"
    assert bad_top_ok, "the wrongly-normalized query should rank kitten first, cat third"
    assert bad_scores_ok, "its scores should collapse to (-0.7754, -0.7884, -0.7884)"
