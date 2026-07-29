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
    for name, emb in db.items():
        print("%-6s -> %s  shape %s" % (name, emb, emb.shape))
    print("(a real backbone gives shape (512,); kitten is slot-by-slot close to cat, car is not)")
    # Predict before measuring — but a COMPUTED prediction, not a typed one. Mean |a - b|
    # per slot uses no lengths and no dot product, so it is an independent third vote.
    pairs = [("cat-kitten", cat, kitten), ("cat-car", cat, car), ("kitten-car", kitten, car)]
    l1 = {name: float(np.abs(a - b).mean()) for name, a, b in pairs}
    print("mean |difference| per slot:", {k: round(v, 4) for k, v in l1.items()})
    predicted_closest = min(l1, key=l1.get)
    print("-> prediction from that crude ruler: the closest pair is", predicted_closest)

    # --- Part 2: cosine similarity ----------------------------------------
    print("\ndot(cat, kitten) =", round(float(cat @ kitten), 4), " length of cat =",
          round(float(np.linalg.norm(cat)), 4), " of kitten =", round(float(np.linalg.norm(kitten)), 4))
    cos_ck, cos_cc, cos_kc = cosine(cat, kitten), cosine(cat, car), cosine(kitten, car)
    print("cosine(cat, kitten) =", round(cos_ck, 4), "-> near 1: same direction, similar")
    print("cosine(cat, car)    =", round(cos_cc, 4), "-> far nearer 0 than 1: unrelated")
    print("cosine(kitten, car) =", round(cos_kc, 4), "-> also unrelated")
    # Honest note: every slot in these stand-ins is positive, so cosine can never reach 0 here.
    # Real 512-number embeddings have negative slots too — that is how the lesson reaches 0.11.

    # --- Part 3: Euclidean / L2 distance ----------------------------------
    l2_ck, l2_cc, l2_kc = l2(cat, kitten), l2(cat, car), l2(kitten, car)
    print("")
    for name, a, b in pairs:   # print the difference vector, then its length
        print("%-11s difference = %s -> l2 = %.4f" % (name, np.round(a - b, 4), l2(a, b)))
    cos_pairs = {"cat-kitten": cos_ck, "cat-car": cos_cc, "kitten-car": cos_kc}
    l2_pairs = {"cat-kitten": l2_ck, "cat-car": l2_cc, "kitten-car": l2_kc}
    cos_order = tuple(sorted(cos_pairs, key=lambda n: -cos_pairs[n]))  # most similar first
    l2_order = tuple(sorted(l2_pairs, key=lambda n: l2_pairs[n]))      # closest first
    l1_order = tuple(sorted(l1, key=lambda n: l1[n]))                  # the Part-1 crude ruler
    print("pairs ordered most-similar-first — cosine:", cos_order, "\n  L2:", l2_order,
          "\n  mean-|diff|:", l1_order, "-> all three rulers agree, so small l2 = high cosine")

    # --- Part 4: nearest-neighbor search, with no training at all ---------
    query = np.array([0.78, 0.12, 0.74, 0.15])   # a NEW cat photo, read by the same borrowed eye
    print("\nquery embedding:", query, "shape", query.shape, "(never stored in the db)")
    ranked = rank_by_cosine(query, db)
    for place, (name, score) in enumerate(ranked, start=1):
        print("  #%d %-6s cosine = %.4f" % (place, name, score))
    ranked_names = tuple(name for name, _ in ranked)
    good_scores = tuple(round(score, 4) for _, score in ranked)
    good_spread = ranked[0][1] - ranked[-1][1]
    print("ranking:", ranked_names, "-> both cats first, the car last: image search with no",
          "training.\nbest score minus worst:", round(good_spread, 4), "(a wide, meaningful gap)")

    # --- Part 5: the gotcha — the WRONG preprocessing ---------------------
    # The pretrained model expects its OWN recipe: its resize, its per-color average and spread.
    # Here the query is centered on 5.0 and divided by 0.1 — a recipe the eye never saw. Same
    # photo, same pixels; only the numbers arrive on the wrong scale.
    query_bad = (query - 5.0) / 0.1
    # Cosine ignores vector LENGTH, so dividing by 0.1 on its own could not change any ranking.
    # It is the -5.0 shift that swings the query away from every stored embedding.
    print("\nquery with the WRONG normalization:", query_bad, "shape", query_bad.shape)
    print("  every number is now large and negative — that scale never reached the eye")
    ranked_bad = rank_by_cosine(query_bad, db)
    for place, (name, score) in enumerate(ranked_bad, start=1):
        print("  #%d %-6s cosine = %.4f" % (place, name, score))
    bad_names = tuple(name for name, _ in ranked_bad)
    bad_scores = tuple(round(score, 4) for _, score in ranked_bad)
    bad, bad_spread = dict(ranked_bad), ranked_bad[0][1] - ranked_bad[-1][1]
    print("ranking:", bad_names, "-> 'cat', the true match, fell from #1 to #%d."
          % (bad_names.index("cat") + 1), "\nbest score minus worst:", round(bad_spread, 4),
          "(was", str(round(good_spread, 4)) + ") -> squashed together, and all negative")
    print("the gap deciding cat vs car: %.7f" % (bad["cat"] - bad["car"]),
          "-> noise, not meaning: the ranking has stopped carrying information")

    # Both queries in ONE matrix multiply — how a real index scores many images at once.
    E = np.stack([cat, kitten, car])                        # (3 stored images, 4 numbers)
    Q = np.stack([query, query_bad])                        # (2 queries, 4 numbers)
    E_unit = E / np.linalg.norm(E, axis=1, keepdims=True)    # divide each ROW by its length
    Q_unit = Q / np.linalg.norm(Q, axis=1, keepdims=True)    # axis=1 = across the 4 numbers
    S = Q_unit @ E_unit.T                                   # (queries, stored) cosine grid
    print("\nE shape", E.shape, " Q shape", Q.shape, " cosine grid shape", S.shape)
    print("  row 0, good query, db order (cat, kitten, car):", np.round(S[0], 4))
    print("  row 1, bad query,  db order (cat, kitten, car):", np.round(S[1], 4))
    grid_matches_loop = (np.allclose(S[0], [cosine(query, e) for e in E])
                         and np.allclose(S[1], [cosine(query_bad, e) for e in E]))
    print("  same numbers as the one-pair-at-a-time loop?", grid_matches_loop)

    # --- Part 6: the one-line takeaway ------------------------------------
    print("\ntakeaway: transfer learning borrows a trained backbone; its penultimate features"
          "\nare an embedding; similar images have close embeddings (high cosine / small L2), so"
          "\nnearest-neighbor search needs no training — but only with the model's exact preprocessing.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down here, so a broken
    # change above cannot quietly re-derive its own "expected" value.
    cos_ck_ok = round(cos_ck, 4) == 0.9878          # cat and kitten point almost the same way
    cos_cc_ok = round(cos_cc, 4) == 0.2514          # cat and car point in unrelated directions
    l2_ck_ok = round(l2_ck, 4) == 0.1694            # ...and sit close in straight-line distance
    l2_cc_ok = round(l2_cc, 4) == 1.4071            # ...while cat and car sit far apart
    # The computed Part-1 prediction and BOTH real rulers must order all three pairs alike.
    rulers_agree = (cos_order == l2_order == l1_order == ("cat-kitten", "cat-car", "kitten-car"))
    search_order_ok = ranked_names == ("cat", "kitten", "car")     # the two cats first
    search_scores_ok = good_scores == (0.9979, 0.9953, 0.2319)     # in that ranked order
    grid_ok = S.shape == (2, 3) and grid_matches_loop              # rows are queries, cols stored
    bad_vector_ok = tuple(np.round(query_bad, 4)) == (-42.2, -48.8, -42.6, -48.5)
    bad_top_ok = bad_names[0] == "kitten"           # the true match is no longer #1
    # Sorted worst-to-best the bad scores are all negative, and cat and car tie to 4 places.
    bad_scores_ok = bad_scores == (-0.7754, -0.7884, -0.7884)

    if (cos_ck_ok and cos_cc_ok and l2_ck_ok and l2_cc_ok and rulers_agree
            and search_order_ok and search_scores_ok and grid_ok and bad_vector_ok
            and bad_top_ok and bad_scores_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected cosine(cat,kitten) == 0.9878, cosine(cat,car) == 0.2514, "
              "l2(cat,kitten) == 0.1694, l2(cat,car) == 1.4071, all three rulers to order the "
              "pairs ('cat-kitten', 'cat-car', 'kitten-car'), the ranking ('cat', 'kitten', "
              "'car') scoring (0.9979, 0.9953, 0.2319), a (2, 3) cosine grid matching the loop, "
              "and the wrongly-normalized query [-42.2, -48.8, -42.6, -48.5] to rank 'kitten' "
              "first with the scores collapsed to (-0.7754, -0.7884, -0.7884)")

    # These asserts make the check hard: a wrong fact stops the program.
    assert cos_ck_ok, "cosine(cat, kitten) should be 0.9878 — near 1, the same direction"
    assert cos_cc_ok, "cosine(cat, car) should be 0.2514 — far nearer 0 than 1"
    assert l2_ck_ok, "l2(cat, kitten) should be 0.1694 — a small straight-line distance"
    assert l2_cc_ok, "l2(cat, car) should be 1.4071 — a large straight-line distance"
    assert rulers_agree, "mean-|diff|, cosine and L2 must all order the pairs the same way"
    assert search_order_ok, "the query should rank cat, then kitten, then car"
    assert search_scores_ok, "those three cosines should be 0.9979, 0.9953, 0.2319"
    assert grid_ok, "the cosine grid must be (2 queries, 3 stored) and match the loop"
    assert bad_vector_ok, "the wrong recipe should give [-42.2, -48.8, -42.6, -48.5]"
    assert bad_top_ok, "the wrongly-normalized query should rank kitten first, not cat"
    assert bad_scores_ok, "its scores should collapse to (-0.7754, -0.7884, -0.7884)"
