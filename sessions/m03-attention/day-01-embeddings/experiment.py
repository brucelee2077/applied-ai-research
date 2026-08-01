# day-01-embeddings — experiment
#
# Today's big idea in two lines of output:
#   A word turns into one ROW of numbers, and one score reads meaning off those
#   rows: cat-vs-dog = 0.99 while cat-vs-car = 0.08, and no spelling is used.
#
# It also proves today's two limits: one frozen row per word, and no SEAT marker.
# Vocabulary note for the whole module: a "slot" is always a COLUMN of a row, counted
# from 1 the way the lesson counts (slot 1 = the first column, is-animal here), and a
# "seat" is always a word's position in the sentence. Day 5 stamps the seats in.
# Run it:  python3 sessions/m03-attention/day-01-embeddings/experiment.py

import numpy as np  # numpy keeps the row math (dot product, arrow length) tidy

# ---- The vocabulary: a ticket number (token id) per word = the ROW it sits on ----
vocab = {"cat": 0, "dog": 1, "car": 2, "the": 3, "bank": 4, "bites": 5, "man": 6}

# ---- The embedding table: one row per word, 4 numbers long ------------------
# Row i = the word with token id i. Made-up columns: is-animal, is-furry, is-vehicle,
# is-fast. These are the exact rows the lesson draws.
E = np.array([
    [0.9, 0.8, 0.0, 0.3],   # row 0 = cat    animal, furry, quite fast
    [0.8, 0.9, 0.0, 0.4],   # row 1 = dog    animal, furry, quite fast
    [0.0, 0.0, 0.9, 0.3],   # row 2 = car    vehicle, not an animal
    [0.1, 0.0, 0.1, 0.1],   # row 3 = the    a glue word: every score tiny
    [0.4, 0.4, 0.4, 0.4],   # row 4 = bank   ONE row for both meanings
    [0.2, 0.1, 0.0, 0.6],   # row 5 = bites  a verb: a quick action
    [0.3, 0.1, 0.0, 0.2],   # row 6 = man    an animal, but not furry
])

def row_ids(text):
    """Turn a sentence into its list of token ids, in reading order."""
    return [vocab[word] for word in text.split()]

def embed(word):
    """A lookup means grabbing a row: hand over the ticket number, get the row."""
    return E[vocab[word]]

def cosine(a, b):
    # Multiply the two rows slot by slot and add it up (that sum is the dot
    # product), then divide by each row's length so only direction is left.
    return float(np.dot(a, b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))

def onehot(word):
    # The old way: all zeros with a single 1 in this word's slot, one slot per known word.
    vec = np.zeros(len(vocab))
    vec[vocab[word]] = 1.0
    return vec


if __name__ == "__main__":
    # --- Part 1: a word becomes a ticket number, then a row ---------------
    token_ids = row_ids("the cat")   # split the sentence, then look up each ticket
    print("table shape:", E.shape, " 'the cat' ->", token_ids, "(the->3, cat->0)")
    print("row for 'cat' (id 0):", embed("cat"), "<- that is the whole lookup")
    # A glue word like "the" sits near the middle of the map, so its arrow is short.
    the_length = round(float(np.linalg.norm(embed("the"))), 2)
    print("arrow length of 'the':", the_length, "(short arrows swing direction easily)")

    # --- Part 2: predict from the table, then measure ---------------------
    # The prediction is COMPUTED, not typed in: more columns above zero in BOTH rows
    # should mean a higher score, and changing the table changes the prediction too.
    shared_dog = int(np.sum((embed("cat") > 0) & (embed("dog") > 0)))
    shared_car = int(np.sum((embed("cat") > 0) & (embed("car") > 0)))
    predicted = "cat-dog" if shared_dog > shared_car else "cat-car"
    print("\nshared columns: cat-dog", shared_dog, " cat-car", shared_car, "->", predicted)
    # Column 2 sits exactly ON the boundary: it is 0.0 in BOTH animal rows. An empty
    # slot is not a shared one, so "above zero" has to stay strict. The loose count is
    # the contrast control: >= 0 would call that empty column shared and report 4.
    loose_dog = int(np.sum((embed("cat") >= 0) & (embed("dog") >= 0)))
    print("column 2 is", embed("cat")[2], "in cat and", embed("dog")[2], "in dog ->",
          "strict >0 counts", shared_dog, " loose >=0 counts", loose_dog)
    dot_cat_dog = round(float(np.dot(embed("cat"), embed("dog"))), 2)
    len_cat = round(float(np.linalg.norm(embed("cat"))), 2)
    len_dog = round(float(np.linalg.norm(embed("dog"))), 2)
    cat_dog = round(cosine(embed("cat"), embed("dog")), 2)
    cat_car = round(cosine(embed("cat"), embed("car")), 2)
    print("cosine(cat, dog) =", dot_cat_dog, "/ (", len_cat, "x", len_dog, ") =", cat_dog,
          "  cosine(cat, car) =", cat_car)
    measured = "cat-dog" if cat_dog > cat_car else "cat-car"
    print("measured winner:", measured, " prediction held?", measured == predicted)
    # Same word, arrow twice as long: the raw total doubles, the cosine does not.
    doubled = 2.0 * embed("cat")
    dot_plain = round(float(np.dot(embed("cat"), embed("cat"))), 2)
    dot_doubled = round(float(np.dot(embed("cat"), doubled)), 2)
    cos_doubled = round(cosine(embed("cat"), doubled), 2)
    print("doubling cat: dot", dot_plain, "->", dot_doubled, " cosine stays", cos_doubled)
    # Worth knowing before Day 2: attention scores words with the RAW dot product, not
    # with cosine, and cures the big-score problem a different way — it divides by the
    # square root of the WIDTH (how many columns), not by each arrow's length. Cosine
    # attention is a real alternative design, not something Day 2 forgot.

    # --- Part 3: limit one — one frozen row per word ----------------------
    # Our wall of seven words cannot spell the lesson's two sentences, so "the bank"
    # and "man bank" stand in for them ("bank" is the last word in both).
    river_ids, money_ids = row_ids("the bank"), row_ids("man bank")
    # the row fetched for "bank" in each sentence (it is the last word of each)
    bank_from_river, bank_from_money = E[river_ids][-1], E[money_ids][-1]
    bank_score = round(cosine(bank_from_river, bank_from_money), 2)
    print("\ncontexts", river_ids, "vs", money_ids, "-> bank", bank_from_river, "and",
          bank_from_money, " cosine =", bank_score, "<- context-free limit")
    # Careful: cosine(row, row) is 1.00 for EVERY row, so the self-check pins the row.

    # --- Part 4: the one-hot contrast ------------------------------------
    onehot_cos = {a + "-" + b: round(cosine(onehot(a), onehot(b)), 2)
                  for a, b in [("cat", "dog"), ("cat", "car"), ("dog", "car")]}
    print("\none-hot 'cat':", onehot("cat"), " width:", len(onehot("cat")), "slots")
    print("one-hot cosines:", onehot_cos, "<- every pair scores the same, all blind")

    # --- Part 5: limit two — nothing inside a row marks its seat ----------
    ids_a, ids_b = row_ids("dog bites man"), row_ids("man bites dog")
    stack_a, stack_b = E[ids_a], E[ids_b]
    bag_a = np.round(stack_a.sum(axis=0), 2)   # add the rows up: the order is gone
    bag_b = np.round(stack_b.sum(axis=0), 2)
    # Compute each printed answer ONCE, so the print and the self-check cannot drift apart.
    same_set = set(ids_a) == set(ids_b)
    same_ordered = np.array_equal(stack_a, stack_b)
    print("\n'dog bites man' ->", ids_a, " 'man bites dog' ->", ids_b, " stacks:", stack_a.shape)
    print("same SET of rows?", same_set, " same ORDERED rows?", same_ordered,
          " added up:", bag_a, "vs", bag_b)

    # --- Self-check: one boolean per claim, pinned to numbers read off a real run --
    lookup_ok = (token_ids == [3, 0] and E.shape == (7, 4) and the_length == 0.17
                 and np.array_equal(embed("cat"), np.array([0.9, 0.8, 0.0, 0.3])))
    scores_ok = (dot_cat_dog == 1.56 and len_cat == 1.24 and len_dog == 1.27
                 and cat_dog == 0.99 and cat_car == 0.08)
    predict_ok = predicted == measured     # a computed guess against a measurement
    # Pin the counts themselves, not just their order: 3 vs 1 is WHY cat-dog was picked,
    # and the loose count pins the boundary column to the not-shared side.
    columns_ok = shared_dog == 3 and shared_car == 1 and loose_dog == 4
    size_ok = (dot_plain == 1.54 and dot_doubled == 3.08 and cos_doubled == 1.0)
    # The frozen ROW, not the self-similarity score, carries the claim here.
    bank_ok = (np.array_equal(bank_from_river, np.array([0.4, 0.4, 0.4, 0.4]))
               and np.array_equal(bank_from_money, bank_from_river) and bank_score == 1.0
               and river_ids == [3, 4] and money_ids == [6, 4])
    onehot_ok = (np.array_equal(onehot("cat"), np.array([1.0, 0, 0, 0, 0, 0, 0]))
                 and len(onehot("cat")) == 7 and set(onehot_cos.values()) == {0.0})
    # A set cannot notice a reordering, so pin the exact id lists and the total. The two
    # booleans asserted here are the SAME values that were printed above.
    order_ok = (ids_a == [1, 5, 6] and ids_b == [6, 5, 1] and same_set
                and not same_ordered and np.array_equal(bag_b, bag_a)
                and np.array_equal(bag_a, np.array([1.3, 1.1, 0.0, 1.2])))
    claims = {"lookup": lookup_ok, "scores": scores_ok, "prediction": predict_ok,
              "columns": columns_ok, "size": size_ok, "frozen": bank_ok,
              "one-hot": onehot_ok, "order": order_ok}
    if all(claims.values()):
        print("\n✅ you got it — cat~dog 0.99 beats cat~car 0.08 (1.56 / (1.24 x 1.27)),")
        print("   one-hot gives every pair 0.0, both 'bank' lookups return [0.4 0.4 0.4 0.4],")
        print("   and both word orders add up to the same [1.3 1.1 0. 1.2].")
    else:
        print("\n❌ not yet — expected ids [3, 0] / length('the') 0.17 / dot 1.56 / cosines")
        print("   0.99 and 0.08 / 1.54 -> 3.08, cosine 1.0 / bank [0.4 0.4 0.4 0.4] from ids")
        print("   [3, 4] and [6, 4] / one-hot 0.0 / [1,5,6] vs [6,5,1] / total [1.3 1.1 0. 1.2]")
        print("   Failed:", [k for k, v in claims.items() if not v])

    # A failed assert stops the script with a message; a passed one stays quiet.
    assert lookup_ok, "'the cat' -> ids [3, 0], row 0 = [0.9 0.8 0. 0.3], 'the' length 0.17"
    assert scores_ok, "expected dot 1.56, lengths 1.24 and 1.27, cosines 0.99 and 0.08"
    assert predict_ok, "the shared-column prediction and the measured winner must agree"
    assert columns_ok, "cat-dog shares 3 above-zero columns and cat-car 1 (loose >=0: 4)"
    assert size_ok, "doubling cat must double the dot (1.54 -> 3.08) and leave cosine 1.0"
    assert bank_ok, "both sentences must fetch the same frozen row [0.4 0.4 0.4 0.4]"
    assert onehot_ok, "one-hot must be 7 wide and score 0.0 for every pair"
    assert order_ok, "expected ids [1,5,6] vs [6,5,1], different stacks, total [1.3 1.1 0. 1.2]"
