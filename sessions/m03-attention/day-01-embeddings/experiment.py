# day-01-embeddings - experiment
#
# GOAL (from today's Produce task): turn words into little lists of numbers
# (embeddings), then measure "how alike are two words?" with ONE friendly score
# called cosine similarity. We prove three things you predicted in the lesson:
#   1. cat and dog score HIGH (they are both furry animals)        -> about 0.99
#   2. cat and car score LOW  (a car is not an animal)             -> about 0.08
#   3. the SAME frozen "bank" row scored against itself is a perfect 1.00,
#      which is the whole "context-free limit" the next lessons will fix.
# We also add the one-hot contrast (the old, clumsy way) to see it score 0.0.
#
# Run it:  python3 sessions/m03-attention/day-01-embeddings/experiment.py

import numpy as np  # numpy just makes the little list-of-numbers math tidy


# ---- Step 1: the vocabulary --------------------------------------------------
# A "vocabulary" is the full set of words the model knows. Each word gets a
# ticket number (its "token id"). The number is just the ROW it lives on in the
# big table below. cat is ticket 0, dog is ticket 1, and so on.
vocab = {"cat": 0, "dog": 1, "car": 2, "the": 3}


# ---- Step 2: turn a sentence into ticket numbers -----------------------------
# A computer cannot read letters as meaning. So the first job is: word -> number.
# We split the sentence into words and look up each word's ticket in the vocab.
sentence = "the cat"
token_ids = [vocab[word] for word in sentence.split()]  # "the"->3, "cat"->0


# ---- Step 3: the embedding table (the big wall of numbered hooks) ------------
# The "embedding table" (embedding matrix) has ONE ROW per word. Row i holds the
# little list of numbers for the word whose ticket id is i. We hand-set the rows
# so cat and dog look alike (both high on the first two scores) and car is the
# odd one out (its big score sits on the third slot instead).
E = np.array([
    [0.9, 0.8, 0.0, 0.3],   # row 0 = cat
    [0.8, 0.9, 0.0, 0.4],   # row 1 = dog
    [0.0, 0.0, 0.9, 0.3],   # row 2 = car
    [0.1, 0.0, 0.1, 0.1],   # row 3 = the
])


# ---- Step 4: a lookup is JUST grabbing a row by its ticket number ------------
# To embed a word, we do not do any math. We go to its row. That is the whole
# "lookup": hand over the ticket, get the row back.
def embed(word):
    row_number = vocab[word]     # the word's ticket id
    return E[row_number]         # the row that lives at that number


# ---- Step 5: cosine similarity — do two arrows point the same way? -----------
# Each row is like an arrow pointing out from the center of a "map of meaning".
# Cosine similarity is a score for how much two arrows agree on a DIRECTION:
#   about 1 = same way (very alike), about 0 = right angle (unrelated),
#   about -1 = opposite way (opposite meaning).
# In plain words: multiply the two lists position-by-position, add it all up
# (that sum is the "dot product"), then divide by how long each arrow is.
def cosine(a, b):
    dot = float(np.dot(a, b))                # multiply position-by-position, add up
    length_a = float(np.linalg.norm(a))      # how long arrow a is
    length_b = float(np.linalg.norm(b))      # how long arrow b is
    return dot / (length_a * length_b)       # divide out length -> only direction counts


# ---- Step 6: the one-hot contrast (the old, clumsy way) ----------------------
# One-hot turns a word into a giant list of zeros with a single 1 in that word's
# slot. cat is slot 0, dog is slot 1 (different slots!). Because the two 1s never
# line up, their arrows sit at a perfect right angle -> cosine is exactly 0.0.
# That is the fatal flaw: one-hot cannot tell ANY two words apart.
def onehot(word):
    vec = np.zeros(len(vocab))   # all zeros, one slot per known word
    vec[vocab[word]] = 1.0       # flip on the single slot for this word
    return vec


if __name__ == "__main__":
    # -- Show the sentence became a list of ticket numbers --------------------
    print("sentence:", sentence)
    print("token ids:", token_ids, " (the->3, cat->0)")

    # -- Show a lookup is just grabbing a row ---------------------------------
    cat_row = embed("cat")
    print("row for 'cat' (id 0):", cat_row)   # -> [0.9 0.8 0.  0.3]

    # -- Score the two pairs: animals should win over cat-vs-car --------------
    cat_dog = round(cosine(embed("cat"), embed("dog")), 2)   # expect ~0.99
    cat_car = round(cosine(embed("cat"), embed("car")), 2)   # expect ~0.08
    print("cosine(cat, dog) =", cat_dog, " <- animals, very alike")
    print("cosine(cat, car) =", cat_car, " <- unrelated, near zero")

    # -- The context-free limit: same frozen row for two meanings of "bank" ---
    # A static table has ONE row for "bank". So the river-bank and the money-bank
    # both fetch the SAME list of numbers. Scoring that row against itself is a
    # perfect 1.00 -- the table literally cannot tell the two meanings apart.
    bank_row = np.array([0.4, 0.4, 0.4])   # the single frozen "bank" row
    bank_river = bank_row                  # "...the bank of the river"
    bank_money = bank_row                  # "...money in the bank"
    bank_score = round(cosine(bank_river, bank_money), 2)   # expect 1.00
    print("cosine(bank_river, bank_money) =", bank_score, " <- context-free limit")

    # -- The one-hot contrast: blind to similarity ---------------------------
    onehot_cat_dog = round(cosine(onehot("cat"), onehot("dog")), 2)   # expect 0.0
    print("cosine(onehot cat, onehot dog) =", onehot_cat_dog, " <- one-hot is blind")

    # -- Self-check: assert every number matches the lesson's stated values ---
    expected = {
        "cat_dog": 0.99,   # animals, very alike
        "cat_car": 0.08,   # unrelated, near zero
        "bank": 1.00,      # context-free limit
        "onehot": 0.0,     # one-hot cannot see similarity
    }
    got = {"cat_dog": cat_dog, "cat_car": cat_car,
           "bank": bank_score, "onehot": onehot_cat_dog}

    if got == expected:
        print("✅ you got it — embeddings see that cat~dog (0.99) beats cat~car (0.08),")
        print("   the frozen 'bank' row scores itself 1.00 (context-free limit),")
        print("   and one-hot is blind (0.0).")
    else:
        print("❌ not yet — expected", expected, "but got", got)

    # A failed assert stops the script with a clear message; a passed one is silent.
    assert got == expected, "expected %r but got %r" % (expected, got)
