# day-01-embeddings - experiment
#
# GOAL (from today's Produce task): turn words into little lists of numbers
# (embeddings), then measure "how alike are two words?" with ONE friendly score
# called cosine similarity. We prove four things you predicted in the lesson:
#   1. cat and dog score HIGH (they are both furry animals)        -> about 0.99
#   2. cat and car score LOW  (a car is not an animal)             -> about 0.08
#   3. the SAME frozen "bank" row scored against itself is a perfect 1.00,
#      which is the whole "context-free limit" the next lessons will fix.
#   4. "dog bites man" and "man bites dog" hand back the SAME SET of rows,
#      which is the "position limit" (nothing inside a row marks its slot).
# We also add the one-hot contrast (the old, clumsy way) to see it score 0.0.
#
# Run it:  python3 sessions/m03-attention/day-01-embeddings/experiment.py

import numpy as np  # numpy just makes the little list-of-numbers math tidy


# ---- Step 1: the vocabulary --------------------------------------------------
# A "vocabulary" is the full set of words the model knows. Each word gets a
# ticket number (its "token id"). The number is just the ROW it lives on in the
# big table below. cat is ticket 0, dog is ticket 1, and so on. Our toy wall
# knows SEVEN words, so every one-hot list further down is seven slots wide.
vocab = {"cat": 0, "dog": 1, "car": 2, "the": 3, "bank": 4, "bites": 5, "man": 6}


# ---- Step 2: turn a sentence into ticket numbers -----------------------------
# A computer cannot read letters as meaning. So the first job is: word -> number.
# We split the sentence into words and look up each word's ticket in the vocab.
sentence = "the cat"
token_ids = [vocab[word] for word in sentence.split()]  # "the"->3, "cat"->0


# ---- Step 3: the embedding table (the big wall of numbered hooks) ------------
# The "embedding table" (embedding matrix) has ONE ROW per word. Row i holds the
# little list of numbers for the word whose ticket id is i. Every row has the
# SAME length (4 numbers here) -- that length is called d_model. Our four made-up
# column meanings are: is-animal, is-furry, is-vehicle, is-fast. We hand-set the
# rows so cat and dog look alike (both high on the first two scores) and car is
# the odd one out (its big score sits on the third slot instead).
E = np.array([
    [0.9, 0.8, 0.0, 0.3],   # row 0 = cat    (animal, furry, quite fast)
    [0.8, 0.9, 0.0, 0.4],   # row 1 = dog    (animal, furry, quite fast)
    [0.0, 0.0, 0.9, 0.3],   # row 2 = car    (vehicle, not an animal)
    [0.1, 0.0, 0.1, 0.1],   # row 3 = the    (a glue word: every score tiny)
    [0.4, 0.4, 0.4, 0.4],   # row 4 = bank   <- ONE row, both meanings
    [0.2, 0.1, 0.0, 0.6],   # row 5 = bites  (a verb: a quick action)
    [0.3, 0.1, 0.0, 0.2],   # row 6 = man    (an animal, but not furry)
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
#   about -1 = opposite way (pointing back the other way).
# In plain words: multiply the two lists position-by-position, add it all up
# (that sum is the "dot product"), then divide by how long each arrow is.
def cosine(a, b):
    dot = float(np.dot(a, b))                # multiply position-by-position, add up
    length_a = float(np.linalg.norm(a))      # how long arrow a is
    length_b = float(np.linalg.norm(b))      # how long arrow b is
    return dot / (length_a * length_b)       # divide out length -> only direction counts


# ---- Step 6: the one-hot contrast (the old, clumsy way) ----------------------
# One-hot turns a word into a giant list of zeros with a single 1 in that word's
# slot -- one slot per KNOWN word, so seven slots here (50,000 in a real model).
# cat is slot 0, dog is slot 1 (different slots!). Because the two 1s never line
# up, their arrows sit at a perfect right angle -> cosine is exactly 0.0.
# That is the fatal flaw: one-hot cannot tell ANY two words apart.
def onehot(word):
    vec = np.zeros(len(vocab))   # all zeros, one slot per known word
    vec[vocab[word]] = 1.0       # flip on the single slot for this word
    return vec


# ---- Step 7: a whole sentence becomes a list of row numbers ------------------
# Embedding a sentence = look up each word's ticket, in order. The ORDER is real,
# but nothing INSIDE a row says which place it came from -- which is why the two
# opposite sentences below hand back the very same SET of rows.
def row_ids(text):
    return [vocab[word] for word in text.split()]


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

    # -- Why we divide the length out: same word, twice as long an arrow ------
    # Doubling every number does not change the word's meaning at all. The raw
    # dot product doubles anyway; the cosine does not move. That is exactly why
    # a ranking by raw score can be won on arrow SIZE instead of direction.
    doubled = 2.0 * embed("cat")
    print("dot(cat, cat)      =", round(float(np.dot(embed("cat"), embed("cat"))), 2),
          "  cos =", round(cosine(embed("cat"), embed("cat")), 2))
    print("dot(cat, 2 x cat)  =", round(float(np.dot(embed("cat"), doubled)), 2),
          "  cos =", round(cosine(embed("cat"), doubled), 2), " <- raw doubled, cosine did not")

    # -- The context-free limit: same frozen row for two meanings of "bank" ---
    # A static table has ONE row for "bank" (row 4). So the river-bank and the
    # money-bank both fetch the SAME list of numbers from the SAME table.
    # Scoring that row against itself is a perfect 1.00 -- the table literally
    # cannot tell the two meanings apart.
    bank_river = embed("bank")             # "...the bank of the river"
    bank_money = embed("bank")             # "...money in the bank"
    print("bank(river) -> row", vocab["bank"], "->", bank_river)
    print("bank(money) -> row", vocab["bank"], "->", bank_money)
    bank_score = round(cosine(bank_river, bank_money), 2)   # expect 1.00
    print("cosine(bank_river, bank_money) =", bank_score, " <- context-free limit")

    # -- The one-hot contrast: blind to similarity ---------------------------
    onehot_cat_dog = round(cosine(onehot("cat"), onehot("dog")), 2)   # expect 0.0
    print("one-hot width:", len(vocab), "slots (one per known word)")
    print("cosine(onehot cat, onehot dog) =", onehot_cat_dog, " <- one-hot is blind")

    # -- The position limit: opposite stories, identical set of rows ----------
    ids_a = row_ids("dog bites man")
    ids_b = row_ids("man bites dog")
    same_set = set(ids_a) == set(ids_b)
    print("rows for 'dog bites man':", ids_a)
    print("rows for 'man bites dog':", ids_b)
    print("same SET of rows?", same_set, " <- position limit")

    # -- Self-check: assert every number matches the lesson's stated values ---
    expected = {
        "cat_dog": 0.99,   # animals, very alike
        "cat_car": 0.08,   # unrelated, near zero
        "bank": 1.00,      # context-free limit
        "onehot": 0.0,     # one-hot cannot see similarity
        "same_set": True,  # position limit
    }
    got = {"cat_dog": cat_dog, "cat_car": cat_car,
           "bank": bank_score, "onehot": onehot_cat_dog, "same_set": same_set}

    if got == expected:
        print("✅ you got it — embeddings see that cat~dog (0.99) beats cat~car (0.08),")
        print("   the frozen 'bank' row scores itself 1.00 (context-free limit),")
        print("   one-hot is blind (0.0), and both word orders give the same rows.")
    else:
        print("❌ not yet — expected", expected, "but got", got)

    # A failed assert stops the script with a clear message; a passed one is silent.
    assert got == expected, "expected %r but got %r" % (expected, got)
