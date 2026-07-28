# day-06-random-seeds — experiment
#
# Today's big idea in two lines of output:
#   With no seed, the "random" numbers are different every run — a bug you saw
#   once may never come back. With the same seed, the exact same numbers return.
#
# This script (1) draws twice with no seed, (2) proves np.random.seed(0) replays
# the same stream, (3) shows default_rng(0) and default_rng(1) are two different
# but each-repeatable streams, (4) walks the lesson's tiny hand recipe, and
# (5) shows the silent failure when four workers share one seed.
# Run it:  python3 sessions/m01-shape-of-data/day-06-random-seeds/experiment.py

import numpy as np  # numpy holds the random-number makers this lesson is about


# ---- Helpers --------------------------------------------------------------

def tiny_prng(seed, count):
    """The lesson's hand-sized recipe: next_state = (5 * state + 3) mod 16.

    A PRNG (pseudo-random number generator) is a fixed formula, not real
    randomness. It keeps one hidden number called the state. Each step scrambles
    the state and hands that value back. The seed is the very first state.
    """
    state = seed
    stream = [state]
    for _ in range(count - 1):
        state = (5 * state + 3) % 16   # the fixed scrambling formula f(state)
        stream.append(state)
    return stream


def worker_stream(base_seed, worker_id, count):
    """Two random draws for one data-loading worker.

    A worker is one parallel helper that prepares training data. Passing
    worker_id = 0 for everybody means every worker starts from the same state.
    """
    rng = np.random.default_rng(base_seed + worker_id)
    return rng.random(count)


if __name__ == "__main__":
    # --- Part 1: no seed — two draws that differ, and change every run ----
    # No seed is set here on purpose. These are the only two numbers in this
    # whole script that are different each time you run it. That is the point.
    first_draw = np.random.rand()    # np.random.rand() gives one number in [0, 1)
    second_draw = np.random.rand()   # the hidden state moved on, so this differs
    print("no seed, draw 1 :", round(float(first_draw), 6))
    print("no seed, draw 2 :", round(float(second_draw), 6))
    print("the two differ  :", first_draw != second_draw,
          "(and both change if you run this file again)")

    # --- Part 2: same seed, same stream (the old global way) --------------
    # np.random.seed(0) sets the starting state of one generator shared by the
    # whole program. The lesson shows this stream as [0.5488, 0.7152, 0.6028].
    np.random.seed(0)
    a = np.random.rand(3)
    np.random.seed(0)                # rewind to the exact same starting state
    b = np.random.rand(3)
    print("\nnp.random.seed(0) -> a =", np.round(a, 4), " shape", a.shape)
    print("np.random.seed(0) -> b =", np.round(b, 4), " shape", b.shape)
    same_stream = np.array_equal(a, b)
    print("a and b identical      :", same_stream, "(equal to the last digit)")

    # --- Part 3: two seeds, two streams, each one repeatable --------------
    # default_rng is the modern way: your own generator object, so two parts of
    # your code never step on each other's hidden state.
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    s0 = rng0.random(3)
    s1 = rng1.random(3)
    print("\ndefault_rng(0).random(3) =", np.round(s0, 4))  # lesson: 0.6370 0.2698 0.0409
    print("default_rng(1).random(3) =", np.round(s1, 4))    # lesson: 0.5118 0.9505 0.1442
    streams_differ = not np.array_equal(s0, s1)
    print("the two streams differ   :", streams_differ)

    # Each stream is still repeatable: build seed 0 again and draw again.
    s0_again = np.random.default_rng(0).random(3)
    s1_again = np.random.default_rng(1).random(3)
    each_repeats = np.array_equal(s0, s0_again) and np.array_equal(s1, s1_again)
    print("rebuilt default_rng(0)   =", np.round(s0_again, 4))
    print("each stream repeats      :", each_repeats)

    # --- Part 4: why the same seed always gives the same numbers ----------
    # Run the lesson's recipe by hand. Seed 1 gives the stream 1, 8, 11, 10.
    hand_stream = tiny_prng(seed=1, count=4)
    hand_again = tiny_prng(seed=1, count=4)
    hand_seed2 = tiny_prng(seed=2, count=4)
    print("\nf(s) = (5*s + 3) mod 16")
    print("seed 1 ->", hand_stream, " run it again ->", hand_again)
    print("seed 2 ->", hand_seed2, " (a different start, a different stream)")
    print("-> one sentence: the seed IS the first state, and every next number is"
          " a fixed formula of the state, so the same seed replays the same chain")

    # --- Part 5: the silent failure — four workers sharing one seed -------
    # Every worker built its generator from the same number, so their "random"
    # draws are copies of each other. Nothing errors. The data loses variety.
    shared = np.array([worker_stream(0, 0, 2) for _ in range(4)])
    # The fix: offset the seed by the worker id, so each worker starts elsewhere.
    offset = np.array([worker_stream(0, w, 2) for w in range(4)])
    print("\nfour workers, one shared seed:\n", np.round(shared, 4), " shape", shared.shape)
    print("four workers, seed + worker_id:\n", np.round(offset, 4), " shape", offset.shape)
    shared_rows_identical = len(np.unique(shared, axis=0)) == 1
    offset_rows_all_different = len(np.unique(offset, axis=0)) == 4
    print("shared-seed rows are copies :", shared_rows_identical, "(the silent bug)")
    print("offset rows are all distinct:", offset_rows_all_different, "(the fix)")

    # --- Self-check: assert the values the lesson states ------------------
    unseeded_differ = bool(first_draw != second_draw)
    # The exact numbers the lesson's playground prints, to four places.
    a_matches = np.allclose(np.round(a, 4), [0.5488, 0.7152, 0.6028])
    # Full precision from a real run; the lesson cuts the last one to 0.0409.
    s0_matches = np.allclose(s0, [0.63696169, 0.26978671, 0.04097352])
    s1_matches = np.allclose(s1, [0.51182162, 0.95046370, 0.14415961])
    hand_matches = (hand_stream == [1, 8, 11, 10]) and (hand_again == hand_stream)
    worker_bug_shown = shared_rows_identical and offset_rows_all_different

    all_ok = (unseeded_differ and same_stream and a_matches and streams_differ
              and each_repeats and s0_matches and s1_matches and hand_matches
              and worker_bug_shown)

    if all_ok:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected two unseeded draws to differ, seed(0) to give"
              " [0.5488, 0.7152, 0.6028] twice, default_rng(0) and default_rng(1) to"
              " give different but each-repeatable streams, and the hand recipe from"
              " seed 1 to give [1, 8, 11, 10]")

    # These asserts stop the program if any claim above is wrong.
    assert unseeded_differ, "two draws with no seed should differ"
    assert same_stream, "the same np.random.seed(0) must replay the exact same array"
    assert a_matches, "np.random.seed(0); rand(3) should be [0.5488, 0.7152, 0.6028]"
    assert streams_differ, "default_rng(0) and default_rng(1) must give different streams"
    assert each_repeats, "rebuilding default_rng(0) must replay the same stream"
    assert s0_matches, "default_rng(0).random(3) should start 0.6370, 0.2698, 0.0410"
    assert s1_matches, "default_rng(1).random(3) should start 0.5118, 0.9505, 0.1442"
    assert hand_matches, "f(s)=(5s+3) mod 16 from seed 1 should give [1, 8, 11, 10]"
    assert worker_bug_shown, "one shared seed must copy the workers; +worker_id must not"
