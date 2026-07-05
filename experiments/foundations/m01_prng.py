"""
Module 1 · Day 6 artifact — Random Seeds & PRNG.

Shows three things you can prove to yourself:
  1. With no seed, two draws differ.
  2. With the SAME seed, the stream repeats exactly (reproducible).
  3. Two different seeds give two different — but each repeatable — streams.

Run:  python3 m01_prng.py
"""

import numpy as np


def no_seed_differs():
    # No seed set: each call advances the hidden global state, so draws differ.
    first = np.random.rand()
    second = np.random.rand()
    print("[1] no seed  -> first :", first)
    print("[1] no seed  -> second:", second)
    # They are (almost surely) different from each other and from run to run.
    print("[1] different from each other:", first != second)
    print()


def same_seed_reproduces():
    # Seed the global generator, draw an array.
    np.random.seed(0)
    a = np.random.rand(3)
    # Reseed with the SAME seed, draw again.
    np.random.seed(0)
    b = np.random.rand(3)
    print("[2] seed 0 -> a:", a)
    print("[2] seed 0 -> b:", b)
    # Same seed => identical stream, to the last bit.
    assert np.array_equal(a, b), "same seed must reproduce the same stream"
    print("[2] assert np.array_equal(a, b) PASSED — reproducible")
    print()


def different_seeds_differ():
    # The modern way: make your own generator objects, one per seed.
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    s0 = rng0.random(3)
    s1 = rng1.random(3)
    print("[3] default_rng(0) -> s0:", s0)
    print("[3] default_rng(1) -> s1:", s1)
    # Different seeds => different streams (each still repeatable on its own).
    assert not np.array_equal(s0, s1), "different seeds must give different streams"
    print("[3] assert not np.array_equal(s0, s1) PASSED — two distinct streams")
    print()

    # Bonus: each stream is itself reproducible.
    s0_again = np.random.default_rng(0).random(3)
    assert np.array_equal(s0, s0_again), "default_rng(0) must be reproducible"
    print("[3] default_rng(0) repeats itself PASSED — each stream is reproducible")


if __name__ == "__main__":
    no_seed_differs()
    same_seed_reproduces()
    different_seeds_differ()
    print("\nAll asserts passed. Same seed = same stream; different seed = different stream.")
