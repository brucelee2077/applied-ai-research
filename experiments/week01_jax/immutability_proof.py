"""
Week 1 Day 1 — Proof: NumPy arrays are MUTABLE, JAX arrays are IMMUTABLE.

The whole point of JAX's functional style is that arrays never change in place.
This script proves that empirically in three steps and fails loudly if any
step behaves differently from what we claim.

Run:   python3 experiments/week01_jax/immutability_proof.py
Exit:  0 = all three demonstrations behaved exactly as expected.
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp


def step1_numpy_is_mutable() -> None:
    """NumPy edits happen in place: the VALUES change but the object (its id) does not."""
    print("=" * 64)
    print("STEP 1 — NumPy array is MUTABLE (edited in place)")
    print("=" * 64)

    arr = np.array([1, 2, 3, 4])
    # id() is the object's identity. If id stays the same, it is literally the same object.
    id_before = id(arr)
    print(f"before:  values={arr}  id={id_before}")

    # In-place assignment: this mutates the existing array, it does NOT create a new one.
    arr[0] = 99
    id_after = id(arr)
    print(f"after:   values={arr}  id={id_after}")

    # Proof: element 0 changed to 99, yet the id is identical -> same object, mutated.
    assert arr[0] == 99, "expected element 0 to become 99"
    assert id_before == id_after, "expected the SAME object (id must be unchanged)"
    print("PROVED: values changed, id unchanged -> the same object was mutated.\n")


def step2_jax_rejects_inplace_edit() -> None:
    """JAX arrays refuse item assignment: `x[0] = 99` raises TypeError."""
    print("=" * 64)
    print("STEP 2 — JAX array is IMMUTABLE (in-place edit is rejected)")
    print("=" * 64)

    x = jnp.array([1, 2, 3, 4])
    print(f"x = {x}")

    try:
        # Exactly what worked for NumPy above — here it must fail because JAX arrays
        # do not support item assignment.
        x[0] = 99
        # Reaching this line means JAX did NOT enforce immutability -> experiment failed.
        raise AssertionError("expected a TypeError, but item assignment succeeded")
    except TypeError as exc:
        # Catch the TypeError and print its message so we can see JAX explain itself.
        print("caught TypeError as expected. Message:")
        print(f"  {exc}")
    print()


def step3_jax_functional_update() -> None:
    """The JAX way: `x.at[0].set(99)` returns a NEW array and leaves x untouched."""
    print("=" * 64)
    print("STEP 3 — JAX functional update with .at[].set()")
    print("=" * 64)

    x = jnp.array([1, 2, 3, 4])

    # .at[idx].set(v) never touches x; it returns a fresh array with the change applied.
    y = x.at[0].set(99)

    print(f"x = {x}   id(x)={id(x)}")
    print(f"y = {y}   id(y)={id(y)}")

    # x must be completely unchanged (still the original [1, 2, 3, 4]).
    assert bool(jnp.all(x == jnp.array([1, 2, 3, 4]))), "expected x to be unchanged"
    # x and y must be different objects (functional update = new array).
    assert x is not y, "expected .at[].set() to return a NEW object"
    print("PROVED: x is unchanged, y is a new array, and x is not y.\n")


def main() -> int:
    print(f"numpy {np.__version__} | jax {jax.__version__}")
    print("Goal: prove NumPy mutates in place while JAX arrays are immutable.\n")
    step1_numpy_is_mutable()
    step2_jax_rejects_inplace_edit()
    step3_jax_functional_update()
    print("ALL THREE DEMONSTRATIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
