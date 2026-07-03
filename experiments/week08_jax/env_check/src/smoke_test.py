"""
Smoke test: is the JAX/Flax/Optax stack ready for the Week 9 Addition Transformer capstone?

Exit code 0 = ready. Any exception = not ready, fix it before Week 9 Monday.
"""
import sys


def main() -> int:
    try:
        import jax
        import jax.numpy as jnp
        import flax
        import flax.linen as nn
        import optax
    except ImportError as exc:
        print(f"IMPORT FAILED: {exc}")
        print("Stack is NOT ready. Install the missing package before Week 9 Monday.")
        return 1

    # Step 1: does JAX see at least one device?
    devices = jax.devices()
    print(f"jax.devices() -> {devices}")
    if not devices:
        print("NO DEVICES FOUND. Stack is NOT ready.")
        return 1

    # Step 2: are flax and optax importable with real version strings?
    print(f"flax.__version__  -> {flax.__version__}")
    print(f"optax.__version__ -> {optax.__version__}")
    if not flax.__version__ or not optax.__version__:
        print("EMPTY VERSION STRING. Stack is NOT ready.")
        return 1

    # Step 3: smallest possible Flax model — a 2-layer Dense MLP.
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=32)(x)
            x = nn.relu(x)
            x = nn.Dense(features=8)(x)
            return x

    # Step 4: one forward pass on a dummy (1, 14) one-hot-ish input.
    # 14 = a stand-in vocab size for the Addition Transformer's tiny alphabet
    # (10 digits + '+' + '=' + space + pad).
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 14))

    model = MLP()
    params = model.init(key, dummy_input)
    output = model.apply(params, dummy_input)

    print(f"forward pass output shape -> {output.shape}")

    print("\nSTACK READY for Week 9 Monday.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
