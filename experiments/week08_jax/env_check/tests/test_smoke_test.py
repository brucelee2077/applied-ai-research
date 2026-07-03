import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "smoke_test.py"


def test_smoke_test_script_exists():
    assert SCRIPT.exists(), f"expected smoke test script at {SCRIPT}"


def test_smoke_test_fails_loudly_without_stack():
    """
    In an environment without jax/flax/optax installed, the script must exit
    non-zero and print a clear message identifying the missing package —
    never silently pass or crash with an unrelated traceback.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "No module named" in result.stdout:
        assert result.returncode != 0
        assert "Stack is NOT ready" in result.stdout
    else:
        # jax/flax/optax ARE installed in this environment: the real
        # acceptance criteria apply.
        assert result.returncode == 0
        assert "STACK READY" in result.stdout
        assert "jax.devices()" in result.stdout
        assert "flax.__version__" in result.stdout
        assert "optax.__version__" in result.stdout
        assert "forward pass output shape" in result.stdout


if __name__ == "__main__":
    test_smoke_test_script_exists()
    test_smoke_test_fails_loudly_without_stack()
    print("All checks passed.")
