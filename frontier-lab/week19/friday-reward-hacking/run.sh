#!/usr/bin/env bash
# =============================================================================
# run.sh -- reproduce the whole reward-hacking demo end to end, one command.
#
#   ./run.sh
#
# Steps:
#   1. Show the raw simulator numbers for both policies.
#   2. Run the pytest correctness gate (honest PASSES, hack FAILS).
#   3. Score both policies under the NAIVE evaluator (hack wins).
#   4. Score both policies under the HARDENED evaluator (hack -> 0, flip).
#
# Exit code is 0 on success. The pytest step is expected to report 1 failure
# (the hack policy correctly fails conservation), so we do not let that abort
# the script -- that failure IS the demonstration.
# =============================================================================
set -u
cd "$(dirname "$0")"

# Use python3 if available, else python.
PY="$(command -v python3 || command -v python)"

echo "############################################################"
echo "# STEP 1 -- raw simulator: honest vs evolved_v7"
echo "############################################################"
"$PY" sim_loadbalancer.py
echo

echo "############################################################"
echo "# STEP 2 -- pytest correctness gate (conservation)"
echo "#   expected: honest PASSED, evolved_v7 FAILED"
echo "############################################################"
# do not abort on the expected failure
"$PY" -m pytest test_conservation.py -q || true
echo

echo "############################################################"
echo "# STEP 3 -- NAIVE evaluator (speed only) -- the hack WINS"
echo "############################################################"
"$PY" evaluator_naive.py
echo

echo "############################################################"
echo "# STEP 4 -- HARDENED evaluator (correctness gate) -- hack -> 0"
echo "############################################################"
"$PY" evaluator_hardened.py
echo

echo "############################################################"
echo "# DONE. See results.md and analysis.md for the writeup."
echo "############################################################"
exit 0
