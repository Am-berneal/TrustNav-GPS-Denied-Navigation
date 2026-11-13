#!/usr/bin/env bash
# Real-data check: degrade a real EuRoC sequence, run the C++ solver, score it.
# Usage: tools/run_real.sh /path/to/MH_01_easy [seed]
#   expects <seq>/mav0/imu0/data.csv and <seq>/mav0/state_groundtruth_estimate0/data.csv
set -euo pipefail
SEQ=${1:?path to sequence folder (contains mav0/)}; SEED=${2:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HERE/real_runs/$(basename "$SEQ")_s$SEED
mkdir -p "$OUT"
g++ -O2 -std=c++17 -o "$OUT/trustnav" "$HERE/src/main.cpp"
python3 "$HERE/tools/degrade.py" "$SEQ/mav0/imu0/data.csv" "$OUT" --seed "$SEED"
"$OUT/trustnav" "$OUT/mav0/imu0/data.csv" "$OUT/estimated_trajectory.txt"
python3 "$HERE/tools/evaluate.py" "$OUT/estimated_trajectory.txt" \
    "$SEQ/mav0/state_groundtruth_estimate0/data.csv" "$OUT/events.json" | tee "$OUT/score.json"
echo "evo cross-check:"
echo "  evo_ape euroc $SEQ/mav0/state_groundtruth_estimate0/data.csv $OUT/estimated_trajectory.txt -va --align --plot"
