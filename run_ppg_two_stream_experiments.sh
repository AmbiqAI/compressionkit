#!/usr/bin/env bash
set -e

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

LOG=results/ppg_two_stream_sweep.log
: > "$LOG"

echo "=== Two-stream PPG codec sweep ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

for cfg in configs/ppg_two_stream_04x.yaml configs/ppg_two_stream_08x.yaml configs/ppg_two_stream_16x.yaml; do
    echo "" | tee -a "$LOG"
    echo "--- Running: $cfg ---" | tee -a "$LOG"
    echo "Start: $(date)" | tee -a "$LOG"
    python train_ppg_two_stream_from_yaml.py "$cfg" 2>&1 | tee -a "$LOG"
    echo "End: $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== All experiments complete: $(date) ===" | tee -a "$LOG"
