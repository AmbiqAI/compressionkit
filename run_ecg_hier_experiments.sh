#!/usr/bin/env bash
# Run hierarchical codebook A/B experiments for ECG (2-level configs only).
# Compares golden [256,256] vs hierarchical [512,128] at iso-CR.
set -euo pipefail

CONFIGS=(
    configs/ecg_rvq_256hz_02x_golden_hier.yaml
    configs/ecg_rvq_256hz_04x_golden_hier.yaml
    configs/ecg_rvq_256hz_08x_golden_hier.yaml
    configs/ecg_rvq_256hz_16x_golden_hier.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================"
    echo "Training: $cfg"
    echo "============================================"
    train-ecg-rvq --config "$cfg"
    echo ""
done

echo "All hierarchical codebook experiments complete."
