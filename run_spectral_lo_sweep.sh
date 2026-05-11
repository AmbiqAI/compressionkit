#!/usr/bin/env bash
# Spectral loss (weight=0.1) sweep across all golden CRs
# ECG: 2x, 4x, 16x, 32x, 64x  (8x already done)
# PPG: 2x, 4x, 8x, 16x
set -e

source /workspaces/compressionkit/.venv/bin/activate

echo "===== ECG spectral_lo sweep ====="
for cr in 02x 04x 16x 32x 64x; do
  echo ""
  echo "--- ECG ${cr} spectral_lo ---"
  train-ecg-rvq --config configs/ecg_rvq_256hz_${cr}_golden_spectral_lo.yaml
done

echo ""
echo "===== PPG spectral_lo sweep ====="
for cr in 2x 4x 8x 16x; do
  echo ""
  echo "--- PPG ${cr} spectral_lo ---"
  train-ppg-rvq --config configs/ppg_h5_rvq_${cr}_mixed_golden_sched_spectral_lo.yaml
done

echo ""
echo "All spectral_lo experiments complete."
