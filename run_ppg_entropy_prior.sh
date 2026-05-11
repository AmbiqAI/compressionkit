#!/usr/bin/env bash
# Run CNN entropy prior on all 4 ExpA2 PPG models.
# Token density varies hugely by CR, so we adjust stride accordingly.
set -e

source /workspaces/compressionkit/.venv/bin/activate
cd /workspaces/compressionkit

COMMON_ARGS=(
  --prior-type cnn
  --context-frames 4
  --epochs 15
  --batch-size 256
  --cnn-embed-dim 48
  --cnn-num-layers 4
  --cnn-kernel 5
  --dropout 0.1
  --learning-rate 3e-4
  --tag cnn_v1
)

echo "========== PPG ExpA2 02x (320 tok/frame) =========="
python scripts/measure_rvq_entropy.py \
  --run-dir results/ppg_rvq_64hz_02x_unified_expA2_codebook \
  --max-train-frames 5000 \
  --max-val-frames 2000 \
  --stride-tokens 32 \
  "${COMMON_ARGS[@]}"

echo ""
echo "========== PPG ExpA2 04x (160 tok/frame) =========="
python scripts/measure_rvq_entropy.py \
  --run-dir results/ppg_rvq_64hz_04x_unified_expA2_codebook \
  --max-train-frames 8000 \
  --max-val-frames 3000 \
  --stride-tokens 16 \
  "${COMMON_ARGS[@]}"

echo ""
echo "========== PPG ExpA2 08x (80 tok/frame) =========="
python scripts/measure_rvq_entropy.py \
  --run-dir results/ppg_rvq_64hz_08x_unified_expA2_codebook \
  --max-train-frames 12000 \
  --max-val-frames 4000 \
  --stride-tokens 8 \
  "${COMMON_ARGS[@]}"

echo ""
echo "========== PPG ExpA2 16x (40 tok/frame) =========="
python scripts/measure_rvq_entropy.py \
  --run-dir results/ppg_rvq_64hz_16x_unified_expA2_codebook \
  --max-train-frames 15000 \
  --max-val-frames 5000 \
  --stride-tokens 8 \
  "${COMMON_ARGS[@]}"

echo ""
echo "========== DONE =========="
echo "Reports saved to:"
for cr in 02x 04x 08x 16x; do
  echo "  results/ppg_rvq_64hz_${cr}_unified_expA2_codebook/entropy_prior/cnn_v1/entropy_report.json"
done
