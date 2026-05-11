#!/usr/bin/env bash
# Train ExpA2 with 200 epochs across all 4 CRs (sequential — each uses full GPU).
set -e

source /workspaces/compressionkit/.venv/bin/activate
cd /workspaces/compressionkit

for cr in 02x 04x 08x 16x; do
  echo ""
  echo "=========================================="
  echo "  Training ExpA2 200ep — ${cr}"
  echo "=========================================="
  echo ""
  train-ppg-rvq --config configs/ppg_rvq_64hz_${cr}_unified_expA2_200ep.yaml
done

echo ""
echo "========== ALL DONE =========="
