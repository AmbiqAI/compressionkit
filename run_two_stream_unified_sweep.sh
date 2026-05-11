#!/bin/bash
# Train two-stream unified sweep: 2x, 8x, 16x
# (4x already trained as ppg_two_stream_04x_unified_all)
set -e

cd /workspaces/compressionkit

echo "=== Training 2x two-stream unified ==="
.venv/bin/python train_ppg_two_stream_from_yaml.py configs/ppg_two_stream_02x_unified_all.yaml

echo "=== Training 8x two-stream unified ==="
.venv/bin/python train_ppg_two_stream_from_yaml.py configs/ppg_two_stream_08x_unified_all.yaml

echo "=== Training 16x two-stream unified ==="
.venv/bin/python train_ppg_two_stream_from_yaml.py configs/ppg_two_stream_16x_unified_all.yaml

echo "=== All done ==="
