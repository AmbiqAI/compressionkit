"""Re-export a golden run directory as a complete deployment package.

Usage:
    python scripts/package_golden_release.py \\
        --golden-dir results/ppg_rvq_64hz_04x_golden \\
        --modality ppg \\
        --sample-rate 64 \\
        --compression-ratio 4

Produces a self-contained `deploy/` subdirectory with all artifacts needed
for HuggingFace release, including float32 decoder TFLite, model card, and
synthetic stimulus data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a golden run for release.")
    parser.add_argument("--golden-dir", type=Path, required=True, help="Path to golden run directory.")
    parser.add_argument("--modality", choices=["ecg", "ppg"], required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--compression-ratio", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output dir (default: golden-dir/deploy).")
    parser.add_argument("--num-stimulus", type=int, default=10, help="Number of synthetic stimulus samples.")
    parser.add_argument("--export-decoder-int8", action="store_true", help="Also export INT8 decoder TFLite.")
    parser.add_argument("--scorecard", type=Path, default=None, help="Path to quality_scorecard.json to embed in model card.")
    args = parser.parse_args()

    golden_dir = args.golden_dir
    if not golden_dir.is_dir():
        logger.error("Golden directory not found: %s", golden_dir)
        sys.exit(1)

    output_dir = args.output_dir or golden_dir / "deploy"

    # Load the trained model
    import keras

    from compressionkit.export.deploy import export_for_deployment
    from compressionkit.export.stimulus import export_stimulus_npz

    # Find best model weights
    weights_path = golden_dir / "best_model.weights.h5"
    model_path = golden_dir / "model.keras"
    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        sys.exit(1)

    logger.info("Loading model from %s", model_path)
    model = keras.models.load_model(model_path)

    # Extract encoder and decoder sub-models
    encoder = model.encoder if hasattr(model, "encoder") else model.get_layer("encoder")
    decoder = model.decoder if hasattr(model, "decoder") else model.get_layer("decoder")

    # Extract RVQ weights
    rvq_layer = None
    for layer in model.layers:
        if "rvq" in layer.name.lower() or "residual_vector_quantiz" in layer.name.lower():
            rvq_layer = layer
            break
    if rvq_layer is None:
        logger.error("Could not find RVQ layer in model.")
        sys.exit(1)
    rvq_weights = rvq_layer.get_weights()

    # Generate representative dataset from synthetic data
    from compressionkit.export.stimulus import generate_stimulus

    frame_size = encoder.input_shape[-2] if len(encoder.input_shape) == 3 else encoder.input_shape[-1]
    rep_dataset = generate_stimulus(
        modality=args.modality,
        num_samples=100,
        frame_size=frame_size,
        sample_rate=args.sample_rate,
        seed=42,
    )
    # Add channel dim if needed
    if len(encoder.input_shape) == 3 and rep_dataset.ndim == 2:
        rep_dataset = rep_dataset[..., np.newaxis]

    # Build model card info
    model_card_info = {
        "modality": args.modality,
        "sample_rate": args.sample_rate,
        "compression_ratio": args.compression_ratio,
        "license": "Apache-2.0",
    }
    if args.scorecard and args.scorecard.exists():
        with open(args.scorecard) as f:
            scorecard = json.load(f)
        model_card_info["scorecard_summary"] = {
            k: scorecard.get(k) for k in ["time_domain", "spectral"] if k in scorecard
        }

    # Generate sample inputs/reconstructions
    sample_inputs = rep_dataset[:10]
    sample_reconstructions = model.predict(sample_inputs, verbose=0)
    if isinstance(sample_reconstructions, dict):
        sample_reconstructions = sample_reconstructions.get("reconstruction", sample_reconstructions.get("output"))

    # Export
    model_name = f"{args.modality}_rvq_{args.sample_rate}hz_{args.compression_ratio}x"
    artifacts = export_for_deployment(
        encoder=encoder,
        decoder=decoder,
        rvq_weights=rvq_weights,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        sample_inputs=sample_inputs,
        sample_targets=sample_inputs,
        sample_reconstructions=np.asarray(sample_reconstructions, dtype=np.float32),
        model_name=model_name,
        export_decoder_float32=True,
        export_decoder_int8=args.export_decoder_int8,
        model_card_info=model_card_info,
    )

    # Export synthetic stimulus separately
    export_stimulus_npz(
        modality=args.modality,
        output_path=output_dir / "sample_stimulus.npz",
        num_samples=args.num_stimulus,
        frame_size=frame_size,
        sample_rate=args.sample_rate,
    )

    logger.info("Package complete: %s", output_dir)
    logger.info("Artifacts: %s", json.dumps(artifacts.as_dict(), indent=2))


if __name__ == "__main__":
    main()
