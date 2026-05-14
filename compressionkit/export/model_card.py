"""Generate HuggingFace-style README.md model cards.

Produces a Markdown model card with YAML frontmatter from a deployment
manifest and optional quality scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(val: float, decimals: int = 4) -> str:
    """Format a float for display in the model card."""
    return f"{val:.{decimals}f}"


def generate_model_card(
    deploy_dir: str | Path,
    scorecard_path: str | Path | None = None,
    license_id: str = "apache-2.0",
) -> str:
    """Generate a HuggingFace-style README.md model card.

    Args:
        deploy_dir: Path to the deployment directory containing
            ``deploy_manifest.json`` and optionally ``model_card.json``.
        scorecard_path: Optional path to ``quality_scorecard.json``.
            If *None*, looks in the parent of ``deploy_dir``.
        license_id: SPDX license identifier for YAML frontmatter.

    Returns:
        The model card as a Markdown string with YAML frontmatter.
    """
    deploy_dir = Path(deploy_dir)

    with open(deploy_dir / "deploy_manifest.json") as f:
        manifest = json.load(f)

    # Try to load model_card.json for extra metadata
    model_card_path = deploy_dir / "model_card.json"
    model_card_info: dict = {}
    if model_card_path.exists():
        with open(model_card_path) as f:
            model_card_info = json.load(f)

    # Try to find scorecard
    scorecard: dict | None = None
    if scorecard_path is not None:
        sc_path = Path(scorecard_path)
        if sc_path.exists():
            with open(sc_path) as f:
                scorecard = json.load(f)
    else:
        # Look in parent directory of deploy_dir
        sc_path = deploy_dir.parent / "quality_scorecard.json"
        if sc_path.exists():
            with open(sc_path) as f:
                scorecard = json.load(f)

    model_name = manifest.get("model_name", "unknown")
    modality = model_card_info.get("modality", _infer_modality(model_name))
    sample_rate = model_card_info.get("sample_rate", scorecard.get("sample_rate") if scorecard else None)
    cr = model_card_info.get("compression_ratio", _infer_cr(manifest))

    # Build codebook info
    cb = manifest.get("codebook", {})
    num_levels = cb.get("num_levels", "?")
    num_embeddings = cb.get("num_embeddings", "?")
    embedding_dim = cb.get("embedding_dim", "?")

    # Encoder info
    enc = manifest.get("encoder", {})
    input_shape = enc.get("input_shape", [])
    output_shape = enc.get("output_shape", [])

    # Tags for HF
    tags = [
        "compressionkit",
        "signal-compression",
        modality,
        "rvq",
        "tflite",
        "edge-ai",
    ]

    # --- Build the card ---
    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"license: {license_id}")
    lines.append("library_name: compressionkit")
    lines.append("pipeline_tag: other")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")

    # Title
    hf_name = f"compressionkit-{modality}-{cr}x" if cr else f"compressionkit-{modality}"
    lines.append(f"# {hf_name}")
    lines.append("")
    lines.append(
        f"A **{modality.upper()}** signal compression codec using Residual Vector Quantization (RVQ), "
        "optimized for edge and wearable devices."
    )
    lines.append("")

    # Model details
    lines.append("## Model Details")
    lines.append("")
    lines.append(f"- **Modality:** {modality.upper()}")
    if sample_rate:
        lines.append(f"- **Sample Rate:** {sample_rate} Hz")
    if cr:
        lines.append(f"- **Compression Ratio:** {cr}x")
    lines.append(f"- **Quantization:** {manifest.get('quantization', 'INT8')}")
    lines.append(f"- **RVQ Levels:** {num_levels}")
    lines.append(f"- **Codebook Size:** {num_embeddings} entries × {embedding_dim}D")
    if input_shape:
        lines.append(f"- **Encoder Input:** `{input_shape}`")
    if output_shape:
        lines.append(f"- **Encoder Output:** `{output_shape}`")
    lines.append("")

    # Quality metrics (if scorecard available)
    if scorecard:
        lines.append("## Quality Metrics")
        lines.append("")
        _add_scorecard_section(lines, scorecard)

    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append("### Python (compressionkit runtime)")
    lines.append("")
    lines.append("```python")
    lines.append("from compressionkit.runtime import RVQCodec")
    lines.append("")
    lines.append(f'codec = RVQCodec.from_pretrained("Ambiq/{hf_name}")')
    lines.append("")
    lines.append("# Encode: float32 signal → RVQ indices")
    lines.append("indices = codec.encode(signal)")
    lines.append("")
    lines.append("# Decode: RVQ indices → reconstructed signal")
    lines.append("recon = codec.decode(indices)")
    lines.append("```")
    lines.append("")
    lines.append("### Local deployment directory")
    lines.append("")
    lines.append("```python")
    lines.append('codec = RVQCodec("path/to/deploy/")')
    lines.append("```")
    lines.append("")

    # Files
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `encoder_int8.tflite` | INT8 quantized encoder (on-device) |")
    lines.append("| `encoder.h` | C header for encoder |")
    lines.append("| `decoder_float32.tflite` | Float32 decoder (server-side evaluation) |")
    lines.append("| `decoder_int8.tflite` | INT8 decoder (optional, on-device) |")
    lines.append("| `codebook.npz` | RVQ codebook tables |")
    lines.append("| `codebook.h` | C header for codebook |")
    lines.append("| `config.json` | Deployment manifest |")
    lines.append("| `sample_stimulus.npz` | Synthetic test data |")
    lines.append("| `quality_scorecard.json` | Full evaluation metrics |")
    lines.append("")

    # Dataset licensing
    lines.append("## Dataset & License")
    lines.append("")
    if modality == "ppg":
        lines.append(
            "Training data: MESA (NSRR restricted). Sample data uses synthetic "
            "physiokit waveforms only — no patient data is redistributed."
        )
    elif modality == "ecg":
        lines.append(
            "Training data: PTB-XL (CC BY 4.0). Sample data may include excerpts under the original license terms."
        )
    lines.append("")
    lines.append(f"Model weights are released under the **{license_id.upper()}** license.")
    lines.append("")

    # Citation
    lines.append("## Citation")
    lines.append("")
    lines.append("```bibtex")
    lines.append("@software{compressionkit,")
    lines.append("  author = {Ambiq AI},")
    lines.append("  title = {compressionKIT: Signal Compression for Edge AI},")
    lines.append("  url = {https://github.com/AmbiqAI/compressionkit}")
    lines.append("}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _add_scorecard_section(lines: list[str], scorecard: dict) -> None:
    """Append quality metrics from scorecard to lines."""
    # Time domain
    td = scorecard.get("time_domain", {})
    if td:
        lines.append("### Time Domain")
        lines.append("")
        lines.append("| Metric | Mean | Median | P90 |")
        lines.append("|--------|------|--------|-----|")
        for key, label in [
            ("prd_percent", "PRD (%)"),
            ("rmse", "RMSE"),
            ("cosine_similarity", "Cosine Similarity"),
        ]:
            if key in td:
                m = td[key]
                lines.append(f"| {label} | {_fmt(m['mean'])} | {_fmt(m['median'])} | {_fmt(m['p90'])} |")
        lines.append("")

    # Spectral
    sp = scorecard.get("spectral", {})
    if sp:
        band_err = sp.get("band_total_rel_error")
        if band_err:
            lines.append("### Spectral")
            lines.append("")
            lines.append(f"- **Band Total Relative Error (median):** {_fmt(band_err['median'])}")
            lines.append("")

    # Bitrate
    br = scorecard.get("bitrate", {})
    if br:
        lines.append("### Bitrate")
        lines.append("")
        cr_codec = br.get("cr_codec_uniform")
        cr_learned = br.get("cr_codec_learned")
        if cr_codec is not None:
            lines.append(f"- **Codec CR (uniform):** {_fmt(cr_codec, 1)}x")
        if cr_learned is not None:
            lines.append(f"- **Codec CR (learned prior):** {_fmt(cr_learned, 2)}x")
        lines.append("")


def _infer_modality(model_name: str) -> str:
    """Infer modality from model name."""
    name_lower = model_name.lower()
    if "ppg" in name_lower:
        return "ppg"
    if "ecg" in name_lower:
        return "ecg"
    return "unknown"


def _infer_cr(manifest: dict) -> int | None:
    """Infer compression ratio from encoder input/output shapes."""
    enc = manifest.get("encoder", {})
    inp = enc.get("input_shape", [])
    out = enc.get("output_shape", [])
    if len(inp) >= 3 and len(out) >= 3:
        t_in = inp[-2] if len(inp) == 4 else inp[-1]
        t_out = out[-2] if len(out) == 4 else out[-1]
        if t_in and t_out and t_out > 0:
            return t_in // t_out
    return None
