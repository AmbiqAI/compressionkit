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


# Known dataset slugs -> (display name, license note). Extend as new sources
# are added to unified_cache configs; unknown slugs degrade gracefully.
_KNOWN_SOURCE_LICENSES: dict[str, tuple[str, str]] = {
    "bidmc": ("BIDMC", "open, PhysioNet"),
    "butppg": ("BUT PPG", "open, PhysioNet"),
    "ppg_dalia": ("PPG-DaLiA", "open, UCI"),
    "wesad": ("WESAD", "open, UCI"),
    "mesa": ("MESA", "NSRR restricted"),
    "ptbxl": ("PTB-XL", "CC BY 4.0"),
    "ptb-xl": ("PTB-XL", "CC BY 4.0"),
}


def _dataset_provenance_note(
    dataset_sources: list[str] | None,
    modality: str,
    *,
    context: str = "Training",
) -> str:
    """Describe dataset provenance from recorded source slugs.

    Falls back to a neutral, non-committal statement when no source list was
    recorded on the package rather than asserting a specific dataset, since
    doing so has previously produced stale/incorrect claims (e.g. reporting
    MESA on packages actually trained on the open unified cache).
    """
    if dataset_sources:
        names: list[str] = []
        restricted: list[str] = []
        for slug in dataset_sources:
            display, note = _KNOWN_SOURCE_LICENSES.get(slug, (slug, "license unknown"))
            names.append(display)
            if "restricted" in note.lower():
                restricted.append(display)
        joined = " + ".join(names)
        if restricted:
            return (
                f"{context} data: {joined}. Includes restricted source(s) ({', '.join(restricted)}); "
                "sample data uses synthetic physiokit waveforms only — no patient data is redistributed."
            )
        return (
            f"{context} data: {joined} (all open, no restricted-access dependency). "
            "Sample data uses synthetic physiokit waveforms only — no patient data is redistributed."
        )
    if modality == "ecg":
        return f"{context} data: PTB-XL (CC BY 4.0). Sample data may include excerpts under the original license terms."
    return (
        f"{context} data provenance is not recorded in this package; "
        "sample data uses synthetic physiokit waveforms only — no patient data is redistributed."
    )


def generate_model_card(
    deploy_dir: str | Path,
    scorecard_path: str | Path | None = None,
    license_id: str = "other",
    repo_id: str | None = None,
) -> str:
    """Generate a HuggingFace-style README.md model card.

    Args:
        deploy_dir: Path to the deployment directory containing
            ``deploy_manifest.json`` and optionally ``model_card.json``.
        scorecard_path: Optional path to ``quality_scorecard.json``.
            If *None*, looks in the parent of ``deploy_dir``.
        license_id: SPDX license identifier for YAML frontmatter.
        repo_id: The actual HuggingFace repo id this card will be published
            to (e.g. ``Ambiq/compressionkit-ppg-4x-v1.0``). When given, the
            title and usage snippets reference this id instead of one
            inferred from the manifest's modality/CR (which may not match
            the real publishing target's release-track suffix).

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
    if license_id == "other":
        lines.append("license_name: ambiq-model-weights-license")
        lines.append("license_link: https://github.com/AmbiqAI/compressionkit/blob/main/LICENSE-MODEL-WEIGHTS.md")
    lines.append("library_name: compressionkit")
    lines.append("pipeline_tag: other")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")

    # Title
    cr_slug = f"{float(cr):g}" if isinstance(cr, (int, float)) else str(cr)
    inferred_repo_id = f"Ambiq/compressionkit-{modality}-{cr_slug}x" if cr else f"Ambiq/compressionkit-{modality}"
    hf_repo_id = repo_id or inferred_repo_id
    lines.append(f"# {hf_repo_id.split('/', 1)[-1]}")
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
    lines.append(f'codec = RVQCodec.from_pretrained("{hf_repo_id}")')
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
    lines.append(_dataset_provenance_note(model_card_info.get("dataset_sources"), modality, context="Training"))
    lines.append("")
    if license_id == "other":
        lines.append(
            "Model weights are released under the **Ambiq Model Weights License** — "
            "deployment is restricted to Ambiq silicon devices. "
            "See `LICENSE-MODEL-WEIGHTS.md` for full terms."
        )
    else:
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


def _add_fidelity_robustness_section(lines: list[str], headline: dict) -> None:
    """Append the paired clean-truth + noise-regime fidelity view.

    Release surfaces must never present faithfulness PRD (vs the recorded,
    still-noisy input) alone, because that yardstick penalizes denoising lanes.
    This block pairs it with truth PRD (vs clean ground truth) and the
    noise-regime anchors so the operating behaviour is read fairly (issue B4).
    """
    faithful = headline.get("faithful_prd_vs_input_pct")
    truth_clean = headline.get("truth_prd_vs_clean_pct")
    truth_native = headline.get("truth_prd_at_native_noise_pct")
    slope = headline.get("prd_degradation_slope_per_db")
    prd_0db = headline.get("prd_at_0db_pct")
    prd_m6db = headline.get("prd_at_-6db_pct")
    imprint = headline.get("imprint_output_autocorr")

    if all(v is None for v in (faithful, truth_clean, truth_native, slope, prd_0db, prd_m6db, imprint)):
        return

    lines.append("### Fidelity & Robustness")
    lines.append("")
    lines.append(
        "Both fidelity yardsticks are reported so the codec is judged fairly: "
        "**faithfulness** is PRD vs the recorded (still-noisy) input, while "
        "**truth fidelity** is PRD vs clean ground truth. Lower is better."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    if truth_clean is not None:
        lines.append(f"| Truth PRD vs clean (%) | {_fmt(truth_clean, 2)} |")
    if truth_native is not None:
        lines.append(f"| Truth PRD at native noise (%) | {_fmt(truth_native, 2)} |")
    if faithful is not None:
        lines.append(f"| Faithful PRD vs input (%) | {_fmt(faithful, 2)} |")
    if slope is not None:
        lines.append(f"| PRD degradation slope (PRD%/dB) | {_fmt(slope, 2)} |")
    if prd_0db is not None:
        lines.append(f"| PRD at 0 dB SNR (%) | {_fmt(prd_0db, 2)} |")
    if prd_m6db is not None:
        lines.append(f"| PRD at -6 dB SNR (%) | {_fmt(prd_m6db, 2)} |")
    if imprint is not None:
        lines.append(f"| Pure-noise imprint autocorr | {_fmt(imprint, 4)} |")
    lines.append("")


def _add_scorecard_section(lines: list[str], scorecard: dict) -> None:
    """Append quality metrics from scorecard to lines."""
    # Fidelity & robustness (paired clean-truth + noise-regime view first)
    headline = scorecard.get("headline", {}) or {}
    if headline:
        _add_fidelity_robustness_section(lines, headline)

    # Time domain
    td = scorecard.get("time_domain", {})
    if td:
        lines.append("### Time Domain")
        lines.append("")
        lines.append(
            "_PRD here is faithfulness (vs the recorded input); see "
            "**Fidelity & Robustness** above for the clean-truth and noise-regime view._"
        )
        lines.append("")
        lines.append("| Metric | Mean | Median | P90 |")
        lines.append("|--------|------|--------|-----|")
        for key, label in [
            ("prd_percent", "PRD vs input — faithfulness (%)"),
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


def generate_spiht_model_card(
    deploy_dir: str | Path,
    scorecard_path: str | Path | None = None,
    license_id: str = "apache-2.0",
    repo_id: str | None = None,
) -> str:
    """Generate a HuggingFace model card for a DSP-only SPIHT codec.

    SPIHT deploy packages have no trained weights — the publishable
    artifact is the bitstream contract plus the vendored C99 reference.
    This card surfaces the operating point, scorecard summary, and
    Python/C quickstart snippets.

    Args:
        deploy_dir: Path to a SPIHT deploy directory.
        scorecard_path: Optional path to ``quality_scorecard.json``.
        license_id: SPDX license identifier (defaults to ``apache-2.0``
            because there are no proprietary weights).
        repo_id: The actual HuggingFace repo id this card will be published
            to. When given, the title and usage snippets reference this id
            instead of one inferred from the manifest's modality/method/CR.

    Returns:
        The model card as a Markdown string with YAML frontmatter.
    """
    deploy_dir = Path(deploy_dir)
    with (deploy_dir / "deploy_manifest.json").open() as f:
        manifest = json.load(f)

    if manifest.get("family") not in ("spiht", "hybrid"):
        raise ValueError(
            f"generate_spiht_model_card expects family in ('spiht', 'hybrid'), got {manifest.get('family')!r}"
        )
    # A hybrid package layers a learned wavelet-gain denoiser (trained weights)
    # in front of the same SPIHT bitstream contract. ``hybrid_manifest.json`` is
    # the authoritative marker regardless of the manifest's own ``family`` value.
    is_hybrid = (deploy_dir / "hybrid_manifest.json").exists()

    codec = manifest.get("codec", {})
    modality = codec.get("modality", "unknown")
    sample_rate = codec.get("sample_rate")
    frame_size = codec.get("frame_size")
    target_cr = codec.get("target_cr")
    wavelet = codec.get("wavelet")
    levels = codec.get("levels")
    use_ac = codec.get("use_ac", True)
    max_bits = codec.get("max_bits")

    scorecard: dict | None = None
    if scorecard_path is not None:
        sc_path = Path(scorecard_path)
        if sc_path.exists():
            with sc_path.open() as f:
                scorecard = json.load(f)
    else:
        # Prefer the corrected ``quality_scorecard.json`` in the parent run dir
        # (it carries the v1 ``headline`` block with the clean-truth / noise
        # view) over the frozen deploy-time ``scorecard.json``; fall back to the
        # embedded ``scorecard_summary`` last.
        for candidate in (
            deploy_dir.parent / "quality_scorecard.json",
            deploy_dir / "scorecard.json",
        ):
            if candidate.exists():
                with candidate.open() as f:
                    scorecard = json.load(f)
                break
        if scorecard is None and isinstance(manifest.get("scorecard_summary"), dict):
            embedded = manifest["scorecard_summary"]
            if embedded:
                scorecard = embedded

    tags = [
        "compressionkit",
        "signal-compression",
        modality,
        "spiht",
        "wavelet",
        "edge-ai",
    ]
    if is_hybrid:
        tags.append("hybrid")
        tags.append("denoising")
    else:
        tags.append("dsp")

    family_label = "hybrid" if is_hybrid else "spiht"
    cr_label = f"{target_cr:g}x" if target_cr else ""
    inferred_repo_id = (
        f"Ambiq/compressionkit-{modality}-{family_label}-{cr_label}"
        if cr_label
        else f"Ambiq/compressionkit-{modality}-{family_label}"
    )
    hf_repo_id = repo_id or inferred_repo_id

    lines: list[str] = []
    lines.append("---")
    lines.append(f"license: {license_id}")
    lines.append("library_name: compressionkit")
    lines.append("pipeline_tag: other")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")

    lines.append(f"# {hf_repo_id.split('/', 1)[-1]}")
    lines.append("")
    if is_hybrid:
        lines.append(
            f"A **{modality.upper()}** signal compression codec: a learned "
            "wavelet-gain denoiser (trained weights) followed by wavelet + "
            "SPIHT + arithmetic coding. The deployable artifact is the "
            "denoiser weights, the bitstream contract, and a portable C99 "
            "reference implementation for the SPIHT stage."
        )
    else:
        lines.append(
            f"A **{modality.upper()}** signal compression codec built on "
            "wavelet + SPIHT + arithmetic coding. **DSP-only — no trained "
            "weights.** The deployable artifact is the bitstream contract "
            "plus a portable C99 reference implementation."
        )
    lines.append("")

    lines.append("## Operating point")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Modality | {modality.upper()} |")
    if sample_rate:
        lines.append(f"| Sample rate | {sample_rate} Hz |")
    if frame_size:
        lines.append(f"| Frame size | {frame_size} samples |")
    if target_cr:
        lines.append(f"| Target CR | {target_cr:g}x |")
    if wavelet:
        lines.append(f"| Wavelet | `{wavelet}` |")
    if levels:
        lines.append(f"| DWT levels | {levels} |")
    if max_bits:
        lines.append(f"| Bit budget | {max_bits} bits/frame |")
    lines.append(f"| Entropy coder | {'arithmetic coding' if use_ac else 'raw SPIHT'} |")
    lines.append("")

    if scorecard:
        lines.append("## Quality metrics")
        lines.append("")
        _add_scorecard_section(lines, scorecard)

    lines.append("## Python quickstart")
    lines.append("")
    lines.append("```python")
    lines.append("from compressionkit.runtime import load_codec")
    lines.append("")
    lines.append(f'codec = load_codec("{hf_repo_id}")')
    lines.append("enc = codec.compress(frame)   # frame: (frame_size,) float32")
    lines.append("recon = codec.decompress(enc)")
    lines.append("```")
    lines.append("")

    lines.append("## C quickstart")
    lines.append("")
    if is_hybrid:
        denoiser_quantization = "INT8"
        hybrid_manifest_path = deploy_dir / "hybrid_manifest.json"
        if hybrid_manifest_path.exists():
            with hybrid_manifest_path.open() as f:
                hybrid_manifest = json.load(f)
            for stage in hybrid_manifest.get("stages", []):
                if isinstance(stage, dict) and stage.get("stage") == "denoise":
                    mode = stage.get("mode") or {}
                    # Direct-mode (non-gain) denoisers replace the coefficient
                    # outright rather than multiplying a bounded [0, 1] gain
                    # into it, so INT8 quantization error propagates directly
                    # into reconstruction error — these ship as INT16X8
                    # instead (see issue #55).
                    denoiser_quantization = "INT8" if mode.get("gain_mode", True) else "INT16X8"
                    break
        lines.append(
            f"The SPIHT stage ships a portable C99 reference. The denoiser ships "
            f"as an {denoiser_quantization} `denoiser_gain_model.tflite` (run via a LiteRT/TFLite "
            "Micro interpreter — no Python/Keras required) alongside the "
            "float32 `denoiser_gain_model.keras` reference. Run the denoiser "
            "stage first and feed its output into the C SPIHT encoder below."
        )
        lines.append("")
    lines.append("```c")
    lines.append('#include "spiht_app_config.h"')
    lines.append("")
    lines.append("float frame[APP_SPIHT_FRAME_SIZE];")
    lines.append("uint8_t bitstream[APP_SPIHT_MAX_BYTES];")
    lines.append("/* ... fill frame from sensor (post-denoise, if hybrid) ... */")
    lines.append("size_t nbits = spiht_encode_frame(&enc, bitstream, APP_SPIHT_MAX_BITS);")
    lines.append("```")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append(f'| `config.json` | Deploy manifest (`family: "{family_label}"`) |')
    lines.append("| `spiht_config.json` | Codec parameters (language-neutral) |")
    lines.append("| `sample_stimulus.npz` | Synthetic test frames |")
    lines.append("| `reference_vectors.npz` | Reference encode/decode vectors |")
    lines.append("| `c_sources/spiht.[ch]` | Portable C99 reference |")
    lines.append("| `spiht_app_config.h` | Codec-specific defines (deploy root, not under `c_sources/`) |")
    lines.append("| `model_card.json` | Provenance metadata |")
    lines.append("| `scorecard.json` | Frozen evaluation summary |")
    if is_hybrid:
        lines.append("| `denoiser_gain_model.keras` | Learned wavelet-gain denoiser (trained weights) |")
        lines.append("| `hybrid_manifest.json` | Pipeline stage order (denoise → SPIHT) |")
    lines.append("")

    lines.append("## Dataset & license")
    lines.append("")
    spiht_model_card_info = manifest.get("model_card")
    dataset_sources = spiht_model_card_info.get("dataset_sources") if isinstance(spiht_model_card_info, dict) else None
    lines.append(_dataset_provenance_note(dataset_sources, modality, context="Evaluation"))
    lines.append("")
    lines.append(f"Codec source released under the **{license_id.upper()}** license.")
    lines.append("")

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
