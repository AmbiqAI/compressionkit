"""RVQ entropy-prior trainer (#27).

Trains a small causal-transformer prior on the discrete RVQ token stream
emitted by a frozen parent codec. Modality-agnostic: the parent's
modality (``"ppg"`` or ``"ecg"``) is resolved from the golden experiment
registry, and the trainer dispatches the right signal loader.

Inputs and outputs are deliberately uniform across modalities so the
generative subpackage and ``TwoStageCodec`` runtime stay portable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.rvq_prior import RvqPriorConfig
from compressionkit.generative import build_prior, extract_rvq_tokens

logger = logging.getLogger(__name__)


def _resolve_parent(cfg: RvqPriorConfig) -> tuple[Any, Path]:
    from compressionkit.experiments.registry import get_golden

    parent_exp = get_golden(cfg.parent_experiment)
    if parent_exp.family != "codec":
        raise ValueError(
            f"prior parent {cfg.parent_experiment!r} must be a codec experiment (got {parent_exp.family!r})"
        )
    run_dir = cfg.parent_run_dir if cfg.parent_run_dir is not None else Path("results") / parent_exp.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Parent run_dir not found: {run_dir}. Train the codec first or pass --parent-run-dir.")
    return parent_exp, run_dir


def _load_parent_compressor(parent_exp: Any, run_dir: Path) -> tuple[keras.Model, Any]:
    if parent_exp.modality == "ecg":
        from compressionkit.configs.ecg_rvq import EcgRvqConfig as ParentConfig
        from compressionkit.trainers.ecg_rvq import build_model
    elif parent_exp.modality == "ppg":
        from compressionkit.configs.ppg_rvq import PpgRvqConfig as ParentConfig
        from compressionkit.trainers.ppg_rvq import build_model
    else:
        raise ValueError(f"Unsupported modality {parent_exp.modality!r}")

    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Parent config.json missing in {run_dir}")
    cfg = ParentConfig.model_validate_json(cfg_path.read_text())
    num_channels = max(1, int(getattr(cfg.data, "num_leads", 1) or 1))
    with tf.device("/CPU:0"):
        model = build_model(cfg)
        dummy = np.zeros((1, 1, cfg.data.frame_size, num_channels), dtype=np.float32)
        model(dummy, training=False)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name
        if p.exists():
            model.load_weights(p)
            break
    else:
        raise FileNotFoundError(f"No weights in {run_dir}")
    return model, cfg


def _load_token_signals(modality: str, parent_cfg: Any, num_files: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    data = parent_cfg.data
    if modality == "ecg":
        from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal

        train_files, _, _ = load_ecg_file_splits(Path(data.datasets_dir), data.dataset_glob, seed=data.shuffle_seed)
        train_files = train_files[:num_files]
        lead_index = getattr(data, "lead_index", 1) or 1
        signals = [load_ecg_signal(p, lead_index=lead_index) for p in train_files]
    else:
        from compressionkit.datasets.ppg import load_ppg_file_splits, load_ppg_signal

        train_files, _, _ = load_ppg_file_splits(Path(data.datasets_dir), data.dataset_glob, seed=data.shuffle_seed)
        train_files = train_files[:num_files]
        target_label = getattr(data, "target_label", "Pleth")
        target_rate = int(getattr(data, "effective_sample_rate", data.sampling_rate))
        signals = [load_ppg_signal(p, target_rate=target_rate, target_label=target_label) for p in train_files]
    return signals, {"num_files": len(signals)}


def _make_windows(tokens_flat: np.ndarray, context_length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    n = tokens_flat.size
    if n < context_length + 1:
        raise ValueError(f"Not enough tokens ({n}) for context {context_length}")
    starts = np.arange(0, n - context_length - 1, max(1, stride))
    xs = np.stack([tokens_flat[s : s + context_length] for s in starts]).astype(np.int32)
    ys = np.stack([tokens_flat[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int32)
    return xs, ys


def _export_prior_tflite(prior: keras.Model, output_dir: Path, vocab_size: int, context_length: int) -> Path:
    """Convert ``prior`` to INT8 TFLite using a representative dataset of random tokens."""
    from compressionkit.export.tflite import _convert_and_export

    rep = np.random.default_rng(0).integers(low=0, high=vocab_size, size=(64, context_length), dtype=np.int32)
    tflite_path, _ = _convert_and_export(
        prior,
        rep_dataset=rep,
        output_dir=output_dir,
        tflite_name="prior_int8.tflite",
        header_name="prior_int8.h",
        c_array_name="prior",
        quantization="INT8",
        io_type="int8",
    )
    return tflite_path


def train_prior_from_config(cfg: RvqPriorConfig) -> dict[str, Any]:
    """Train an entropy prior described by ``cfg`` against its parent codec.

    Side effects:
        - Writes ``<parent_run_dir>/prior/prior.weights.h5``.
        - Writes ``<parent_run_dir>/prior/prior_config.json`` (cfg dump).
        - When ``cfg.export_tflite``: writes ``<parent_run_dir>/deploy/prior_int8.tflite``
          + ``prior_manifest.json``.

    Returns:
        Summary dict with ``parent_run_dir``, ``prior_dir``, ``deploy_dir``,
        ``vocab_size``, ``context_length``, and final training history.
    """
    parent_exp, parent_run_dir = _resolve_parent(cfg)
    logger.info("Training prior for parent %s (run_dir=%s)", parent_exp.experiment_id, parent_run_dir)

    compressor, parent_cfg = _load_parent_compressor(parent_exp, parent_run_dir)
    data = parent_cfg.data
    mcfg = parent_cfg.model
    num_channels = max(1, int(getattr(data, "num_leads", 1) or 1))
    vocab_size = int(mcfg.latent_width)
    frame_size = int(data.frame_size)
    tokens_per_frame = frame_size // (2**mcfg.num_stages)
    if mcfg.num_levels != 1:
        logger.warning("Prior prototype uses level-0 tokens only; parent has num_levels=%d", mcfg.num_levels)

    signals, _ = _load_token_signals(parent_exp.modality, parent_cfg, cfg.training.num_train_files)
    tokens = extract_rvq_tokens(
        compressor,
        signals,
        frame_size=frame_size,
        num_leads=num_channels,
        epsilon=data.epsilon,
        batch_size=64,
    )
    level0 = tokens[..., 0].reshape(-1)
    context_length = cfg.training.context_frames * tokens_per_frame
    xs, ys = _make_windows(level0, context_length, cfg.training.stride_tokens)

    n_val = max(1, int(xs.shape[0] * cfg.training.val_fraction))
    rng = np.random.default_rng(0)
    perm = rng.permutation(xs.shape[0])
    xs, ys = xs[perm], ys[perm]
    x_val, y_val = xs[:n_val], ys[:n_val]
    x_tr, y_tr = xs[n_val:], ys[n_val:]

    prior = build_prior(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=cfg.arch.embed_dim,
        num_layers=cfg.arch.num_layers,
        num_heads=cfg.arch.num_heads,
        ffn_dim=cfg.arch.ffn_dim,
    )
    prior.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
    )
    history = prior.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        verbose=2,
    )

    prior_dir = parent_run_dir / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    weights_path = prior_dir / "prior.weights.h5"
    prior.save_weights(weights_path)
    (prior_dir / "prior_config.json").write_text(cfg.model_dump_json(indent=2))

    deploy_dir = parent_run_dir / "deploy"
    tflite_path: Path | None = None
    if cfg.export_tflite:
        from compressionkit.export.release import write_checksums

        deploy_dir.mkdir(parents=True, exist_ok=True)
        tflite_path = _export_prior_tflite(prior, deploy_dir, vocab_size, context_length)
        manifest = {
            "parent_experiment": parent_exp.experiment_id,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "tokens_per_frame": tokens_per_frame,
            "arch": cfg.arch.model_dump(),
            "training": cfg.training.model_dump(),
        }
        (deploy_dir / "prior_manifest.json").write_text(json.dumps(manifest, indent=2))
        write_checksums(deploy_dir)

    return {
        "parent_run_dir": str(parent_run_dir),
        "prior_dir": str(prior_dir),
        "deploy_dir": str(deploy_dir) if cfg.export_tflite else "",
        "vocab_size": vocab_size,
        "context_length": context_length,
        "tflite_path": str(tflite_path) if tflite_path is not None else "",
        "final_loss": float(history.history.get("loss", [float("nan")])[-1]),
        "final_val_loss": float(history.history.get("val_loss", [float("nan")])[-1]),
    }


__all__ = ["train_prior_from_config"]
