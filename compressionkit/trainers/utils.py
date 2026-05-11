"""Shared trainer utilities.

Reusable loss builders, learning rate constructors, callback helpers, and
logger setup used by both the PPG and ECG RVQ training pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import keras

# ---------------------------------------------------------------------------
# Protocol for config objects accepted by shared builders
# ---------------------------------------------------------------------------

class _TrainingSection(Protocol):
    learning_rate: float
    lr_schedule: Any
    selection_metric: str
    val_metric: str
    val_mode: str
    early_stop_patience: int
    reduce_lr_on_plateau: bool
    reduce_lr_factor: float
    reduce_lr_patience: int
    reduce_lr_min_lr: float


class _OutputSection(Protocol):
    tensorboard: bool
    wandb: Any


class _TrainableConfig(Protocol):
    run_name: str
    training: _TrainingSection
    output: _OutputSection


# ---------------------------------------------------------------------------
# Derivative (smoothness) loss
# ---------------------------------------------------------------------------

def build_derivative_loss(weight: float) -> callable:
    """Build a first-difference penalty scaled by *weight*.

    The loss measures how well the reconstruction preserves the first
    derivative (sample-to-sample differences) of the target signal.
    Input shape is ``(B, 1, T, 1)``; the diff is computed along the time
    axis (axis=2).
    """
    import keras.ops as ops

    def derivative_loss(y_true, y_pred):
        dt = ops.subtract(y_true[:, :, 1:, :], y_true[:, :, :-1, :])
        dp = ops.subtract(y_pred[:, :, 1:, :], y_pred[:, :, :-1, :])
        return weight * ops.mean(ops.square(ops.subtract(dt, dp)))

    derivative_loss.__name__ = "derivative_loss"
    return derivative_loss


# ---------------------------------------------------------------------------
# Multi-scale spectral loss
# ---------------------------------------------------------------------------

def build_multi_scale_spectral_loss(
    weight: float,
    fft_sizes: list[int],
) -> callable:
    """Multi-resolution STFT loss combining spectral convergence and log-mag L1.

    For each FFT size *n* the loss computes:

    * **Spectral convergence** — Frobenius-norm ratio of the magnitude
      difference to the reference magnitude.
    * **Log-magnitude L1** — mean absolute difference of log-magnitude
      spectra.

    The two terms are summed per scale and the result is averaged over
    scales and weighted by *weight*.

    Input shape is ``(B, 1, T, 1)``; the time axis is axis 2.
    """
    import keras.ops as ops

    def _stft_mag(x, fft_length: int, hop_length: int):
        real, imag = ops.stft(
            x,
            sequence_length=fft_length,
            sequence_stride=hop_length,
            fft_length=fft_length,
        )
        return ops.sqrt(ops.square(real) + ops.square(imag) + 1e-8)

    def spectral_loss(y_true, y_pred):
        num_ch = y_true.shape[-1] if y_true.shape[-1] is not None else 1
        total = ops.convert_to_tensor(0.0)
        for ch in range(num_ch):
            yt = y_true[:, 0, :, ch]
            yp = y_pred[:, 0, :, ch]
            for n in fft_sizes:
                hop = n // 4
                st = _stft_mag(yt, n, hop)
                sp = _stft_mag(yp, n, hop)

                diff_sq = ops.sum(ops.square(st - sp), axis=(1, 2))
                ref_sq = ops.sum(ops.square(st), axis=(1, 2))
                sc = ops.mean(ops.sqrt(diff_sq + 1e-8) / (ops.sqrt(ref_sq + 1e-8) + 1e-6))

                log_mag = ops.mean(ops.abs(ops.log(st + 1e-8) - ops.log(sp + 1e-8)))

                total = total + sc + log_mag

        return weight * total / (len(fft_sizes) * num_ch)

    spectral_loss.__name__ = "spectral_loss"
    return spectral_loss


# ---------------------------------------------------------------------------
# Learning rate builder
# ---------------------------------------------------------------------------

def build_learning_rate(
    cfg: _TrainableConfig,
    *,
    steps_per_epoch: int,
) -> float | keras.optimizers.schedules.LearningRateSchedule:
    """Build optimizer learning rate or schedule from config."""
    lr_cfg = cfg.training.lr_schedule
    if not lr_cfg.enabled:
        return float(cfg.training.learning_rate)

    stype = lr_cfg.type.strip().lower()

    if stype == "cosine_decay":
        decay_steps = lr_cfg.first_decay_steps
        if decay_steps is None:
            decay_steps = max(1, cfg.data.epochs * steps_per_epoch)
        else:
            decay_steps = max(1, decay_steps)
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(cfg.training.learning_rate),
            decay_steps=decay_steps,
            alpha=lr_cfg.alpha,
        )

    if stype == "cosine_restarts":
        first_decay_steps = lr_cfg.first_decay_steps
        if first_decay_steps is None:
            first_decay_steps = max(1, steps_per_epoch)
        else:
            first_decay_steps = max(1, first_decay_steps)
        return keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=float(cfg.training.learning_rate),
            first_decay_steps=first_decay_steps,
            t_mul=lr_cfg.t_mul,
            m_mul=lr_cfg.m_mul,
            alpha=lr_cfg.alpha,
        )

    raise ValueError(f"Unsupported lr_schedule.type: {stype}")


# ---------------------------------------------------------------------------
# Callback builder
# ---------------------------------------------------------------------------

def build_callbacks(
    cfg: _TrainableConfig,
    *,
    run_dir: Path,
    lr_value: float | keras.optimizers.schedules.LearningRateSchedule,
    wandb_run: Any,
) -> list[keras.callbacks.Callback]:
    """Build the list of Keras training callbacks from config."""
    from compressionkit.logging.wandb_utils import build_wandb_callbacks

    tcfg = cfg.training
    selection_monitor = tcfg.selection_metric if tcfg.selection_metric.startswith("val_") else f"val_{tcfg.selection_metric}"
    best_ckpt_path = run_dir / "best_model.weights.h5"

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor=selection_monitor,
            mode=tcfg.val_mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=f"val_{tcfg.val_metric}",
            patience=tcfg.early_stop_patience,
            mode=tcfg.val_mode,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(run_dir / f"training_history_{cfg.run_name}.csv"),
    ]

    using_schedule = isinstance(lr_value, keras.optimizers.schedules.LearningRateSchedule)
    if tcfg.reduce_lr_on_plateau and not using_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=f"val_{tcfg.val_metric}",
                factor=tcfg.reduce_lr_factor,
                patience=tcfg.reduce_lr_patience,
                mode=tcfg.val_mode,
                min_lr=tcfg.reduce_lr_min_lr,
                verbose=1,
            )
        )

    if cfg.output.tensorboard:
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_dir, write_graph=False, write_images=False, update_freq="epoch",
            )
        )

    callbacks.extend(
        build_wandb_callbacks(run=wandb_run, log_model=cfg.output.wandb.log_model)
    )
    return callbacks


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(
    logger: logging.Logger,
    run_dir: Path,
    log_file: str | None,
) -> None:
    """Configure file and stream handlers for a trainer logger."""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = run_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
