"""Pydantic configuration for the two-stream PPG codec.

Two-stream architecture: a tiny codec for the slowly-varying baseline
channel and a standard RVQ for the pulsatile (detrended) residual channel.
Stitching is naturally improved because the baseline can be smoothly
interpolated at frame boundaries while the pulsatile stream is bounded
and mean-zero.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from compressionkit.configs.ppg_rvq import (
    DerivativeLossConfig,
    LrScheduleConfig,
    OutputConfig,
    SpectralLossConfig,
    UnifiedSourceConfig,
)


class DecomposeConfig(BaseModel):
    """Baseline/pulsatile decomposition parameters."""

    baseline_cutoff_hz: float = 0.5
    filter_order: int = 3
    epsilon: float = 1e-6


class BaselineModelConfig(BaseModel):
    """Model config for the baseline (trend) stream.

    The baseline is smooth and low-bandwidth, so we aggressively
    downsample it before coding. A tiny RVQ with few levels and a small
    codebook suffices.
    """

    downsample_factor: int = 16
    """Extra downsampling on top of the frame-level segmentation."""

    embedding_dim: int = 8
    latent_width: int = 64
    num_levels: int = 1
    num_stages: int = 2
    base_filters: int = 16
    multiplier: float = 1.25
    beta: float = 0.25
    use_ema: bool = True
    ema_decay: float = 0.99
    encoder_block_norm: str = "batch"
    decoder_block_norm: str = "none"
    decoder_head_norm: str = "layer"


class PulsatileModelConfig(BaseModel):
    """Model config for the pulsatile (residual) stream.

    Standard RVQ autoencoder on the detrended signal. Inherits the same
    architecture as the golden PPG RVQ but operates on a cleaner input
    (no baseline wander to waste capacity on).
    """

    embedding_dim: int = 16
    latent_width: int = 256
    num_levels: int = 2
    num_stages: int = 3
    base_filters: int = 48
    multiplier: float = 1.25
    beta: float = 0.25
    use_ema: bool = True
    ema_decay: float = 0.99
    encoder_block_norm: str = "batch"
    encoder_head_norm: str = "none"
    decoder_block_norm: str = "none"
    decoder_head_norm: str = "layer"


class TwoStreamDataConfig(BaseModel):
    """Data loading configuration for the two-stream pipeline."""

    datasets_dir: str = "/home/vscode/datasets"
    dataset_glob: str = "mesa-commercial-use/polysomnography/edfs/*.edf"
    sampling_rate: int = 64
    frame_size: int = 320
    segment_samples: int = 640
    offset_samples: int = 192
    target_label: str = "Pleth"
    batch_size: int = 64
    buffer_size: int = 10_000
    steps_per_epoch: int = 50
    epochs: int = 200
    shuffle_seed: int = 42
    gaussian_noise: list[float] = Field(default_factory=lambda: [0.005, 0.05])
    decompose: DecomposeConfig = Field(default_factory=DecomposeConfig)

    # TFRecord cache (reuse from golden pipeline)
    cache_enabled: bool = True
    cache_root: str = "datasets/ppg_tfrecord_cache"
    windows_per_subject_train: int = 512
    windows_per_subject_val: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.2

    # Unified per-source cache (overrides cache_enabled when set)
    unified_cache_enabled: bool = False
    unified_cache_root: str = "/home/vscode/datasets/ppg_cache"
    unified_sources: list[UnifiedSourceConfig] = Field(default_factory=list)
    max_train_windows: int | None = None
    max_val_windows: int | None = None


class FilteredLossConfig(BaseModel):
    """Lowpass-filtered MSE to focus on representable frequencies."""

    enabled: bool = False
    weight: float = 1.0
    cutoff_hz: float = 8.0
    num_taps: int = 65


class TwoStreamTrainingConfig(BaseModel):
    """Training hyperparameters for the two-stream pipeline."""

    learning_rate: float = 1e-3
    lr_schedule: LrScheduleConfig = Field(default_factory=LrScheduleConfig)
    derivative_loss: DerivativeLossConfig = Field(default_factory=DerivativeLossConfig)
    filtered_loss: FilteredLossConfig = Field(default_factory=FilteredLossConfig)
    spectral_loss: SpectralLossConfig = Field(default_factory=SpectralLossConfig)
    baseline_loss_weight: float = 1.0
    pulsatile_loss_weight: float = 1.0
    val_metric: str = "mse"
    val_mode: str = "min"
    early_stop_patience: int = 50
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 8
    reduce_lr_min_lr: float = 1e-5
    selection_metric: str = "val_loss"


class StitchingEvalConfig(BaseModel):
    """Stitching evaluation configuration."""

    enabled: bool = True
    methods: list[str] = Field(default_factory=lambda: ["hard_concat", "overlap_add", "linear_crossfade"])
    hop_ratio: float = 0.5
    duration_sec: float = 60.0
    num_recordings: int = 50
    batch_size: int = 32


class SpectralEvalConfig(BaseModel):
    """Spectral (frequency-domain) evaluation bands."""

    enabled: bool = True
    bands: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0.0, 0.5),  # sub-pulse (baseline/respiration)
            (0.5, 2.0),  # cardiac fundamental (30-120 BPM)
            (2.0, 4.0),  # first harmonic
            (4.0, 8.0),  # higher harmonics
            (8.0, 32.0),  # out-of-band (noise floor)
        ]
    )


class PhysiokitEvalConfig(BaseModel):
    """HR/HRV evaluation via PhysioKit."""

    enabled: bool = True
    low_hz: float = 0.5
    high_hz: float = 8.0
    order: int = 3
    min_peaks: int = 5


class TwoStreamEvaluationConfig(BaseModel):
    """Comprehensive evaluation for two-stream PPG codec."""

    num_samples: int = 200
    num_plot_samples: int = 20
    input_bit_depth: int = 16
    physiokit_metrics: PhysiokitEvalConfig = Field(default_factory=PhysiokitEvalConfig)
    spectral_metrics: SpectralEvalConfig = Field(default_factory=SpectralEvalConfig)
    stitching: StitchingEvalConfig = Field(default_factory=StitchingEvalConfig)


class PpgTwoStreamConfig(BaseModel):
    """Top-level configuration for the two-stream PPG codec pipeline."""

    run_name: str = "ppg_two_stream_run"
    data: TwoStreamDataConfig = Field(default_factory=TwoStreamDataConfig)
    baseline_model: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    pulsatile_model: PulsatileModelConfig = Field(default_factory=PulsatileModelConfig)
    training: TwoStreamTrainingConfig = Field(default_factory=TwoStreamTrainingConfig)
    evaluation: TwoStreamEvaluationConfig = Field(default_factory=TwoStreamEvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> PpgTwoStreamConfig:
        """Load config from a YAML file, merged with defaults."""
        from pathlib import Path as _Path

        import yaml

        with _Path(path).open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
        return cls.model_validate(user_cfg)
