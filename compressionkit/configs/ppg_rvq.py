"""Pydantic configuration models for PPG RVQ compression training."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StreamingConfig(BaseModel):
    """Config for subject-level streaming dataset mode."""

    enabled: bool = False
    windows_per_subject_train: int = 8
    windows_per_subject_val: int = 4
    subject_buffer_size: int = 512
    window_buffer_size: int = 20_000
    interleave_cycle_length: int = 16


class CacheConfig(BaseModel):
    """Config for TFRecord cache dataset mode."""

    enabled: bool = False
    cache_root: str = "datasets/ppg_tfrecord_cache"
    auto_build: bool = True
    force_rebuild: bool = False
    min_segment_scale: float = 1.2
    windows_per_subject_train: int = 64
    windows_per_subject_val: int = 16
    train_ratio: float = 0.8
    val_ratio: float = 0.2


class SyntheticMixConfig(BaseModel):
    """Config for synthetic PPG mixing during in-memory training."""

    enabled: bool = False
    fraction: float = 0.1
    seed: int = 1337
    heart_rate_bpm: list[float] = Field(default_factory=lambda: [50.0, 120.0])
    frequency_modulation: list[float] = Field(default_factory=lambda: [0.1, 0.5])
    ibi_randomness: list[float] = Field(default_factory=lambda: [0.02, 0.2])


class FilterConfig(BaseModel):
    """Config for bandpass filtering of input or target signals."""

    enabled: bool = False
    low_hz: float = 0.5
    high_hz: float = 8.0
    order: int = 3


class TransformConfig(BaseModel):
    """Signal transform domain for compression input.

    When ``domain`` is ``"raw"`` (default), the autoencoder operates on
    the raw time-domain signal.  ``"dwt"`` applies a multi-level discrete
    wavelet transform so the model compresses packed wavelet coefficients.
    """

    domain: str = Field(default="raw", description="Transform domain: 'raw' or 'dwt'.")
    dwt_levels: int = Field(default=4, description="Number of DWT decomposition levels.")
    dwt_wavelet: str = Field(default="db4", description="Wavelet name for DWT: haar, db4.")


class AugmentationConfig(BaseModel):
    """Tier-1 PPG augmentation settings for cross-domain robustness."""

    enabled: bool = False
    baseline_wander_prob: float = 0.3
    motion_artifact_prob: float = 0.3
    motion_snr_range: tuple[float, float] = Field(
        default=(12.0, 25.0),
        description="SNR range in dB for synthetic motion artifact injection.",
    )
    beat_scale_prob: float = 0.2
    time_warp_prob: float = 0.2
    empirical_noise_prob: float = 0.3
    empirical_snr_range: tuple[float, float] = Field(
        default=(10.0, 25.0),
        description="SNR range in dB for empirical noise injection.",
    )
    noise_bank_sources: list[str] = Field(
        default_factory=lambda: ["ppg_dalia", "wesad"],
        description="H5 dataset slugs to build noise bank from.",
    )
    noise_bank_max_segments: int = 2000
    noise_bank_root: str = "/home/vscode/datasets"


class UnifiedSourceConfig(BaseModel):
    """One source in a unified multi-source training mix."""

    slug: str
    weight: float | None = Field(
        default=None,
        description="Sampling weight. None = proportional to window count.",
    )


class UnifiedCacheConfig(BaseModel):
    """Config for unified per-source TFRecord cache dataset mode.

    When enabled, training data is loaded from pre-built per-source
    TFRecord caches via ``scripts/build_ppg_cache.py``.  Multiple
    sources can be combined with optional per-source sampling weights.
    """

    enabled: bool = False
    cache_root: str = "datasets/ppg_cache"
    sources: list[UnifiedSourceConfig] = Field(default_factory=list)
    shuffle_buffer: int = 10_000


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    datasets_dir: str = "/home/vscode/datasets"
    dataset_glob: str = "mesa-commercial-use/polysomnography/edfs/*.edf"
    sampling_rate: int = 64
    frame_size: int = 320
    segment_samples: int = 7680  # 64 Hz * 120 s
    offset_samples: int = 192  # 64 Hz * 3 s
    target_label: str = "Pleth"
    batch_size: int = 64
    buffer_size: int = 10_000
    steps_per_epoch: int = 250
    epochs: int = 100
    epsilon: float = 1e-3
    gaussian_noise: list[float] = Field(default_factory=lambda: [0.01, 0.1])
    shuffle_seed: int = 42
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    unified_cache: UnifiedCacheConfig = Field(default_factory=UnifiedCacheConfig)
    synthetic_mix: SyntheticMixConfig = Field(default_factory=SyntheticMixConfig)
    input_filter: FilterConfig = Field(default_factory=FilterConfig)
    target_filter: FilterConfig = Field(default_factory=FilterConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class ModelConfig(BaseModel):
    """RVQ autoencoder model architecture configuration."""

    embedding_dim: int = 16
    latent_width: int = 128
    codebook_sizes: list[int] | None = Field(
        default=None,
        description=(
            "Per-level codebook sizes for hierarchical / multi-rate RVQ. "
            "Length must equal num_levels. When set, overrides latent_width "
            "for the RVQ bottleneck (e.g. [512, 256, 128, 64] gives a "
            "coarse-to-fine hierarchy). When None, all levels use latent_width."
        ),
    )
    num_levels: int = 2
    num_stages: int = 4
    base_filters: int = 32
    multiplier: float = 1.25
    beta: float = 0.25
    encoder_block_norm: str = "batch"
    encoder_head_norm: str = "none"
    decoder_block_norm: str = "none"
    decoder_head_norm: str = "layer"
    use_ema: bool = False
    ema_decay: float = 0.99
    revive_dead_codes: bool = Field(
        default=False,
        description="Enable EnCodec/DAC-style dead-code revival in EMA RVQ.",
    )
    revive_threshold: float = Field(
        default=0.03,
        description=(
            "Codes whose normalized usage falls below revive_threshold/K are "
            "resampled from the current batch."
        ),
    )
    kmeans_init: bool = Field(
        default=False,
        description=(
            "Mini-batch k-means warm-start of the EMA RVQ codebooks before "
            "training. Run once eagerly via the trainer."
        ),
    )


class LrScheduleConfig(BaseModel):
    """Learning rate schedule configuration."""

    enabled: bool = False
    type: str = "cosine_restarts"
    first_decay_steps: int | None = None
    t_mul: float = 2.0
    m_mul: float = 1.0
    alpha: float = 1e-2


class DerivativeLossConfig(BaseModel):
    """First-derivative (smoothness) penalty on reconstruction."""

    enabled: bool = False
    weight: float = 0.1


class SpectralLossConfig(BaseModel):
    """Multi-scale spectral loss for frequency-domain reconstruction quality.

    Computes STFT at multiple FFT sizes and penalises differences in both
    spectral convergence (Frobenius norm ratio) and log-magnitude L1.
    """

    enabled: bool = False
    weight: float = 1.0
    fft_sizes: list[int] = Field(default_factory=lambda: [16, 32, 64, 128])


class TrainingConfig(BaseModel):
    """Training hyperparameters and schedule configuration."""

    learning_rate: float = 1e-3
    lr_schedule: LrScheduleConfig = Field(default_factory=LrScheduleConfig)
    derivative_loss: DerivativeLossConfig = Field(default_factory=DerivativeLossConfig)
    spectral_loss: SpectralLossConfig = Field(default_factory=SpectralLossConfig)
    val_metric: str = "mse"
    val_mode: str = "min"
    early_stop_patience: int = 25
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 8
    reduce_lr_min_lr: float = 1e-5
    selection_metric: str = "val_mse"
    validation_steps: int | None = None


class BandMetricsConfig(BaseModel):
    """Band-limited evaluation configuration."""

    enabled: bool = False
    low_hz: float = 0.5
    high_hz: float = 8.0
    order: int = 3


class PhysiokitMetricsConfig(BaseModel):
    """PhysioKit HR/HRV evaluation configuration."""

    enabled: bool = False
    low_hz: float = 0.5
    high_hz: float = 8.0
    order: int = 3
    min_peaks: int = 5


class LongRecordingEvalConfig(BaseModel):
    """Long-recording overlap-add evaluation for clinically meaningful HR/HRV."""

    enabled: bool = False
    duration_sec: float = 30.0
    hop_ratio: float = 0.5
    num_recordings: int = 10
    batch_size: int = 32


class EvaluationConfig(BaseModel):
    """Post-training evaluation configuration."""

    num_samples: int = 1000
    """How many random validation samples to evaluate (CSV + metrics)."""
    num_plot_samples: int = 50
    """Subset of ``num_samples`` that also receive a PNG plot artifact."""
    tflite_rep_batches: int = 8
    input_bit_depth: int = 16
    band_metrics: BandMetricsConfig = Field(default_factory=BandMetricsConfig)
    physiokit_metrics: PhysiokitMetricsConfig = Field(default_factory=PhysiokitMetricsConfig)
    long_recording: LongRecordingEvalConfig = Field(default_factory=LongRecordingEvalConfig)


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "compression-kit"
    entity: str | None = None
    group: str | None = None
    job_type: str = "train"
    tags: list[str] = Field(default_factory=list)
    mode: str = "auto"
    log_model: bool = False
    artifact_summary_only: bool = True


class OutputConfig(BaseModel):
    """Output directory and logging configuration."""

    results_root: str = "results"
    log_file: str | None = None
    tensorboard: bool = True
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class PpgRvqConfig(BaseModel):
    """Top-level configuration for PPG RVQ training pipeline."""

    run_name: str = "ppg_rvq_run"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> PpgRvqConfig:
        """Load config from a YAML file, merged with defaults."""
        from pathlib import Path as _Path

        import yaml

        with _Path(path).open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
        return cls.model_validate(user_cfg)
