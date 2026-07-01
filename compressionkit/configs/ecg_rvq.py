"""Pydantic configuration models for ECG RVQ compression training."""

from __future__ import annotations

from pydantic import BaseModel, Field

from compressionkit.configs.paths import default_datasets_dir


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
    cache_root: str = "datasets/ecg_tfrecord_cache"
    auto_build: bool = True
    force_rebuild: bool = False
    min_segment_scale: float = 1.2
    windows_per_subject_train: int = 64
    windows_per_subject_val: int = 16
    train_ratio: float = 0.8
    val_ratio: float = 0.2


class FilterConfig(BaseModel):
    """Config for bandpass filtering of input or target signals."""

    enabled: bool = False
    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 3
    forward_backward: bool = True


class AugmentationConfig(BaseModel):
    """Configurable augmentation and abstention regimes applied in training."""

    gaussian_noise: list[float] = Field(default_factory=lambda: [0.01, 0.1])
    empirical_noise_prob: float = 0.0
    empirical_snr_range: list[float] = Field(default_factory=lambda: [10.0, 25.0])
    noise_bank_max_segments: int = 20_000
    noise_bank_threshold_std: float = 2.0
    amplitude_warp: bool = False
    amplitude_warp_amplitude: list[float] = Field(default_factory=lambda: [0.05, 0.2])
    amplitude_warp_frequency: list[float] = Field(default_factory=lambda: [1.0, 5.0])
    random_cutout: bool = False
    cutout_factor: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    long_cutout_prob: float = 0.0
    long_cutout_factor: list[float] = Field(default_factory=lambda: [0.3, 0.9])
    null_frame_prob: float = 0.0

    # Wide contact-artifact augmentation (continuous severity, non-bimodal).
    # When enabled, every window is mixed with a sampled artifact at a Beta-drawn
    # power fraction (plus a continuous near-clean tail), spanning the full
    # contact-artifact family set. See RandomArtifactNoise1D.
    artifact_noise_enabled: bool = False
    artifact_severity_beta: list[float] = Field(default_factory=lambda: [0.9, 1.3])
    artifact_clean_prob: float = 0.08
    artifact_clean_severity_max: float = 0.05
    artifact_families: list[str] = Field(
        default_factory=lambda: ["colored", "mains", "motion", "lead_off", "weak_leak"]
    )
    artifact_bank_size: int = 2000


class SyntheticMixConfig(BaseModel):
    """Config for synthetic ECG mixing during training."""

    enabled: bool = False
    fraction: float = 0.1
    seed: int = 1337
    heart_rate_bpm: list[float] = Field(default_factory=lambda: [50.0, 100.0])
    noise_multiplier: list[float] = Field(default_factory=lambda: [0.2, 1.0])
    impedance: list[float] = Field(default_factory=lambda: [0.5, 1.5])


class DerivativeLossConfig(BaseModel):
    """First-derivative (smoothness) penalty on reconstruction."""

    enabled: bool = False
    weight: float = 0.1


class FilteredLossConfig(BaseModel):
    """Lowpass-filtered MSE to focus on frequencies the model can represent.

    Applies a fixed FIR lowpass filter to both the target and prediction
    before computing MSE.  This prevents the model from being penalised for
    high-frequency content that the bottleneck cannot represent.
    """

    enabled: bool = False
    weight: float = 1.0
    cutoff_hz: float = 10.0
    num_taps: int = 65


class SpectralLossConfig(BaseModel):
    """Multi-scale spectral loss for frequency-domain reconstruction quality.

    Computes STFT at multiple FFT sizes and penalises differences in both
    spectral convergence (Frobenius norm ratio) and log-magnitude L1.
    """

    enabled: bool = False
    weight: float = 1.0
    fft_sizes: list[int] = Field(default_factory=lambda: [32, 64, 128, 256])


class DwtLossConfig(BaseModel):
    """Frequency-weighted MSE via Haar DWT subbands.

    Decomposes both target and prediction into wavelet subbands and
    computes a weighted MSE per band.  This gives fine-grained control
    over which frequency octaves the model is penalised for.

    ``band_weights`` has length ``levels + 1``: the first entry weights
    the approximation (lowest frequency), and the remaining entries
    weight the detail coefficients from coarsest to finest.
    """

    enabled: bool = False
    weight: float = 1.0
    levels: int = 4
    band_weights: list[float] = Field(
        default_factory=lambda: [4.0, 2.0, 1.0, 0.25, 0.1],
    )


class TransformConfig(BaseModel):
    """Signal transform domain for compression input.

    When ``domain`` is ``"raw"`` (default), the autoencoder operates on
    the raw time-domain signal.  ``"dwt"`` applies a multi-level discrete
    wavelet transform (flat-packed, same length), and ``"stft"`` applies a
    short-time Fourier transform producing a 2-D time-frequency image.
    """

    domain: str = Field(
        default="raw",
        description="Transform domain: 'raw', 'dwt', or 'stft'.",
    )
    # DWT parameters
    dwt_levels: int = Field(default=4, description="Number of DWT decomposition levels.")
    dwt_wavelet: str = Field(default="haar", description="Wavelet name for DWT.")
    # STFT parameters
    stft_n_fft: int = Field(default=64, description="FFT window length for STFT.")
    stft_hop_length: int = Field(default=16, description="Hop length for STFT.")
    stft_window: str = Field(default="hann", description="Window function for STFT.")


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    dataset_id: str | None = Field(
        default=None,
        description="Identifier registered in compressionkit.datasets.contract (see #26).",
    )
    datasets_dir: str = Field(default_factory=default_datasets_dir)
    dataset_glob: str = "ptbxl/*.h5"
    sampling_rate: int = 500
    target_sample_rate: int | None = None
    frame_size: int = 1024
    segment_samples: int = 5000
    lead_index: int = 1
    leads: list[int] | None = None
    batch_size: int = 64
    buffer_size: int = 10_000
    steps_per_epoch: int = 250
    epochs: int = 100
    epsilon: float = 1e-3
    gaussian_noise: list[float] = Field(default_factory=lambda: [0.01, 0.1])
    shuffle_seed: int = 42
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    synthetic_mix: SyntheticMixConfig = Field(default_factory=SyntheticMixConfig)
    input_filter: FilterConfig = Field(default_factory=FilterConfig)
    target_filter: FilterConfig = Field(default_factory=FilterConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)

    @property
    def effective_sample_rate(self) -> int:
        """Sample rate after optional resampling."""
        return self.target_sample_rate if self.target_sample_rate is not None else self.sampling_rate

    @property
    def num_leads(self) -> int:
        """Number of ECG leads to load (1 when using single lead_index)."""
        return len(self.leads) if self.leads is not None else 1


class ModelConfig(BaseModel):
    """RVQ autoencoder model architecture configuration."""

    model_type: str = Field(
        default="default",
        description=(
            "Model architecture family: 'default' (standard RVQ autoencoder) "
            "or 'dwt_rvq' (DWT front-end + learned codec + iDWT back-end)."
        ),
    )
    dwt_wavelet: str = Field(
        default="bior4.4",
        description="Wavelet name for DWT codec model_type='dwt_rvq'.",
    )
    dwt_levels: int = Field(
        default=6,
        description="DWT decomposition levels for model_type='dwt_rvq'.",
    )

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
    use_residual: bool = False
    use_ema: bool = False
    ema_decay: float = 0.99
    encoder_block_norm: str = "batch"
    encoder_head_norm: str = "none"
    decoder_block_norm: str = "none"
    decoder_head_norm: str = "layer"
    encoder_type: str = Field(
        default="default",
        description="Encoder architecture: 'default' or 'inverted_residual'.",
    )
    expand_ratio: float = Field(
        default=4.0,
        description="MBConv expansion ratio (inverted_residual only).",
    )
    causal: bool = Field(
        default=False,
        description="Use causal (left-only) padding in the encoder.",
    )
    discard_tail: int = Field(
        default=0,
        description="Leading latent positions to discard (warm-up for causal mode).",
    )
    bottleneck_type: str = Field(
        default="rvq",
        description="Bottleneck quantizer: 'rvq' or 'fsq'.",
    )
    fsq_levels: list[int] | None = Field(
        default=None,
        description=(
            "Per-dimension FSQ level counts (used when bottleneck_type='fsq'). "
            "Implicit codebook size = product of levels; embedding_dim is forced "
            "to len(fsq_levels)."
        ),
    )
    decoder_type: str = Field(
        default="default",
        description=(
            "Decoder family: 'default' (conv), 'ssm' (S4D blocks), or "
            "'hierarchical' (coarse + residual/detail RVQ branches), or "
            "'hierarchical_hybrid' (summed RVQ coarse path plus per-level detail), or "
            "'hierarchical_adaptor' (summed RVQ trunk with per-level residual adaptors)."
        ),
    )
    decoder_state_size: int = Field(
        default=32,
        description="State channels per S4D block (decoder_type='ssm').",
    )
    decoder_num_ssm_blocks: int = Field(
        default=2,
        description="Number of S4DBlocks per decoder stage (decoder_type='ssm').",
    )
    hier_detail_scale: float = Field(
        default=0.25,
        description=("Scale for residual/detail contributions when using hierarchical decoder types."),
    )
    revive_dead_codes: bool = Field(
        default=False,
        description="Enable EnCodec/DAC-style dead-code revival in EMA RVQ.",
    )
    revive_threshold: float = Field(
        default=0.03,
        description=(
            "Codes whose normalized usage falls below revive_threshold/K are resampled from the current batch."
        ),
    )
    kmeans_init: bool = Field(
        default=False,
        description=(
            "Mini-batch k-means warm-start of the EMA RVQ codebooks before training. Run once eagerly via the trainer."
        ),
    )
    structured_dropout: bool = Field(
        default=False,
        description=(
            "Enable structured RVQ dropout: randomly truncate the number of "
            "active RVQ levels during training so the decoder learns to "
            "reconstruct from partial quantization (variable bitrate)."
        ),
    )
    dropout_levels: list[int] | None = Field(
        default=None,
        description=(
            "Allowed active-level counts for structured dropout, sampled "
            "uniformly each forward pass. E.g. [1, 2, 4, 8] for powers-of-2. "
            "If None, auto-generates powers of 2 up to num_levels."
        ),
    )
    decoder_activation: str = Field(
        default="relu",
        description=(
            "Activation function for decoder blocks: 'relu', 'snake', or any Keras-registered activation name."
        ),
    )
    encoder_blocks_per_stage: int = Field(
        default=1,
        description=(
            "Number of conv/DW blocks per encoder stage (before the stride-2 "
            "downsample). Increasing this adds capacity without changing the "
            "temporal downsample factor."
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


class BetaAnnealConfig(BaseModel):
    """Schedule for the RVQ commitment-loss weight (beta).

    When enabled, ``beta`` is annealed from ``start`` to ``end`` over
    ``epochs`` epochs. Useful for stabilising codebook usage in
    multi-level RVQ with small per-level codebooks: a high initial beta
    locks the encoder near the codebook (preventing dead codes), then
    annealing toward a low beta releases the encoder for fine-tuning.
    """

    enabled: bool = False
    start: float = 1.0
    end: float = 0.1
    epochs: int = 50
    mode: str = "cosine"


class RvqPrefixLossConfig(BaseModel):
    """Auxiliary coarse-to-fine supervision for multi-level RVQ prefixes."""

    enabled: bool = False
    weights: list[float] = Field(
        default_factory=list,
        description=(
            "Loss weights for decoded RVQ prefixes. For num_levels=2, a single "
            "weight supervises level 1 before the full two-level reconstruction."
        ),
    )
    target: str = Field(
        default="lowpass",
        description="Prefix target: 'lowpass' for coarse morphology or 'full'.",
    )
    lowpass_kernel: int = Field(
        default=9,
        description="Odd moving-average kernel width for lowpass prefix targets.",
    )
    start_epoch: int = Field(
        default=0,
        description="Epoch before which prefix loss scale is held at zero.",
    )
    ramp_epochs: int = Field(
        default=0,
        description="Number of epochs to linearly ramp prefix loss scale from zero to one.",
    )


class LabelTrustConfig(BaseModel):
    """Reference-free label-trust loss weighting (``n0``-aware strong/weak labels).

    Estimates the inherent corruption already present in each *target* window
    and down-weights the reconstruction loss where the target is unreliable, so
    the model applies higher scrutiny to clean (low-``n0``) targets. This is a
    training-time-only signal applied via ``sample_weight``; it does not affect
    the exported model. Only supported for raw-domain training.
    """

    enabled: bool = False
    granularity: str = "window"
    w_min: float = 0.25
    gamma: float = 1.0
    half_sat_ratio: float = 0.25
    hf_window_ms: float = 40.0
    baseline_window_ms: float = 900.0
    smooth_ms: float = 300.0
    normalize: bool = True


class TrainingConfig(BaseModel):
    """Training hyperparameters and schedule configuration."""

    learning_rate: float = 1e-3
    lr_schedule: LrScheduleConfig = Field(default_factory=LrScheduleConfig)
    val_metric: str = "mse"
    val_mode: str = "min"
    early_stop_patience: int = 25
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 8
    reduce_lr_min_lr: float = 1e-5
    selection_metric: str = "val_mse"
    validation_steps: int | None = None
    derivative_loss: DerivativeLossConfig = Field(default_factory=DerivativeLossConfig)
    filtered_loss: FilteredLossConfig = Field(default_factory=FilteredLossConfig)
    spectral_loss: SpectralLossConfig = Field(default_factory=SpectralLossConfig)
    dwt_loss: DwtLossConfig = Field(default_factory=DwtLossConfig)
    beta_anneal: BetaAnnealConfig = Field(default_factory=BetaAnnealConfig)
    rvq_prefix_loss: RvqPrefixLossConfig = Field(default_factory=RvqPrefixLossConfig)
    label_trust: LabelTrustConfig = Field(default_factory=LabelTrustConfig)


class BandMetricsConfig(BaseModel):
    """Band-limited evaluation configuration."""

    enabled: bool = False
    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 3


class StitchingEvalConfig(BaseModel):
    """Long-recording stitching evaluation.

    When enabled, evaluates the configured stitching strategies on a
    handful of full-length validation recordings and reports
    reconstruction quality plus a seam-discontinuity ratio for each
    method. Useful for measuring how sensitive downstream analyses are
    to the choice of stitching beyond frame-level PRD.
    """

    enabled: bool = False
    methods: list[str] = Field(
        default_factory=lambda: ["overlap_add", "hard_concat"],
        description="Stitching methods to evaluate (see compressionkit.evaluation.stitching.STITCH_METHODS).",
    )
    hop_ratio: float = 0.5
    num_recordings: int = 10
    duration_sec: float = 30.0
    seam_radius: int = 4
    batch_size: int = 32
    hr_hrv: bool = True
    """Compute HR/HRV metrics on each stitched trace (issue #3)."""


class EvaluationConfig(BaseModel):
    """Post-training evaluation configuration."""

    num_samples: int = 1000
    """How many random validation samples to evaluate (CSV + metrics)."""
    num_plot_samples: int = 50
    """Subset of ``num_samples`` that also receive a PNG plot artifact.
    Capped at ``num_samples``. Plots are slow to render and large to ship,
    so this is typically much smaller than ``num_samples``."""
    tflite_rep_batches: int = 8
    input_bit_depth: int = 16
    band_metrics: BandMetricsConfig = Field(default_factory=BandMetricsConfig)
    stitching: StitchingEvalConfig = Field(default_factory=StitchingEvalConfig)


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


class EcgRvqConfig(BaseModel):
    """Top-level configuration for ECG RVQ training pipeline."""

    run_name: str = "ecg_rvq_run"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> EcgRvqConfig:
        """Load config from a YAML file, merged with defaults."""
        from pathlib import Path as _Path

        import yaml

        with _Path(path).open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
        return cls.model_validate(user_cfg)
