"""Model architecture modules for compressionkit."""

from compressionkit.models.rvq_autoencoder import (
    PrefixSupervisedVQAutoencoder,
    build_decoder_2d,
    build_encoder_2d,
    build_rvq_autoencoder,
    compute_compression_stats,
)
from compressionkit.models.ssm_autoencoder import (
    build_ssm_autoencoder,
    build_ssm_decoder,
    build_ssm_encoder,
    compute_compression_ratio,
)

__all__ = [
    "PrefixSupervisedVQAutoencoder",
    "build_decoder_2d",
    "build_encoder_2d",
    "build_rvq_autoencoder",
    "build_ssm_autoencoder",
    "build_ssm_decoder",
    "build_ssm_encoder",
    "compute_compression_ratio",
    "compute_compression_stats",
]
