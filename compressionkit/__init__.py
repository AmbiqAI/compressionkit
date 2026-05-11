"""compressionKIT — AI-powered compression for physiological signals.

Top-level convenience imports. The authoritative definitions live in the
relevant submodules; re-exports here let users write::

    from compressionkit import build_rvq_autoencoder, PRD

instead of navigating the full module tree.
"""

from compressionkit.evaluation import (
    PRD,
    TruePRD,
    compute_signal_metrics,
    evaluate_long_recordings,
    reconstruct_overlap_add,
)
from compressionkit.layers import (
    EmaResidualVectorQuantizer,
    FiniteScalarQuantizer,
    ResidualVectorQuantizer,
    VectorQuantizer,
)
from compressionkit.models import (
    build_decoder_2d,
    build_encoder_2d,
    build_rvq_autoencoder,
    compute_compression_stats,
)

__version__ = "0.1.0"

__all__ = [
    "PRD",
    "EmaResidualVectorQuantizer",
    "FiniteScalarQuantizer",
    "ResidualVectorQuantizer",
    "TruePRD",
    "VectorQuantizer",
    "__version__",
    "build_decoder_2d",
    "build_encoder_2d",
    "build_rvq_autoencoder",
    "compute_compression_stats",
    "compute_signal_metrics",
    "evaluate_long_recordings",
    "reconstruct_overlap_add",
]
