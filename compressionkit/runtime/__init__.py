"""Lightweight inference runtime for compressionkit models.

This module provides a minimal-dependency runtime for running RVQ
codec models using only LiteRT (``ai-edge-litert``) and numpy.
No Keras, TensorFlow, or training dependencies are required.

Example::

    from compressionkit.runtime import RVQCodec

    # Load from a local deploy directory
    codec = RVQCodec("path/to/deploy/")
    indices = codec.encode(signal)
    recon = codec.decode(indices)
"""

from compressionkit.runtime.codec import RVQCodec

__all__ = ["RVQCodec"]
