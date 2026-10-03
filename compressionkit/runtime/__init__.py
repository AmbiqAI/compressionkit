"""Lightweight inference runtime for compressionkit models.

RVQ model inference uses LiteRT and NumPy. Importing this module also
loads the compressionkit package and its Keras and evaluation dependencies;
install the package dependencies even when using exported LiteRT models.

Example::

    from compressionkit.runtime import RVQCodec

    # Load from a local deploy directory
    codec = RVQCodec("path/to/deploy/")
    indices = codec.encode(signal)
    recon = codec.decode(indices)

    # Two-stage compression with entropy prior
    from compressionkit.runtime.prior import EntropyPrior
    from compressionkit.runtime.two_stage import TwoStageCodec

    prior = EntropyPrior("prior.tflite")
    two_stage = TwoStageCodec(codec, prior)
    result = two_stage.compress(signal)
"""

from compressionkit.runtime.base import Codec, EncodedFrame
from compressionkit.runtime.codec import RVQCodec
from compressionkit.runtime.hybrid import HybridSpihtCodec
from compressionkit.runtime.loader import load_codec, resolve_deploy_dir
from compressionkit.runtime.spiht import SpihtCodec

__all__ = [
    "Codec",
    "EncodedFrame",
    "HybridSpihtCodec",
    "RVQCodec",
    "SpihtCodec",
    "load_codec",
    "resolve_deploy_dir",
]
