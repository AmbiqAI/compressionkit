---
title: Hybrid compression
description: Understand learned preprocessing followed by a SPIHT codec.
---

The hybrid codec applies a learned wavelet-gain preprocessor before SPIHT compression. Consider it when noise suppression is part of your objective and you want to compare that behavior with direct waveform compression.

## How it works

1. Apply the package's learned preprocessor to a signal frame.
2. Encode the processed frame with the package's SPIHT settings.
3. Decode the SPIHT payload to reconstruct the processed signal.

This is not the RVQ entropy-prior path. The hybrid package combines learned preprocessing with SPIHT; a two-stage RVQ package instead entropy-codes RVQ token indices.

## Preserve the distinction between fidelity and denoising

A reconstruction can differ from a noisy input because preprocessing removed noise, because compression discarded information, or both. Compare against the observed input and, where available, a clean reference. Report the reference used with every metric rather than interpreting a single waveform error as a denoising score.

## What you download

The package includes SPIHT codec settings and a hybrid manifest identifying the learned preprocessor. The runtime selects the quantized LiteRT preprocessor when the package supplies one and otherwise can use its Keras artifact. Inspect the manifest to establish the dependencies for the particular package you select.

```python title="Hybrid round-trip"
import numpy as np
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-hybrid-4x-v1.0")
frame = np.zeros(codec.frame_size, dtype=np.float32)
encoded = codec.compress(frame)
reconstructed = codec.decompress(encoded)
print(encoded.nbits, reconstructed.shape)
```

Install from the [getting-started guide](/compressionkit/getting-started/#1-install). This synthetic example checks the interface, not reconstruction quality.

## Before choosing hybrid

Compare hybrid and direct SPIHT on the same recordings. Check clean-signal distortion, artifact behavior, and the physiological measurements relevant to your application. Include preprocessing in your execution-time and memory measurements.

Continue with [method comparison](/compressionkit/methods/), the [validation scorecard](/compressionkit/validation-scorecard/) and [deployment](/compressionkit/deployment/).
