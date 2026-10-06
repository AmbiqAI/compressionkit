---
title: Wavelet and SPIHT
description: Understand the DSP codec, its bit budget and package requirements.
---

SPIHT compresses wavelet coefficients without training a neural encoder. Start here when you want to evaluate a DSP baseline alongside RVQ using the same recordings and quality metrics.

## How it works

1. Split the signal into frames at the package's sample rate and frame size.
2. Transform each frame into wavelet coefficients.
3. Encode significant coefficients with SPIHT, subject to the configured bit budget. The package can also enable arithmetic coding.
4. Decode the coefficients and apply the inverse transform to reconstruct the frame.

The package records the wavelet, transform levels, bit budget and arithmetic-coding setting. Keep these settings with the encoded data; the decoder must use the matching configuration.

## What you download

A SPIHT package describes the codec parameters rather than neural model weights. Use the manifest and codec specification to inspect the operating point, and the saved stimulus and scorecard to evaluate it. Inspect the actual package before assuming that embedded source or a particular integration target is included.

## Try a package

After following [installation](/compressionkit/getting-started/#1-install), use the same API as the other codec families:

```python title="SPIHT round-trip"
import numpy as np
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x-v1.0")
frame = np.zeros(codec.frame_size, dtype=np.float32)
encoded = codec.compress(frame)
reconstructed = codec.decompress(encoded)
print(encoded.nbits, reconstructed.shape)
```

A zero frame checks loading and data flow only. Compare quality on representative recordings, including noise and artifacts. A target ratio does not establish the actual packet size or suitability for your signal.

## What to compare

- Reconstructed waveform quality at the same bit budget as RVQ.
- Signal-derived measurements, using the same reference and preprocessing.
- Actual encoded bits and required side information.
- Measured execution time and working memory on your intended platform.

See [PPG models](/compressionkit/models/ppg/), [ECG models](/compressionkit/models/ecg/) and [deployment](/compressionkit/deployment/) for results and integration boundaries.
