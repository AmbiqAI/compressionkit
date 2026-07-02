---
icon: lucide/activity
---

# PPG Workflow

This page describes the current production-oriented PPG path in compressionKIT:
published RVQ bundles for immediate use, plus reproducible RVQ, SPIHT, and hybrid
golden lanes for deeper evaluation.

## What Is Supported Today

The current supported task is:

- Input: 64 Hz PPG windows from the open unified PPG v1 cache or from your own signal.
- Published runtime path: load `Ambiq/compressionkit-ppg-{2,4,8,16,32}x-v1.0` bundles.
- Golden comparison lanes: RVQ, SPIHT, and hybrid runs evaluated through the same scorecard shape.
- Deployment path: export encoder/codebook artifacts plus manifests, checksums, and reference vectors.
- Evaluation path: compare waveform metrics, HR/HRV preservation, noise buckets, and artifact sweeps.

In practice, this gives teams a path from a quick bundle test to a reproducible
release package without changing toolchains midway through the project.

## End-To-End Flow

1. Try a published RVQ bundle on a synthetic or customer-provided waveform.
2. Inspect the PPG model page for CR, HR/HRV, SPIHT, RVQ, hybrid, noise, and artifact tradeoffs.
3. Build the open PPG cache only when you need to reproduce or train.
4. Run a golden experiment through `compressionkit golden run <id>`.
5. Validate the generated `deploy/` package before integrating encoder/codebook artifacts.

## Core Design Choices

### Fixed-window processing

The default flow uses a fixed frame size so training, evaluation, and deployment all share the same model assumptions. That keeps exported inference predictable on constrained targets.

### YAML-driven configuration

The full run is driven from YAML, which makes it easy to keep experiments reproducible and to compare configurations across compression ratios.

### Embedded-oriented outputs

The workflow does not stop at training. It produces deployment artifacts, evaluation summaries, and reconstruction samples in the same result directory.

## Primary Inputs And Outputs

| Stage | Input | Output |
|------|------|------|
| Runtime test | Published HuggingFace bundle or local deploy package | Encoded tokens and reconstructed waveform |
| Data loading | Open PPG cache or customer waveform | Windowed training/evaluation frames |
| Codec model | PPG frames | RVQ tokens, SPIHT bitstream, or hybrid output depending on lane |
| Evaluation | Original/noisy/reconstructed signals | PRD, cosine, HR/HRV, noise buckets, artifact scorecards |
| Export | Golden run output | LiteRT/TFLite, C headers, manifests, checksums, reference vectors |

## Recommended Entry Points

Use the golden runner when you want a release-grade reproduction:

```bash
uv run compressionkit golden run ppg-rvq-8x
```

Use the published runtime path when you only need to try a codec:

```python
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-8x-v1.0")
```

## Reference Operating Points

compressionKIT currently publishes five PPG RVQ bundles and keeps DSP/hybrid
comparison lanes reproducible locally:

- 2x for highest fidelity.
- 4x for balanced quality and savings.
- 8x for an aggressive but broadly useful operating point.
- 16x for high compression with more visible HRV tradeoff.
- 32x for storage/radio-constrained telemetry.

The measured RVQ metrics and SPIHT/RVQ/hybrid noise-artifact tradeoffs are in
[PPG Models](../models/ppg.md).

## Related Demo

The workflow is also exposed through a browser demo that shows reconstruction quality, compression tradeoffs, and live controls in a more visual format. See [PPG Codec Demo](../demo/ppg-codec.md).
