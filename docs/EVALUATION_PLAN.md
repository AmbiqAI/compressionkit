# Evaluation Architecture

> This document describes the evaluation harness built into CompressionKit.
> The harness provides codec-agnostic, reproducible quality measurement for
> any signal compression algorithm (classical or neural) targeting
> physiological waveforms (ECG, PPG).

---

## 1. Infrastructure Overview

| Concern | Module | Status |
|---|---|---|
| Signal-fidelity metrics (PRD, SNR, cosine) | `compressionkit/evaluation/metrics.py` | ✅ |
| PPG peak alignment, physiokit HR/HRV | `compressionkit/evaluation/metrics.py` | ✅ |
| ECG HR/HRV, R-peak alignment | `compressionkit/evaluation/metrics.py` | ✅ |
| Spectral metrics (band power, coherence, weighted freq PRD) | `compressionkit/evaluation/spectral_metrics.py` | ✅ |
| Noise floor / QRS SNR / HF residual | `compressionkit/evaluation/noise.py` | ✅ |
| Stitching: hard concat, linear crossfade, Tukey OLA, seam metric | `compressionkit/evaluation/stitching.py` | ✅ |
| Long-recording overlap-add evaluation | `compressionkit/evaluation/overlap_add.py` | ✅ |
| Scorecard renderer | `compressionkit/evaluation/scorecard.py` | ✅ |
| ECG-specific stitching helpers | `compressionkit/evaluation/ecg_stitching.py` | ✅ |
| Sample artifact dumping | `compressionkit/evaluation/artifacts.py` | ✅ |
| Unified Codec protocol + adapters | `compressionkit/evaluation/codec.py` | ✅ |
| RVQ adapter (load-and-evaluate) | `compressionkit/evaluation/rvq_codec.py` | ✅ |
| Adversarial input battery | `compressionkit/evaluation/adversarial.py` | ✅ |
| RVQ QoS / confidence indicator | `compressionkit/evaluation/qos.py` | ✅ |
| Stitching A/B report | `compressionkit/evaluation/stitching_report.py` | ✅ |
| PPG random cutout augmentation | `compressionkit/preprocessing/ppg.py` | ✅ |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  compressionkit.evaluation                                   │
│                                                              │
│  ┌─────────────────┐                                         │
│  │ Codec protocol  │  ←  SPIHT-AC, RVQ, Identity, …          │
│  │  encode/decode  │                                         │
│  │  EncodedFrame   │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│  ┌────────▼──────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ Fidelity metrics  │  │ Spectral metrics │  │  Noise    │  │
│  └───────────────────┘  └──────────────────┘  └───────────┘  │
│                                                              │
│  ┌───────────────────┐  ┌──────────────────────┐             │
│  │ Stitching + A/B   │  │ Adversarial inputs   │             │
│  └───────────────────┘  └──────────────────────┘             │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ RVQ QoS (confidence indicator)                   │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ Scorecard builder + CLI (scripts/eval_codec.py)  │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### 3.1 Codec Protocol (`evaluation/codec.py`)

```python
@runtime_checkable
class Codec(Protocol):
    name: str
    target_cr: float
    modality: str       # "ppg" | "ecg"
    sample_rate: int
    frame_size: int

    def encode(self, frame: np.ndarray) -> EncodedFrame: ...
    def decode(self, encoded: EncodedFrame) -> np.ndarray: ...
```

Adapters: `SpihtAcCodec`, `RvqCodec`, `IdentityCodec`.

### 3.2 Adversarial Input Battery (`evaluation/adversarial.py`)

Generators for zero input, Gaussian noise, DC offset, sinusoids, steps, zero-bursts, and signal-noise mixtures. Injectors for baseline wander and powerline interference.

`run_adversarial_battery(codec)` returns `AdversarialResult` per scenario with:
- `energy_ratio` — output/input energy; detects hallucination on silent input.
- `hallucinated_peaks` — peaks in the physiological band when input had none.
- `output_band_power` — fraction of output power in the physio band.
- `reconstruction_prd_mean` — PRD for signal-present tests.

### 3.3 RVQ QoS (`evaluation/qos.py`)

Per-frame confidence ∈ [0, 1] derived from:
1. Mean quantization distance (how well the codebook fits the latent).
2. Relative residual (what fraction of latent energy the codebook cannot explain).
3. Initial latent norm (two-sided guard: anomalously low or high energy).

`RvqQoSCalibrator` is fit on in-distribution validation frames and maps raw QoS
to a percentile-based confidence via geometric mean of the three sub-scores.

### 3.4 Stitching A/B (`evaluation/stitching_report.py`)

`compare_stitching_methods(codec, signal)` produces a DataFrame sweeping
4 methods × N hop ratios. Reports PRD and seam discontinuity ratio per
combination.

---

## 4. CLI

```bash
python scripts/eval_codec.py \
    --codec spiht_ac --modality ppg --cr 4 \
    --tiers fidelity adversarial stitching \
    --out results/evals/spiht_ppg_04x

python scripts/eval_codec.py \
    --codec rvq --modality ppg \
    --rvq-run results/ppg_rvq_64hz_04x_golden \
    --tiers fidelity adversarial stitching qos \
    --out results/evals/rvq_ppg_04x_golden
```

Tiers: `fidelity`, `spectral`, `adversarial`, `stitching`, `qos`.

Outputs: `report.json` + `report.md`.

---

## 5. Scorecard Acceptance Criteria

A codec passes evaluation when its scorecard contains measured values for:

| Tier | Metrics |
|---|---|
| Fidelity | PRD, SNR, cosine (mean/median/p10/p90) |
| Spectral | Per-band PSD error, coherence, weighted-frequency PRD |
| Domain | HR/HRV error, peak F1, peak timing MAE |
| Adversarial | Zero-input energy ratio & hallucinated peaks; noise-robustness at 5/10/20 dB |
| Stitching | Seam discontinuity for ≥2 methods at ≥2 hop sizes |
| QoS (neural) | Confidence distribution on in-dist vs OOD |

---

## 6. Hallucination Defense (Neural Codecs)

Neural codecs (RVQ) defend against hallucination through four layers:

1. **Bounded codebook** — the reconstruction is always a sum of learned codewords; the output vocabulary is finite.
2. **Measured worst case** — adversarial battery quantifies what the codec produces on pathological inputs.
3. **Per-frame QoS gate** — the calibrated confidence flag allows downstream systems to reject untrusted frames.
4. **Cutout training** — `random_cutout` augmentation forces the model to produce silence for silent inputs.

Classical codecs (SPIHT-AC) are deterministic by construction: zero in → zero out.

---

## 7. Migration Roadmap (HeliaEdge / PhysioKit)

| Block | Eventual home | Rationale |
|---|---|---|
| Fidelity metrics (PRD/SNR/cosine) | PhysioKit | Generic signal processing |
| HR/HRV, peak alignment | PhysioKit | Pure physiology |
| Spectral metrics | PhysioKit | Signal processing |
| Noise floor estimation | PhysioKit | Physio noise floors |
| Stitching (windows, OLA) | HeliaEdge | Generic DSP |
| Codec protocol + adapters | CompressionKit | Codec-specific |
| Adversarial battery | CompressionKit | Codec evaluation |
| RVQ QoS | HeliaEdge | Reusable AI tooling |
| Scorecard + CLI | CompressionKit | Project-specific output |
