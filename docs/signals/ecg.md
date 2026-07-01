---
icon: lucide/heart
---

# ECG (Electrocardiography)

## What is ECG?

Electrocardiography (ECG/EKG) records the electrical activity of the heart over time using electrodes placed on the skin. It's the gold standard for cardiac monitoring and is used to detect:

- **Arrhythmias** (atrial fibrillation, ventricular tachycardia)
- **Myocardial infarction** (heart attack)
- **Conduction abnormalities**
- **Heart rate and HRV**

The ECG waveform consists of the characteristic P-QRS-T complex, where the sharp QRS complex represents ventricular depolarization.

## Signal Characteristics

| Property | Value |
|----------|-------|
| Typical sampling rate | 250–500 Hz |
| Useful bandwidth | 0.5–40 Hz |
| Morphology | Sharp QRS complexes, smooth P and T waves |
| Key challenge | Preserving QRS timing and amplitude |

## ECG in compressionKIT

ECG compression in the v1 release has published RVQ bundles plus local
SPIHT and hybrid comparison lanes. The RVQ architecture is tuned for the higher
sampling rate and sharper morphology of ECG signals; SPIHT is the clean-signal
DSP baseline, and hybrid AI+DSP runs help characterize artifact-heavy regimes.

### Pipeline

- **Data source**: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — 21,799 12-lead ECG recordings
- **Preprocessing**: Resample 500 → 256 Hz, Lead II (`lead_index=1`), layer normalization
- **Model**: Conv2D encoder/decoder + EMA RVQ bottleneck (256-entry codebooks)
- **Compression range**: 2× – 64× (six golden configs)
- **Loss**: MSE + derivative loss (weight 0.1)

### Golden Models

See **[ECG Models (v1.0)](../models/ecg.md)** for the full results table, architecture breakdown, and training instructions.

| Model | CR | PRD (%) | Cosine |
|-------|----|---------|--------|
| ecg-rvq-02x | 2× | 2.50 | 0.9997 |
| ecg-rvq-04x | 4× | 4.09 | 0.9992 |
| ecg-rvq-08x | 8× | 7.48 | 0.9972 |
| ecg-rvq-16x | 16× | 11.18 | 0.9936 |
| ecg-rvq-32x | 32× | 16.04 | 0.9868 |
| ecg-rvq-64x | 64× | 22.35 | 0.9742 |

### Training

```bash
uv run compressionkit golden run ecg-rvq-8x
```

### Stitching evaluation

At higher compression ratios, frame-by-frame reconstruction can introduce
visible seams at frame boundaries. `compressionkit.evaluation.stitching`
provides four comparable strategies so the impact can be measured:

| Method | Window | Notes |
| --- | --- | --- |
| `hard_concat` | rectangular, no overlap | baseline (worst seams) |
| `overlap_add` | Hann, 50 % overlap | canonical default |
| `linear_crossfade` | triangular, 50 % overlap | LUT-free alternative |
| `tukey_overlap_add` | Tukey(α=0.25) | minimal taper knob |

The helper `seam_discontinuity_ratio` reports a first-difference RMS
ratio at seam neighbourhoods versus the rest of the signal — a scalar
that complements HR/HRV for driving future stitching work.

Enable it in any ECG RVQ YAML under `evaluation.stitching:` (see
[configs/ecg_rvq_256hz_32x_golden.yaml](../../configs/ecg_rvq_256hz_32x_golden.yaml)),
or run it against an already-trained model without retraining:

```bash
uv run python scripts/eval_ecg_stitching.py \
    --run-dir results/ecg_rvq_256hz_32x_golden \
    --duration-sec 30 --num-recordings 10
```
