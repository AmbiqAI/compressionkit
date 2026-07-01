# Validation Scorecard

A framework for validating that lossy compression preserves physiologically and clinically relevant features — not just waveform fidelity.

## Core claim

> The codec is a task-preserving, morphology-bounded, versioned lossy compression system for ECG and PPG, with explicit validation of clinically and analytically relevant features.

## Evidence layers

1. **Waveform fidelity** — Time-domain and frequency-domain reconstruction error.
2. **Physiological feature fidelity** — HR, HRV, pulse timing, ECG morphology, PPG AC/DC, SpO2-sensitive ratios.
3. **Downstream task preservation** — Same algorithms run on raw and reconstructed signals produce non-inferior outputs.
4. **Operational safety** — Boundary handling, fallback behavior, bitrate adaptation, versioning, and decoder compatibility.

---

## Measurement philosophy

### Agreement vs accuracy

Most metrics in this scorecard measure **agreement** between raw and compressed signals (same algorithm applied to both). This is distinct from **task accuracy** (compressed vs ground truth). Agreement tells you "did compression break anything?" but is bounded by the algorithm's own repeatability.

### Algorithm repeatability baseline

Before attributing a metric delta to the codec, establish the algorithm's inherent jitter:

- Run the same metric on the raw signal with ±1 LSB noise injection (10 repetitions)
- The standard deviation of those measurements is the **measurement floor**
- Compression error within the measurement floor is indistinguishable from algorithm noise

### Noise preservation vs noise removal

A codec that denoises (e.g., learned RVQ) may show higher PRD against a noisy ground truth but better downstream task performance. Conversely, a signal-agnostic codec (e.g., SPIHT) preserves noise faithfully — lower PRD but potentially worse clinical metrics at high compression.

The scorecard must report both:

- PRD / PRDN (noise-adjusted PRD)
- Downstream metric agreement (HR, HRV, morphology)

A codec that "hides" bad signal quality (low SQI raw → high SQI reconstructed) is dangerous. The scorecard explicitly checks for this.

---

## Implementation tiers

| Tier | Scope | Target |
|------|-------|--------|
| 1 — Required for v1 | HR/HRV preservation, PRD/SNR, boundary behavior, versioning | Now |
| 2 — Pre-deployment | Morphology metrics (QRS width, ST segment, pulse shape), band power, SQI honesty | Before deployment |
| 3 — Post-v1 | Arrhythmia classifier agreement, SpO2 ratio, full bucket analysis | Roadmap |

---

## ECG validation

### HR / HRV preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| Timing | R-peak timing error (ms) | Per beat, report mean / p95 / max | Core metric |
| Heart rate | HR error (bpm) | Per-window and aggregate | |
| HRV time-domain | RMSSD delta (ms or %) | Sensitive to R-R timing | |
| HRV time-domain | SDNN delta (ms or %) | Window-length dependent | |
| HRV time-domain | pNN50 delta (%) | Sensitive to beat timing errors | |
| Beat intervals | R-R interval error (ms) | Per beat and per window | |
| Signal quality | SQI agreement | Codec should not hide bad signal | |
| Boundary | False/missed R-peaks near chunk edges | Report separately | |

### Morphology preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| Waveform | PRD / PRDN (%) | Useful but not sufficient alone | |
| Waveform | MAE / RMSE | Report by segment type | |
| Frequency | Band-power error (%) | Include diagnostic frequency ranges | |
| QRS | QRS width delta (ms) | Important for conduction abnormalities | |
| QRS | QRS amplitude delta | Normalize carefully | |
| ST segment | ST deviation delta | Critical for diagnostic use cases | |
| T-wave | T-wave amplitude / morphology delta | Include abnormal T-wave cases | |
| Baseline | Baseline drift preservation/removal | Low-frequency error | |
| Beat morphology | Beat embedding similarity | Cosine distance with frozen model | |
| Boundary | Morphology distortion near chunk edges | Report first/last N ms | |

### Downstream task preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| R-peak detection | Sensitivity / PPV delta | Same detector on raw and reconstructed | |
| Arrhythmia | Classifier agreement (F1 / kappa) | AFib, PVC, PAC, tachy/brady | |
| Ectopic beats | Detection agreement (F1) | Important for HRV validity | |
| ST analysis | ST event agreement | If target use case requires it | |
| Signal quality | SQI classifier agreement | Codec should not mask poor signal | |

---

## PPG validation

### HR / Pulse-rate preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| Timing | Pulse peak timing error (ms) | Per pulse and per window | |
| Timing | Pulse onset timing error (ms) | Relevant for PAT/PTT | |
| Heart rate | Pulse-rate error (bpm) | Compare to raw-derived PR | |
| PRV | Inter-pulse interval error (ms) | Analogous to R-R error | |
| PRV | RMSSD / SDNN delta | Window-length dependent | |
| Frequency | Band-power error (%) | Include respiration-related bands | |
| Signal quality | PPG SQI agreement | Codec should not make bad PPG look good | |
| Boundary | False/missed pulse peaks near chunk edges | Report separately | |

### Morphology preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| Waveform | PRD / PRDN (%) | Not sufficient alone | |
| Waveform | MAE / RMSE | Report by activity / SQI bucket | |
| AC component | AC amplitude error (%) | Critical for pulse morphology | |
| DC component | DC / baseline error | Critical for optical interpretation | |
| AC/DC | AC/DC ratio delta (%) | Important for SpO2-related use | |
| Pulse shape | Systolic peak amplitude delta | Normalize by baseline | |
| Pulse shape | Systolic upstroke slope delta (%) | Useful for vascular features | |
| Pulse shape | Dicrotic notch preservation | Detection agreement | |
| Low-frequency | Baseline trend error | Profile-specific | |
| Boundary | Baseline step at chunk boundary | Important for stitching | |

### SpO2-sensitive preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| Red channel | Red AC/DC error (%) | If red channel available | |
| IR channel | IR AC/DC error (%) | If IR channel available | |
| Ratio-of-ratios | Red/IR AC/DC ratio delta (%) | Core SpO2-sensitive metric | |
| Downstream | SpO2 algorithm output delta | Use reference algorithm | |
| Low perfusion | Performance in low-perfusion segments | Report separately | |
| Motion | Performance under motion artifact | Report separately | |

### Downstream task preservation

| Category | Metric | Comparison | Notes |
|----------|--------|------------|-------|
| HR estimation | HR estimator output delta (bpm) | Same estimator on both | |
| SpO2 | SpO2 output delta (points) | If applicable | |
| Respiratory rate | RR estimator delta (breaths/min) | If respiratory modulation is used | |
| Motion artifact | Artifact detector agreement (F1) | Important for wearables | |
| Sleep features | Sleep feature delta | If required by use case | |

---

## Cross-cutting validation

### Stitching and streaming behavior

| Category | Metric | Notes |
|----------|--------|-------|
| Value discontinuity | Amplitude step at chunk boundary | ECG and PPG |
| Slope discontinuity | First-derivative step | Detect artificial edges |
| Baseline continuity | Baseline step across chunks | Especially important for PPG |
| Peak integrity | False peaks near boundaries | ECG R-peaks and PPG pulses |
| Missed events | Missed peaks near boundaries | Report first/last N ms |
| HR/HRV impact | Boundary-induced HR/HRV error | Boundary vs non-boundary windows |
| Spectral | Boundary spectral leakage | Important for HRV/respiration |

### Noise and artifact behavior

| Category | Metric | Notes |
|----------|--------|-------|
| Additive noise | Feature preservation under injected noise | |
| Motion artifact | Performance by motion/SQI bucket | PPG especially |
| Baseline wander | Preservation/removal behavior | Profile-specific |
| Muscle/electrode noise | ECG feature preservation | |
| Low SNR | Performance at low SNR | Report degradation curve |
| SQI honesty | SQI before/after compression agreement | Codec should not falsely improve SQI |
| Rare morphology | Preservation of abnormal/rare events | Avoid smoothing away anomalies |

### Adaptive bitrate and fallback

| Category | Metric | Notes |
|----------|--------|-------|
| Bitrate | Average and p95 bitrate (bps) | By modality/profile |
| Complexity response | Bitrate vs signal complexity | Should spend bits on hard segments |
| Risk response | Bitrate vs clinical risk | Should spend bits on risky segments |
| Fallback | Residual/raw fallback rate (%) | By bucket |
| Fallback quality | Post-fallback metric recovery | Demonstrates safety mechanism |
| Failure mode | Windows exceeding preservation bounds | Must be explicitly reported |

### Versioning and compatibility

| Category | Metric | Notes |
|----------|--------|-------|
| Codec ID | Stream includes codec family ID | Prevent silent incompatibility |
| Encoder version | Stream includes encoder version | Required for auditability |
| Decoder version | Stream includes decoder version/hash | Decoder must be identifiable |
| Codebook version | RVQ codebook version/hash | Codebook changes are breaking |
| Entropy model | Entropy model version included | Especially with CNN prior |
| Profile ID | Compression profile included | HRV vs morphology vs SpO2 |
| Golden vectors | Known streams decode to expected outputs | Regression/conformance testing |
| Backward compat | Old streams decode with old decoder | Avoid silent reinterpretation |

---

## Signal condition buckets

Metrics should be reported per-bucket, not only in aggregate.

### ECG buckets

- Clean sinus rhythm
- Low-amplitude ECG
- Baseline wander
- Muscle artifact
- Electrode motion
- Premature atrial contractions (PAC)
- Premature ventricular contractions (PVC)
- Atrial fibrillation
- Bradycardia / tachycardia
- QRS widening
- ST elevation / depression
- T-wave abnormality
- Noisy but clinically interpretable

### PPG buckets

- Clean resting PPG
- Low perfusion
- High motion
- Loose sensor fit
- Ambient light contamination
- Exercise transition
- Sleep
- Cold / vasoconstriction
- Arrhythmia-affected pulse intervals
- SpO2 desaturation (if available)
- Noisy but physiologically interpretable

---

## Positioning guidance

Avoid:

> "The model learns what is important."

Prefer:

> "The codec is constrained and validated to preserve specific physiological observables — timing, morphology, spectral content, baseline behavior, AC/DC structure, and downstream algorithm outputs. When preservation confidence is low, the system can increase bitrate or transmit fallback residuals."

---

## Pass/fail summary (per modality and profile)

For each codec version, report:

1. Average performance
2. p95 performance
3. Worst-case bounded error
4. Performance by signal-quality bucket
5. Performance by pathology/artifact bucket
6. Boundary-specific performance
7. Downstream task agreement
8. Fallback/escalation rate
9. Version compatibility status

A codec version is releasable for a profile only if it passes all required preservation bounds for that profile.
