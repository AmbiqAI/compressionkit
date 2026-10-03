---
title: "Validation scorecard"
description: "Understand waveform, physiological and deployment measurements when comparing codecs."
---

A scorecard records how a reconstructed signal differs from its reference. Read it together with the dataset, sample count, preprocessing, and codec configuration. Missing measurements do not count as passing results.

## Read the metrics

| Metric | What it tells you | How to interpret it |
| --- | --- | --- |
| Compression ratio (CR) | Raw payload divided by encoded payload | State the original sample precision and whether framing overhead is included. |
| Effective CR | Payload reduction after entropy coding | Use measured bitstream size; a prior is not a fixed improvement across signals. |
| PRD | Percentage root-mean-square difference from a reference | Lower is better for a fixed reference and evaluation. |
| Faithful PRD | Error against the recorded input | Includes differences in the input's noise and artifacts. |
| Truth PRD | Error against the clean reference used by a fixture | Depends on how the reference was obtained; inspect the fixture. |
| PRDN-noise | Noise-normalized distortion | Read with the reference definition and other error metrics. A low value alone does not prove useful denoising. |
| MSE | Mean squared sample error | Sensitive to signal scale and normalization. |
| Cosine similarity | Similarity of waveform direction | Does not establish amplitude or downstream task agreement. |
| HR MAE | Mean absolute heart-rate difference, in bpm | Check the algorithm, window duration, and which frames were scored. |
| SDNN / RMSSD error | Difference in interval-variability measurements | Sensitive to detected peaks and recording duration. |
| Band error / coherence | Frequency-domain agreement | Compare using the same frequency bands and estimator. |
| Seam ratio | Boundary energy relative to window-center energy | Inspect long-recording reconstructions as well as the aggregate. |

<span id="agreement-vs-accuracy"></span>

<span id="algorithm-repeatability-baseline"></span>

## Agreement is different from accuracy

Applying the same algorithm to original and reconstructed signals measures agreement. It does not establish that either result is accurate against independently labeled ground truth.

A codec may preserve noise faithfully or suppress some of it. Compare error against the recorded input and, when available, a separate clean reference. Report how that reference was constructed.

<span id="core-claim"></span>

<span id="evidence-layers"></span>

<span id="measurement-philosophy"></span>

<span id="implementation-tiers"></span>

<span id="positioning-guidance"></span>

## Compare like with like

Record the following alongside each comparison:

- Dataset and split, number of frames, and selection rules.
- Sample rate, frame duration, channel selection, and normalization.
- Codec version, compression ratio, and any entropy model.
- Reference signal and metric definitions.
- Noise or artifact conditions and excluded/invalid windows.
- Reconstruction overlap, stitching, and boundary handling.

The [customer evidence](/compressionkit/customer-evidence/) and model pages preserve different evaluation views. Their numbers should not be combined as if they came from one run.

<span id="hr-hrv-preservation"></span>

<span id="morphology-preservation"></span>

<span id="downstream-task-preservation"></span>

## ECG validation

Start with waveform error and R-peak or heart-rate agreement. If your application uses morphology or rhythm features, evaluate those specific outputs on representative recordings as well. Inspect chunk boundaries and difficult signal segments separately.

The [ECG model page](/compressionkit/models/ecg/) and [ECG noise-aware tables](/compressionkit/methods/cr_vs_fidelity_ecg/) show the recorded measurements. They do not establish coverage for every downstream task.

<span id="hr-pulse-rate-preservation"></span>

<span id="morphology-preservation_1"></span>

<span id="spo2-sensitive-preservation"></span>

<span id="downstream-task-preservation_1"></span>

## PPG validation

Inspect pulse timing, heart-rate agreement, waveform error, and behavior under motion or baseline drift. For interval variability, use sufficiently long recordings and report the reconstruction and peak-detection procedure.

The [PPG model page](/compressionkit/models/ppg/) and [PPG noise-aware tables](/compressionkit/methods/cr_vs_fidelity_ppg/) show the recorded measurements. Single-channel reconstruction results do not establish multi-channel optical measurement performance.

<span id="noise-preservation-vs-noise-removal"></span>

<span id="noise-and-artifact-behavior"></span>

<span id="ecg-buckets"></span>

<span id="ppg-buckets"></span>

## Signal condition buckets

Separate clean and difficult recordings rather than relying only on an overall average. Useful groups include low signal amplitude, motion, baseline drift, and boundary-adjacent samples. Use labels and thresholds that match the dataset and publish the number of samples in each group.

<span id="cross-cutting-validation"></span>

<span id="stitching-and-streaming-behavior"></span>

<span id="adaptive-bitrate-and-fallback"></span>

<span id="versioning-and-compatibility"></span>

<span id="passfail-summary-per-modality-and-profile"></span>

## Package and runtime validation

A quality score does not verify that a package loads correctly or that another runtime reproduces its output. Check manifests, file integrity, reference vectors, and encoder/decoder compatibility using the [deployment workflow](/compressionkit/deployment/#recommended-validation).

Measure memory and latency in the intended environment. Define acceptance thresholds for the application before evaluating a candidate.
