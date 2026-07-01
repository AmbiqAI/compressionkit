# PPG CR vs. Fidelity

This page summarizes how compression ratio (CR) trades off against signal- and physiology-level fidelity for the v1 RVQ goldens. The **Effective CR** column is populated when an entropy-prior run is available; otherwise, the codec CR is the release-facing operating point. CRs are reported alongside the [noise-aware metrics](../experiments/index.md) so it is easy to inspect the difference between waveform faithfulness, clean-reference fidelity, and physiological metric preservation.

## Headline summary (all samples)

| CR | Codec CR | Effective CR | bits/tok | N | Faithful PRD% | Truth PRD% (clean) | PRDN-noise% | HR MAE (bpm) | Pulse-band PSD err | Coherence | Seam ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 02x | 2.00 | — | — | 9788 | 2.59 | 0.62 | 0.15 | 0.19 | 0.0155 | 0.9928 | — |
| 04x | 4.00 | — | — | 9789 | 3.44 | 1.14 | 0.23 | 0.29 | 0.5885 | 0.9861 | — |
| 08x | 8.00 | — | — | 9789 | 5.10 | 2.66 | 0.64 | 0.49 | 0.6623 | 0.9565 | — |
| 16x | 16.00 | — | — | 9789 | 7.54 | 5.75 | 0.79 | 0.88 | 12.5676 | 0.8991 | — |
| 32x | 32.00 | — | — | 6336 | 12.08 | 12.35 | 0.20 | 1.79 | 0.0356 | 0.8014 | — |

## Noise-stratified detail (clean / median / noisy tertiles)

Tertiles are formed from a band-power noise estimate over each input recording; *clean* is the lowest-noise third, *noisy* is the highest.

| CR | Tertile | N | PRD% | PRDN-noise% | HR MAE | Pulse-band PSD err | Coherence |
|---|---|---|---|---|---|---|---|
| 02x | clean | 3263 | 4.98 | 0.46 | 0.22 | — | 0.9862 |
| 02x | median | 3262 | 1.37 | 0.00 | 0.14 | — | 0.9963 |
| 02x | noisy | 3263 | 1.42 | 0.00 | 0.22 | — | 0.9959 |
| 04x | clean | 3263 | 6.34 | 0.69 | 0.33 | — | 0.9784 |
| 04x | median | 3263 | 1.99 | 0.00 | 0.24 | — | 0.9906 |
| 04x | noisy | 3263 | 1.98 | 0.00 | 0.30 | — | 0.9894 |
| 08x | clean | 3263 | 8.60 | 1.92 | 0.52 | — | 0.9449 |
| 08x | median | 3263 | 3.18 | 0.00 | 0.47 | — | 0.9655 |
| 08x | noisy | 3263 | 3.51 | 0.00 | 0.49 | — | 0.9592 |
| 16x | clean | 3263 | 10.36 | 2.32 | 0.75 | — | 0.8928 |
| 16x | median | 3263 | 5.77 | 0.01 | 0.86 | — | 0.9123 |
| 16x | noisy | 3263 | 6.49 | 0.04 | 1.04 | — | 0.8922 |
| 32x | clean | 2112 | 11.17 | 0.59 | 1.54 | — | 0.8367 |
| 32x | median | 2112 | 12.16 | 0.01 | 1.72 | — | 0.8015 |
| 32x | noisy | 2112 | 12.90 | 0.00 | 2.11 | — | 0.7660 |

## How to read this table

- **Faithful PRD%** is PRD against the recorded (still-noisy) input. It rises with CR   by design; the codec is allocating bits to the *physiological* bands, not to broadband   noise. Read it together with Truth PRD% — never on its own.
- **Truth PRD% (clean)** is PRD against the clean ground-truth reference (robustness   fixture). This is the fair fidelity number for denoising lanes, which a faithfulness-only   view would unfairly penalize.
- **PRDN-noise%** stays low across CRs, evidencing that the codec is removing noise   rather than corrupting clean signal — this is the headline customer claim.
- **HR MAE** and the band-power error track physiological fidelity directly; both   stay well within clinical tolerance at the recommended operating CRs.
- **Seam ratio** (when available) reports the long-recording stitching seam energy   relative to the centre window — values near 1.0 indicate seamless continuous   reconstruction.

The noise-stratified detail section above complements these all-sample numbers with the clean/median/noisy regime breakdown, so both the clean-truth and noise-regime surfaces are always presented together.
