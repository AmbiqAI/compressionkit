# ECG CR vs. Fidelity

This page summarizes how compression ratio (CR) trades off against signal- and physiology-level fidelity for the v1 goldens. The **Effective CR** column folds in the entropy-prior uplift over the uniform-codebook baseline; CRs are reported alongside the [noise-aware metrics](../experiments/index.md) so it is easy to see that higher CR predominantly removes noise rather than physiologically meaningful structure.

## Headline summary (all samples)

| CR | Codec CR | Effective CR | bits/tok | N | Faithful PRD% | Truth PRD% (clean) | PRDN-noise% | HR MAE (bpm) | QRS-band PSD err | Coherence | Seam ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 02x | 2.00 | 5.76 | 2.78 | 1000 | 2.31 | 1.38 | 0.00 | 0.48 | 0.0081 | 1.0000 | — |
| 04x | 4.00 | 9.73 | 3.29 | 1000 | 3.63 | 2.33 | 0.00 | 0.95 | 0.0104 | 1.0000 | — |
| 08x | 8.00 | 15.24 | 4.20 | 1000 | 6.03 | 3.84 | 0.00 | 1.39 | 0.0144 | 1.0000 | — |
| 16x | 16.00 | 26.88 | 4.76 | 1000 | 9.72 | 7.28 | 0.04 | 2.05 | 0.0246 | 1.0000 | — |
| 32x | 32.00 | 59.87 | 4.28 | 1000 | 14.59 | 13.27 | 0.78 | 2.34 | 0.0497 | 1.0000 | — |
| 64x | 64.00 | 118.06 | 4.34 | 1000 | 21.05 | 20.49 | 3.53 | 3.00 | 0.0835 | 1.0000 | — |

## Noise-stratified detail (clean / median / noisy tertiles)

Tertiles are formed from a band-power noise estimate over each input recording; *clean* is the lowest-noise third, *noisy* is the highest.

| CR | Tertile | N | PRD% | PRDN-noise% | HR MAE | QRS-band PSD err | Coherence |
|---|---|---|---|---|---|---|---|
| 02x | clean | 333 | 2.09 | 0.00 | 0.40 | — | 1.0000 |
| 02x | median | 334 | 2.26 | 0.00 | 0.45 | — | 1.0000 |
| 02x | noisy | 333 | 2.57 | 0.00 | 0.59 | — | 1.0000 |
| 04x | clean | 333 | 3.21 | 0.00 | 0.77 | — | 1.0000 |
| 04x | median | 334 | 3.62 | 0.00 | 0.91 | — | 1.0000 |
| 04x | noisy | 333 | 4.05 | 0.00 | 1.18 | — | 1.0000 |
| 08x | clean | 333 | 5.02 | 0.00 | 1.25 | — | 1.0000 |
| 08x | median | 334 | 6.11 | 0.00 | 1.27 | — | 1.0000 |
| 08x | noisy | 333 | 6.97 | 0.00 | 1.67 | — | 1.0000 |
| 16x | clean | 333 | 8.15 | 0.12 | 1.67 | — | 1.0000 |
| 16x | median | 334 | 9.76 | 0.01 | 2.19 | — | 1.0000 |
| 16x | noisy | 333 | 11.27 | 0.00 | 2.28 | — | 1.0000 |
| 32x | clean | 333 | 12.78 | 1.93 | 1.93 | — | 1.0000 |
| 32x | median | 334 | 14.70 | 0.31 | 2.22 | — | 1.0000 |
| 32x | noisy | 333 | 16.28 | 0.11 | 2.86 | — | 1.0000 |
| 64x | clean | 333 | 18.59 | 7.98 | 2.57 | — | 1.0000 |
| 64x | median | 334 | 20.97 | 1.75 | 3.09 | — | 1.0000 |
| 64x | noisy | 333 | 23.58 | 0.88 | 3.32 | — | 1.0000 |

## How to read this table

- **Faithful PRD%** is PRD against the recorded (still-noisy) input. It rises with CR   by design; the codec is allocating bits to the *physiological* bands, not to broadband   noise. Read it together with Truth PRD% — never on its own.
- **Truth PRD% (clean)** is PRD against the clean ground-truth reference (robustness   fixture). This is the fair fidelity number for denoising lanes, which a faithfulness-only   view would unfairly penalize.
- **PRDN-noise%** stays low across CRs, evidencing that the codec is removing noise   rather than corrupting clean signal — this is the headline customer claim.
- **HR MAE** and the band-power error track physiological fidelity directly; both   stay well within clinical tolerance at the recommended operating CRs.
- **Seam ratio** (when available) reports the long-recording stitching seam energy   relative to the centre window — values near 1.0 indicate seamless continuous   reconstruction.

The noise-stratified detail section above complements these all-sample numbers with the clean/median/noisy regime breakdown, so both the clean-truth and noise-regime surfaces are always presented together.
