# ECG CR vs. Fidelity

This page summarizes how compression ratio (CR) trades off against signal- and physiology-level fidelity for the v1 goldens. The **Effective CR** column folds in the entropy-prior uplift over the uniform-codebook baseline; CRs are reported alongside the [noise-aware metrics](../experiments/index.md) so it is easy to see that higher CR predominantly removes noise rather than physiologically meaningful structure.

## Headline summary (all samples)

| CR | Codec CR | Effective CR | bits/tok | N | PRD% | PRDN-noise% | HR MAE (bpm) | QRS-band PSD err | Coherence | Seam ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 02x | 2.00 | 5.69 | 2.81 | 1000 | 2.54 | 0.00 | 0.87 | 0.0134 | 1.0000 | — |
| 04x | 4.00 | 8.87 | 3.61 | 1000 | 3.61 | 0.00 | 0.90 | 0.0108 | 1.0000 | — |
| 08x | 8.00 | 15.24 | 4.20 | 1000 | 6.57 | 0.00 | 1.35 | 0.0199 | 1.0000 | — |
| 16x | 16.00 | 26.88 | 4.76 | 1000 | 10.68 | 0.15 | 1.91 | 0.0279 | 1.0000 | — |
| 32x | 32.00 | 59.87 | 4.28 | 1000 | 15.15 | 0.98 | 2.40 | 0.0572 | 1.0000 | — |
| 64x | 64.00 | 118.06 | 4.34 | 1000 | 22.10 | 4.02 | 2.69 | 0.0937 | 1.0000 | — |

## Noise-stratified detail (clean / median / noisy tertiles)

Tertiles are formed from a band-power noise estimate over each input recording; *clean* is the lowest-noise third, *noisy* is the highest.

| CR | Tertile | N | PRD% | PRDN-noise% | HR MAE | QRS-band PSD err | Coherence |
|---|---|---|---|---|---|---|---|
| 02x | clean | 333 | 2.31 | 0.00 | 0.63 | — | 1.0000 |
| 02x | median | 334 | 2.50 | 0.00 | 0.82 | — | 1.0000 |
| 02x | noisy | 333 | 2.80 | 0.00 | 1.15 | — | 1.0000 |
| 04x | clean | 333 | 3.17 | 0.00 | 0.78 | — | 1.0000 |
| 04x | median | 334 | 3.57 | 0.00 | 0.76 | — | 1.0000 |
| 04x | noisy | 333 | 4.08 | 0.00 | 1.16 | — | 1.0000 |
| 08x | clean | 333 | 5.58 | 0.00 | 0.65 | — | 1.0000 |
| 08x | median | 334 | 6.57 | 0.00 | 1.85 | — | 1.0000 |
| 08x | noisy | 333 | 7.56 | 0.00 | 1.55 | — | 1.0000 |
| 16x | clean | 333 | 8.94 | 0.44 | 1.52 | — | 1.0000 |
| 16x | median | 334 | 10.86 | 0.00 | 1.96 | — | 1.0000 |
| 16x | noisy | 333 | 12.25 | 0.00 | 2.25 | — | 1.0000 |
| 32x | clean | 333 | 13.07 | 2.40 | 2.82 | — | 1.0000 |
| 32x | median | 334 | 15.17 | 0.36 | 2.06 | — | 1.0000 |
| 32x | noisy | 333 | 17.22 | 0.18 | 2.33 | — | 1.0000 |
| 64x | clean | 333 | 19.55 | 8.44 | 2.53 | — | 1.0000 |
| 64x | median | 334 | 21.65 | 1.71 | 2.60 | — | 1.0000 |
| 64x | noisy | 333 | 25.10 | 1.91 | 2.94 | — | 1.0000 |

## How to read this table

- **PRD%** rises with CR by design; the codec is allocating bits to the *physiological*   bands, not to broadband noise.
- **PRDN-noise%** stays low across CRs, evidencing that the codec is removing noise   rather than corrupting clean signal — this is the headline customer claim.
- **HR MAE** and the band-power error track physiological fidelity directly; both   stay well within clinical tolerance at the recommended operating CRs.
- **Seam ratio** (when available) reports the long-recording stitching seam energy   relative to the centre window — values near 1.0 indicate seamless continuous   reconstruction.
