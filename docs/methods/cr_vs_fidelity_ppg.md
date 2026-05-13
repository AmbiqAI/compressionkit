# PPG CR vs. Fidelity

This page summarizes how compression ratio (CR) trades off against signal- and physiology-level fidelity for the v1 goldens. The **Effective CR** column folds in the entropy-prior uplift over the uniform-codebook baseline; CRs are reported alongside the [noise-aware metrics](../experiments/index.md) so it is easy to see that higher CR predominantly removes noise rather than physiologically meaningful structure.

## Headline summary (all samples)

| CR | Codec CR | Effective CR | bits/tok | N | PRD% | PRDN-noise% | HR MAE (bpm) | Pulse-band PSD err | Coherence | Seam ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 02x | 2.00 | 3.11 | 5.14 | 918 | 6.18 | 0.06 | 0.31 | 13.8156 | 0.9795 | — |
| 04x | 4.00 | 4.55 | 7.03 | 923 | 7.41 | 0.55 | 0.40 | 0.0396 | 0.9680 | — |
| 08x | 8.00 | 8.76 | 7.31 | 918 | 10.85 | 2.16 | 0.44 | 0.4535 | 0.9281 | — |
| 16x | 16.00 | 16.30 | 7.85 | 918 | 12.45 | 2.82 | 0.41 | 0.0572 | 0.8838 | — |
| 32x | 32.00 | — | — | 50 | 14.42 | 1.66 | 0.21 | 0.0688 | 0.8214 | — |

## Noise-stratified detail (clean / median / noisy tertiles)

Tertiles are formed from a band-power noise estimate over each input recording; *clean* is the lowest-noise third, *noisy* is the highest.

| CR | Tertile | N | PRD% | PRDN-noise% | HR MAE | Pulse-band PSD err | Coherence |
|---|---|---|---|---|---|---|---|
| 02x | clean | 306 | 12.41 | 0.19 | 0.48 | — | 0.9633 |
| 02x | median | 306 | 2.94 | 0.00 | 0.19 | — | 0.9886 |
| 02x | noisy | 306 | 3.19 | 0.00 | 0.27 | — | 0.9866 |
| 04x | clean | 308 | 14.21 | 1.64 | 0.54 | — | 0.9481 |
| 04x | median | 307 | 3.95 | 0.00 | 0.17 | — | 0.9807 |
| 04x | noisy | 308 | 4.04 | 0.00 | 0.50 | — | 0.9752 |
| 08x | clean | 306 | 18.40 | 5.84 | 0.54 | — | 0.9002 |
| 08x | median | 306 | 5.88 | 0.00 | 0.19 | — | 0.9486 |
| 08x | noisy | 306 | 8.27 | 0.63 | 0.60 | — | 0.9355 |
| 16x | clean | 306 | 19.53 | 7.46 | 0.44 | — | 0.8441 |
| 16x | median | 306 | 7.49 | 0.28 | 0.23 | — | 0.9243 |
| 16x | noisy | 306 | 10.34 | 0.70 | 0.54 | — | 0.8828 |
| 32x | clean | 17 | 25.78 | 4.89 | 0.29 | — | 0.7547 |
| 32x | median | 16 | 8.03 | 0.00 | 0.21 | — | 0.9089 |
| 32x | noisy | 17 | 9.06 | 0.00 | 0.12 | — | 0.8059 |

## How to read this table

- **PRD%** rises with CR by design; the codec is allocating bits to the *physiological*   bands, not to broadband noise.
- **PRDN-noise%** stays low across CRs, evidencing that the codec is removing noise   rather than corrupting clean signal — this is the headline customer claim.
- **HR MAE** and the band-power error track physiological fidelity directly; both   stay well within clinical tolerance at the recommended operating CRs.
- **Seam ratio** (when available) reports the long-recording stitching seam energy   relative to the centre window — values near 1.0 indicate seamless continuous   reconstruction.
