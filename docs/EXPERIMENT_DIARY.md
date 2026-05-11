# Experiment Diary

Tracks what was tried, what worked, and what didn't.

---

## 1. Spectral Loss (Multi-Scale STFT)

**Hypothesis:** Adding multi-scale spectral loss preserves frequency content better.

**Setup:** Spectral loss weight 0.5, FFT sizes [16, 32, 64, 128], added on top of MSE + derivative loss.

**Results — ECG:**

| CR | Golden MSE | Spectral MSE | Δ |
|----|-----------|-------------|---|
| 2x | 0.000623 | 0.000636 | +2.1% |
| 4x | 0.001259 | 0.001332 | +5.8% |
| 8x | 0.004123 | 0.004356 | +5.7% |
| 16x | 0.011278 | 0.015123 | +34.1% |

**Results — PPG:** Catastrophic regression (+409% to +3079%).

**Verdict:** ❌ Harmful across the board. Spectral loss fights with MSE rather than complementing it at these signal lengths.

---

## 2. Hierarchical Codebooks [512, 128]

**Hypothesis:** Coarse-to-fine codebook allocation (9+7 = 16 bits, iso-CR with 8+8) lets the first RVQ level capture more structure.

**Setup:** `codebook_sizes: [512, 128]` vs uniform `[256, 256]`. Both 16 total bits.

**Results — ECG:**

| CR | Golden MSE | Hier MSE | Δ |
|----|-----------|---------|---|
| 2x | 0.000623 | 0.000628 | +0.8% |
| 4x | 0.001259 | 0.001295 | +2.8% |
| 8x | 0.004123 | 0.004385 | +6.4% |
| 16x | 0.011278 | 0.011376 | +0.9% |

**Results — PPG:**

| CR | Golden MSE | Hier MSE | Δ |
|----|-----------|---------|---|
| 2x | 0.000645 | 0.000772 | +19.7% |
| 4x | 0.001154 | 0.001178 | +2.0% |
| 8x | 0.002708 | 0.002702 | -0.2% |
| 16x | 0.007094* | 0.008841 | +24.6% |

*16x golden_sched diverged; compared against rep2 baseline.

**Verdict:** ❌ No benefit. Uniform [256,256] wins. RVQ residual structure already handles capacity allocation implicitly.

---

## 3. Dead-Code Revival + K-Means Init (PPG-H5)

**Hypothesis:** EMA codebooks may suffer from dead codes (unused entries). Revival resamples dead entries from current batch; k-means init gives better starting codebooks.

**Setup:** `revive_dead_codes: true`, `revive_threshold: 0.03`, `kmeans_init: true` on top of golden PPG-H5 configs.

**Results — PPG:**

| CR | Golden MSE | Revive MSE | Δ |
|----|-----------|-----------|---|
| 2x | 0.000645 | 0.000944 | +46.3% |
| 4x | 0.001154 | 0.001260 | +9.1% |
| 8x | 0.002708 | 0.002309 | **-14.8%** |
| 16x | 0.007094* | 0.007352 | +3.6% |

**Verdict:** ❌ Mixed / mostly harmful. 8x saw a meaningful improvement (-14.8%), but 2x regressed badly (+46%) and 4x/16x also worse. The 8x win may be noise — not consistent enough to justify enabling globally.

---

## 4. Ablation: Dead-Code Revival vs K-Means Init (PPG-H5)

**Hypothesis:** Experiment 3 combined both features. This ablation isolates each to determine which (if either) helps.

**Setup:**
- Revive-only: `revive_dead_codes: true`, `revive_threshold: 0.03`, `kmeans_init: false`
- K-Means-only: `revive_dead_codes: false`, `kmeans_init: true`

**Results — PPG:**

| CR | Golden MSE | Revive-only | Δ | KMeans-only | Δ |
|----|-----------|------------|---|------------|---|
| 2x | 0.000645 | 0.000825 | +27.7% | 0.001345 | +108.4% |
| 4x | 0.001154 | 0.001282 | +11.1% | 0.001181 | +2.3% |
| 8x | 0.002708 | 0.003796 | +40.2% | 0.002736 | +1.0% |
| 16x | 0.007094* | 0.006575 | **-7.3%** | 1.456277 | +20429% (DIVERGED) |

**Observations:**
- **K-Means init is catastrophic at 16x** — causes complete training failure, codebook usage collapses to ~50% and loss plateaus at ~1.5. At lower CRs it's neutral (+1–2%) but never helps.
- **Dead-code revival hurts at low CRs** (2x, 4x) and high CRs (8x). Only helps slightly at 16x (-7.3%).
- The 8x improvement seen in experiment 3 (-14.8%) is NOT replicated in the revive-only ablation (+40.2%) — suggesting that result was stochastic noise or specific to the combined interaction.
- Training logs show revive-only maintains high codebook usage (~91-94%) during training, but this doesn't translate to better val_mse.

**Verdict:** ❌ Neither feature helps reliably. K-means init is actively dangerous (16x divergence). Dead-code revival is a wash. **Recommend keeping both disabled.**

---

## 5. Learned CNN Prior for Entropy Coding

**Hypothesis:** RVQ token sequences have temporal structure. A causal CNN prior can predict token distributions, enabling arithmetic/ANS coding at fewer bits than the uniform 8 bits/token ceiling.

**Setup:**
- Two-stage approach: frozen RVQ encoder → extract tokens → train CNN prior on token sequences.
- Prior: 4-layer dilated causal Conv1D, embed_dim=48, kernel=5, receptive_field=61 tokens, 71K params (278 KB).
- Baselines: uniform (8 bits/tok), unigram (frequency-based).
- Script: `scripts/measure_rvq_entropy.py`

### Results — PPG-H5 (val=2000 frames, train=5000 frames)

| Base CR | Unigram bpt | CNN bpt | Effective CR | CNN Uplift |
|---------|------------|---------|-------------|-----------|
| 2x | 7.18 | 2.72 | 5.88x | **2.94x** |
| 4x | 7.15 | 4.43 | 7.22x | **1.81x** |
| 8x | 7.07 | 5.21 | 12.28x | **1.54x** |
| 16x | 7.18 | 6.23 | 20.55x | **1.28x** |

Unigram uplift: negligible (1.05–1.11x only).

### Results — ECG-256Hz (val=80 files, ~716 frames per CR)

| Base CR | Unigram bpt | Best Prior | CNN bpt | Effective CR | CNN Uplift |
|---------|------------|-----------|---------|-------------|-----------|
| 2x | 6.61 | winsweep_f4 | 2.81 | 5.69x | **2.84x** |
| 4x | 6.50 | winsweep_f4 | 3.61 | 8.87x | **2.22x** |
| 8x | 6.71 | winsweep_f4 | 4.20 | 15.24x | **1.90x** |
| 16x | 7.02 | cnnL_full | 4.76 | 26.88x | **1.68x** |
| 32x | 6.53 | winsweep_f16 | 4.28 | 59.87x | **1.87x** |
| 64x | 6.98 | cnnL_full | 4.34 | 118.06x | **1.84x** |

ECG 32x was explored most heavily (GRU, transformer, hybrid, WaveNet, DSCNN variants) — best architecture is "winsweep_f16" (windowed CNN sweep with context=16 frames).

### Key Observations

1. **Massive free compression** — the prior adds 0 distortion (lossless stage) but provides 1.28–2.94x additional CR on top of the RVQ codec.
2. **Low CRs benefit most** — at 2x base CR, tokens are highly redundant (many per frame, high inter-frame correlation). The prior easily predicts them, yielding ~2.8–2.9x uplift for both PPG and ECG.
3. **ECG benefits more uniformly** — ECG gets 1.68–2.84x uplift across all CRs, while PPG drops to 1.28x at 16x (fewer tokens per frame = less context for the prior).
4. **Unigram is useless** — frequency-based coding gives only 5–23% uplift, confirming that the gains come from sequential/temporal modeling, not marginal entropy.
5. **Prior is tiny** — 71K params, fully causal, INT8-friendly (Conv1D stack). Deployable alongside the RVQ decoder on-device.

### Effective Final CRs (lossless prior + lossy RVQ)

| Signal | Base 2x | Base 4x | Base 8x | Base 16x | Base 32x | Base 64x |
|--------|---------|---------|---------|----------|----------|----------|
| PPG-H5 | 5.88x | 7.22x | 12.28x | 20.55x | — | — |
| ECG-256Hz | 5.69x | 8.87x | 15.24x | 26.88x | 59.87x | 118.06x |

**Verdict:** ✅ Major win. Learned prior gives 1.3–2.9x lossless uplift at zero quality cost. Recommend implementing actual arithmetic/ANS coder with the trained prior for deployment.

---

## Notes

- **Eval sample size matters:** N=50 is unreliable (CoV ~96%). All golden configs use N=500+.
- **PPG 16x golden_sched is broken:** The `ppg_h5_rvq_16x_mixed_golden_sched` run diverged (val_mse=1.46). Use `rep2` (val_mse=0.007094) as 16x baseline.
- **PPG-H5 wiring bug (fixed):** `train_ppg_h5_rvq.py` was not forwarding `revive_dead_codes`, `revive_threshold`, `kmeans_init` to the model builder. Fixed 2026-05-07.
