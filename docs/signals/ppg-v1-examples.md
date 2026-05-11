---
icon: lucide/sliders-horizontal
---

# PPG 2x-16x Examples

compressionKIT currently ships a set of four reference PPG RVQ operating points that make the compression-versus-fidelity tradeoff easy to evaluate. They share the same overall training flow and differ mainly in how aggressively the signal is compressed.

## At A Glance

| Variant | Archived v1 PRD | Retest PRD | Archived v1 MSE | Retest MSE | Best Use |
|--------|------------------|-----------:|----------------:|-----------:|---------|
| 2× | 4.76% | 4.18% | 0.001688 | 0.001303 | Highest reconstruction fidelity |
| 4× | 6.18% | 5.38% | 0.002850 | 0.002151 | Balanced deployment point |
| 8× | 8.67% | 7.83% | 0.005595 | 0.004556 | Strong savings with preserved morphology |
| 16× | 11.42% | 10.43% | 0.009705 | 0.008099 | Maximum reduction for constrained systems |

The retest values above were regenerated from the current modular flow and improved on the archived v1 golden baselines across all four operating points.

## 2× Example

The 2× model is the most conservative option. It is the right starting point when signal fidelity is the first priority and compression is a secondary constraint.

- Compression ratio: 2×
- Archived v1 PRD: 4.76%
- Current retest PRD: 4.18%
- Archived v1 MSE: 0.001688
- Current retest MSE: 0.001303

This variant is useful when teams want the reconstructed waveform to remain as close as possible to the source signal while still reducing storage and transfer cost.

## 4× Example

The 4× model is a strong middle ground and often the best first deployment candidate when the product needs both quality and efficiency.

- Compression ratio: 4×
- Archived v1 PRD: 6.18%
- Current retest PRD: 5.38%
- Archived v1 MSE: 0.002850
- Current retest MSE: 0.002151

This is the operating point to evaluate when a program needs visible savings without pushing reconstruction quality to the edge.

## 8× Example

The 8× model is a more aggressive compression point that still retains the dominant PPG structure well enough for interactive comparison and product evaluation.

- Compression ratio: 8×
- Archived v1 PRD: 8.67%
- Current retest PRD: 7.83%
- Archived v1 MSE: 0.005595
- Current retest MSE: 0.004556

For many teams, 8× is where the system-level benefit starts to become immediately tangible without losing the core waveform story.

## 16× Example

The 16× model is the most aggressive reference point in the documented v1 set. It is aimed at scenarios where bandwidth, storage, or radio energy dominate the design conversation.

- Compression ratio: 16×
- Archived v1 PRD: 11.42%
- Current retest PRD: 10.43%
- Archived v1 MSE: 0.009705
- Current retest MSE: 0.008099

This operating point is useful when the question is not whether compression helps, but how far it can be pushed while still preserving the signal information that matters.

## Configs Used For The Reference Family

The corresponding golden-style configs live at the repository root:

- [configs/ppg_rvq_64hz_02x_golden.yaml](../../configs/ppg_rvq_64hz_02x_golden.yaml)
- [configs/ppg_rvq_64hz_04x_golden.yaml](../../configs/ppg_rvq_64hz_04x_golden.yaml)
- [configs/ppg_rvq_64hz_08x_golden.yaml](../../configs/ppg_rvq_64hz_08x_golden.yaml)
- [configs/ppg_rvq_64hz_16x_golden.yaml](../../configs/ppg_rvq_64hz_16x_golden.yaml)

## How To Evaluate Them

The easiest way to explore the family is:

1. Start at 2× to anchor the quality ceiling.
2. Move to 4× to establish a balanced operating point.
3. Jump to 16× to understand the efficiency ceiling.
4. Use 8× to decide whether the middle-to-aggressive tradeoff fits the target product.

If you want to see these tradeoffs in a more visual and interactive setting, the browser demo is covered in [PPG Codec Demo](../demo/ppg-codec.md).
