---
icon: lucide/layers
---

# Compression Methods

compressionKIT implements multiple compression approaches for physiological signals, ranging from classical signal processing to learned neural methods.

## Method Comparison

| Method | Type | Compression | Latency | Quality | Deployable |
|--------|------|-------------|---------|---------|------------|
| [**RVQ Autoencoder**](rvq.md) | Learned | 2×–16× documented | Low | High | :material-check: INT8 TFLite |
| **Wavelet + SPIHT** | Classical | 2×–16× | Very Low | Medium | :material-progress-clock: Planned |
| **Decimation** | Baseline | 2×–16× | Minimal | Low | Trivial |

## Design Principles

All compression methods in compressionKIT follow these principles:

1. **Portable to embedded C** — No dynamic allocation, fixed memory layouts
2. **Quantization-friendly** — Only operators that work with INT8/INT16x8 quantization
3. **Configurable via YAML** — Major parameters exposed through configuration files
4. **Evaluated on clinical metrics** — Not just MSE, but HR/HRV preservation
