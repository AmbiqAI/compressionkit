---
title: "Dataset Setup"
description: "Prepare PPG and ECG datasets, configure cache locations, or evaluate your own signal."
---


compressionKIT separates **using a published codec** from **training or
evaluating on real data**:

- **Running a golden codec** (load from HuggingFace, compress/decompress a
  frame) needs **no dataset at all** — see [Getting Started](/compressionkit/getting-started/)
  and the example notebooks.
- **Reproducing a golden run** (training) or **evaluating on physiological
  recordings** needs the raw datasets below.

If you only want to try a codec on your own signals, skip straight to
[Bring your own data](#bring-your-own-data) or the synthetic generators.

## The `datasets/` convention

Every config and script resolves data through a single root directory. By
default that root is a relative `datasets/` folder next to the repository, and
it can be overridden with one environment variable:

```bash
# Point the whole toolkit at any location (no config edits required).
export COMPRESSIONKIT_DATASETS_DIR=/path/to/your/datasets
```

| Resolution order | Value |
|------------------|-------|
| 1. Environment variable | `COMPRESSIONKIT_DATASETS_DIR` |
| 2. Fallback default | `datasets` (relative to the working directory) |

In YAML configs the same root appears as `data.datasets_dir: datasets`, and
caches live under it (e.g. `datasets/ppg_cache/`). Override the env var and
every golden config follows.


:::note[Dev container shortcut]
If your raw data lives elsewhere, either set
`COMPRESSIONKIT_DATASETS_DIR`, or create a local (git-ignored) symlink:
```bash
ln -s /mnt/data/datasets datasets
```

:::

## Expected layout

```
datasets/
├── ptbxl/                 # ECG · PTB-XL (converted to .h5)
│   └── *.h5
├── bidmc/                 # PPG · BIDMC
│   └── *.h5
├── ppg_dalia/             # PPG · PPG-DaLiA
│   └── *.h5
├── wesad/                 # PPG · WESAD
│   └── *.h5
├── butppg/                # PPG · BUT PPG
│   └── *.h5
├── mesa-commercial-use/   # PPG · MESA (NSRR, restricted)
│   └── polysomnography/edfs/*.edf
├── ppg_cache_strict_sanitize/  # Generated PPG v1 golden cache (built once)
│   ├── bidmc/{train,val}.tfrecord
│   ├── butppg/{train,val}.tfrecord
│   ├── ppg_dalia/{train,val}.tfrecord
│   └── wesad/{train,val}.tfrecord
├── ppg_cache/             # Optional ad hoc PPG caches
└── ecg_tfrecord_cache/    # Generated TFRecord cache (built once)
```

<span id="ecg-ptb-xl"></span>

## ECG — PTB-XL

PTB-XL is released under **CC BY 4.0** and may be redistributed. The loader
ships a one-time downloader that fetches and converts records to HDF5:

```python
from compressionkit.datasets import PtbxlDataset

ds = PtbxlDataset(path="datasets/ptbxl")
ds.download()          # one-time: fetch + convert to .h5
sig = ds.load_signal(patient_id=1)   # (12, 5000) float32 @ 500 Hz
```

Golden ECG configs read 256 Hz, lead II windows via `data.dataset_glob:
ptbxl/*.h5`.

<span id="ppg-mesa-open-sources"></span>

## PPG — MESA + open sources

The PPG golden codecs train on a **unified mixture** of open PPG datasets.
MESA is **NSRR-restricted and must not be
redistributed** — request access through the [NSRR](https://sleepdata.org/).
The remaining sources are openly available:

| Slug | Dataset | Format | Notes |
|------|---------|--------|-------|
| `mesa` | MESA (PSG) | `.edf` | NSRR restricted; not used in published goldens |
| `bidmc` | BIDMC | `.h5` | Open |
| `ppg_dalia` | PPG-DaLiA | `.h5` | Open |
| `wesad` | WESAD | `.h5` | Open |
| `butppg` | BUT PPG | `.h5` | Open; quality-filtered |


:::note[Published goldens are MESA-free]
The v1 PPG goldens train on `bidmc`, `butppg`, `ppg_dalia`, and `wesad`
so they can be reproduced without restricted data.

:::

### Download or prepare the open PPG sources

Install the ingestion helpers, then fetch or convert each open source into the
canonical `datasets/<slug>/*.h5` layout. The download scripts accept `--limit`
for a smoke test before a full pull.

```bash
uv sync --group ingest

# Optional smoke tests first.
uv run python scripts/datasets/download_bidmc.py --limit 3
uv run python scripts/datasets/download_butppg.py --limit 5
uv run python scripts/datasets/download_ppg_dalia.py --limit 2
uv run python scripts/datasets/download_wesad.py --limit 2

# Full source preparation for the v1 PPG golden cache.
uv run python scripts/datasets/download_bidmc.py
uv run python scripts/datasets/download_butppg.py
uv run python scripts/datasets/download_ppg_dalia.py
uv run python scripts/datasets/download_wesad.py
```

If a source is already present, the scripts skip work or reuse existing files
where possible. Keep each source under the same root selected by
`COMPRESSIONKIT_DATASETS_DIR`.

### Build the PPG cache

Each source is cached once into TFRecords, then any combination can be mixed
during training:

```bash
# Build the exact open-source cache expected by the v1 PPG goldens.
uv run python scripts/build_ppg_cache.py \
  --sources bidmc butppg ppg_dalia wesad \
  --cache-root datasets/ppg_cache_strict_sanitize

# Custom locations (otherwise resolved from COMPRESSIONKIT_DATASETS_DIR).
uv run python scripts/build_ppg_cache.py --sources bidmc \
    --datasets-root /data/datasets \
  --cache-root /data/ppg_cache_strict_sanitize
```

The golden runner checks for `train.tfrecord`, `val.tfrecord`, and
`metadata.json` under each required source directory. If the cache is missing,
`compressionkit golden run ppg-rvq-8x` fails early with the same build command
instead of failing deep in training.

## Bring your own data

Load a one-dimensional waveform, resample it to the codec sample rate, and follow its preprocessing contract. This example assumes `your_signal` is already a float32 array with those properties:

```python
import numpy as np
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-4x-v1.0")
fs, n = codec.sample_rate, codec.frame_size   # PPG: 64 Hz

# your_signal: 1-D float32 sampled at `fs`
frames = your_signal[: len(your_signal) // n * n].reshape(-1, n)
recon = np.concatenate([codec.decompress(codec.compress(f)) for f in frames])
```

This snippet drops an incomplete final frame and does not implement overlap or stitching. See the [evaluation notebook](/compressionkit/guides/02_evaluate_on_your_data/) for a fuller workflow.

## No data? Use the synthetic generators

For demos, CI, and quick sanity checks, compressionKIT ships analytical
PPG/ECG generators (no datasets, fully license-safe):

```python
from compressionkit.synthetic import ppg_dynamical, ecg_mcsharry, add_noise, NoiseSpec

clean = ppg_dynamical(duration_s=30.0, sample_rate=64.0, hr_mean=72.0)
ecg = ecg_mcsharry(duration_s=10.0, sample_rate=256.0, hr_mean=60.0)
noisy, noise = add_noise(clean, sample_rate=64.0, snr_db=15.0)
```

These pair a ground-truth-clean waveform with a controlled noise harness, which
is exactly what the example notebooks use so they run anywhere.

## Licensing summary

| Dataset | Modality | License | Redistribute? |
|---------|----------|---------|---------------|
| PTB-XL | ECG | CC BY 4.0 | Yes |
| MESA | PPG | NSRR DUA | **No** |
| BIDMC | PPG | Open (PhysioNet) | Per source terms |
| PPG-DaLiA | PPG | Open | Per source terms |
| WESAD | PPG | Open | Per source terms |
| BUT PPG | PPG | Open | Per source terms |
| Synthetic (physiokit) | PPG/ECG | Generated | Yes |

Always confirm each source's terms before redistributing derived artifacts.
