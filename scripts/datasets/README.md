# Dataset ingestion scripts

Each `download_<slug>.py` here knows how to fetch one public physiological-signal
dataset, convert it to the project's canonical h5 layout under
`datasets/<slug>/`, and (eventually) upload the sanitized bundle to
`s3://ambiq-ai-datasets/<slug>/<slug>.zip` so future users can pull it back
with a single S3 fetch.

## Canonical layout

Each dataset becomes a directory of per-record h5 files:

```
datasets/<slug>/
    00001.h5
    00002.h5
    ...
```

Inside each h5 we follow the PTB-XL convention used by the existing trainer:

| Member | Type | Meaning |
|---|---|---|
| `data` | `(num_leads, num_samples)` float32 | ECG / PPG signal in millivolts (or device units, see `attrs["units"]`) |
| `attrs["fs"]` | int | Native sampling rate in Hz |
| `attrs["lead_names"]` | str (csv) | Channel names, e.g. `"I,II,III,..."` |
| `attrs["source"]` | str | Dataset slug |
| `attrs["acquisition"]` | str | One of: `clinic-12lead`, `holter`, `patch-1lead`, `smartphone-1lead`, `wearable`, `telehealth`, `bench-noise` |
| `attrs["patient_id"]` | str | Source-specific id; opaque |
| `attrs["age"]`, `attrs["sex"]` | optional | Demographics if available |
| `attrs["dx_codes"]` | optional str | csv-joined diagnostic labels |
| `r_peaks`, `blabels`, `slabels`, `segmentations`, `fiducials` | optional | Annotations if the source provides them |

## Inventory

Status legend: ✅ done (canonical h5 on disk) · ☁️ pre-published on Ambiq S3 · ⏳ raw source only (needs converter) · ❌ no public source

Confirmed Ambiq S3 contents (from a `list_objects_v2` against
`s3://ambiq-ai-datasets/`): ptbxl/, lsad/, ludb/, qtdb/, icentia11k/.
Everything else needs to be fetched from PhysioNet/Zenodo and converted.

| Bucket | Slug | Subjects | Lead/Fs | Acquisition | S3 source | Public source | Script | Status |
|---|---|---|---|---|---|---|---|---|
| Clinic 12-lead | `ptbxl` | 18,885 | 12 / 500 Hz | clinic-12lead | `ptbxl/ptbxl.zip` ☁️ | PhysioNet | (existing PtbxlDataset) | ✅ |
| Clinic 12-lead | `lsad` | 45,150 | 12 / 500 Hz | clinic-12lead | `lsad/lsad.zip` ☁️ | PhysioNet `ecg-arrhythmia` (Chapman+Ningbo combined) | [download_lsad.py](download_lsad.py) | ✅ |
| Clinic 12-lead | `cpsc2018` | 6,877 | 12 / 500 Hz | clinic-12lead | (none) | PhysioNet `cpsc2018` | TODO | ⏳ |
| Clinic 12-lead long | `incartdb` | 75 | 12 / 257 Hz | clinic-12lead-30min | (none) | PhysioNet `incartdb` | [download_incartdb.py](download_incartdb.py) | ✅ 75 records / 1.6 GB |
| Beat/wave delineation | `ludb` | 200 | 12 / 500 Hz | clinic-12lead | `ludb/ludb.zip` ☁️ (h5) | PhysioNet `ludb` | [download_ludb.py](download_ludb.py) | ✅ |
| Beat/wave delineation | `qtdb` | 102 | 2 / 250 Hz | clinic-2lead | `qtdb/qtdb.zip` ☁️ (raw WFDB; needs convert) | PhysioNet `qtdb` | [download_qtdb.py](download_qtdb.py) | ✅ (existing h5, skip re-download) |
| Holter / arrhythmia | `mitdb` | 48 | 2 / 360 Hz | holter | (none) | PhysioNet `mitdb` | [download_mitdb.py](download_mitdb.py) | ✅ |
| Holter / long AF | `ltafdb` | 84 | 2 / 128 Hz | holter-24h | (none) | PhysioNet `ltafdb` | TODO | ⏳ |
| Holter / SVA | `svdb` | 78 | 2 / 128 Hz | holter | (none) | PhysioNet `svdb` | TODO | ⏳ |
| Patch single-lead | `icentia11k` | 11,000 (600 pulled) | 1 / 250 Hz | patch-1lead-2week | `icentia11k/p*****.h5` ☁️ (object prefix) | PhysioNet `icentia11k-continuous-ecg` | [download_icentia11k.py](download_icentia11k.py) (`--limit`) | ✅ 600 patients / 16 GB |
| Smartphone single-lead | `alivecor2017` | 8,528 records | 1 / 300 Hz | smartphone-1lead | (none) | PhysioNet `challenge-2017` | [download_alivecor2017.py](download_alivecor2017.py) | ✅ |
| Telehealth | `code15` | ≥ 5k records (subsample) | 12 / 400 Hz | telehealth-12lead | (none) | Zenodo `code-15%` | TODO | ⏳ |
| Ambulatory ICU | `sharee` | 139 | 3 / 128 Hz | holter | (none) | PhysioNet `shareedb` | TODO | ⏳ |
| Bench / noise | `nstdb` | n/a | 2 / 360 Hz | bench-noise | (none) | PhysioNet `nstdb` | TODO | ⏳ |
| Motion-artifact ECG | `macecgdb` | 25 | 2 / 200 Hz | wearable-motion | (none) | PhysioNet `macecgdb` | TODO | ⏳ |

### PPG datasets

| Bucket | Slug | Subjects | Channels / Fs | Acquisition | Public source | Script | Status |
|---|---|---|---|---|---|---|---|
| Clinic bedside PPG | `bidmc` | 53 | PPG/ECG/RESP @ 125 Hz | clinic-bedside-ppg | PhysioNet `bidmc` | [download_bidmc.py](download_bidmc.py) | ✅ |
| Smartphone camera PPG | `butppg` | 50 (3,888 records) | PPG@30 Hz / ECG@1000 Hz / ACC@100 Hz | smartphone-1lead-ppg | PhysioNet `butppg` v2.0.0 | [download_butppg.py](download_butppg.py) | ✅ 3,882 / 3,888 records (6 dropped: missing ECG) / 243 MB |
| Wearable wrist PPG (free-living) | `ppg_dalia` | 15 | BVP@64 Hz / ECG@700 Hz / ACC@32 Hz / HR-GT@0.5 Hz | wearable-wrist-ppg | UCI 495 (PPG-DaLiA) | [download_ppg_dalia.py](download_ppg_dalia.py) | ✅ 15 subjects / 772 MB |
| Wearable wrist PPG (stress) | `wesad` | 15 | BVP@64 Hz / ECG@700 Hz / ACC@32 Hz / stress-label@700 Hz | wearable-wrist-ppg-stress | UCI 465 / Sciebo (WESAD) | [download_wesad.py](download_wesad.py) | ✅ 15 subjects / 634 MB |
| MIMIC bedside PPG | `mimic_perform` | 200+ | PPG/ECG/RESP @ 125 Hz | clinic-bedside-ppg | Charlton et al. mimic_perform | TODO | ⏳ |
| Capnography reference PPG | `capnobase` | 42 | PPG/ECG/CO2 @ 300 Hz | clinic-bedside-ppg | Borealis 10.5683/SP2/NLB8IT | TODO (Dataverse fiddly) | ⏳ |

## Workflow per dataset

1. **Smoke test** — set `--limit N` to ingest a handful of records and verify the canonical h5 layout.
2. **Full run** — drop `--limit`. Network-bound; usually safe to leave running.
3. **Verify** — open an h5 with h5py, confirm `data`, `fs`, `lead_names`, `acquisition` fields.
4. **(Later) Re-publish** — once the canonical bundle is stable, run with `--upload-s3 <bucket>/<slug>/<slug>.zip` to mirror it on the Ambiq S3 bucket so other devs (and the public dev container) can pull a single zip.

## Tackle order (for this iteration)

Done so far in this session:
1. ✅ **`ludb`** — smoke test of the S3+extract path. 200 patients downloaded.
2. ✅ **`lsad`** — Ambiq S3 single zip; 45,150 patients on disk in <15 s. (Confirmed to include both Chapman-Shaoxing and Ningbo subsets — no separate `chapman` script needed.)
3. ✅ **`icentia11k`** — Ambiq S3 object prefix; smoke verified at `--limit 2`. Use `--limit N` to sub-sample patients (full pull is ~330 GB).
4. ✅ **`mitdb`** — full PhysioNet pull complete; 48 records on disk with `r_peaks` + `beat_symbols`. Closes weakness #2 (beat-detection ground truth).
5. ✅ **`incartdb`** — 75 records / 1.6 GB. 12-lead × 30 min × 257 Hz with beat annotations → closes the long-recording / stitching gap.
6. ✅ **`alivecor2017`** — 8,528 records × 30 s × 300 Hz with rhythm-class labels. Closes the smartphone single-lead acquisition gap.
7. ✅ **`qtdb`** — 102 h5 files already present from a prior heartkit run. (Note: Ambiq `qtdb.zip` ships raw WFDB, not h5 — script has a guard against re-extraction clobbering existing files.)

Next up:
- **`code15`** — telehealth Zenodo bundle; subsample 5–10 k records.
- **`nstdb`** — noise stress benchmark; small but useful for adversarial PSD analysis (closes weakness #1, the 15-40 Hz QRS-detail PSD bleed).
- **`ltafdb` / `svdb`** — extra Holter / supraventricular variety.
- **`macecgdb`** — motion-artifact wearable; useful for bench-noise PSD.
- **(Optional re-publish)** — once the canonical layouts settle, run any of the scripts with `--upload-s3 ambiq-ai-datasets/<slug>/<slug>.zip` to mirror the new bundles to S3 for future devs.
