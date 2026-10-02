# Documentation ownership and refresh

The Astro pages are authored content. Do not replace them with the older full-page experiment or evidence templates: those templates also write narrative and layout that has been revised here.

| Surface | Authority | Update contract |
| --- | --- | --- |
| Experiment facts | `compressionkit/experiments/registry.py` and referenced configs | The build exports the registry in an isolated Pydantic environment; ExperimentFacts renders overview values. Authored reproduction, dataset and explanation sections remain review-required when registry/config inputs change. `check:migration` fails on input drift. |
| Recorded evidence | Original documented evaluation snapshot; refreshed run scorecards when supplied | Customer evidence lives in content-data/customer-evidence.json and renders through EvidenceTable. Preserve sample counts and metric definitions. A complete scorecard set can refresh data without overwriting prose. |
| Notebook guides | `examples/*.ipynb` | `prepare:docs` renders saved cells/outputs and copies downloads byte for byte. It never executes notebooks. |
| Python API | Allowlisted Python modules and their docstrings | `prepare:docs` regenerates with static Griffe extraction. |
| Navigation and explanatory prose | `src/navigation.mjs` and `src/content/docs` | Edit directly and validate the built site. |

## Evidence refresh boundary

Run evidence exporters against an explicit results directory and write to a temporary review location, not over the Astro pages. The old exporters remain in use by the published site until cutover.

Experiment overview facts and customer-evidence tables have data-only exports and validated Astro renderers. Refresh customer evidence from a complete results tree with:

```sh
npm run refresh:evidence -- --results-dir /path/to/results --output /tmp/customer-evidence.json
```

Review the data diff, then replace `content-data/customer-evidence.json` and rebuild. Missing runs fail the refresh; absent measurements remain dashes. The provenance digest covers source scorecards and deploy artifacts. The initial snapshot preserves the source documentation values and is labeled accordingly. Detailed PPG and ECG fidelity tables render from `content-data/fidelity-ppg.json` and `fidelity-ecg.json`. Refresh both together with:

```sh
npm run refresh:fidelity -- --results-dir /path/to/results --output-dir /tmp/fidelity-review
```

Review both JSON diffs before copying them into `content-data/`. The exporter requires all eleven scorecards before writing either file. Zero-valued errors are preserved; absent measurements remain dashes. Run `npm run check:content`, rebuild and inspect the tables. Model and signal overview pages still contain separately recorded evaluations and need coordinated editorial review when measurements change. The migration's hash guard detects changed inputs; it does not prove that new evaluation results have been incorporated. Do not update recorded source hashes merely to silence it: review and update affected pages first.

## Migration checks

After `npm run build`, run `npm run check:migration` and `npm run check:links`. These check inventoried source changes, original page routes, migrated files and byte-identical notebook downloads. `npm run check:external` records HTTP availability separately because authentication, rate limiting and network policy can prevent verification without implying a broken destination. Legacy fragment dispositions are explicit in legacy-anchor-dispositions.json. The migration guard verifies retained aliases and relocated API targets; retired internal/demo sections retain documented reasons.
