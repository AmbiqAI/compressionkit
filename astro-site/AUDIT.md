# compressionKIT documentation audit

Scope: the local Astro replacement under issue #68. The published site is unchanged.

## Content and layout revised

- Home: replaced the repetitive release narrative, duplicated installation commands, and overlapping navigation tables with a product explanation, benefit cards, a four-step workflow, and four starting points.
- Use cases: replaced unsupported battery-life, clinical-suitability, and universal fidelity claims with concrete integration scenarios and explicitly calculated payload examples.
- Evidence: separated quality, package footprint, and detailed metrics; explained differing evaluation views and missing measurements. Recorded metric values are preserved.
- Models and methods: simplified the model index, split wide tables, retained primary plots, and made secondary plots expandable. Corrected stale SPIHT/hybrid availability statements.
- Experiments: grouped the index by signal, made page titles readable, retained stable IDs, and separated reproduction from optional publication. Removed the misleading dataset-bypass smoke command.
- Guides: clarified Python prerequisites, provided the missing step to run the first script, removed duplicate installation instructions, added the ECG notebook to the example overview, and repaired prose-only file references.
- Reference: added context to topic pages, removed internal planning/refactoring material from public architecture and contract pages, and clarified scorecard coverage versus application acceptance criteria.
- Generated API: converted reStructuredText literal examples into fenced code before rendering; Python comments no longer become page headings. Two regression tests cover this conversion.
- Layout: bounded narrative line length, consistent section spacing, compact card headings, readable tables, expandable long examples, a smaller mobile capability grid, and room for the product name in the mobile header.

## Verified on this revision

- Production build passes.
- Astro check: 0 errors, warnings, or hints.
- Docstring regression tests: 2 passing.
- Internal link, asset and anchor audit: 1,518 checks, no failures.
- Production-rendered route audit: 97 routes × 2 viewports (390px and 1440px) × 2 themes = 388 checks, no failures. Checks cover response status, horizontal overflow, visible images, one page heading, empty links and code accidentally rendered inside cards.
- Screenshots inspected for home, evidence, model and experiment pages in desktop/mobile and light/dark configurations.
- Mobile section selection, scoped sidebar, plot disclosure and API search tested; `load_codec` returns one API match.
- All three downloadable notebooks match canonical source bytes. Model/fidelity tables retain numeric source cells.
- Live Hugging Face API: all 33 linked bundle repositories returned 200 and file inventories. Recorded revisions are in `bundle-audit.json`. Availability does not establish successful runtime loading or evaluation.

## Remaining migration work

- Integrate experiment/evidence generators with the authored Astro structure and add drift checks. Retain source ownership rather than overwriting the editorial pass with old templates.
- Independently review the migration and wire final CI/cutover. No deployment occurred in this pass.
- Runtime execution, notebook execution, fresh model evaluation, and hardware integration are not verified by this documentation audit.
- The internal-link audit and bundle check are not a complete external-link availability audit.
- Muted code backgrounds and distinct notebook input/output presentation are applied through supported theme configuration. Shared collapsible CodeBlock/Callout options remain a separate proposed helia-ui feature.

## Gap closure pass

- Added SPIHT and hybrid method guides, a practical selection workflow and device/transport/host deployment boundaries. An independent source review found inaccurate minimal Python dependency claims; those are corrected.
- Preview CI added in `.github/workflows/docs-preview.yml`. Existing published-site deployment is unchanged. Remote CI has not run.
- Build and Astro check pass; 99 routes / 396 layout checks pass. Rendered deployment screenshot inspected.
- `check:migration` passes for 115 inventoried items, original page routes and notebook bytes. Registry/config/generator input hashes are watched. This catches drift but is not data-only generator integration.
- External audit attempted 81 URLs: last retry returned 49 successful responses, 31 HTTP 503 responses and one connection failure. No 404 responses observed; failures remain unresolved rather than classified as dead links.
- Source-heading comparison identified 131 candidate old fragment IDs absent from the new render. `anchor-review.json` is a triage list, not verified old-site IDs; exact old rendered anchors still need checking and aliases where appropriate.
- Runtime execution blocked: the local dev container is stopped and `devcontainer` CLI unavailable. Requested the user's intended execution environment. No notebooks executed.
- Evidence scorecards are not present in this checkout. Automated evidence refresh needs the source scorecards and data-only export/render integration; preserved snapshot values remain unchanged.

Published-site anchor verification: fetched all 69 migrated original pages, inspected 499 heading IDs with no fetch failures. Found 168 missing old IDs before remediation. Added 43 aliases where the corresponding heading text matches exactly after punctuation normalization. Remaining renamed/removed sections and relocated API symbols need deliberate destination mapping. This supersedes the earlier heuristic 131-candidate note.

## Independent review follow-up

Resolved the review findings: false minimal-dependency claims in source docstrings and the HF guide, prior artifact path, same-site absolute link coverage, hosted notebook setup guidance, and inventory coverage for new notebooks. Five regression tests and 1,553 internal links pass. Follow-up review confirmed fixes and 37 registry entries matched to 37 experiment pages. Overview facts now render from the registry export, with target ratio explicitly labeled. This does not regenerate evaluation scorecards or validate runtime execution.

## Evidence data and old-anchor disposition

Customer evidence is now a validated data snapshot rendered separately from prose. An explicit refresh command accepts a complete scorecard/results tree and records source provenance; it never treats missing measurements as zero or a partial file sum as Total. Fixture coverage verifies these boundaries. Preserved values have not been replaced with fresh model results. Detailed fidelity/modality tables remain separately maintained snapshots.

Published original-site anchor audit: 168 previously missing IDs now have 104 heading aliases, 32 API entry-point links and 32 intentional retirements (internal guidance and demo documentation moved to the maintained hub). Migration checks verify active mappings. Eight content tests, 1,558 local links and 396 rendered layout checks pass; evidence tables and a legacy API link inspected. No runtime or notebook execution, remote CI or deployment claimed.

## Detailed fidelity refresh and production recheck

PPG and ECG fidelity pages now render five validated tables each from data-only snapshots. Independent review confirmed exact original cells and source hashes, full ratio/noise coverage, zero preservation and failure before writing partial results. Refresh tooling does not overwrite authored prose; no fresh evaluation is claimed.

The production audit exposed stale ExpressiveCode CSS references after a theme change. Build now forces the Astro content cache to refresh, and the browser audit fails resource HTTP errors. All 396 production route/viewport/theme checks pass with this stronger guard; mobile muted code styling and scrolling inspected. Ten fixture/content tests, 1,558 local links, 115 migration entries and 168 anchor dispositions pass. Build and type checks pass. Runtime/notebook execution, unresolved external availability and remote CI remain unverified.

## Local execution and clean checkout verification

The three canonical notebooks executed successfully on 2026-10-02 in a local Linux ARM64 CPU container with Python 3.12 and locked project dependencies: 22 code cells, no errors. Source hashes and coverage are recorded in notebook-execution-audit.json. CLI --help passed. Source notebooks and displayed saved outputs remain unchanged. This verifies default examples, not training, GPU, hardware or all model tiers.

A clean source checkout passed npm ci, types/build, ten content tests, 1,558 links and migration checks. Second independent delivery review found no actionable findings. External retry resolved 76 of 81 links; the GitHub source link was verified via authenticated API and sleepdata.org returned 200 via curl. Three PhysioNet URLs remain DNS-unverified. No PR, remote CI or deployment yet.

## PR review corrections

Corrected the quickstart to use the public target_cr property and executed the exact snippet in the local CPU container. Added meaningful alternative text to all seven saved notebook plots; the generator escapes it and rejects missing descriptions. All three notebooks were rerun successfully after their metadata changes. Preview CI now covers root project metadata and lockfile changes. Twelve content tests, build/types, 1,558 links, migration checks and 396 rendered checks pass.
