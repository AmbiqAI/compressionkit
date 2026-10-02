# compressionKIT documentation

The Astro replacement is in progress under issue #68. Zensical remains the published site until cutover approval.

Use Node 24 and npm 11:

```sh
cd astro-site
npm ci
npm run dev
npm run build
npm run check
```

Authored replacement pages live in `src/content/docs`. Navigation lives in `src/navigation.mjs`; it does not read `zensical.toml`. The content inventory in `migration-manifest.json` records baseline source hashes and dispositions. See AUDIT.md for completed checks and remaining coverage.

`prepare:docs` renders existing notebooks from `../examples` without executing cells, and extracts Python API documentation with pinned Griffe static analysis. The public module allowlist is `scripts/public-api.json`. Generated pages and assets are ignored. `postbuild` publishes full API Markdown and updates the LLM bundle.

The preview-only CI workflow checks types, build, content, internal links, migration inventory and rendered layouts. It does not deploy. Registry facts, customer evidence and detailed fidelity tables have data-only rendering and refresh checks. Legacy heading mappings are inventoried and checked. Pending before cutover: three unresolved PhysioNet links, remote CI and approval. The three default notebooks and CLI help have passed in a local CPU container; see notebook-execution-audit.json. See CONTENT-OWNERSHIP.md for source ownership and refresh boundaries. Python runtime validation uses the repository container workflow. A rendered notebook is not an executed notebook.

Production builds use `astro build --force` so cached Markdown cannot retain obsolete ExpressiveCode asset references after theme changes. The rendered audit checks failed resource responses as well as page layout.
