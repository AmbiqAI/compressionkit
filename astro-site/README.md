# compressionKIT documentation

Astro/Starlight is the documentation build under issue #68. Merging the cutover deploys the validated artifact to GitHub Pages.

Use Node 24 and npm 11:

```sh
cd astro-site
npm ci
npm run dev
npm run build
npm run check
```

Authored replacement pages live in `src/content/docs`. Navigation lives in `src/navigation.mjs`. The content inventory in `migration-manifest.json` records baseline source hashes and dispositions. See AUDIT.md for completed checks and remaining coverage.

`prepare:docs` renders existing notebooks from `../examples` without executing cells, and extracts Python API documentation with pinned Griffe static analysis. The public module allowlist is `scripts/public-api.json`. Generated pages and assets are ignored. `postbuild` publishes full API Markdown and updates the LLM bundle.

The pull-request CI workflow checks types, build, content, internal links, migration inventory and rendered layouts. It does not deploy. The Pages workflow runs the same validation and deploys only from main or a manual invocation. Registry facts, customer evidence and detailed fidelity tables have data-only rendering and refresh checks. Legacy heading mappings are inventoried and checked. Pending before cutover: three unresolved PhysioNet links, remote CI and approval. The three default notebooks and CLI help have passed in a local CPU container; see notebook-execution-audit.json. See CONTENT-OWNERSHIP.md for source ownership and refresh boundaries. Python runtime validation uses the repository container workflow. A rendered notebook is not an executed notebook.

Production builds use `astro build --force` so cached Markdown cannot retain obsolete ExpressiveCode asset references after theme changes. The rendered audit checks failed resource responses as well as page layout.

## Deployment and rollback

The Pages workflow uploads `astro-site/dist` with the `/compressionkit/` base path. PR checks produce a preview artifact without deploying. After merge, verify the production home, notebook, API and legacy redirect URLs. To roll back, revert the migration merge on main and manually run the restored Pages workflow. The pre-migration source is retained in Git history at `69b94d836dd8a24f227d411e99ef4ff2a7de2b7d`.
