# internal/

Scratch area for **internal-only** material. Everything under this directory
is git-ignored except this README — use it for drafts and experiments that
are **not yet ready** to be shared with customers or published to the public
repository.

## Layout

```
internal/
├── README.md          ← tracked (this file)
├── docs/              ← internal-only docs (positioning, customer prep, drafts)
├── experiments/       ← scratch scripts, throwaway notebooks, WIP configs
└── results/           ← intermediate results we don't want to publish yet
```

## What belongs here

- Customer-readiness / positioning docs that contain commentary on competitors,
  internal weaknesses, or pre-publication findings.
- Experimental scripts that are not yet promoted to `scripts/`.
- Intermediate model artifacts or evaluation runs that are not yet
  scorecard-ready.

## What does NOT belong here

- Code that is reusable across the project — promote to `compressionkit/`.
- Customer-facing docs — promote to `astro-site/src/content/docs/`.
- Finalized scripts — promote to `scripts/`.
- Published results — golden runs go under `results/` (also git-ignored, but
  reproducible from configs) and are published to HuggingFace per the
  release process in `AGENTS.md`.

## Promotion checklist

When something in `internal/` is ready to be shared:

1. Strip internal-only commentary (competitor names, unflushed numbers,
   weakness lists).
2. Move to the appropriate public location (`astro-site/src/content/docs/`, `scripts/`,
   `compressionkit/`).
3. Add tests if it's code.
4. Update `astro-site/src/content/docs/index.mdx` if it's customer-facing documentation.
