# Home-page consistency handoff

Goal: align compressionKIT with the KIT documentation sites and physioKIT.

State:
- Done locally in `issue-79-kit-home-consistency`: replace the hero dot with the existing product icon, render the links below the hero with the shared helia-ui Button, and set `header.titleRegularPrefix` so the prefix is regular and KIT stays bold. The hero headline blends from icon-matched yellow on “Carry” toward bright green across “less data.” The home body follows the shared order: overview, four feature cards, source-checkout installation, and four navigation cards. Cards use CardHeader's full-card link and shared interactive focus treatment.
- Verified: Astro check and build pass. Link and layout audits pass (1,867 links; 404 layout checks), as do 12 content checks. The layout audit verifies readable shared quick buttons in both themes and pointer clicks in card bodies reaching their intended pages. Desktop, 785px and 390px mobile screenshots were inspected in light and dark mode; the icon renders and the page has no horizontal overflow.
- Preview: http://127.0.0.1:8778/compressionkit/
- Tracked by AmbiqAI/compressionkit#79. Draft PR #80 is open; no release or deployment has occurred.

Decisions:
- Reuse the repository's existing product icon and keep its current hero color and destination links.
- Use the shared Button component for the links under the hero; do not introduce a site-specific pill design.
- The header option is implemented in the local helia-ui worktree `/Users/adam.page/Ambiq/helia/helia-ui-header-prefix` but is not released. Preview dependencies have a local, untracked copy of that implementation. The consumer package pin stays at its current immutable version until a new helia-ui tag exists.

Next:
1. Review draft site PR #80, linked to issue #79 and request review.
2. After helia-ui#193 is merged and released, pin its immutable tag, regenerate the lockfile, rebuild, and review both themes.
3. Resolve review and CI findings, then ask Adam for approval. Do not merge without that approval.

Gotcha: `npm ci` resets the preview-only helia-ui copy in `node_modules`; restore or use the released tag before rebuilding.

Publication: final review fixes are pushed. Fresh CI and Copilot re-reviews requested; verify the final head before approval. Product PRs remain draft pending shared helia-ui approval, release and immutable dependency pins.

Not-found handling: restrict the product hero to the home route so unknown routes show the 404 page; built output and browser checks cover the fallback.
