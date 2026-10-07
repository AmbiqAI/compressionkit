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
- Pin helia-ui v0.1.0-alpha.24, published from f7158bf, with an npm 11.19.0 lockfile. Clean installs reproduce the shared header without a local patch.

Next: finish checks against the released package, final review and CI, merge under Adam's authorization, then verify the Pages deployment.

Not-found handling: restrict the product hero to the home route so unknown routes show the 404 page; built output and browser checks cover the fallback.


Release validation: clean installation of alpha.24 passed. Final check/build/output checks pass, with 404 rendered acceptance checks passing against the clean released dependency. User authorized merging after green CI.
