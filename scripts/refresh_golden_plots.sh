#!/usr/bin/env bash
# Regenerate golden results summary and documentation plots.
#
# Usage:
#   bash scripts/refresh_golden_plots.sh
#
# Reads results/*/summary.json → results/golden_summary.{csv,json}
# Generates astro-site/public/assets/plots/*.png (light + dark theme variants)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "==> Collecting golden results..."
python scripts/collect_golden_results.py

echo "==> Generating plots..."
python scripts/plot_golden_results.py

echo "==> Done. Plots saved to astro-site/public/assets/plots/"
ls -lh astro-site/public/assets/plots/*.png
