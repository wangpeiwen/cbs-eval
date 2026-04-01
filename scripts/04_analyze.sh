#!/bin/bash
# Step 4: Analyze results and generate plots
set -e
cd "$(dirname "$0")/.."

echo "=== Generating analysis ==="
python -m analysis.compare --results-dir results/
python -m analysis.ablation --results-dir results/sim/
python -m analysis.sensitivity --results-dir results/sim/

echo "=== Generating plots ==="
python -c "
from plot.thesis_plots import generate_all
generate_all('results/', 'results/figures/')
"

echo "=== Analysis complete. Figures saved to results/figures/ ==="
