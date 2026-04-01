#!/bin/bash
# Step 1: Collect MLWD data on V100 GPUs
# Run this on the GPU server with 4x V100

set -e
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-7b" "llama-3.2-3b")
OUTPUT_BASE="results/mlwd"

for model in "${MODELS[@]}"; do
    echo "=== Collecting MLWD for ${model} ==="
    mkdir -p "${OUTPUT_BASE}/${model}"

    # Sensitivity collection (4D interference sensitivity)
    echo "  [1/4] Collecting sensitivity..."
    python -m mlwd.collect_sensitivity \
        --model "${model}" \
        --output "${OUTPUT_BASE}/${model}/sensitivity.json" \
        --batch-sizes 1 4 \
        --seq-lengths 32 64 128 512 2048

    # Nsys profiling (execution mode features)
    echo "  [2/4] Collecting nsys features..."
    python -m mlwd.collect_nsys \
        --model "${model}" \
        --output "${OUTPUT_BASE}/${model}/nsys.json" \
        --batch-sizes 1 4 \
        --seq-lengths 32 64 128 512 2048

    # Compute intensity
    echo "  [3/4] Collecting CI..."
    python -m mlwd.collect_ci \
        --model "${model}" \
        --output "${OUTPUT_BASE}/${model}/ci.json" \
        --batch-sizes 1 4 \
        --seq-lengths 32 64 128 512 2048

    # Merge
    echo "  [4/4] Merging MLWD data..."
    python -m mlwd.merge \
        --sensitivity "${OUTPUT_BASE}/${model}/sensitivity.json" \
        --nsys "${OUTPUT_BASE}/${model}/nsys.json" \
        --ci "${OUTPUT_BASE}/${model}/ci.json" \
        --output "${OUTPUT_BASE}/${model}/mlwd_complete.json"

    echo "  Done: ${OUTPUT_BASE}/${model}/mlwd_complete.json"
done

# Colocation experiments + interference calibration
for model in "${MODELS[@]}"; do
    echo "=== Colocation experiments for ${model} ==="
    python -m mlwd.colocation \
        --model "${model}" \
        --mlwd "${OUTPUT_BASE}/${model}/mlwd_complete.json" \
        --output "${OUTPUT_BASE}/${model}/colocation.json"

    python -m mlwd.interference \
        --mlwd "${OUTPUT_BASE}/${model}/mlwd_complete.json" \
        --colocation "${OUTPUT_BASE}/${model}/colocation.json" \
        --output "${OUTPUT_BASE}/${model}/weights.json"
done

echo "=== MLWD collection complete ==="
