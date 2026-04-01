#!/bin/bash
# Step 3: Run simulation experiments (8-node and 16-node)
set -e
cd "$(dirname "$0")/.."

SCALES=("small" "medium" "large")
BASELINES=("disagg_static" "coloc_sarathi" "cbs_nomig" "cbs_norole" "cbs_full")
WORKLOADS=("uniform" "bursty" "long_context")
RATES_UNIFORM="2 4 6 8 10 12 14 16"
RATES_LONG="1 2 4 6 8"

# Build interference table from MLWD data
echo "=== Building interference tables ==="
for model in "qwen2.5-7b" "llama-3.2-3b"; do
    python -m sim.profile_bridge \
        --mlwd-json "results/mlwd/${model}/mlwd_complete.json" \
        --weights-json "results/mlwd/${model}/weights.json" \
        --model-name "$model" \
        --output "results/sim/interference_${model}.json"
done

# Run simulations
for scale in "${SCALES[@]}"; do
    for baseline in "${BASELINES[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            if [ "$workload" = "long_context" ]; then
                rates=$RATES_LONG
            elif [ "$workload" = "bursty" ]; then
                rates="4 8"
            else
                rates=$RATES_UNIFORM
            fi

            for rate in $rates; do
                echo "=== sim: ${scale}/${baseline}/${workload}/${rate} ==="
                python -m sim.run_sim \
                    --config configs/sim_scenarios.yaml \
                    --scale "$scale" \
                    --baseline "$baseline" \
                    --workload "$workload" \
                    --rate "$rate" \
                    --interference-table "results/sim/interference_qwen2.5-7b.json" \
                    --output-dir "results/sim/${scale}/${workload}/rate_${rate}"
            done
        done
    done
done

echo "=== Simulation experiments complete ==="
