#!/bin/bash
# Step 3: Run simulation experiments (8-node and 16-node)
set -e
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
SIM_DURATION="${SIM_DURATION:-60}"
SIM_FORCE="${SIM_FORCE:-1}"
SCALES=("small" "medium" "large")
BASELINES=("disagg_static" "coloc_sarathi" "cbs_nomig" "cbs_norole" "cbs_full")
WORKLOADS=("uniform" "bursty" "long_context")
RATES_UNIFORM="2 4 6 8 10 12 14 16"
RATES_LONG="1 2 4 6 8"

# Build interference table from MLWD data
echo "=== Building interference tables ==="
for model in "qwen2.5-7b" "llama-3.1-8b"; do
    weights_arg=()
    if [ -f "results/mlwd/${model}/weights.json" ]; then
        weights_arg=(--weights-json "results/mlwd/${model}/weights.json")
    fi

    "$PYTHON_BIN" -m sim.profile_bridge \
        --mlwd-json "results/mlwd/${model}/mlwd_complete.json" \
        "${weights_arg[@]}" \
        --model-name "$model" \
        --output "results/sim/interference_${model}.json"
done

# Run simulations in batches. One Python process handles all scales/baselines
# for a workload/rate pair, avoiding 225 interpreter startups.
for workload in "${WORKLOADS[@]}"; do
    if [ "$workload" = "long_context" ]; then
        rates=$RATES_LONG
    elif [ "$workload" = "bursty" ]; then
        rates="4 8"
    else
        rates=$RATES_UNIFORM
    fi

    for rate in $rates; do
        if [ "$SIM_FORCE" != "1" ]; then
            all_present=1
            for scale in "${SCALES[@]}"; do
                for baseline in "${BASELINES[@]}"; do
                    output_file="results/sim/${workload}/rate_${rate}.0/${scale}_${baseline}.json"
                    if [ ! -f "$output_file" ]; then
                        all_present=0
                    fi
                done
            done
            if [ "$all_present" = "1" ]; then
                echo "=== skip existing batch: ${workload}/${rate} ==="
                continue
            fi
        fi

        echo "=== sim batch: ${workload}/${rate} ==="
        "$PYTHON_BIN" -m sim.run_sim \
            --config configs/sim_scenarios.yaml \
            --scale all \
            --baseline all \
            --workload "$workload" \
            --rate "$rate" \
            --duration "$SIM_DURATION" \
            --interference-table "results/sim/interference_qwen2.5-7b.json" \
            --output-dir "results/sim"
    done
done

echo "=== Simulation experiments complete ==="
