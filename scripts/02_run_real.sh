#!/bin/bash
# Step 2: Run real 2P2D experiments on 4x V100
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-7b" "llama-3.2-3b")
SCENARIOS=("disagg_2p2d" "coloc_4" "cbs_2p2d")
WORKLOADS=("uniform" "bursty" "long_context")
RATES_UNIFORM="2 4 6 8 10"
RATES_LONG="1 2 4 6"
DURATION=600

for model in "${MODELS[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            if [ "$workload" = "long_context" ]; then
                rates=$RATES_LONG
            elif [ "$workload" = "bursty" ]; then
                rates="4"  # bursty uses fixed base rate
            else
                rates=$RATES_UNIFORM
            fi

            for rate in $rates; do
                echo "=== ${model} / ${scenario} / ${workload} / ${rate} req/s ==="
                python -m real.run_real \
                    --scenario "$scenario" \
                    --model "$model" \
                    --workload "$workload" \
                    --rate "$rate" \
                    --duration "$DURATION" \
                    --output-dir "results/real/${model}/${scenario}/${workload}/rate_${rate}"
            done
        done
    done
done

echo "=== Real experiments complete ==="
