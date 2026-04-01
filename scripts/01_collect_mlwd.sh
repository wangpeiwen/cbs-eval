#!/bin/bash
# Step 1: Collect MLWD data on V100 GPUs
# 4 张卡并行：GPU0=qwen-sensitivity, GPU1=qwen-ci, GPU2=llama-sensitivity, GPU3=llama-ci
# nsys 和 merge 在各自完成后串行执行

set -e
cd "$(dirname "$0")/.."

# Build CUDA stress kernels if not already built
if [ ! -f build/cuda/libstress_interface.so ]; then
    echo "=== Building CUDA stress kernels ==="
    bash scripts/00_build_cuda.sh
fi

OUTPUT_BASE="results/mlwd"
mkdir -p "${OUTPUT_BASE}/qwen2.5-7b" "${OUTPUT_BASE}/llama-3.1-8b"

# ── 阶段 1: 4 卡并行跑 sensitivity 和 ci ──
echo "=== Phase 1: Parallel sensitivity + CI collection (4 GPUs) ==="

CUDA_VISIBLE_DEVICES=0 python -m mlwd.collect_sensitivity \
    --model "qwen2.5-7b" \
    --output "${OUTPUT_BASE}/qwen2.5-7b/sensitivity.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_Q_SENS=$!

CUDA_VISIBLE_DEVICES=1 python -m mlwd.collect_ci \
    --model "qwen2.5-7b" \
    --output "${OUTPUT_BASE}/qwen2.5-7b/ci.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_Q_CI=$!

CUDA_VISIBLE_DEVICES=2 python -m mlwd.collect_sensitivity \
    --model "llama-3.1-8b" \
    --output "${OUTPUT_BASE}/llama-3.1-8b/sensitivity.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_L_SENS=$!

CUDA_VISIBLE_DEVICES=3 python -m mlwd.collect_ci \
    --model "llama-3.1-8b" \
    --output "${OUTPUT_BASE}/llama-3.1-8b/ci.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_L_CI=$!

echo "  Waiting for 4 parallel jobs..."
wait $PID_Q_SENS $PID_Q_CI $PID_L_SENS $PID_L_CI
echo "  Phase 1 done."

# ── 阶段 2: 2 卡并行跑 nsys ──
echo "=== Phase 2: Parallel nsys profiling (2 GPUs) ==="

CUDA_VISIBLE_DEVICES=0 python -m mlwd.collect_nsys \
    --model "qwen2.5-7b" \
    --output "${OUTPUT_BASE}/qwen2.5-7b/nsys.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_Q_NSYS=$!

CUDA_VISIBLE_DEVICES=1 python -m mlwd.collect_nsys \
    --model "llama-3.1-8b" \
    --output "${OUTPUT_BASE}/llama-3.1-8b/nsys.json" \
    --batch_sizes 1 4 \
    --seq_lengths 32 64 128 512 2048 &
PID_L_NSYS=$!

wait $PID_Q_NSYS $PID_L_NSYS
echo "  Phase 2 done."

# ── 阶段 3: Merge（CPU，秒级） ──
echo "=== Phase 3: Merge ==="
for model in "qwen2.5-7b" "llama-3.1-8b"; do
    python -m mlwd.merge \
        --sensitivity "${OUTPUT_BASE}/${model}/sensitivity.json" \
        --nsys "${OUTPUT_BASE}/${model}/nsys.json" \
        --ci "${OUTPUT_BASE}/${model}/ci.json" \
        --output "${OUTPUT_BASE}/${model}/mlwd_complete.json"
    echo "  Merged: ${OUTPUT_BASE}/${model}/mlwd_complete.json"
done

# ── 阶段 4: 2 卡并行跑 colocation ──
echo "=== Phase 4: Parallel colocation experiments (2 GPUs) ==="

CUDA_VISIBLE_DEVICES=0 python -m mlwd.colocation \
    --model "qwen2.5-7b" \
    --mlwd "${OUTPUT_BASE}/qwen2.5-7b/mlwd_complete.json" \
    --output "${OUTPUT_BASE}/qwen2.5-7b/colocation.json" &
PID_Q_COL=$!

CUDA_VISIBLE_DEVICES=1 python -m mlwd.colocation \
    --model "llama-3.1-8b" \
    --mlwd "${OUTPUT_BASE}/llama-3.1-8b/mlwd_complete.json" \
    --output "${OUTPUT_BASE}/llama-3.1-8b/colocation.json" &
PID_L_COL=$!

wait $PID_Q_COL $PID_L_COL

# Interference calibration (CPU)
for model in "qwen2.5-7b" "llama-3.1-8b"; do
    python -m mlwd.interference \
        --mlwd "${OUTPUT_BASE}/${model}/mlwd_complete.json" \
        --colocation "${OUTPUT_BASE}/${model}/colocation.json" \
        --output "${OUTPUT_BASE}/${model}/weights.json"
done

echo "=== MLWD collection complete ==="
