#!/bin/bash
# 补充 colocation 实验：α_p 测量 + Qwen3-14B 矩阵扩展
# Phase 1: 2 卡并行 — GPU0=Qwen2.5-7B, GPU1=Llama-3.1-8B
# Phase 2: 2 卡 TP=2 — GPU0+1=Qwen3-14B

set -e
cd "$(dirname "$0")/.."

RESULTS="results"
BATCH="1 4"
SEQ="32 64 128 512 2048"

echo "=== Supplement colocation: α_p measurement ==="
echo ""

# ── Phase 1: 两张卡并行跑 7B/8B ──
echo ">>> Phase 1: GPU0=Qwen2.5-7B, GPU1=Llama-3.1-8B (parallel)"
echo ""

CUDA_VISIBLE_DEVICES=0 python -m mlwd.supplement \
    --model qwen2.5-7b \
    --input "${RESULTS}/qwen2.5-7b-colocation.json" \
    --batch_sizes ${BATCH} --seq_lengths ${SEQ} \
    --num_runs 5 --warmup 2 --max_tokens 32 \
    2>&1 | tee "${RESULTS}/log_supplement_qwen25.txt" &
PID_Q25=$!

CUDA_VISIBLE_DEVICES=1 python -m mlwd.supplement \
    --model llama-3.1-8b \
    --input "${RESULTS}/llama-3.1-8b-colocation.json" \
    --batch_sizes ${BATCH} --seq_lengths ${SEQ} \
    --num_runs 5 --warmup 2 --max_tokens 32 \
    2>&1 | tee "${RESULTS}/log_supplement_llama31.txt" &
PID_L31=$!

echo "  PIDs: Qwen2.5=$PID_Q25  Llama3.1=$PID_L31"
wait $PID_Q25 $PID_L31
echo ">>> Phase 1 done."
echo ""

# ── Phase 2: 两张卡 TP=2 跑 Qwen3-14B ──
echo ">>> Phase 2: GPU0+1=Qwen3-14B (tp=2)"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m mlwd.supplement \
    --model qwen3-14b \
    --input "${RESULTS}/qwen3-14b-colocation.json" \
    --batch_sizes ${BATCH} --seq_lengths ${SEQ} \
    --num_runs 5 --warmup 2 --max_tokens 32 \
    --tp 2 \
    2>&1 | tee "${RESULTS}/log_supplement_qwen3.txt"

echo ">>> Phase 2 done."
echo ""

echo "=== All supplement experiments complete ==="

# ── 验证 ──
python3 -c "
import json
for f in ['${RESULTS}/qwen2.5-7b-colocation.json',
          '${RESULTS}/llama-3.1-8b-colocation.json',
          '${RESULTS}/qwen3-14b-colocation.json']:
    d = json.load(open(f))
    pairs = d.get('pairs', [])
    n_ap = sum(1 for p in pairs if 'alpha_p' in p and p['alpha_p'] != 'N/A')
    print(f'{d[\"model\"]}: {len(pairs)} pairs, {n_ap} with α_p')
"
