"""补充 α_p 测量 + Qwen3-14B 矩阵扩展。

读取已有 colocation 结果，补充缺失的：
  1. prefill baselines（所有模型都缺）
  2. α_p（所有模型的已有 pairs 都缺）
  3. Qwen3-14B 的 b4 decode baseline + b4 组合 pairs（含 α_d 和 α_p）

不重新测量已有的 α_d 数据。

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m mlwd.supplement \
        --model qwen2.5-7b \
        --input results/qwen2.5-7b-colocation.json \
        --output results/qwen2.5-7b-colocation.json
"""

import argparse, json, os, time, gc
from itertools import product
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZES, DEFAULT_SEQ_LENGTHS,
    FULL_BATCH_SIZES, FULL_SEQ_LENGTHS,
    resolve_model_path,
)
from .colocation import (
    _save, _load, _median, _free_gpu,
    measure_decode_only, measure_prefill_only,
    measure_pd_colocation, measure_prefill_colocation,
)


def supplement_experiment(model_path, input_path, output_path,
                          batch_sizes=None, seq_lengths=None,
                          num_runs=5, warmup=2, max_tokens=32,
                          tp=1, max_model_len=4096):
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = DEFAULT_SEQ_LENGTHS

    os.environ["VLLM_USE_V1"] = "0"

    data = _load(input_path)
    model_name = data.get("model", Path(model_path).name)

    print(f"{'='*60}")
    print(f"  Supplement: {model_name}")
    print(f"  Input: {input_path}")
    print(f"  Target batch sizes: {batch_sizes}, Seq lengths: {seq_lengths}")
    print(f"{'='*60}\n")

    # ── 分析缺失数据 ──
    existing_baselines = set(data.get("baselines", {}).keys())
    existing_pfx_baselines = set(data.get("prefill_baselines", {}).keys())
    existing_pairs = {p["key"]: p for p in data.get("pairs", [])}

    need_decode_baselines = [b for b in batch_sizes if f"b{b}" not in existing_baselines]
    need_pfx_baselines = []
    for b_p, s_p in product(batch_sizes, seq_lengths):
        key = f"b{b_p}_s{s_p}"
        if key not in existing_pfx_baselines:
            need_pfx_baselines.append((b_p, s_p))

    need_alpha_p = [k for k, p in existing_pairs.items() if "alpha_p" not in p or p["alpha_p"] == "N/A"]
    need_new_pairs = []
    for b_d in batch_sizes:
        for b_p, s_p in product(batch_sizes, seq_lengths):
            key = f"d{b_d}_p{b_p}x{s_p}"
            if key not in existing_pairs:
                need_new_pairs.append((b_d, b_p, s_p))

    print(f"  Missing decode baselines: {need_decode_baselines}")
    print(f"  Missing prefill baselines: {len(need_pfx_baselines)}")
    print(f"  Pairs needing α_p: {len(need_alpha_p)}")
    print(f"  New pairs to measure: {len(need_new_pairs)}")

    total_work = (len(need_decode_baselines) + len(need_pfx_baselines)
                  + len(need_alpha_p) + len(need_new_pairs))
    if total_work == 0:
        print("\n  Nothing to do — all data complete.")
        return

    # ── 加载模型 ──
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\nLoading {model_name}...")
    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.85,
              max_model_len=max_model_len, tensor_parallel_size=tp)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.\n")

    # ── Step A: 补充 decode baselines ──
    if need_decode_baselines:
        print(f"Step A: Decode baselines for {need_decode_baselines}")
        if "baselines" not in data:
            data["baselines"] = {}
        for b in need_decode_baselines:
            med, lats = measure_decode_only(llm, tokenizer, b, 0, max_tokens,
                                             num_runs, warmup)
            data["baselines"][f"b{b}"] = {
                "batch_size": b, "median_ms": med, "all_ms": lats
            }
            print(f"  b={b}: {med:.1f} ms")
        _save(output_path, data)
        print()

    # ── Step B: 补充 prefill baselines ──
    if need_pfx_baselines:
        print(f"Step B: Prefill baselines ({len(need_pfx_baselines)} combos)")
        if "prefill_baselines" not in data:
            data["prefill_baselines"] = {}
        for b_p, s_p in need_pfx_baselines:
            key = f"b{b_p}_s{s_p}"
            med, lats = measure_prefill_only(llm, tokenizer, b_p, s_p,
                                              num_runs, warmup)
            data["prefill_baselines"][key] = {
                "batch_size": b_p, "seq_len": s_p,
                "median_ms": med, "all_ms": lats
            }
            print(f"  b={b_p}, s={s_p}: {med:.1f} ms")
        _save(output_path, data)
        print()

    # ── Step C: 补充已有 pairs 的 α_p ──
    if need_alpha_p:
        print(f"Step C: Supplement α_p for {len(need_alpha_p)} existing pairs")
        pairs_list = data["pairs"]
        for idx, pair in enumerate(pairs_list):
            if pair["key"] not in need_alpha_p:
                continue
            b_d = pair["decode_batch"]
            b_p = pair["prefill_batch"]
            s_p = pair["prefill_seq"]
            pfx_key = f"b{b_p}_s{s_p}"
            pfx_baseline = data["prefill_baselines"][pfx_key]["median_ms"]

            pfx_coloc_med, pfx_coloc_lats = measure_prefill_colocation(
                llm, tokenizer,
                decode_batch=b_d, prefill_batch=b_p, prefill_seq=s_p,
                max_tokens=max_tokens, num_runs=num_runs, warmup=warmup)
            alpha_p = (pfx_coloc_med - pfx_baseline) / pfx_baseline if pfx_baseline > 0 else 0

            pair["prefill_baseline_ms"] = round(pfx_baseline, 4)
            pair["prefill_coloc_ms"] = round(pfx_coloc_med, 4)
            pair["alpha_p"] = round(alpha_p, 6)
            print(f"  {pair['key']}: α_p={alpha_p:.4f}")

        data["pairs"] = pairs_list
        _save(output_path, data)
        print()

    # ── Step D: 新增 pairs（Qwen3-14B 的 b4 组合等） ──
    if need_new_pairs:
        print(f"Step D: New pairs ({len(need_new_pairs)} combos)")
        if "pairs" not in data:
            data["pairs"] = []

        for i, (b_d, b_p, s_p) in enumerate(need_new_pairs):
            key = f"d{b_d}_p{b_p}x{s_p}"
            print(f"  [{i+1}/{len(need_new_pairs)}] {key}...", end=" ", flush=True)

            baseline = data["baselines"][f"b{b_d}"]["median_ms"]
            coloc_med, coloc_lats = measure_pd_colocation(
                llm, tokenizer,
                decode_batch=b_d, decode_seq=0,
                prefill_batch=b_p, prefill_seq=s_p,
                max_tokens=max_tokens, num_runs=num_runs, warmup=warmup)
            alpha_d = (coloc_med - baseline) / baseline if baseline > 0 else 0

            pfx_key = f"b{b_p}_s{s_p}"
            pfx_baseline = data["prefill_baselines"][pfx_key]["median_ms"]
            pfx_coloc_med, pfx_coloc_lats = measure_prefill_colocation(
                llm, tokenizer,
                decode_batch=b_d, prefill_batch=b_p, prefill_seq=s_p,
                max_tokens=max_tokens, num_runs=num_runs, warmup=warmup)
            alpha_p = (pfx_coloc_med - pfx_baseline) / pfx_baseline if pfx_baseline > 0 else 0

            entry = {
                "key": key,
                "model": model_name,
                "decode_batch": b_d,
                "prefill_batch": b_p, "prefill_seq": s_p,
                "victim_phase": "decode", "aggressor_phase": "prefill",
                "victim_b": b_d, "victim_s": max_tokens,
                "aggressor_b": b_p, "aggressor_s": s_p,
                "baseline_ms": round(baseline, 4),
                "coloc_ms": round(coloc_med, 4),
                "alpha_d": round(alpha_d, 6),
                "prefill_baseline_ms": round(pfx_baseline, 4),
                "prefill_coloc_ms": round(pfx_coloc_med, 4),
                "alpha_p": round(alpha_p, 6),
                "all_ms": coloc_lats,
            }
            data["pairs"].append(entry)
            print(f"α_d={alpha_d:.4f}  α_p={alpha_p:.4f}")

        _save(output_path, data)
        print()

    del llm
    _free_gpu()

    n = len(data.get("pairs", []))
    n_with_ap = sum(1 for p in data["pairs"] if "alpha_p" in p and p["alpha_p"] != "N/A")
    print(f"Done. {n} pairs ({n_with_ap} with α_p) → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="补充 α_p 测量 + 矩阵扩展")
    parser.add_argument("--model", required=True, help="模型短名称或路径")
    parser.add_argument("--input", required=True, help="已有 colocation 结果路径")
    parser.add_argument("--output", help="输出路径（默认覆盖 input）")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    supplement_experiment(
        model_path=resolve_model_path(args.model),
        input_path=args.input,
        output_path=args.output or args.input,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_runs=args.num_runs,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        tp=args.tp,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
