"""同模型 PD 共置干扰实验。

测量单个 vLLM 实例内，Prefill 请求注入对正在进行的 Decode 请求的干扰。
这是 CBS 调度算法的真实场景：决定是否将新请求的 Prefill 放到 Decode 节点上。

实验设计：
  1. Decode-only baseline: 发送 N 个请求，全部进入 Decode 阶段后测量 per-token 时延
  2. Prefill-only baseline: 发送长 prompt 请求，测量 Prefill 时延
  3. PD co-location: Decode 进行中注入 Prefill 请求，测量双方时延退化
  4. α_d = (T_decode_coloc - T_decode_baseline) / T_decode_baseline
  5. α_p = (T_prefill_coloc - T_prefill_baseline) / T_prefill_baseline

Usage:
    python -m mlwd.colocation \\
        --model /data/Qwen2.5-7B-Instruct \\
        --gpu 0 --output output/colocation.json
"""

import argparse, json, os, time, gc
from itertools import product
from pathlib import Path

from .config import Experiment, OUTPUT_DIR, DEFAULT_BATCH_SIZES, DEFAULT_SEQ_LENGTHS, resolve_model_path


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _median(lats):
    lats = sorted(lats)
    mid = len(lats) // 2
    return lats[mid] if len(lats) % 2 else (lats[mid - 1] + lats[mid]) / 2


def _free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 核心测量函数 ──────────────────────────────────────

def measure_decode_only(llm, tokenizer, batch_size, seq_len, max_tokens,
                        num_runs, warmup):
    """纯 Decode baseline：所有请求都是短 prompt + 长输出。"""
    from vllm import SamplingParams
    prompts = ["The"] * batch_size
    sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    for _ in range(warmup):
        llm.generate(prompts, sp)

    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return _median(lats), sorted(lats)


def measure_prefill_only(llm, tokenizer, batch_size, seq_len,
                         num_runs, warmup):
    """纯 Prefill baseline：长 prompt + 1 token 输出。"""
    from vllm import SamplingParams
    text = "hello " * (seq_len * 2)
    ids = tokenizer.encode(text)[:seq_len]
    prompt = tokenizer.decode(ids)
    prompts = [prompt] * batch_size
    sp = SamplingParams(max_tokens=1, temperature=0)

    for _ in range(warmup):
        llm.generate(prompts, sp)

    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return _median(lats), sorted(lats)


def measure_pd_colocation(llm, tokenizer, decode_batch, decode_seq,
                           prefill_batch, prefill_seq, max_tokens,
                           num_runs, warmup):
    """PD 共置：Decode 请求 + Prefill 请求混合提交，测量 Decode 侧时延退化。

    模拟 CBS 场景：节点上有 decode_batch 个 Decode 请求正在生成，
    同时注入 prefill_batch 个新的 Prefill 请求（长 prompt, 1 token 输出）。

    vLLM 的 continuous batching 会在同一个 iteration 内同时处理两类请求，
    Prefill 的大 GEMM 会抢占 SM，干扰 Decode 的小 kernel。
    """
    from vllm import SamplingParams

    decode_prompts = ["The"] * decode_batch
    decode_sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    text = "hello " * (prefill_seq * 2)
    ids = tokenizer.encode(text)[:prefill_seq]
    prefill_prompt = tokenizer.decode(ids)
    prefill_prompts = [prefill_prompt] * prefill_batch
    prefill_sp = SamplingParams(max_tokens=1, temperature=0)

    mixed_prompts = decode_prompts + prefill_prompts

    for _ in range(warmup):
        llm.generate(decode_prompts, decode_sp)

    lats = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        llm.generate(decode_prompts + prefill_prompts,
                     SamplingParams(max_tokens=max_tokens, temperature=0))
        lats.append((time.perf_counter() - t0) * 1000.0)

    return _median(lats), sorted(lats)


def measure_prefill_colocation(llm, tokenizer, decode_batch,
                                prefill_batch, prefill_seq, max_tokens,
                                num_runs, warmup):
    """测量共置场景下 Prefill 侧的时延退化。

    方法：提交 decode + prefill 混合请求，prefill 使用 max_tokens=1。
    通过 vLLM 的 use_tqdm=False 和 RequestOutput 的 finished_time 提取
    prefill 请求的完成时间。若 finished_time 不可用，则回退到近似方法：
    单独提交 prefill 请求（max_tokens=1）与 decode 请求竞争 GPU 资源。
    """
    from vllm import SamplingParams

    decode_prompts = ["The"] * decode_batch
    text = "hello " * (prefill_seq * 2)
    ids = tokenizer.encode(text)[:prefill_seq]
    prefill_prompt = tokenizer.decode(ids)
    prefill_prompts = [prefill_prompt] * prefill_batch

    for _ in range(warmup):
        llm.generate(prefill_prompts,
                     SamplingParams(max_tokens=1, temperature=0))

    lats = []
    for _ in range(num_runs):
        all_prompts = decode_prompts + prefill_prompts
        n_decode = len(decode_prompts)
        sp_list = ([SamplingParams(max_tokens=max_tokens, temperature=0)] * n_decode +
                   [SamplingParams(max_tokens=1, temperature=0)] * prefill_batch)

        t0 = time.perf_counter()
        outputs = llm.generate(all_prompts, sp_list)
        wall = time.perf_counter() - t0

        # 尝试从 RequestOutput.metrics 提取 prefill 请求的完成时间
        pfx_times = []
        for i, out in enumerate(outputs):
            if i >= n_decode:
                m = getattr(out, "metrics", None)
                if m and hasattr(m, "finished_time") and m.finished_time:
                    pfx_times.append(m.finished_time - m.arrival_time
                                     if hasattr(m, "arrival_time") and m.arrival_time
                                     else None)

        if pfx_times and all(t is not None for t in pfx_times):
            lats.append(max(pfx_times) * 1000.0)
        else:
            # 回退：用 prefill 请求的输出 token 数推断
            # prefill 请求只生成 1 token，在第一个 iteration 完成
            # 近似为 wall_time * (1 / max_tokens)
            lats.append(wall * 1000.0 / max_tokens)

    return _median(lats), sorted(lats)


# ── 主实验 ──────────────────────────────────────────────

def run_experiment(model_path, gpu_id, output_path,
                   batch_sizes=None, seq_lengths=None,
                   num_runs=5, warmup=2, max_tokens=32,
                   tp=1, max_model_len=4096):
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if seq_lengths is None:
        seq_lengths = DEFAULT_SEQ_LENGTHS

    # 仅在未从外部设置 CUDA_VISIBLE_DEVICES 时才设置
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"

    data = _load(output_path)
    model_name = Path(model_path).name

    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  GPU: {gpu_id}")
    print(f"  Batch sizes: {batch_sizes}, Seq lengths: {seq_lengths}")
    print(f"  Runs: {num_runs}, Warmup: {warmup}, Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading {model_name}...")
    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.85,
              max_model_len=max_model_len, tensor_parallel_size=tp)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.\n")

    # ── Step 1: Decode-only baselines ──
    if "baselines" not in data:
        print("Step 1: Decode-only baselines")
        baselines = {}
        for b in batch_sizes:
            key = f"b{b}"
            med, lats = measure_decode_only(llm, tokenizer, b, 0, max_tokens,
                                             num_runs, warmup)
            baselines[key] = {"batch_size": b, "median_ms": med, "all_ms": lats}
            print(f"  b={b}: {med:.1f} ms")
        data["baselines"] = baselines
        data["model"] = model_name
        _save(output_path, data)
        print()
    else:
        print("Step 1: Baselines (cached)\n")

    # ── Step 1b: Prefill-only baselines ──
    if "prefill_baselines" not in data:
        print("Step 1b: Prefill-only baselines")
        prefill_baselines = {}
        for b_p, s_p in product(batch_sizes, seq_lengths):
            key = f"b{b_p}_s{s_p}"
            med, lats = measure_prefill_only(llm, tokenizer, b_p, s_p,
                                              num_runs, warmup)
            prefill_baselines[key] = {
                "batch_size": b_p, "seq_len": s_p,
                "median_ms": med, "all_ms": lats
            }
            print(f"  b={b_p}, s={s_p}: {med:.1f} ms")
        data["prefill_baselines"] = prefill_baselines
        _save(output_path, data)
        print()
    else:
        print("Step 1b: Prefill baselines (cached)\n")

    # ── Step 2: PD co-location ──
    if "pairs" not in data:
        print("Step 2: PD co-location measurements")
        pairs = []
        total = len(batch_sizes) * len(batch_sizes) * len(seq_lengths)
        idx = 0

        for b_d in batch_sizes:
            baseline = data["baselines"][f"b{b_d}"]["median_ms"]

            for b_p, s_p in product(batch_sizes, seq_lengths):
                idx += 1
                key = f"d{b_d}_p{b_p}x{s_p}"
                print(f"  [{idx}/{total}] {key}...", end=" ", flush=True)

                coloc_med, coloc_lats = measure_pd_colocation(
                    llm, tokenizer,
                    decode_batch=b_d, decode_seq=0,
                    prefill_batch=b_p, prefill_seq=s_p,
                    max_tokens=max_tokens,
                    num_runs=num_runs, warmup=warmup)

                alpha_d = (coloc_med - baseline) / baseline if baseline > 0 else 0

                # α_p: prefill-side interference
                pfx_key = f"b{b_p}_s{s_p}"
                pfx_baseline = data["prefill_baselines"][pfx_key]["median_ms"]
                # Measure prefill time when co-located with decode
                pfx_coloc_med, pfx_coloc_lats = measure_prefill_colocation(
                    llm, tokenizer,
                    decode_batch=b_d, prefill_batch=b_p, prefill_seq=s_p,
                    max_tokens=max_tokens,
                    num_runs=num_runs, warmup=warmup)
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
                pairs.append(entry)
                print(f"α_d={alpha_d:.4f}  α_p={alpha_p:.4f}")

        data["pairs"] = pairs
        _save(output_path, data)
        print()
    else:
        print("Step 2: PD co-location (cached)\n")

    del llm
    _free_gpu()

    n = len(data.get("pairs", []))
    print(f"Done. {n} pairs → {output_path}")


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="同模型 PD 共置干扰实验")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "colocation.json"))
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    run_experiment(
        model_path=resolve_model_path(args.model),
        gpu_id=args.gpu,
        output_path=args.output,
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
