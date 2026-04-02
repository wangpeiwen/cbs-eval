"""合并 sensitivity + ci → mlwd_complete.json（不依赖 nsys）。

从 ci.json 推算 r_attn/r_ffn，其余 nsys 字段留空。

Usage:
    python3 -m mlwd.merge_simple \
        --sensitivity results/qwen-2.5-7b-sensitivity.json \
        --ci results/qwen2.5-7B-ci.json \
        --model qwen2.5-7b \
        --output results/qwen2.5-7b-mlwd_complete.json
"""

import argparse, json, os
from pathlib import Path
from .config import resolve_model_path, get_model_params


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _estimate_l2(entry, mp):
    """粗略估算 L2 命中率：基于工作集大小 vs L2 容量。"""
    from .config import V100_L2_BYTES
    b = entry.get("batch_size", 1)
    s = entry.get("seq_len", 128)
    phase = entry.get("phase", "prefill")
    h, d, nkv, L = mp["hidden"], mp["head_dim"], mp["kv_heads"], mp["layers"]

    # Attention 工作集：KV Cache
    kv_bytes = 2 * L * nkv * d * s * 2 * b  # fp16
    l2_attn = max(0.0, min(1.0, V100_L2_BYTES / max(kv_bytes, 1)))

    # FFN 工作集：权重（一层）
    inter = mp["inter"]
    ffn_weight_bytes = 3 * h * inter * 2  # 3 matrices, fp16
    l2_ffn = max(0.0, min(1.0, V100_L2_BYTES / max(ffn_weight_bytes, 1)))

    return round(l2_attn, 4), round(l2_ffn, 4)


def _estimate_ipc(ci):
    """粗略估算 IPC：CI 越高越接近计算瓶颈，IPC 越高。"""
    # V100 理论峰值 IPC ~4.0（每 SM 每周期 4 条 warp 指令）
    # 简单映射：IPC ≈ min(ci / 10, 1.0) * 2.0
    return round(min(ci / 10.0, 1.0) * 2.0, 4)


def main():
    parser = argparse.ArgumentParser(description="合并 sensitivity + ci → mlwd_complete（无 nsys）")
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--ci", required=True)
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sens = _load(args.sensitivity)
    ci = _load(args.ci)

    try:
        mp = get_model_params(resolve_model_path(args.model))
    except ValueError:
        mp = None

    print(f"Loaded: sensitivity={len(sens)} entries, ci={len(ci)} entries")

    # 从 sensitivity 中提取所有 (b, s, phase) 组合
    complete = {}

    for key, sv in sens.items():
        # key = "b1_s32_prefill" or "b1_s32_decode"
        parts = key.split("_")
        if len(parts) < 3:
            continue
        b = sv.get("batch_size")
        s = sv.get("seq_len")
        phase = sv.get("phase")
        if b is None or s is None or phase is None:
            continue

        entry = {"batch_size": b, "seq_len": s, "phase": phase}

        # ── sensitivity 字段 ──
        for f in ["baseline_ms", "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]:
            if f in sv:
                entry[f] = sv[f]

        # ── ci 字段 ──
        ci_key = f"b{b}_s{s}"
        if ci_key in ci:
            cv = ci[ci_key]
            for f in ["ci_attn", "ci_ffn", "attn_tflops", "ffn_tflops"]:
                if f in cv and cv[f] is not None:
                    entry[f] = cv[f]

            # 推算 r_attn, r_ffn（从累计时延比例）
            attn_t = cv.get("attn_time_us", 0)
            ffn_t = cv.get("ffn_time_us", 0)
            total_t = attn_t + ffn_t
            if total_t > 0:
                entry["r_attn"] = round(attn_t / total_t, 4)
                entry["r_ffn"] = round(ffn_t / total_t, 4)

        # ── 理论估算 ──
        if mp:
            l2_attn, l2_ffn = _estimate_l2(entry, mp)
            entry["l2_attn"] = l2_attn
            entry["l2_ffn"] = l2_ffn

            avg_ci = (entry.get("ci_attn", 1.0) + entry.get("ci_ffn", 1.0)) / 2
            entry["ipc"] = _estimate_ipc(avg_ci)

        # ── 不可采字段留空 ──
        for f in ["t_attn", "t_ffn", "g_launch", "f_switch"]:
            entry.setdefault(f, None)

        # ── 完整性标记 ──
        has_sens = all(entry.get(f"sigma_{d}") is not None for d in ["bs", "cu", "l2", "bw"])
        has_ci = entry.get("ci_ffn") is not None
        entry["complete"] = has_sens and has_ci

        complete[key] = entry

    # 统计
    n = len(complete)
    nc = sum(1 for v in complete.values() if v["complete"])
    print(f"\n{nc}/{n} entries complete (sensitivity + ci)")

    for key in sorted(complete.keys()):
        v = complete[key]
        tag = "OK" if v["complete"] else "--"
        svals = [v.get(f'sigma_{d}') for d in ['bs','cu','l2','bw']]
        sigma = "σ=[" + ",".join(f"{x:.2f}" if isinstance(x, (int, float)) else "?" for x in svals) + "]"
        ci_a, ci_f = v.get('ci_attn'), v.get('ci_ffn')
        ci_str = f"CI=[{ci_a},{ci_f}]" if ci_a is not None else "CI=missing"
        print(f"  {key:25s} {tag}  {sigma}  {ci_str}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(complete, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
