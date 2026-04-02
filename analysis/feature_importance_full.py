"""MLWD 完整特征集的特征重要性分析。

读取 mlwd_complete.json + colocation.json，用全部可用特征预测 α_d。

Usage:
    python3 -m analysis.feature_importance_full \
        --mlwd results/qwen2.5-7b-mlwd_complete.json \
        --colocation results/qwen2.5-7b-colocation.json \
        --output results/qwen-feature-importance-full.json
"""

import argparse, json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# 从 mlwd_complete 中提取的特征（跳过 None 字段）
CANDIDATE_FEATURES = [
    "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",  # 干扰敏感度
    "ci_attn", "ci_ffn",                              # 资源竞争强度
    "r_attn", "r_ffn",                                # 执行模式（从 CI 推算）
    "l2_attn", "l2_ffn",                              # L2 命中率（理论估算）
    "ipc",                                            # IPC（理论估算）
    "baseline_ms",                                    # 基线时延
]


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_dataset(
    mlwd: Dict, colocation: Dict,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """构建 (X, y) 数据集。

    X: [victim_features, aggressor_features]
    y: α_d
    """
    pairs = colocation.get("pairs", [])
    X_rows, y_rows, pair_keys = [], [], []

    for pair in pairs:
        alpha_d = pair.get("alpha_d")
        if alpha_d is None:
            continue

        b_d = pair["decode_batch"]
        b_p = pair["prefill_batch"]
        s_p = pair["prefill_seq"]

        # victim = decode 端的 MLWD
        # decode 用短 seq（共置实验中 decode prompt = "The"，实际 seq 由 max_tokens 决定）
        victim_key = f"b{b_d}_s32_decode"
        # aggressor = prefill 端的 MLWD
        aggressor_key = f"b{b_p}_s{s_p}_prefill"

        victim = mlwd.get(victim_key)
        aggressor = mlwd.get(aggressor_key)
        if victim is None or aggressor is None:
            continue

        v_feats = _extract(victim)
        a_feats = _extract(aggressor)
        if v_feats is None or a_feats is None:
            continue

        X_rows.append(np.concatenate([v_feats, a_feats]))
        y_rows.append(alpha_d)
        pair_keys.append(pair.get("key", ""))

    # 特征名
    feat_names = [f"v_{f}" for f in CANDIDATE_FEATURES] + [f"a_{f}" for f in CANDIDATE_FEATURES]
    return np.array(X_rows), np.array(y_rows), pair_keys, feat_names


def _extract(entry: Dict) -> Optional[np.ndarray]:
    feats = []
    for f in CANDIDATE_FEATURES:
        val = entry.get(f)
        if val is None:
            val = 0.0  # 缺失字段用 0 填充
        feats.append(float(val))
    return np.array(feats)


def fit_ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """OLS 拟合，返回 (weights, MAE, R²)。"""
    X_b = np.column_stack([X, np.ones(len(X))])
    w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    y_pred = X_b @ w
    mae = np.mean(np.abs(y - y_pred))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return w, mae, r2


def leave_one_out(X, y, feat_names):
    _, base_mae, base_r2 = fit_ols(X, y)
    results = []
    for i in range(X.shape[1]):
        X_abl = np.delete(X, i, axis=1)
        _, abl_mae, abl_r2 = fit_ols(X_abl, y)
        delta = abl_mae - base_mae
        results.append({
            "name": feat_names[i],
            "ablated_mae": round(abl_mae, 6),
            "ablated_r2": round(abl_r2, 4),
            "delta_mae": round(delta, 6),
            "importance_pct": round(delta / base_mae * 100, 2) if base_mae > 0 else 0,
        })
    results.sort(key=lambda x: -x["importance_pct"])
    return {"baseline_mae": round(base_mae, 6), "baseline_r2": round(base_r2, 4), "features": results}


def forward_selection(X, y, feat_names, threshold=1.05):
    n_feat = X.shape[1]
    _, full_mae, _ = fit_ols(X, y)
    selected, remaining = [], list(range(n_feat))
    history = []

    for step in range(n_feat):
        best_mae, best_idx = float("inf"), None
        for idx in remaining:
            trial = selected + [idx]
            _, mae, _ = fit_ols(X[:, trial], y)
            if mae < best_mae:
                best_mae, best_idx = mae, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        _, _, r2 = fit_ols(X[:, selected], y)
        history.append({
            "step": step + 1,
            "added": feat_names[best_idx],
            "selected": [feat_names[i] for i in selected],
            "mae": round(best_mae, 6),
            "r2": round(r2, 4),
        })

    # 找最小子集
    min_subset = None
    for h in history:
        if h["mae"] <= full_mae * threshold:
            min_subset = h
            break

    return {"history": history, "recommended_subset": min_subset, "full_mae": round(full_mae, 6)}


def print_report(loo, fwd, n_samples, n_features):
    print(f"\n{'='*65}")
    print(f"MLWD 完整特征集重要性分析（{n_samples} 样本, {n_features} 特征）")
    print(f"{'='*65}")
    print(f"\n全特征 MAE: {loo['baseline_mae']:.6f}, R²: {loo['baseline_r2']:.4f}")

    print(f"\n── Leave-One-Out 重要性排序 ──")
    print(f"{'特征':<20} {'去掉后MAE':>10} {'ΔMAE':>10} {'重要性%':>10} {'去掉后R²':>8}")
    print("-" * 62)
    for f in loo["features"]:
        marker = " ***" if f["importance_pct"] > 10 else " **" if f["importance_pct"] > 3 else " *" if f["importance_pct"] > 1 else ""
        print(f"{f['name']:<20} {f['ablated_mae']:>10.6f} {f['delta_mae']:>+10.6f} {f['importance_pct']:>9.2f}%{marker} {f['ablated_r2']:>8.4f}")

    print(f"\n── 前向选择 ──")
    print(f"{'步骤':>4} {'加入特征':<20} {'MAE':>10} {'R²':>8}")
    print("-" * 46)
    for h in fwd["history"]:
        print(f"{h['step']:>4} {h['added']:<20} {h['mae']:>10.6f} {h['r2']:>8.4f}")

    if fwd["recommended_subset"]:
        rs = fwd["recommended_subset"]
        print(f"\n推荐最小子集（MAE ≤ 全特征 105%）：")
        print(f"  特征数: {rs['step']}/{n_features}")
        print(f"  MAE: {rs['mae']:.6f} (全特征: {fwd['full_mae']:.6f})")
        print(f"  R²: {rs['r2']:.4f}")
        print(f"  特征: {rs['selected']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlwd", required=True, help="mlwd_complete.json")
    parser.add_argument("--colocation", required=True, help="colocation.json")
    parser.add_argument("--output", default="results/feature-importance-full.json")
    args = parser.parse_args()

    mlwd = load_json(args.mlwd)
    coloc = load_json(args.colocation)

    X, y, keys, feat_names = build_dataset(mlwd, coloc)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"α_d range: [{y.min():.4f}, {y.max():.4f}], mean: {y.mean():.4f}")

    if len(X) < X.shape[1]:
        print(f"WARNING: samples ({len(X)}) < features ({X.shape[1]}), results may be unstable")

    loo = leave_one_out(X, y, feat_names)
    fwd = forward_selection(X, y, feat_names)
    print_report(loo, fwd, len(X), X.shape[1])

    result = {"leave_one_out": loo, "forward_selection": fwd,
              "dataset": {"n_samples": len(X), "n_features": X.shape[1],
                          "alpha_d_range": [float(y.min()), float(y.max())],
                          "alpha_d_mean": float(y.mean())}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
