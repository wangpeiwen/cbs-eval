"""MLWD 特征重要性分析与降维。

基于干扰预测的特征选择：用 MLWD 特征预测 α_d/α_p，
逐个去掉特征看 MAE 变化（leave-one-out ablation）。

Usage:
    python -m analysis.feature_importance \
        --sensitivity results/qwen-2.5-7b-sensitivity.json \
        --colocation results/qwen2.5-7b-colocation.json \
        --output results/feature_importance.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

# MLWD 第一层 15 个特征名
MLWD_FEATURES = [
    "ci_attn", "ci_ffn",           # 资源竞争强度
    "l2_attn", "l2_ffn",
    "t_attn", "t_ffn",             # 执行模式
    "g_launch",
    "sigma_bs", "sigma_cu",        # 干扰敏感度
    "sigma_l2", "sigma_bw",
    "r_attn", "r_ffn",
    "f_switch",
    "ipc",
]

# 按语义分组（用于分组 ablation）
FEATURE_GROUPS = {
    "资源竞争强度": ["ci_attn", "ci_ffn", "l2_attn", "l2_ffn"],
    "干扰敏感度": ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"],
    "执行模式": ["t_attn", "t_ffn", "g_launch", "r_attn", "r_ffn", "f_switch"],
    "Pipeline饱和度": ["ipc"],
}


def load_sensitivity(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_colocation(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("pairs", [])


def build_dataset(
    sensitivity: Dict,
    colocation: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """从 sensitivity + colocation 数据构建 (X, y) 数据集。

    X: 每行是 [victim_features, aggressor_features] 拼接
    y: α_d 真值

    Returns: (X, y, pair_keys)
    """
    X_rows, y_rows, keys = [], [], []

    for pair in colocation:
        alpha_d = pair.get("alpha_d")
        if alpha_d is None:
            continue

        # victim = decode 请求的 MLWD 特征
        b_d = pair["decode_batch"]
        # aggressor = prefill 请求的 MLWD 特征
        b_p = pair["prefill_batch"]
        s_p = pair["prefill_seq"]

        # 从 sensitivity 中查找最近的特征
        victim_key = f"b{b_d}_s32_decode"  # decode 用短 seq
        aggressor_key = f"b{b_p}_s{s_p}_prefill"

        victim = sensitivity.get(victim_key)
        aggressor = sensitivity.get(aggressor_key)

        if victim is None or aggressor is None:
            continue

        # 提取可用特征（sensitivity 中有的子集）
        v_feats = _extract_features(victim)
        a_feats = _extract_features(aggressor)

        if v_feats is None or a_feats is None:
            continue

        X_rows.append(np.concatenate([v_feats, a_feats]))
        y_rows.append(alpha_d)
        keys.append(pair.get("key", ""))

    return np.array(X_rows), np.array(y_rows), keys


def _extract_features(entry: Dict) -> np.ndarray:
    """从 sensitivity entry 中提取可用特征向量。

    sensitivity.json 中直接可用的特征：sigma_bs, sigma_cu, sigma_l2, sigma_bw, baseline_ms
    其他特征（ci, t_attn 等）需要从 ci.json / nsys.json 获取，这里用 baseline_ms 作为代理。
    """
    feats = []
    for f in ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw", "baseline_ms"]:
        val = entry.get(f)
        if val is None:
            return None
        feats.append(val)
    return np.array(feats)


# 特征名（victim + aggressor 各 5 个 = 10 个）
AVAILABLE_FEATURES = [
    "v_sigma_bs", "v_sigma_cu", "v_sigma_l2", "v_sigma_bw", "v_baseline",
    "a_sigma_bs", "a_sigma_cu", "a_sigma_l2", "a_sigma_bw", "a_baseline",
]


def fit_linear(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS 拟合，返回 (weights, MAE)。"""
    # 加 bias 列
    X_b = np.column_stack([X, np.ones(len(X))])
    w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    y_pred = X_b @ w
    mae = np.mean(np.abs(y - y_pred))
    return w, mae


def leave_one_out_ablation(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Dict:
    """逐个去掉特征，看 MAE 变化。"""
    n_features = X.shape[1]
    assert n_features == len(feature_names)

    # 全特征 baseline
    _, baseline_mae = fit_linear(X, y)

    results = {"baseline_mae": baseline_mae, "features": []}

    for i in range(n_features):
        # 去掉第 i 个特征
        X_ablated = np.delete(X, i, axis=1)
        _, ablated_mae = fit_linear(X_ablated, y)
        delta = ablated_mae - baseline_mae
        importance = delta / baseline_mae * 100  # 百分比

        results["features"].append({
            "name": feature_names[i],
            "ablated_mae": round(ablated_mae, 6),
            "delta_mae": round(delta, 6),
            "importance_pct": round(importance, 2),
        })

    # 按重要性排序
    results["features"].sort(key=lambda x: -x["importance_pct"])
    return results


def group_ablation(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Dict:
    """按语义分组去掉特征，看 MAE 变化。"""
    _, baseline_mae = fit_linear(X, y)

    # 映射 AVAILABLE_FEATURES 到分组
    groups = {
        "干扰敏感度(victim)": ["v_sigma_bs", "v_sigma_cu", "v_sigma_l2", "v_sigma_bw"],
        "干扰敏感度(aggressor)": ["a_sigma_bs", "a_sigma_cu", "a_sigma_l2", "a_sigma_bw"],
        "时延代理(victim)": ["v_baseline"],
        "时延代理(aggressor)": ["a_baseline"],
    }

    results = {"baseline_mae": baseline_mae, "groups": []}

    for group_name, group_feats in groups.items():
        indices = [i for i, f in enumerate(feature_names) if f in group_feats]
        if not indices:
            continue
        X_ablated = np.delete(X, indices, axis=1)
        _, ablated_mae = fit_linear(X_ablated, y)
        delta = ablated_mae - baseline_mae
        importance = delta / baseline_mae * 100

        results["groups"].append({
            "name": group_name,
            "features": group_feats,
            "ablated_mae": round(ablated_mae, 6),
            "delta_mae": round(delta, 6),
            "importance_pct": round(importance, 2),
        })

    results["groups"].sort(key=lambda x: -x["importance_pct"])
    return results


def forward_selection(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    max_features: int = None,
) -> Dict:
    """前向特征选择：从空集开始，每次加入使 MAE 降低最多的特征。"""
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))
    history = []

    for step in range(min(max_features, n_features)):
        best_mae = float("inf")
        best_feat = None

        for feat_idx in remaining:
            trial = selected + [feat_idx]
            X_sub = X[:, trial]
            _, mae = fit_linear(X_sub, y)
            if mae < best_mae:
                best_mae = mae
                best_feat = feat_idx

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append({
            "step": step + 1,
            "added_feature": feature_names[best_feat],
            "selected_features": [feature_names[i] for i in selected],
            "mae": round(best_mae, 6),
        })

    return {"history": history}


def print_report(loo: Dict, group: Dict, fwd: Dict):
    """打印分析报告。"""
    print("\n" + "=" * 60)
    print("MLWD 特征重要性分析报告")
    print("=" * 60)

    print(f"\n全特征 MAE: {loo['baseline_mae']:.6f}")

    print("\n── Leave-One-Out 单特征重要性 ──")
    print(f"{'特征':<20} {'去掉后MAE':>10} {'ΔMAE':>10} {'重要性%':>10}")
    print("-" * 52)
    for f in loo["features"]:
        marker = " ***" if f["importance_pct"] > 5 else " *" if f["importance_pct"] > 1 else ""
        print(f"{f['name']:<20} {f['ablated_mae']:>10.6f} {f['delta_mae']:>+10.6f} {f['importance_pct']:>9.2f}%{marker}")

    print("\n── 分组重要性 ──")
    print(f"{'组':<25} {'去掉后MAE':>10} {'ΔMAE':>10} {'重要性%':>10}")
    print("-" * 57)
    for g in group["groups"]:
        print(f"{g['name']:<25} {g['ablated_mae']:>10.6f} {g['delta_mae']:>+10.6f} {g['importance_pct']:>9.2f}%")

    print("\n── 前向选择（最优特征子集） ──")
    print(f"{'步骤':>4} {'加入特征':<20} {'MAE':>10} {'已选特征'}")
    print("-" * 70)
    for h in fwd["history"]:
        print(f"{h['step']:>4} {h['added_feature']:<20} {h['mae']:>10.6f} {h['selected_features']}")

    # 找到 MAE 接近全特征 95% 的最小子集
    full_mae = loo["baseline_mae"]
    for h in fwd["history"]:
        if h["mae"] <= full_mae * 1.05:
            print(f"\n推荐最小特征子集（MAE ≤ 全特征的 105%）：")
            print(f"  特征数: {h['step']}/{len(loo['features'])}")
            print(f"  特征: {h['selected_features']}")
            print(f"  MAE: {h['mae']:.6f} (全特征: {full_mae:.6f})")
            break


def main():
    parser = argparse.ArgumentParser(description="MLWD 特征重要性分析")
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--colocation", required=True)
    parser.add_argument("--output", default="results/feature_importance.json")
    args = parser.parse_args()

    sensitivity = load_sensitivity(args.sensitivity)
    colocation = load_colocation(args.colocation)

    print(f"Loaded {len(sensitivity)} sensitivity entries, {len(colocation)} colocation pairs")

    X, y, keys = build_dataset(sensitivity, colocation)
    print(f"Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"α_d range: [{y.min():.4f}, {y.max():.4f}], mean: {y.mean():.4f}")

    if len(X) < 5:
        print("Too few samples for meaningful analysis. Need more colocation data.")
        return

    # 三种分析
    loo = leave_one_out_ablation(X, y, AVAILABLE_FEATURES)
    group = group_ablation(X, y, AVAILABLE_FEATURES)
    fwd = forward_selection(X, y, AVAILABLE_FEATURES)

    print_report(loo, group, fwd)

    # 保存
    result = {
        "leave_one_out": loo,
        "group_ablation": group,
        "forward_selection": fwd,
        "dataset_info": {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "alpha_d_range": [float(y.min()), float(y.max())],
            "alpha_d_mean": float(y.mean()),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
