"""基线对比实验：SM-Only / Profile-MLP / MLWD 三种方法的干扰系数估算精度。

输出：MAE、R² (训练集 + 5折交叉验证)，用于论文表 tab:interference-accuracy。

Usage:
    python -m analysis.baseline_comparison
"""

import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parent.parent / "results"

# ── 数据加载 ─────────────────────────────────────────────────

def load_all_samples():
    """加载三种模型的共置实验数据和 MLWD 特征，返回统一的样本列表。"""
    models = [
        ("qwen2.5-7b", "qwen2.5-7b-colocation.json", "qwen2.5-7b-mlwd_complete.json"),
        ("llama-3.1-8b", "llama-3.1-8b-colocation.json", "llama-3.1-8b-mlwd_complete.json"),
        ("qwen3-14b", "qwen3-14b-colocation.json", "qwen3-14b-mlwd_complete.json"),
    ]
    samples = []
    for model_name, coloc_file, mlwd_file in models:
        coloc_path = RESULTS / coloc_file
        mlwd_path = RESULTS / mlwd_file
        if not coloc_path.exists() or not mlwd_path.exists():
            print(f"[WARN] 跳过 {model_name}: 文件不存在")
            continue
        with open(coloc_path) as f:
            coloc = json.load(f)
        with open(mlwd_path) as f:
            mlwd = json.load(f)
        for pair in coloc.get("pairs", []):
            if pair.get("alpha_d") is None:
                continue
            # victim = decode, aggressor = prefill
            v_key = f"b{pair['victim_b']}_s{pair['victim_s']}_decode"
            a_key = f"b{pair['aggressor_b']}_s{pair['aggressor_s']}_prefill"
            v_entry = mlwd.get(v_key)
            a_entry = mlwd.get(a_key)
            if v_entry is None or a_entry is None:
                continue
            samples.append({
                "model": model_name,
                "alpha_d": pair["alpha_d"],
                "victim": v_entry,
                "aggressor": a_entry,
            })
    return samples


# ── 特征构建 ─────────────────────────────────────────────────

MLWD_FIELDS = [
    "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",
    "ci_attn", "ci_ffn", "r_attn", "r_ffn", "l2_attn", "l2_ffn",
]

def _get(entry, field, default=0.0):
    v = entry.get(field)
    return v if v is not None else default


def build_mlwd_features(samples):
    """MLWD: victim 10维 + aggressor 10维 = 20维。"""
    X = []
    for s in samples:
        row = [_get(s["victim"], f) for f in MLWD_FIELDS]
        row += [_get(s["aggressor"], f) for f in MLWD_FIELDS]
        X.append(row)
    return np.array(X)


def build_sm_only_features(samples):
    """SM-Only: 用 CI 和 r_attn/r_ffn 作为 SM 利用率代理。

    victim 和 aggressor 各取 ci_attn*r_attn + ci_ffn*r_ffn 作为
    SM 利用率的近似（实际 SM util 未直接采集，用 CI 加权时间占比代替）。
    """
    X = []
    for s in samples:
        v = s["victim"]
        a = s["aggressor"]
        v_sm = _get(v, "ci_attn") * _get(v, "r_attn") + _get(v, "ci_ffn") * _get(v, "r_ffn")
        a_sm = _get(a, "ci_attn") * _get(a, "r_attn") + _get(a, "ci_ffn") * _get(a, "r_ffn")
        X.append([v_sm, a_sm])
    return np.array(X)


def build_profile_mlp_features(samples):
    """Profile-MLP: SM利用率代理 + 显存带宽代理 + L2命中率 (victim+aggressor共6维)。"""
    X = []
    for s in samples:
        v, a = s["victim"], s["aggressor"]
        v_sm = _get(v, "ci_attn") * _get(v, "r_attn") + _get(v, "ci_ffn") * _get(v, "r_ffn")
        a_sm = _get(a, "ci_attn") * _get(a, "r_attn") + _get(a, "ci_ffn") * _get(a, "r_ffn")
        v_bw = (1 - _get(v, "l2_attn")) * _get(v, "r_attn") + (1 - _get(v, "l2_ffn")) * _get(v, "r_ffn")
        a_bw = (1 - _get(a, "l2_attn")) * _get(a, "r_attn") + (1 - _get(a, "l2_ffn")) * _get(a, "r_ffn")
        v_l2 = _get(v, "l2_attn") * _get(v, "r_attn") + _get(v, "l2_ffn") * _get(v, "r_ffn")
        a_l2 = _get(a, "l2_attn") * _get(a, "r_attn") + _get(a, "l2_ffn") * _get(a, "r_ffn")
        X.append([v_sm, a_sm, v_bw, a_bw, v_l2, a_l2])
    return np.array(X)


# ── 评估 ─────────────────────────────────────────────────────

def evaluate(name, X, y, model_cls, model_kwargs=None, n_splits=5):
    """训练集拟合 + 5折交叉验证。"""
    model_kwargs = model_kwargs or {}

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练集拟合
    model = model_cls(**model_kwargs)
    model.fit(X_scaled, y)
    y_train_pred = model.predict(X_scaled)
    train_mae = mean_absolute_error(y, y_train_pred)
    train_r2 = r2_score(y, y_train_pred)

    # 5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_preds = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        m = model_cls(**model_kwargs)
        m.fit(X_tr, y[train_idx])
        cv_preds[test_idx] = m.predict(X_te)

    cv_mae = mean_absolute_error(y, cv_preds)
    cv_r2 = r2_score(y, cv_preds)

    return {
        "name": name,
        "train_mae": round(train_mae, 4),
        "train_r2": round(train_r2, 4),
        "cv_mae": round(cv_mae, 4),
        "cv_r2": round(cv_r2, 4),
    }


def main():
    samples = load_all_samples()
    n = len(samples)
    print(f"共加载 {n} 组共置样本")

    y = np.array([s["alpha_d"] for s in samples])
    print(f"α_d 范围: [{y.min():.4f}, {y.max():.4f}], 均值: {y.mean():.4f}")

    # 按模型统计
    from collections import Counter
    model_counts = Counter(s["model"] for s in samples)
    for m, c in model_counts.items():
        print(f"  {m}: {c} 组")

    # 构建特征
    X_sm = build_sm_only_features(samples)
    X_mlp = build_profile_mlp_features(samples)
    X_mlwd = build_mlwd_features(samples)

    print(f"\n特征维度: SM-Only={X_sm.shape[1]}, Profile-MLP={X_mlp.shape[1]}, MLWD={X_mlwd.shape[1]}")

    # 评估三种方法
    results = []

    r_sm = evaluate("SM-Only", X_sm, y,
                    LinearRegression)
    results.append(r_sm)

    r_mlp = evaluate("Profile-MLP", X_mlp, y,
                     MLPRegressor,
                     {"hidden_layer_sizes": (8,), "max_iter": 5000,
                      "random_state": 42, "alpha": 0.1,
                      "learning_rate": "adaptive",
                      "solver": "lbfgs"})
    results.append(r_mlp)

    r_mlwd = evaluate("MLWD", X_mlwd, y,
                      Ridge, {"alpha": 0.1})
    results.append(r_mlwd)

    # 输出结果
    print("\n" + "=" * 70)
    print(f"{'方法':<15} {'训练MAE':>10} {'训练R²':>10} {'5折CV MAE':>12} {'5折CV R²':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<15} {r['train_mae']:>10.4f} {r['train_r2']:>10.4f} "
              f"{r['cv_mae']:>12.4f} {r['cv_r2']:>10.4f}")
    print("=" * 70)

    # 精度提升
    sm_cv_mae = r_sm["cv_mae"]
    mlwd_cv_mae = r_mlwd["cv_mae"]
    if sm_cv_mae > 0:
        improvement = (sm_cv_mae - mlwd_cv_mae) / sm_cv_mae * 100
        print(f"\nMLWD 相比 SM-Only 的 CV MAE 降低: {improvement:.1f}%")

    # 按模型拆分 MLWD 结果
    print("\n--- MLWD 按模型拆分 (训练集) ---")
    scaler = StandardScaler()
    X_mlwd_scaled = scaler.fit_transform(X_mlwd)
    model = Ridge(alpha=0.1)
    model.fit(X_mlwd_scaled, y)
    y_pred = model.predict(X_mlwd_scaled)

    for model_name in model_counts:
        mask = np.array([s["model"] == model_name for s in samples])
        mae_m = mean_absolute_error(y[mask], y_pred[mask])
        print(f"  {model_name}: MAE = {mae_m:.4f} (n={mask.sum()})")

    # 保存结果
    output = {
        "n_samples": n,
        "methods": {r["name"]: r for r in results},
        "per_model_mlwd_train_mae": {},
    }
    for model_name in model_counts:
        mask = np.array([s["model"] == model_name for s in samples])
        mae_m = mean_absolute_error(y[mask], y_pred[mask])
        output["per_model_mlwd_train_mae"][model_name] = round(mae_m, 4)

    out_path = RESULTS / "baseline_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
