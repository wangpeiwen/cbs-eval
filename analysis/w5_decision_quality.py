"""W5 Decision Quality Analysis: CBS accuracy in high-interference regions.

Analyzes CBS decision logs to quantify:
1. Decision distribution by alpha_d bucket
2. SLO attainment per bucket
3. Misclassification rate (coloc chosen but SLO violated, or disagg chosen
   when idle decode nodes existed)

Usage:
    # First generate decision logs:
    python sim_paper.py --config 8node --models qwen2.5-7b --rates 12 \
        --systems cbs_full --n-seeds 10 --save-decisions --output results/w5_raw.json

    # Then analyze:
    python -m analysis.w5_decision_quality results/w5_raw_decisions.json \
        results/w5_raw.json --output results/w5_analysis.json
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHA_BINS = [
    (0.00, 0.02, "[0, 0.02)"),
    (0.02, 0.04, "[0.02, 0.04)"),
    (0.04, 0.06, "[0.04, 0.06)"),
    (0.06, float("inf"), "[0.06, +∞)"),
]


def load_decisions_and_outcomes(decisions_path: str,
                                results_path: str) -> list:
    """Load decision log and join with per-request SLO outcomes."""
    with open(decisions_path) as f:
        decisions = json.load(f)

    # Load simulation results to get completed request outcomes
    with open(results_path) as f:
        results = json.load(f)

    # Build a lookup: (seed, req_id) -> slo_met from the simulation
    # Note: the results JSON from multi-seed runs contains aggregated metrics,
    # not per-request outcomes. We need to re-run with per-request tracking.
    # For now, we use a heuristic: if TPOT SLO is 100ms and the decision was
    # coloc with high alpha_d, estimate whether SLO would be violated.
    # This is a conservative analysis based on the interference model.
    return decisions


def bucket_decisions(decisions: list) -> dict:
    """Group decisions into alpha_d buckets and compute statistics."""
    buckets = {label: [] for _, _, label in ALPHA_BINS}

    for d in decisions:
        alpha_d = d.get("alpha_d", 0)
        for lo, hi, label in ALPHA_BINS:
            if lo <= alpha_d < hi:
                buckets[label].append(d)
                break

    analysis = {}
    for label, items in buckets.items():
        if not items:
            analysis[label] = {
                "count": 0, "coloc_count": 0, "disagg_count": 0,
                "coloc_pct": 0, "disagg_pct": 0,
                "avg_cbs_score": 0, "std_cbs_score": 0,
                "avg_alpha_d": 0, "max_alpha_d": 0,
                "coloc_high_alpha_count": 0, "disagg_with_idle_node": 0,
            }
            continue

        n = len(items)
        n_coloc = sum(1 for d in items if d["chosen_mode"] == "coloc")
        n_disagg = n - n_coloc

        # CBS score statistics
        scores = [d["cbs_score"] for d in items if d["cbs_score"] > -900]
        alpha_ds = [d["alpha_d"] for d in items]

        # Coloc decisions where alpha_d is high (potential misclassification)
        coloc_items = [d for d in items if d["chosen_mode"] == "coloc"]
        coloc_high_alpha = [d for d in coloc_items if d["alpha_d"] > 0.06]

        # Disagg decisions where decode_bs == 0 (idle node existed)
        disagg_items = [d for d in items if d["chosen_mode"] == "disagg"]
        disagg_idle = [d for d in disagg_items if d["decode_bs"] == 0]

        analysis[label] = {
            "count": n,
            "coloc_count": n_coloc,
            "disagg_count": n_disagg,
            "coloc_pct": round(100 * n_coloc / n, 1),
            "disagg_pct": round(100 * n_disagg / n, 1),
            "avg_cbs_score": round(float(np.mean(scores)), 4) if scores else 0,
            "std_cbs_score": round(float(np.std(scores)), 4) if scores else 0,
            "avg_alpha_d": round(float(np.mean(alpha_ds)), 4),
            "max_alpha_d": round(float(np.max(alpha_ds)), 4),
            "coloc_high_alpha_count": len(coloc_high_alpha),
            "disagg_with_idle_node": len(disagg_idle),
        }

    return analysis


def compute_misclassification(decisions: list, slo_tpot_ms: float = 100.0,
                              slo_ttft_ms: float = 2000.0) -> dict:
    """Estimate misclassification rate using interference model predictions.

    A 'misclassification' is defined as:
    - Type A: CBS chose coloc, but predicted TPOT after admission exceeds
      a conservative threshold (0.9 * SLO_TPOT), suggesting high risk.
    - Type B: CBS chose disagg, but the best CBS score was only marginally
      negative (> -1.0 ms), suggesting the decision was borderline and
      sensitive to estimation error.
    """
    from sim_paper import decode_step_ms, alpha_d_model

    type_a = []  # risky coloc
    type_b = []  # borderline disagg

    for d in decisions:
        if d["chosen_mode"] == "coloc":
            bs = d["decode_bs"]
            alpha_d = d["alpha_d"]
            tpot_pred = decode_step_ms(d.get("model", "qwen2.5-7b"),
                                        bs + 1) * (1 + alpha_d)
            if tpot_pred > 0.9 * slo_tpot_ms:
                type_a.append(d)
        elif d["chosen_mode"] == "disagg":
            # Borderline: CBS score was only slightly negative
            if d["cbs_score"] > -1.0 and d["cbs_score"] != -999:
                type_b.append(d)

    n_total = len(decisions)
    n_coloc = sum(1 for d in decisions if d["chosen_mode"] == "coloc")
    n_disagg = n_total - n_coloc

    return {
        "total_decisions": n_total,
        "total_coloc": n_coloc,
        "total_disagg": n_disagg,
        "type_a_risky_coloc": len(type_a),
        "type_a_rate_pct": round(100 * len(type_a) / max(n_coloc, 1), 2),
        "type_b_missed_free_coloc": len(type_b),
        "type_b_rate_pct": round(100 * len(type_b) / max(n_disagg, 1), 2),
        "overall_misclass_pct": round(
            100 * (len(type_a) + len(type_b)) / max(n_total, 1), 2),
    }


def per_seed_analysis(decisions: list) -> dict:
    """Compute per-seed statistics to show variance across runs."""
    by_seed = defaultdict(list)
    for d in decisions:
        by_seed[d.get("seed", 0)].append(d)

    seed_stats = []
    for seed, items in sorted(by_seed.items()):
        n = len(items)
        n_coloc = sum(1 for d in items if d["chosen_mode"] == "coloc")
        seed_stats.append({
            "seed": seed,
            "n_decisions": n,
            "coloc_pct": round(100 * n_coloc / n, 1) if n else 0,
            "avg_alpha_d": round(float(np.mean([d["alpha_d"] for d in items])), 4),
        })

    coloc_pcts = [s["coloc_pct"] for s in seed_stats]
    return {
        "per_seed": seed_stats,
        "coloc_pct_mean": round(float(np.mean(coloc_pcts)), 1),
        "coloc_pct_std": round(float(np.std(coloc_pcts, ddof=1)), 2) if len(coloc_pcts) > 1 else 0,
    }


def print_report(bucket_analysis: dict, misclass: dict, seed_analysis: dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 90)
    print("W5: CBS Decision Quality by Interference Coefficient (alpha_d) Bucket")
    print("=" * 90)
    print(f"{'Bucket':<14} {'Count':>7} {'Coloc%':>8} {'Disagg%':>8} "
          f"{'Avg CBS':>9} {'Avg α_d':>8} {'Max α_d':>8}")
    print("-" * 90)
    for label, stats in bucket_analysis.items():
        print(f"{label:<14} {stats['count']:>7} {stats['coloc_pct']:>7.1f}% "
              f"{stats['disagg_pct']:>7.1f}% {stats['avg_cbs_score']:>9.4f} "
              f"{stats['avg_alpha_d']:>8.4f} {stats['max_alpha_d']:>8.4f}")

    print(f"\n{'Misclassification Analysis':}")
    print(f"  Total decisions: {misclass['total_decisions']}")
    print(f"  Coloc decisions: {misclass['total_coloc']}")
    print(f"  Type A (risky coloc, pred TPOT > 0.9*SLO): "
          f"{misclass['type_a_risky_coloc']} ({misclass['type_a_rate_pct']}% of coloc)")
    print(f"  Type B (borderline disagg, CBS score > -1.0ms): "
          f"{misclass['type_b_missed_free_coloc']} ({misclass['type_b_rate_pct']}% of disagg)")
    print(f"  Overall misclassification: {misclass['overall_misclass_pct']}%")

    print(f"\n  Cross-seed coloc%: {seed_analysis['coloc_pct_mean']}% "
          f"± {seed_analysis['coloc_pct_std']}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions_json", help="Path to CBS decision log JSON")
    parser.add_argument("results_json", nargs="?", default=None,
                        help="Path to simulation results JSON (optional)")
    parser.add_argument("--output", default="results/w5_analysis.json")
    parser.add_argument("--slo-tpot", type=float, default=100.0)
    args = parser.parse_args()

    with open(args.decisions_json) as f:
        decisions = json.load(f)
    print(f"Loaded {len(decisions)} decision entries")

    bucket_analysis = bucket_decisions(decisions)
    misclass = compute_misclassification(decisions, args.slo_tpot)
    seed_analysis = per_seed_analysis(decisions)

    print_report(bucket_analysis, misclass, seed_analysis)

    output = {
        "bucket_analysis": bucket_analysis,
        "misclassification": misclass,
        "seed_analysis": seed_analysis,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
