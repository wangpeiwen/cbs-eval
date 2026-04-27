"""W2 Jitter Sensitivity Analysis.

Runs the CBS simulator under multiple jitter configurations and reports
goodput / SLO% / P99 latencies with 95% confidence intervals.

Usage:
    python -m analysis.w2_jitter_sensitivity --config 8node --n-seeds 10
"""

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim_paper import (SYSTEMS, CONFIGS, run_experiment_multi_seed)

JITTER_CONFIGS = {
    "deterministic": {"prefill": 0.0,  "decode": 0.0,  "kv": 0.0},
    "mild":          {"prefill": 0.05, "decode": 0.05, "kv": 0.10},
    "moderate":      {"prefill": 0.10, "decode": 0.10, "kv": 0.20},
    "severe":        {"prefill": 0.15, "decode": 0.15, "kv": 0.30},
}

METRICS = ["goodput", "slo_pct", "p99_ttft", "p99_tpot"]


def run_jitter_matrix(config_name: str, models: list, rates: list,
                      systems: list, n_seeds: int, duration_s: float,
                      pattern: str = "uniform") -> list:
    results = []
    combos = [(m, r, s, j) for m in models for r in rates
              for s in systems for j in JITTER_CONFIGS]
    total = len(combos)
    for i, (model, rate, system, jname) in enumerate(combos, 1):
        jcfg = JITTER_CONFIGS[jname]
        print(f"[{i}/{total}] {model} rate={rate} {system} jitter={jname} "
              f"x{n_seeds} ...", end=" ", flush=True)
        t0 = time.time()
        agg = run_experiment_multi_seed(config_name, model, rate, system,
                                        pattern, duration_s, n_seeds, jcfg)
        agg["jitter_name"] = jname
        results.append(agg)
        elapsed = time.time() - t0
        print(f"goodput={agg['goodput']}±{agg['goodput_ci']} ({elapsed:.1f}s)")
    return results


def print_comparison_table(results: list):
    """Print a table comparing deterministic vs jittered results."""
    # Group by (model, rate, system)
    from collections import defaultdict
    groups = defaultdict(dict)
    for r in results:
        key = (r["model"], r["rate"], r["system"])
        groups[key][r["jitter_name"]] = r

    print("\n" + "=" * 120)
    print("W2 Jitter Sensitivity: Goodput (mean ± 95% CI)")
    print("=" * 120)
    hdr = f"{'Model':<16} {'Rate':>5} {'System':<16}"
    for jname in JITTER_CONFIGS:
        hdr += f" {jname:>18}"
    print(hdr)
    print("-" * 120)

    for (model, rate, system), jmap in sorted(groups.items()):
        line = f"{model:<16} {rate:>5.0f} {system:<16}"
        det_gp = jmap.get("deterministic", {}).get("goodput", 0)
        for jname in JITTER_CONFIGS:
            r = jmap.get(jname, {})
            gp = r.get("goodput", 0)
            ci = r.get("goodput_ci", 0)
            delta = gp - det_gp if jname != "deterministic" else 0
            if jname == "deterministic":
                line += f" {gp:>6.2f}±{ci:<4.2f}     "
            else:
                sign = "+" if delta >= 0 else ""
                line += f" {gp:>6.2f}±{ci:<4.2f}({sign}{delta:.2f})"
            line = line.rstrip()
        print(line)

    # SLO% table
    print(f"\n{'Model':<16} {'Rate':>5} {'System':<16}", end="")
    for jname in JITTER_CONFIGS:
        print(f" {'SLO%':>8}", end="")
    print()
    print("-" * 100)
    for (model, rate, system), jmap in sorted(groups.items()):
        line = f"{model:<16} {rate:>5.0f} {system:<16}"
        for jname in JITTER_CONFIGS:
            r = jmap.get(jname, {})
            line += f" {r.get('slo_pct', 0):>7.1f}%"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="8node", choices=list(CONFIGS.keys()))
    parser.add_argument("--models", default="qwen2.5-7b,llama-3.1-8b")
    parser.add_argument("--rates", default="4,8,12,16")
    parser.add_argument("--systems", default="disagg_static,cbs_full")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--pattern", default="uniform")
    parser.add_argument("--output", default="results/w2_jitter_sensitivity.json")
    args = parser.parse_args()

    models = args.models.split(",")
    rates = [float(r) for r in args.rates.split(",")]
    systems = args.systems.split(",")

    results = run_jitter_matrix(args.config, models, rates, systems,
                                args.n_seeds, args.duration, args.pattern)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
