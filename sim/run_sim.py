"""CLI entry point for simulation experiments."""

import argparse
import json
import os
import time
from pathlib import Path

from sim.scenario import load_config, build_scenario
from workload.generator import WorkloadConfig


def run_single(scale, baseline, sim_config, workload_cfg, interference_table, output_dir):
    """Run a single simulation scenario and save results."""
    env, cluster, arrivals, requests = build_scenario(
        scale, baseline, sim_config, workload_cfg, interference_table,
    )

    print(f"  Running {scale}/{baseline} with {len(requests)} requests...")
    t0 = time.time()

    try:
        env.run()
    except Exception as e:
        print(f"  Simulation error: {e}")
        return None

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Extract results from cluster
    results = extract_results(cluster, requests)
    results["meta"] = {
        "scale": scale,
        "baseline": baseline,
        "num_requests": len(requests),
        "sim_elapsed_s": elapsed,
    }

    # Save
    out_path = Path(output_dir) / f"{scale}_{baseline}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")
    return results


def extract_results(cluster, requests):
    """Extract per-request metrics from completed simulation."""
    completed = []
    for req in requests:
        entry = {
            "request_id": req.id,
            "input_len": req.input_length,
            "output_len": req.output_length,
        }
        if hasattr(req, "ttft"):
            entry["ttft_ms"] = req.ttft
        if hasattr(req, "tpot"):
            entry["tpot_ms"] = req.tpot
        if hasattr(req, "finish_time") and hasattr(req, "arrival_time"):
            entry["total_latency_ms"] = req.finish_time - req.arrival_time
        if hasattr(req, "mode"):
            entry["mode"] = req.mode  # "colocate" or "disaggregate"
        completed.append(entry)
    return {"requests": completed}


def main():
    parser = argparse.ArgumentParser(description="Run CBS simulation experiments")
    parser.add_argument("--config", default="configs/sim_scenarios.yaml")
    parser.add_argument("--scale", default=None, help="Scale name (small/medium/large or all)")
    parser.add_argument("--baseline", default=None, help="Baseline name or 'all'")
    parser.add_argument("--workload", default="uniform", choices=["uniform", "bursty", "long_context"])
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--interference-table", default=None)
    parser.add_argument("--output-dir", default="results/sim")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim_config = load_config(args.config)

    # Build workload config
    wl_params = {
        "uniform": {"input_len_range": (128, 2048), "output_len_range": (32, 512)},
        "bursty": {"input_len_range": (128, 2048), "output_len_range": (32, 512)},
        "long_context": {"input_len_range": (1024, 4096), "output_len_range": (64, 256)},
    }
    arrival_type = "poisson_bursty" if args.workload == "bursty" else "poisson"
    wl_cfg = WorkloadConfig(
        arrival=arrival_type,
        rate=args.rate,
        duration_s=args.duration,
        seed=args.seed,
        **wl_params[args.workload],
    )

    # Determine which scenarios to run
    scales = list(sim_config["scales"].keys()) if args.scale in (None, "all") else [args.scale]
    baselines = list(sim_config["baselines"].keys()) if args.baseline in (None, "all") else [args.baseline]

    output_dir = os.path.join(args.output_dir, args.workload, f"rate_{args.rate}")

    print(f"Running {len(scales)} scales x {len(baselines)} baselines = {len(scales)*len(baselines)} scenarios")
    for scale in scales:
        for baseline in baselines:
            run_single(scale, baseline, sim_config, wl_cfg, args.interference_table, output_dir)

    print("All simulations complete.")


if __name__ == "__main__":
    main()
