"""CLI entry point for simulation experiments."""

import argparse
import json
import os
import time
from pathlib import Path

from sim.scenario import load_config, build_scenario
from sim.engine.scheduler import put_requests_with_interarrivals
from workload.generator import WorkloadConfig


def run_single(scale, baseline, sim_config, workload_cfg, interference_table, output_dir):
    """Run a single simulation scenario and save results."""
    env, cluster, arrivals, requests = build_scenario(
        scale, baseline, sim_config, workload_cfg, interference_table,
    )

    print(f"  Running {scale}/{baseline} with {len(requests)} requests...")
    t0 = time.time()

    try:
        cluster.run()
        put_requests_with_interarrivals(env, cluster.scheduler, arrivals, requests)
        env.run()
    except Exception as e:
        print(f"  Simulation error: {e}")
        raise

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Extract results from cluster
    results = extract_results(cluster, requests)
    results["meta"] = {
        "scale": scale,
        "baseline": baseline,
        "workload": workload_cfg.arrival,
        "rate": workload_cfg.rate,
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
        arrival_ms = req.arrival_time
        first_ms = req.first_token_time
        end_ms = req.finish_time
        completed_ok = end_ms is not None
        ttft_ms = (first_ms - arrival_ms) if first_ms is not None and arrival_ms is not None else None
        tpot_ms = None
        if completed_ok and first_ms is not None:
            if req.output_lens <= 1:
                tpot_ms = 0.0
            else:
                tpot_ms = (end_ms - first_ms) / max(req.output_lens - 1, 1)

        entry = {
            "request_id": req.req_id,
            "input_len": req.prefill_lens,
            "output_len": req.output_lens,
            "arrival_ts": arrival_ms / 1000.0 if arrival_ms is not None else None,
            "first_token_ts": first_ms / 1000.0 if first_ms is not None else None,
            "end_ts": end_ms / 1000.0 if end_ms is not None else None,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "completed": completed_ok,
            "mode": "colocate" if getattr(req, "is_colocated", False) else "disaggregate",
        }
        if completed_ok and arrival_ms is not None:
            entry["total_latency_ms"] = end_ms - arrival_ms
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
