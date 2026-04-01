"""CLI entry point for real LLM inference experiments.

Usage examples:
    python -m real.run_real --scenario disagg_2p2d --model qwen2.5-7b
    python -m real.run_real --scenario cbs_2p2d --model qwen2.5-7b --workload bursty --rate 6
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from real.benchmark import BenchmarkClient
from real.collect_metrics import MetricsCollector
from real.gateway import Gateway
from real.launch import ClusterLauncher
from workload.generator import WorkloadConfig, generate_workload

log = logging.getLogger("real")

_BASE = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _BASE / "configs"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _build_workload_config(
    wl_name: str, rate: float, duration: int, workloads_cfg: dict
) -> WorkloadConfig:
    """Map a workload name + overrides to a WorkloadConfig."""
    wl = workloads_cfg["workloads"].get(wl_name)
    if wl is None:
        raise ValueError(f"Unknown workload: {wl_name}")

    arrival = wl.get("arrival", "poisson")
    input_range = tuple(wl.get("input_len_range", [128, 2048]))
    output_range = tuple(wl.get("output_len_range", [32, 512]))

    kwargs: dict = dict(
        arrival=arrival,
        rate=rate,
        input_len_range=input_range,
        output_len_range=output_range,
        duration_s=float(duration),
    )
    if arrival == "poisson_bursty":
        kwargs["burst_multiplier"] = wl.get("burst_multiplier", 4.0)
        kwargs["burst_interval_s"] = wl.get("burst_interval_s", 30.0)
        kwargs["burst_duration_s"] = wl.get("burst_duration_s", 5.0)

    return WorkloadConfig(**kwargs)


async def _run_experiment(args: argparse.Namespace) -> None:
    # -- load configs ----------------------------------------------------------
    scenarios_cfg = _load_yaml(_CONFIG_DIR / "real_scenarios.yaml")
    models_cfg = _load_yaml(_CONFIG_DIR / "models.yaml")
    workloads_cfg = _load_yaml(_CONFIG_DIR / "workloads.yaml")

    scenario = scenarios_cfg["scenarios"].get(args.scenario)
    if scenario is None:
        sys.exit(f"Unknown scenario: {args.scenario}")
    model = models_cfg["models"].get(args.model)
    if model is None:
        sys.exit(f"Unknown model: {args.model}")

    slo = workloads_cfg.get("slo", {})
    slo_ttft = slo.get("ttft_ms", 2000)
    slo_tpot = slo.get("tpot_ms", 100)

    # -- output directory ------------------------------------------------------
    out_dir = Path(args.output_dir) / f"{args.scenario}_{args.model}_{args.workload}_r{args.rate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- generate workload -----------------------------------------------------
    wl_cfg = _build_workload_config(
        args.workload, args.rate, args.duration, workloads_cfg
    )
    requests = generate_workload(wl_cfg)
    log.info("Generated %d requests (%.1f req/s, %ds)", len(requests), args.rate, args.duration)

    # -- launch cluster --------------------------------------------------------
    launcher = ClusterLauncher(scenario, model)
    launcher.launch()
    try:
        launcher.wait_all_ready(timeout=args.launch_timeout)
    except TimeoutError:
        log.error("Cluster did not become healthy in time, aborting.")
        launcher.teardown()
        sys.exit(1)

    # -- start gateway ---------------------------------------------------------
    routing = scenario.get("routing", "round_robin")
    cbs_params = scenario.get("cbs_params")
    gateway = Gateway(
        prefill_urls=launcher.prefill_urls,
        decode_urls=launcher.decode_urls or launcher.all_urls,
        routing=routing,
        cbs_params=cbs_params,
    )
    gateway_port = args.gateway_port
    await gateway.start(port=gateway_port)
    gateway_url = f"http://localhost:{gateway_port}"

    # -- start metrics collector -----------------------------------------------
    collector = MetricsCollector(
        instance_urls=launcher.all_urls,
        interval_s=args.metrics_interval,
    )
    await collector.start()

    # -- run benchmark ---------------------------------------------------------
    log.info("Starting benchmark against %s ...", gateway_url)
    bench = BenchmarkClient(
        target_url=gateway_url,
        model_name=model["path"],
        workload_requests=requests,
        slo_ttft=slo_ttft,
        slo_tpot=slo_tpot,
    )
    await bench.run()

    # -- collect & save --------------------------------------------------------
    collector.stop()
    bench.save_results(out_dir / "results.json")
    collector.save(out_dir / "metrics.json")

    summary = bench.summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary: %s", json.dumps(summary, indent=2))

    # -- save experiment metadata ----------------------------------------------
    meta = {
        "scenario": args.scenario,
        "model": args.model,
        "workload": args.workload,
        "rate": args.rate,
        "duration": args.duration,
        "num_requests": len(requests),
        "routing": routing,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- teardown --------------------------------------------------------------
    await gateway.stop()
    launcher.teardown()
    log.info("Experiment complete. Results in %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real LLM inference experiment on 4x V100 GPUs."
    )
    parser.add_argument("--scenario", required=True,
                        choices=["disagg_2p2d", "coloc_4", "cbs_2p2d"],
                        help="Cluster topology / routing strategy")
    parser.add_argument("--model", required=True,
                        help="Model key from configs/models.yaml")
    parser.add_argument("--workload", default="uniform",
                        help="Workload key from configs/workloads.yaml")
    parser.add_argument("--rate", type=float, default=4.0,
                        help="Request arrival rate (req/s)")
    parser.add_argument("--duration", type=int, default=600,
                        help="Experiment duration in seconds")
    parser.add_argument("--output-dir", default="results/real",
                        help="Base output directory")
    parser.add_argument("--gateway-port", type=int, default=8080,
                        help="Port for the request gateway")
    parser.add_argument("--launch-timeout", type=int, default=300,
                        help="Max seconds to wait for vLLM instances")
    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Metrics scrape interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(_run_experiment(args))


if __name__ == "__main__":
    main()
