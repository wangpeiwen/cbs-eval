"""Build SimPy simulation scenarios from YAML configuration."""

import yaml
import simpy
from pathlib import Path
from typing import Dict, Any, Tuple, List

from workload.generator import WorkloadConfig, generate_workload, generate_arrivals


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_scenario(
    scale_name: str,
    baseline_name: str,
    sim_config: Dict,
    workload_cfg: WorkloadConfig,
    interference_table_path: str = None,
) -> Tuple[Any, Any, List[float], list]:
    """
    Build a SimPy environment and cluster from configuration.

    Returns: (env, cluster, arrivals, requests)
    """
    from sim.engine.params import DisaggRunParam, VLLMRunParam
    from sim.engine.request import Request as SimRequest

    scale = sim_config["scales"][scale_name]
    baseline = sim_config["baselines"][baseline_name]
    sim_params = sim_config["simulation"]

    n_prefill = scale["n_prefill"]
    n_decode = scale["n_decode"]

    # Generate workload
    wl_requests = generate_workload(workload_cfg)
    arrivals = generate_arrivals(wl_requests)

    # Convert to simulator request format
    sim_requests = []
    for r in wl_requests:
        sim_requests.append(SimRequest(
            id=r.request_id,
            input_length=r.input_len,
            output_length=r.output_len,
        ))

    env = simpy.Environment()

    scheduler_type = baseline.get("scheduler", "round_robin")

    if scheduler_type == "round_robin" and not baseline.get("enable_migration", False):
        # Pure disaggregated baseline
        from sim.engine.disagg_cluster import DisaggCluster
        from sim.engine.scheduler import Scheduler as RRScheduler

        param = DisaggRunParam(
            name=f"{scale_name}_{baseline_name}",
            arrival=arrivals,
            requests=sim_requests,
            N_prefill_instance=n_prefill,
            N_decode_instance=n_decode,
            PP_prefill=1, PP_decode=1,
            prefill_max_batch_size=sim_params.get("prefill_max_batch_size", 32),
            model_type=sim_params.get("model_type", "qwen2.5-7b"),
            TP_Prefill=1, TP_Decode=1,
            chunked_prefill_max_tokens=sim_params.get("chunked_prefill_max_tokens", 512),
        )
        cluster = DisaggCluster(env, param)

    elif scheduler_type == "cbs":
        # CBS-based scheduling (includes coloc_sarathi when mu=0)
        from sim.engine.cbs_cluster import CBSCluster
        from sim.engine.interference_model import InterferenceModel

        interference_model = InterferenceModel(
            table_path=interference_table_path
        ) if interference_table_path else InterferenceModel()

        param = DisaggRunParam(
            name=f"{scale_name}_{baseline_name}",
            arrival=arrivals,
            requests=sim_requests,
            N_prefill_instance=n_prefill,
            N_decode_instance=n_decode,
            PP_prefill=1, PP_decode=1,
            prefill_max_batch_size=sim_params.get("prefill_max_batch_size", 32),
            model_type=sim_params.get("model_type", "qwen2.5-7b"),
            TP_Prefill=1, TP_Decode=1,
            chunked_prefill_max_tokens=sim_params.get("chunked_prefill_max_tokens", 512),
        )

        cluster = CBSCluster(
            env, param,
            mu=baseline.get("mu", 2.0),
            lambda_ext=baseline.get("lambda_ext", 1.0),
            kappa_dispatch=baseline.get("kappa_dispatch", 0.1),
            interference_model=interference_model,
            enable_migration=baseline.get("enable_migration", False),
            enable_role_adaptation=baseline.get("enable_role_adaptation", False),
            kv_transfer_latency=sim_params.get("kv_transfer_latency_ms", 5.0),
            control_latency=sim_params.get("control_latency_ms", 2.0),
            slo_tpot=sim_params.get("slo_tpot_ms", 100.0),
            slo_ttft=sim_params.get("slo_ttft_ms", 2000.0),
            theta_ceil=baseline.get("theta_ceil", 0.3),
            theta_floor=baseline.get("theta_floor", 0.4),
            theta_dispatch=baseline.get("theta_dispatch", 0.85),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return env, cluster, arrivals, sim_requests
