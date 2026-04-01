"""Synthetic workload distributions."""

from typing import List
from workload.generator import WorkloadConfig, Request, generate_workload


def uniform_workload(rate: float, duration_s: float = 600.0, seed: int = 42) -> List[Request]:
    return generate_workload(WorkloadConfig(
        arrival="poisson", rate=rate,
        input_len_range=(128, 2048), output_len_range=(32, 512),
        duration_s=duration_s, seed=seed,
    ))


def bursty_workload(base_rate: float, duration_s: float = 600.0, seed: int = 42) -> List[Request]:
    return generate_workload(WorkloadConfig(
        arrival="poisson_bursty", rate=base_rate,
        input_len_range=(128, 2048), output_len_range=(32, 512),
        burst_multiplier=4.0, burst_interval_s=30.0, burst_duration_s=5.0,
        duration_s=duration_s, seed=seed,
    ))


def long_context_workload(rate: float, duration_s: float = 600.0, seed: int = 42) -> List[Request]:
    return generate_workload(WorkloadConfig(
        arrival="poisson", rate=rate,
        input_len_range=(1024, 4096), output_len_range=(64, 256),
        duration_s=duration_s, seed=seed,
    ))
