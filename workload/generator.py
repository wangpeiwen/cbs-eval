"""Unified workload generator for both real and simulation experiments."""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Literal


@dataclass
class WorkloadConfig:
    arrival: Literal["poisson", "poisson_bursty", "gamma"]
    rate: float  # req/s
    input_len_range: Tuple[int, int] = (128, 2048)
    output_len_range: Tuple[int, int] = (32, 512)
    # Bursty params
    burst_multiplier: float = 4.0
    burst_interval_s: float = 30.0
    burst_duration_s: float = 5.0
    # Gamma params
    cv: float = 2.0
    # Duration
    duration_s: float = 600.0
    seed: int = 42


@dataclass
class Request:
    request_id: int
    arrival_time: float  # seconds
    input_len: int
    output_len: int


def generate_workload(cfg: WorkloadConfig) -> List[Request]:
    """Generate a list of requests with arrival times and sequence lengths."""
    rng = random.Random(cfg.seed)
    requests = []
    t = 0.0
    rid = 0

    while t < cfg.duration_s:
        # Determine current rate
        if cfg.arrival == "poisson_bursty":
            cycle_pos = t % cfg.burst_interval_s
            if cycle_pos >= (cfg.burst_interval_s - cfg.burst_duration_s):
                current_rate = cfg.rate * cfg.burst_multiplier
            else:
                current_rate = cfg.rate
        else:
            current_rate = cfg.rate

        # Sample inter-arrival time
        if cfg.arrival in ("poisson", "poisson_bursty"):
            iat = rng.expovariate(current_rate)
        elif cfg.arrival == "gamma":
            shape = 1.0 / (cfg.cv ** 2)
            scale = 1.0 / (current_rate * shape)
            iat = rng.gammavariate(shape, scale)
        else:
            raise ValueError(f"Unknown arrival type: {cfg.arrival}")

        t += iat
        if t >= cfg.duration_s:
            break

        input_len = rng.randint(*cfg.input_len_range)
        output_len = rng.randint(*cfg.output_len_range)
        requests.append(Request(rid, t, input_len, output_len))
        rid += 1

    return requests


def generate_arrivals(requests: List[Request]) -> List[float]:
    """Extract inter-arrival times from request list (for simulator compatibility)."""
    if not requests:
        return []
    arrivals = [requests[0].arrival_time]
    for i in range(1, len(requests)):
        arrivals.append(requests[i].arrival_time - requests[i - 1].arrival_time)
    return arrivals
