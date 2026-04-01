"""Compute TTFT, TPOT, goodput, SLO attainment from request logs.

Expected JSON format (list of request records):
[
  {
    "req_id": 0,
    "arrival_ts": 0.5,
    "first_token_ts": 0.62,
    "end_ts": 1.45,
    "input_len": 512,
    "output_len": 128,
    "ttft_ms": 120.0,
    "tpot_ms": 6.5,
    "completed": true
  },
  ...
]

If the file instead contains a top-level dict with a "requests" key, that list
is used.  Fields ``ttft_ms`` / ``tpot_ms`` may be absent -- they will be
derived from timestamps when missing.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union


@dataclass
class ExperimentMetrics:
    total_requests: int
    completed_requests: int
    goodput: float          # req/s -- only SLO-compliant requests
    slo_attainment: float   # percentage 0-100
    mean_ttft_ms: float
    p50_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    p50_tpot_ms: float
    p99_tpot_ms: float
    throughput: float       # req/s -- all completed requests

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_requests(results_path: str) -> list:
    """Load request records from a JSON file."""
    path = Path(results_path)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("requests", data.get("results", []))
    return data


def _derive_ttft(req: dict) -> float:
    """Return TTFT in ms, deriving from timestamps if needed."""
    if "ttft_ms" in req and req["ttft_ms"] is not None:
        return float(req["ttft_ms"])
    arrival = req.get("arrival_ts", req.get("arrival_time", 0.0))
    first = req.get("first_token_ts", req.get("first_token_time", arrival))
    return (first - arrival) * 1000.0


def _derive_tpot(req: dict) -> float:
    """Return TPOT in ms, deriving from timestamps if needed."""
    if "tpot_ms" in req and req["tpot_ms"] is not None:
        return float(req["tpot_ms"])
    first = req.get("first_token_ts", req.get("first_token_time"))
    end = req.get("end_ts", req.get("end_time"))
    output_len = req.get("output_len", req.get("output_length", 1))
    if first is None or end is None or output_len <= 1:
        return 0.0
    return (end - first) * 1000.0 / max(output_len - 1, 1)


def _filter_warmup(requests: list, warmup_s: float) -> list:
    """Remove requests that arrived during the warmup window."""
    if warmup_s <= 0 or not requests:
        return requests
    t0 = min(
        r.get("arrival_ts", r.get("arrival_time", 0.0)) for r in requests
    )
    cutoff = t0 + warmup_s
    return [
        r for r in requests
        if r.get("arrival_ts", r.get("arrival_time", 0.0)) >= cutoff
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_goodput(
    requests: list,
    slo_ttft: float,
    slo_tpot: float,
    duration_s: float,
) -> float:
    """Count requests meeting both SLO constraints divided by duration."""
    if duration_s <= 0:
        return 0.0
    compliant = sum(
        1 for r in requests
        if r.get("completed", True)
        and _derive_ttft(r) <= slo_ttft
        and _derive_tpot(r) <= slo_tpot
    )
    return compliant / duration_s


def compute_metrics(
    results_path: str,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> ExperimentMetrics:
    """Compute all evaluation metrics from a results JSON file.

    Parameters
    ----------
    results_path : str
        Path to the JSON file with request-level results.
    slo_ttft : float
        TTFT SLO threshold in milliseconds.
    slo_tpot : float
        TPOT SLO threshold in milliseconds.
    warmup_s : float
        Seconds of warmup to discard from the beginning.
    """
    all_requests = _load_requests(results_path)
    requests = _filter_warmup(all_requests, warmup_s)

    total = len(requests)
    completed = [r for r in requests if r.get("completed", True)]
    n_completed = len(completed)

    if n_completed == 0:
        return ExperimentMetrics(
            total_requests=total,
            completed_requests=0,
            goodput=0.0,
            slo_attainment=0.0,
            mean_ttft_ms=0.0,
            p50_ttft_ms=0.0,
            p99_ttft_ms=0.0,
            mean_tpot_ms=0.0,
            p50_tpot_ms=0.0,
            p99_tpot_ms=0.0,
            throughput=0.0,
        )

    ttfts = np.array([_derive_ttft(r) for r in completed])
    tpots = np.array([_derive_tpot(r) for r in completed])

    # Duration: time span covered by the (post-warmup) completed requests
    arrivals = [
        r.get("arrival_ts", r.get("arrival_time", 0.0)) for r in completed
    ]
    ends = [r.get("end_ts", r.get("end_time", 0.0)) for r in completed]
    duration_s = max(ends) - min(arrivals) if len(arrivals) > 1 else 1.0
    duration_s = max(duration_s, 1e-6)

    slo_ok = np.sum((ttfts <= slo_ttft) & (tpots <= slo_tpot))

    return ExperimentMetrics(
        total_requests=total,
        completed_requests=n_completed,
        goodput=float(slo_ok) / duration_s,
        slo_attainment=float(slo_ok) / n_completed * 100.0,
        mean_ttft_ms=float(np.mean(ttfts)),
        p50_ttft_ms=float(np.percentile(ttfts, 50)),
        p99_ttft_ms=float(np.percentile(ttfts, 99)),
        mean_tpot_ms=float(np.mean(tpots)),
        p50_tpot_ms=float(np.percentile(tpots, 50)),
        p99_tpot_ms=float(np.percentile(tpots, 99)),
        throughput=float(n_completed) / duration_s,
    )
