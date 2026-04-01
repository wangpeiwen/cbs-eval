"""Hyperparameter sensitivity analysis.

Analyses the effect of key CBS hyperparameters on goodput and SLO attainment
by loading results from sweep directories.

Expected layout for a lambda sweep::

    results_dir/
        lambda_0.5/
            rate_4.json
        lambda_1.0/
            rate_4.json
        ...

Similar layouts for threshold and mu sweeps.
"""

from pathlib import Path
from typing import Dict, List, Optional

from analysis.metrics import compute_metrics


def _sweep_results(
    results_dir: str,
    prefix: str,
    values: List,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Generic helper: load results for a parameter sweep.

    Looks for directories named ``{prefix}_{value}`` under *results_dir*.
    Returns a dict mapping each value to its ExperimentMetrics (as a dict).
    """
    rdir = Path(results_dir)
    out: Dict = {"values": [], "goodput": [], "slo_attainment": [],
                 "throughput": [], "p99_ttft_ms": [], "p99_tpot_ms": [],
                 "metrics": {}}

    for v in values:
        # Support both "lambda_0.5" and "lambda_0.50" naming
        candidates = [
            rdir / f"{prefix}_{v}",
            rdir / f"{prefix}_{v:.1f}" if isinstance(v, float) else None,
            rdir / f"{prefix}_{v:.2f}" if isinstance(v, float) else None,
        ]
        sys_dir = None
        for c in candidates:
            if c is not None and c.is_dir():
                sys_dir = c
                break
        if sys_dir is None:
            continue

        jsons = sorted(sys_dir.glob("*.json"))
        if not jsons:
            continue

        m = compute_metrics(
            str(jsons[0]),
            slo_ttft=slo_ttft,
            slo_tpot=slo_tpot,
            warmup_s=warmup_s,
        )
        out["values"].append(v)
        out["goodput"].append(m.goodput)
        out["slo_attainment"].append(m.slo_attainment)
        out["throughput"].append(m.throughput)
        out["p99_ttft_ms"].append(m.p99_ttft_ms)
        out["p99_tpot_ms"].append(m.p99_tpot_ms)
        out["metrics"][v] = m.to_dict()

    return out


def lambda_sensitivity(
    results_dir: str,
    lambda_values: Optional[List[float]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Analyze effect of lambda_ext on goodput and SLO attainment.

    Parameters
    ----------
    results_dir : str
        Directory containing ``lambda_{value}/`` sub-directories.
    lambda_values : list[float] | None
        Values to sweep.  Defaults to [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0].
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    return _sweep_results(
        results_dir, "lambda", lambda_values,
        slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s,
    )


def threshold_sensitivity(
    results_dir: str,
    theta_ceil_values: Optional[List[float]] = None,
    theta_floor_values: Optional[List[float]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Analyze effect of theta_ceil and theta_floor on performance.

    Runs two independent sweeps and returns both under keys
    ``"theta_ceil"`` and ``"theta_floor"``.
    """
    if theta_ceil_values is None:
        theta_ceil_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    if theta_floor_values is None:
        theta_floor_values = [0.2, 0.3, 0.4, 0.5, 0.6]

    return {
        "theta_ceil": _sweep_results(
            results_dir, "theta_ceil", theta_ceil_values,
            slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s,
        ),
        "theta_floor": _sweep_results(
            results_dir, "theta_floor", theta_floor_values,
            slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s,
        ),
    }


def mu_sensitivity(
    results_dir: str,
    mu_values: Optional[List[float]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Analyze effect of mu (colocation threshold) on performance.

    Parameters
    ----------
    mu_values : list[float] | None
        Defaults to [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0].
    """
    if mu_values is None:
        mu_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    return _sweep_results(
        results_dir, "mu", mu_values,
        slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s,
    )
