"""Ablation study analysis.

Compares CBS-Full against variants with individual components disabled to
quantify each component's contribution.

Macro ablations (Section 6.4.1):
    CBS-Full  vs  CBS-NoMig   -> migration contribution
    CBS-Full  vs  CBS-NoRole  -> role-adaptation contribution

Component ablations (Section 6.4.2):
    CBS-Full  vs  CBS-NoDispatch  -> dispatch-aware scheduling
    CBS-Full  vs  CBS-NoRisk      -> risk-budget migration trigger
    CBS-Full  vs  CBS-NoBudget    -> budget-constrained migration
"""

from pathlib import Path
from typing import Dict, List, Optional

from analysis.metrics import compute_metrics, ExperimentMetrics


_MACRO_ABLATIONS = ["cbs_nomig", "cbs_norole"]
_COMPONENT_ABLATIONS = ["cbs_nodispatch", "cbs_norisk", "cbs_nobudget"]


def _find_result(results_dir: Path, system: str) -> Optional[str]:
    """Return the path to the first JSON result for *system*, or None."""
    sys_dir = results_dir / system
    if not sys_dir.is_dir():
        return None
    jsons = sorted(sys_dir.glob("*.json"))
    return str(jsons[0]) if jsons else None


def _delta(base: float, variant: float) -> float:
    """Relative change from *base* to *variant* as a percentage."""
    if base == 0:
        return 0.0
    return (variant - base) / base * 100.0


def ablation_analysis(
    results_dir: str,
    base_system: str = "cbs_full",
    ablation_systems: Optional[List[str]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Compare CBS-Full vs macro ablation variants.

    Returns a dict keyed by ablation system name, each containing the
    variant's metrics plus delta_goodput and delta_slo_attainment relative
    to the base system.
    """
    rdir = Path(results_dir)
    if ablation_systems is None:
        ablation_systems = _MACRO_ABLATIONS

    base_path = _find_result(rdir, base_system)
    if base_path is None:
        raise FileNotFoundError(
            f"No results found for base system '{base_system}' in {rdir}"
        )
    base_metrics = compute_metrics(
        base_path, slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s
    )

    results: Dict = {
        "__base__": {
            "system": base_system,
            "metrics": base_metrics.to_dict(),
        }
    }

    for sys_name in ablation_systems:
        path = _find_result(rdir, sys_name)
        if path is None:
            continue
        m = compute_metrics(
            path, slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s
        )
        results[sys_name] = {
            "metrics": m.to_dict(),
            "delta_goodput_pct": _delta(base_metrics.goodput, m.goodput),
            "delta_slo_attainment_pct": _delta(
                base_metrics.slo_attainment, m.slo_attainment
            ),
            "delta_throughput_pct": _delta(base_metrics.throughput, m.throughput),
            "delta_p99_ttft_pct": _delta(base_metrics.p99_ttft_ms, m.p99_ttft_ms),
            "delta_p99_tpot_pct": _delta(base_metrics.p99_tpot_ms, m.p99_tpot_ms),
        }

    return results


def cbs_component_ablation(
    results_dir: str,
    base_system: str = "cbs_full",
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> Dict:
    """Compare CBS-Full vs fine-grained component ablation variants.

    Variants:
        cbs_nodispatch -- disable dispatch-aware scheduling
        cbs_norisk     -- disable risk-budget migration trigger
        cbs_nobudget   -- disable budget-constrained migration
    """
    return ablation_analysis(
        results_dir,
        base_system=base_system,
        ablation_systems=_COMPONENT_ABLATIONS,
        slo_ttft=slo_ttft,
        slo_tpot=slo_tpot,
        warmup_s=warmup_s,
    )
