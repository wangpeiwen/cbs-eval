"""Compare results across different scheduling systems.

Expected directory layout::

    results_dir/
        disagg_static/
            rate_2.json
            rate_4.json
            ...
        coloc_sarathi/
            rate_2.json
            ...
        cbs_full/
            ...
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from analysis.metrics import compute_metrics, ExperimentMetrics


# Canonical column order for tables
_TABLE_COLS = [
    ("throughput", "Throughput (req/s)"),
    ("goodput", "Goodput (req/s)"),
    ("slo_attainment", "SLO Att. (\\%)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
]


def compare_systems(
    results_dir: str,
    systems: Optional[List[str]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
    rate: Optional[int] = None,
) -> Dict[str, ExperimentMetrics]:
    """Load results for each system and compute metrics.

    Parameters
    ----------
    results_dir : str
        Root directory containing one sub-directory per system.
    systems : list[str] | None
        System names to compare.  ``None`` discovers all sub-dirs.
    rate : int | None
        If given, only load ``rate_{rate}.json`` from each system dir.
        Otherwise load the first JSON found.
    """
    rdir = Path(results_dir)
    if systems is None:
        systems = sorted(
            d.name for d in rdir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    comparison: Dict[str, ExperimentMetrics] = {}
    for sys_name in systems:
        sys_dir = rdir / sys_name
        if not sys_dir.is_dir():
            continue
        # Pick the results file
        if rate is not None:
            candidate = sys_dir / f"rate_{rate}.json"
        else:
            jsons = sorted(sys_dir.glob("*.json"))
            candidate = jsons[0] if jsons else None
        if candidate is None or not candidate.exists():
            continue
        comparison[sys_name] = compute_metrics(
            str(candidate),
            slo_ttft=slo_ttft,
            slo_tpot=slo_tpot,
            warmup_s=warmup_s,
        )
    return comparison


def generate_latex_table(
    comparison: Dict[str, ExperimentMetrics],
    caption: str = "System comparison",
    label: str = "tab:comparison",
    system_labels: Optional[Dict[str, str]] = None,
) -> str:
    """Generate a LaTeX tabularx table from comparison results.

    Parameters
    ----------
    comparison : dict
        Mapping of system name -> ExperimentMetrics.
    caption, label : str
        LaTeX caption and label.
    system_labels : dict | None
        Pretty-print names for systems.  Falls back to the key itself.
    """
    if system_labels is None:
        from plot.style import LABELS
        system_labels = LABELS

    cols = _TABLE_COLS
    n_data_cols = len(cols)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    col_spec = "l " + "r " * n_data_cols
    lines.append(rf"\begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append(r"\toprule")

    header = "System & " + " & ".join(h for _, h in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for sys_name, metrics in comparison.items():
        pretty = system_labels.get(sys_name, sys_name)
        md = metrics.to_dict()
        vals = []
        for key, _ in cols:
            v = md[key]
            if "attainment" in key:
                vals.append(f"{v:.1f}")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(f"{pretty} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_summary(comparison: Dict[str, ExperimentMetrics]) -> None:
    """Print a formatted comparison table to stdout."""
    from plot.style import LABELS

    header_keys = [k for k, _ in _TABLE_COLS]
    header_labels = [h.replace("\\%", "%") for _, h in _TABLE_COLS]

    # Column widths
    name_w = max(len(LABELS.get(s, s)) for s in comparison) + 2
    col_w = max(len(h) for h in header_labels) + 2

    # Header
    print(f"{'System':<{name_w}}", end="")
    for h in header_labels:
        print(f"{h:>{col_w}}", end="")
    print()
    print("-" * (name_w + col_w * len(header_labels)))

    # Rows
    for sys_name, metrics in comparison.items():
        pretty = LABELS.get(sys_name, sys_name)
        md = metrics.to_dict()
        print(f"{pretty:<{name_w}}", end="")
        for key in header_keys:
            v = md[key]
            if "attainment" in key:
                print(f"{v:>{col_w}.1f}", end="")
            elif isinstance(v, float):
                print(f"{v:>{col_w}.2f}", end="")
            else:
                print(f"{v:>{col_w}}", end="")
        print()
