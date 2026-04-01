"""Bridge MLWD JSON data to simulator interference table format."""

import json
from pathlib import Path
from typing import Dict, List


def load_mlwd_data(mlwd_json_path: str) -> Dict:
    """Load MLWD complete JSON from mlwd-collector output."""
    with open(mlwd_json_path) as f:
        return json.load(f)


def build_interference_table(
    mlwd_data: Dict,
    weights: Dict = None,
    decode_batch_sizes: List[int] = None,
    prefill_lengths: List[int] = None,
) -> Dict:
    """
    Convert MLWD data + calibrated weights into the interference table
    format expected by sim.engine.interference_model.InterferenceModel.

    Output format:
    {
        "model_name": {
            "entries": [
                {"decode_bs": 1, "prefill_len": 128, "alpha_p": 0.03, "alpha_d": 0.05},
                ...
            ]
        }
    }
    """
    if decode_batch_sizes is None:
        decode_batch_sizes = [1, 2, 4, 8, 16]
    if prefill_lengths is None:
        prefill_lengths = [32, 64, 128, 256, 512, 1024, 2048]

    if weights is None:
        # Default weights from OLS calibration
        weights = {
            "w_bs": 0.15, "w_cu": 0.30, "w_l2": 0.25,
            "w_bw": 0.20, "w_ipc": 0.05, "w_overlap": 0.05,
        }

    entries = []
    for dbs in decode_batch_sizes:
        for plen in prefill_lengths:
            alpha_p = _estimate_alpha_p(mlwd_data, weights, dbs, plen)
            alpha_d = _estimate_alpha_d(mlwd_data, weights, dbs, plen)
            entries.append({
                "decode_bs": dbs,
                "prefill_len": plen,
                "alpha_p": round(alpha_p, 6),
                "alpha_d": round(alpha_d, 6),
            })

    return entries


def _get_mlwd_entry(mlwd_data: Dict, batch_size: int, seq_len: int, phase: str) -> Dict:
    """Lookup nearest MLWD entry for given (b, s, phase)."""
    key = f"b{batch_size}_s{seq_len}_{phase}"
    if key in mlwd_data:
        return mlwd_data[key]
    # Nearest neighbor fallback
    best_key, best_dist = None, float("inf")
    for k, v in mlwd_data.items():
        if not k.endswith(f"_{phase}"):
            continue
        b = v.get("batch_size", 0)
        s = v.get("seq_len", 0)
        dist = abs(b - batch_size) * 1000 + abs(s - seq_len)
        if dist < best_dist:
            best_dist = dist
            best_key = k
    return mlwd_data.get(best_key, {})


def _estimate_alpha_d(mlwd_data: Dict, weights: Dict, decode_bs: int, prefill_len: int) -> float:
    """Estimate decode interference coefficient using MLWD weighted mapping rule."""
    decode_entry = _get_mlwd_entry(mlwd_data, decode_bs, 128, "decode")
    prefill_entry = _get_mlwd_entry(mlwd_data, 1, prefill_len, "prefill")

    if not decode_entry or not prefill_entry:
        return 0.0

    # Victim sensitivity (decode)
    sigma_bs = decode_entry.get("sigma_bs", 0)
    sigma_cu = decode_entry.get("sigma_cu", 0)
    sigma_l2 = decode_entry.get("sigma_l2", 0)
    sigma_bw = decode_entry.get("sigma_bw", 0)

    # Aggressor strength (prefill)
    ci_attn = prefill_entry.get("ci_attn", 1.0)
    ci_ffn = prefill_entry.get("ci_ffn", 1.0)
    l2_attn = prefill_entry.get("l2_attn", 0.5)
    l2_ffn = prefill_entry.get("l2_ffn", 0.5)
    g_launch = prefill_entry.get("g_launch", 100.0)
    r_attn = prefill_entry.get("r_attn", 0.3)
    r_ffn = prefill_entry.get("r_ffn", 0.5)
    ipc = prefill_entry.get("ipc", 0.5)

    A_bs = min(1.0 / max(g_launch, 1.0) * 1000, 1.0)
    A_cu = (ci_attn * r_attn + ci_ffn * r_ffn) / max(r_attn + r_ffn, 0.01)
    A_cu = min(A_cu / 10.0, 1.0)  # normalize
    A_l2 = ((1 - l2_attn) * r_attn + (1 - l2_ffn) * r_ffn) / max(r_attn + r_ffn, 0.01)
    A_bw = 1.0 / max(A_cu * 10.0, 0.1)
    A_bw = min(A_bw, 1.0)

    # Overlap factor
    r_decode = decode_entry.get("r_attn", 0.3) + decode_entry.get("r_ffn", 0.5)
    r_prefill = r_attn + r_ffn
    omega = min(r_decode, r_prefill)

    alpha = (
        weights["w_bs"] * sigma_bs * A_bs
        + weights["w_cu"] * sigma_cu * A_cu
        + weights["w_l2"] * sigma_l2 * A_l2
        + weights["w_bw"] * sigma_bw * A_bw
        + weights["w_ipc"] * ipc * sigma_cu
        + weights["w_overlap"] * omega
    )

    # Dilution by decode batch size
    alpha = alpha / max(decode_bs, 1)
    return max(0.0, min(alpha, 0.5))


def _estimate_alpha_p(mlwd_data: Dict, weights: Dict, decode_bs: int, prefill_len: int) -> float:
    """Estimate prefill interference coefficient."""
    prefill_entry = _get_mlwd_entry(mlwd_data, 1, prefill_len, "prefill")
    decode_entry = _get_mlwd_entry(mlwd_data, decode_bs, 128, "decode")

    if not prefill_entry or not decode_entry:
        return 0.0

    sigma_bs = prefill_entry.get("sigma_bs", 0)
    sigma_cu = prefill_entry.get("sigma_cu", 0)
    sigma_l2 = prefill_entry.get("sigma_l2", 0)
    sigma_bw = prefill_entry.get("sigma_bw", 0)

    ci_attn = decode_entry.get("ci_attn", 0.5)
    ci_ffn = decode_entry.get("ci_ffn", 0.5)
    l2_attn = decode_entry.get("l2_attn", 0.5)
    l2_ffn = decode_entry.get("l2_ffn", 0.5)
    g_launch = decode_entry.get("g_launch", 100.0)
    r_attn = decode_entry.get("r_attn", 0.3)
    r_ffn = decode_entry.get("r_ffn", 0.5)
    ipc = decode_entry.get("ipc", 0.3)

    A_bs = min(1.0 / max(g_launch, 1.0) * 1000, 1.0)
    A_cu = (ci_attn * r_attn + ci_ffn * r_ffn) / max(r_attn + r_ffn, 0.01)
    A_cu = min(A_cu / 10.0, 1.0)
    A_l2 = ((1 - l2_attn) * r_attn + (1 - l2_ffn) * r_ffn) / max(r_attn + r_ffn, 0.01)
    A_bw = 1.0 / max(A_cu * 10.0, 0.1)
    A_bw = min(A_bw, 1.0)

    r_decode = r_attn + r_ffn
    r_prefill = prefill_entry.get("r_attn", 0.3) + prefill_entry.get("r_ffn", 0.5)
    omega = min(r_decode, r_prefill)

    alpha = (
        weights["w_bs"] * sigma_bs * A_bs
        + weights["w_cu"] * sigma_cu * A_cu
        + weights["w_l2"] * sigma_l2 * A_l2
        + weights["w_bw"] * sigma_bw * A_bw
        + weights["w_ipc"] * ipc * sigma_cu
        + weights["w_overlap"] * omega
    )
    return max(0.0, min(alpha, 0.5))


def save_interference_table(entries: list, model_name: str, output_path: str):
    """Save interference table in simulator-compatible format."""
    table = {model_name: {"entries": entries}}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert MLWD data to simulator interference table")
    parser.add_argument("--mlwd-json", required=True, help="Path to MLWD complete JSON")
    parser.add_argument("--weights-json", default=None, help="Path to OLS weights JSON")
    parser.add_argument("--model-name", default="qwen2.5-7b")
    parser.add_argument("--output", default="results/interference_table.json")
    args = parser.parse_args()

    mlwd_data = load_mlwd_data(args.mlwd_json)
    weights = None
    if args.weights_json:
        with open(args.weights_json) as f:
            weights = json.load(f)

    entries = build_interference_table(mlwd_data, weights)
    save_interference_table(entries, args.model_name, args.output)
    print(f"Saved {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
