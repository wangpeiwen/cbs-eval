"""Latency estimators used by the SimPy engine.

The simulator runs in milliseconds.  These estimators use the repository's
V100 profile coefficients and keep the older call signatures used by workers
and schedulers.
"""

from sim.v100_profile import V100_DECODE_COEFFS, V100_PREFILL_COEFFS


def _model_name(model_type=None):
    model = str(model_type or "qwen2.5-7b")
    if model in V100_PREFILL_COEFFS:
        return model
    return "qwen2.5-7b"


def get_prefill_time(
    num_tokens=None,
    pp=1,
    bs=1,
    decode_bs=0,
    model_type="qwen2.5-7b",
    TP=1,
    prefill_len_list=None,
    engine_type="distserve",
    **kw,
):
    """Estimate batched prefill latency in milliseconds."""
    model = _model_name(model_type)
    coeff = V100_PREFILL_COEFFS[model]

    if prefill_len_list is None:
        tokens = int(num_tokens or 0)
        prefill_len_list = [tokens] if tokens > 0 else []
    if not prefill_len_list:
        return 0.0

    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum(x ** 2 for x in prefill_len_list)
    delay = (
        coeff["a"]
        + coeff["b"] * num_total_tokens
        + coeff["c"] * sum_num_tokens_sqr
    )
    return delay / max(pp, 1) + max(pp, 1)


def get_decode_time(
    num_requests,
    pp=1,
    model_type="qwen2.5-7b",
    TP=1,
    token_generated_list=None,
    engine_type="distserve",
    **kw,
):
    """Estimate one decode iteration latency in milliseconds."""
    batch_size = int(num_requests or 0)
    if batch_size <= 0:
        return 0.0

    model = _model_name(model_type)
    coeff = V100_DECODE_COEFFS[model]
    delay = coeff["base_ms"] + coeff["per_token_ms"] * batch_size
    return delay / max(pp, 1)
