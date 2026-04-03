"""V100-specific latency profile data for simulation."""

# Prefill latency model: a + b * tokens + c * tokens^2 (ms)
# Measured on V100-32GB with Qwen2.5-7B FP16, enforce_eager
V100_PREFILL_COEFFS = {
    "qwen2.5-7b": {"a": 15.0, "b": 0.025, "c": 2.5e-7},
    "qwen2.5-32b": {"a": 60.0, "b": 0.10, "c": 1.0e-6},
    "qwen3-14b": {"a": 30.0, "b": 0.05, "c": 5.0e-7},
    "llama-3.1-8b": {"a": 16.0, "b": 0.028, "c": 2.8e-7},
}

# Decode step latency: base_ms + per_token_ms * batch_size (ms)
V100_DECODE_COEFFS = {
    "qwen2.5-7b": {"base_ms": 28.0, "per_token_ms": 1.2},
    "qwen2.5-32b": {"base_ms": 110.0, "per_token_ms": 5.0},
    "qwen3-14b": {"base_ms": 55.0, "per_token_ms": 2.5},
    "llama-3.1-8b": {"base_ms": 30.0, "per_token_ms": 1.3},
}


def estimate_prefill_latency(model: str, input_tokens: int) -> float:
    """Estimate prefill latency in ms for given model and input length."""
    c = V100_PREFILL_COEFFS.get(model, V100_PREFILL_COEFFS["qwen2.5-7b"])
    return c["a"] + c["b"] * input_tokens + c["c"] * input_tokens ** 2


def estimate_decode_step_latency(model: str, batch_size: int) -> float:
    """Estimate single decode step latency in ms."""
    c = V100_DECODE_COEFFS.get(model, V100_DECODE_COEFFS["qwen2.5-7b"])
    return c["base_ms"] + c["per_token_ms"] * batch_size


def estimate_kv_transfer_latency(input_tokens: int, model: str = "qwen2.5-7b") -> float:
    """Estimate KV cache transfer latency in ms (intra-node NVLink)."""
    # KV size = 2 * layers * kv_heads * head_dim * seq_len * sizeof(fp16)
    model_params = {
        "qwen2.5-7b": {"layers": 28, "kv_heads": 4, "head_dim": 128},
        "qwen2.5-32b": {"layers": 64, "kv_heads": 8, "head_dim": 128},
        "qwen3-14b": {"layers": 40, "kv_heads": 8, "head_dim": 128},
        "llama-3.1-8b": {"layers": 32, "kv_heads": 8, "head_dim": 128},
    }
    p = model_params.get(model, model_params["qwen2.5-7b"])
    kv_bytes = 2 * p["layers"] * p["kv_heads"] * p["head_dim"] * input_tokens * 2  # fp16
    # NVLink bandwidth ~300 GB/s effective (V100 NVLink)
    bw_bytes_per_ms = 300e9 / 1000
    return kv_bytes / bw_bytes_per_ms + 0.5  # 0.5ms setup overhead
