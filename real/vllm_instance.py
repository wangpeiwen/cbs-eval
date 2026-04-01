"""Single vLLM inference instance wrapper.

Manages the lifecycle of one `vllm serve` process bound to a specific GPU,
optionally configured for disaggregated prefill/decode via KV connectors.
"""

import logging
import os
import signal
import subprocess
import time

import requests as req

log = logging.getLogger(__name__)


class VLLMInstance:
    """Wraps a single ``vllm serve`` process."""

    def __init__(
        self,
        model_path: str,
        port: int,
        gpu_id: int,
        role: str = "neutral",
        kv_connector: str | None = None,
        kv_role: str | None = None,
        kv_rank: int | None = None,
        kv_parallel_size: int | None = None,
        dtype: str = "float16",
        enforce_eager: bool = True,
        gpu_mem_util: float = 0.90,
        max_model_len: int = 8192,
        max_num_batched_tokens: int = 8192,
        trust_remote_code: bool = True,
        enable_chunked_prefill: bool = True,
    ):
        self.model_path = model_path
        self.port = port
        self.gpu_id = gpu_id
        self.role = role  # "prefill", "decode", or "neutral"
        self.kv_connector = kv_connector
        self.kv_role = kv_role
        self.kv_rank = kv_rank
        self.kv_parallel_size = kv_parallel_size
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.gpu_mem_util = gpu_mem_util
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.trust_remote_code = trust_remote_code
        self.enable_chunked_prefill = enable_chunked_prefill
        self._proc: subprocess.Popen | None = None

    # -- command construction --------------------------------------------------

    def _build_cmd(self) -> list[str]:
        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_mem_util),
            "--max-model-len", str(self.max_model_len),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        # Disaggregated KV-transfer flags (vLLM >= 0.18)
        if self.kv_connector is not None:
            cmd.extend(["--kv-connector", self.kv_connector])
        if self.kv_role is not None:
            cmd.extend(["--kv-role", self.kv_role])
        if self.kv_rank is not None:
            cmd.extend(["--kv-rank", str(self.kv_rank)])
        if self.kv_parallel_size is not None:
            cmd.extend(["--kv-parallel-size", str(self.kv_parallel_size)])

        return cmd

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> subprocess.Popen:
        """Launch the vLLM serve process with the target GPU visible."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Disable vLLM v1 engine to use the stable serving path
        env["VLLM_USE_V1"] = "0"

        cmd = self._build_cmd()
        log.info("Starting vLLM [gpu=%d role=%s port=%d]: %s",
                 self.gpu_id, self.role, self.port, " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return self._proc

    def wait_ready(self, timeout: int = 300) -> None:
        """Poll the ``/health`` endpoint until the instance is serving."""
        deadline = time.monotonic() + timeout
        url = f"{self.url}/health"
        while time.monotonic() < deadline:
            try:
                r = req.get(url, timeout=5)
                if r.status_code == 200:
                    log.info("vLLM ready on port %d (gpu %d)", self.port, self.gpu_id)
                    return
            except req.ConnectionError:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"vLLM on port {self.port} (gpu {self.gpu_id}) not ready "
            f"after {timeout}s"
        )

    def stop(self) -> None:
        """Terminate the vLLM process and wait for cleanup."""
        if self._proc is None:
            return
        log.info("Stopping vLLM on port %d (gpu %d)", self.port, self.gpu_id)
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("Force-killing vLLM on port %d", self.port)
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def metrics_url(self) -> str:
        return f"http://localhost:{self.port}/metrics"

    def __repr__(self) -> str:
        return (f"VLLMInstance(gpu={self.gpu_id}, port={self.port}, "
                f"role={self.role}, alive={self.alive})")
