"""Async benchmark client for LLM inference.

Sends requests according to pre-generated arrival times, measures TTFT and
inter-token latency via SSE streaming, and collects per-request results.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import aiohttp

from workload.generator import Request as WorkloadRequest

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    request_id: int
    input_len: int
    output_len: int
    ttft_ms: float
    tpot_ms: float  # mean time-per-output-token (excluding first)
    total_latency_ms: float
    success: bool
    error: str = ""


class BenchmarkClient:
    """Drive a workload against a gateway / vLLM endpoint and record metrics."""

    def __init__(
        self,
        target_url: str,
        model_name: str,
        workload_requests: List[WorkloadRequest],
        slo_ttft: float = 2000.0,
        slo_tpot: float = 100.0,
    ):
        self.target_url = target_url.rstrip("/")
        self.model_name = model_name
        self.requests = workload_requests
        self.slo_ttft = slo_ttft
        self.slo_tpot = slo_tpot
        self.results: List[BenchmarkResult] = []

    # -- public API ------------------------------------------------------------

    async def run(self) -> List[BenchmarkResult]:
        """Send all requests respecting their arrival times and collect results."""
        if not self.requests:
            return self.results

        connector = aiohttp.TCPConnector(limit=256)
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            tasks: list[asyncio.Task] = []
            t0 = time.monotonic()

            for wreq in self.requests:
                # Sleep until this request's scheduled arrival
                now = time.monotonic() - t0
                delay = wreq.arrival_time - now
                if delay > 0:
                    await asyncio.sleep(delay)
                task = asyncio.create_task(self._send_request(session, wreq))
                tasks.append(task)

            self.results = await asyncio.gather(*tasks)

        log.info("Benchmark complete: %d requests", len(self.results))
        return self.results

    async def _send_request(
        self, session: aiohttp.ClientSession, wreq: WorkloadRequest
    ) -> BenchmarkResult:
        """POST a single streaming completion request and measure latencies."""
        # Build a dummy prompt of approximately the right token count
        prompt = "hello " * wreq.input_len
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": wreq.output_len,
            "temperature": 0,
            "stream": True,
        }

        url = f"{self.target_url}/v1/completions"
        send_time = time.monotonic()
        first_token_time: float | None = None
        token_times: list[float] = []

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return BenchmarkResult(
                        request_id=wreq.request_id,
                        input_len=wreq.input_len,
                        output_len=wreq.output_len,
                        ttft_ms=0, tpot_ms=0, total_latency_ms=0,
                        success=False,
                        error=f"HTTP {resp.status}: {body[:200]}",
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    now = time.monotonic()
                    # Record token arrival
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text"):
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)

        except Exception as exc:
            return BenchmarkResult(
                request_id=wreq.request_id,
                input_len=wreq.input_len,
                output_len=wreq.output_len,
                ttft_ms=0, tpot_ms=0, total_latency_ms=0,
                success=False,
                error=str(exc),
            )

        end_time = time.monotonic()
        total_ms = (end_time - send_time) * 1000.0
        ttft_ms = ((first_token_time - send_time) * 1000.0
                   if first_token_time else total_ms)

        # Mean inter-token latency (excluding first token)
        if len(token_times) >= 2:
            deltas = [
                (token_times[i] - token_times[i - 1]) * 1000.0
                for i in range(1, len(token_times))
            ]
            tpot_ms = sum(deltas) / len(deltas)
        else:
            tpot_ms = 0.0

        return BenchmarkResult(
            request_id=wreq.request_id,
            input_len=wreq.input_len,
            output_len=wreq.output_len,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_ms,
            success=True,
        )

    # -- persistence -----------------------------------------------------------

    def save_results(self, path: str | Path) -> None:
        """Write results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(r) for r in self.results]
        path.write_text(json.dumps(records, indent=2))
        log.info("Saved %d results to %s", len(records), path)

    # -- summary ---------------------------------------------------------------

    def summary(self) -> dict:
        """Compute aggregate statistics and SLO attainment."""
        ok = [r for r in self.results if r.success]
        if not ok:
            return {"total": len(self.results), "success": 0}

        ttfts = [r.ttft_ms for r in ok]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
        latencies = [r.total_latency_ms for r in ok]

        def _percentile(vals: list[float], p: float) -> float:
            vals_s = sorted(vals)
            idx = int(len(vals_s) * p / 100.0)
            return vals_s[min(idx, len(vals_s) - 1)]

        slo_ttft_ok = sum(1 for t in ttfts if t <= self.slo_ttft)
        slo_tpot_ok = sum(1 for t in tpots if t <= self.slo_tpot) if tpots else 0

        return {
            "total": len(self.results),
            "success": len(ok),
            "failed": len(self.results) - len(ok),
            "ttft_p50_ms": _percentile(ttfts, 50),
            "ttft_p99_ms": _percentile(ttfts, 99),
            "ttft_slo_pct": slo_ttft_ok / len(ok) * 100,
            "tpot_p50_ms": _percentile(tpots, 50) if tpots else 0,
            "tpot_p99_ms": _percentile(tpots, 99) if tpots else 0,
            "tpot_slo_pct": (slo_tpot_ok / len(tpots) * 100) if tpots else 0,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p99_ms": _percentile(latencies, 99),
        }
