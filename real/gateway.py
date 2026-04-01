"""Lightweight request gateway with routing strategies.

Sits between the benchmark client and the vLLM instances.  Supports:
  - round_robin  : cycle across prefill (or all) instances
  - random       : uniform random selection
  - cbs_aware    : simplified CBS score to decide disagg vs. coloc per-request
"""

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
from aiohttp import web

log = logging.getLogger(__name__)


class Gateway:
    """Async HTTP gateway that proxies /v1/completions to vLLM backends."""

    def __init__(
        self,
        prefill_urls: list[str],
        decode_urls: list[str],
        routing: str = "round_robin",
        cbs_params: dict[str, float] | None = None,
    ):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.routing = routing
        self.cbs_params = cbs_params or {}
        self._rr_idx = 0
        self._session: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None

    # -- request handler -------------------------------------------------------

    async def handle_request(self, request: web.Request) -> web.StreamResponse:
        """Route an incoming /v1/completions request to a backend."""
        body = await request.json()
        input_text = body.get("prompt", "")
        # Rough token count estimate (whitespace split) for CBS routing
        input_len = len(input_text.split())

        if self.routing == "cbs_aware":
            target_url = await self._cbs_select(input_len)
        elif self.routing == "random":
            import random
            target_url = random.choice(self.prefill_urls or self.decode_urls)
        else:
            target_url = self._round_robin_select()

        # Check if client wants streaming
        stream = body.get("stream", False)

        backend_url = f"{target_url}/v1/completions"
        if stream:
            return await self._proxy_stream(request, backend_url, body)
        else:
            return await self._proxy_simple(backend_url, body)

    # -- routing strategies ----------------------------------------------------

    def _round_robin_select(self) -> str:
        """Cycle across prefill instances (or all instances if no split)."""
        pool = self.prefill_urls if self.prefill_urls else self.decode_urls
        url = pool[self._rr_idx % len(pool)]
        self._rr_idx += 1
        return url

    async def _cbs_select(self, input_len: int) -> str:
        """Compute a simplified CBS score to decide disagg vs. coloc.

        CBS = mu * (decode_queue_pressure) - lambda_ext * (prefill_cost)
              - kappa_dispatch

        If CBS > 0 the decode nodes are under-loaded enough to colocate;
        route to the least-loaded decode node.  Otherwise disaggregate to
        the least-loaded prefill node.
        """
        mu = self.cbs_params.get("mu", 2.0)
        lam = self.cbs_params.get("lambda_ext", 1.0)
        kappa = self.cbs_params.get("kappa_dispatch", 0.1)

        decode_metrics = await self._fetch_queue_depths(self.decode_urls)
        prefill_metrics = await self._fetch_queue_depths(self.prefill_urls)

        # Decode pressure: average fraction of free KV-cache across decode nodes
        avg_decode_free = 1.0
        if decode_metrics:
            avg_decode_free = sum(
                1.0 - m.get("gpu_cache_usage_perc", 0.0)
                for m in decode_metrics.values()
            ) / len(decode_metrics)

        # Prefill cost proxy: normalised input length
        prefill_cost = input_len / 8192.0

        cbs = mu * avg_decode_free - lam * prefill_cost - kappa

        if cbs > 0 and decode_metrics:
            # Colocate: pick decode node with fewest running requests
            best = min(decode_metrics, key=lambda u: decode_metrics[u].get(
                "num_requests_running", float("inf")))
            return best
        else:
            # Disaggregate: pick prefill node with fewest waiting requests
            if prefill_metrics:
                best = min(prefill_metrics, key=lambda u: prefill_metrics[u].get(
                    "num_requests_waiting", float("inf")))
                return best
            return self._round_robin_select()

    async def _fetch_queue_depths(
        self, urls: list[str]
    ) -> dict[str, dict[str, float]]:
        """Scrape lightweight metrics from each instance's /metrics endpoint."""
        results: dict[str, dict[str, float]] = {}
        if not self._session:
            return results
        for url in urls:
            try:
                async with self._session.get(
                    f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    text = await resp.text()
                    results[url] = _parse_prometheus_subset(text)
            except Exception:
                log.debug("Failed to scrape metrics from %s", url)
        return results

    # -- proxying --------------------------------------------------------------

    async def _proxy_simple(
        self, backend_url: str, body: dict
    ) -> web.Response:
        assert self._session is not None
        async with self._session.post(backend_url, json=body) as resp:
            data = await resp.read()
            return web.Response(
                body=data,
                status=resp.status,
                content_type=resp.content_type,
            )

    async def _proxy_stream(
        self, request: web.Request, backend_url: str, body: dict
    ) -> web.StreamResponse:
        assert self._session is not None
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)

        async with self._session.post(backend_url, json=body) as resp:
            async for chunk in resp.content.iter_any():
                await response.write(chunk)

        await response.write_eof()
        return response

    # -- server lifecycle ------------------------------------------------------

    async def start(self, port: int = 8080) -> None:
        """Start the gateway HTTP server."""
        self._session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_post("/v1/completions", self.handle_request)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", port)
        await site.start()
        log.info("Gateway listening on 0.0.0.0:%d  routing=%s", port, self.routing)

    async def stop(self) -> None:
        """Shut down the gateway."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._session:
            await self._session.close()
            self._session = None


# -- helpers -------------------------------------------------------------------

def _parse_prometheus_subset(text: str) -> dict[str, float]:
    """Extract a handful of vLLM metrics from Prometheus text format."""
    metrics: dict[str, float] = {}
    targets = {
        "vllm:num_requests_running": "num_requests_running",
        "vllm:num_requests_waiting": "num_requests_waiting",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_perc",
    }
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for prom_name, key in targets.items():
            if line.startswith(prom_name):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics[key] = float(parts[-1])
                    except ValueError:
                        pass
    return metrics
