"""Collect vLLM Prometheus metrics during experiments.

Periodically scrapes each instance's ``/metrics`` endpoint and stores
time-series snapshots for post-hoc analysis.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

# Prometheus metric names we care about
_TARGET_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:avg_prompt_throughput_toks_per_s",
)


class MetricsCollector:
    """Background task that scrapes vLLM Prometheus metrics at a fixed cadence."""

    def __init__(self, instance_urls: list[str], interval_s: float = 1.0):
        self.urls = instance_urls
        self.interval = interval_s
        self.timeseries: list[dict[str, Any]] = []
        self._running = False
        self._task: asyncio.Task | None = None

    # -- public API ------------------------------------------------------------

    async def start(self) -> None:
        """Begin periodic scraping in the background."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("MetricsCollector started (%d instances, %.1fs interval)",
                 len(self.urls), self.interval)

    def stop(self) -> None:
        """Signal the collection loop to stop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        log.info("MetricsCollector stopped (%d snapshots collected)",
                 len(self.timeseries))

    def save(self, path: str | Path) -> None:
        """Persist collected time-series to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.timeseries, indent=2))
        log.info("Saved %d metric snapshots to %s", len(self.timeseries), path)

    # -- internals -------------------------------------------------------------

    async def _loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self._running:
                snapshot: dict[str, Any] = {
                    "timestamp": time.time(),
                    "instances": {},
                }
                scrape_tasks = [
                    self._scrape(session, url) for url in self.urls
                ]
                results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                for url, result in zip(self.urls, results):
                    if isinstance(result, Exception):
                        log.debug("Scrape failed for %s: %s", url, result)
                        snapshot["instances"][url] = {"error": str(result)}
                    else:
                        snapshot["instances"][url] = result
                self.timeseries.append(snapshot)
                await asyncio.sleep(self.interval)

    async def _scrape(
        self, session: aiohttp.ClientSession, url: str
    ) -> dict[str, float]:
        """GET /metrics and parse the Prometheus text exposition format."""
        async with session.get(f"{url}/metrics") as resp:
            text = await resp.text()
        return _parse_prometheus(text)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Extract target metrics from Prometheus text format.

    Lines look like:
        # HELP vllm:num_requests_running ...
        # TYPE vllm:num_requests_running gauge
        vllm:num_requests_running{model_name="..."} 3.0
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        for target in _TARGET_METRICS:
            if line.startswith(target):
                # Value is always the last whitespace-separated token
                parts = line.split()
                if parts:
                    try:
                        metrics[target] = float(parts[-1])
                    except ValueError:
                        pass
                break
    return metrics
