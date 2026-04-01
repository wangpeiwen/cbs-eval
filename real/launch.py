"""Launch vLLM cluster for real experiments.

Supports three topologies on 4x V100-32GB:
  - disagg_2p2d : 2 Prefill + 2 Decode with KV transfer
  - coloc_4     : 4 colocated mixed-PD instances
  - cbs_2p2d    : 2P+2D with CBS-aware gateway routing
"""

import logging
from typing import Any

from real.vllm_instance import VLLMInstance

log = logging.getLogger(__name__)

# Default port ranges
_PREFILL_PORT_BASE = 8100
_DECODE_PORT_BASE = 8200
_COLOC_PORT_BASE = 8100


class ClusterLauncher:
    """Orchestrates a set of VLLMInstance processes."""

    def __init__(self, scenario_config: dict, model_config: dict):
        self.scenario = scenario_config
        self.model = model_config
        self.instances: list[VLLMInstance] = []
        self.prefill_instances: list[VLLMInstance] = []
        self.decode_instances: list[VLLMInstance] = []

    # -- common vLLM kwargs from config ----------------------------------------

    def _common_kwargs(self) -> dict[str, Any]:
        return dict(
            model_path=self.model["path"],
            dtype=self.model.get("dtype", "float16"),
            enforce_eager=True,
            gpu_mem_util=0.90,
            max_model_len=self.model.get("max_model_len", 8192),
            max_num_batched_tokens=self.model.get("max_num_batched_tokens", 8192),
            trust_remote_code=True,
            enable_chunked_prefill=True,
        )

    # -- topology launchers ----------------------------------------------------

    def launch_disagg_2p2d(self) -> None:
        """2 Prefill (GPU 0,1) + 2 Decode (GPU 2,3) with KV transfer."""
        prefill_gpus = self.scenario.get("prefill_gpus", [0, 1])
        decode_gpus = self.scenario.get("decode_gpus", [2, 3])
        connector = self.scenario.get("kv_connector", "PyNcclConnector")
        total = len(prefill_gpus) + len(decode_gpus)
        kw = self._common_kwargs()

        for i, gpu in enumerate(prefill_gpus):
            inst = VLLMInstance(
                port=_PREFILL_PORT_BASE + i,
                gpu_id=gpu,
                role="prefill",
                kv_connector=connector,
                kv_role="kv_producer",
                kv_rank=i,
                kv_parallel_size=total,
                **kw,
            )
            self.prefill_instances.append(inst)
            self.instances.append(inst)

        for i, gpu in enumerate(decode_gpus):
            inst = VLLMInstance(
                port=_DECODE_PORT_BASE + i,
                gpu_id=gpu,
                role="decode",
                kv_connector=connector,
                kv_role="kv_consumer",
                kv_rank=len(prefill_gpus) + i,
                kv_parallel_size=total,
                **kw,
            )
            self.decode_instances.append(inst)
            self.instances.append(inst)

    def launch_coloc_4(self) -> None:
        """4 colocated mixed prefill+decode instances."""
        gpus = self.scenario.get("gpus", [0, 1, 2, 3])
        kw = self._common_kwargs()

        for i, gpu in enumerate(gpus):
            inst = VLLMInstance(
                port=_COLOC_PORT_BASE + i,
                gpu_id=gpu,
                role="neutral",
                **kw,
            )
            self.instances.append(inst)

    def launch_cbs_2p2d(self) -> None:
        """Same physical layout as disagg_2p2d; gateway uses CBS routing."""
        self.launch_disagg_2p2d()

    # -- lifecycle helpers -----------------------------------------------------

    def launch(self) -> None:
        """Dispatch to the correct topology based on scenario mode."""
        mode = self.scenario.get("mode", "disaggregate")
        dispatch = {
            "disaggregate": self.launch_disagg_2p2d,
            "colocate": self.launch_coloc_4,
            "cbs": self.launch_cbs_2p2d,
        }
        fn = dispatch.get(mode)
        if fn is None:
            raise ValueError(f"Unknown scenario mode: {mode}")
        fn()

        log.info("Starting %d vLLM instances ...", len(self.instances))
        for inst in self.instances:
            inst.start()

    def wait_all_ready(self, timeout: int = 300) -> None:
        """Block until every instance passes its health check."""
        log.info("Waiting for all instances to become healthy ...")
        for inst in self.instances:
            inst.wait_ready(timeout=timeout)
        log.info("All %d instances healthy.", len(self.instances))

    def teardown(self) -> None:
        """Stop every instance."""
        log.info("Tearing down cluster ...")
        for inst in reversed(self.instances):
            try:
                inst.stop()
            except Exception:
                log.exception("Error stopping %s", inst)
        self.instances.clear()
        self.prefill_instances.clear()
        self.decode_instances.clear()

    @property
    def prefill_urls(self) -> list[str]:
        return [i.url for i in self.prefill_instances]

    @property
    def decode_urls(self) -> list[str]:
        return [i.url for i in self.decode_instances]

    @property
    def all_urls(self) -> list[str]:
        return [i.url for i in self.instances]
