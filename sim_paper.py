"""Paper-faithful SimPy simulator for CBS dynamic scheduling.

Implements exactly the algorithms described in the thesis:
- CBS scoring (chap4 §4.3): C_disagg - C_coloc with psi factor
- Bilateral migration (chap4 §4.5): Mitigation + Consolidation + role adaptation
- Baselines: Disagg-Static, Coloc-Sarathi
- Interference model: simplified alpha_d/alpha_p from real colocation data

Usage:
    python sim_paper.py --config 8node --model qwen2.5-7b --rates 4,8,12,16
"""

import argparse, json, math, os, random, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import simpy

# ── V100 Latency Profile ─────────────────────────────────────

PREFILL_COEFFS = {
    # V100 vLLM enforce_eager, FP16. Includes scheduling overhead and KV write.
    # Measured: 512 tokens ~55ms, 1024 tokens ~110ms, 2048 tokens ~250ms.
    "qwen2.5-7b": {"a": 20.0, "b": 0.08, "c": 1.5e-5},
    "llama-3.1-8b": {"a": 22.0, "b": 0.09, "c": 1.6e-5},
}
DECODE_COEFFS = {
    # Recalibrated: base is per-step latency at bs=1 with short context.
    # V100 vLLM enforce_eager, FP16. Measured range: bs=1 ~15ms, bs=8 ~22ms.
    "qwen2.5-7b": {"base_ms": 14.0, "per_token_ms": 1.0},
    "llama-3.1-8b": {"base_ms": 15.0, "per_token_ms": 1.1},
}
KV_PARAMS = {
    "qwen2.5-7b": {"layers": 28, "kv_heads": 4, "head_dim": 128},
    "llama-3.1-8b": {"layers": 32, "kv_heads": 8, "head_dim": 128},
}

def prefill_latency_ms(model: str, tokens: int, jitter: float = 0.0,
                       rng: random.Random = None) -> float:
    c = PREFILL_COEFFS.get(model, PREFILL_COEFFS["qwen2.5-7b"])
    base = c["a"] + c["b"] * tokens + c["c"] * tokens ** 2
    if jitter > 0 and rng is not None:
        base *= max(1.0 + rng.gauss(0, jitter), 0.5)
    return base

def decode_step_ms(model: str, batch_size: int, jitter: float = 0.0,
                   rng: random.Random = None) -> float:
    c = DECODE_COEFFS.get(model, DECODE_COEFFS["qwen2.5-7b"])
    base = c["base_ms"] + c["per_token_ms"] * batch_size
    if jitter > 0 and rng is not None:
        base *= max(1.0 + rng.gauss(0, jitter), 0.5)
    return base

def kv_transfer_ms(model: str, seq_len: int, jitter: float = 0.0,
                   rng: random.Random = None) -> float:
    p = KV_PARAMS.get(model, KV_PARAMS["qwen2.5-7b"])
    kv_bytes = 2 * p["layers"] * p["kv_heads"] * p["head_dim"] * seq_len * 2
    bw_bytes_per_ms = 300e9 / 1000  # NVLink 300 GB/s
    base = kv_bytes / bw_bytes_per_ms + 0.5
    if jitter > 0 and rng is not None:
        base *= max(1.0 + rng.gauss(0, jitter), 0.5)
    return base

# ── Interference Model (from real colocation data) ────────────

# Simplified interference coefficients calibrated from 41 colocation samples.
# alpha_d(decode_bs, prefill_len): Decode slowdown when colocated with Prefill
# alpha_p(decode_bs, prefill_len): Prefill slowdown when colocated with Decode
# These are fitted from combined-ols-calibration.json (Ridge, alpha=0.1).

S_CHUNK = [256, 512, 1024, 2048]  # discrete candidate set

def resolve_chunk_size(model: str, decode_bs: int, input_len: int, hp) -> int:
    """Adaptive chunk size: minimize total colocation cost across all chunks,
    subject to TPOT SLO constraint.

    Total cost = n_chunks * (delta_prefill + lambda*delta_ext + delta_dispatch)
    where n_chunks = ceil(input_len / c). The prefill latency's quadratic term
    creates a U-shaped cost curve, making intermediate chunk sizes (e.g. 1024)
    optimal for long sequences.
    """
    best_c = S_CHUNK[0]
    best_cost = float("inf")
    for c in S_CHUNK:
        s_eff = min(input_len, c)
        # TPOT SLO hard constraint
        a_d = alpha_d_model(model, decode_bs, s_eff)
        tpot = decode_step_ms(model, decode_bs + 1) * (1 + a_d)
        if tpot > hp.slo_tpot_ms:
            continue
        n_chunks = math.ceil(input_len / c) if input_len > c else 1
        a_p = alpha_p_model(model, decode_bs, s_eff)
        t_pf0 = prefill_latency_ms(model, s_eff)
        tau = t_pf0 * (1 + a_p)
        delta_pf = a_p * t_pf0
        delta_ext = decode_bs * tau * a_d
        delta_disp = hp.kappa * hp.r_attn_prefill * hp.r_attn_decode * tau
        total_cost = n_chunks * (delta_pf + hp.lam * delta_ext + delta_disp)
        if total_cost < best_cost:
            best_cost = total_cost
            best_c = c
    return best_c

def alpha_d_model(model: str, decode_bs: int, prefill_len: int) -> float:
    """Estimate Decode interference coefficient."""
    # From real data: alpha_d decreases with decode_bs (amortization),
    # increases mildly with prefill_len. kv_heads=8 models are more sensitive.
    if model == "llama-3.1-8b":
        base = 0.08 if decode_bs <= 1 else 0.04 if decode_bs <= 2 else 0.015
    else:  # qwen2.5-7b (kv_heads=4, less sensitive)
        base = 0.05 if decode_bs <= 1 else 0.025 if decode_bs <= 2 else 0.01
    # Mild prefill length effect
    len_factor = 1.0 + 0.1 * (prefill_len / 1024)
    return max(base * len_factor, 0.0)

def alpha_p_model(model: str, decode_bs: int, prefill_len: int) -> float:
    """Estimate Prefill interference coefficient."""
    # Prefill is less affected by colocation (compute-bound)
    return alpha_d_model(model, decode_bs, prefill_len) * 0.6

# ── Data Structures ───────────────────────────────────────────

@dataclass
class Request:
    id: int
    arrival_ms: float
    input_len: int
    output_len: int
    model: str
    # Filled during execution
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    completion_ms: float = 0.0
    mode: str = ""  # "disagg" or "coloc"
    slo_met: bool = False

@dataclass
class DecodeTask:
    req: Request
    tokens_generated: int = 0
    current_seq_len: int = 0  # input_len + tokens_generated
    start_decode_ms: float = 0.0
    last_step_ms: float = 0.0
    migrating: bool = False
    cooldown_until: float = 0.0

@dataclass
class Worker:
    id: int
    role: str  # "prefill" or "decode"
    decode_tasks: List[DecodeTask] = field(default_factory=list)
    prefill_queue: List = field(default_factory=list)
    busy: bool = False
    # Utilization tracking
    busy_ms: float = 0.0       # cumulative time spent computing
    total_ms: float = 0.0      # cumulative wall-clock time tracked
    tokens_processed: int = 0  # total tokens processed (prefill + decode)

# ── Hyperparameters (Table 4.1) ───────────────────────────────

@dataclass
class HyperParams:
    lam: float = 1.0           # externality weight lambda
    kappa: float = 0.1         # dispatch contention coefficient
    o_hist: float = 256.0      # historical average output length
    chunk_size: int = 512      # chunked prefill token limit (Sarathi baseline)
    slo_ttft_ms: float = 2000.0
    slo_tpot_ms: float = 100.0
    theta_dispatch: float = 0.85
    r_max: int = 2
    t_cool_ms: float = 10000.0
    t_role_thresh_ms: float = 1000.0  # 0.5 * SLO_TTFT
    migration_scan_ms: float = 1000.0
    r_attn_prefill: float = 0.55  # typical Attention time ratio for Prefill
    r_attn_decode: float = 0.55   # typical Attention time ratio for Decode
    # Jitter parameters for W2 sensitivity analysis (multiplicative Gaussian σ)
    jitter_prefill: float = 0.0
    jitter_decode: float = 0.0
    jitter_kv: float = 0.0

# ── CBS Scoring (chap4 §4.3) ──────────────────────────────────

def compute_cbs(req: Request, worker: Worker, prefill_workers: List[Worker],
                hp: HyperParams) -> float:
    """Compute CBS(r_j, d) = C_disagg - C_coloc for candidate Decode worker."""
    model = req.model
    decode_bs = len(worker.decode_tasks)

    # Adaptive chunk size for CBS; Sarathi baseline stays fixed
    adaptive_chunk = resolve_chunk_size(model, decode_bs, req.input_len, hp)
    s_eff = min(req.input_len, adaptive_chunk)

    # ── C_disagg: best Prefill node queue + KV transfer ──
    best_queue_ms = float("inf")
    for pw in prefill_workers:
        q_len = len(pw.prefill_queue) + (1 if pw.busy else 0)
        avg_service = prefill_latency_ms(model, 512)  # approximate avg
        best_queue_ms = min(best_queue_ms, q_len * avg_service)
    t_kv = kv_transfer_ms(model, req.input_len)
    c_disagg = best_queue_ms + t_kv

    # ── C_coloc components ──
    if decode_bs == 0:
        # Empty Decode node: no interference, no queue delay
        # Degenerates to: CBS = c_disagg - 0 = c_disagg > 0 (pure coloc regime)
        c_coloc = 0.0
    else:
        # Queue delay on Decode node (wait for current iteration)
        t_queue_d = decode_step_ms(model, decode_bs) * 0.5

        # Prefill interference loss
        a_p = alpha_p_model(model, decode_bs, s_eff)
        t_prefill_0 = prefill_latency_ms(model, s_eff)
        delta_prefill = a_p * t_prefill_0

        # Decode externality cost with psi factor
        a_d = alpha_d_model(model, decode_bs, s_eff)
        tau_j = t_prefill_0 * (1 + a_p)

        # psi: interference persistence factor (chap4 §4.2.2)
        avg_o_remain = np.mean([max(t.req.output_len - t.tokens_generated, 1)
                                for t in worker.decode_tasks])
        psi = math.log(1 + avg_o_remain) / math.log(1 + hp.o_hist)

        delta_ext = decode_bs * tau_j * a_d * psi

        # Dispatch contention
        delta_dispatch = hp.kappa * hp.r_attn_prefill * hp.r_attn_decode * tau_j

        c_coloc = t_queue_d + delta_prefill + hp.lam * delta_ext + delta_dispatch

        # Compute budget constraint: predicted TPOT after admission
        tpot_after = decode_step_ms(model, decode_bs + 1) * (1 + a_d)
        if tpot_after > hp.slo_tpot_ms:
            return -float("inf")

    return c_disagg - c_coloc


# ── Simulation Engine ─────────────────────────────────────────

class Simulator:
    def __init__(self, n_prefill: int, n_decode: int, model: str,
                 hp: HyperParams, system: str = "cbs_full", seed: int = 42):
        self.env = simpy.Environment()
        self.model = model
        self.hp = hp
        self.system = system  # "disagg_static", "coloc_sarathi", "cbs_nomig", "cbs_norole", "cbs_full"
        self.completed: List[Request] = []
        self.cbs_decisions: List[dict] = []  # W5 decision log
        self.rng = random.Random(seed + 9999)  # separate RNG for jitter

        # Create workers
        self.workers: List[Worker] = []
        for i in range(n_prefill):
            self.workers.append(Worker(id=i, role="prefill"))
        for i in range(n_decode):
            self.workers.append(Worker(id=n_prefill + i, role="decode"))

    @property
    def prefill_workers(self):
        return [w for w in self.workers if w.role == "prefill"]

    @property
    def decode_workers(self):
        return [w for w in self.workers if w.role == "decode"]

    def run(self, requests: List[Request], duration_ms: float):
        """Run simulation with given request stream."""
        self.env.process(self._arrival_process(requests))
        for w in self.workers:
            if w.role == "decode":
                self.env.process(self._decode_loop(w))
            else:
                self.env.process(self._prefill_loop(w))
        # Migration process (only for CBS variants with migration)
        if self.system in ("cbs_norole", "cbs_full"):
            self.env.process(self._migration_loop())
        self.env.run(until=duration_ms)

    def _arrival_process(self, requests: List[Request]):
        """Inject requests at their arrival times."""
        for req in requests:
            wait = req.arrival_ms - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            self._schedule_request(req)

    def _schedule_request(self, req: Request):
        """CBS or baseline scheduling decision."""
        if self.system == "disagg_static":
            self._route_disagg(req)
        elif self.system == "coloc_sarathi":
            self._route_coloc(req)
        else:
            self._route_cbs(req)

    def _route_disagg(self, req: Request):
        """Pure disaggregation: always use Prefill node."""
        req.mode = "disagg"
        best_pw = min(self.prefill_workers,
                      key=lambda w: len(w.prefill_queue) + (1 if w.busy else 0))
        best_pw.prefill_queue.append(req)

    def _route_coloc(self, req: Request):
        """Pure colocation (Sarathi): always colocate on Decode node."""
        req.mode = "coloc"
        # Pick Decode node with lowest batch size
        dw = min(self.decode_workers, key=lambda w: len(w.decode_tasks))
        dw.prefill_queue.append(req)

    def _route_cbs(self, req: Request):
        """CBS dynamic decision."""
        best_score = -float("inf")
        best_dw = None
        for dw in self.decode_workers:
            score = compute_cbs(req, dw, self.prefill_workers, self.hp)
            if score > best_score:
                best_score = score
                best_dw = dw

        if best_score > 0 and best_dw is not None:
            req.mode = "coloc"
            best_dw.prefill_queue.append(req)
        else:
            req.mode = "disagg"
            self._route_disagg(req)

        # W5: log decision details
        best_bs = len(best_dw.decode_tasks) if best_dw else 0
        chunk = resolve_chunk_size(self.model, best_bs, req.input_len, self.hp)
        s_eff = min(req.input_len, chunk)
        self.cbs_decisions.append({
            "req_id": req.id,
            "time_ms": self.env.now,
            "alpha_d": alpha_d_model(self.model, best_bs, s_eff),
            "alpha_p": alpha_p_model(self.model, best_bs, s_eff),
            "decode_bs": best_bs,
            "prefill_len": req.input_len,
            "s_eff": s_eff,
            "cbs_score": best_score if best_score != -float("inf") else -999,
            "chosen_mode": req.mode,
            "best_node_id": best_dw.id if best_dw else None,
        })

    def _prefill_loop(self, worker: Worker):
        """Process Prefill requests on a Prefill worker."""
        while True:
            if not worker.prefill_queue:
                worker.total_ms += 1
                yield self.env.timeout(1)  # poll interval
                continue
            req = worker.prefill_queue.pop(0)
            worker.busy = True
            t_prefill = prefill_latency_ms(self.model, req.input_len,
                                           self.hp.jitter_prefill, self.rng)
            worker.busy_ms += t_prefill
            worker.total_ms += t_prefill
            worker.tokens_processed += req.input_len
            yield self.env.timeout(t_prefill)
            req.ttft_ms = self.env.now - req.arrival_ms
            worker.busy = False
            t_kv = kv_transfer_ms(self.model, req.input_len,
                                  self.hp.jitter_kv, self.rng)
            yield self.env.timeout(t_kv)
            # Assign to least-loaded Decode node
            dw = min(self.decode_workers, key=lambda w: len(w.decode_tasks))
            task = DecodeTask(req=req, current_seq_len=req.input_len,
                              start_decode_ms=self.env.now)
            dw.decode_tasks.append(task)

    def _decode_loop(self, worker: Worker):
        """Continuous batching decode loop on a Decode worker."""
        while True:
            # Process any pending Prefill requests (colocation mode)
            while worker.prefill_queue:
                req = worker.prefill_queue.pop(0)
                bs = len(worker.decode_tasks)
                # Sarathi uses fixed chunk; CBS variants use adaptive chunk
                if self.system == "coloc_sarathi":
                    chunk = self.hp.chunk_size
                else:
                    chunk = resolve_chunk_size(self.model, bs, req.input_len, self.hp)
                s_eff = min(req.input_len, chunk)
                a_p = alpha_p_model(self.model, bs, s_eff)
                t_prefill = prefill_latency_ms(self.model, s_eff,
                                               self.hp.jitter_prefill, self.rng) * (1 + a_p)
                worker.busy_ms += t_prefill
                worker.total_ms += t_prefill
                worker.tokens_processed += s_eff
                yield self.env.timeout(t_prefill)
                req.ttft_ms = self.env.now - req.arrival_ms
                remaining = req.input_len - s_eff
                while remaining > 0:
                    # Re-resolve chunk size: decode load may have changed
                    bs = len(worker.decode_tasks)
                    if self.system == "coloc_sarathi":
                        chunk = self.hp.chunk_size
                    else:
                        chunk = resolve_chunk_size(self.model, bs, remaining, self.hp)
                    c = min(remaining, chunk)
                    a_p = alpha_p_model(self.model, bs, c)
                    t_chunk = prefill_latency_ms(self.model, c,
                                                 self.hp.jitter_prefill, self.rng) * (1 + a_p)
                    worker.busy_ms += t_chunk
                    worker.total_ms += t_chunk
                    worker.tokens_processed += c
                    yield self.env.timeout(t_chunk)
                    remaining -= c
                task = DecodeTask(req=req, current_seq_len=req.input_len,
                                  start_decode_ms=self.env.now)
                worker.decode_tasks.append(task)

            if not worker.decode_tasks:
                worker.total_ms += 1
                yield self.env.timeout(1)
                continue

            bs = len(worker.decode_tasks)
            base_step = decode_step_ms(self.model, bs,
                                       self.hp.jitter_decode, self.rng)
            step_ms = base_step
            worker.busy_ms += step_ms
            worker.total_ms += step_ms
            worker.tokens_processed += bs

            yield self.env.timeout(step_ms)

            # Update all tasks
            finished = []
            for task in worker.decode_tasks:
                if task.migrating:
                    continue
                task.tokens_generated += 1
                task.current_seq_len += 1
                task.last_step_ms = step_ms
                if task.tokens_generated >= task.req.output_len:
                    task.req.completion_ms = self.env.now
                    task.req.tpot_ms = ((self.env.now - task.start_decode_ms)
                                        / max(task.tokens_generated - 1, 1))
                    task.req.slo_met = (task.req.ttft_ms <= self.hp.slo_ttft_ms
                                        and task.req.tpot_ms <= self.hp.slo_tpot_ms)
                    self.completed.append(task.req)
                    finished.append(task)
            for t in finished:
                worker.decode_tasks.remove(t)

    def _predict_tpot(self, worker: Worker) -> float:
        """Predict TPOT on a Decode worker, accounting for Prefill colocation."""
        bs = len(worker.decode_tasks)
        if bs == 0:
            return 0.0
        base_step = decode_step_ms(self.model, bs)
        # Use average input length from pending queue, or default
        avg_input = self.hp.chunk_size  # fallback
        if worker.prefill_queue:
            avg_input = int(np.mean([r.input_len for r in worker.prefill_queue]))
        has_coloc = len(worker.prefill_queue) > 0
        if has_coloc:
            chunk = resolve_chunk_size(self.model, bs, avg_input, self.hp)
            a_d = alpha_d_model(self.model, bs, chunk)
            avg_prefill_time = prefill_latency_ms(self.model, chunk)
            prefill_overhead_ratio = avg_prefill_time / (avg_prefill_time + base_step)
            return base_step * (1 + a_d) + avg_prefill_time * prefill_overhead_ratio
        else:
            chunk = resolve_chunk_size(self.model, bs, avg_input, self.hp)
            a_d = alpha_d_model(self.model, bs, chunk)
            return base_step * (1 + a_d)

    def _migration_loop(self):
        """Bilateral migration: cost-driven Mitigation + Consolidation + role adaptation."""
        while True:
            yield self.env.timeout(self.hp.migration_scan_ms)
            migrations_this_scan = 0

            # ── Phase 1: Mitigation (observed TPOT + reaction time) ──
            candidates = []
            for dw in self.decode_workers:
                if not dw.decode_tasks:
                    continue
                for task in dw.decode_tasks:
                    if task.migrating or self.env.now < task.cooldown_until:
                        continue
                    o_remain = max(task.req.output_len - task.tokens_generated, 1)
                    # Skip requests about to finish: migration cost exceeds remaining lifetime
                    t_react = self.hp.migration_scan_ms + kv_transfer_ms(self.model, task.current_seq_len)
                    if o_remain * self.hp.slo_tpot_ms < t_react:
                        continue  # request finishes before migration completes
                    obs_tpot = task.last_step_ms if task.last_step_ms > 0 else 0
                    tau_mit = self.hp.slo_tpot_ms - t_react / o_remain
                    if obs_tpot > tau_mit:
                        candidates.append((obs_tpot, task, dw))
            candidates.sort(key=lambda x: -x[0])  # highest observed TPOT first

            for obs_tpot, task, src_dw in candidates:
                if migrations_this_scan >= self.hp.r_max:
                    break
                best_target = None
                best_tpot_after = float("inf")
                for dw in self.decode_workers:
                    if dw is src_dw:
                        continue
                    new_bs = len(dw.decode_tasks) + 1
                    base_step = decode_step_ms(self.model, new_bs)
                    mig_chunk = resolve_chunk_size(self.model, new_bs, self.hp.chunk_size, self.hp)
                    a_d_after = alpha_d_model(self.model, new_bs, mig_chunk)
                    tpot_after = base_step * (1 + a_d_after)
                    if dw.prefill_queue:
                        avg_pf = prefill_latency_ms(self.model, mig_chunk)
                        tpot_after += avg_pf * avg_pf / (avg_pf + base_step)
                    if (tpot_after <= self.hp.theta_dispatch * self.hp.slo_tpot_ms
                            and tpot_after < best_tpot_after):
                        best_tpot_after = tpot_after
                        best_target = dw
                if best_target is not None:
                    src_dw.decode_tasks.remove(task)
                    best_target.decode_tasks.append(task)
                    task.cooldown_until = self.env.now + self.hp.t_cool_ms
                    migrations_this_scan += 1

            # ── Phase 2: Consolidation (net benefit driven) ──
            # Compute V_free: value of freeing a node (Prefill queue pressure)
            avg_pf_queue = 0.0
            if self.prefill_workers:
                avg_pf_queue = np.mean([len(pw.prefill_queue) for pw in self.prefill_workers])
            avg_pf_queue_ms = avg_pf_queue * prefill_latency_ms(self.model, 512)
            n_prefill = max(len(self.prefill_workers), 1)
            v_free = avg_pf_queue_ms / n_prefill  # queue reduction from adding one Prefill node

            # Minimum benefit threshold: must exceed cheapest possible KV transfer
            min_kv_cost = kv_transfer_ms(self.model, 128)  # shortest possible sequence

            consol_candidates = []
            for dw in self.decode_workers:
                if not dw.decode_tasks:
                    continue
                # Compute migration cost for all tasks on this node
                total_cost = 0.0
                feasible = True
                for task in dw.decode_tasks:
                    if self.env.now < task.cooldown_until:
                        feasible = False
                        break
                    t_kv = kv_transfer_ms(self.model, task.current_seq_len)
                    # Estimate externality on best target
                    best_ext = float("inf")
                    for target_dw in self.decode_workers:
                        if target_dw is dw:
                            continue
                        new_bs = len(target_dw.decode_tasks) + 1
                        consol_chunk = resolve_chunk_size(self.model, new_bs, self.hp.chunk_size, self.hp)
                        a_d_after = alpha_d_model(self.model, new_bs, consol_chunk)
                        tpot_after = decode_step_ms(self.model, new_bs) * (1 + a_d_after)
                        if tpot_after <= self.hp.theta_dispatch * self.hp.slo_tpot_ms:
                            # Externality: interference increase on target
                            tau = prefill_latency_ms(self.model, consol_chunk) * (1 + alpha_p_model(self.model, new_bs, consol_chunk))
                            ext = len(target_dw.decode_tasks) * tau * a_d_after
                            best_ext = min(best_ext, ext)
                    if best_ext == float("inf"):
                        feasible = False
                        break
                    total_cost += t_kv + self.hp.lam * best_ext
                if feasible:
                    net_benefit = v_free - total_cost
                    if net_benefit > min_kv_cost:
                        consol_candidates.append((net_benefit, dw))

            consol_candidates.sort(key=lambda x: -x[0])  # highest benefit first

            for net_benefit, src_dw in consol_candidates:
                if migrations_this_scan >= self.hp.r_max:
                    break
                if len(self.decode_workers) <= 1:
                    break
                all_placed = True
                tasks_to_move = list(src_dw.decode_tasks)
                for task in tasks_to_move:
                    placed = False
                    for dw in self.decode_workers:
                        if dw is src_dw:
                            continue
                        new_bs = len(dw.decode_tasks) + 1
                        consol_chunk = resolve_chunk_size(self.model, new_bs, self.hp.chunk_size, self.hp)
                        a_d_after = alpha_d_model(self.model, new_bs, consol_chunk)
                        tpot_after = decode_step_ms(self.model, new_bs) * (1 + a_d_after)
                        if tpot_after <= self.hp.theta_dispatch * self.hp.slo_tpot_ms:
                            src_dw.decode_tasks.remove(task)
                            dw.decode_tasks.append(task)
                            task.cooldown_until = self.env.now + self.hp.t_cool_ms
                            migrations_this_scan += 1
                            placed = True
                            break
                    if not placed:
                        all_placed = False
                        break

                # Role adaptation (only for cbs_full)
                if (all_placed and not src_dw.decode_tasks
                        and self.system == "cbs_full"):
                    avg_queue = np.mean([len(pw.prefill_queue)
                                        for pw in self.prefill_workers]) if self.prefill_workers else 0
                    avg_queue_ms = avg_queue * prefill_latency_ms(self.model, 512)
                    if (avg_queue_ms > self.hp.t_role_thresh_ms
                            and len(self.decode_workers) > 1):
                        src_dw.role = "prefill"
                        self.env.process(self._prefill_loop(src_dw))


# ── Workload Generation ───────────────────────────────────────

def generate_requests(rate: float, duration_ms: float, model: str,
                      pattern: str = "uniform", seed: int = 42) -> List[Request]:
    rng = random.Random(seed)
    requests = []
    rid = 0
    t = 0.0
    while t < duration_ms:
        # Poisson inter-arrival
        if pattern == "bursty":
            cycle_pos = t % 30000  # 30s cycle
            effective_rate = rate * 4 if cycle_pos < 5000 else rate
        else:
            effective_rate = rate
        interval = rng.expovariate(effective_rate / 1000)  # convert to ms
        t += interval
        if t >= duration_ms:
            break

        if pattern == "long_context":
            input_len = rng.randint(1024, 4096)
            output_len = rng.randint(64, 256)
        else:
            input_len = rng.randint(128, 2048)
            output_len = rng.randint(32, 512)

        requests.append(Request(id=rid, arrival_ms=t, input_len=input_len,
                                output_len=output_len, model=model))
        rid += 1
    return requests


# ── Metrics ───────────────────────────────────────────────────

def compute_metrics(completed: List[Request], workers: List[Worker] = None,
                    warmup_ms: float = 120000):
    """Compute goodput, SLO attainment, P99 latencies, and GPU utilization."""
    stable = [r for r in completed if r.arrival_ms >= warmup_ms]
    if not stable:
        return {"goodput": 0, "slo_pct": 0, "p99_ttft": 0, "p99_tpot": 0,
                "n": 0, "gpu_util_pct": 0, "prefill_util_pct": 0, "decode_util_pct": 0}

    duration_s = (max(r.completion_ms for r in stable)
                  - min(r.arrival_ms for r in stable)) / 1000
    if duration_s <= 0:
        duration_s = 1.0

    slo_met = [r for r in stable if r.slo_met]
    ttfts = [r.ttft_ms for r in stable]
    tpots = [r.tpot_ms for r in stable if r.tpot_ms > 0]

    result = {
        "goodput": round(len(slo_met) / duration_s, 2),
        "slo_pct": round(100 * len(slo_met) / len(stable), 1),
        "p99_ttft": round(np.percentile(ttfts, 99), 1) if ttfts else 0,
        "p99_tpot": round(np.percentile(tpots, 99), 1) if tpots else 0,
        "n_completed": len(stable),
        "n_slo_met": len(slo_met),
    }

    # GPU utilization: fraction of time each worker spent computing
    if workers:
        prefill_workers = [w for w in workers if w.role == "prefill"]
        decode_workers = [w for w in workers if w.role == "decode"]

        def avg_util(ws):
            utils = [w.busy_ms / max(w.total_ms, 1) * 100 for w in ws]
            return round(np.mean(utils), 1) if utils else 0.0

        result["prefill_util_pct"] = avg_util(prefill_workers)
        result["decode_util_pct"] = avg_util(decode_workers)
        result["gpu_util_pct"] = avg_util(workers)  # cluster-wide average

    return result


# ── Main ──────────────────────────────────────────────────────

SYSTEMS = ["disagg_static", "coloc_sarathi", "cbs_nomig", "cbs_norole", "cbs_full"]
CONFIGS = {
    "8node": {"n_prefill": 2, "n_decode": 6},
    "16node": {"n_prefill": 4, "n_decode": 12},
    "2p2d": {"n_prefill": 2, "n_decode": 2},
}

def run_experiment(config_name: str, model: str, rate: float, system: str,
                   pattern: str = "uniform", duration_s: float = 600,
                   seed: int = 42, hp: HyperParams = None) -> dict:
    cfg = CONFIGS[config_name]
    if hp is None:
        hp = HyperParams()
    duration_ms = duration_s * 1000

    requests = generate_requests(rate, duration_ms, model, pattern, seed)
    sim = Simulator(cfg["n_prefill"], cfg["n_decode"], model, hp, system, seed)
    sim.run(requests, duration_ms)

    metrics = compute_metrics(sim.completed, sim.workers)
    metrics["system"] = system
    metrics["model"] = model
    metrics["rate"] = rate
    metrics["config"] = config_name
    metrics["pattern"] = pattern
    metrics["seed"] = seed
    metrics["cbs_decisions"] = sim.cbs_decisions
    return metrics


def aggregate_with_ci(results: List[dict], ci: float = 0.95) -> dict:
    """Aggregate multiple seed runs: mean + CI for numeric metrics."""
    from scipy import stats as sp_stats
    numeric_keys = ["goodput", "slo_pct", "p99_ttft", "p99_tpot",
                    "gpu_util_pct", "prefill_util_pct", "decode_util_pct"]
    agg = {}
    n = len(results)
    for k in numeric_keys:
        vals = [r.get(k, 0) for r in results]
        mean = np.mean(vals)
        if n > 1:
            se = sp_stats.sem(vals)
            t_val = sp_stats.t.ppf((1 + ci) / 2, n - 1)
            half = se * t_val
        else:
            half = 0.0
        agg[k] = round(float(mean), 2)
        agg[f"{k}_ci"] = round(float(half), 2)
        agg[f"{k}_std"] = round(float(np.std(vals, ddof=1)) if n > 1 else 0.0, 3)
    # Copy non-numeric fields from first result
    for k in ["system", "model", "rate", "config", "pattern"]:
        agg[k] = results[0].get(k)
    agg["n_seeds"] = n
    return agg


def run_experiment_multi_seed(config_name: str, model: str, rate: float,
                              system: str, pattern: str = "uniform",
                              duration_s: float = 600, n_seeds: int = 10,
                              jitter_config: dict = None,
                              save_decisions: bool = False) -> dict:
    """Run experiment with multiple seeds, return aggregated metrics."""
    per_seed = []
    all_decisions = []
    for i in range(n_seeds):
        seed = 42 + i
        hp = HyperParams()
        if jitter_config:
            hp.jitter_prefill = jitter_config.get("prefill", 0.0)
            hp.jitter_decode = jitter_config.get("decode", 0.0)
            hp.jitter_kv = jitter_config.get("kv", 0.0)
        result = run_experiment(config_name, model, rate, system, pattern,
                                duration_s, seed, hp)
        decisions = result.pop("cbs_decisions", [])
        if save_decisions:
            for d in decisions:
                d["seed"] = seed
            all_decisions.extend(decisions)
        per_seed.append(result)
    agg = aggregate_with_ci(per_seed)
    if jitter_config:
        agg["jitter"] = jitter_config
    if save_decisions:
        agg["_decisions"] = all_decisions
    return agg


def main():
    parser = argparse.ArgumentParser(description="Paper-faithful CBS simulator")
    parser.add_argument("--config", default="8node", choices=list(CONFIGS.keys()))
    parser.add_argument("--models", default="qwen2.5-7b,llama-3.1-8b")
    parser.add_argument("--rates", default="4,8,12,16")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--pattern", default="uniform")
    parser.add_argument("--duration", type=float, default=600, help="seconds")
    parser.add_argument("--output", default="sim_results.json")
    # W2: jitter sensitivity
    parser.add_argument("--jitter-prefill", type=float, default=0.0)
    parser.add_argument("--jitter-decode", type=float, default=0.0)
    parser.add_argument("--jitter-kv", type=float, default=0.0)
    # Multi-seed
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds (1=single deterministic run)")
    # W5: decision logging
    parser.add_argument("--save-decisions", action="store_true",
                        help="Save CBS decision log to separate JSON")
    args = parser.parse_args()

    models = args.models.split(",")
    rates = [float(r) for r in args.rates.split(",")]
    systems = args.systems.split(",")

    jitter_config = None
    if args.jitter_prefill > 0 or args.jitter_decode > 0 or args.jitter_kv > 0:
        jitter_config = {"prefill": args.jitter_prefill,
                         "decode": args.jitter_decode,
                         "kv": args.jitter_kv}

    all_results = []
    all_decisions = []
    total = len(models) * len(rates) * len(systems)
    done = 0

    for model in models:
        for rate in rates:
            for system in systems:
                done += 1
                seeds_str = f"x{args.n_seeds}" if args.n_seeds > 1 else ""
                jitter_str = ""
                if jitter_config:
                    jitter_str = f" jitter=({args.jitter_prefill},{args.jitter_decode},{args.jitter_kv})"
                print(f"[{done}/{total}] {model} rate={rate} {system}{seeds_str}{jitter_str} ...",
                      end=" ", flush=True)
                t0 = time.time()

                if args.n_seeds > 1:
                    result = run_experiment_multi_seed(
                        args.config, model, rate, system, args.pattern,
                        args.duration, args.n_seeds, jitter_config,
                        args.save_decisions)
                    decisions = result.pop("_decisions", [])
                    if decisions:
                        all_decisions.extend(decisions)
                    elapsed = time.time() - t0
                    ci_str = f"±{result.get('goodput_ci', 0)}"
                    print(f"goodput={result['goodput']}{ci_str} slo={result['slo_pct']}% ({elapsed:.1f}s)")
                else:
                    hp = HyperParams()
                    if jitter_config:
                        hp.jitter_prefill = jitter_config.get("prefill", 0.0)
                        hp.jitter_decode = jitter_config.get("decode", 0.0)
                        hp.jitter_kv = jitter_config.get("kv", 0.0)
                    result = run_experiment(args.config, model, rate, system,
                                            args.pattern, args.duration, 42, hp)
                    decisions = result.pop("cbs_decisions", [])
                    if args.save_decisions and decisions:
                        all_decisions.extend(decisions)
                    elapsed = time.time() - t0
                    print(f"goodput={result['goodput']} slo={result['slo_pct']}% ({elapsed:.1f}s)")

                all_results.append(result)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if all_decisions and args.save_decisions:
        dec_path = out_path.with_name(out_path.stem + "_decisions.json")
        with open(dec_path, "w") as f:
            json.dump(all_decisions, f)
        print(f"Decision log saved to {dec_path} ({len(all_decisions)} entries)")

    # Print summary table
    has_ci = args.n_seeds > 1
    hdr = f"{'Model':<16} {'Rate':>6} {'System':<16} {'Goodput':>8}"
    if has_ci:
        hdr += f" {'±CI':>6}"
    hdr += f" {'SLO%':>6} {'P99 TTFT':>9} {'P99 TPOT':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in all_results:
        line = (f"{r['model']:<16} {r['rate']:>6.0f} {r['system']:<16} "
                f"{r['goodput']:>8.2f}")
        if has_ci:
            line += f" {r.get('goodput_ci', 0):>5.2f}"
        line += (f" {r['slo_pct']:>5.1f}% {r['p99_ttft']:>8.1f} "
                 f"{r['p99_tpot']:>8.1f}")
        print(line)


if __name__ == "__main__":
    main()

