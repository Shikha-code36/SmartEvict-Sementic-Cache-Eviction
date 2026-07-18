"""Bounded semantic cache simulator.

Replays a request trace through a fixed-capacity semantic cache:
  - hit/miss = cosine similarity of embeddings vs. a threshold
    (same shape as GPTCache's similarity evaluator)
  - on miss + full cache, the EvictionPolicy chooses a victim
  - "cost saved" on a hit = the cached entry's response_tokens
    (regeneration-cost proxy; no live LLM calls, log-replay style)

The simulator exposes per-entry stats (insert time, access times, hit count,
response tokens) — exactly the raw material the feature extractor needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CacheEntry:
    eid: int
    text: str
    emb: np.ndarray
    response_tokens: int
    insert_t: float
    last_access_t: float
    hit_count: int = 0
    access_ts: list = field(default_factory=list)  # times of hits


@dataclass
class SimResult:
    requests: int
    hits: int
    tokens_saved: int
    tokens_total: int
    evictions: int

    @property
    def hit_ratio(self) -> float:
        return self.hits / max(1, self.requests)

    @property
    def tokens_saved_frac(self) -> float:
        return self.tokens_saved / max(1, self.tokens_total)


class SemanticCache:
    def __init__(self, max_size: int, sim_threshold: float, policy):
        self.max_size = max_size
        self.thr = sim_threshold
        self.policy = policy
        self.entries: dict[int, CacheEntry] = {}
        self.lru_order: list[int] = []   # front = coldest
        self.insert_order: list[int] = []
        self._mat: np.ndarray | None = None
        self._mat_ids: list[int] = []
        self._dirty = True
        self._next_id = 0

    # -- internal ---------------------------------------------------------
    def _matrix(self):
        if self._dirty:
            self._mat_ids = list(self.entries.keys())
            self._mat = (np.stack([self.entries[i].emb for i in self._mat_ids])
                         if self._mat_ids else np.zeros((0, 1), dtype=np.float32))
            self._dirty = False
        return self._mat, self._mat_ids

    def _touch(self, eid: int):
        self.lru_order.remove(eid)
        self.lru_order.append(eid)

    # -- api --------------------------------------------------------------
    def lookup(self, emb: np.ndarray) -> tuple[int | None, float]:
        mat, ids = self._matrix()
        if len(ids) == 0:
            return None, -1.0
        sims = mat @ emb
        j = int(np.argmax(sims))
        s = float(sims[j])
        return (ids[j], s) if s >= self.thr else (None, s)

    def insert(self, text: str, emb: np.ndarray, tokens: int, now: float) -> int | None:
        """Returns evicted eid if an eviction happened."""
        evicted = None
        if len(self.entries) >= self.max_size:
            victim = self.policy.choose_victim(self, now)
            self._remove(victim)
            evicted = victim
        eid = self._next_id
        self._next_id += 1
        self.entries[eid] = CacheEntry(eid, text, emb, tokens, now, now)
        self.lru_order.append(eid)
        self.insert_order.append(eid)
        self._dirty = True
        return evicted

    def _remove(self, eid: int):
        del self.entries[eid]
        self.lru_order.remove(eid)
        self.insert_order.remove(eid)
        self._dirty = True

    def record_hit(self, eid: int, now: float):
        e = self.entries[eid]
        e.hit_count += 1
        e.access_ts.append(now)
        e.last_access_t = now
        self._touch(eid)


def run_simulation(records: list[dict], embeddings: np.ndarray, max_size: int,
                   sim_threshold: float, policy, hooks=None) -> SimResult:
    """Replay trace. `hooks.on_eviction(cache, now, victim, candidates)` and
    `hooks.on_step(i)` are optional callbacks for trajectory logging."""
    cache = SemanticCache(max_size, sim_threshold, policy)
    if hasattr(policy, "bind"):
        policy.bind(cache, hooks)
    hits = tokens_saved = tokens_total = evictions = 0
    for i, rec in enumerate(records):
        now, emb = rec["t"], embeddings[i]
        tokens_total += rec["response_tokens"]
        eid, _ = cache.lookup(emb)
        if eid is not None:
            hits += 1
            tokens_saved += cache.entries[eid].response_tokens
            cache.record_hit(eid, now)
        else:
            ev = cache.insert(rec["text"], emb, rec["response_tokens"], now)
            if ev is not None:
                evictions += 1
            if hooks and hasattr(hooks, "on_insert"):
                hooks.on_insert(cache._next_id - 1, i)
        if hooks and hasattr(hooks, "on_step"):
            hooks.on_step(i)
    return SimResult(len(records), hits, tokens_saved, tokens_total, evictions)
