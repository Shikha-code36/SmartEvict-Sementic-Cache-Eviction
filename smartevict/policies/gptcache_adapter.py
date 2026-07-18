"""GPTCache integration: drop the learned eviction policy into a real
GPTCache `Cache`, not just this repo's own simulator/wrapper.

Why this needs its own adapter instead of a config string: GPTCache's
built-in eviction backend (`gptcache.manager.eviction.memory_cache
.MemoryCacheEviction`) hardcodes `policy` to one of a fixed set of
`cachetools` classes (LRU/LFU/FIFO/RR) -- there is no name for "custom".
But the underlying `EvictionBase` GPTCache eviction backends implement is
just a 3-method ABC (`put`/`get`/`policy`), and `get_data_manager(...,
eviction_base=...)` accepts an already-constructed `EvictionBase`
*instance* (see gptcache/manager/factory.py) -- so a subclass of that ABC
drops straight in without patching GPTCache itself:

    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    from policies.gptcache_adapter import LearnedEviction

    eviction = LearnedEviction(model_path="results/learned_policy.npz", maxsize=1000)
    data_manager = get_data_manager(CacheBase("sqlite"),
                                    VectorBase("faiss", dimension=384),
                                    eviction_base=eviction,
                                    max_size=1000)

Caveat: GPTCache's `put`/`get` hooks only pass opaque cache-storage row
ids, not the prompt/response text or token counts, so this adapter tracks
its own per-id bookkeeping (insert time, last-access time, hit count,
access timestamps) -- the same fields `features/extract.py` reads off the
simulator's `CacheEntry`. Regeneration cost (response tokens) isn't
visible to GPTCache's eviction hooks at all, so the caller should report
it via `note_cost(id, tokens)` right after inserting; entries with no
reported cost default to a flat 1.0, and the policy degrades gracefully
to a recency/frequency-only signal for those (no cost dimension, still
never worse than the hard LRU fallback below).
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

import numpy as np

from smartevict.model.dueling_net import DuelingEvictionNet

from gptcache.manager.eviction.base import EvictionBase


class _EntryMeta:
    __slots__ = ("insert_t", "last_access_t", "hit_count", "access_ts", "response_tokens")

    def __init__(self, now: float, response_tokens: float):
        self.insert_t = now
        self.last_access_t = now
        self.hit_count = 0
        self.access_ts: list[float] = []
        self.response_tokens = response_tokens


def _entry_features(e: _EntryMeta, now: float) -> np.ndarray:
    """Same 6 features as features/extract.py:entry_features, computed off
    this adapter's own bookkeeping instead of the simulator's CacheEntry."""
    age = max(now - e.insert_t, 1e-6)
    idle = max(now - e.last_access_t, 0.0)
    if e.hit_count >= 1:
        ts = [e.insert_t] + e.access_ts
        gaps = np.diff(ts)
        mean_gap = float(np.mean(gaps)) if len(gaps) else age
    else:
        mean_gap = age
    return np.array([
        np.log1p(age),
        np.log1p(e.response_tokens),
        np.log1p(e.hit_count),
        np.log1p(idle),
        np.log1p(mean_gap),
        min(idle / age, 1.0),
    ], dtype=np.float32)


class LearnedEviction(EvictionBase):
    """GPTCache `EvictionBase` backed by the dueling-net policy benchmarked
    in this repo. Same K-tail sampling + hard LRU fallback as
    `policies.eviction.LearnedPolicy`, adapted to GPTCache's put(ids)/get(id)
    hooks instead of a bound simulator cache.
    """

    def __init__(self, model_path: Optional[str] = None, maxsize: int = 1000,
                clean_size: int = 0, k_tail: int = 8,
                on_evict: Optional[Callable[[List[Any]], None]] = None,
                default_cost: float = 1.0):
        self.maxsize = maxsize
        self.clean_size = clean_size or max(1, int(maxsize * 0.2))
        self.k_tail = k_tail
        self.on_evict = on_evict
        self.default_cost = default_cost
        self.fallbacks = 0
        self.decisions = 0

        self.net: Optional[DuelingEvictionNet] = None
        if model_path:
            try:
                self.net = DuelingEvictionNet.load(model_path)
            except Exception:
                self.net = None  # missing/broken model -> pure LRU (see _choose_victim)

        self._meta: dict[Any, _EntryMeta] = {}
        self._lru_order: list[Any] = []  # front = coldest, back = most recent
        self._pending_cost: dict[Any, float] = {}

    # -- optional: report real regeneration cost (response tokens) ----------
    def note_cost(self, obj_id: Any, response_tokens: float):
        """Call right after inserting `obj_id` (e.g. after `cache.import_data`)
        so eviction can weigh this entry's true regeneration cost. Safe to
        call before `put()` sees the id too -- the cost is staged until then."""
        if obj_id in self._meta:
            self._meta[obj_id].response_tokens = response_tokens
        else:
            self._pending_cost[obj_id] = response_tokens

    # -- EvictionBase interface ----------------------------------------------
    def put(self, objs: List[Any]):
        now = time.time()
        for obj in objs:
            cost = self._pending_cost.pop(obj, self.default_cost)
            self._meta[obj] = _EntryMeta(now, cost)
            self._lru_order.append(obj)
        self._evict_if_needed(now)

    def get(self, obj: Any):
        e = self._meta.get(obj)
        if e is None:
            return None
        now = time.time()
        e.hit_count += 1
        e.access_ts.append(now)
        e.last_access_t = now
        self._lru_order.remove(obj)
        self._lru_order.append(obj)
        return obj

    @property
    def policy(self) -> str:
        return "learned"

    # -- eviction logic --------------------------------------------------------
    def _choose_victim(self, now: float) -> Any:
        self.decisions += 1
        try:
            if self.net is None:
                raise RuntimeError("no model loaded")
            cands = self._lru_order[: self.k_tail]
            feats = np.stack([_entry_features(self._meta[c], now) for c in cands])
            q = self.net.q_values(feats)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("non-finite Q values")
            return cands[int(np.argmin(q))]
        except Exception:
            self.fallbacks += 1
            return self._lru_order[0]  # safe default: plain LRU

    def _evict_if_needed(self, now: float):
        evicted = []
        while len(self._lru_order) > self.maxsize:
            victim = self._choose_victim(now)
            self._lru_order.remove(victim)
            del self._meta[victim]
            evicted.append(victim)
        if evicted and self.on_evict:
            self.on_evict(evicted)
