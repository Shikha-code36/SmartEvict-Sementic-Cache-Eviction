"""Eviction policies.

- LRUPolicy / FIFOPolicy: faithful reimplementations of the naive baselines
  (GPTCache's cache_size-count eviction is LRU/FIFO over entry count).
- GDSFPolicy / CostWeightedRecencyPolicy: non-learned, cost-aware baselines.
  These exist to isolate "does considering cost at all help" from "does
  learning help beyond a hand-written formula" — without them the learned
  net is only ever compared against recency-only policies, which can't
  answer that question. Both operate over the same K-tail candidate pool
  as LearnedPolicy/OraclePolicy for a fair comparison.
- LearnedPolicy: Cold-RL pattern — sample the K coldest entries (K-tail of
  the LRU list), score them with the dueling net, evict argmin-Q. Any
  exception or missing model => instant fallback to LRU. Fallbacks are
  counted so the benchmark can report them.
- OraclePolicy: clairvoyant — evicts the K-tail candidate with the least
  *actual* discounted future demand (needs precomputed future-match lists).
  Not deployable; it's the honesty ceiling for the benchmark.
"""
from __future__ import annotations

import numpy as np

from smartevict.features.extract import candidates_features


class BasePolicy:
    name = "base"

    def bind(self, cache, hooks=None):
        self.cache, self.hooks = cache, hooks

    def choose_victim(self, cache, now: float) -> int:
        raise NotImplementedError


class LRUPolicy(BasePolicy):
    name = "lru"

    def choose_victim(self, cache, now):
        return cache.lru_order[0]


class FIFOPolicy(BasePolicy):
    name = "fifo"

    def choose_victim(self, cache, now):
        return cache.insert_order[0]


class GDSFPolicy(BasePolicy):
    """Greedy-Dual-Size-Frequency, size=1 (this cache is entry-count bounded,
    not byte-bounded, so the size term drops out): H(e) = L + hits(e) * cost(e).
    L is an inflation clock set to the max H evicted so far — the standard
    GDSF trick that stops a stale high-priority entry from being permanently
    protected by a score it earned long ago. Classic non-learned cost-aware
    caching baseline (frequency x cost), evaluated over the same K-tail pool
    as LearnedPolicy/OraclePolicy."""
    name = "gdsf"

    def __init__(self, k_tail: int = 8):
        self.k = k_tail
        self.L = 0.0

    def choose_victim(self, cache, now):
        cands = cache.lru_order[: self.k]
        scores = []
        for c in cands:
            e = cache.entries[c]
            scores.append(self.L + e.hit_count * e.response_tokens)
        i = int(np.argmin(scores))
        self.L = max(self.L, scores[i])
        return cands[i]


class CostWeightedRecencyPolicy(BasePolicy):
    """score(e) = cost(e) / (1 + idle_time(e)), evict argmin. Cruder than
    GDSF and not from the literature, but maps directly onto the same
    age/cost signals LearnedPolicy consumes as features — a sanity-check
    baseline for "would a simple formula over the same signals do?"."""
    name = "cost_weighted_recency"

    def __init__(self, k_tail: int = 8):
        self.k = k_tail

    def choose_victim(self, cache, now):
        cands = cache.lru_order[: self.k]
        scores = []
        for c in cands:
            e = cache.entries[c]
            idle = max(now - e.last_access_t, 0.0)
            scores.append(e.response_tokens / (1.0 + idle))
        return cands[int(np.argmin(scores))]


class LearnedPolicy(BasePolicy):
    """K-tail candidate sampling + dueling net + hard fallback to LRU."""
    name = "learned"

    def __init__(self, net, k_tail: int = 8, feature_indices=None):
        """feature_indices: optional column subset matching what the net was
        trained on (see build_dataset's feature_indices) -- for
        feature-ablation studies. None = all 6 features."""
        self.net, self.k = net, k_tail
        self.feature_indices = feature_indices
        self.fallbacks = 0
        self.decisions = 0

    def choose_victim(self, cache, now):
        self.decisions += 1
        try:
            if self.net is None:
                raise RuntimeError("no model loaded")
            cands = cache.lru_order[: self.k]
            feats = candidates_features(cache, cands, now)
            if self.feature_indices is not None:
                feats = feats[:, self.feature_indices]
            q = self.net.q_values(feats)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("non-finite Q values")
            return cands[int(np.argmin(q))]
        except Exception:
            self.fallbacks += 1
            return cache.lru_order[0]  # safe default: plain LRU


class OraclePolicy(BasePolicy):
    """Clairvoyant upper bound (also used to label training data).

    future_matches[req_idx] = sorted future request indices whose embedding
    matches the entry inserted by request req_idx (precomputed offline from
    an infinite-cache demand replay). Pass this policy object itself as
    `hooks=` to run_simulation so on_insert can map eid -> req_idx.
    """
    name = "oracle"

    def __init__(self, future_matches: dict[int, list[int]],
                 req_times: np.ndarray, req_tokens: np.ndarray,
                 k_tail: int = 8, gamma: float = 0.999, horizon: float = 5000.0):
        self.k, self.gamma, self.h = k_tail, gamma, horizon
        self.future_matches = future_matches
        self.req_times = req_times
        self.req_tokens = req_tokens
        self.req_of_eid: dict[int, int] = {}

    # hook: called by run_simulation after every insert
    def on_insert(self, eid: int, req_idx: int):
        self.req_of_eid[eid] = req_idx

    def demand(self, eid: int, now: float) -> float:
        ri = self.req_of_eid.get(eid)
        if ri is None:
            return 0.0
        ms = self.future_matches.get(ri, [])
        tok = float(self.req_tokens[ri])
        d = 0.0
        for j in ms:
            dt = self.req_times[j] - now
            if dt <= 0:
                continue
            if dt > self.h:
                break
            d += (self.gamma ** dt) * tok
        return d

    def choose_victim(self, cache, now):
        cands = cache.lru_order[: self.k]
        demands = [self.demand(c, now) for c in cands]
        return cands[int(np.argmin(demands))]
