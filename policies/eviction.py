"""Eviction policies.

- LRUPolicy / FIFOPolicy: faithful reimplementations of the naive baselines
  (GPTCache's cache_size-count eviction is LRU/FIFO over entry count).
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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import candidates_features  # noqa: E402


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


class LearnedPolicy(BasePolicy):
    """K-tail candidate sampling + dueling net + hard fallback to LRU."""
    name = "learned"

    def __init__(self, net, k_tail: int = 8):
        self.net, self.k = net, k_tail
        self.fallbacks = 0
        self.decisions = 0

    def choose_victim(self, cache, now):
        self.decisions += 1
        try:
            if self.net is None:
                raise RuntimeError("no model loaded")
            cands = cache.lru_order[: self.k]
            feats = candidates_features(cache, cands, now)
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
