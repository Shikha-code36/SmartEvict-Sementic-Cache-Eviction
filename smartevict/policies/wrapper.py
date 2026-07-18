"""LearnedSemanticCache — the standalone, backend-agnostic wrapper (Plan §9).

Sits between your app and the vector store; replaces ONLY the eviction
decision. Similarity search and storage are delegated to FAISS (v1 backend).
Matches the interface proposed in the plan:

    cache = LearnedSemanticCache(
        embedding_fn=my_embed,          # list[str] -> np.ndarray (L2-normed)
        max_size=1000,
        eviction_policy="learned",      # or "lru" / "fifo"
        model_path="results/learned_policy.npz",
        sim_threshold=0.8,
    )
    cache.set(prompt, response)
    hit = cache.get(prompt)             # -> response str | None

Safety behavior (Cold-RL pattern): every learned decision is wrapped in
try/except; any failure (missing model, bad weights, non-finite scores)
falls back to plain LRU. `cache.stats()` reports fallback counts.

FAISS notes: IndexFlatIP over normalized vectors = cosine similarity.
Deletions use IDMap remove_ids. If faiss is not installed, a brute-force
NumPy store is used transparently (same semantics, fine up to ~100K entries).
"""
from __future__ import annotations

import time
import numpy as np

from smartevict.features.extract import entry_features
from smartevict.model.dueling_net import DuelingEvictionNet

try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False


# --------------------------------------------------------------------------
class _FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, eid: int, vec: np.ndarray):
        self.index.add_with_ids(vec[None, :].astype(np.float32),
                                np.array([eid], dtype=np.int64))

    def remove(self, eid: int):
        self.index.remove_ids(np.array([eid], dtype=np.int64))

    def nearest(self, vec: np.ndarray):
        if self.index.ntotal == 0:
            return None, -1.0
        D, I = self.index.search(vec[None, :].astype(np.float32), 1)
        return (int(I[0, 0]), float(D[0, 0])) if I[0, 0] != -1 else (None, -1.0)


class _NumpyStore:
    def __init__(self, dim: int):
        self.vecs: dict[int, np.ndarray] = {}

    def add(self, eid, vec):
        self.vecs[eid] = vec.astype(np.float32)

    def remove(self, eid):
        self.vecs.pop(eid, None)

    def nearest(self, vec):
        if not self.vecs:
            return None, -1.0
        ids = list(self.vecs)
        sims = np.stack([self.vecs[i] for i in ids]) @ vec
        j = int(np.argmax(sims))
        return ids[j], float(sims[j])


# --------------------------------------------------------------------------
class _Entry:
    __slots__ = ("eid", "prompt", "response", "response_tokens",
                 "insert_t", "last_access_t", "hit_count", "access_ts", "emb")

    def __init__(self, eid, prompt, response, tokens, now, emb):
        self.eid, self.prompt, self.response = eid, prompt, response
        self.response_tokens = tokens
        self.insert_t = self.last_access_t = now
        self.hit_count, self.access_ts = 0, []
        self.emb = emb


class LearnedSemanticCache:
    def __init__(self, embedding_fn, max_size: int = 1000,
                 eviction_policy: str = "learned", model_path: str | None = None,
                 sim_threshold: float = 0.8, k_tail: int = 8,
                 token_counter=None, clock=None):
        assert eviction_policy in ("learned", "lru", "fifo")
        self.embed = embedding_fn
        self.max_size = max_size
        self.policy_name = eviction_policy
        self.thr = sim_threshold
        self.k = k_tail
        self.count_tokens = token_counter or (lambda s: max(1, len(s) // 4))
        self.clock = clock or time.monotonic

        self.net = None
        if eviction_policy == "learned" and model_path:
            try:
                self.net = DuelingEvictionNet.load(model_path)
            except Exception:
                self.net = None  # will fall back to LRU at decision time

        self.entries: dict[int, _Entry] = {}
        self.lru_order: list[int] = []
        self.insert_order: list[int] = []
        self.store = None
        self._next_id = 0
        self._stats = {"gets": 0, "hits": 0, "sets": 0, "evictions": 0,
                       "fallbacks": 0, "tokens_saved": 0}

    # -- internals ---------------------------------------------------------
    def _ensure_store(self, dim: int):
        if self.store is None:
            self.store = (_FaissStore(dim) if _HAVE_FAISS else _NumpyStore(dim))

    def _choose_victim(self, now: float) -> int:
        if self.policy_name == "fifo":
            return self.insert_order[0]
        if self.policy_name == "lru" or self.net is None:
            if self.policy_name == "learned":
                self._stats["fallbacks"] += 1
            return self.lru_order[0]
        try:
            cands = self.lru_order[: self.k]
            feats = np.stack([entry_features(self.entries[c], now) for c in cands])
            q = self.net.q_values(feats)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("non-finite Q")
            return cands[int(np.argmin(q))]
        except Exception:
            self._stats["fallbacks"] += 1
            return self.lru_order[0]

    def _evict(self, eid: int):
        del self.entries[eid]
        self.lru_order.remove(eid)
        self.insert_order.remove(eid)
        self.store.remove(eid)
        self._stats["evictions"] += 1

    # -- public api --------------------------------------------------------
    def get(self, prompt: str) -> str | None:
        self._stats["gets"] += 1
        if not self.entries:
            return None
        vec = np.asarray(self.embed([prompt])[0], dtype=np.float32)
        eid, sim = self.store.nearest(vec)
        if eid is None or sim < self.thr or eid not in self.entries:
            return None
        e = self.entries[eid]
        now = self.clock()
        e.hit_count += 1
        e.access_ts.append(now)
        e.last_access_t = now
        self.lru_order.remove(eid)
        self.lru_order.append(eid)
        self._stats["hits"] += 1
        self._stats["tokens_saved"] += e.response_tokens
        return e.response

    def set(self, prompt: str, response: str):
        self._stats["sets"] += 1
        vec = np.asarray(self.embed([prompt])[0], dtype=np.float32)
        self._ensure_store(len(vec))
        now = self.clock()
        if len(self.entries) >= self.max_size:
            self._evict(self._choose_victim(now))
        eid = self._next_id
        self._next_id += 1
        self.entries[eid] = _Entry(eid, prompt, response,
                                   self.count_tokens(response), now, vec)
        self.lru_order.append(eid)
        self.insert_order.append(eid)
        self.store.add(eid, vec)

    def stats(self) -> dict:
        s = dict(self._stats)
        s["size"] = len(self.entries)
        s["hit_ratio"] = s["hits"] / max(1, s["gets"])
        s["backend"] = "faiss" if isinstance(self.store, _FaissStore) else "numpy"
        return s
