"""Offline training pipeline (Phases 2-3 of the plan).

1. Demand precompute (infinite-cache replay, chunked):
   for every request i, find all future requests j with cos(e_i, e_j) >= thr.
   This gives each potential cache entry's *demand curve* independent of any
   eviction policy — sidestepping the off-policy counterfactual problem of
   "we evicted it under LRU so we never saw its reuse".

2. Trajectory collection: replay the trace under LRU (behavior policy).
   At every eviction decision, snapshot the K-tail candidates' features.

3. Targets: for each candidate at decision time t,
       y = log1p( sum_{j in matches, t < t_j <= t+H} gamma^(t_j - t) * tokens )
   i.e. discounted future regeneration cost saved if kept. Eviction ends an
   entry's episode, so this is the fitted-Q target for the terminal-on-evict
   formulation (POC simplification of full multi-step Q-learning).

4. Train the dueling net with MSE on (features -> y), grouped per decision.
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cache_sim import run_simulation                 # noqa: E402
from features.extract import candidates_features, N_FEATURES   # noqa: E402
from policies.eviction import LRUPolicy                        # noqa: E402
from model.dueling_net import DuelingEvictionNet               # noqa: E402


# ---------------------------------------------------------------------------
def compute_future_matches(emb: np.ndarray, thr: float,
                           chunk: int = 1024, max_matches: int = 200):
    """future_matches[i] = sorted j>i with cos-sim >= thr (capped)."""
    n = emb.shape[0]
    out: dict[int, list[int]] = {}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = emb[s:e] @ emb.T                     # (chunk, n)
        for r in range(e - s):
            i = s + r
            js = np.nonzero(sims[r, i + 1:] >= thr)[0] + i + 1
            if len(js):
                out[i] = js[:max_matches].tolist()
    return out


def demand_at(future_matches, req_times, req_tokens, req_idx: int,
              now: float, gamma: float, horizon: float) -> float:
    d = 0.0
    for j in future_matches.get(req_idx, []):
        dt = req_times[j] - now
        if dt <= 0:
            continue
        if dt > horizon:
            break
        d += (gamma ** dt) * float(req_tokens[req_idx])
    return d


# ---------------------------------------------------------------------------
class TrajectoryRecorder:
    """hooks object for run_simulation: wraps LRU as behavior policy and
    records (candidate features, candidate req-indices, decision time)."""

    def __init__(self, k_tail: int):
        self.k = k_tail
        self.req_of_eid: dict[int, int] = {}
        self.decisions: list[tuple[np.ndarray, list[int], float]] = []

    def on_insert(self, eid, req_idx):
        self.req_of_eid[eid] = req_idx


class RecordingLRUPolicy(LRUPolicy):
    def __init__(self, recorder: TrajectoryRecorder):
        self.rec = recorder

    def choose_victim(self, cache, now):
        cands = cache.lru_order[: self.rec.k]
        feats = candidates_features(cache, cands, now)
        ridx = [self.rec.req_of_eid[c] for c in cands]
        self.rec.decisions.append((feats, ridx, now))
        return cache.lru_order[0]


# ---------------------------------------------------------------------------
def build_dataset(records, emb, max_size, thr, k_tail,
                  gamma=0.999, horizon=5000.0, future_matches=None):
    if future_matches is None:
        future_matches = compute_future_matches(emb, thr)
    req_times = np.array([r["t"] for r in records])
    req_tokens = np.array([r["response_tokens"] for r in records])

    rec = TrajectoryRecorder(k_tail)
    run_simulation(records, emb, max_size, thr, RecordingLRUPolicy(rec), hooks=rec)

    X, y, groups = [], [], []
    for gi, (feats, ridx, now) in enumerate(rec.decisions):
        for f, ri in zip(feats, ridx):
            X.append(f)
            y.append(np.log1p(demand_at(future_matches, req_times, req_tokens,
                                        ri, now, gamma, horizon)))
            groups.append(gi)
    return (np.stack(X).astype(np.float32), np.array(y, np.float32),
            np.array(groups), future_matches)


def train_model(X, y, groups, epochs=8, batch_decisions=256,
                lr=1e-3, seed=0, val_frac=0.15, verbose=True):
    net = DuelingEvictionNet(n_features=N_FEATURES, seed=seed)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    n_val = max(1, int(len(ug) * val_frac))
    val_g, tr_g = set(ug[:n_val].tolist()), ug[n_val:]
    val_mask = np.isin(groups, list(val_g))
    Xv, yv, gv = X[val_mask], y[val_mask], groups[val_mask]

    hist = []
    for ep in range(epochs):
        rng.shuffle(tr_g)
        losses = []
        for s in range(0, len(tr_g), batch_decisions):
            gs = set(tr_g[s:s + batch_decisions].tolist())
            m = np.isin(groups, list(gs))
            losses.append(net.train_batch(X[m], y[m], groups[m], lr=lr))
        # val loss (forward only): reuse train_batch math w/o update — cheap way:
        qv = _forward_q(net, Xv, gv)
        vloss = float(np.mean((qv - yv) ** 2))
        hist.append((float(np.mean(losses)), vloss))
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  train_mse={hist[-1][0]:.4f}  val_mse={vloss:.4f}")
    return net, hist


def _forward_q(net, X, groups):
    q = np.zeros(len(X), np.float32)
    for g in np.unique(groups):
        m = groups == g
        q[m] = net.q_values(X[m])
    return q
