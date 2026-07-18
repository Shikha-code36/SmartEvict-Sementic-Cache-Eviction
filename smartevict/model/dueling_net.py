"""Tiny dueling network, pure NumPy (no torch dependency).

Architecture mirrors Cold-RL's: shared trunk 128 -> 64, then a dueling split:
  - value head V(context): sees pooled context of all K candidates
  - advantage head A(candidate): sees each candidate's own features
  Q(candidate) = V(context) + (A(candidate) - mean_k A(k))

Eviction rule: evict argmin Q — the candidate with the least predicted
future value. ~12K params, trains on CPU in seconds.

Training here is fitted-Q on offline decision points: because an evicted
entry's episode terminates at the decision, the 1-step target reduces to the
(discounted) future regeneration cost saved by keeping the entry — computed
from an infinite-cache "demand" replay (see smartevict/model/train.py). This is a
deliberate POC simplification of full multi-step Q-learning; stated in the
README.
"""
from __future__ import annotations

import numpy as np


def _he(rng, n_in, n_out):
    return (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)


class DuelingEvictionNet:
    def __init__(self, n_features: int = 6, h1: int = 128, h2: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": _he(rng, n_features, h1), "b1": np.zeros(h1, np.float32),
            "W2": _he(rng, h1, h2),         "b2": np.zeros(h2, np.float32),
            "Wa": _he(rng, h2, 1),          "ba": np.zeros(1, np.float32),
            # value head over pooled (mean) candidate features
            "Wv1": _he(rng, n_features, 32), "bv1": np.zeros(32, np.float32),
            "Wv2": _he(rng, 32, 1),          "bv2": np.zeros(1, np.float32),
        }
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.p.items()}
        self._t = 0

    # ---- forward ---------------------------------------------------------
    def _trunk(self, X):
        z1 = X @ self.p["W1"] + self.p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ self.p["W2"] + self.p["b2"]; a2 = np.maximum(z2, 0)
        adv = (a2 @ self.p["Wa"] + self.p["ba"]).squeeze(-1)
        return adv, (X, z1, a1, z2, a2)

    def _value(self, ctx):
        zv1 = ctx @ self.p["Wv1"] + self.p["bv1"]; av1 = np.maximum(zv1, 0)
        v = (av1 @ self.p["Wv2"] + self.p["bv2"]).squeeze(-1)
        return v, (ctx, zv1, av1)

    def q_values(self, cand_feats: np.ndarray) -> np.ndarray:
        """cand_feats: (K, F) for one decision. Returns Q per candidate."""
        adv, _ = self._trunk(cand_feats)
        v, _ = self._value(cand_feats.mean(axis=0, keepdims=True))
        return v[0] + adv - adv.mean()

    # ---- training (MSE on per-candidate targets) -------------------------
    def train_batch(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    lr: float = 1e-3) -> float:
        """X: (N,F) candidate features, y: (N,) targets, groups: (N,) decision
        ids (candidates sharing a decision share a context/value)."""
        adv, tc = self._trunk(X)
        # pooled context per group
        ug, inv = np.unique(groups, return_inverse=True)
        ctx = np.zeros((len(ug), X.shape[1]), np.float32)
        cnt = np.zeros(len(ug), np.float32)
        np.add.at(ctx, inv, X); np.add.at(cnt, inv, 1.0)
        ctx /= cnt[:, None]
        v, vc = self._value(ctx)
        adv_mean = np.zeros(len(ug), np.float32)
        np.add.at(adv_mean, inv, adv); adv_mean /= cnt
        q = v[inv] + adv - adv_mean[inv]

        err = (q - y).astype(np.float32)
        loss = float(np.mean(err ** 2))
        g_q = (2.0 / len(y)) * err

        # grads through dueling combine
        g_adv = g_q.copy()
        g_admean = np.zeros(len(ug), np.float32); np.add.at(g_admean, inv, g_q)
        g_adv -= (g_admean / cnt)[inv]              # d(adv - mean)/d adv
        g_v = g_admean                              # dQ/dV summed per group

        grads = {k: np.zeros_like(vv) for k, vv in self.p.items()}
        # advantage trunk backward
        Xc, z1, a1, z2, a2 = tc
        gz = g_adv[:, None] * self.p["Wa"].T        # (N,h2) pre-relu? Wa applied on a2
        grads["Wa"] = a2.T @ g_adv[:, None]; grads["ba"] = np.array([g_adv.sum()], np.float32)
        gz2 = gz * (z2 > 0)
        grads["W2"] = a1.T @ gz2; grads["b2"] = gz2.sum(0)
        gz1 = (gz2 @ self.p["W2"].T) * (z1 > 0)
        grads["W1"] = Xc.T @ gz1; grads["b1"] = gz1.sum(0)
        # value head backward
        ctx_in, zv1, av1 = vc
        gv = g_v[:, None]
        grads["Wv2"] = av1.T @ gv; grads["bv2"] = gv.sum(0)
        gzv1 = (gv @ self.p["Wv2"].T) * (zv1 > 0)
        grads["Wv1"] = ctx_in.T @ gzv1; grads["bv1"] = gzv1.sum(0)

        # Adam
        self._t += 1
        b1m, b2m, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            m, s = self._adam[k]
            g = grads[k].astype(np.float32).reshape(self.p[k].shape)
            m[:] = b1m * m + (1 - b1m) * g
            s[:] = b2m * s + (1 - b2m) * g * g
            mh = m / (1 - b1m ** self._t); sh = s / (1 - b2m ** self._t)
            self.p[k] -= lr * mh / (np.sqrt(sh) + eps)
        return loss

    # ---- persistence -----------------------------------------------------
    def save(self, path: str):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path: str) -> "DuelingEvictionNet":
        net = cls()
        data = np.load(path)
        for k in net.p:
            net.p[k] = data[k].astype(np.float32)
        return net

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())
