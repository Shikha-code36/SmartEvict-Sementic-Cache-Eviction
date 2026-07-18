"""Per-entry features for eviction decisions (Section 4 of the plan).

6 features per cached entry, mirroring Cold-RL's feature set translated to
the semantic-cache domain:

  0. age                  — now - insert_t                  (log1p)
  1. token cost           — response_tokens (regen proxy)   (log1p)
  2. hit count            — hits since insertion            (log1p)
  3. time since last use  — now - last_access_t             (log1p)
  4. mean inter-arrival   — avg gap between accesses;
                            age if never re-accessed        (log1p)
  5. staleness ratio      — (now - last_access) / age  in [0, 1]
                            (proxy for drift: fraction of lifetime idle)

All raw values are on trace-time / token scales; log1p keeps them in a range
a tiny MLP is happy with. No dataset-dependent normalization constants are
baked in.
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 6


def entry_features(e, now: float) -> np.ndarray:
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


def candidates_features(cache, cand_ids: list[int], now: float) -> np.ndarray:
    return np.stack([entry_features(cache.entries[c], now) for c in cand_ids])
