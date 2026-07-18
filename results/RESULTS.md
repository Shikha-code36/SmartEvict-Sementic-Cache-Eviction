# Benchmark results

## Real workload (LMSYS-Chat-1M)

Setup: 50,000 real conversations from `lmsys/lmsys-chat-1m`, cache size 400,
similarity threshold 0.8, K-tail = 8, time-ordered 60/40 train/test split
(30K train / 20K held-out test, no leakage). Reproduce with:
`smartevict-download-lmsys --n 50000 --out data/lmsys_trace.json`
then `smartevict-benchmark --trace data/lmsys_trace.json --seeds 0 1 2 3 4 [--embedder minilm]`.

`GDSF` and `Cost-weighted-recency` are non-learned, cost-aware baselines
(added to isolate "does considering cost at all help" from "does *learning*
help beyond a hand-written formula" — see "Reading these honestly" below).
Both are deterministic, like FIFO/LRU/Oracle.

| Embedder | FIFO | LRU | GDSF | Cost-weighted-recency | **Learned (5-seed mean ± std)** | Oracle (K-tail greedy) |
|---|---|---|---|---|---|---|
| HashingEmbedder (proxy) | −4.4% | baseline | **+27.7%** | +4.6% | +16.67% ± 1.09% | +22.4% |
| MiniLM (real semantic) | −18.2% | baseline | **+19.5%** | +2.7% | +16.96% ± 1.41% | +15.7% |

Hit ratios are much lower than on synthetic data (0.10–0.14 vs. 0.32–0.70) —
real LMSYS prompts have far less exact/near-duplicate structure than the
synthetic generator assumes, which is expected and realistic, not a bug.

### The headline finding: a simple heuristic beats the learned model here

**On both embedders, GDSF — a 1-line, non-learned formula — beats the
learned RL net**, and with HashingEmbedder it also beats the K-tail
"oracle." This was unexpected going in, and it changes the honest framing of
this project's real-data result: *learning does not clearly beat a
well-chosen cost-aware heuristic on this trace.*

**Why GDSF wins, mechanistically** (verified by instrumenting the K-tail
candidate pool across all 17,250 real eviction decisions):
- GDSF's aging term (`L`, added to every candidate at a given decision) is a
  constant across the K-tail comparison, so it cancels out of the argmin.
  The rule actually implemented reduces to **`score = hit_count × cost`,
  evict the minimum** — despite the name, the classic Greedy-Dual aging
  mechanism does no real work in this K-tail-simultaneous-comparison
  setting (it would matter in a sequential admission-vs-threshold cache,
  which this isn't).
- In **99.5%** of real decisions, at least one K-tail candidate has never
  been reused (`hit_count == 0`), which forces its score to the minimum
  possible. So in practice GDSF is close to a pure rule: *"if anything in
  the cold-8 has never been reused, evict that first, full stop"* — an
  aggressive LFU-style gate that's a strong prior specifically because
  LMSYS reuse is sparse (≈12% hit rate): protecting anything ever reused,
  regardless of its recency or cost, turns out to be almost always correct.

**Why the learned net doesn't discover this rule on its own:**
- The regression target (discounted future demand) is **90.2% exactly
  zero** across the 209,800 real-trace training samples — the net is fit
  with per-candidate MSE against a mostly-degenerate signal, which dilutes
  the pressure to sharply separate "definitely worthless" from "rare real
  future value."
- We tested whether this was fixable: switching the training objective from
  MSE regression to a **pairwise ranking loss** (directly optimizing
  "does the net rank the true best-to-keep candidate above the true
  worst-to-keep candidate," ignoring the exact zero-heavy target value) did
  **not** close the gap (3-seed mean ≈ +15.6% vs. GDSF's +27.7%, same
  ballpark as the original MSE net).
- We also tried a **hybrid policy**: gate on GDSF's exact `hit_count == 0`
  rule first, then let the learned net (trained either way) break ties
  *within* the gated pool instead of GDSF's naive oldest-first tie-break.
  This also underperformed plain GDSF (+11.2% to +20.0% across
  seeds/losses) — the net's cost-based tie-breaking, applied *within* the
  set of never-reused candidates, is actively worse than just picking the
  oldest one.
- That last result is the most informative one: it shows **cost is not a
  useful predictor of which never-reused entries will eventually be
  reused** on this trace — only how long an entry has sat unreused is.
  The net tends to protect expensive-but-old unreused entries, betting
  they'll pay off; on real LMSYS traffic that bet loses more often than a
  plain recency tie-break. Three independent fixes converging on the same
  root cause (not a bug in any one of them) is why we stopped chasing this
  gap rather than trying a fourth patch.

**Where cost-awareness *does* still show a (narrow) edge:** on the
synthetic high-duplicate regime below, the learned net edges out GDSF
(+6.10% vs. +5.40%) — the one regime dense enough in repeat reuse that cost
can meaningfully break ties among entries that have *already* proven
valuable, rather than needing to bet on unproven ones. That's a much
narrower claim than "learned beats heuristics," and it's the one the
evidence actually supports.

**On the oracle no longer being a hard ceiling:** GDSF exceeding it
(HashingEmbedder: +27.7% vs. +22.4%) isn't a contradiction. The "oracle" is
a *greedy, per-decision* policy — at each eviction it picks the least
valuable candidate *within the current K=8 tail only*, with no lookahead
and no consideration of the rest of the cache. GDSF isn't bound by that
same local greediness in the same way, since its scoring reflects the
*entire history* of each entry's hit count rather than a one-step future
estimate. This is a real limitation of calling it an "oracle" and is
flagged here rather than renamed away.

## Synthetic workload

Setup: 20,000 requests per regime, cache size 400 entries, similarity
threshold 0.8, K-tail = 8, time-based 60/40 train/test split (model trained
only on the first 60% of the trace, all numbers below are held-out).
Reproduce with: `smartevict-benchmark --n 20000 --cache-size 400`

Metric of record: **tokens saved** (regeneration-cost proxy), which is what
the reward optimizes — not raw hit ratio.

## Tokens saved vs. LRU (held-out trace)

| Regime (duplicate density) | FIFO | LRU | GDSF | Cost-weighted-recency | **Learned** | Oracle (ceiling) |
|---|---|---|---|---|---|---|
| High (tail=0.15) | −4.4% | baseline | +5.4% | +5.0% | **+6.1%** | +9.3% |
| Medium (tail=0.35) | −7.3% | baseline | **+8.0%** | +1.7% | +6.5% | +12.9% |
| Low (tail=0.60) | −10.7% | baseline | **+7.5%** | −0.9% | +3.6% | +15.5% |

Learned only wins outright in the high-duplicate regime; GDSF wins medium
and low. The pattern matches the real-trace finding above: as duplicate
density drops (less proven reuse to rank by cost, more decisions about
unproven entries), the simple heuristic's edge grows and the learned net's
shrinks.

## Hit ratio (held-out trace)

| Regime | FIFO | LRU | GDSF | Cost-weighted-recency | Learned | Oracle |
|---|---|---|---|---|---|---|
| High | 0.645 | 0.698 | 0.699 | 0.697 | 0.702 | 0.704 |
| Medium | 0.477 | 0.522 | 0.526 | 0.521 | 0.528 | 0.533 |
| Low | 0.317 | 0.355 | 0.358 | 0.353 | 0.362 | 0.365 |

## Reading these honestly

- **The central claim of this project — "learning beats non-learned
  eviction" — is not supported as a blanket statement.** A simple
  cost-aware heuristic (GDSF) matches or beats the learned net in most
  regimes tested, on both synthetic and real data. What *is* supported: a
  cost-aware policy (learned or heuristic) reliably beats recency-only
  LRU/FIFO, and learning shows a narrow edge specifically when reuse is
  dense enough that cost can break ties among already-proven-valuable
  entries (synthetic high-dup regime only).
- **The win is in tokens saved, not hit ratio**, for every cost-aware
  policy (GDSF, cost-weighted-recency, learned) — hit-ratio deltas are
  small (≤ 0.7 pt) while token-savings deltas are large. All three convert
  a similar number of hits into more regeneration cost avoided by
  specifically protecting expensive/proven entries, which is the intended
  mechanism and the honest headline regardless of which policy does it
  best.
- **The oracle gap shows headroom, but the "oracle" is a soft ceiling, not
  a hard one** — see the GDSF-exceeds-oracle discussion above. Treat the
  oracle numbers as "a strong, local reference," not a proven upper bound.
- **Gains shrink as duplicate density drops** for the learned policy
  (+6.5% → +3.6%) but *grow* for GDSF (+5.4% → +7.5%) — opposite trends,
  consistent with the mechanism explanation above.
- **Zero fallbacks fired** across all learned runs, and the fallback path
  is separately tested to reproduce LRU exactly when the model is absent
  or broken. This safety property holds regardless of the eviction-quality
  finding above.
- **Synthetic caveat.** This workload is generated (Zipf intents +
  paraphrase noise + unique tail); the generator was designed *before*
  results were collected and the density knob is swept to show sensitivity.
  Treat these tables as proof-of-mechanism, not proof-of-production-value —
  the LMSYS-Chat-1M numbers above are the ones that should headline any
  public claim.
- **Embedding caveat.** Hashing n-gram embeddings make paraphrase matching
  easier than real semantic matching would; the real-workload section
  above covers both HashingEmbedder and real MiniLM sentence embeddings.
  The synthetic tables in this section still use `HashingEmbedder` only.

See also [ABLATIONS.md](ABLATIONS.md) for feature/architecture/hyperparameter
ablations that dig further into what's driving these numbers.
