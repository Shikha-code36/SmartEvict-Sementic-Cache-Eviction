# SmartEvict

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![GPTCache](https://img.shields.io/badge/GPTCache-Compatible-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A learned, cost-aware eviction policy for semantic LLM caches.

------------------------------------------------------------------------

## Why SmartEvict?

Traditional semantic caches use heuristic eviction policies such as LRU
and FIFO. These policies only consider *when* an entry was last
accessed, not *how expensive* it would be to regenerate.

SmartEvict predicts which cache entries are worth keeping by
considering:

-   Regeneration cost
-   Access history
-   Semantic reuse patterns
-   Response size

### Benefits

-   Higher regeneration-token savings
-   Lower latency
-   Reduced LLM cost
-   Safe fallback to classical LRU

------------------------------------------------------------------------

## Architecture

``` text
                  User Query
                      │
                      ▼
               Embedding Model
                      │
                      ▼
             Semantic Similarity Search
                      │
             ┌────────┴────────┐
             │                 │
          Cache Hit        Cache Miss
             │                 │
             ▼                 ▼
      Return Response      Call LLM
                                │
                                ▼
                     Store Prompt–Response Pair
                                │
                                ▼
                     SmartEvict Policy Engine
                                │
                                ▼
                   Select Entry to Evict
```

------------------------------------------------------------------------

## Key Features

-   Lightweight learned eviction model (\~9.5K parameters)
-   Cost-aware eviction decisions
-   Offline training using replayed request traces
-   Exact LRU fallback for production safety
-   GPTCache integration
-   Backend-agnostic wrapper
-   FAISS or NumPy vector search backend
-   Fully reproducible benchmarking pipeline

------------------------------------------------------------------------

## Why Not LRU?

Traditional cache eviction policies assume that recently used entries
are the most valuable to keep. While this works well for conventional
caches, semantic LLM caches have an additional consideration:
**regeneration cost**.

  Cached Entry            Regeneration Cost LRU Decision
  --------------------- ------------------- ----------------
  FAQ answer                            Low Keep if recent
  Multi-page analysis                  High Evict if old

Although both entries may be equally old, regenerating the multi-page
analysis is significantly more expensive. SmartEvict learns to
prioritize cache entries based on their expected future value rather
than recency alone.

------------------------------------------------------------------------

## Project Goals

-   Improve regeneration-token savings over classical eviction policies
-   Maintain production safety through deterministic LRU fallback
-   Keep inference lightweight enough for CPU-only deployment
-   Provide a reproducible benchmark for learned semantic cache eviction

------------------------------------------------------------------------

## Workflow

``` text
Incoming Prompt
      │
      ▼
Generate Embedding
      │
      ▼
Semantic Cache Lookup
      │
 ┌────┴─────┐
 │          │
Hit        Miss
 │          │
 ▼          ▼
Return     Call LLM
              │
              ▼
      Store Response
              │
              ▼
    SmartEvict decides
    whether another cache
    entry should be evicted
```

------------------------------------------------------------------------


Semantic LLM caches (GPTCache, LangChain's cache, Redis semantic caching)
evict by recency alone: LRU treats a cheap cached FAQ answer and an
expensive multi-page cached analysis as equally disposable the moment
neither has been touched in a while. But they aren't equally disposable —
one costs orders of magnitude more to regenerate than the other. This
project replaces recency-only eviction with a **learned, cost-aware
policy**: a ~9.5K-param model that predicts which entries are worth
keeping warm based on the regeneration cost they'd save if reused, not
just how recently they were touched.

It follows a Cold-RL-style pattern (arXiv:2508.12485: K-tail candidate
sampling + tiny dueling network + hard fallback to a classical policy),
trains offline on replayed traces, is safe by default (falls back to LRU
exactly if the model is absent or errors), and is benchmarked honestly
against LRU/FIFO and a clairvoyant oracle rather than against itself.

Scope note: this project decides **which cached entries to keep warm**. It
deliberately does *not* touch semantic-match correctness (whether a cached
answer is actually valid for a new prompt) — that is a separate problem.

## Results at a glance

On a 20K-request synthetic conversational workload (held-out split, cache
size 400): the learned policy saves **+3.6% to +6.5% more regeneration
tokens than LRU** depending on duplicate density, capturing ~40–65% of the
clairvoyant-oracle headroom, with **zero fallbacks fired**.

On a real 50K-request trace from LMSYS-Chat-1M (held-out 20K-request tail),
averaged across 5 training seeds (fifo/lru/oracle are deterministic, so
only the learned net varies): **+16.7% ± 1.1%** more regeneration tokens
than LRU with the local `HashingEmbedder`, and **+17.0% ± 1.4%** with real
**MiniLM sentence embeddings** (`--embedder minilm`) — where the gap to the
clairvoyant oracle ceiling essentially closes. Full tables and all caveats:
[results/RESULTS.md](results/RESULTS.md).

## Install

Not published on PyPI (still under active benchmarking/validation) — install
straight from GitHub or a local clone:

```bash
# directly from GitHub, no clone needed
pip install "git+https://github.com/Shikha-code36/SmartEvict-Sementic-Cache-Eviction.git"
pip install "smartevict[all] @ git+https://github.com/Shikha-code36/SmartEvict-Sementic-Cache-Eviction.git"

# or, if you already have a local clone
git clone https://github.com/Shikha-code36/SmartEvict-Sementic-Cache-Eviction.git
cd SmartEvict-Sementic-Cache-Eviction
pip install -e .              # core package (numpy only)
pip install -e ".[all]"       # + faiss, LMSYS download, MiniLM, GPTCache adapter
```

Extras are also installable individually: `.[faiss]`, `.[lmsys]`, `.[minilm]`,
`.[gptcache]`.

## Quickstart — use the wrapper

```python
from smartevict.features.embeddings import HashingEmbedder   # or your own embed fn
from smartevict.policies.wrapper import LearnedSemanticCache

cache = LearnedSemanticCache(
    embedding_fn=HashingEmbedder(dim=64).embed,   # list[str] -> normalized np.ndarray
    max_size=1000,
    eviction_policy="learned",                     # or "lru" / "fifo"
    model_path="results/learned_policy.npz",
    sim_threshold=0.8,
)

resp = cache.get(prompt)          # None on miss
if resp is None:
    resp = call_llm(prompt)       # your model call
    cache.set(prompt, resp)

print(cache.stats())              # hits, evictions, fallbacks, tokens_saved, backend
```

Backend: FAISS (`IndexFlatIP` over normalized vectors) if installed,
transparent brute-force NumPy store otherwise. The wrapper is
backend-agnostic by design (Plan §9): it only owns the eviction decision.

For real semantic quality, swap the embedder:

```python
from smartevict.features.embeddings import sentence_transformers_embedder
cache = LearnedSemanticCache(embedding_fn=sentence_transformers_embedder(), ...)
```

## Quickstart — use it inside GPTCache

Already using [GPTCache](https://github.com/zilliztech/GPTCache)? You don't
need to switch caching libraries to try this — GPTCache's own eviction
backend (`gptcache.manager.eviction.memory_cache.MemoryCacheEviction`) is
hardcoded to a fixed set of `cachetools` policies (LRU/LFU/FIFO/RR), but
the interface underneath it is a plain 3-method ABC, and
`get_data_manager(..., eviction_base=...)` accepts an already-built
instance of it. `smartevict/policies/gptcache_adapter.py` implements that
interface using the same net + hard-fallback logic benchmarked in this repo:

```python
pip install -e ".[gptcache]"

from gptcache import Cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from smartevict.policies.gptcache_adapter import LearnedEviction

eviction = LearnedEviction(model_path="results/learned_policy.npz", maxsize=1000)
data_manager = get_data_manager(CacheBase("sqlite"),
                                VectorBase("faiss", dimension=384),
                                eviction_base=eviction, max_size=1000)

cache = Cache()
cache.init(data_manager=data_manager, ...)  # your usual embedding_func / similarity_evaluation
```

Caveat: GPTCache's `put`/`get` eviction hooks only pass opaque row ids, not
the prompt/response text or token counts — this adapter tracks its own
per-id age/hit-count/idle-time bookkeeping, but has no visibility into
regeneration cost unless you tell it. Call
`eviction.note_cost(id, response_tokens)` right after each insert to get
the full cost-aware behavior benchmarked in [results/RESULTS.md](results/RESULTS.md);
without it, cost defaults to a flat value and the policy degrades to a
recency/frequency-only signal (still safe — falls back to plain LRU
exactly if the model is missing or errors, same as the standalone wrapper).

## Reproduce the benchmark

```bash
pip install -e ".[all]"
python tests/test_all.py                                # ~30s sanity suite
smartevict-benchmark                                     # full 3-regime sweep, ~5 min CPU
smartevict-benchmark --quick                             # fast sanity version
```

(`smartevict-benchmark` is a console script installed by `pip install -e .`;
equivalently `python -m smartevict.benchmark.run_benchmark`.)

Outputs `results/benchmark.json` + `results/learned_policy.npz`.

### Run on real data (LMSYS-Chat-1M)

`lmsys/lmsys-chat-1m` is a **gated dataset** — installing the `lmsys` extra
is not enough on its own, you need an approved Hugging Face access request
and a token:

1. Create a free account at https://huggingface.co if you don't have one.
2. Visit https://huggingface.co/datasets/lmsys/lmsys-chat-1m while logged
   in and submit the access request (fills in affiliation/use-case). This
   is reviewed manually by the dataset owner and is **not instant** — it
   can take anywhere from minutes to a day or more to be approved. Recheck
   the page until it shows you have access.
3. Generate a read-scoped token at https://huggingface.co/settings/tokens.
4. Put the token in a `.env` file in the repo root (gitignored, never commit
   it):
   ```
   HF_TOKEN=hf_your_token_here
   ```
5. Install deps and download:
   ```bash
   pip install -e ".[lmsys]"
   smartevict-download-lmsys --n 50000 --out data/lmsys_trace.json
   smartevict-benchmark --trace data/lmsys_trace.json
   ```

If step 5 fails with `DatasetNotFoundError: ... is a gated dataset`, the
token is either missing/invalid or your access request from step 2 hasn't
been approved yet — it is not a code/setup bug.

To benchmark with real sentence embeddings instead of the local hashing
proxy, add `pip install -e ".[minilm]"` (first run downloads
`all-MiniLM-L6-v2`, ~90MB) and pass `--embedder minilm`:

```bash
pip install -e ".[minilm]"
smartevict-benchmark --trace data/lmsys_trace.json --embedder minilm --out results/benchmark_minilm.json
```

This writes to `results/benchmark_minilm.json` and
`results/minilm_learned_policy.npz` rather than the default filenames, so it
won't overwrite the hashing-embedder results/model.

To check the result isn't a lucky single seed, add `--seeds`: fifo/lru/oracle
are deterministic so they're computed once, and only the learned net is
retrained per seed, reporting mean ± std vs LRU:

```bash
smartevict-benchmark --trace data/lmsys_trace.json --seeds 0 1 2 3 4 --out results/benchmark_multiseed_hashing.json
smartevict-benchmark --trace data/lmsys_trace.json --embedder minilm --seeds 0 1 2 3 4 --out results/benchmark_multiseed_minilm.json
```

## How it works (Cold-RL → semantic cache mapping)

| Cold-RL (NGINX) | Here |
|---|---|
| HTTP object | Cached prompt–response pair |
| age, size, hits, inter-arrival, TTL, RTT | age, response tokens (cost proxy), hits, idle time, mean inter-access gap, staleness ratio |
| K-tail from LRU list | K=8 coldest entries |
| Dueling DQN (~10K params) | Dueling net, **9,474 params**, pure NumPy, CPU-trains in seconds |
| ONNX sidecar, 500µs SLO | In-process inference (~0.2ms/decision; semantic-cache evictions are rare) |
| Hard timeout → LRU fallback | try/except → LRU fallback; fallback path unit-tested to match LRU exactly |
| Reward: +1 if reused before TTL | Reward: discounted future **regeneration tokens saved** |

Training is offline fitted-Q on decision points collected from an LRU replay,
with targets from an infinite-cache "demand" pre-pass (every request's future
matches, computed policy-free) — this sidesteps the off-policy counterfactual
problem of never observing reuse of entries the behavior policy evicted.
Because eviction terminates an entry's episode, the 1-step target reduces to
discounted future demand; that simplification (vs. full multi-step
Q-learning) is deliberate for this POC and stated here so nobody mistakes it
for the full algorithm.

## Repo layout

```
pyproject.toml       package metadata; `pip install -e .` / `.[all]`
smartevict/
  data/         synthetic workload generator + LMSYS download script
  simulator/    bounded semantic cache simulator, replay harness
  features/     embeddings (hashing / sentence-transformers) + 6-feature extractor
  model/        NumPy dueling net + offline training pipeline
  policies/     LRU, FIFO, Learned (w/ fallback), Oracle, LearnedSemanticCache
                wrapper, and the GPTCache EvictionBase adapter
  benchmark/    reproducible comparison script
tests/          sanity suite (simulator, training, fallback, wrapper, FAISS, GPTCache)
results/        benchmark output + honest write-up (RESULTS.md)
```

## Known limitations

- The synthetic-data tables still only use `HashingEmbedder`. The LMSYS
  real-trace benchmark has now been run with both `HashingEmbedder` and
  real MiniLM sentence embeddings (`--embedder minilm`), each averaged
  across 5 training seeds — see [results/RESULTS.md](results/RESULTS.md)
  for the full comparison.
- LMSYS results are still from a single trace/split (only the training
  seed is varied, not the data); a different 50K-request sample or a
  different train/test split isn't yet covered.
- The model generalizes across duplicate-density regimes here, but was not
  tested across *time-scale* shifts (e.g., traces with very different
  arrival rates); features use log-scaled absolute times, so a per-deployment
  fine-tune (seconds of CPU) is recommended.
- The GPTCache adapter (`smartevict/policies/gptcache_adapter.py`) is tested against a
  real in-process GPTCache `Cache` (sqlite + FAISS), but not against every
  backend combination GPTCache supports (Redis, Milvus, etc.), and cost
  weighting there requires the caller to call `note_cost()` explicitly —
  GPTCache's eviction hooks don't expose response length on their own.
- Single-threaded wrapper; no persistence of cache contents across restarts.
