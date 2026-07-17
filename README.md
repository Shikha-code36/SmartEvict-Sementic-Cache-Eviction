# semantic-cache-eviction

A drop-in, **learned eviction policy for semantic LLM caches**, replacing
naive LRU/FIFO with a Cold-RL-style tiny model (arXiv:2508.12485 pattern:
K-tail candidate sampling + tiny dueling network + hard fallback to a
classical policy). Trained offline on replayed traces, safe by default,
benchmarked honestly against LRU/FIFO and a clairvoyant oracle.

Scope note: this project decides **which cached entries to keep warm**. It
deliberately does *not* touch semantic-match correctness (whether a cached
answer is actually valid for a new prompt) — that is a separate problem.

## Results at a glance

On a 20K-request synthetic conversational workload (held-out split, cache
size 400): the learned policy saves **+3.6% to +6.5% more regeneration
tokens than LRU** depending on duplicate density, capturing ~40–65% of the
clairvoyant-oracle headroom, with **zero fallbacks fired**. Full tables and
all caveats (synthetic data, hashing embeddings): [results/RESULTS.md](results/RESULTS.md).

## Quickstart — use the wrapper

```python
from features.embeddings import HashingEmbedder   # or your own embed fn
from policies.wrapper import LearnedSemanticCache

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
from features.embeddings import sentence_transformers_embedder
cache = LearnedSemanticCache(embedding_fn=sentence_transformers_embedder(), ...)
```

## Reproduce the benchmark

```bash
pip install -r requirements.txt
python tests/test_all.py                       # ~30s sanity suite
python benchmark/run_benchmark.py              # full 3-regime sweep, ~5 min CPU
python benchmark/run_benchmark.py --quick      # fast sanity version
```

Outputs `results/benchmark.json` + `results/learned_policy.npz`.

### Run on real data (LMSYS-Chat-1M)

Requires Hugging Face access (accept the dataset license on the hub first):

```bash
pip install datasets
python data/download_lmsys.py --n 50000 --out data/lmsys_trace.json
python benchmark/run_benchmark.py --trace data/lmsys_trace.json
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
data/        synthetic workload generator + LMSYS download script
simulator/   bounded semantic cache simulator, replay harness
features/    embeddings (hashing / sentence-transformers) + 6-feature extractor
model/       NumPy dueling net + offline training pipeline
policies/    LRU, FIFO, Learned (w/ fallback), Oracle + LearnedSemanticCache wrapper
benchmark/   reproducible comparison script
tests/       sanity suite (simulator, training, fallback, wrapper, FAISS)
results/     benchmark output + honest write-up (RESULTS.md)
```

## Known limitations

- Benchmarked on synthetic data with hashing embeddings so far (build
  environment had no Hugging Face access). Rerun on LMSYS + MiniLM before
  quoting numbers anywhere public.
- The model generalizes across duplicate-density regimes here, but was not
  tested across *time-scale* shifts (e.g., traces with very different
  arrival rates); features use log-scaled absolute times, so a per-deployment
  fine-tune (seconds of CPU) is recommended.
- GPTCache `eviction=` adapter is a stretch goal (Plan §9), not yet built.
- Single-threaded wrapper; no persistence of cache contents across restarts.
