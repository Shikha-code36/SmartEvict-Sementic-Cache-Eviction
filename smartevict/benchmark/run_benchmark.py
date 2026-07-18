"""Reproducible benchmark: learned eviction vs LRU/FIFO (+ clairvoyant oracle).

Protocol (train/test split by TIME, no leakage):
  - generate (or load) a request trace
  - embed all prompts once
  - train the dueling net ONLY on the first `train_frac` of the trace
  - evaluate all policies on the held-out remainder
  - repeat across duplicate-density regimes (tail_frac knob)

Usage:
  python benchmark/run_benchmark.py                 # full run, writes results/
  python benchmark/run_benchmark.py --quick         # smaller trace, sanity run
  python benchmark/run_benchmark.py --trace path.json   # real trace (e.g. LMSYS)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from smartevict.data.generate_synthetic import generate_workload
from smartevict.features.embeddings import HashingEmbedder, sentence_transformers_embedder
from smartevict.simulator.cache_sim import run_simulation
from smartevict.policies.eviction import (LRUPolicy, FIFOPolicy, LearnedPolicy, OraclePolicy,
                                          GDSFPolicy, CostWeightedRecencyPolicy)
from smartevict.model.train import build_dataset, train_model, compute_future_matches


def evaluate_one(records, emb, max_size, thr, pol, hooks=None):
    t0 = time.time()
    res = run_simulation(records, emb, max_size, thr, pol, hooks=hooks)
    row = {
        "hit_ratio": res.hit_ratio,
        "tokens_saved": res.tokens_saved,
        "tokens_saved_frac": res.tokens_saved_frac,
        "evictions": res.evictions,
        "wall_s": round(time.time() - t0, 2),
    }
    if isinstance(pol, LearnedPolicy):
        row["fallbacks"] = pol.fallbacks
        row["decisions"] = pol.decisions
    return row


def evaluate(records, emb, max_size, thr, k_tail, net, future_matches):
    req_times = np.array([r["t"] for r in records])
    req_tokens = np.array([r["response_tokens"] for r in records])
    rows = {}
    for name, factory in {
        "fifo": lambda: (FIFOPolicy(), None),
        "lru": lambda: (LRUPolicy(), None),
        "gdsf": lambda: (GDSFPolicy(k_tail=k_tail), None),
        "cost_weighted_recency": lambda: (CostWeightedRecencyPolicy(k_tail=k_tail), None),
        "learned": lambda: (LearnedPolicy(net, k_tail=k_tail), None),
        "oracle": lambda: (
            (p := OraclePolicy(future_matches, req_times, req_tokens, k_tail=k_tail)), p),
    }.items():
        pol, hooks = factory()
        rows[name] = evaluate_one(records, emb, max_size, thr, pol, hooks=hooks)
    return rows


def make_embedder(args):
    if args.embedder == "hashing":
        return HashingEmbedder(dim=args.dim).embed
    return sentence_transformers_embedder()


def run_regime(tag, records, args, out):
    embed = make_embedder(args)
    texts = [r["text"] for r in records]
    emb = embed(texts)

    split = int(len(records) * args.train_frac)
    tr_rec, te_rec = records[:split], records[split:]
    tr_emb, te_emb = emb[:split], emb[split:]
    # re-zero test timestamps so 'age' scales match training
    t0 = te_rec[0]["t"]
    te_rec = [{**r, "t": r["t"] - t0} for r in te_rec]

    print(f"[{tag}] building training set ({split} train / {len(te_rec)} test requests)")
    X, y, groups, _ = build_dataset(tr_rec, tr_emb, args.cache_size, args.threshold,
                                    args.k_tail, gamma=args.gamma, horizon=args.horizon)
    print(f"[{tag}] {len(np.unique(groups))} eviction decisions, {len(X)} candidate samples")
    fm_test = compute_future_matches(te_emb, args.threshold)

    if args.seeds:
        # fifo/lru/oracle are deterministic (no learned component), so compute
        # them once; only the learned policy varies with the training seed.
        base_rows = {}
        for name, factory in {
            "fifo": lambda: (FIFOPolicy(), None),
            "lru": lambda: (LRUPolicy(), None),
            "gdsf": lambda: (GDSFPolicy(k_tail=args.k_tail), None),
            "cost_weighted_recency": lambda: (CostWeightedRecencyPolicy(k_tail=args.k_tail), None),
            "oracle": lambda: ((p := OraclePolicy(
                fm_test, np.array([r["t"] for r in te_rec]),
                np.array([r["response_tokens"] for r in te_rec]), k_tail=args.k_tail)), p),
        }.items():
            pol, hooks = factory()
            base_rows[name] = evaluate_one(te_rec, te_emb, args.cache_size, args.threshold,
                                           pol, hooks=hooks)

        seed_runs = []
        net = None
        for seed in args.seeds:
            net, _ = train_model(X, y, groups, epochs=args.epochs, seed=seed, verbose=False)
            row = evaluate_one(te_rec, te_emb, args.cache_size, args.threshold,
                               LearnedPolicy(net, k_tail=args.k_tail))
            seed_runs.append({"seed": seed, **row})
            print(f"  seed {seed}: learned hit={row['hit_ratio']:.4f} "
                  f"tokens_saved={row['tokens_saved']:>10,}")

        tokens = np.array([r["tokens_saved"] for r in seed_runs], dtype=np.float64)
        hits = np.array([r["hit_ratio"] for r in seed_runs], dtype=np.float64)
        base = base_rows["lru"]["tokens_saved"]
        deltas = (tokens / base - 1) * 100 if base else np.full_like(tokens, float("nan"))
        learned_agg = {
            "tokens_saved_mean": float(tokens.mean()), "tokens_saved_std": float(tokens.std()),
            "hit_ratio_mean": float(hits.mean()), "hit_ratio_std": float(hits.std()),
            "vs_lru_pct_mean": float(deltas.mean()), "vs_lru_pct_std": float(deltas.std()),
        }
        rows = {**base_rows, "learned_per_seed": seed_runs, "learned_agg": learned_agg}
        out[tag] = {"config": {"n_requests": len(records), "cache_size": args.cache_size,
                               "threshold": args.threshold, "k_tail": args.k_tail,
                               "embedder": args.embedder, "seeds": args.seeds},
                    "results": rows}
        print(f"[{tag}] learned vs LRU across {len(args.seeds)} seeds: "
              f"{learned_agg['vs_lru_pct_mean']:+.2f}% +/- {learned_agg['vs_lru_pct_std']:.2f}%")
        for name in ("fifo", "lru", "gdsf", "cost_weighted_recency", "oracle"):
            r = base_rows[name]
            delta = (r["tokens_saved"] / base - 1) * 100 if base else float("nan")
            print(f"  {name:8s} hit={r['hit_ratio']:.4f}  tokens_saved={r['tokens_saved']:>10,}"
                  f"  vs LRU: {delta:+.2f}%")
        return net

    net, hist = train_model(X, y, groups, epochs=args.epochs, verbose=True)
    print(f"[{tag}] evaluating policies on held-out trace")
    rows = evaluate(te_rec, te_emb, args.cache_size, args.threshold,
                    args.k_tail, net, fm_test)
    out[tag] = {"config": {"n_requests": len(records), "cache_size": args.cache_size,
                           "threshold": args.threshold, "k_tail": args.k_tail,
                           "embedder": args.embedder},
                "train_hist": hist, "results": rows}
    base = rows["lru"]["tokens_saved"]
    for name, r in rows.items():
        delta = (r["tokens_saved"] / base - 1) * 100 if base else float("nan")
        print(f"  {name:8s} hit={r['hit_ratio']:.4f}  tokens_saved={r['tokens_saved']:>10,}"
              f"  vs LRU: {delta:+.2f}%")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--cache-size", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--k-tail", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--horizon", type=float, default=5000.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trace", type=str, default=None,
                    help="JSON trace file (e.g. from data/download_lmsys.py)")
    ap.add_argument("--embedder", choices=["hashing", "minilm"], default="hashing",
                    help="hashing: local, deterministic, no download. "
                         "minilm: sentence-transformers/all-MiniLM-L6-v2, real semantic "
                         "embeddings, requires `pip install sentence-transformers` + "
                         "one-time model download.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="if given, retrain the learned net once per seed (fifo/lru/oracle "
                         "are deterministic so computed once) and report mean +/- std vs LRU "
                         "instead of a single run, e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--out", type=str, default="results/benchmark.json")
    ap.add_argument("--model-out", type=str, default=None,
                    help="where to save the trained net (default: derived from --out, "
                         "e.g. results/benchmark_foo.json -> results/foo_learned_policy.npz)")
    args = ap.parse_args()
    if args.model_out is None:
        base = os.path.splitext(os.path.basename(args.out))[0]
        suffix = base.replace("benchmark", "").strip("_")
        fname = f"{suffix}_learned_policy.npz" if suffix else "learned_policy.npz"
        args.model_out = os.path.join(os.path.dirname(args.out) or ".", fname)
    if args.quick:
        args.n, args.epochs = 6000, 4

    out = {}
    if args.trace:
        with open(args.trace) as f:
            records = json.load(f)
        net = run_regime("real_trace", records, args, out)
    else:
        # duplicate-density sweep: high / medium / low
        regimes = {"high_dup (tail=0.15)": 0.15,
                   "med_dup  (tail=0.35)": 0.35,
                   "low_dup  (tail=0.60)": 0.60}
        net = None
        for tag, tail in regimes.items():
            records = generate_workload(n_requests=args.n, tail_frac=tail, seed=7)
            net = run_regime(tag, records, args, out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if net is not None:
        net.save(args.model_out)
    print(f"\nwrote {args.out} and {args.model_out}")


if __name__ == "__main__":
    main()
