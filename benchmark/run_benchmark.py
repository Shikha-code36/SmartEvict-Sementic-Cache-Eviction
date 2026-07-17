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
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_synthetic import generate_workload                    # noqa: E402
from features.embeddings import HashingEmbedder                          # noqa: E402
from simulator.cache_sim import run_simulation                           # noqa: E402
from policies.eviction import LRUPolicy, FIFOPolicy, LearnedPolicy, OraclePolicy  # noqa: E402
from model.train import build_dataset, train_model, compute_future_matches  # noqa: E402


def evaluate(records, emb, max_size, thr, k_tail, net, future_matches):
    req_times = np.array([r["t"] for r in records])
    req_tokens = np.array([r["response_tokens"] for r in records])
    rows = {}
    for name, factory in {
        "fifo": lambda: (FIFOPolicy(), None),
        "lru": lambda: (LRUPolicy(), None),
        "learned": lambda: (LearnedPolicy(net, k_tail=k_tail), None),
        "oracle": lambda: (
            (p := OraclePolicy(future_matches, req_times, req_tokens, k_tail=k_tail)), p),
    }.items():
        pol, hooks = factory()
        t0 = time.time()
        res = run_simulation(records, emb, max_size, thr, pol, hooks=hooks)
        rows[name] = {
            "hit_ratio": res.hit_ratio,
            "tokens_saved": res.tokens_saved,
            "tokens_saved_frac": res.tokens_saved_frac,
            "evictions": res.evictions,
            "wall_s": round(time.time() - t0, 2),
        }
        if name == "learned":
            rows[name]["fallbacks"] = pol.fallbacks
            rows[name]["decisions"] = pol.decisions
    return rows


def run_regime(tag, records, args, out):
    embedder = HashingEmbedder(dim=args.dim)
    texts = [r["text"] for r in records]
    emb = embedder.embed(texts)

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
    net, hist = train_model(X, y, groups, epochs=args.epochs, verbose=True)

    fm_test = compute_future_matches(te_emb, args.threshold)
    print(f"[{tag}] evaluating policies on held-out trace")
    rows = evaluate(te_rec, te_emb, args.cache_size, args.threshold,
                    args.k_tail, net, fm_test)
    out[tag] = {"config": {"n_requests": len(records), "cache_size": args.cache_size,
                           "threshold": args.threshold, "k_tail": args.k_tail},
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
    ap.add_argument("--out", type=str, default="results/benchmark.json")
    args = ap.parse_args()
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
        net.save("results/learned_policy.npz")
    print(f"\nwrote {args.out} and results/learned_policy.npz")


if __name__ == "__main__":
    main()
