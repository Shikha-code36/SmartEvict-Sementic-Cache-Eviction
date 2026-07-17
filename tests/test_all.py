"""Sanity tests. Run: python tests/test_all.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.embeddings import HashingEmbedder
from data.generate_synthetic import generate_workload
from simulator.cache_sim import run_simulation
from policies.eviction import LRUPolicy, FIFOPolicy, LearnedPolicy
from policies.wrapper import LearnedSemanticCache
from model.dueling_net import DuelingEvictionNet
from model.train import build_dataset, train_model


def test_embedder_paraphrase_similarity():
    e = HashingEmbedder(dim=64)
    v = e.embed(["how do i reset my password on github",
                 "hey how do i reset my password on github thanks",
                 "recipe for chocolate cake"])
    para = float(v[0] @ v[1]); diff = float(v[0] @ v[2])
    assert para > 0.85, para
    assert diff < 0.5, diff
    print(f"ok embedder: paraphrase sim={para:.3f}, unrelated sim={diff:.3f}")


def test_simulator_lru_vs_fifo():
    recs = generate_workload(n_requests=2000, seed=1)
    emb = HashingEmbedder().embed([r["text"] for r in recs])
    r_lru = run_simulation(recs, emb, 100, 0.8, LRUPolicy())
    r_fifo = run_simulation(recs, emb, 100, 0.8, FIFOPolicy())
    assert r_lru.hits > 0 and r_fifo.hits > 0
    assert r_lru.requests == 2000
    print(f"ok simulator: LRU hit={r_lru.hit_ratio:.3f} FIFO hit={r_fifo.hit_ratio:.3f}")


def test_learned_fallback_on_broken_model():
    recs = generate_workload(n_requests=1000, seed=2)
    emb = HashingEmbedder().embed([r["text"] for r in recs])
    pol = LearnedPolicy(net=None, k_tail=8)  # no model -> must fall back
    res = run_simulation(recs, emb, 80, 0.8, pol)
    assert pol.fallbacks == pol.decisions > 0
    lru = run_simulation(recs, emb, 80, 0.8, LRUPolicy())
    assert res.hits == lru.hits  # fallback == LRU exactly
    print(f"ok fallback: {pol.fallbacks}/{pol.decisions} decisions fell back to LRU, matches LRU")


def test_training_reduces_loss():
    recs = generate_workload(n_requests=3000, seed=3)
    emb = HashingEmbedder().embed([r["text"] for r in recs])
    X, y, g, _ = build_dataset(recs, emb, 100, 0.8, 8)
    net, hist = train_model(X, y, g, epochs=3, verbose=False)
    assert hist[-1][0] < hist[0][0], hist
    assert net.n_params < 50000
    print(f"ok training: mse {hist[0][0]:.3f} -> {hist[-1][0]:.3f}, {net.n_params} params")


def test_wrapper_api():
    emb = HashingEmbedder(dim=64)
    c = LearnedSemanticCache(embedding_fn=emb.embed, max_size=3,
                             eviction_policy="learned", model_path="does_not_exist.npz",
                             sim_threshold=0.8)
    assert c.get("anything") is None
    c.set("how tall is the eiffel tower", "330 meters")
    c.set("capital of france", "paris")
    hit = c.get("hey how tall is the eiffel tower please")
    assert hit == "330 meters", hit
    assert c.get("best sushi in osaka") is None
    c.set("a", "1"); c.set("b", "2")  # forces eviction with broken model path
    s = c.stats()
    assert s["evictions"] >= 1 and s["fallbacks"] >= 1 and s["size"] == 3
    print(f"ok wrapper: backend={s['backend']}, evictions={s['evictions']}, "
          f"fallbacks={s['fallbacks']}, hit_ratio={s['hit_ratio']:.2f}")


def test_net_save_load_roundtrip():
    net = DuelingEvictionNet(seed=5)
    F = np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32)
    q1 = net.q_values(F)
    net.save("/tmp/net_test.npz")
    q2 = DuelingEvictionNet.load("/tmp/net_test.npz").q_values(F)
    assert np.allclose(q1, q2)
    print("ok save/load roundtrip")


if __name__ == "__main__":
    test_embedder_paraphrase_similarity()
    test_simulator_lru_vs_fifo()
    test_learned_fallback_on_broken_model()
    test_training_reduces_loss()
    test_wrapper_api()
    test_net_save_load_roundtrip()
    print("\nall tests passed")
