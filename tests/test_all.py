"""Sanity tests. Run: python tests/test_all.py (requires `pip install -e .`)"""
import os

import numpy as np

from smartevict.features.embeddings import HashingEmbedder
from smartevict.data.generate_synthetic import generate_workload
from smartevict.simulator.cache_sim import run_simulation
from smartevict.policies.eviction import LRUPolicy, FIFOPolicy, LearnedPolicy
from smartevict.policies.wrapper import LearnedSemanticCache
from smartevict.model.dueling_net import DuelingEvictionNet
from smartevict.model.train import build_dataset, train_model


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


def test_gptcache_adapter():
    try:
        import gptcache  # noqa: F401
    except ImportError:
        print("skip gptcache adapter: `pip install gptcache` not installed")
        return

    import shutil
    import tempfile
    import time as time_mod

    from gptcache import Cache
    from gptcache.adapter.api import put, get
    from gptcache.manager import get_data_manager, CacheBase, VectorBase
    from gptcache.processor.pre import get_prompt
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

    from smartevict.policies.gptcache_adapter import LearnedEviction

    embedder = HashingEmbedder(dim=64)

    def embed_one(text, **_):
        return embedder.embed([text])[0]

    tmpdir = tempfile.mkdtemp(prefix="gptcache_adapter_test_")
    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        evicted_log = []
        # no model_path -> must behave exactly like the hard LRU fallback
        eviction = LearnedEviction(model_path=None, maxsize=5, k_tail=3,
                                   on_evict=lambda ids: evicted_log.append(ids))
        dm = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=64),
                              eviction_base=eviction, max_size=5)
        cache = Cache()
        cache.init(pre_embedding_func=get_prompt, embedding_func=embed_one,
                  data_manager=dm, similarity_evaluation=SearchDistanceEvaluation())

        for i in range(8):
            put(f"question number {i}", f"answer {i}", cache_obj=cache)
            time_mod.sleep(0.001)

        assert eviction.fallbacks == eviction.decisions > 0
        assert len(eviction._lru_order) == 5
        hit = get("question number 7", cache_obj=cache)
        assert hit == "answer 7", hit
        miss = get("completely unrelated topic zzz", cache_obj=cache)
        assert miss is None, miss
        print(f"ok gptcache adapter: {eviction.decisions} decisions, "
              f"{sum(len(e) for e in evicted_log)} evicted, size={len(eviction._lru_order)}")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_embedder_paraphrase_similarity()
    test_simulator_lru_vs_fifo()
    test_learned_fallback_on_broken_model()
    test_training_reduces_loss()
    test_wrapper_api()
    test_net_save_load_roundtrip()
    test_gptcache_adapter()
    print("\nall tests passed")
