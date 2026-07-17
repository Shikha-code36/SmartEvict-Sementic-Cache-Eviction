"""Embedding functions for the semantic cache.

The cache/simulator only requires `embed(texts: list[str]) -> np.ndarray`
(L2-normalized rows). Two implementations:

- HashingEmbedder: fully local, deterministic, zero downloads. Character
  n-gram hashing + signed random projection. Paraphrases share most n-grams,
  so they land close in cosine space. Good enough to *simulate* semantic
  matching; NOT a claim of real semantic quality.

- sentence_transformers_embedder(): thin wrapper around
  sentence-transformers/all-MiniLM-L6-v2 for machines with Hugging Face
  access. Use this for real benchmarks.
"""
from __future__ import annotations

import hashlib
import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 64, ngram: tuple[int, int] = (3, 5),
                 n_buckets: int = 2 ** 16, seed: int = 0):
        self.dim = dim
        self.ngram = ngram
        self.n_buckets = n_buckets
        rng = np.random.default_rng(seed)
        # Fixed random projection: buckets -> dense dim
        self.proj = rng.standard_normal((n_buckets, dim)).astype(np.float32)
        self.proj /= np.sqrt(dim)

    def _bucketize(self, text: str) -> dict[int, float]:
        t = " " + " ".join(text.lower().split()) + " "
        counts: dict[int, float] = {}
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little")
                b = h % self.n_buckets
                counts[b] = counts.get(b, 0.0) + 1.0
        return counts

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = self._bucketize(text)
            if not counts:
                continue
            idx = np.fromiter(counts.keys(), dtype=np.int64)
            w = np.fromiter(counts.values(), dtype=np.float32)
            w = np.log1p(w)  # sublinear tf
            out[i] = w @ self.proj[idx]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def sentence_transformers_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Returns an embed(texts)->ndarray callable. Requires network/HF access."""
    from sentence_transformers import SentenceTransformer  # lazy import
    m = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        v = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    return embed
