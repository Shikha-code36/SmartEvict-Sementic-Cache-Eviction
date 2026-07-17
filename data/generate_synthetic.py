"""Synthetic conversational workload with controllable duplicate density.

Why synthetic: this environment cannot reach Hugging Face, so LMSYS-Chat-1M /
ShareGPT can't be pulled here. The generator mimics the property that matters
for eviction research: a Zipf-skewed pool of "intents" (popular questions get
asked repeatedly, phrased slightly differently) mixed with a long tail of
one-off prompts. Duplicate density is an explicit knob so you can report how
gains scale with it (mirrors Cold-RL's pressure regimes).

Swap in real data with data/download_lmsys.py on a machine with HF access —
the simulator only needs a list of (text, response_tokens) records.
"""
from __future__ import annotations

import numpy as np

TOPICS = [
    "reset my password on {svc}", "cancel my {svc} subscription",
    "difference between {a} and {b} in python", "write a {adj} poem about {thing}",
    "explain {concept} like i am five", "fix {err} error in {lang}",
    "best way to learn {lang} in 2026", "summarize the plot of {thing}",
    "convert {n} {unit} to {unit2}", "how do i center a div in css",
    "what is the capital of {place}", "write sql to find duplicate rows in {tbl}",
    "debug why my {lang} loop is infinite", "email to my boss asking for {thing}",
    "meal plan for a {adj} week", "regex to match {thing}",
    "explain {concept} vs {concept2}", "translate hello world to {lang}",
    "why is my docker container {err}", "itinerary for 3 days in {place}",
]
FILL = {
    "svc": ["netflix", "spotify", "github", "aws", "notion"],
    "a": ["list", "tuple", "set", "dict", "array"],
    "b": ["tuple", "generator", "dict", "dataframe", "deque"],
    "adj": ["short", "funny", "healthy", "cheap", "quick"],
    "thing": ["autumn", "coffee", "a raise", "dune", "an email address", "my cat"],
    "concept": ["entropy", "recursion", "inflation", "attention", "gradient descent"],
    "concept2": ["enthalpy", "iteration", "deflation", "convolution", "newton's method"],
    "err": ["segfault", "keyerror", "oomkilled", "timeout", "403"],
    "lang": ["python", "rust", "go", "sql", "javascript"],
    "n": ["10", "42", "100", "3.5", "250"],
    "unit": ["miles", "kg", "usd", "celsius", "gb"],
    "unit2": ["km", "lbs", "eur", "fahrenheit", "mb"],
    "place": ["kyoto", "lisbon", "delhi", "peru", "iceland"],
    "tbl": ["orders", "users", "events", "logs", "payments"],
}
PARA_PREFIX = ["", "", "hey ", "please ", "can you ", "quick question - ", "urgent: "]
PARA_SUFFIX = ["", "", " thanks", " asap", " for me", " today", "?"]


def _fill(template: str, rng: np.random.Generator) -> str:
    out = template
    for key, opts in FILL.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", str(rng.choice(opts)))
    return out


def generate_workload(n_requests: int = 20000, n_intents: int = 1500,
                      zipf_a: float = 1.15, tail_frac: float = 0.35,
                      seed: int = 7) -> list[dict]:
    """Returns list of {"t": float, "text": str, "response_tokens": int, "intent": int}.

    tail_frac: fraction of requests that are unique one-offs (never repeat).
    Lower tail_frac / lower zipf_a => higher duplicate density.
    """
    rng = np.random.default_rng(seed)
    # Materialize intent pool: canonical text + fixed response cost
    intents = []
    for i in range(n_intents):
        tmpl = TOPICS[i % len(TOPICS)]
        intents.append({
            "canon": _fill(tmpl, rng) + f" v{i}",
            "tokens": int(np.clip(rng.lognormal(5.3, 0.9), 20, 4000)),  # ~200 median
        })

    ranks = rng.zipf(zipf_a, size=n_requests * 2)
    ranks = ranks[ranks <= n_intents][:n_requests] - 1

    records, t = [], 0.0
    for j in range(n_requests):
        t += rng.exponential(1.0)  # Poisson arrivals, mean 1 time-unit apart
        if rng.random() < tail_frac:
            text = _fill(str(rng.choice(TOPICS)), rng) + f" uniq{j}x{rng.integers(1e9)}"
            tok = int(np.clip(rng.lognormal(5.3, 0.9), 20, 4000))
            intent = -1
        else:
            it = intents[int(ranks[j])]
            # Paraphrase: cosmetic edits around the canonical form
            text = str(rng.choice(PARA_PREFIX)) + it["canon"] + str(rng.choice(PARA_SUFFIX))
            tok, intent = it["tokens"], int(ranks[j])
        records.append({"t": t, "text": text, "response_tokens": tok, "intent": intent})
    return records
