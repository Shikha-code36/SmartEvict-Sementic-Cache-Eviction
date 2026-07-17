"""Download + prep a real workload (run locally; needs Hugging Face access).

Produces the same record schema the simulator consumes:
    [{"t": float, "text": str, "response_tokens": int}, ...]

Usage:
    pip install datasets python-dotenv
    # put HF_TOKEN=hf_xxx in a .env file in the repo root (gitignored)
    python data/download_lmsys.py --n 50000 --out data/lmsys_trace.json

Note: LMSYS-Chat-1M requires accepting its license on the HF hub first.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    ap.add_argument("--out", default="data/lmsys_trace.json")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # datasets' streaming reader prefetches the next parquet shard ahead of
    # where we actually stop reading; breaking out of the loop below then
    # abandons that in-flight request, and huggingface_hub logs a harmless
    # retry warning for it. Quiet that logger since it's not a real failure.
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    ds = ds.take(args.n * 2)  # generous cap: some rows get skipped below

    records, t = [], 0.0
    for row in ds:
        conv = row.get("conversation") or []
        user = next((m["content"] for m in conv if m["role"] == "user"), None)
        asst = next((m["content"] for m in conv if m["role"] == "assistant"), None)
        if not user or not asst:
            continue
        t += 1.0
        records.append({
            "t": t,
            "text": user[:2000],
            # ~4 chars/token heuristic; swap in a real tokenizer if you want
            "response_tokens": max(1, len(asst) // 4),
        })
        if len(records) >= args.n:
            break

    with open(args.out, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
