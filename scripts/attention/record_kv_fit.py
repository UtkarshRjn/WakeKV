#!/usr/bin/env python
"""One long needle. Records whether the full KV fits, and a short log if it does.

Eager attention on the whole prompt allocates a heads-by-T-by-T score
matrix. Prefill is chunked so an out-of-memory error means the KV cache
itself did not fit, which is the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wakekv.instrument import run_instrumented_generation
from wakekv.page_summary import prefetch_recall
from wakekv.tasks.niah import build_niah_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--context-tokens", type=int, default=65536)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--out", default="runs/longctx")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[kv-fit] dtype={dtype} context={args.context_tokens}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    model.eval()
    spec = build_niah_prompt(
        tokenizer, args.context_tokens, depth_fraction=0.5, seed=0
    )
    torch.cuda.reset_peak_memory_stats()
    record = {
        "model": args.model,
        "context_tokens": args.context_tokens,
        "prefill_chunk": args.prefill_chunk,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    try:
        log = run_instrumented_generation(
            model,
            tokenizer,
            spec["prompt"],
            needle_substring=spec["needle"],
            max_new_tokens=args.max_new_tokens,
            prefill_chunk=args.prefill_chunk,
            page_size=args.page_size,
            meta={"model": args.model, "task": "niah", "job": "longctx"},
        )
        saved = log.save(out / "log")
        record["fit"] = True
        record["prompt_len"] = log.meta["prompt_len"]
        record["generated"] = log.meta["generated_text"]
        record["answer"] = spec["answer"]
        record["answer_present"] = spec["answer"] in log.meta["generated_text"]
        record["log"] = str(saved)
        if log.page_max is not None:
            record["page_summary"] = prefetch_recall(
                log.page_max, log.topk_idx, args.page_size, args.budget
            )
    except torch.OutOfMemoryError as exc:
        record["fit"] = False
        record["error"] = str(exc).split("\n", 1)[0]
        print(f"[kv-fit] KV did not fit: {record['error']}", flush=True)
    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out / "fit.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("fit", "peak_allocated_bytes")}), flush=True)


if __name__ == "__main__":
    main()
