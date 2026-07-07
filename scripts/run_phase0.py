#!/usr/bin/env python
"""Phase 0 runner: instrumented generation + per-step head logging.

Examples:
  python scripts/run_phase0.py --model Qwen/Qwen2.5-7B-Instruct \
      --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2

  python scripts/run_phase0.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --task cot --max-new-tokens 2048 --limit 5

  python scripts/run_phase0.py --model Qwen/Qwen2.5-7B-Instruct --task multiturn

Requires a CUDA GPU. Models are loaded with attn_implementation="eager"
(required to materialize attention weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wakekv.instrument import run_instrumented_generation
from wakekv.tasks.cot import build_cot_prompt, load_problems
from wakekv.tasks.multiturn import build_multiturn_prompt
from wakekv.tasks.niah import build_niah_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["niah", "cot", "multiturn"], required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--context-tokens", type=int, default=5000, help="niah only")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5], help="niah only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="niah only")
    ap.add_argument("--limit", type=int, default=5, help="cot: number of problems")
    ap.add_argument("--cot-source", default="auto", help="auto|math500|aime2024|bundled")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="auto = bf16 where supported (Ampere+), else fp16 (e.g. 2080Ti)")
    args = ap.parse_args()

    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = getattr(torch, args.dtype)
    print(f"[phase0] using dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    model_slug = args.model.replace("/", "__")
    out_root = Path(args.out) / model_slug / args.task

    jobs: list[tuple[str, dict]] = []
    if args.task == "niah":
        for depth in args.depths:
            for seed in args.seeds:
                spec = build_niah_prompt(
                    tokenizer, args.context_tokens, depth_fraction=depth, seed=seed
                )
                jobs.append((f"d{depth:.2f}_s{seed}", spec))
    elif args.task == "cot":
        for prob in load_problems(args.cot_source, args.limit):
            spec = build_cot_prompt(tokenizer, prob["problem"])
            jobs.append((prob["id"], spec))
    else:
        jobs.append(("recall", build_multiturn_prompt(tokenizer)))

    for name, spec in jobs:
        print(f"[phase0] {args.task}/{name} ...", flush=True)
        log = run_instrumented_generation(
            model,
            tokenizer,
            spec["prompt"],
            needle_substring=spec.get("needle"),
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
            meta={
                "model": args.model,
                "task": args.task,
                "job": name,
                "expected_answer": spec.get("answer"),
            },
        )
        path = log.save(out_root / name)
        answer = spec.get("answer")
        got = log.meta["generated_text"]
        verdict = ""
        if answer:
            verdict = " [ANSWER OK]" if answer in got else " [ANSWER MISSING]"
        print(f"  -> {path}  gen={len(log.gen_token_ids)} tokens{verdict}", flush=True)


if __name__ == "__main__":
    main()
