#!/usr/bin/env python
"""Task accuracy under full attention, destructive eviction, and recall.

A stock 4D mask is shared across heads. This script patches Mistral's eager
attention so each query head can hide its own pages. Demand is the top-64
of a full-attention forward. The token is then taken from a second forward
that can see only the pages the policy left resident. Eviction never
re-admits a page. Recall fetches it back under the same LRU budget.

One NIAH prompt and one MATH-500 problem. The NIAH full-attention pass also
records page maxima and scores a causal prefetch against the top-64 pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wakekv.instrument import chunked_prefill, page_max_from_row
from wakekv.page_summary import prefetch_recall
from wakekv.reactive_controller import ReactiveController
from wakekv.tasks.cot import build_cot_prompt, load_problems
from wakekv.tasks.niah import build_niah_prompt


class _Mask:
    active = False
    by_layer: dict[int, torch.Tensor] = {}


def install_mistral_mask() -> None:
    import transformers.models.mistral.modeling_mistral as mm

    if getattr(mm, "_wakekv_mask_installed", False):
        return
    orig_names = True  # patch the module global the layer forward calls

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = mm.repeat_kv(key, module.num_key_value_groups)
        value_states = mm.repeat_kv(value, module.num_key_value_groups)
        attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, : key_states.shape[-2]]
        extra = _Mask.by_layer.get(module.layer_idx) if _Mask.active else None
        if extra is not None:
            attn = attn + extra[..., : attn.shape[-1]]
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout:
            attn = torch.nn.functional.dropout(attn, p=dropout, training=module.training)
        out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
        return out, attn

    mm.eager_attention_forward = wrapped
    mm._wakekv_mask_installed = orig_names


def clone_cache(past):
    from transformers.cache_utils import DynamicCache

    new = DynamicCache()
    if hasattr(past, "key_cache"):
        new.key_cache = [t.clone() for t in past.key_cache]
        new.value_cache = [t.clone() for t in past.value_cache]
        if hasattr(past, "_seen_tokens"):
            new._seen_tokens = past._seen_tokens
        return new
    raise TypeError(f"unsupported cache type {type(past)}")


def demand_from_attentions(attentions, page_size: int, topk: int) -> list[set[int]]:
    """One set of pages per (layer, query head), flattened layer-major."""
    wanted: list[set[int]] = []
    for layer in attentions:
        # layer: [batch, heads, q, k]
        row = layer[0, :, -1, :].float()
        k = min(topk, row.shape[-1])
        idx = torch.topk(row, k, dim=-1).indices
        pages = idx // page_size
        for head in range(pages.shape[0]):
            wanted.append(set(pages[head].tolist()))
    return wanted


def resident_mask(controller: ReactiveController, n_layers: int, n_heads: int,
                  ctx: int, page_size: int, device) -> dict[int, torch.Tensor]:
    masks = {}
    for layer in range(n_layers):
        allowed = torch.zeros(n_heads, ctx, device=device)
        for head in range(n_heads):
            unit = layer * n_heads + head
            for page in controller._resident[unit]:
                start = page * page_size
                end = min(ctx, start + page_size)
                if start < ctx:
                    allowed[head, start:end] = 1
        allowed[:, -1] = 1  # the query token stays visible
        additive = torch.zeros(1, n_heads, 1, ctx, device=device)
        additive = additive.masked_fill(allowed[None, :, None, :] == 0, float("-inf"))
        masks[layer] = additive
    return masks


@torch.no_grad()
def generate(model, input_ids, past, mode: str, budget: int, page_size: int,
             max_new_tokens: int, topk: int) -> tuple[list[int], dict]:
    """``past`` holds every prompt token except the last. ``input_ids[:, -1]``
    is that last prompt token and is not yet in the cache."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_units = n_layers * n_heads
    device = input_ids.device
    eos = cfg.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    demotion = "offload" if mode == "reactive" else "evict"
    ctrl = ReactiveController(n_units, max(budget, 1), rerank_interval=1, demotion=demotion)
    past_full = past
    past_body = clone_cache(past) if mode != "full" else None
    cur = input_ids[:, -1:]
    page_rows: list = []
    top_rows: list = []
    gen: list[int] = []

    def observe(out) -> None:
        wanted = demand_from_attentions(out.attentions, page_size, topk)
        ctrl.begin_step(ctrl.stats.steps)
        for unit, pages in enumerate(wanted):
            ctrl.on_unit_lru(unit, pages)
        ctrl.end_step()
        if mode == "full":
            row = torch.stack([a[0, :, -1, :].float() for a in out.attentions])
            page_rows.append(page_max_from_row(row, page_size).cpu().numpy())
            k = min(topk, row.shape[-1])
            top_rows.append(torch.topk(row, k, dim=-1).indices.cpu().numpy())

    def next_token(query, full_past, masked_past):
        _Mask.active = False
        oracle = model(
            query, past_key_values=full_past, use_cache=True, output_attentions=True
        )
        observe(oracle)
        if mode == "full":
            return int(oracle.logits[0, -1].argmax()), oracle.past_key_values, None
        ctx = masked_past.key_cache[0].shape[-2] + query.shape[-1]
        _Mask.by_layer = resident_mask(
            ctrl, n_layers, n_heads, ctx, page_size, device
        )
        _Mask.active = True
        masked = model(
            query, past_key_values=masked_past, use_cache=True, output_attentions=False
        )
        _Mask.active = False
        return int(masked.logits[0, -1].argmax()), oracle.past_key_values, masked.past_key_values

    nxt, past_full, past_masked = next_token(cur, past_full, past_body)
    for _ in range(max_new_tokens):
        gen.append(nxt)
        if nxt in eos_ids:
            break
        cur = torch.tensor([[nxt]], device=device)
        nxt, past_full, past_masked = next_token(cur, past_full, past_masked)
    extra: dict = {}
    if mode != "full":
        extra["miss_rate"] = ctrl.stats.as_dict()["miss_rate"]
    if page_rows:
        import numpy as np
        width = max(r.shape[-1] for r in page_rows)
        padded = [
            r if r.shape[-1] == width else np.pad(r, [(0, 0), (0, 0), (0, width - r.shape[-1])])
            for r in page_rows
        ]
        page_max = np.stack(padded).astype(np.float16)
        topk_idx = np.stack([
            np.pad(r, [(0, 0), (0, 0), (0, topk - r.shape[-1])], constant_values=-1)
            if r.shape[-1] < topk else r
            for r in top_rows
        ]).astype(np.int32)
        extra["page_summary"] = {
            str(b): prefetch_recall(page_max, topk_idx, page_size, b)
            for b in (8, 16, 32)
        }
    return gen, extra


def run_prompt(model, tokenizer, prompt: str, answer: str, kind: str, args) -> list[dict]:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    base = model.model
    print(f"[mask] prefill {kind} tokens={input_ids.shape[1]}", flush=True)
    past = chunked_prefill(base, input_ids, args.prefill_chunk)
    rows = []
    jobs = [("full", 0)] + [
        (mode, budget) for budget in args.budgets for mode in ("evict", "reactive")
    ]
    for mode, budget in jobs:
        print(f"[mask] {kind} mode={mode} budget={budget}", flush=True)
        gen, extra = generate(
            model, input_ids, clone_cache(past), mode, budget or 1,
            args.page_size, args.max_new_tokens, args.topk,
        )
        text = tokenizer.decode(gen)
        rows.append({
            "task": kind,
            "mode": mode,
            "budget": budget or None,
            "correct": answer in text,
            "generated": text,
            "answer": answer,
            **extra,
        })
        print(f"  correct={rows[-1]['correct']} miss_rate={extra['miss_rate']}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--context-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--math-tokens", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--out", default="runs/masked_accuracy.json")
    args = ap.parse_args()

    install_mistral_mask()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    model.eval()
    niah = build_niah_prompt(tokenizer, args.context_tokens, depth_fraction=0.5, seed=0)
    rows = run_prompt(model, tokenizer, niah["prompt"], niah["answer"], "niah", args)
    problems = load_problems("math500", 1)
    math = build_cot_prompt(tokenizer, problems[0]["problem"])
    # MATH-500 stores the answer separately; load it directly.
    from datasets import load_dataset
    answer = load_dataset("HuggingFaceH4/MATH-500", split="test")[0]["answer"]
    saved_tokens = args.max_new_tokens
    args.max_new_tokens = args.math_tokens
    rows.extend(run_prompt(model, tokenizer, math["prompt"], answer, "math500-0", args))
    args.max_new_tokens = saved_tokens
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "niah_context_tokens": args.context_tokens,
        "rows": rows,
    }, indent=2))
    print(f"[mask] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
