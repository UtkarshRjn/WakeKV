#!/usr/bin/env python
"""Task accuracy under full / frozen / destructive / reversible residency.

Oracle full-attention demand drives the residency controller; the next
token is taken from a second forward that can see only resident pages.
Both forwards share the same generated-token prefix within a cell.
Across cells, each (mode, budget) generates its own history.

Writes ``--out`` after every completed cell and skips finished keys on
resume. Still an oracle-assisted accuracy experiment, not a deployable
offloading benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wakekv.grading import grade
from wakekv.instrument import chunked_prefill, page_max_from_row
from wakekv.masked_eval import atomic_write, cell_key, job_grid, load_checkpoint
from wakekv.page_summary import prefetch_recall
from wakekv.reactive_controller import ReactiveController
from wakekv.residency import classify_unstable
from wakekv.tasks.cot import build_cot_prompt, load_problems
from wakekv.tasks.multiturn import build_multiturn_prompt
from wakekv.tasks.niah import build_niah_prompt


class _Mask:
    active = False
    by_layer: dict[int, torch.Tensor] = {}


def install_mistral_mask() -> None:
    import transformers.models.mistral.modeling_mistral as mm

    if getattr(mm, "_wakekv_mask_installed", False):
        return

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
    mm._wakekv_mask_installed = True


def clone_cache(past):
    from transformers.cache_utils import DynamicCache

    if past is None:
        return None
    # Legacy tuple[(k,v), ...] from older HF returns.
    if isinstance(past, tuple):
        if hasattr(DynamicCache, "from_legacy_cache"):
            past = DynamicCache.from_legacy_cache(past)
        else:
            new = DynamicCache()
            new.key_cache = [k.clone() for k, _ in past]
            new.value_cache = [v.clone() for _, v in past]
            return new
    if hasattr(past, "key_cache"):
        new = DynamicCache()
        new.key_cache = [t.clone() for t in past.key_cache]
        new.value_cache = [t.clone() for t in past.value_cache]
        if hasattr(past, "_seen_tokens"):
            new._seen_tokens = past._seen_tokens
        return new
    if hasattr(past, "to_legacy_cache") and hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past.to_legacy_cache())
    raise TypeError(f"unsupported cache type {type(past)}")


def demand_from_attentions(attentions, page_size: int, topk: int) -> list[set[int]]:
    wanted: list[set[int]] = []
    for layer in attentions:
        row = layer[0, :, -1, :].float()
        k = min(topk, row.shape[-1])
        idx = torch.topk(row, k, dim=-1).indices
        pages = idx // page_size
        for head in range(pages.shape[0]):
            wanted.append(set(int(p) for p in pages[head].tolist()))
    return wanted


def attention_mass_retained(
    attentions, controller: ReactiveController, n_layers: int, n_heads: int, page_size: int
) -> float:
    """Fraction of oracle attention mass that lands on currently resident pages."""
    kept = total = 0.0
    for layer_i, layer in enumerate(attentions):
        row = layer[0, :, -1, :].float()  # [H, ctx]
        ctx = row.shape[-1]
        for head in range(n_heads):
            unit = layer_i * n_heads + head
            mass = row[head]
            total += float(mass.sum())
            for page in controller.resident_pages(unit):
                start = page * page_size
                end = min(ctx, start + page_size)
                if start < ctx:
                    kept += float(mass[start:end].sum())
            # query token always visible
            kept += float(mass[-1])
    return kept / total if total > 0 else 1.0


def resident_mask(
    controller: ReactiveController,
    n_layers: int,
    n_heads: int,
    ctx: int,
    page_size: int,
    device,
    unstable: set[int] | None = None,
    budget: int = 0,
) -> dict[int, torch.Tensor]:
    masks = {}
    unstable = unstable or set()
    for layer in range(n_layers):
        allowed = torch.zeros(n_heads, ctx, device=device)
        for head in range(n_heads):
            unit = layer * n_heads + head
            if unit in unstable:
                allowed[head, :] = 1
            else:
                pages = controller.resident_pages(unit)
                if budget > 0 and len(pages) > budget:
                    raise AssertionError(
                        f"unit {unit} has {len(pages)} resident pages > budget {budget}"
                    )
                for page in pages:
                    start = page * page_size
                    end = min(ctx, start + page_size)
                    if start < ctx:
                        allowed[head, start:end] = 1
            allowed[head, -1] = 1
        additive = torch.zeros(1, n_heads, 1, ctx, device=device)
        additive = additive.masked_fill(allowed[None, :, None, :] == 0, float("-inf"))
        masks[layer] = additive
    return masks


def _fit_unstable_from_demands(
    demand_history: list[list[set[int]]], n_units: int, unstable_frac: float, prefix: int
) -> set[int]:
    prefix_stream = demand_history[:prefix] if demand_history else []
    if len(prefix_stream) < 2:
        return set()
    return classify_unstable(prefix_stream, n_units, unstable_frac)


@torch.no_grad()
def generate(
    model,
    input_ids,
    past,
    mode: str,
    budget: int,
    page_size: int,
    max_new_tokens: int,
    topk: int,
    unstable_frac: float = 0.25,
    frozen_prefix: int = 32,
    frozen_refresh: int = 16,
) -> tuple[list[int], dict]:
    """``past`` holds every prompt token except the last.

    Oracle and masked forwards share the same query token each step.
    Cross-policy histories diverge because each mode emits its own tokens.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_units = n_layers * n_heads
    device = input_ids.device
    eos = cfg.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    demotion = "offload" if mode in ("reactive", "frozen", "full") else "evict"
    # Frozen uses a large LRU for unstable heads via mask bypass; stable use budget.
    ctrl_budget = max(budget, 1) if mode != "full" else max(n_units * 64, 1)
    ctrl = ReactiveController(n_units, ctrl_budget, rerank_interval=1, demotion=demotion)
    past_full = past
    past_body = clone_cache(past) if mode != "full" else None
    cur = input_ids[:, -1:]
    page_rows: list = []
    top_rows: list = []
    gen: list[int] = []
    demand_history: list[list[set[int]]] = []
    unstable: set[int] = set()
    mass_sum = 0.0
    mass_steps = 0
    window_demand: list[dict[int, int]] = [dict() for _ in range(n_units)]

    def observe(out) -> list[set[int]]:
        nonlocal unstable
        wanted = demand_from_attentions(out.attentions, page_size, topk)
        demand_history.append(wanted)
        if mode == "frozen" and len(demand_history) == frozen_prefix:
            unstable = _fit_unstable_from_demands(
                demand_history, n_units, unstable_frac, frozen_prefix
            )
        ctrl.begin_step(ctrl.stats.steps)
        if mode == "frozen":
            step = ctrl.stats.steps
            for unit, pages in enumerate(wanted):
                if unit in unstable:
                    # Keep everything ever demanded for unstable heads.
                    for p in pages:
                        ctrl._resident[unit][p] = step
                    ctrl.stats.total_wanted += len(pages)
                else:
                    ctrl.on_unit_lru(unit, pages)
                    for p in pages:
                        window_demand[unit][p] = window_demand[unit].get(p, 0) + 1
            if (step + 1) % frozen_refresh == 0:
                for unit in range(n_units):
                    if unit in unstable or not window_demand[unit]:
                        continue
                    top = sorted(
                        window_demand[unit], key=window_demand[unit].get, reverse=True
                    )[:budget]
                    ctrl._resident[unit] = {p: step for p in top}
                    window_demand[unit] = {}
        elif mode == "full":
            for unit, pages in enumerate(wanted):
                for p in pages:
                    ctrl._resident[unit][p] = ctrl.stats.steps
                ctrl.stats.total_wanted += len(pages)
        else:
            for unit, pages in enumerate(wanted):
                ctrl.on_unit_lru(unit, pages)
                if budget > 0:
                    assert len(ctrl.resident_pages(unit)) <= budget
        ctrl.end_step()
        mass = attention_mass_retained(out.attentions, ctrl, n_layers, n_heads, page_size)
        nonlocal mass_sum, mass_steps
        mass_sum += mass
        mass_steps += 1
        if mode == "full":
            row = torch.stack([a[0, :, -1, :].float() for a in out.attentions])
            page_rows.append(page_max_from_row(row, page_size).cpu().numpy())
            k = min(topk, row.shape[-1])
            top_rows.append(torch.topk(row, k, dim=-1).indices.cpu().numpy())
        return wanted

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
            ctrl, n_layers, n_heads, ctx, page_size, device,
            unstable=unstable if mode == "frozen" else None,
            budget=budget if mode in ("evict", "reactive") else 0,
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

    stats = ctrl.stats.as_dict()
    extra: dict = {
        "miss_rate": 0.0 if mode == "full" else float(stats["miss_rate"]),
        "attention_mass_retained": mass_sum / mass_steps if mass_steps else 1.0,
        "stats": stats,
        "n_unstable": len(unstable),
    }
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


def build_task_specs(tokenizer, args) -> list[dict]:
    specs = []
    for depth in args.niah_depths:
        for seed in args.niah_seeds:
            niah = build_niah_prompt(
                tokenizer, args.context_tokens, depth_fraction=depth, seed=seed
            )
            specs.append({
                "task_id": f"niah-d{depth}-s{seed}",
                "kind": "niah",
                "prompt": niah["prompt"],
                "answer": niah["answer"],
                "max_new_tokens": args.max_new_tokens,
            })
    problems = load_problems("math500", args.n_math)
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, problem in enumerate(problems):
        math = build_cot_prompt(tokenizer, problem["problem"])
        specs.append({
            "task_id": f"math500-{i}",
            "kind": "math",
            "prompt": math["prompt"],
            "answer": ds[i]["answer"],
            "max_new_tokens": args.math_tokens,
        })
    mt = build_multiturn_prompt(tokenizer)
    specs.append({
        "task_id": "multiturn-0",
        "kind": "multiturn",
        "prompt": mt["prompt"],
        "answer": mt["answer"],
        "max_new_tokens": args.multiturn_tokens,
    })
    return specs


def run_cell(model, tokenizer, past, input_ids, spec: dict, mode: str, budget: int | None, args) -> dict:
    gen, extra = generate(
        model,
        input_ids,
        clone_cache(past),
        mode,
        budget or 1,
        args.page_size,
        spec["max_new_tokens"],
        args.topk,
        unstable_frac=args.unstable_frac,
        frozen_prefix=args.frozen_prefix,
    )
    text = tokenizer.decode(gen)
    return {
        "task": spec["task_id"],
        "kind": spec["kind"],
        "mode": mode,
        "budget": budget,
        "correct": grade(spec["kind"], text, spec["answer"]),
        "generated": text,
        "answer": spec["answer"],
        **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--context-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--math-tokens", type=int, default=256)
    ap.add_argument("--multiturn-tokens", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--niah-depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--niah-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-math", type=int, default=5)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--unstable-frac", type=float, default=0.25)
    ap.add_argument("--frozen-prefix", type=int, default=32)
    ap.add_argument("--out", default="runs/masked_accuracy.json")
    args = ap.parse_args()

    install_mistral_mask()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    model.eval()

    out = Path(args.out)
    state = load_checkpoint(out)
    rows: list[dict] = list(state["rows"])
    done: dict = dict(state["done"])
    specs = build_task_specs(tokenizer, args)
    jobs = job_grid(args.budgets)
    device = next(model.parameters()).device

    for spec in specs:
        input_ids = tokenizer(spec["prompt"], return_tensors="pt").input_ids.to(device)
        print(f"[mask] prefill {spec['task_id']} tokens={input_ids.shape[1]}", flush=True)
        past = chunked_prefill(model.model, input_ids, args.prefill_chunk)
        for mode, budget in jobs:
            key = cell_key(spec["task_id"], mode, budget)
            if key in done and "error" not in done[key]:
                print(f"[mask] skip {key}", flush=True)
                continue
            print(f"[mask] {key}", flush=True)
            try:
                row = run_cell(model, tokenizer, past, input_ids, spec, mode, budget, args)
            except Exception as exc:  # noqa: BLE001 — per-cell isolation
                row = {
                    "task": spec["task_id"],
                    "kind": spec["kind"],
                    "mode": mode,
                    "budget": budget,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "miss_rate": None,
                    "attention_mass_retained": None,
                }
                print(f"  ERROR {row['error']}", flush=True)
            else:
                print(
                    f"  correct={row['correct']} miss_rate={row['miss_rate']} "
                    f"mass={row['attention_mass_retained']:.3f}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # replace prior failed attempt if any
            rows = [r for r in rows if cell_key(r["task"], r["mode"], r.get("budget")) != key]
            rows.append(row)
            done[key] = row
            atomic_write(out, {
                "model": args.model,
                "niah_context_tokens": args.context_tokens,
                "budgets": args.budgets,
                "rows": rows,
            })
    print(f"[mask] wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
