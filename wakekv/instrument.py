"""Instrumented greedy decoding with per-step, per-head attention logging.

Logs, for every decode step and every (layer, query-head):
- top-k attended positions and their attention weights
- continuous needle score (2602.11162 Eq. 2: needle mass / non-sink,
  non-local mass) when a needle span is given
- binary copy-paste score (2602.11162 Eq. 1: argmax position inside the
  needle AND the attended token equals the generated token)

Memory design: the model must run with ``attn_implementation="eager"`` so
attention weights are materialized, but we never accumulate full rows —
each step's [L, H, ctx] tensor is reduced to top-k + scalars immediately.
Prefill runs WITHOUT attention output (the full [T, T] map would not fit);
the last prompt token's forward doubles as the step-0 / prefill-baseline
attention row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class RunLog:
    """Arrays are [steps, layers, heads, ...]; step 0 = last prompt token."""

    topk_idx: np.ndarray  # int32 [S, L, H, K], padded with -1
    topk_val: np.ndarray  # float16 [S, L, H, K]
    ctx_len: np.ndarray  # int32 [S] — attention-row length at each step
    needle_score: np.ndarray | None  # float32 [S, L, H]
    copy_paste: np.ndarray | None  # bool [S, L, H]
    page_max: np.ndarray | None = None  # float16 [S, L, H, P], 0 past ctx
    # Optional Phase-3 detector fields (large; written beside log.npz).
    query: np.ndarray | None = None  # float16 [S, L, H, D]
    gen_token_ids: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # page -> (kmin, kmax) per (layer, head); saved as kmin.npz / kmax.npz
    key_extrema: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {
            "topk_idx": self.topk_idx,
            "topk_val": self.topk_val,
            "ctx_len": self.ctx_len,
            "gen_token_ids": np.array(self.gen_token_ids, dtype=np.int64),
        }
        if self.needle_score is not None:
            arrays["needle_score"] = self.needle_score
            arrays["copy_paste"] = self.copy_paste
        if self.page_max is not None:
            arrays["page_max"] = self.page_max
        np.savez_compressed(out / "log.npz", **arrays)
        if self.query is not None:
            np.save(out / "query.npy", self.query)
            self.meta["has_query"] = True
        if self.key_extrema:
            kmin = {k: v[0] for k, v in self.key_extrema.items()}
            kmax = {k: v[1] for k, v in self.key_extrema.items()}
            np.savez_compressed(out / "kmin.npz", **kmin)
            np.savez_compressed(out / "kmax.npz", **kmax)
            self.meta["has_key_extrema"] = True
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return out


def page_key_extrema(keys: np.ndarray, page_size: int) -> tuple[np.ndarray, np.ndarray]:
    """``keys`` [T, D] → dense ``kmin``/``kmax`` [n_pages, D]."""
    t, d = keys.shape
    n_pages = (t + page_size - 1) // page_size
    kmin = np.full((n_pages, d), np.inf, dtype=np.float32)
    kmax = np.full((n_pages, d), -np.inf, dtype=np.float32)
    for p in range(n_pages):
        chunk = keys[p * page_size : min(t, (p + 1) * page_size)]
        kmin[p] = chunk.min(axis=0)
        kmax[p] = chunk.max(axis=0)
    return kmin, kmax


def find_token_span(tokenizer, full_text: str, substring: str) -> tuple[int, int]:
    """Locate ``substring`` in ``full_text`` as a [start, end) token span."""
    char_start = full_text.index(substring)
    char_end = char_start + len(substring)
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=True)
    tok_start = tok_end = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s == e:  # special token
            continue
        if tok_start is None and e > char_start:
            tok_start = i
        if s < char_end:
            tok_end = i + 1
    if tok_start is None or tok_end is None:
        raise ValueError("substring not found at token level")
    return tok_start, tok_end


def chunked_prefill(base, input_ids: torch.Tensor, chunk_size: int = 512):
    """Prefill every token except the last, one chunk at a time.

    Eager attention materializes the score matrix. A single forward of a
    65,536-token prompt asks for hundreds of gigabytes. Chunks keep each
    score matrix at ``heads × chunk × cache`` instead of ``heads × T × T``.
    Returns the KV cache. The KV itself may still not fit.
    """
    from transformers.cache_utils import DynamicCache

    past = None
    body = input_ids[:, :-1]
    pos = 0
    n = body.shape[1]
    while pos < n:
        end = min(n, pos + chunk_size)
        out = base(body[:, pos:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
        # Normalize legacy tuple caches so callers can clone consistently.
        if isinstance(past, tuple) and hasattr(DynamicCache, "from_legacy_cache"):
            past = DynamicCache.from_legacy_cache(past)
        pos = end
    return past


def page_max_from_row(att: torch.Tensor, page_size: int) -> torch.Tensor:
    """Max attention in each page. ``att`` is [L, H, ctx]."""
    ctx = att.shape[-1]
    n_pages = (ctx + page_size - 1) // page_size
    pad = n_pages * page_size - ctx
    if pad:
        att = torch.nn.functional.pad(att, (0, pad))
    return att.view(*att.shape[:-1], n_pages, page_size).amax(dim=-1)


@torch.no_grad()
def run_instrumented_generation(
    model,
    tokenizer,
    prompt_text: str,
    needle_substring: str | None = None,
    max_new_tokens: int = 256,
    topk: int = 64,
    sink_tokens: int = 4,
    local_window: int = 32,
    meta: dict | None = None,
    prefill_chunk: int = 512,
    page_size: int = 0,
) -> RunLog:
    device = next(model.parameters()).device
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    needle_span = None
    if needle_substring is not None:
        needle_span = find_token_span(tokenizer, prompt_text, needle_substring)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads

    # Prefill everything except the last prompt token, no attention output.
    # Call the base model (no LM head) so we don't materialize full-sequence
    # logits — logits.float() over [seq, vocab] is a multi-GB peak on long
    # contexts and the prefill logits are unused (only the KV cache is kept).
    base = getattr(model, "model", model)
    past = chunked_prefill(base, input_ids, prefill_chunk)
    cur = input_ids[:, -1:]

    full_tokens: list[int] = input_ids[0].tolist()
    steps_idx, steps_val, steps_ctx = [], [], []
    steps_needle, steps_copy = [], []
    steps_page: list[np.ndarray] = []
    gen_ids: list[int] = []
    eos_ids = set(
        cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    )

    for _ in range(max_new_tokens):
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
        past = out.past_key_values
        # att: [L, H, ctx] for the single current query position
        att = torch.stack([a[0, :, 0, :] for a in out.attentions]).float()
        ctx = att.shape[-1]
        k = min(topk, ctx)
        val, idx = torch.topk(att, k, dim=-1)
        if k < topk:
            pad = topk - k
            idx = torch.nn.functional.pad(idx, (0, pad), value=-1)
            val = torch.nn.functional.pad(val, (0, pad), value=0.0)
        next_tok = int(out.logits[0, -1].argmax())

        steps_idx.append(idx.cpu().numpy().astype(np.int32))
        steps_val.append(val.cpu().numpy().astype(np.float16))
        steps_ctx.append(ctx)
        if page_size:
            steps_page.append(
                page_max_from_row(att, page_size).cpu().numpy().astype(np.float16)
            )

        if needle_span is not None:
            s, e = needle_span
            # Needle score = needle mass / non-sink, non-local mass
            # (2602.11162 Eq. 2). The sink/local exclusions must never remove
            # needle mass itself, or denom collapses when the needle overlaps
            # the local window — which happens for CoT (needle = the question
            # at the start) while the context is still short, blowing the ratio
            # past 1 (to ~1e5). Excluding the needle span from sink/local keeps
            # the ratio in [0, 1] and is a no-op for NIAH, where the needle is
            # disjoint from both regions.
            pos = torch.arange(ctx, device=att.device)
            needle_m = (pos >= s) & (pos < e)
            sink_m = (pos < min(sink_tokens, ctx)) & ~needle_m
            local_m = (pos >= max(0, ctx - local_window)) & ~needle_m
            needle_mass = att[..., needle_m].sum(-1)
            denom = (att.sum(-1) - att[..., sink_m].sum(-1)
                     - att[..., local_m].sum(-1)).clamp_min(1e-6)
            steps_needle.append((needle_mass / denom).cpu().numpy().astype(np.float32))
            argmax_pos = att.argmax(-1)  # [L, H]
            in_needle = (argmax_pos >= s) & (argmax_pos < e)
            tok_lookup = torch.tensor(full_tokens[:ctx], device=argmax_pos.device)
            attended_tok = tok_lookup[argmax_pos]
            steps_copy.append(
                (in_needle & (attended_tok == next_tok)).cpu().numpy()
            )

        gen_ids.append(next_tok)
        full_tokens.append(next_tok)
        if next_tok in eos_ids:
            break
        cur = torch.tensor([[next_tok]], device=device)

    log_meta = {
        "prompt_len": prompt_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", n_heads),
        "topk": topk,
        "sink_tokens": sink_tokens,
        "local_window": local_window,
        "needle_token_span": list(needle_span) if needle_span else None,
        "generated_text": tokenizer.decode(gen_ids),
        "page_size": page_size or None,
        "prefill_chunk": prefill_chunk,
        **(meta or {}),
    }
    page_max = None
    if steps_page:
        width = max(p.shape[-1] for p in steps_page)
        padded = []
        for p in steps_page:
            if p.shape[-1] < width:
                p = np.pad(p, [(0, 0), (0, 0), (0, width - p.shape[-1])])
            padded.append(p)
        page_max = np.stack(padded)
    return RunLog(
        topk_idx=np.stack(steps_idx),
        topk_val=np.stack(steps_val),
        ctx_len=np.array(steps_ctx, dtype=np.int32),
        needle_score=np.stack(steps_needle) if steps_needle else None,
        copy_paste=np.stack(steps_copy) if steps_copy else None,
        page_max=page_max,
        gen_token_ids=gen_ids,
        meta=log_meta,
    )
