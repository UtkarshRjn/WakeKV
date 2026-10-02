#!/usr/bin/env python
"""Wake-up prediction report from logged attention.

  python scripts/attention/analyze_wakeups.py runs/<model>/<task> \
      [--page-size 16] [--pcie-gbps 21] [--decode-step-ms 30]

For every run with needle scores, extracts per-head wake-up events and
evaluates each candidate signal's precision and recall at lead times
{1,2,4,8,16,32} steps, then compares that lead with the time to prefetch
one head's working set from CPU. The paper finds these signals are not
reliable enough to prefetch.

Candidates: the four original attention-derived signals (online_rco, drift,
entropy_trend, needle_mass_delta), ensemble_vote (how many of the four
z-scored attention signals agree), and a data-driven token/wake-event
correlation study — NOT a hand-picked discourse-marker word list, but an
unbiased scan of every generated token id, significance-tested and evaluated
discovery-half vs. held-out-half (see the "## Data-driven token/wake
correlation" section below and wakekv.signals.token_wake_stats). The token
study needs gen_token_ids in the log (raw ids, no tokenizer/decoding
required); it's skipped per-run (reported, not silently dropped) otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakekv.signals import (
    bonferroni_z_bar,
    causal_zscore,
    ensemble_vote,
    evaluate_fixed_threshold,
    evaluate_signal,
    event_horizon_mask,
    page_kv_bytes,
    pooled_threshold,
    signal_discovered_tokens,
    signal_drift,
    signal_entropy_trend,
    signal_needle_mass_delta,
    signal_online_rco,
    token_wake_stats,
    transfer_steps_needed,
    wake_events,
)


def per_head_page_sets(topk_idx: np.ndarray, unit: int, page_size: int) -> list[set]:
    S, L, H, K = topk_idx.shape
    flat = topk_idx.reshape(S, L * H, K)
    return [
        set((flat[s, unit][flat[s, unit] >= 0] // page_size).tolist()) for s in range(S)
    ]


def token_correlation_section(token_bundles: list[dict], args) -> list[str]:
    """Report lines for the data-driven token/wake correlation study.

    Discovery: pool the FIRST HALF of every run's steps, test every
    sufficiently-frequent token id for a significant lift in "wake soon"
    rate (two-proportion z-test, Bonferroni-corrected across all tokens
    tested). Held-out: evaluate the discovered set's precision/recall ONLY
    on the SECOND HALF of each run — data disjoint from what selected the
    tokens, so the reported numbers aren't inflated by cherry-picking over
    a large vocabulary (see wakekv.signals module note above token_wake_stats).
    """
    lines = ["", "## Data-driven token/wake correlation (discovery + held-out)",
             "No hand-picked word list: every token occurring >= "
             f"{args.marker_min_count} times in the discovery half is tested "
             "(two-proportion z-test vs. the run's base wake rate), "
             f"Bonferroni-corrected across all tokens tested (alpha={args.marker_alpha}), "
             "and only tokens clearing the corrected bar are evaluated — "
             "on the held-out half only.", ""]
    if not token_bundles:
        lines.append("No run had usable gen_token_ids (matching length S) — skipped.")
        return lines

    disc_tokens, disc_wake = [], []
    for b in token_bundles:
        half = b["S"] // 2
        mask = event_horizon_mask(b["all_events"], b["S"], args.marker_lead)
        disc_tokens.append(b["token_ids"][:half])
        disc_wake.append(mask[:half])
    disc_tokens = np.concatenate(disc_tokens)
    disc_wake = np.concatenate(disc_wake)

    stats = token_wake_stats(disc_tokens, disc_wake, min_count=args.marker_min_count)
    z_bar = bonferroni_z_bar(len(stats), alpha=args.marker_alpha)
    discovered = {s.token_id for s in stats if s.z > z_bar and s.lift > 1.0}

    lines.append(f"- discovery half: {len(disc_tokens)} steps pooled across "
                 f"{len(token_bundles)} run(s), {len(stats)} distinct token ids "
                 f"tested (min_count={args.marker_min_count})")
    lines.append(f"- Bonferroni z-bar (alpha={args.marker_alpha}, {len(stats)} tests): "
                 f"{z_bar:.2f}")
    lines.append(f"- discovered marker tokens: {len(discovered)}")

    if discovered:
        top = sorted((s for s in stats if s.token_id in discovered), key=lambda s: -s.z)[:10]
        lines += ["", "| token_id | count (discovery half) | hit rate | base rate | lift | z |",
                  "|---|---|---|---|---|---|"]
        for s in top:
            lines.append(f"| {s.token_id} | {s.count} | {s.hit_rate:.3f} "
                         f"| {s.base_rate:.3f} | {s.lift:.2f} | {s.z:.2f} |")

    held_raw = []
    for b in token_bundles:
        half = b["S"] // 2
        sig = signal_discovered_tokens(b["token_ids"], discovered)
        ev_held = [e for e in b["all_events"] if e >= half]
        held_raw.append((sig, ev_held))

    if not discovered:
        lines.append("")
        lines.append("No token cleared the Bonferroni-corrected bar — no evidence "
                     "any single generated token id predicts wake bursts beyond "
                     "chance at this alpha.")
    elif not any(ev for _, ev in held_raw):
        lines.append("")
        lines.append("(no held-out wake events to evaluate against)")
    else:
        lines += ["", "Held-out precision/recall (never seen during discovery):", "",
                  "| lead | precision | recall | events | alarms |",
                  "|---|---|---|---|---|"]
        res = evaluate_fixed_threshold(held_raw, threshold=0.5)
        for lead in sorted(res):
            r = res[lead]
            lines.append(f"| {lead} | {r.precision:.2f} | {r.recall:.2f} "
                         f"| {r.n_events} | {r.n_alarms} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--pcie-gbps", type=float, default=21.0)
    ap.add_argument("--decode-step-ms", type=float, default=30.0)
    ap.add_argument("--max-heads", type=int, default=0, help="0 = only heads with events")
    ap.add_argument("--fixed-quantile", type=float, default=0.9,
                    help="global threshold = this quantile of pooled signal values")
    ap.add_argument("--z-threshold", type=float, default=2.0,
                    help="fire when causal per-head z-score exceeds this")
    ap.add_argument("--z-window", type=int, default=32,
                    help="trailing window for the causal per-head z-score")
    ap.add_argument("--marker-lead", type=int, default=8,
                    help="event-horizon window (steps) defining 'wake soon' "
                         "for token/wake correlation discovery")
    ap.add_argument("--marker-min-count", type=int, default=5,
                    help="skip tokens occurring fewer than this many times "
                         "in the discovery half (z-test needs a real sample)")
    ap.add_argument("--marker-alpha", type=float, default=0.05,
                    help="family-wise significance level (Bonferroni-corrected "
                         "across every distinct token tested)")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    agg: dict[str, list] = defaultdict(list)
    raw: dict[str, list] = defaultdict(list)  # name -> [(signal_array, events)]
    token_bundles: list[dict] = []  # per-run data for the token/wake study
    total_events = 0
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        if "needle_score" not in data:
            print(f"[skip] {rd.name}: no needle score (cot without needle?)")
            continue
        score = data["needle_score"]  # [S, L, H]
        S, L, H = score.shape
        flat_score = score.reshape(S, L * H)
        topk_val = data["topk_val"].reshape(S, L * H, -1)
        # Read topk_idx once per run: decompressing this large npz member on
        # every head (via zipfile) is slow and segfaults numpy's inflate path
        # after many repeated reads.
        topk_idx = data["topk_idx"]
        pool = [max(1, int(c) // args.page_size + 1) for c in data["ctx_len"]]

        units_with_events = []
        for u in range(L * H):
            ev = wake_events(flat_score[:, u])
            if ev:
                units_with_events.append((u, ev))
        total_events += sum(len(ev) for _, ev in units_with_events)
        print(f"[{rd.name}] heads with wake-ups: {len(units_with_events)}, "
              f"events: {sum(len(e) for _, e in units_with_events)}")

        if "gen_token_ids" in data:
            gen_ids = data["gen_token_ids"]
            if len(gen_ids) == S:
                token_bundles.append({
                    "name": rd.name,
                    "token_ids": np.asarray(gen_ids),
                    "S": S,
                    # "any head wakes" pooled across units — the token/wake
                    # study targets synchronized bursts, not one head's own
                    # events, consistent with the prior burstiness finding
                    # (heads tend to wake together, not independently).
                    "all_events": sorted({e for _, ev in units_with_events for e in ev}),
                })
            else:
                print(f"[warn] {rd.name}: gen_token_ids length {len(gen_ids)} "
                      f"!= S={S}, skipping for token-correlation discovery")

        for u, ev in units_with_events[: args.max_heads or None]:
            sets = per_head_page_sets(topk_idx, u, args.page_size)
            sigs = {
                "online_rco": signal_online_rco(sets, pool, k=None),
                "drift": signal_drift(sets),
                "entropy_trend": signal_entropy_trend(topk_val[:, u]),
                "needle_mass_delta": signal_needle_mass_delta(flat_score[:, u]),
            }
            # Ensemble: each of the four attention-derived signals fires if
            # its own causal z-score exceeds args.z_threshold; the ensemble
            # signal is the vote count. None of the four is precise enough
            # to prefetch on its own. This checks whether requiring agreement
            # helps because their false positives differ.
            zvote_inputs = [
                causal_zscore(sigs[n], window=args.z_window)
                for n in ("online_rco", "drift", "entropy_trend", "needle_mass_delta")
            ]
            sigs["ensemble_vote"] = ensemble_vote(zvote_inputs, threshold=args.z_threshold)
            for name, sig in sigs.items():
                for r in evaluate_signal(sig, ev):
                    agg[name].append(r)
                raw[name].append((sig, ev))

    if not agg:
        sys.exit("no wake-up events found — check thresholds or the attention log")

    # Transfer-time bar: promoting one head's full top-k working set.
    kb = page_kv_bytes()
    n_layers = 32  # reported per-layer; actual fetch is per (layer, head)
    steps_bar = transfer_steps_needed(
        pages_to_fetch=64, page_kv_bytes=kb,
        pcie_gbps=args.pcie_gbps, decode_step_ms=args.decode_step_ms,
    )

    lines = [f"# Wake-up prediction — {task_dir}", "",
             f"- total wake-up events: {total_events}",
             f"- transfer bar: promoting 64 pages x 1 KV head ≈ "
             f"{steps_bar:.2f} decode steps "
             f"(@{args.pcie_gbps} GB/s, {args.decode_step_ms} ms/step)", "",
             "| signal | lead | best precision | recall @ that θ |",
             "|---|---|---|---|"]
    best_by_signal = {}
    for name, results in agg.items():
        by_lead = defaultdict(list)
        for r in results:
            by_lead[r.lead].append(r)
        for lead in sorted(by_lead):
            rs = by_lead[lead]
            # aggregate across heads/runs at matching thresholds: mean P/R
            best = max(
                rs, key=lambda r: (0 if np.isnan(r.precision) else r.precision)
            )
            lines.append(
                f"| {name} | {lead} | {best.precision:.2f} | {best.recall:.2f} |"
            )
            if lead >= steps_bar:
                cur = best_by_signal.get(name)
                if cur is None or best.precision > cur[1]:
                    best_by_signal[name] = (lead, best.precision, best.recall)
    lines += ["", "## Prefetch lead",
              f"A signal is early enough to prefetch only if some lead >= "
              f"{steps_bar:.1f} steps still has useful precision and recall."]
    for name, (lead, p, r) in sorted(best_by_signal.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"- {name}: lead {lead} -> precision {p:.2f}, recall {r:.2f}")
    if not best_by_signal:
        lines.append("- No signal clears the transfer bar. The paper therefore "
                     "reacts to demand instead of prefetching.")

    # Honest version: ONE global threshold per signal (what the controller
    # actually uses), events/alarms pooled across all heads — no per-head
    # cherry-picking. This is the grading a runtime policy can actually use.
    lines += ["", f"## Fixed global threshold (quantile {args.fixed_quantile})",
              "One cutoff per signal, applied to every head, pooled P/R. "
              "This is what a runtime controller can actually achieve.", "",
              "| signal | lead | precision | recall | events | alarms |",
              "|---|---|---|---|---|---|"]
    fixed_best = {}
    for name, per_head in raw.items():
        th = pooled_threshold(per_head, args.fixed_quantile)
        res = evaluate_fixed_threshold(per_head, th)
        for lead in sorted(res):
            r = res[lead]
            lines.append(f"| {name} | {lead} | {r.precision:.2f} | {r.recall:.2f} "
                         f"| {r.n_events} | {r.n_alarms} |")
            if lead >= steps_bar and not np.isnan(r.precision):
                cur = fixed_best.get(name)
                # rank by F1 so we don't reward precision at zero recall
                f1 = 0.0 if (r.precision + r.recall) == 0 else \
                    2 * r.precision * r.recall / (r.precision + r.recall)
                if cur is None or f1 > cur[0]:
                    fixed_best[name] = (f1, lead, r.precision, r.recall)
    lines += ["", "Winner under a fixed threshold (best F1 at lead >= bar):"]
    for name, (f1, lead, p, r) in sorted(fixed_best.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"- {name}: lead {lead} -> P {p:.2f}, R {r:.2f} (F1 {f1:.2f})")

    # Deployable middle ground: causal per-head z-score (each head normalized
    # against its OWN trailing window, using only past values) + one global
    # z-threshold. Fair across heterogeneous heads and fires online — this is
    # the number an online policy can actually hit.
    lines += ["", f"## Causal per-head z-score + global z>{args.z_threshold} (deployable)",
              "Each head's signal normalized against its own past "
              f"(window {args.z_window}); one z-threshold for all heads. "
              "No label peeking, fires online. ensemble_vote is already a "
              "combination of z-scored signals (see above), so it's graded "
              "directly at >=2 of 4 agreeing rather than re-z-scored.", "",
              "| signal | lead | precision | recall | events | alarms |",
              "|---|---|---|---|---|---|"]
    z_best = {}
    for name, per_head in raw.items():
        if name == "ensemble_vote":
            zper, th = per_head, 1.5  # already vote-counts; >=2 of 4 agree
        else:
            zper = [(causal_zscore(sig, window=args.z_window), ev) for sig, ev in per_head]
            th = args.z_threshold
        res = evaluate_fixed_threshold(zper, th)
        for lead in sorted(res):
            r = res[lead]
            lines.append(f"| {name} | {lead} | {r.precision:.2f} | {r.recall:.2f} "
                         f"| {r.n_events} | {r.n_alarms} |")
            if lead >= steps_bar and not np.isnan(r.precision):
                f1 = 0.0 if (r.precision + r.recall) == 0 else \
                    2 * r.precision * r.recall / (r.precision + r.recall)
                cur = z_best.get(name)
                if cur is None or f1 > cur[0]:
                    z_best[name] = (f1, lead, r.precision, r.recall)
    lines += ["", "Winner under causal z-score (best F1 at lead >= bar):"]
    for name, (f1, lead, p, r) in sorted(z_best.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"- {name}: lead {lead} -> P {p:.2f}, R {r:.2f} (F1 {f1:.2f})")

    lines += token_correlation_section(token_bundles, args)

    (task_dir / "signal_study.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/signal_study.md")


if __name__ == "__main__":
    main()
