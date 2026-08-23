# Combined Phase 1 signal study — Sun Aug 23 05:26:00 PDT 2026

<!-- ============ Qwen__Qwen2.5-3B-Instruct/multiturn ============ -->
# Phase 1 signal study — /home/utranjan/dynamic-head-kv/runs/Qwen__Qwen2.5-3B-Instruct/multiturn

- total wake-up events: 73
- transfer bar: promoting 64 pages x 1 KV head ≈ 0.00 decode steps (@21.0 GB/s, 30.0 ms/step)

| signal | lead | best precision | recall @ that θ |
|---|---|---|---|
| online_rco | 1 | 0.11 | 1.00 |
| online_rco | 2 | 0.17 | 1.00 |
| online_rco | 4 | 0.28 | 1.00 |
| online_rco | 8 | 0.50 | 0.33 |
| online_rco | 16 | 0.56 | 0.67 |
| online_rco | 32 | 1.00 | 0.67 |
| drift | 1 | 0.14 | 1.00 |
| drift | 2 | 0.29 | 1.00 |
| drift | 4 | 0.60 | 1.00 |
| drift | 8 | 1.00 | 1.00 |
| drift | 16 | 1.00 | 0.50 |
| drift | 32 | 1.00 | 0.50 |
| entropy_trend | 1 | 0.50 | 1.00 |
| entropy_trend | 2 | 0.50 | 1.00 |
| entropy_trend | 4 | 0.50 | 1.00 |
| entropy_trend | 8 | 0.50 | 1.00 |
| entropy_trend | 16 | 1.00 | 1.00 |
| entropy_trend | 32 | 1.00 | 0.50 |
| needle_mass_delta | 1 | 0.06 | 1.00 |
| needle_mass_delta | 2 | 0.11 | 1.00 |
| needle_mass_delta | 4 | 0.17 | 1.00 |
| needle_mass_delta | 8 | 0.44 | 1.00 |
| needle_mass_delta | 16 | 0.44 | 1.00 |
| needle_mass_delta | 32 | 0.67 | 1.00 |
| ensemble_vote | 1 | 1.00 | 1.00 |
| ensemble_vote | 2 | 1.00 | 1.00 |
| ensemble_vote | 4 | 1.00 | 1.00 |
| ensemble_vote | 8 | 1.00 | 1.00 |
| ensemble_vote | 16 | 1.00 | 1.00 |
| ensemble_vote | 32 | 1.00 | 1.00 |

## G1 gate
A signal passes if at some lead >= 0.0 steps it keeps useful precision/recall (judge trade-off; see plan §4).
- online_rco: lead 32 -> precision 1.00, recall 0.67
- drift: lead 8 -> precision 1.00, recall 1.00
- entropy_trend: lead 16 -> precision 1.00, recall 1.00
- ensemble_vote: lead 1 -> precision 1.00, recall 1.00
- needle_mass_delta: lead 32 -> precision 0.67, recall 1.00

## Fixed global threshold (quantile 0.9)
One cutoff per signal, applied to every head, pooled P/R. This is what a runtime controller can actually achieve.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.05 | 73 | 1067 |
| online_rco | 2 | 0.01 | 0.08 | 73 | 1067 |
| online_rco | 4 | 0.02 | 0.23 | 73 | 1067 |
| online_rco | 8 | 0.04 | 0.36 | 73 | 1067 |
| online_rco | 16 | 0.08 | 0.45 | 73 | 1067 |
| online_rco | 32 | 0.14 | 0.55 | 73 | 1067 |
| drift | 1 | 0.01 | 0.08 | 73 | 1057 |
| drift | 2 | 0.02 | 0.14 | 73 | 1057 |
| drift | 4 | 0.03 | 0.14 | 73 | 1057 |
| drift | 8 | 0.06 | 0.15 | 73 | 1057 |
| drift | 16 | 0.15 | 0.23 | 73 | 1057 |
| drift | 32 | 0.24 | 0.23 | 73 | 1057 |
| entropy_trend | 1 | 0.01 | 0.12 | 73 | 1067 |
| entropy_trend | 2 | 0.02 | 0.19 | 73 | 1067 |
| entropy_trend | 4 | 0.04 | 0.34 | 73 | 1067 |
| entropy_trend | 8 | 0.06 | 0.41 | 73 | 1067 |
| entropy_trend | 16 | 0.11 | 0.55 | 73 | 1067 |
| entropy_trend | 32 | 0.17 | 0.59 | 73 | 1067 |
| needle_mass_delta | 1 | 0.00 | 0.00 | 73 | 1067 |
| needle_mass_delta | 2 | 0.00 | 0.00 | 73 | 1067 |
| needle_mass_delta | 4 | 0.00 | 0.00 | 73 | 1067 |
| needle_mass_delta | 8 | 0.00 | 0.00 | 73 | 1067 |
| needle_mass_delta | 16 | 0.01 | 0.07 | 73 | 1067 |
| needle_mass_delta | 32 | 0.05 | 0.21 | 73 | 1067 |
| ensemble_vote | 1 | 0.02 | 0.10 | 73 | 400 |
| ensemble_vote | 2 | 0.03 | 0.14 | 73 | 400 |
| ensemble_vote | 4 | 0.04 | 0.21 | 73 | 400 |
| ensemble_vote | 8 | 0.06 | 0.23 | 73 | 400 |
| ensemble_vote | 16 | 0.07 | 0.27 | 73 | 400 |
| ensemble_vote | 32 | 0.11 | 0.42 | 73 | 400 |

Winner under a fixed threshold (best F1 at lead >= bar):
- entropy_trend: lead 32 -> P 0.17, R 0.59 (F1 0.26)
- drift: lead 32 -> P 0.24, R 0.23 (F1 0.24)
- online_rco: lead 32 -> P 0.14, R 0.55 (F1 0.22)
- ensemble_vote: lead 32 -> P 0.11, R 0.42 (F1 0.17)
- needle_mass_delta: lead 32 -> P 0.05, R 0.21 (F1 0.08)

## Causal per-head z-score + global z>2.0 (deployable)
Each head's signal normalized against its own past (window 32); one z-threshold for all heads. No label peeking, fires online. ensemble_vote is already a combination of z-scored signals (see above), so it's graded directly at >=2 of 4 agreeing rather than re-z-scored.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.01 | 0.10 | 73 | 687 |
| online_rco | 2 | 0.01 | 0.12 | 73 | 687 |
| online_rco | 4 | 0.03 | 0.29 | 73 | 687 |
| online_rco | 8 | 0.06 | 0.40 | 73 | 687 |
| online_rco | 16 | 0.08 | 0.51 | 73 | 687 |
| online_rco | 32 | 0.13 | 0.67 | 73 | 687 |
| drift | 1 | 0.01 | 0.10 | 73 | 1208 |
| drift | 2 | 0.01 | 0.10 | 73 | 1208 |
| drift | 4 | 0.02 | 0.16 | 73 | 1208 |
| drift | 8 | 0.04 | 0.25 | 73 | 1208 |
| drift | 16 | 0.10 | 0.52 | 73 | 1208 |
| drift | 32 | 0.20 | 0.70 | 73 | 1208 |
| entropy_trend | 1 | 0.01 | 0.03 | 73 | 353 |
| entropy_trend | 2 | 0.02 | 0.08 | 73 | 353 |
| entropy_trend | 4 | 0.07 | 0.22 | 73 | 353 |
| entropy_trend | 8 | 0.09 | 0.30 | 73 | 353 |
| entropy_trend | 16 | 0.15 | 0.45 | 73 | 353 |
| entropy_trend | 32 | 0.20 | 0.56 | 73 | 353 |
| needle_mass_delta | 1 | 0.02 | 0.19 | 73 | 779 |
| needle_mass_delta | 2 | 0.04 | 0.33 | 73 | 779 |
| needle_mass_delta | 4 | 0.05 | 0.36 | 73 | 779 |
| needle_mass_delta | 8 | 0.07 | 0.41 | 73 | 779 |
| needle_mass_delta | 16 | 0.07 | 0.42 | 73 | 779 |
| needle_mass_delta | 32 | 0.10 | 0.52 | 73 | 779 |
| ensemble_vote | 1 | 0.02 | 0.10 | 73 | 400 |
| ensemble_vote | 2 | 0.03 | 0.14 | 73 | 400 |
| ensemble_vote | 4 | 0.04 | 0.21 | 73 | 400 |
| ensemble_vote | 8 | 0.06 | 0.23 | 73 | 400 |
| ensemble_vote | 16 | 0.07 | 0.27 | 73 | 400 |
| ensemble_vote | 32 | 0.11 | 0.42 | 73 | 400 |

Winner under causal z-score (best F1 at lead >= bar):
- drift: lead 32 -> P 0.20, R 0.70 (F1 0.31)
- entropy_trend: lead 32 -> P 0.20, R 0.56 (F1 0.29)
- online_rco: lead 32 -> P 0.13, R 0.67 (F1 0.22)
- ensemble_vote: lead 32 -> P 0.11, R 0.42 (F1 0.17)
- needle_mass_delta: lead 32 -> P 0.10, R 0.52 (F1 0.17)

## Data-driven token/wake correlation (discovery + held-out)
No hand-picked word list: every token occurring >= 5 times in the discovery half is tested (two-proportion z-test vs. the run's base wake rate), Bonferroni-corrected across all tokens tested (alpha=0.05), and only tokens clearing the corrected bar are evaluated — on the held-out half only.

- discovery half: 86 steps pooled across 1 run(s), 1 distinct token ids tested (min_count=5)
- Bonferroni z-bar (alpha=0.05, 1 tests): 1.64
- discovered marker tokens: 0

No token cleared the Bonferroni-corrected bar — no evidence any single generated token id predicts wake bursts beyond chance at this alpha.

<!-- ============ Qwen__Qwen2.5-3B-Instruct/niah ============ -->
# Phase 1 signal study — /home/utranjan/dynamic-head-kv/runs/Qwen__Qwen2.5-3B-Instruct/niah

- total wake-up events: 236
- transfer bar: promoting 64 pages x 1 KV head ≈ 0.00 decode steps (@21.0 GB/s, 30.0 ms/step)

| signal | lead | best precision | recall @ that θ |
|---|---|---|---|
| online_rco | 1 | 0.33 | 1.00 |
| online_rco | 2 | 0.33 | 1.00 |
| online_rco | 4 | 0.33 | 1.00 |
| online_rco | 8 | 0.67 | 0.50 |
| online_rco | 16 | 1.00 | 1.00 |
| online_rco | 32 | 1.00 | 1.00 |
| drift | 1 | 0.33 | 0.33 |
| drift | 2 | 0.33 | 0.33 |
| drift | 4 | 1.00 | 0.50 |
| drift | 8 | 1.00 | 0.50 |
| drift | 16 | 1.00 | 0.50 |
| drift | 32 | 1.00 | 0.50 |
| entropy_trend | 1 | 0.33 | 0.50 |
| entropy_trend | 2 | 0.33 | 0.50 |
| entropy_trend | 4 | 0.33 | 1.00 |
| entropy_trend | 8 | 0.67 | 0.50 |
| entropy_trend | 16 | 0.67 | 1.00 |
| entropy_trend | 32 | 1.00 | 1.00 |
| needle_mass_delta | 1 | 0.33 | 1.00 |
| needle_mass_delta | 2 | 0.33 | 1.00 |
| needle_mass_delta | 4 | 0.33 | 1.00 |
| needle_mass_delta | 8 | 0.33 | 1.00 |
| needle_mass_delta | 16 | 0.67 | 1.00 |
| needle_mass_delta | 32 | 0.67 | 1.00 |
| ensemble_vote | 1 | 1.00 | 1.00 |
| ensemble_vote | 2 | 1.00 | 1.00 |
| ensemble_vote | 4 | 1.00 | 1.00 |
| ensemble_vote | 8 | 1.00 | 1.00 |
| ensemble_vote | 16 | 1.00 | 1.00 |
| ensemble_vote | 32 | 1.00 | 1.00 |

## G1 gate
A signal passes if at some lead >= 0.0 steps it keeps useful precision/recall (judge trade-off; see plan §4).
- online_rco: lead 16 -> precision 1.00, recall 1.00
- drift: lead 4 -> precision 1.00, recall 0.50
- entropy_trend: lead 32 -> precision 1.00, recall 1.00
- ensemble_vote: lead 1 -> precision 1.00, recall 1.00
- needle_mass_delta: lead 16 -> precision 0.67, recall 1.00

## Fixed global threshold (quantile 0.9)
One cutoff per signal, applied to every head, pooled P/R. This is what a runtime controller can actually achieve.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.08 | 236 | 5197 |
| online_rco | 2 | 0.01 | 0.11 | 236 | 5197 |
| online_rco | 4 | 0.02 | 0.17 | 236 | 5197 |
| online_rco | 8 | 0.03 | 0.23 | 236 | 5197 |
| online_rco | 16 | 0.06 | 0.31 | 236 | 5197 |
| online_rco | 32 | 0.13 | 0.40 | 236 | 5197 |
| drift | 1 | 0.00 | 0.06 | 236 | 5197 |
| drift | 2 | 0.01 | 0.08 | 236 | 5197 |
| drift | 4 | 0.01 | 0.08 | 236 | 5197 |
| drift | 8 | 0.03 | 0.10 | 236 | 5197 |
| drift | 16 | 0.05 | 0.11 | 236 | 5197 |
| drift | 32 | 0.09 | 0.13 | 236 | 5197 |
| entropy_trend | 1 | 0.01 | 0.12 | 236 | 5197 |
| entropy_trend | 2 | 0.01 | 0.18 | 236 | 5197 |
| entropy_trend | 4 | 0.02 | 0.29 | 236 | 5197 |
| entropy_trend | 8 | 0.04 | 0.47 | 236 | 5197 |
| entropy_trend | 16 | 0.08 | 0.67 | 236 | 5197 |
| entropy_trend | 32 | 0.16 | 0.79 | 236 | 5197 |
| needle_mass_delta | 1 | 0.00 | 0.00 | 236 | 5197 |
| needle_mass_delta | 2 | 0.00 | 0.00 | 236 | 5197 |
| needle_mass_delta | 4 | 0.00 | 0.00 | 236 | 5197 |
| needle_mass_delta | 8 | 0.00 | 0.00 | 236 | 5197 |
| needle_mass_delta | 16 | 0.01 | 0.22 | 236 | 5197 |
| needle_mass_delta | 32 | 0.05 | 0.53 | 236 | 5197 |
| ensemble_vote | 1 | 0.01 | 0.06 | 236 | 1175 |
| ensemble_vote | 2 | 0.02 | 0.08 | 236 | 1175 |
| ensemble_vote | 4 | 0.02 | 0.11 | 236 | 1175 |
| ensemble_vote | 8 | 0.03 | 0.13 | 236 | 1175 |
| ensemble_vote | 16 | 0.06 | 0.23 | 236 | 1175 |
| ensemble_vote | 32 | 0.12 | 0.39 | 236 | 1175 |

Winner under a fixed threshold (best F1 at lead >= bar):
- entropy_trend: lead 32 -> P 0.16, R 0.79 (F1 0.26)
- online_rco: lead 32 -> P 0.13, R 0.40 (F1 0.20)
- ensemble_vote: lead 32 -> P 0.12, R 0.39 (F1 0.18)
- drift: lead 32 -> P 0.09, R 0.13 (F1 0.11)
- needle_mass_delta: lead 32 -> P 0.05, R 0.53 (F1 0.10)

## Causal per-head z-score + global z>2.0 (deployable)
Each head's signal normalized against its own past (window 32); one z-threshold for all heads. No label peeking, fires online. ensemble_vote is already a combination of z-scored signals (see above), so it's graded directly at >=2 of 4 agreeing rather than re-z-scored.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.01 | 0.06 | 236 | 2885 |
| online_rco | 2 | 0.01 | 0.11 | 236 | 2885 |
| online_rco | 4 | 0.02 | 0.16 | 236 | 2885 |
| online_rco | 8 | 0.03 | 0.24 | 236 | 2885 |
| online_rco | 16 | 0.05 | 0.40 | 236 | 2885 |
| online_rco | 32 | 0.11 | 0.62 | 236 | 2885 |
| drift | 1 | 0.00 | 0.08 | 236 | 4667 |
| drift | 2 | 0.01 | 0.15 | 236 | 4667 |
| drift | 4 | 0.02 | 0.20 | 236 | 4667 |
| drift | 8 | 0.04 | 0.25 | 236 | 4667 |
| drift | 16 | 0.08 | 0.41 | 236 | 4667 |
| drift | 32 | 0.14 | 0.55 | 236 | 4667 |
| entropy_trend | 1 | 0.01 | 0.06 | 236 | 2120 |
| entropy_trend | 2 | 0.01 | 0.08 | 236 | 2120 |
| entropy_trend | 4 | 0.02 | 0.14 | 236 | 2120 |
| entropy_trend | 8 | 0.04 | 0.26 | 236 | 2120 |
| entropy_trend | 16 | 0.09 | 0.47 | 236 | 2120 |
| entropy_trend | 32 | 0.16 | 0.67 | 236 | 2120 |
| needle_mass_delta | 1 | 0.02 | 0.19 | 236 | 2075 |
| needle_mass_delta | 2 | 0.03 | 0.22 | 236 | 2075 |
| needle_mass_delta | 4 | 0.03 | 0.24 | 236 | 2075 |
| needle_mass_delta | 8 | 0.04 | 0.27 | 236 | 2075 |
| needle_mass_delta | 16 | 0.07 | 0.39 | 236 | 2075 |
| needle_mass_delta | 32 | 0.14 | 0.64 | 236 | 2075 |
| ensemble_vote | 1 | 0.01 | 0.06 | 236 | 1175 |
| ensemble_vote | 2 | 0.02 | 0.08 | 236 | 1175 |
| ensemble_vote | 4 | 0.02 | 0.11 | 236 | 1175 |
| ensemble_vote | 8 | 0.03 | 0.13 | 236 | 1175 |
| ensemble_vote | 16 | 0.06 | 0.23 | 236 | 1175 |
| ensemble_vote | 32 | 0.12 | 0.39 | 236 | 1175 |

Winner under causal z-score (best F1 at lead >= bar):
- entropy_trend: lead 32 -> P 0.16, R 0.67 (F1 0.26)
- needle_mass_delta: lead 32 -> P 0.14, R 0.64 (F1 0.22)
- drift: lead 32 -> P 0.14, R 0.55 (F1 0.22)
- ensemble_vote: lead 32 -> P 0.12, R 0.39 (F1 0.18)
- online_rco: lead 32 -> P 0.11, R 0.62 (F1 0.18)

## Data-driven token/wake correlation (discovery + held-out)
No hand-picked word list: every token occurring >= 5 times in the discovery half is tested (two-proportion z-test vs. the run's base wake rate), Bonferroni-corrected across all tokens tested (alpha=0.05), and only tokens clearing the corrected bar are evaluated — on the held-out half only.

- discovery half: 1152 steps pooled across 9 run(s), 64 distinct token ids tested (min_count=5)
- Bonferroni z-bar (alpha=0.05, 64 tests): 3.16
- discovered marker tokens: 0

No token cleared the Bonferroni-corrected bar — no evidence any single generated token id predicts wake bursts beyond chance at this alpha.

<!-- ============ deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot ============ -->
# Phase 1 signal study — /home/utranjan/dynamic-head-kv/runs/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot

- total wake-up events: 1137
- transfer bar: promoting 64 pages x 1 KV head ≈ 0.00 decode steps (@21.0 GB/s, 30.0 ms/step)

| signal | lead | best precision | recall @ that θ |
|---|---|---|---|
| online_rco | 1 | 0.14 | 1.00 |
| online_rco | 2 | 0.29 | 1.00 |
| online_rco | 4 | 0.29 | 1.00 |
| online_rco | 8 | 0.29 | 1.00 |
| online_rco | 16 | 0.44 | 0.88 |
| online_rco | 32 | 0.71 | 0.80 |
| drift | 1 | 0.33 | 0.17 |
| drift | 2 | 1.00 | 1.00 |
| drift | 4 | 1.00 | 1.00 |
| drift | 8 | 1.00 | 1.00 |
| drift | 16 | 1.00 | 1.00 |
| drift | 32 | 1.00 | 0.33 |
| entropy_trend | 1 | 0.14 | 0.50 |
| entropy_trend | 2 | 0.29 | 0.50 |
| entropy_trend | 4 | 0.29 | 0.67 |
| entropy_trend | 8 | 0.43 | 1.00 |
| entropy_trend | 16 | 0.43 | 0.50 |
| entropy_trend | 32 | 0.71 | 0.67 |
| needle_mass_delta | 1 | 0.05 | 0.75 |
| needle_mass_delta | 2 | 0.05 | 0.80 |
| needle_mass_delta | 4 | 0.08 | 1.00 |
| needle_mass_delta | 8 | 0.11 | 1.00 |
| needle_mass_delta | 16 | 0.29 | 0.67 |
| needle_mass_delta | 32 | 0.57 | 1.00 |
| ensemble_vote | 1 | 0.09 | 0.17 |
| ensemble_vote | 2 | 0.17 | 1.00 |
| ensemble_vote | 4 | 1.00 | 0.50 |
| ensemble_vote | 8 | 1.00 | 0.50 |
| ensemble_vote | 16 | 1.00 | 0.17 |
| ensemble_vote | 32 | 1.00 | 1.00 |

## G1 gate
A signal passes if at some lead >= 0.0 steps it keeps useful precision/recall (judge trade-off; see plan §4).
- drift: lead 2 -> precision 1.00, recall 1.00
- ensemble_vote: lead 4 -> precision 1.00, recall 0.50
- online_rco: lead 32 -> precision 0.71, recall 0.80
- entropy_trend: lead 32 -> precision 0.71, recall 0.67
- needle_mass_delta: lead 32 -> precision 0.57, recall 1.00

## Fixed global threshold (quantile 0.9)
One cutoff per signal, applied to every head, pooled P/R. This is what a runtime controller can actually achieve.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.09 | 1137 | 76021 |
| online_rco | 2 | 0.00 | 0.14 | 1137 | 76021 |
| online_rco | 4 | 0.01 | 0.25 | 1137 | 76021 |
| online_rco | 8 | 0.01 | 0.45 | 1137 | 76021 |
| online_rco | 16 | 0.03 | 0.69 | 1137 | 76021 |
| online_rco | 32 | 0.05 | 0.85 | 1137 | 76021 |
| drift | 1 | 0.00 | 0.17 | 1137 | 72465 |
| drift | 2 | 0.01 | 0.18 | 1137 | 72465 |
| drift | 4 | 0.01 | 0.19 | 1137 | 72465 |
| drift | 8 | 0.02 | 0.19 | 1137 | 72465 |
| drift | 16 | 0.04 | 0.21 | 1137 | 72465 |
| drift | 32 | 0.07 | 0.22 | 1137 | 72465 |
| entropy_trend | 1 | 0.00 | 0.07 | 1137 | 76075 |
| entropy_trend | 2 | 0.00 | 0.13 | 1137 | 76075 |
| entropy_trend | 4 | 0.00 | 0.24 | 1137 | 76075 |
| entropy_trend | 8 | 0.01 | 0.48 | 1137 | 76075 |
| entropy_trend | 16 | 0.02 | 0.67 | 1137 | 76075 |
| entropy_trend | 32 | 0.04 | 0.82 | 1137 | 76075 |
| needle_mass_delta | 1 | 0.00 | 0.00 | 1137 | 76075 |
| needle_mass_delta | 2 | 0.00 | 0.00 | 1137 | 76075 |
| needle_mass_delta | 4 | 0.00 | 0.00 | 1137 | 76075 |
| needle_mass_delta | 8 | 0.00 | 0.00 | 1137 | 76075 |
| needle_mass_delta | 16 | 0.01 | 0.49 | 1137 | 76075 |
| needle_mass_delta | 32 | 0.02 | 0.72 | 1137 | 76075 |
| ensemble_vote | 1 | 0.00 | 0.01 | 1137 | 12956 |
| ensemble_vote | 2 | 0.00 | 0.02 | 1137 | 12956 |
| ensemble_vote | 4 | 0.00 | 0.03 | 1137 | 12956 |
| ensemble_vote | 8 | 0.01 | 0.07 | 1137 | 12956 |
| ensemble_vote | 16 | 0.03 | 0.26 | 1137 | 12956 |
| ensemble_vote | 32 | 0.05 | 0.43 | 1137 | 12956 |

Winner under a fixed threshold (best F1 at lead >= bar):
- drift: lead 32 -> P 0.07, R 0.22 (F1 0.10)
- online_rco: lead 32 -> P 0.05, R 0.85 (F1 0.10)
- ensemble_vote: lead 32 -> P 0.05, R 0.43 (F1 0.09)
- entropy_trend: lead 32 -> P 0.04, R 0.82 (F1 0.08)
- needle_mass_delta: lead 32 -> P 0.02, R 0.72 (F1 0.05)

## Causal per-head z-score + global z>2.0 (deployable)
Each head's signal normalized against its own past (window 32); one z-threshold for all heads. No label peeking, fires online. ensemble_vote is already a combination of z-scored signals (see above), so it's graded directly at >=2 of 4 agreeing rather than re-z-scored.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.04 | 1137 | 43261 |
| online_rco | 2 | 0.00 | 0.07 | 1137 | 43261 |
| online_rco | 4 | 0.00 | 0.13 | 1137 | 43261 |
| online_rco | 8 | 0.01 | 0.28 | 1137 | 43261 |
| online_rco | 16 | 0.03 | 0.54 | 1137 | 43261 |
| online_rco | 32 | 0.05 | 0.79 | 1137 | 43261 |
| drift | 1 | 0.00 | 0.13 | 1137 | 45014 |
| drift | 2 | 0.01 | 0.14 | 1137 | 45014 |
| drift | 4 | 0.01 | 0.18 | 1137 | 45014 |
| drift | 8 | 0.02 | 0.23 | 1137 | 45014 |
| drift | 16 | 0.03 | 0.31 | 1137 | 45014 |
| drift | 32 | 0.05 | 0.44 | 1137 | 45014 |
| entropy_trend | 1 | 0.00 | 0.02 | 1137 | 26282 |
| entropy_trend | 2 | 0.00 | 0.06 | 1137 | 26282 |
| entropy_trend | 4 | 0.01 | 0.11 | 1137 | 26282 |
| entropy_trend | 8 | 0.01 | 0.24 | 1137 | 26282 |
| entropy_trend | 16 | 0.02 | 0.40 | 1137 | 26282 |
| entropy_trend | 32 | 0.04 | 0.60 | 1137 | 26282 |
| needle_mass_delta | 1 | 0.00 | 0.04 | 1137 | 37624 |
| needle_mass_delta | 2 | 0.00 | 0.06 | 1137 | 37624 |
| needle_mass_delta | 4 | 0.00 | 0.07 | 1137 | 37624 |
| needle_mass_delta | 8 | 0.00 | 0.09 | 1137 | 37624 |
| needle_mass_delta | 16 | 0.02 | 0.51 | 1137 | 37624 |
| needle_mass_delta | 32 | 0.04 | 0.79 | 1137 | 37624 |
| ensemble_vote | 1 | 0.00 | 0.01 | 1137 | 12956 |
| ensemble_vote | 2 | 0.00 | 0.02 | 1137 | 12956 |
| ensemble_vote | 4 | 0.00 | 0.03 | 1137 | 12956 |
| ensemble_vote | 8 | 0.01 | 0.07 | 1137 | 12956 |
| ensemble_vote | 16 | 0.03 | 0.26 | 1137 | 12956 |
| ensemble_vote | 32 | 0.05 | 0.43 | 1137 | 12956 |

Winner under causal z-score (best F1 at lead >= bar):
- drift: lead 32 -> P 0.05, R 0.44 (F1 0.10)
- online_rco: lead 32 -> P 0.05, R 0.79 (F1 0.10)
- ensemble_vote: lead 32 -> P 0.05, R 0.43 (F1 0.09)
- entropy_trend: lead 32 -> P 0.04, R 0.60 (F1 0.08)
- needle_mass_delta: lead 32 -> P 0.04, R 0.79 (F1 0.08)

## Data-driven token/wake correlation (discovery + held-out)
No hand-picked word list: every token occurring >= 5 times in the discovery half is tested (two-proportion z-test vs. the run's base wake rate), Bonferroni-corrected across all tokens tested (alpha=0.05), and only tokens clearing the corrected bar are evaluated — on the held-out half only.

- discovery half: 3182 steps pooled across 5 run(s), 129 distinct token ids tested (min_count=5)
- Bonferroni z-bar (alpha=0.05, 129 tests): 3.36
- discovered marker tokens: 4

| token_id | count (discovery half) | hit rate | base rate | lift | z |
|---|---|---|---|---|---|
| 1396 | 8 | 1.000 | 0.218 | 4.59 | 5.33 |
| 20597 | 5 | 1.000 | 0.218 | 4.59 | 4.22 |
| 220 | 168 | 0.357 | 0.218 | 1.64 | 4.21 |
| 10461 | 9 | 0.778 | 0.218 | 3.57 | 4.05 |

Held-out precision/recall (never seen during discovery):

| lead | precision | recall | events | alarms |
|---|---|---|---|---|
| 1 | 0.05 | 0.17 | 140 | 460 |
| 2 | 0.07 | 0.24 | 140 | 460 |
| 4 | 0.11 | 0.39 | 140 | 460 |
| 8 | 0.20 | 0.59 | 140 | 460 |
| 16 | 0.31 | 0.76 | 140 | 460 |
| 32 | 0.48 | 0.89 | 140 | 460 |

<!-- ============ deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot ============ -->
# Phase 1 signal study — /home/utranjan/dynamic-head-kv/runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot

- total wake-up events: 407
- transfer bar: promoting 64 pages x 1 KV head ≈ 0.00 decode steps (@21.0 GB/s, 30.0 ms/step)

| signal | lead | best precision | recall @ that θ |
|---|---|---|---|
| online_rco | 1 | 0.12 | 0.20 |
| online_rco | 2 | 0.17 | 1.00 |
| online_rco | 4 | 0.22 | 0.40 |
| online_rco | 8 | 0.38 | 0.60 |
| online_rco | 16 | 0.57 | 0.20 |
| online_rco | 32 | 0.88 | 0.60 |
| drift | 1 | 0.20 | 1.00 |
| drift | 2 | 0.40 | 1.00 |
| drift | 4 | 0.80 | 1.00 |
| drift | 8 | 1.00 | 1.00 |
| drift | 16 | 1.00 | 1.00 |
| drift | 32 | 1.00 | 1.00 |
| entropy_trend | 1 | 0.12 | 0.20 |
| entropy_trend | 2 | 0.17 | 1.00 |
| entropy_trend | 4 | 0.17 | 1.00 |
| entropy_trend | 8 | 0.20 | 0.80 |
| entropy_trend | 16 | 0.36 | 0.90 |
| entropy_trend | 32 | 0.64 | 1.00 |
| needle_mass_delta | 1 | 0.11 | 0.50 |
| needle_mass_delta | 2 | 0.14 | 1.00 |
| needle_mass_delta | 4 | 0.15 | 0.67 |
| needle_mass_delta | 8 | 0.26 | 0.67 |
| needle_mass_delta | 16 | 0.62 | 0.60 |
| needle_mass_delta | 32 | 1.00 | 0.43 |
| ensemble_vote | 1 | 0.14 | 0.25 |
| ensemble_vote | 2 | 0.33 | 1.00 |
| ensemble_vote | 4 | 0.33 | 1.00 |
| ensemble_vote | 8 | 1.00 | 0.20 |
| ensemble_vote | 16 | 1.00 | 0.20 |
| ensemble_vote | 32 | 1.00 | 0.20 |

## G1 gate
A signal passes if at some lead >= 0.0 steps it keeps useful precision/recall (judge trade-off; see plan §4).
- drift: lead 8 -> precision 1.00, recall 1.00
- needle_mass_delta: lead 32 -> precision 1.00, recall 0.43
- ensemble_vote: lead 8 -> precision 1.00, recall 0.20
- online_rco: lead 32 -> precision 0.88, recall 0.60
- entropy_trend: lead 32 -> precision 0.64, recall 1.00

## Fixed global threshold (quantile 0.9)
One cutoff per signal, applied to every head, pooled P/R. This is what a runtime controller can actually achieve.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.08 | 407 | 18325 |
| online_rco | 2 | 0.00 | 0.14 | 407 | 18325 |
| online_rco | 4 | 0.01 | 0.20 | 407 | 18325 |
| online_rco | 8 | 0.01 | 0.34 | 407 | 18325 |
| online_rco | 16 | 0.03 | 0.56 | 407 | 18325 |
| online_rco | 32 | 0.07 | 0.71 | 407 | 18325 |
| drift | 1 | 0.00 | 0.17 | 407 | 18305 |
| drift | 2 | 0.01 | 0.18 | 407 | 18305 |
| drift | 4 | 0.02 | 0.19 | 407 | 18305 |
| drift | 8 | 0.03 | 0.20 | 407 | 18305 |
| drift | 16 | 0.05 | 0.20 | 407 | 18305 |
| drift | 32 | 0.09 | 0.21 | 407 | 18305 |
| entropy_trend | 1 | 0.00 | 0.12 | 407 | 18329 |
| entropy_trend | 2 | 0.01 | 0.20 | 407 | 18329 |
| entropy_trend | 4 | 0.01 | 0.33 | 407 | 18329 |
| entropy_trend | 8 | 0.02 | 0.46 | 407 | 18329 |
| entropy_trend | 16 | 0.05 | 0.63 | 407 | 18329 |
| entropy_trend | 32 | 0.08 | 0.74 | 407 | 18329 |
| needle_mass_delta | 1 | 0.00 | 0.00 | 407 | 18329 |
| needle_mass_delta | 2 | 0.00 | 0.00 | 407 | 18329 |
| needle_mass_delta | 4 | 0.00 | 0.00 | 407 | 18329 |
| needle_mass_delta | 8 | 0.00 | 0.00 | 407 | 18329 |
| needle_mass_delta | 16 | 0.02 | 0.43 | 407 | 18329 |
| needle_mass_delta | 32 | 0.05 | 0.71 | 407 | 18329 |
| ensemble_vote | 1 | 0.00 | 0.00 | 407 | 3149 |
| ensemble_vote | 2 | 0.00 | 0.03 | 407 | 3149 |
| ensemble_vote | 4 | 0.01 | 0.04 | 407 | 3149 |
| ensemble_vote | 8 | 0.01 | 0.07 | 407 | 3149 |
| ensemble_vote | 16 | 0.04 | 0.24 | 407 | 3149 |
| ensemble_vote | 32 | 0.08 | 0.46 | 407 | 3149 |

Winner under a fixed threshold (best F1 at lead >= bar):
- entropy_trend: lead 32 -> P 0.08, R 0.74 (F1 0.15)
- ensemble_vote: lead 32 -> P 0.08, R 0.46 (F1 0.13)
- online_rco: lead 32 -> P 0.07, R 0.71 (F1 0.13)
- drift: lead 32 -> P 0.09, R 0.21 (F1 0.13)
- needle_mass_delta: lead 32 -> P 0.05, R 0.71 (F1 0.09)

## Causal per-head z-score + global z>2.0 (deployable)
Each head's signal normalized against its own past (window 32); one z-threshold for all heads. No label peeking, fires online. ensemble_vote is already a combination of z-scored signals (see above), so it's graded directly at >=2 of 4 agreeing rather than re-z-scored.

| signal | lead | precision | recall | events | alarms |
|---|---|---|---|---|---|
| online_rco | 1 | 0.00 | 0.03 | 407 | 10703 |
| online_rco | 2 | 0.00 | 0.08 | 407 | 10703 |
| online_rco | 4 | 0.00 | 0.12 | 407 | 10703 |
| online_rco | 8 | 0.01 | 0.23 | 407 | 10703 |
| online_rco | 16 | 0.03 | 0.45 | 407 | 10703 |
| online_rco | 32 | 0.06 | 0.69 | 407 | 10703 |
| drift | 1 | 0.00 | 0.06 | 407 | 10305 |
| drift | 2 | 0.01 | 0.08 | 407 | 10305 |
| drift | 4 | 0.01 | 0.13 | 407 | 10305 |
| drift | 8 | 0.02 | 0.19 | 407 | 10305 |
| drift | 16 | 0.05 | 0.30 | 407 | 10305 |
| drift | 32 | 0.08 | 0.41 | 407 | 10305 |
| entropy_trend | 1 | 0.00 | 0.02 | 407 | 4377 |
| entropy_trend | 2 | 0.00 | 0.03 | 407 | 4377 |
| entropy_trend | 4 | 0.01 | 0.07 | 407 | 4377 |
| entropy_trend | 8 | 0.01 | 0.13 | 407 | 4377 |
| entropy_trend | 16 | 0.03 | 0.28 | 407 | 4377 |
| entropy_trend | 32 | 0.06 | 0.45 | 407 | 4377 |
| needle_mass_delta | 1 | 0.00 | 0.01 | 407 | 8383 |
| needle_mass_delta | 2 | 0.00 | 0.04 | 407 | 8383 |
| needle_mass_delta | 4 | 0.00 | 0.04 | 407 | 8383 |
| needle_mass_delta | 8 | 0.00 | 0.06 | 407 | 8383 |
| needle_mass_delta | 16 | 0.02 | 0.42 | 407 | 8383 |
| needle_mass_delta | 32 | 0.06 | 0.72 | 407 | 8383 |
| ensemble_vote | 1 | 0.00 | 0.00 | 407 | 3149 |
| ensemble_vote | 2 | 0.00 | 0.03 | 407 | 3149 |
| ensemble_vote | 4 | 0.01 | 0.04 | 407 | 3149 |
| ensemble_vote | 8 | 0.01 | 0.07 | 407 | 3149 |
| ensemble_vote | 16 | 0.04 | 0.24 | 407 | 3149 |
| ensemble_vote | 32 | 0.08 | 0.46 | 407 | 3149 |

Winner under causal z-score (best F1 at lead >= bar):
- drift: lead 32 -> P 0.08, R 0.41 (F1 0.14)
- ensemble_vote: lead 32 -> P 0.08, R 0.46 (F1 0.13)
- entropy_trend: lead 32 -> P 0.06, R 0.45 (F1 0.11)
- online_rco: lead 32 -> P 0.06, R 0.69 (F1 0.11)
- needle_mass_delta: lead 32 -> P 0.06, R 0.72 (F1 0.10)

## Data-driven token/wake correlation (discovery + held-out)
No hand-picked word list: every token occurring >= 5 times in the discovery half is tested (two-proportion z-test vs. the run's base wake rate), Bonferroni-corrected across all tokens tested (alpha=0.05), and only tokens clearing the corrected bar are evaluated — on the held-out half only.

- discovery half: 2186 steps pooled across 5 run(s), 99 distinct token ids tested (min_count=5)
- Bonferroni z-bar (alpha=0.05, 99 tests): 3.29
- discovered marker tokens: 1

| token_id | count (discovery half) | hit rate | base rate | lift | z |
|---|---|---|---|---|---|
| 220 | 93 | 0.559 | 0.384 | 1.46 | 3.39 |

Held-out precision/recall (never seen during discovery):

| lead | precision | recall | events | alarms |
|---|---|---|---|---|
| 1 | 0.04 | 0.05 | 139 | 173 |
| 2 | 0.08 | 0.10 | 139 | 173 |
| 4 | 0.08 | 0.12 | 139 | 173 |
| 8 | 0.13 | 0.22 | 139 | 173 |
| 16 | 0.28 | 0.46 | 139 | 173 |
| 32 | 0.43 | 0.68 | 139 | 173 |
