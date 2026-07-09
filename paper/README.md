# paper/

Raw LaTeX draft of the WakeKV workshop paper (Phase 0–2a snapshot).
**Kept in-repo, not published.** Rawness is expected — see
`docs/preprint_outline.md` for the plan and `docs/phase2b_scoping.md`
for what the follow-up paper will add.

Real numbers are already filled in from the committed results. Prose
that needs a real pass is marked `\NEEDPROSE{...}`; figures not yet
generated are marked `\NEEDFIG`. Both render in red in the PDF.

## Build

Standard LaTeX toolchain, no workshop style file yet:

```bash
cd paper
pdflatex wakekv.tex && pdflatex wakekv.tex   # twice for refs
```

Uses only base + `amsmath`, `booktabs`, `hyperref`, `graphicx`, `xcolor`,
`geometry`, `url` — should build on any TeX Live install.

## What's real, what's not

| Section | Numbers | Prose |
|---|---|---|
| §3 Churn | ✅ real (Table 1) | needs a pass |
| §4 Prediction | ✅ real (Table 2, all P/R/F1) | needs a pass |
| §5 Clustering | ✅ real (z, boundary-recall) | needs a pass |
| §6 Simulator | ✅ real (Table 3, matched-memory Pareto) | needs a pass |
| §7 Related work | ✅ paragraph from RESEARCH_PLAN | needs expansion |
| §8 Limitations | draft written | probably fine |
| Figures | ❌ pending — 4 figs need generation | — |
| References | ❌ NEEDPROSE placeholders | pull from notes/ |

## Figures to generate (all no-GPU, from committed logs)

1. **Per-step per-head activity heatmap** — CoT run, twelve most-active heads.
2. **Precision-recall curves** — four signals at lead 32.
3. **Wake-up histogram + null envelope** — math500-1.
4. **Memory-vs-miss Pareto** — reactive vs frozen, both regimes.

These become the paper's four figures. Scripts live in
`scripts/analyze_*.py`; a `scripts/make_figures.py` will emit the PDFs
into `paper/figures/` once written.
