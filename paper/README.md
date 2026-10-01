# paper/

Accepted manuscript for the NeurIPS 2026 Workshop on ML for Systems:

[`wakekv_mlforsys.pdf`](wakekv_mlforsys.pdf)

Source for that PDF is [`wakekv_mlforsys.tex`](wakekv_mlforsys.tex)
(`\workshoptitle{ML for Systems 2026}`). [`wakekv.tex`](wakekv.tex) is an
earlier draft and is not the accepted version.

The checked-in PDF is the submission build (line numbers, and the NeurIPS
style’s “Submitted … Do not distribute” footer). Camera-ready copy drops
both of those when the style is loaded with the `final` option:

```latex
\usepackage[sglblindworkshop,final,nonatbib]{neurips_2026}
```

## Build

```bash
cd paper
pdflatex wakekv_mlforsys.tex && pdflatex wakekv_mlforsys.tex
```
