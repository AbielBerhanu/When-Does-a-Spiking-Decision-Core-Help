# AIxDke2026: Neuromorphic spoken-action decision core

This repository contains the six-page IEEE conference manuscript, reproducible simulation code, held-out prediction scores, and tables. The manuscript is [`paper1_action/paper.pdf`](paper1_action/paper.pdf); its LaTeX source is [`paper1_action/paper.tex`](paper1_action/paper.tex).

## Main finding

The spiking core is competitive for recognized words, but its small French lead over leak-matched tanh is inconclusive across seeds. A GRU is stronger on acoustic states; TF--IDF logistic regression is stronger on recognized words and on SLURP selective risk and simulated action cost. No hardware energy or clinical claim is made.

## Contents

- `paper1_action/*.py`: preparation, training, evaluation, analysis, validation, plotting, and table-export code.
- `paper1_action/tables/*.csv`: machine-readable result tables.
- `paper1_action/results/*.json`: per-run metrics and summaries.
- `paper1_action/results/*.npz`: held-out logits, labels, and paired item IDs for independent analysis; no waveform files.
- `paper1_action/fresh_benchmark_protocol.md`: the fixed French extension protocol.
- `paper1_action/reproduce.md`: dataset sources, commands, and interpretation limits.

## Rebuild

From the repository root, install `paper1_action/requirements.txt` in a Python environment and follow `paper1_action/reproduce.md`. Raw SLURP, Speech-MASSIVE, and EasyCall recordings and trained checkpoints are intentionally omitted. The included predictions and summaries support result audits without redistributing those corpora. To build the PDF, run `tectonic -C -o paper1_action paper1_action/paper.tex` with Tectonic installed.
