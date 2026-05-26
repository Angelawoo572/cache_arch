# Notebook layout

This directory contains offline ML/model-development notebooks.

## Active GRU notebooks

- `gru_sweep_v8.ipynb` — current active GRU delta-prediction notebook. Use this first for the current research direction.
- `gru_sweep_cross_trace.ipynb` — earlier controlled-variable GRU feature sweep. Kept for comparison and methodology notes.

## Older model-zoo notebooks

- `neural_prefetcher_zoo_v3.ipynb`
- `neural_prefetcher_zoo_v2.ipynb`

These are useful as baselines/history, but the current direction is the GRU V8 / resource-aware formulation rather than direct next-offset prediction.

## Generated files

Do not commit model-generated prefetch lists from Colab. Files such as `prefetch_list_GRU_V8.txt` should stay local and are ignored by git.
