# Results directory layout

This directory is for experiment outputs. Keep the paths used by scripts stable, but keep generated files separated by type.

## Tracked summary files

These are small CSV/TEX files that are useful for reports and slides and may be committed intentionally:

- `baseline.csv` — baseline IPC/MPKI summary from `scripts/run_baseline.sh`
- `upper_bound.csv` — built-in-policy sweep summary from `scripts/run_upper_bound.sh`
- `bypass_summary.csv` — bypass experiment summary from `scripts/run_bypass.sh`
- `nn_demo_summary.csv` — neural replay summary from `scripts/run_nn_replay.sh`
- `mlp_demo_summary.csv` — old ChampSim-ML demo summary
- `gru_sweep_summary.csv` — GRU sweep summary, if generated
- `slide8_data.tex` — generated LaTeX snippets for slides

## Local-only generated files

These should normally not be committed:

- `logs/` — simulator logs and long stdout/stderr captures
- `raw/` — raw per-run data that is too large or too messy for the repo
- `tmp/` — scratch files
- `generated/` — generated helper files that can be recreated
- `access_trace.<TRACE>.csv` — large trace-dumper CSVs for Colab/model training
- `*.log` — any run log
- `*.txt` — ad-hoc generated text outputs

## Root-level text files kept for compatibility

The bypass scripts still default to root-level paths such as `bypass_pc_list.txt`. Those files are intentionally left at the repo root so commands like this do not break:

```bash
bash scripts/run_bypass.sh
BYPASS_PC_LIST=bypass_pc_list_10.txt bash scripts/run_bypass.sh
```

Model-generated prefetch lists such as `prefetch_list.txt` and `prefetch_list_MLP.txt` are local-only by default. They can be regenerated from the notebook and are ignored unless explicitly forced with `git add -f`.
