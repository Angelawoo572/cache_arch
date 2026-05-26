# Results directory layout

This directory is for experiment outputs. Keep generated files separated by type and avoid committing large/raw artifacts.

## Tracked summary files

These are small CSV/JSON/TEX files that are useful for reports and slides and may be committed intentionally:

- `baseline.csv` — baseline IPC/MPKI summary from `projects/legacy_gru_prefetch/scripts/run_baseline.sh`
- `upper_bound.csv` — built-in-policy sweep summary from `projects/legacy_gru_prefetch/scripts/run_upper_bound.sh`
- `bypass_summary.csv` — bypass experiment summary from `projects/legacy_gru_prefetch/scripts/run_bypass.sh`
- `nn_demo_summary.csv` — neural replay summary from `projects/legacy_gru_prefetch/scripts/run_nn_replay.sh`
- `mlp_demo_summary.csv` — old ChampSim-ML demo summary
- `gru_sweep_summary.csv` — GRU sweep summary, if generated
- `gru_v8_summary.json` — current GRU V8 offline accuracy/latency summary
- `slide8_data.tex` — generated LaTeX snippets for slides

## Local-only generated files

These should normally not be committed:

- `logs/` — simulator logs and long stdout/stderr captures
- `raw/` — raw per-run data that is too large or too messy for the repo
- `tmp/` — scratch files
- `generated/` — generated helper files that can be recreated
- `generated/prefetch_lists/` — archived Colab/model-generated prefetch lists
- `access_trace.<TRACE>.csv` — large trace-dumper CSVs for Colab/model training
- `*.log` — any run log
- `*.txt` — ad-hoc generated text outputs

## Config files moved out of root

Tracked bypass PC-list configs now live under:

```text
projects/legacy_gru_prefetch/configs/bypass/
```

`run_bypass.sh` defaults to:

```text
projects/legacy_gru_prefetch/configs/bypass/bypass_pc_list.txt
```

To sweep list sizes:

```bash
BYPASS_PC_LIST=projects/legacy_gru_prefetch/configs/bypass/bypass_pc_list_25.txt TAG=top25 bash projects/legacy_gru_prefetch/scripts/run_bypass.sh
```

Model-generated prefetch lists such as `prefetch_list_GRU_V8.txt` are local-only by default. They can be regenerated from the notebook and are ignored unless explicitly forced with `git add -f`.
