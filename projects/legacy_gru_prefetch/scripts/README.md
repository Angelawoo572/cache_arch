# Scripts layout

Keep executable scripts directly under `projects/legacy_gru_prefetch/scripts/` so existing commands such as `bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh` keep working. This README groups them by role.

## Setup / build

- `setup_champsim.sh` — clone/update external ChampSim repos, download traces, and run a smoke test.
- `install_and_build.sh` — build baseline, trace dumper, and list replayer binaries.
- `install_bypass.sh` — build the bypass experiment binary.

## Trace and data generation

- `dump_trace.sh` — run ChampSim trace dumper and generate `results/access_trace.<TRACE>.csv`.
- `profile_bypass_pcs.py` — analyze an access-trace CSV and generate `projects/legacy_gru_prefetch/configs/bypass/bypass_pc_list*.txt`.
- `make_slide8_data.py` — generate small LaTeX/data snippets for slides.

## Experiment runners

- `run_baseline.sh` — baseline run.
- `run_upper_bound.sh` — built-in policy / upper-bound style sweep.
- `run_bypass.sh` — LRU vs LRU+bypass using a PC list from `projects/legacy_gru_prefetch/configs/bypass/`.
- `run_nn_replay.sh` — replay a model-generated `prefetch_list*.txt` through ChampSim.
- `run_mlp_demo.sh` — old ChampSim-ML demo path.
- `run_all.sh`, `run_all_models.sh` — convenience batch runners.

## GRU-specific runners

- `run_gru_sweep.sh` — older GRU V1--V4 controlled feature sweep replay.
- `run_gru_v8.sh` — current GRU V8 replay wrapper.
- `run_gru_v9.sh`, `run_gru_v9_sweep.sh` — local V9 scripts if you decide to commit them.

## Cleanup

- `organize_results.sh` — move logs and generated prefetch lists into ignored result subdirectories.

Do not commit generated `*.log`, `prefetch_list*.txt`, or large `results/access_trace*.csv` files.
