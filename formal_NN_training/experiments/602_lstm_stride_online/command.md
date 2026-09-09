# Commands and paths

The experiment uses Sacramento CPU execution with two jobs and one PyTorch
thread per worker. No Colab, GPU, Overleaf or new system installation is used.
Raw runs are `/home/qianruw/cache_online_runs`; existing runs under `~/cache`
remain read-only. Authentication is performed in the user's terminal.

## Connect and execute on Sacramento

The authenticated master was opened locally with:

```sh
ssh -M -S /tmp/cache602-online-ssh -o ControlPersist=8h qianruw@sacramento.ece.local.cmu.edu
```

The healthy live branch was used to create this isolated server worktree:

```sh
git -C /home/qianruw/cache worktree add -b experiment/602-stride-online-learning /home/qianruw/cache_online experiment/602-stride-live-inference
```

Do not repeat branch creation when resuming. The actual trace is
`/home/qianruw/cache/traces/602.gcc_s-734B.champsimtrace.xz`. The four seed-7
checkpoint directories are below:

```text
/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points/h8/i1m/seed7
/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points/h8/i20m/seed7
/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points/h16/i1m/seed7
/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points/h16/i20m/seed7
```

Each supplies `offline/model.pt`, `export/model.bin`, and metadata. Frozen arms
use the existing exported weights. Warm adaptation loads the i1m checkpoint;
random arms initialize from their specified seed with no pretraining statistics.

The installed environment is `/home/qianruw/venvs/cache-few-nn-cpu/bin/python`
with Torch 2.5.1+cpu and NumPy 1.26.4. These commands were executed:

```sh
cd /home/qianruw/cache_online
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ONLINE_PY=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
ONLINE_ENTRY=/home/qianruw/cache_online/formal_NN_training/experiments/602_lstm_stride_online/run.py
"$ONLINE_PY" "$ONLINE_ENTRY" build
"$ONLINE_PY" "$ONLINE_ENTRY" test
"$ONLINE_PY" "$ONLINE_ENTRY" pilot
"$ONLINE_PY" "$ONLINE_ENTRY" lock
"$ONLINE_PY" "$ONLINE_ENTRY" run --select historical
"$ONLINE_PY" "$ONLINE_ENTRY" run
"$ONLINE_PY" "$ONLINE_ENTRY" status
```

The development pilot uses records 20M--20.1M and 100k simulated retired
instructions. Final cold runs skip 25M records and execute exactly the next 25M;
historical runs retain original 25M warmup + 25M measurement/overshoot semantics.
The exact per-run executable command, paths, exit status, time and RSS are in
each raw `run.json`, `run.log`, and `host_time.txt`. Snapshot/counter output is
`snapshots.jsonl`; actual optimizer statistics are `worker_stats.json`.

`build` always creates a new isolated build directory. The current binary is
recorded in `/home/qianruw/cache_online_runs/build.json`. Builds reconstruct the
inspected ChampSim base and apply `patches/inherited_cache.patch` before the new
hooks, reuse the existing libbf build, and never clean the active simulator.
The final measurement build is `builds/build_04`; the historical parity checks
used `builds/build_03`. Their forward arithmetic, historical service behavior
and legacy counters match; build04 adds decoded-buffer/retained-bank accounting.

## Safe stopping and resuming

These commands act only on recorded, verified process groups belonging to this
experiment. Stop also prevents the driver from launching queued jobs. Resume
clears that stop marker, skips complete runs, and preserves failed/incomplete
attempts under `failed/` before starting fresh attempts. It does not resume an
optimizer halfway through a run or overwrite older raw results.

```sh
ssh -S /tmp/cache602-online-ssh qianruw@sacramento.ece.local.cmu.edu '/home/qianruw/venvs/cache-few-nn-cpu/bin/python /home/qianruw/cache_online/formal_NN_training/experiments/602_lstm_stride_online/run.py status'
ssh -S /tmp/cache602-online-ssh qianruw@sacramento.ece.local.cmu.edu '/home/qianruw/venvs/cache-few-nn-cpu/bin/python /home/qianruw/cache_online/formal_NN_training/experiments/602_lstm_stride_online/run.py stop'
ssh -S /tmp/cache602-online-ssh qianruw@sacramento.ece.local.cmu.edu '/home/qianruw/venvs/cache-few-nn-cpu/bin/python /home/qianruw/cache_online/formal_NN_training/experiments/602_lstm_stride_online/run.py resume'
```

`stop` was used once after a saved-tensor measurement-hook reference cycle was
identified. A detached saved view preserves the exact training arithmetic while
avoiding old graph retention. The affected attempts were preserved under
`/home/qianruw/cache_online_runs/pre_hook_fix_1788928492320801860`. The worker
tests and pilot were repeated, then `resume` reran the online matrix with the
locked recipe. Finished frozen references were reused. No other user's or
earlier experiment's process was targeted.

## Local source and report destination

The damaged local `cache/.git` was not repaired or reset. A healthy adjacent
checkout supplied an independent clone at:

```text
/Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement/work/cache_online
```

The user approved the new report destination:
`/Users/angelawoo/Documents/CMU/ece/cache/docs/602_stride_online_learning`.
Existing slides and `docs/index.html` are preserved.

Report generation uses the already installed local dynamics Python environment
for matplotlib, then Sacramento's existing `pdflatex` to compile. The following commands generated the final figures and compiled PDF from all
24 completed runs. The full final software suite passed 29 tests.


```sh
cd /Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement
rsync -a -e 'ssh -S /tmp/cache602-online-ssh' --exclude='initial_model.bin' qianruw@sacramento.ece.local.cmu.edu:/home/qianruw/cache_online_runs/final/ work/raw_final/
rsync -a -e 'ssh -S /tmp/cache602-online-ssh' --exclude='initial_model.bin' qianruw@sacramento.ece.local.cmu.edu:/home/qianruw/cache_online_runs/pilot/ work/raw_pilot/
MPLCONFIGDIR=/tmp/online602-matplotlib /Users/angelawoo/Documents/CMU/ece/cache_live_dynamics/formal_NN_training/experiments/602_lstm_stride_live_dynamics/.venv/bin/python work/cache_online/formal_NN_training/experiments/602_lstm_stride_online/run.py report --run-dir work/raw_final --out-dir work/cache_online/formal_NN_training/experiments/602_lstm_stride_online/report --historical-root /Users/angelawoo/Documents/CMU/ece/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7 --compile-via qianruw@sacramento.ece.local.cmu.edu --ssh-control /tmp/cache602-online-ssh
```

The local Python environment already contains matplotlib. The report entry point
uses a fresh private build folder under `/home/qianruw/cache_online_runs/report_builds`,
runs the existing `/usr/bin/pdflatex` twice, and copies the compiled PDF back.
All rendered pages were inspected with the installed Poppler tools. Neither a
TeX installation nor an Overleaf upload was necessary. Figure CSV/PDF inputs
and the generated LaTeX are sufficient to compile the report independently.

The final delivery copies preserve all existing presentations:

```sh
mkdir -p /Users/angelawoo/Documents/CMU/ece/cache/docs/602_stride_online_learning
rsync -a --exclude='*.png' work/cache_online/formal_NN_training/experiments/602_lstm_stride_online/report/ /Users/angelawoo/Documents/CMU/ece/cache/docs/602_stride_online_learning/
rsync -a --exclude='*.png' work/cache_online/formal_NN_training/experiments/602_lstm_stride_online/report/ outputs/
cp work/cache_online/formal_NN_training/experiments/602_lstm_stride_online/command.md outputs/command.md
```

Only the new experiment directory is committed and only the new branch is pushed:

```sh
cd /Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement/work/cache_online
git add formal_NN_training/experiments/602_lstm_stride_online
git commit -m "Run causal online Stride learning and report measured limits"
git push -u origin experiment/602-stride-online-learning
```

The normal `resume` command above verifies/skips completed compatible outputs;
it restarts missing or interrupted independent runs with a fresh optimizer.
The build/test/pilot/lock/run sequence reproduces the suite in the configured
run tree. It never silently overwrites a completed result. Raw traces, checkpoint
weights, training IPC payloads and large simulator logs stay outside Git.
