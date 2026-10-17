# Offline training and online inference: executed commands

The active branch is `experiment/602-stride-dynamic-inference`. It uses existing
offline-trained seed-7 weights and runs only NoPF, conventional Stride and frozen
inference. The 13 retained scientific runs are compatible existing measurements;
the new 100k-record paired run is a software regression. No full matrix was
re-executed during this refactor. No training worker is launched.

## Sacramento and environment

The authenticated control connection was supplied by the user:

```sh
ssh -M -S /tmp/cache602-dynamic-ssh -o ControlPersist=8h qianruw@sacramento.ece.local.cmu.edu
```

Subsequent commands use that connection. The verified repository is
`/home/qianruw/cache`, with the new isolated worktree
`/home/qianruw/cache_dynamic_inference`. The inspected original simulator is
`/home/qianruw/cache/external/ChampSim`; build archives its base and applies the
preserved source patch to a new directory. It does not reset the active simulator.

```sh
ssh -S /tmp/cache602-dynamic-ssh qianruw@sacramento.ece.local.cmu.edu
cd /home/qianruw/cache_dynamic_inference
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
DYNAMIC_PY=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
DYNAMIC_EXP=formal_NN_training/experiments/602_lstm_stride_dynamic_inference
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" build
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" test
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" pilot
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" status
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" resume
```

The build completed in `cache_dynamic_inference_runs/builds/build_01/simulator`.
All 24 tests passed (11 metrics, 7 runtime, 2 measurement, 4 runner). The new pilot
completed in 2.959 seconds, peak host RSS 9,772 KiB. The last `resume` recognized
all 13 compatible completed references and launched no replacement simulations.
`run` selects the same 13-row frozen matrix as `resume`; `--select state64_mac4`
selects its finite h8 MAC4 row. A failed attempt is moved to the new raw tree's
`failed/` directory before retry, preserving the original output.

Paths in `config.json`:

- Trace: `/home/qianruw/cache/traces/602.gcc_s-734B.champsimtrace.xz`.
- Checkpoint root: `/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points`.
- Each h8/h16 i1m/i20m point uses `seed7/export/model.bin`; its original offline checkpoint remains `seed7/offline/model.pt`.
- New raw outputs: `/home/qianruw/cache_dynamic_inference_runs`.
- Read-only scientific references: `/home/qianruw/cache_online_runs/final`. The legacy directory name identifies original data, not an active training experiment.

## Paired frozen regression

The new pilot skips 20M records and executes 100k with empty caches/histories,
h8-i1m, 64 state entries and 4 MACs/cycle. A separate invocation of the preserved
binary used the identical selected records and frozen settings. It is retained
under `cache_dynamic_inference_runs/pilot/reference_frozen_h8_i1m_state64_mac4`.
The command actually executed was equivalent to the following environment and
binary invocation (the run metadata records the complete command and host time):

```sh
DYNAMIC_REFERENCE=/home/qianruw/cache_dynamic_inference_runs/pilot/reference_frozen_h8_i1m_state64_mac4
ONLINE602_METHOD=frozen ONLINE602_MACS=4 ONLINE602_STATE_CAPACITY=64 \
ONLINE602_OUTPUT_LIMIT=32 ONLINE602_SKIP_RECORDS=20000000 ONLINE602_MAX_RECORDS=100000 \
ONLINE602_STATS="$DYNAMIC_REFERENCE/snapshots.jsonl" \
STRIDE_LSTM_MODEL_BIN=/home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/points/h8/i1m/seed7/export/model.bin \
/usr/bin/time -v -o "$DYNAMIC_REFERENCE/host_time.txt" \
/home/qianruw/cache_online_runs/builds/build_04/simulator/bin/perceptron-no-online602-no-ship-1core \
--l2c_prefetcher_types=online602 --stride_num_trackers=64 --stride_pref_degree=2 \
--warmup_instructions=0 --simulation_instructions=100000 \
-traces /home/qianruw/cache/traces/602.gcc_s-734B.champsimtrace.xz
```

This is the predecessor's frozen mode, not an online training process. Do not
redirect over the completed reference; retain its existing output or choose a
new run directory for any deliberate repeat. Compare the completed logs with:

```sh
"$DYNAMIC_PY" "$DYNAMIC_EXP/validation/compare_frozen_runs.py" \
  --new-run /home/qianruw/cache_dynamic_inference_runs/pilot/development_frozen_h8_i1m_state64_mac4 \
  --reference-run /home/qianruw/cache_dynamic_inference_runs/pilot/reference_frozen_h8_i1m_state64_mac4
```

Both executions retired 100,000 instructions, observed 1,633 eligible L2 LOADs,
completed 408 neural decisions, and used 325,790 cycles (IPC 0.306946). All 30
original parser fields match. Cache/service timelines match; only the documented
removed fields, initialized zeros and smaller event/queue layout differ.

## Safe stop and resume

No long-running jobs remain at delivery. These commands are available for a
future run; stopping was not necessary for this completed pilot:

```sh
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" stop
"$DYNAMIC_PY" "$DYNAMIC_EXP/run.py" resume
```

Stop checks the recorded process command belongs to this experiment's build
root before signaling its process group. It does not target all Python or
ChampSim processes. At most two simulator jobs run concurrently.

## Report generation and local copy

The following local command was successfully used to regenerate the 13 retained
rows, measured figures and the compiled PDF (17 pages after the observation supplement). It uses the available local
plotting environment and Sacramento's existing `pdflatex`, with two TeX passes.
No Colab, Overleaf, installation or cloud execution was used.

```sh
cd /Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement
MPLCONFIGDIR=/tmp/dynamic602-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/Users/angelawoo/Documents/CMU/ece/cache_live_dynamics/formal_NN_training/experiments/602_lstm_stride_live_dynamics/.venv/bin/python \
work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/run.py report \
--run-dir work/raw_final \
--historical-root /Users/angelawoo/Documents/CMU/ece/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7 \
--compile-via qianruw@sacramento.ece.local.cmu.edu --ssh-control /tmp/cache602-dynamic-ssh
```

For Sacramento-only regeneration, the corresponding entry is:

```sh
cd /home/qianruw/cache_dynamic_inference
/home/qianruw/venvs/cache-few-nn-cpu/bin/python \
formal_NN_training/experiments/602_lstm_stride_dynamic_inference/run.py report \
--historical-root /home/qianruw/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7
```

The source/PDF and required figure inputs are copied to a new
`602_stride_dynamic_inference/` subfolder under the user's approved
`cache/docs/602_stride_online_learning` destination, preserving old reports:

```sh
cd /Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement
mkdir -p outputs/602_stride_dynamic_inference
rsync -a --exclude '*.png' \
work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/report/ \
outputs/602_stride_dynamic_inference/
rsync -a outputs/602_stride_dynamic_inference/ \
/Users/angelawoo/Documents/CMU/ece/cache/docs/602_stride_online_learning/602_stride_dynamic_inference/
```

The PDF was rendered with `pdftoppm` and all pages visually inspected.

## Git delivery and old branch retirement

The new branch was created from the existing work, then committed and pushed
without force. Source changes are confined to the experiment replacement.

```sh
git -C work/cache_dynamic_inference add formal_NN_training/experiments/602_lstm_stride_online formal_NN_training/experiments/602_lstm_stride_dynamic_inference
git -C work/cache_dynamic_inference commit -m 'Separate frozen dynamic inference and preserve measured results'
git -C work/cache_dynamic_inference push -u origin experiment/602-stride-dynamic-inference
```

At the user's request, the one existing old branch
`experiment/602-stride-online-learning` is retired locally, on GitHub and on
Sacramento after delivery of the new branch. No separate online-training branch
exists. Old worktree directories and raw artifacts are preserved; the local
`work/cache_online/.git` is the common Git directory backing the new worktree.
Deleting that directory would break the new checkout. Branch retirement does
not rewrite Git history or delete checkpoints.

```sh
git -C work/cache_online switch --detach
git -C work/cache_dynamic_inference branch -d experiment/602-stride-online-learning
git -C work/cache_dynamic_inference push origin --delete experiment/602-stride-online-learning
ssh -S /tmp/cache602-dynamic-ssh qianruw@sacramento.ece.local.cmu.edu \
'git -C /home/qianruw/cache_online switch --detach && git -C /home/qianruw/cache_dynamic_inference branch -d experiment/602-stride-online-learning'
```

## Completed observation/quality supplement

The supplementary entry point has ten fixed cases: historical NoPF, historical
live Stride, four original checkpoint replays, original Stride replay, and h8-i1m
history/latency measurements at the existing zero/4/16-MAC settings. It introduces
no training, inference-window change, reset or threshold. All new raw outputs
are separate under `/home/qianruw/cache_dynamic_inference_runs/observation_study`.
The existing 13 compatible scientific outputs remain unchanged.

Commands used on Sacramento (same authenticated SSH connection):

```sh
cd /home/qianruw/cache_dynamic_inference
/home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_lstm_stride_dynamic_inference/run.py build
/home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_lstm_stride_dynamic_inference/run.py test
/home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measure_observations.py run
/home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measure_observations.py status
```

`measure_observations.py resume` retries failed own cases while preserving their
prior outputs and skips complete cases. The same scoped `run.py stop` check also
recognizes this supplementary raw subtree. The build is isolated as `build_02`.

Local measured analysis commands:

```sh
cd /Users/angelawoo/Documents/Codex/2026-09-08/work-on-angelawoo572-cache-arch-implement
rsync -a -e 'ssh -S /tmp/cache602-dynamic-ssh' qianruw@sacramento.ece.local.cmu.edu:/home/qianruw/cache_dynamic_inference_runs/observation_study/ work/observation_raw/
python3 work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/python/compare_quality.py --run-dir work/raw_final --run-dir work/observation_raw --out-dir work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results
python3 work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/python/history_summary.py --run-dir work/observation_raw --out-dir work/cache_dynamic_inference/formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results
```

The report entry above additionally consumes the quality/history CSVs when
present. It preserves all maximal strict intervals and the full raw denominators
in CSV; cumulative and 100k/1M interval results are distinct. Overleaf requires
the user-approved project link/access; local/server PDF compilation does not.

All ten supplementary runs completed. Thirty original end fields match for each of six reference repetitions and three instrumented NN repetitions (270 equal fields). The current suite has 36 focused tests: 11 metrics, 4 runner, 8 runtime/history, 2 measurement, 10 quality comparisons, and 1 original replay. A final logger flush fix preserves complete bounded records in future runs; the original truncated tail is explicitly excluded from the real excerpts.
