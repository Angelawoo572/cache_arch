# Mac, Sacramento, Colab, and live ChampSim commands

All measurements and generated files stay below the ignored runs directory.
Commands marked “estimate” are planning estimates, not measured results.

## A. Mac login

Start at the Mac prompt:

~~~bash
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % ssh qianruw@sacramento.ece.local.cmu.edu
~~~

## B. Sacramento safety checks

~~~bash
cd ~/cache
git status --short
git branch --show-current
git submodule status
df -h
du -sh external/ChampSim
ls -lh traces/602.gcc_s-734B.champsimtrace.xz
~~~

If git status prints anything, stop and inspect it. Do not automatically stash,
reset, delete, or clean a user working tree.

## C. Switch to the one feature branch

For a first checkout:

~~~bash
cd ~/cache
git fetch origin
git switch --track origin/experiment/602-stride-live-inference
~~~

If the local branch already exists:

~~~bash
cd ~/cache
git switch experiment/602-stride-live-inference
git pull --ff-only
~~~

Verify:

~~~bash
git branch --show-current
git submodule status
~~~

Do not merge main into this workflow and do not work directly on main.

## Shared Sacramento variables

~~~bash
cd ~/cache
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_ID=602_gcc_stride_prefix_seed7
RUN_DIR=$EXP/runs/$RUN_ID
TRACE=traces/602.gcc_s-734B.champsimtrace.xz
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/incoming"
~~~

## D. Pre-run validation

The Sacramento host may use Python 3.6 and a NumPy release older than 1.17.
The host-side workflow deliberately requires no pandas. Record the versions
before validation:

~~~bash
python3 --version
python3 - <<'PY'
import numpy as np
print("NumPy", np.__version__)
print("[PASS] legacy RandomState available:", hasattr(np.random, "RandomState"))
print("[PASS] pandas is not required by Sacramento stages")
PY
~~~

If an older branch revision failed on `subprocess(..., text=True)` or
`numpy.random.default_rng`, pull the current branch and restart here at section
D. Do not begin collection until both validators and synthetic parity pass.

~~~bash
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
python3 "$EXP/validation/validate_live_contract.py"
python3 -m compileall -q formal_NN_training/common "$EXP/python" "$EXP/validation"
find "$EXP" -name '*.sh' -print0 | xargs -0 -n1 bash -n
if command -v shellcheck >/dev/null 2>&1; then
  find "$EXP" -name '*.sh' -print0 | xargs -0 shellcheck
else
  echo "shellcheck not installed; bash -n completed"
fi
RUN_DIR="$RUN_DIR" FORCE=1 bash "$EXP/linux/build_standalone_runtime.sh"
python3 "$EXP/validation/compare_python_cpp_outputs.py" \
  --cpp-runner "$RUN_DIR/bin/stride_lstm_standalone" \
  --work-dir "$RUN_DIR/parity/synthetic" \
  --synthetic \
  --output "$RUN_DIR/parity/synthetic/summary.json"
~~~

Expected synthetic output ends with status PASS and
action_mismatch_count 0.

## E. Exact training-prefix collection

The collector runs warmup 0 and simulation N for each of the 17 budgets. It
does not substitute callback rows for retired instructions. Each resulting
stream is shared by h8 and h16.

Foreground, all budgets:

~~~bash
RUN_DIR="$RUN_DIR" BUDGETS=all \
  bash "$EXP/linux/collect_training_prefixes.sh" collect
~~~

Background, all budgets:

~~~bash
nohup env RUN_DIR="$RUN_DIR" BUDGETS=all \
  bash "$EXP/linux/collect_training_prefixes.sh" collect \
  > "$RUN_DIR/logs/collect_all.nohup.log" 2>&1 &
echo $! > "$RUN_DIR/logs/collect_all.pid"
tail -f "$RUN_DIR/logs/collect_all.nohup.log"
~~~

Status/resume:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/collect_training_prefixes.sh" status
RUN_DIR="$RUN_DIR" BUDGETS=all \
  bash "$EXP/linux/collect_training_prefixes.sh" collect
~~~

One budget, a subset, and an explicit one-budget rerun:

~~~bash
RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i1m \
  bash "$EXP/linux/collect_training_prefixes.sh" collect
RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i100,i250,i500,i1k \
  bash "$EXP/linux/collect_training_prefixes.sh" collect
RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i1m FORCE=1 \
  bash "$EXP/linux/collect_training_prefixes.sh" collect
~~~

Outputs:

~~~text
$RUN_DIR/training_events/602.gcc_s-734B.i*.events.csv.gz
$RUN_DIR/training_prefixes/i*/602.gcc_s-734B.i*.train_stream.csv.gz
$RUN_DIR/training_prefixes/i*/training_manifest.json
$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz
$RUN_DIR/training_prefix_data_summary.csv
$RUN_DIR/training_prefix_data_summary.json
~~~

## F. Inspect supervision

~~~bash
column -s, -t < "$RUN_DIR/training_prefix_data_summary.csv" | less -S
jq '.points[] | {
  budget_tag,instruction_budget,decision_rows,silent_rows,
  positive_count_rows,action_atoms,k_histogram,unique_pcs,
  unique_positive_pcs,status,failure_reason
}' "$RUN_DIR/training_prefix_data_summary.json"
jq . "$RUN_DIR/training_prefixes/i20m/training_manifest.json"
python3 "$EXP/validation/validate_training_sweep.py" --run-dir "$RUN_DIR" --require-all
~~~

A report sentence can be read directly as: X retired instructions produced Y
decision rows, Z positive-count rows, and W action atoms.

## G. Package Colab input

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/package_colab_input.sh"
INPUT_ARCHIVE=$RUN_DIR/602_stride_live_colab_input.tar.gz
tar -tzf "$INPUT_ARCHIVE" | less
python3 "$EXP/validation/validate_training_sweep.py" --run-dir "$RUN_DIR" --require-all
~~~

## H. Sacramento to Mac Documents

Run these on the Mac:

~~~bash
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % mkdir -p ~/Documents/cache_arch_stride_live
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % scp qianruw@sacramento.ece.local.cmu.edu:~/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/602_stride_live_colab_input.tar.gz ~/Documents/cache_arch_stride_live/
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % cd ~/Documents/cache_arch_stride_live
(base) angelawoo@Angelas-MacBook-Pro-5 cache_arch_stride_live % tar -tzf 602_stride_live_colab_input.tar.gz | less
~~~

## I. Colab branch checkout

The notebook performs this exact fresh-clone operation:

~~~bash
git clone \
  --branch experiment/602-stride-live-inference \
  --single-branch \
  https://github.com/Angelawoo572/cache_arch.git \
  /content/cache_arch
cd /content/cache_arch
git branch --show-current
~~~

For an existing Colab clone:

~~~bash
cd /content/cache_arch
git fetch origin
git switch experiment/602-stride-live-inference
git pull --ff-only
git branch --show-current
~~~

Open colab/train_prefix_sweep.ipynb, restart the runtime, select a GPU, and run
all cells from top to bottom. Do not rely on variables from an earlier runtime.

## J. Colab training and export

Upload this single file:

~~~text
602_stride_live_colab_input.tar.gz
~~~

The notebook then performs these reproducible stages:

1. validates the archive structure, all 17 prefix streams, and evaluation stream;
2. extracts into /content/stride_run;
3. runs h16 × i20m first;
4. runs h8 × i20m second;
5. resumes into the complete h8/h16 seed-7 matrix;
6. records tiny untrainable points without future oversampling;
7. exports every trained checkpoint to model.bin/model_metadata.json;
8. packages points into one output archive.

Equivalent training commands inside Colab are:

~~~bash
EXP=/content/cache_arch/formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_DIR=/content/stride_run
EVAL=$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz

python3 "$EXP/python/run_training_prefix_sweep.py" \
  --run-dir "$RUN_DIR" --evaluation-stream "$EVAL" \
  --hidden-sizes 16 --budgets i20m --seeds 7 --device auto --resume

python3 "$EXP/python/run_training_prefix_sweep.py" \
  --run-dir "$RUN_DIR" --evaluation-stream "$EVAL" \
  --hidden-sizes 8 --budgets i20m --seeds 7 --device auto --resume

python3 "$EXP/python/run_training_prefix_sweep.py" \
  --run-dir "$RUN_DIR" --evaluation-stream "$EVAL" \
  --hidden-sizes 8,16 --budgets all --seeds 7 --device auto --resume
~~~

One explicit export example:

~~~bash
POINT=$RUN_DIR/points/h16/i20m/seed7
python3 "$EXP/python/export_live_model.py" \
  --checkpoint "$POINT/offline/model.pt" \
  --run-metadata "$POINT/offline/run_metadata.json" \
  --out-dir "$POINT/export" \
  --hidden-size 16 --instruction-budget 20000000 --seed 7
~~~

Expected downloaded files:

~~~text
602_stride_live_colab_output.tar.gz
~~~

## K. Colab output to Mac

~~~bash
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % cd ~/Documents/cache_arch_stride_live
(base) angelawoo@Angelas-MacBook-Pro-5 cache_arch_stride_live % tar -tzf 602_stride_live_colab_output.tar.gz | less
~~~

## L. Mac to Sacramento

~~~bash
(base) angelawoo@Angelas-MacBook-Pro-5 cache_arch_stride_live % scp 602_stride_live_colab_output.tar.gz qianruw@sacramento.ece.local.cmu.edu:~/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/incoming/
~~~

## M. Install and validate Colab output

Back on Sacramento:

~~~bash
cd ~/cache
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_DIR=$EXP/runs/602_gcc_stride_prefix_seed7
ARCHIVE=$RUN_DIR/incoming/602_stride_live_colab_output.tar.gz \
RUN_DIR="$RUN_DIR" \
  bash "$EXP/linux/install_colab_output.sh" install

RUN_DIR="$RUN_DIR" bash "$EXP/linux/install_colab_output.sh" status
jq . "$RUN_DIR/points/h16/i20m/seed7/point_metadata.json"
jq . "$RUN_DIR/points/h8/i20m/seed7/point_metadata.json"
find "$RUN_DIR/points" -path '*/offline/model.pt' -print | sort
python3 "$EXP/validation/validate_training_sweep.py" --run-dir "$RUN_DIR" --require-all
~~~

The installer requires h16=5,220 and h8=1,908 parameters and both 20M
checkpoint anchors. It does not claim their IPC until Sacramento replay runs.

## N. Offline keyed-replay sweep

Foreground all valid points:

~~~bash
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8,16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run
~~~

Background and monitoring:

~~~bash
nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8,16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run \
  > "$RUN_DIR/logs/offline_sweep.nohup.log" 2>&1 &
echo $! > "$RUN_DIR/logs/offline_sweep.pid"
tail -f "$RUN_DIR/logs/offline_sweep.nohup.log"
~~~

Resume, one-point rerun, h8-only, h16-only, and all valid:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/run_offline_sweep.sh" run
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=i1m FORCE=1 \
  bash "$EXP/linux/run_offline_sweep.sh" run
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8,16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run
RUN_DIR="$RUN_DIR" bash "$EXP/linux/run_offline_sweep.sh" status
~~~

Offline replay is the learning-curve/reference phase. It is not proof that
live callback inference works.

## O. h16-first torch-free export validation and standalone parity

All checkpoints and `model.bin` files are exported in Colab. Sacramento does
not re-import PyTorch checkpoints. Validate the complete h16 export first:

~~~bash
POINT=$RUN_DIR/points/h16/i20m/seed7
python3 "$EXP/python/validate_export.py" \
  --model-bin "$POINT/export/model.bin" \
  --metadata "$POINT/export/model_metadata.json"
~~~

Then h8:

~~~bash
POINT=$RUN_DIR/points/h8/i20m/seed7
python3 "$EXP/python/validate_export.py" \
  --model-bin "$POINT/export/model.bin" \
  --metadata "$POINT/export/model_metadata.json"
~~~

Build and synthetic parity:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/build_standalone_runtime.sh"
python3 "$EXP/validation/compare_python_cpp_outputs.py" \
  --cpp-runner "$RUN_DIR/bin/stride_lstm_standalone" \
  --work-dir "$RUN_DIR/parity/synthetic" \
  --synthetic \
  --output "$RUN_DIR/parity/synthetic/summary.json"
~~~

Full recorded-stream parity, h16 then h8:

~~~bash
for H in 16 8; do
  POINT=$RUN_DIR/points/h$H/i20m/seed7
  python3 "$EXP/validation/compare_python_cpp_outputs.py" \
    --cpp-runner "$RUN_DIR/bin/stride_lstm_standalone" \
    --work-dir "$POINT/parity/recorded" \
    --model-bin "$POINT/export/model.bin" \
    --stream "$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz" \
    --point-metadata "$POINT/point_metadata.json" \
    --output "$POINT/parity/recorded/summary.json"
done
~~~

Expected PASS fields are exact action mismatch count 0, mismatch rate 0, and
first mismatch null. If it fails:

~~~bash
jq . "$POINT/parity/recorded/summary.json"
head -1 "$POINT/parity/recorded/cpp_outputs.jsonl"
~~~

On a machine that has PyTorch, exact model.pt-to-model.bin validation and an
optional old-checkpoint action comparison remain available:

~~~bash
python3 "$EXP/python/validate_export.py" \
  --model-bin "$POINT/export/model.bin" \
  --metadata "$POINT/export/model_metadata.json" \
  --checkpoint "$POINT/offline/model.pt" \
  --reference-checkpoint /absolute/path/to/old/model.pt \
  --evaluation-stream "$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz" \
  --candidate-run-metadata "$POINT/offline/run_metadata.json" \
  --reference-run-metadata /absolute/path/to/old/run_metadata.json \
  --metric-tolerances "$EXP/config/regression_tolerances.json"
~~~

## P. Install and build ChampSim live prefetcher

~~~bash
git submodule status
CHAMP_DIR=external/ChampSim \
  bash "$EXP/runtime/champsim/install_live_prefetcher.sh" status
CHAMP_DIR=external/ChampSim \
  bash "$EXP/runtime/champsim/install_live_prefetcher.sh" install
RUN_DIR="$RUN_DIR" bash "$EXP/linux/build_live_champsim.sh"
ls -lh "$RUN_DIR/bin/champsim.602_stride_lstm_live"
~~~

Rebuild explicitly:

~~~bash
RUN_DIR="$RUN_DIR" FORCE=1 bash "$EXP/linux/build_live_champsim.sh"
~~~

After installation/build:

~~~bash
git submodule status
git -C external/ChampSim status --short
CHAMP_DIR=external/ChampSim \
  bash "$EXP/runtime/champsim/install_live_prefetcher.sh" status
~~~

The listed live files must be exactly the installer-owned paths. Any
unexpected tracked modification must be inspected. Restore only the named
installed sources, without deleting binaries:

~~~bash
CHAMP_DIR=external/ChampSim \
  bash "$EXP/runtime/champsim/install_live_prefetcher.sh" restore
~~~

## Q. Live binary audit

The recommended workflow proceeds from full recorded-stream parity directly
to selected full live runs; no smoke simulation is required. Confirm that no
learning/teacher implementation is linked:

~~~bash
if strings "$RUN_DIR/bin/champsim.602_stride_lstm_live" | \
  grep -Ei 'torch|libtorch|onnx|tensorflow|optimizer|backprop|PFETCH_LIST_PATH|replay.csv'; then
  echo "FAIL: inspect forbidden linkage/string"
else
  echo "PASS: no learning runtime or action-list linkage"
fi
~~~

## R. Recommend, approve, and run selected live points

Generate recommendations only:

~~~bash
python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/select_live_budgets.py" \
  --offline-results "$RUN_DIR/offline_sweep_results.json" \
  --output "$RUN_DIR/selected_live_budgets.json" \
  --selection-config "$EXP/config/live_selection.json"
jq . "$RUN_DIR/selected_live_budgets.json"
jq . "$EXP/config/live_selection.json"
~~~

Review reasons, edit the config if needed, then set approved to true. The
selection script does not launch simulations.

Foreground and background selected sweep:

~~~bash
RUN_DIR="$RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run

nohup env RUN_DIR="$RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_selected.nohup.log" 2>&1 &
echo $! > "$RUN_DIR/logs/live_selected.pid"
tail -f "$RUN_DIR/logs/live_selected.nohup.log"
~~~

Status/resume and one-point rerun:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/run_live_sweep.sh" status
RUN_DIR="$RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run
RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=i1m SEEDS=7 \
STATE_MODE=parity RUN_MODE=live FORCE=1 \
  bash "$EXP/linux/run_live_sweep.sh" run
~~~

Expensive, explicit all-valid override:

~~~bash
LIVE_ALL_VALID=1 RUN_DIR="$RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run
~~~

Do not set LIVE_ALL_VALID=1 for the default seed-7 exploration.

## S. Aggregate, plot, and package the Overleaf report

~~~bash
python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/aggregate_live_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/compare_offline_live.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/plot_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/validation/validate_training_sweep.py" --run-dir "$RUN_DIR" --require-all
jq . "$RUN_DIR/report/conclusions.json"
column -s, -t < "$RUN_DIR/pareto_frontier.csv"
~~~

Sacramento does not need to compile LaTeX. Package the measured conclusions
and all 14 plots as a self-contained Overleaf project:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/package_overleaf_report.sh"
ls -lh "$RUN_DIR/602_stride_live_overleaf.zip"
unzip -l "$RUN_DIR/602_stride_live_overleaf.zip"
find "$RUN_DIR" -maxdepth 2 -type f | sort
~~~

The archive contains only `main.tex`, `report/generated_conclusions.tex`, and
the 14 PNG plots. It excludes checkpoints, logs, event streams, replay lists,
and Python environments. If a previously inspected archive must be replaced,
rerun the packaging command with `FORCE=1`.

Expected aggregate artifacts:

~~~text
training_prefix_data_summary.csv/json
offline_sweep_results.csv/json
live_sweep_results.csv/json
offline_vs_live.csv/json
pareto_frontier.csv/json
selected_live_budgets.json
plots/01_*.png through plots/14_*.png
report/conclusions.json
report/generated_conclusions.tex
602_stride_live_overleaf.zip
~~~

## T. Copy the Overleaf project to Mac

Run on the Mac:

~~~bash
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % mkdir -p ~/Documents/cache_arch_stride_live
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % scp qianruw@sacramento.ece.local.cmu.edu:~/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/602_stride_live_overleaf.zip ~/Documents/cache_arch_stride_live/
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % unzip -l ~/Documents/cache_arch_stride_live/602_stride_live_overleaf.zip
~~~

In Overleaf choose **New Project -> Upload Project**, upload the ZIP, and keep
`main.tex` as the main document. No absolute Sacramento path is embedded.

## U. Final repository checks

~~~bash
cd ~/cache
git status --short
git diff --check
git diff --stat
git diff
git ls-files "$EXP/runs"
git ls-files | grep -E '(\.pt|\.pth|\.ckpt|model\.bin|\.log|events\.csv|replay\.csv|\.tar\.gz)$' && echo "FAIL: inspect tracked artifacts" || echo "PASS: no forbidden artifacts"
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
python3 "$EXP/validation/validate_live_contract.py"
git submodule status
git -C external/ChampSim status --short
~~~

Generated runs may make git status quiet because they are ignored. The
installer's untracked submodule files are expected only while installed;
inspect their paths or run the named restore command before source commits.

## V. Commit and push source only

Do not use git add dot. Add only these source/doc paths:

~~~bash
git add \
  .gitignore \
  formal_NN_training/README.md \
  formal_NN_training/common/stride_direct_action_model.py \
  formal_NN_training/experiments/602_lstm_stride_live_inference \
  formal_NN_training/experiments/602_offline_lstm_stride/python/train_and_offline_infer.py

git status --short
git diff --cached --check
git diff --cached --stat
git commit -m "Complete 602 frozen live inference workflow"
git push -u origin experiment/602-stride-live-inference
~~~

Do not merge main in this task.

## W. Only after a future PR merge

These are future cleanup commands, not part of the current experiment:

~~~bash
git switch main
git pull --ff-only
git branch -d experiment/602-stride-live-inference
~~~

Never delete the feature branch before its results and PR are accepted.

## Complete nohup command index

Run this setup once in the Sacramento shell. Do not start a foreground and a
nohup copy of the same stage at the same time.

~~~bash
cd ~/cache
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_ID=602_gcc_stride_prefix_seed7
RUN_DIR=$EXP/runs/$RUN_ID
mkdir -p "$RUN_DIR/logs"
export EXP RUN_DIR
~~~

Collect every prefix and the fixed evaluation stream:

~~~bash
nohup env RUN_DIR="$RUN_DIR" BUDGETS=all \
  bash "$EXP/linux/collect_training_prefixes.sh" collect \
  > "$RUN_DIR/logs/collect_all.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/collect_all.pid"
tail -f "$RUN_DIR/logs/collect_all.nohup.log"
~~~

Collection subset, one point, or explicit rerun:

~~~bash
nohup env RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i100,i250,i500,i1k \
  bash "$EXP/linux/collect_training_prefixes.sh" collect \
  > "$RUN_DIR/logs/collect_tiny.nohup.log" 2>&1 < /dev/null &

nohup env RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i1m \
  bash "$EXP/linux/collect_training_prefixes.sh" collect \
  > "$RUN_DIR/logs/collect_i1m.nohup.log" 2>&1 < /dev/null &

nohup env RUN_DIR="$RUN_DIR" BUILD=0 BUDGETS=i1m FORCE=1 \
  bash "$EXP/linux/collect_training_prefixes.sh" collect \
  > "$RUN_DIR/logs/collect_i1m_force.nohup.log" 2>&1 < /dev/null &
~~~

Training is run by restart-and-run-all in the Colab notebook; it is not a
Sacramento nohup stage. After installing the single Colab output archive, run
the offline replay sweep:

~~~bash
nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8,16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run \
  > "$RUN_DIR/logs/offline_all.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/offline_all.pid"
tail -f "$RUN_DIR/logs/offline_all.nohup.log"
~~~

Offline h8-only, h16-only, or one forced point:

~~~bash
nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run \
  > "$RUN_DIR/logs/offline_h8.nohup.log" 2>&1 < /dev/null &

nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=16 BUDGETS=all \
  bash "$EXP/linux/run_offline_sweep.sh" run \
  > "$RUN_DIR/logs/offline_h16.nohup.log" 2>&1 < /dev/null &

nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=i1m FORCE=1 \
  bash "$EXP/linux/run_offline_sweep.sh" run \
  > "$RUN_DIR/logs/offline_h8_i1m_force.nohup.log" 2>&1 < /dev/null &
~~~

h16 anchor export validation/full parity/build, then h8 through the same
torch-free validation path. These commands do not launch smoke simulations:

~~~bash
nohup env RUN_DIR="$RUN_DIR" \
  bash "$EXP/linux/launch_stage.sh" stage0 \
  > "$RUN_DIR/logs/stage0_h16.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/stage0_h16.pid"
tail -f "$RUN_DIR/logs/stage0_h16.nohup.log"

nohup env RUN_DIR="$RUN_DIR" \
  bash "$EXP/linux/launch_stage.sh" stage1 \
  > "$RUN_DIR/logs/stage1_h8.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/stage1_h8.pid"
tail -f "$RUN_DIR/logs/stage1_h8.nohup.log"
~~~

Run the actual full h16 20M live reference first, then the actual full h8 20M
live reference. These are measured runs, not reduced-instruction tests:

~~~bash
nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=16 BUDGETS=i20m SEEDS=7 \
  STATE_MODE=parity RUN_MODE=live \
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_h16_i20m.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/live_h16_i20m.pid"
tail -f "$RUN_DIR/logs/live_h16_i20m.nohup.log"

nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=i20m SEEDS=7 \
  STATE_MODE=parity RUN_MODE=live \
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_h8_i20m.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/live_h8_i20m.pid"
tail -f "$RUN_DIR/logs/live_h8_i20m.nohup.log"
~~~

Standalone live ChampSim build by itself:

~~~bash
nohup env RUN_DIR="$RUN_DIR" \
  bash "$EXP/linux/build_live_champsim.sh" \
  > "$RUN_DIR/logs/build_live.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/build_live.pid"
tail -f "$RUN_DIR/logs/build_live.nohup.log"
~~~

Recommended selected live sweep:

~~~bash
nohup env RUN_DIR="$RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_selected.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/live_selected.pid"
tail -f "$RUN_DIR/logs/live_selected.nohup.log"
~~~

One forced live point and the expensive all-valid override:

~~~bash
nohup env RUN_DIR="$RUN_DIR" HIDDEN_SIZES=8 BUDGETS=i1m SEEDS=7 \
  STATE_MODE=parity RUN_MODE=live FORCE=1 \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_h8_i1m_force.nohup.log" 2>&1 < /dev/null &

nohup env LIVE_ALL_VALID=1 RUN_DIR="$RUN_DIR" \
  STATE_MODE=parity RUN_MODE=live \
  bash "$EXP/linux/run_live_sweep.sh" run \
  > "$RUN_DIR/logs/live_all_valid.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/live_all_valid.pid"
tail -f "$RUN_DIR/logs/live_all_valid.nohup.log"
~~~

Final aggregation, plots, and Overleaf packaging:

~~~bash
nohup bash -c '
set -e
cd "$HOME/cache"
python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/aggregate_live_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/compare_offline_live.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/plot_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/validation/validate_training_sweep.py" --run-dir "$RUN_DIR" --require-all
RUN_DIR="$RUN_DIR" bash "$EXP/linux/package_overleaf_report.sh"
' > "$RUN_DIR/logs/analysis_report.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/logs/analysis_report.pid"
tail -f "$RUN_DIR/logs/analysis_report.nohup.log"
~~~

Status commands are safe to run while detached jobs are active:

~~~bash
RUN_DIR="$RUN_DIR" bash "$EXP/linux/collect_training_prefixes.sh" status
RUN_DIR="$RUN_DIR" bash "$EXP/linux/install_colab_output.sh" status
RUN_DIR="$RUN_DIR" bash "$EXP/linux/run_offline_sweep.sh" status
RUN_DIR="$RUN_DIR" bash "$EXP/linux/run_live_sweep.sh" status
jobs -l
~~~

## Runtime and storage estimates

These are estimates only; actual Sacramento throughput, callback density,
Colab GPU assignment, filesystem compression, and selected-point count can
change them.

| Stage | Where | Estimate | Generated/track status |
| --- | --- | --- | --- |
| 17 prefix collections + fixed evaluation | Sacramento | 0.5–3 h; about 89M total simulated retired instructions | ignored event/stream files |
| 34 seed-7 training/evaluation points | Colab GPU | 1–6 h | ignored checkpoints, lists, metadata |
| All valid keyed offline replays + 3 references | Sacramento | 10–40 h if most of 34 train | ignored logs/events |
| Synthetic standalone parity | Sacramento or Mac/Linux | under 1 min | ignored fixtures |
| Full held-out parity per anchor | Sacramento/Colab CPU | 2–20 min, callback-count dependent | ignored JSONL |
| Selected full live sweep | Sacramento | roughly 0.5–2 h per selected point; commonly 4–20 h total | ignored logs |
| Plot/report inspection | Sacramento or Mac with Python/TeX | 1–10 min | ignored plots/PDF |

Estimated disk:

- compressed prefix/event/evaluation streams: roughly 1–10 GB total;
- 34 PyTorch checkpoints: roughly 5–30 MB total (weights are tiny; container
  overhead dominates);
- replay lists and per-run event logs: roughly 1–20 GB, workload dependent;
- plain simulator text logs: roughly 0.1–2 GB;
- Colab output archive: usually tens to hundreds of MB if replay lists are
  included;
- exact model weight bytes: h8 7,632; h16 20,880;
- raw recurrent tensors: h8 64 bytes per observed PC; h16 128 bytes per
  observed PC; peak raw state is 64×U or 128×U bytes for U live PCs, excluding
  hash-map allocator overhead.

Collection, keyed replay, ChampSim build, and full live runs are
Sacramento tasks. Training needs Colab/GPU. Archive verification, SCP, JSON/CSV
inspection, and final PDF viewing can be done on the Mac. All run outputs are
ignored; source, configs, notebook, validators, TeX, README, and COMMANDS are
tracked.
