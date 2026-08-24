# 602 offline training-prefix sufficiency: Sacramento commands

These commands operate on the existing branch and completed ignored run
trees. The fairness, regression, aggregation, LaTeX generation, and Overleaf
packaging commands do **not** train a model or launch ChampSim. They require
only Python 3's standard library plus `zip` support; Sacramento does not need
matplotlib, `pdflatex`, or another local TeX installation. Do not set `FORCE=1`
unless a specific rerun has been reviewed and approved.

## 1. Switch to the existing branch

```bash
cd ~/cache
git fetch origin
git switch experiment/602-stride-live-inference
git pull --ff-only origin experiment/602-stride-live-inference
git branch --show-current
```

The last command must print:

```text
experiment/602-stride-live-inference
```

Set absolute paths once. This avoids the empty-variable `/report` failure from
an earlier shell session.

```bash
ROOT_DIR=$(pwd -P)
EXP_REL=formal_NN_training/experiments/602_lstm_stride_live_inference
EXP_DIR="$ROOT_DIR/$EXP_REL"
NEW_RUN_DIR="$EXP_DIR/runs/602_gcc_stride_prefix_seed7"
export ROOT_DIR EXP_REL EXP_DIR NEW_RUN_DIR

test -d "$NEW_RUN_DIR"
printf 'NEW_RUN_DIR=%s\n' "$NEW_RUN_DIR"
```

## 2. Locate the old default 20M run

List candidates without changing them:

```bash
find "$ROOT_DIR/formal_NN_training/experiments/602_offline_lstm_stride/runs" \
  -maxdepth 1 -type d \
  -name '602_offline_lstm_stride_compact_hurdle_v9_seed7' -print
```

Set the exact old directory explicitly:

```bash
OLD_RUN_DIR="$ROOT_DIR/formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_compact_hurdle_v9_seed7"
export OLD_RUN_DIR
printf 'OLD_RUN_DIR=%s\nNEW_RUN_DIR=%s\n' "$OLD_RUN_DIR" "$NEW_RUN_DIR"
```

Missing old files are allowed, but the regression validator will then report
`historical-metric regression only` and will not claim exact artifact parity.

## 3. Confirm the completed new sweep

```bash
python3 "$EXP_DIR/validation/validate_training_sweep.py" \
  --run-dir "$NEW_RUN_DIR" --require-all

find "$NEW_RUN_DIR/points" \
  -path '*/offline/replay.log' -type f | sort | wc -l

test -s "$NEW_RUN_DIR/points/h8/i1m/seed7/offline/replay.log"
test -s "$NEW_RUN_DIR/points/h8/i20m/seed7/offline/replay.log"
test -s "$NEW_RUN_DIR/points/h16/i1m/seed7/offline/replay.log"
test -s "$NEW_RUN_DIR/points/h16/i20m/seed7/offline/replay.log"
```

These checks read completed artifacts only.

## 4. Run the machine-readable fairness validator

```bash
python3 "$EXP_DIR/validation/validate_offline_fairness.py" \
  --run-dir "$NEW_RUN_DIR" \
  --keyed-replayer-binary \
    "$ROOT_DIR/external/ChampSim/bin/champsim.602_offline_replay" \
  --require-all

jq '{status, summary, source_identities, shared_artifacts}' \
  "$NEW_RUN_DIR/report/offline_fairness_report.json"

jq -r '.points[] as $point | $point.checks[] |
  select(.status == "FAIL" or (.status == "UNAVAILABLE" and .required)) |
  "h\($point.hidden_size) \($point.budget_tag) \(.status) \(.name)"' \
  "$NEW_RUN_DIR/report/offline_fairness_report.json"
```

The final `jq` command prints nothing on a complete PASS. The validator also
prints each distinct failed/unavailable check name and its point count, so one
shared counter-semantics problem is not mistaken for many scientific failures.

Generated analysis artifacts:

```text
report/offline_fairness_report.json
report/offline_fairness_report.csv
report/offline_fairness_report.tex
```

The JSON contains a check row for every configured h8/h16 budget. It verifies
that prefix-local class weights equal
`decision_rows / (2 * prefix_label_frequency)`.

## 5. Run the old-versus-new i20m regression validator

The command accepts explicit old/new paths as required:

```bash
python3 "$EXP_DIR/validation/validate_20m_regression.py" \
  --old-run-dir "$OLD_RUN_DIR" \
  --new-run-dir "$NEW_RUN_DIR"

jq '{status, mode, exact_artifact_parity_status, failures,
     h8: .points.h8.historical_ipc_anchor,
     h16: .points.h16.historical_ipc_anchor}' \
  "$NEW_RUN_DIR/report/regression_20m_report.json"
```

Expected historical anchors are approximately h8 IPC 0.40924 and h16 IPC
0.40994. The validator does not pass exact artifact parity from those IPC
numbers alone.

Generated outputs:

```text
report/regression_20m_report.json
report/regression_20m_report.csv
report/regression_20m_report.tex
```

## 6. Aggregate the completed offline sweep

```bash
python3 "$EXP_DIR/python/summarize_training_prefixes.py" \
  --prefix-root "$NEW_RUN_DIR/training_prefixes" \
  --out-dir "$NEW_RUN_DIR"

python3 "$EXP_DIR/python/aggregate_offline_results.py" \
  --run-dir "$NEW_RUN_DIR"
```

No live aggregate is needed for the primary offline conclusion.

## 7. Generate the offline-primary analysis and LaTeX figures

```bash
python3 "$EXP_DIR/python/compare_offline_live.py" \
  --run-dir "$NEW_RUN_DIR"

python3 "$EXP_DIR/python/generate_overleaf_figures.py" \
  --run-dir "$NEW_RUN_DIR"
```

Inspect the same-hidden-size conclusion:

```bash
jq '{scope, comparison_rule,
     h8: .conclusions.h8.offline_stable_plateau,
     h16: .conclusions.h16.offline_stable_plateau}' \
  "$NEW_RUN_DIR/offline_sufficiency.json"

column -s, -t < "$NEW_RUN_DIR/offline_sufficiency.csv" | less -S

find "$NEW_RUN_DIR/report" -maxdepth 1 \
  -type f -name 'generated_figure*.tex' -print | sort
```

The three required primary figure sources are:

```text
generated_figure1_offline_ipc.tex
generated_figure2_same_h_differences.tex
generated_figure3_training_supervision.tex
```

They contain measured PGFPlots coordinates and are rendered by Overleaf. The
existing older PNG plots in `plots/` or `report_plots/` are not read, deleted,
or overwritten.

## 8. Package the redesigned LaTeX for Overleaf

Do not run `plot_results.py`, install matplotlib, or run `pdflatex` on
Sacramento. Package only `main.tex` plus generated `.tex` inputs:

```bash
RUN_DIR="$NEW_RUN_DIR" FORCE=1 \
  bash "$EXP_DIR/linux/package_overleaf_report.sh"

ls -lh "$NEW_RUN_DIR/602_stride_offline_sufficiency_overleaf.zip"
unzip -l "$NEW_RUN_DIR/602_stride_offline_sufficiency_overleaf.zip"
```

The ZIP must contain 11 files: `main.tex` and ten `report/*.tex` files. It must
not contain PNG/PDF files, raw JSON/CSV, logs, streams, replay lists,
checkpoints, or binaries. Upload the ZIP as a new Overleaf project and keep
`main.tex` as the main document. Overleaf supplies PGFPlots and builds the PDF.

## 9. Run the four primary functional-live validation points

Only run this after the offline report is complete and the four frozen exports
exist. This is a real ChampSim stage, unlike steps 3-8.

```bash
for H in 8 16; do
  for B in i1m i20m; do
    test -s "$NEW_RUN_DIR/points/h$H/$B/seed7/export/model.bin"
  done
done

RUN_DIR="$NEW_RUN_DIR" \
HIDDEN_SIZES=8,16 BUDGETS=i1m,i20m SEEDS=7 \
STATE_MODE=parity RUN_MODE=live \
WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run
```

These four points answer whether i1m versus i20m remains close under
functional live inference with zero modeled NN inference latency.

## 10. Run the two aggressive live candidates

Do not combine the selectors into a cross product.

```bash
RUN_DIR="$NEW_RUN_DIR" \
HIDDEN_SIZES=8 BUDGETS=i250k SEEDS=7 \
STATE_MODE=parity RUN_MODE=live \
WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run

RUN_DIR="$NEW_RUN_DIR" \
HIDDEN_SIZES=16 BUDGETS=i100k SEEDS=7 \
STATE_MODE=parity RUN_MODE=live \
WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run
```

## 11. Optionally run every valid live point

This is not required for the primary conclusion. Choose exactly one of the
following two alternatives. The full curve uses every measured valid export;
it does not interpolate or smooth missing points.

### Alternative A: one sequential process

```bash
LIVE_ALL_VALID=1 RUN_DIR="$NEW_RUN_DIR" \
STATE_MODE=parity RUN_MODE=live \
WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run
```

### Alternative B: h8 and h16 concurrently

The two processes below select disjoint point directories, so they may run at
the same time. Do not also start Alternative A.

```bash
mkdir -p "$NEW_RUN_DIR/logs"

nohup env LIVE_ALL_VALID=1 HIDDEN_SIZES=8 \
  RUN_DIR="$NEW_RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run \
  >"$NEW_RUN_DIR/logs/live_full_curve_h8.nohup.log" 2>&1 &
LIVE_H8_PID=$!

nohup env LIVE_ALL_VALID=1 HIDDEN_SIZES=16 \
  RUN_DIR="$NEW_RUN_DIR" STATE_MODE=parity RUN_MODE=live \
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000 \
  bash "$EXP_DIR/linux/run_live_sweep.sh" run \
  >"$NEW_RUN_DIR/logs/live_full_curve_h16.nohup.log" 2>&1 &
LIVE_H16_PID=$!

printf 'h8 PID=%s\nh16 PID=%s\n' "$LIVE_H8_PID" "$LIVE_H16_PID"
wait "$LIVE_H8_PID"; H8_STATUS=$?
wait "$LIVE_H16_PID"; H16_STATUS=$?
printf 'h8 status=%s\nh16 status=%s\n' "$H8_STATUS" "$H16_STATUS"
test "$H8_STATUS" -eq 0
test "$H16_STATUS" -eq 0
```

From a second Sacramento login, monitor without changing results:

```bash
tail -n 30 -f \
  "$NEW_RUN_DIR/logs/live_full_curve_h8.nohup.log" \
  "$NEW_RUN_DIR/logs/live_full_curve_h16.nohup.log"
```

Existing valid runs are skipped. Never launch two processes for the same
hidden size, never aggregate while either process is still running, and use
`FORCE=1` only for a specifically approved rerun.

## 12. Aggregate live results and regenerate only the secondary appendix

```bash
python3 "$EXP_DIR/python/aggregate_live_results.py" \
  --run-dir "$NEW_RUN_DIR"

python3 "$EXP_DIR/python/compare_offline_live.py" \
  --run-dir "$NEW_RUN_DIR"

python3 "$EXP_DIR/python/generate_overleaf_figures.py" \
  --run-dir "$NEW_RUN_DIR"

RUN_DIR="$NEW_RUN_DIR" FORCE=1 \
  bash "$EXP_DIR/linux/package_overleaf_report.sh"

jq '.points[] | {
  hidden_size, budget_tag,
  same_checkpoint_offline_ipc,
  ipc,
  live_minus_same_checkpoint_offline_ipc,
  same_hidden_size_live_20m_ipc,
  relative_live_ipc_difference_vs_same_h_live_20m
}' "$NEW_RUN_DIR/live_validation_comparisons.json"
```

The live appendix is a generated table, so no additional image dependency is
introduced. Live data remain secondary and cannot alter the offline fairness
conclusion.

## 13. Copy the Overleaf ZIP to the Mac Documents directory

Run from the Mac Terminal, not Sacramento:

```bash
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % mkdir -p ~/Documents/cache_arch_stride_sufficiency
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % scp qianruw@sacramento.ece.local.cmu.edu:~/cache/formal_NN_training/experiments/602_lstm_stride_live_inference/runs/602_gcc_stride_prefix_seed7/602_stride_offline_sufficiency_overleaf.zip ~/Documents/cache_arch_stride_sufficiency/
(base) angelawoo@Angelas-MacBook-Pro-5 ~ % unzip -l ~/Documents/cache_arch_stride_sufficiency/602_stride_offline_sufficiency_overleaf.zip
```

Upload that ZIP to Overleaf. The PDF is intentionally built there, not on
Sacramento.

## 14. Check ignored/generated artifacts

```bash
cd "$ROOT_DIR"
git check-ignore -v \
  "$NEW_RUN_DIR/offline_sufficiency.json" \
  "$NEW_RUN_DIR/report/generated_figure1_offline_ipc.tex" \
  "$NEW_RUN_DIR/602_stride_offline_sufficiency_overleaf.zip"

test -z "$(git ls-files "$EXP_REL/runs")" \
  && echo 'PASS: no generated run artifact is tracked'

git ls-files | grep -E \
  '(\.pt|\.pth|\.ckpt|model\.bin|\.log|events\.csv|replay\.csv|\.tar\.gz)$' \
  && echo 'FAIL: inspect tracked artifacts' \
  || echo 'PASS: no forbidden generated artifact is tracked'
```

## 15. Review source diff and status

```bash
cd "$ROOT_DIR"
git status --short
git diff --check
git diff --stat
git diff -- \
  "$EXP_REL/README.md" \
  "$EXP_REL/COMMANDS.md" \
  "$EXP_REL/python" \
  "$EXP_REL/validation" \
  "$EXP_REL/report" \
  "$EXP_REL/linux/package_overleaf_report.sh"
```

Also run source tests:

```bash
python3 -m unittest discover -s "$EXP_DIR/validation" -p 'test_*.py'
python3 "$EXP_DIR/validation/validate_live_contract.py"
python3 -m compileall -q "$EXP_DIR/python" "$EXP_DIR/validation"
bash -n "$EXP_DIR/linux/package_overleaf_report.sh"
```

## 16. Source-only commit and push to the same branch

Do not use `git add .`.

```bash
cd "$ROOT_DIR"
git add \
  "$EXP_REL/README.md" \
  "$EXP_REL/COMMANDS.md" \
  "$EXP_REL/python/analysis_policy.py" \
  "$EXP_REL/python/offline_sufficiency.py" \
  "$EXP_REL/python/compare_offline_live.py" \
  "$EXP_REL/python/generate_overleaf_figures.py" \
  "$EXP_REL/python/select_live_budgets.py" \
  "$EXP_REL/validation/provenance_utils.py" \
  "$EXP_REL/validation/validate_offline_fairness.py" \
  "$EXP_REL/validation/validate_20m_regression.py" \
  "$EXP_REL/validation/validate_20m_backward_regression.py" \
  "$EXP_REL/validation/validate_live_contract.py" \
  "$EXP_REL/validation/test_analysis_policy.py" \
  "$EXP_REL/validation/test_offline_audits.py" \
  "$EXP_REL/validation/test_offline_fairness_analysis.py" \
  "$EXP_REL/report/602_stride_training_budget_live_inference.tex" \
  "$EXP_REL/report/README.md" \
  "$EXP_REL/linux/package_overleaf_report.sh"

git diff --cached --check
git diff --cached --stat
git status --short
git commit -m "Make 602 prefix sufficiency offline-keyed-replay primary"
git push origin experiment/602-stride-live-inference
```

Do not merge to `main` in this workflow.
