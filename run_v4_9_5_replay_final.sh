#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/cache"

RUN_ID="v4_9_5_one_shot_all_seed7"
RUN_DIR="$PWD/formal_NN_training/artifacts/v4_9_5/runs/$RUN_ID/worker_00_of_01"
OUT_ROOT="$PWD/formal_NN_training/results/prefetch_experiments/${RUN_ID}_v4_9_5"

# Must be the parent containing normal/events.
NORMAL_EVENT_ROOT="$PWD/formal_NN_training/results/prefetch_experiments/normal_full_evidence_20260705"

# Fixed normal matrix summary.
NORMAL_SUMMARY="$PWD/formal_NN_training/results/prefetcher_baselines/summary.csv"

echo "[start] $(date)"
echo "[repo] $PWD"
echo "[RUN_DIR] $RUN_DIR"
echo "[OUT_ROOT] $OUT_ROOT"
echo "[NORMAL_EVENT_ROOT] $NORMAL_EVENT_ROOT"
echo "[NORMAL_SUMMARY] $NORMAL_SUMMARY"
echo "[MAX_JOBS] ${MAX_JOBS:-1}"
echo "[FORCE] ${FORCE:-0}"
echo "[CLEAN_OUT] ${CLEAN_OUT:-0}"

test -d "$RUN_DIR" || { echo "[ERR] missing RUN_DIR: $RUN_DIR" >&2; exit 2; }
test -f "$RUN_DIR/plan/v4_9_combined_replay_plan.csv" || { echo "[ERR] missing replay plan" >&2; exit 2; }
test -f "$RUN_DIR/plan/v4_9_offline_export_acceptance.csv" || { echo "[ERR] missing offline acceptance" >&2; exit 2; }
test -f "$RUN_DIR/v4_9_all_job_metadata.csv" || { echo "[ERR] missing metadata" >&2; exit 2; }
test -f "$RUN_DIR/server/v4_9_combined_replay.sh" || { echo "[ERR] missing generated replay script" >&2; exit 2; }
test -d "$NORMAL_EVENT_ROOT/normal/events" || { echo "[ERR] missing normal events: $NORMAL_EVENT_ROOT/normal/events" >&2; exit 2; }
test -f "$NORMAL_SUMMARY" || { echo "[ERR] missing normal summary: $NORMAL_SUMMARY" >&2; exit 2; }

echo "[normal event count]"
find "$NORMAL_EVENT_ROOT/normal/events" -type f | wc -l

echo "[artifact counts]"
du -sh "$RUN_DIR" || true
echo -n "metadata.json: "
find "$RUN_DIR" -name "metadata.json" | wc -l
echo -n "prefetch lists: "
find "$RUN_DIR" \( -name "prefetch_list*.csv" -o -name "prefetch_list*.csv.gz" \) | wc -l

echo "[compile checks]"
python3 -m py_compile formal_NN_training/scripts/replay/resolve_replay_plan.py
python3 -m py_compile formal_NN_training/scripts/09_parse_standalone_lstm_replay.py
python3 -m py_compile formal_NN_training/scripts/12_analyze_prefetch_event_attribution.py
python3 -m py_compile formal_NN_training/scripts/15_summarize_prefetch_evidence.py
python3 -m py_compile formal_NN_training/scripts/22_resource_summary.py
python3 -m py_compile formal_NN_training/scripts/23_analyze_v4_8_replay.py
bash -n "$RUN_DIR/server/v4_9_combined_replay.sh"
bash -n formal_NN_training/scripts/11_run_prefetch_event_attribution.sh

echo "[tag uniqueness check]"
python3 - <<PY
import csv
from pathlib import Path
from collections import Counter

paths = [
    Path("$RUN_DIR/plan/v4_9_combined_replay_plan.csv"),
    Path("$RUN_DIR/plan/v4_9_offline_export_acceptance.csv"),
    Path("$RUN_DIR/v4_9_all_job_metadata.csv"),
]
for p in paths:
    rows = list(csv.DictReader(p.open()))
    c = Counter(r.get("tag","") for r in rows)
    dup = [k for k,v in c.items() if v > 1 and k]
    unsafe = [r.get("tag","") for r in rows if "|" in r.get("tag","")]
    print(p, "rows", len(rows), "duplicates", len(dup), "pipe_tags", len(unsafe))
    if dup:
        raise SystemExit("duplicate tags remain in %s: %s" % (p, dup[:5]))
    if unsafe:
        raise SystemExit("unsafe pipe tags remain in %s: %s" % (p, unsafe[:5]))
PY

echo "[plan resolver test]"
python3 formal_NN_training/scripts/replay/resolve_replay_plan.py \
  --plan "$RUN_DIR/plan/v4_9_combined_replay_plan.csv" \
  --root "$PWD" \
  --out "$RUN_DIR/plan/replay_plan_entries.preview.tsv"

wc -l "$RUN_DIR/plan/replay_plan_entries.preview.tsv"
head -3 "$RUN_DIR/plan/replay_plan_entries.preview.tsv"

echo "[plan summary]"
python3 - <<PY
import csv
from pathlib import Path
from collections import Counter

run = Path("$RUN_DIR")
for name in ["v4_9_combined_replay_plan.csv", "v4_9_offline_export_acceptance.csv"]:
    p = run / "plan" / name
    rows = list(csv.DictReader(p.open()))
    print("\\n[file]", p)
    print("rows", len(rows))
    if rows and "replay_eligible" in rows[0]:
        print("eligible", sum(str(r.get("replay_eligible","0")) == "1" for r in rows))
    if rows and "trace" in rows[0]:
        print("by_trace", dict(Counter(r.get("trace","") for r in rows)))
    if rows and "job_status" in rows[0]:
        print("job_status", dict(Counter(r.get("job_status","") for r in rows)))
PY

if [[ "${CLEAN_OUT:-0}" == "1" ]]; then
  echo "[clean previous OUT_ROOT]"
  rm -rf "$OUT_ROOT"
else
  echo "[resume mode] keeping OUT_ROOT if present"
fi

echo "[run replay] $(date)"

NORMAL_EVENT_ROOT="$NORMAL_EVENT_ROOT" \
NORMAL_SUMMARY="$NORMAL_SUMMARY" \
MAX_JOBS="${MAX_JOBS:-1}" \
FORCE="${FORCE:-0}" \
WARMUP=25000000 \
SIM=25000000 \
PATCH_LOGGER=1 \
RESET_PATCH=0 \
COLLECT_EVENT_LOGS=1 \
RUN_SAME_BINARY_NO_PREF=1 \
bash "$RUN_DIR/server/v4_9_combined_replay.sh" "$PWD" "$RUN_DIR"

echo "[replay done] $(date)"

echo "[quick ipc grep]"
grep -R "CPU 0 cumulative IPC" "$OUT_ROOT" 2>/dev/null \
  | tee "$RUN_DIR/quick_ipc_grep.txt" \
  | tail -120 || true

echo "[result files]"
find "$OUT_ROOT" -type f \
  \( -name "*.log" -o -name "*summary*.csv" -o -name "*replay*.csv" -o -name "*analysis*.csv" -o -name "quick_ipc_grep.txt" \) \
  | sort \
  | tail -250 || true

echo "[post-run counts]"
echo -n "lstm event gz: "
find "$OUT_ROOT/lstm" -name "*.events.csv.gz" 2>/dev/null | wc -l
echo -n "key=pc_line_occ logs: "
grep -R "key=pc_line_occ" "$OUT_ROOT/lstm" 2>/dev/null | wc -l
echo -n "IPC lines: "
grep -R "CPU 0 cumulative IPC" "$OUT_ROOT/lstm" 2>/dev/null | wc -l

echo "[package full evidence]"
FULL_EVID="$PWD/formal_NN_training/artifacts/v4_9_5/runs/${RUN_ID}_FULL_replay_evidence_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$FULL_EVID" \
  -C "$PWD/formal_NN_training/artifacts/v4_9_5/runs" "$RUN_ID" \
  -C "$PWD/formal_NN_training/results/prefetch_experiments" "${RUN_ID}_v4_9_5"

ls -lh "$FULL_EVID"

echo "[package compact evidence for upload]"
COMPACT_DIR="$PWD/formal_NN_training/artifacts/v4_9_5/runs/${RUN_ID}_compact_upload"
rm -rf "$COMPACT_DIR"
mkdir -p "$COMPACT_DIR"

mkdir -p "$COMPACT_DIR/run_plan"
cp -a "$RUN_DIR/plan" "$COMPACT_DIR/run_plan/"
cp -a "$RUN_DIR/v4_9_all_job_metadata.csv" "$COMPACT_DIR/run_plan/" || true
cp -a "$RUN_DIR/quick_ipc_grep.txt" "$COMPACT_DIR/run_plan/" 2>/dev/null || true

mkdir -p "$COMPACT_DIR/offline_tables"
find "$RUN_DIR/traces" -maxdepth 3 -type f \
  \( -name "*offline*.csv" -o -name "*ablation*.csv" -o -name "*comparison*.csv" -o -name "*metadata*.csv" -o -name "resource_sweep_plan.csv" -o -name "replay_plan.csv" \) \
  -exec cp --parents {} "$COMPACT_DIR/offline_tables/" \; 2>/dev/null || true

mkdir -p "$COMPACT_DIR/replay_results"
for f in \
  "$OUT_ROOT/lstm_summary.csv" \
  "$OUT_ROOT/lstm_winners.csv" \
  "$OUT_ROOT/resource_summary.csv" \
  "$OUT_ROOT/replay_plan.csv"
do
  [ -f "$f" ] && cp -a "$f" "$COMPACT_DIR/replay_results/"
done

[ -d "$OUT_ROOT/analysis" ] && cp -a "$OUT_ROOT/analysis" "$COMPACT_DIR/replay_results/"
[ -d "$OUT_ROOT/v4_9_analysis" ] && cp -a "$OUT_ROOT/v4_9_analysis" "$COMPACT_DIR/replay_results/"
[ -d "$OUT_ROOT/evidence" ] && cp -a "$OUT_ROOT/evidence" "$COMPACT_DIR/replay_results/"

mkdir -p "$COMPACT_DIR/log_tails"
find "$OUT_ROOT" -type f -name "*.log" | sort | while read -r log; do
  safe="$(echo "$log" | sed 's#^'"$OUT_ROOT"'/##; s#[/ ]#__#g')"
  {
    echo "===== $log ====="
    tail -200 "$log" || true
  } > "$COMPACT_DIR/log_tails/${safe}.tail.txt"
done

COMPACT_TAR="$PWD/formal_NN_training/artifacts/v4_9_5/runs/${RUN_ID}_COMPACT_upload_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$COMPACT_TAR" -C "$PWD/formal_NN_training/artifacts/v4_9_5/runs" "${RUN_ID}_compact_upload"
ls -lh "$COMPACT_TAR"

echo "[compact tar] $COMPACT_TAR"
echo "[full tar] $FULL_EVID"
echo "[finish] $(date)"
