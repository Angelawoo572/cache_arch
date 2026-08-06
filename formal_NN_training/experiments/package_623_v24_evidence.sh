#!/usr/bin/env bash
# Validate and package one completed 623 v24 track without checksum sidecars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACK="${1:-}"

case "$TRACK" in
  stride)
    EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_stride"
    ;;
  spp)
    EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
    ;;
  *)
    echo "usage: $0 stride|spp" >&2
    exit 2
    ;;
esac

CONTRACT="$EXP/python/model_contract.py"
RUN_ID="$(python3 "$CONTRACT" --field run_id)"
RUN_DIR="$EXP/runs/$RUN_ID"
TAGS_CSV="$(python3 "$CONTRACT" --tags-csv)"
ARCHIVE="$RUN_DIR/$RUN_ID.evidence.tar.gz"
IFS=',' read -r -a TAGS <<< "$TAGS_CSV"

test -s "$RUN_DIR/matched_comparison.json"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
if [[ "$TRACK" == spp ]]; then
  python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
fi

files=(
  matched_comparison.json
  matched_comparison.csv
  insight_summary.csv
  model_diagnosis.json
  model_diagnosis.csv
  replay.nohup.log
  colab_output/sweep_manifest.json
  colab_output/validated_collection_manifest.json
)
for tag in "${TAGS[@]}"; do
  files+=(
    "colab_output/$tag/run_metadata.json"
    "colab_output/$tag/training_history.csv"
    "colab_output/$tag/trainer.stdout_stderr.log"
  )
done
for name in "${files[@]}"; do
  test -s "$RUN_DIR/$name"
done

temporary="$(mktemp "$RUN_DIR/.evidence.XXXXXX.tar.gz")"
tar -C "$RUN_DIR" -czf "$temporary" "${files[@]}"
mv -f "$temporary" "$ARCHIVE"
gzip -t "$ARCHIVE"
echo "[PASS] $ARCHIVE"
