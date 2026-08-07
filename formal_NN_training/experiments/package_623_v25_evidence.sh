#!/usr/bin/env bash
# Validate and package both completed 623 v25 tracks without checksum sidecars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

package_track() {
  local track="$1"
  local exp="$ROOT/formal_NN_training/experiments/623_offline_lstm_$track"
  local contract="$exp/python/model_contract.py"
  local run_id tags_csv run_dir archive temporary tag name
  local -a tags files

  [[ -f "$contract" ]] || {
    echo "[error] missing model contract for $track: $contract" >&2
    exit 2
  }
  run_id="$(python3 "$contract" --field run_id)"
  tags_csv="$(python3 "$contract" --tags-csv)"
  [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "[error] unsafe $track run_id: $run_id" >&2
    exit 2
  }
  IFS=',' read -r -a tags <<< "$tags_csv"
  [[ "${#tags[@]}" -gt 0 ]] || {
    echo "[error] $track model_contract returned no tags" >&2
    exit 2
  }
  for tag in "${tags[@]}"; do
    [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
      echo "[error] unsafe $track model tag: $tag" >&2
      exit 2
    }
  done

  run_dir="$exp/runs/$run_id"
  archive="$run_dir/$run_id.evidence.tar.gz"

  if [[ "$track" == spp ]]; then
    python3 "$exp/python/check_matched_comparison.py" --run-dir "$run_dir"
  fi

  python3 - "$run_dir/matched_comparison.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("missing {}".format(path))
payload = json.loads(path.read_text())
if payload.get("status") != "PASS":
    raise SystemExit("{} root status is not PASS".format(path))
if payload.get("failures") != []:
    raise SystemExit("{} root failures are not empty".format(path))
if payload.get("fair_comparison_claim_allowed") is not True:
    raise SystemExit("{} does not allow a fair-comparison claim".format(path))
print("[PASS] root matched comparison {}".format(path))
PY

  python3 "$exp/python/diagnose_completed_run.py" --run-dir "$run_dir"

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
  for tag in "${tags[@]}"; do
    files+=(
      "colab_output/$tag/run_metadata.json"
      "colab_output/$tag/training_history.csv"
      "colab_output/$tag/trainer.stdout_stderr.log"
    )
  done
  for name in "${files[@]}"; do
    [[ -s "$run_dir/$name" ]] || {
      echo "[error] missing evidence file $run_dir/$name" >&2
      exit 2
    }
  done

  temporary="$(mktemp "$run_dir/.evidence.XXXXXX.tar.gz")"
  trap 'rm -f -- "$temporary"' RETURN
  tar -czf "$temporary" -C "$run_dir" "${files[@]}"
  gzip -t "$temporary"
  mv -f "$temporary" "$archive"
  trap - RETURN
  echo "[PASS] $track evidence: $archive"
}

package_track stride
package_track spp
