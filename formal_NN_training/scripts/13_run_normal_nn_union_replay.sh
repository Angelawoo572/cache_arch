#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_PREFETCHER="${BASE_PREFETCHER:-}"
TRACES_STR="${TRACES:-}"
[[ -n "$BASE_PREFETCHER" ]] || { echo "BASE_PREFETCHER is required" >&2; exit 2; }
[[ -n "$TRACES_STR" ]] || { echo "TRACES is required" >&2; exit 2; }

case "$BASE_PREFETCHER" in
  spp|spp_dev2) BASE_TYPE="spp_dev2" ;;
  spp_ppf|spp_ppf_dev) BASE_TYPE="spp_ppf_dev" ;;
  no_pref|none|nopref|list_replayer) echo "invalid base" >&2; exit 2 ;;
  *) BASE_TYPE="$BASE_PREFETCHER" ;;
esac

RUN_TAG="${RUN_TAG:-normal_nn_union_${BASE_PREFETCHER}}"
OUT_DIR="${OUT_DIR:-formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}}"
CFG_DIR="$ROOT/formal_NN_training/_cfg/normal_nn_union"
CFG="$CFG_DIR/${RUN_TAG}.${BASE_PREFETCHER}.ini"
mkdir -p "$CFG_DIR" "$OUT_DIR"

{
  echo "l2c_prefetcher_types = $BASE_TYPE"
  echo "l2c_prefetcher_types = list_replayer"
  if [[ "$BASE_TYPE" == "spp_dev2" || "$BASE_TYPE" == "spp_ppf_dev" ]]; then
    echo "spp_dev2_fill_threshold = 90"
    echo "spp_dev2_pf_threshold = 40"
  fi
} > "$CFG"

printf 'RUN_KIND=normal_plus_keyed_nn_union_control\nBASE_PREFETCHER=%s\nBASE_TYPE=%s\nL2_CONFIG=%s\nTRACES=%s\nRUN_TAG=%s\n' "$BASE_PREFETCHER" "$BASE_TYPE" "$CFG" "$TRACES_STR" "$RUN_TAG" > "$OUT_DIR/UNION_RUN_INFO.txt"

RUN_TAG="$RUN_TAG" \
OUT_DIR="$OUT_DIR" \
TRACES="$TRACES_STR" \
L2_REPLAYER_KNOB="--config=$CFG" \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh

for trace in $TRACES_STR; do
  log="$OUT_DIR/logs/${trace}.oracle_replacer.log"
  grep -Fq "adding L2C_PREFETCHER: list_replayer" "$log" || { echo "list replayer missing" >&2; exit 3; }
done

echo "[summary] $OUT_DIR/summary.csv"
