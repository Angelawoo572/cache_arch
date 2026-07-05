#!/usr/bin/env bash
# Canonical v4 ceiling-replay entrypoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT/formal_NN_training/scripts/20_run_oracle_ceiling_replay.sh" "$@"
