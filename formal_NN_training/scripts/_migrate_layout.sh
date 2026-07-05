#!/usr/bin/env bash
# One-shot migration from historical numbered scripts to a role-based layout.
# Run this locally only after reviewing it.
set -euo pipefail

[[ "${1:-}" == "--apply" ]] || { echo "usage: $0 --apply" >&2; exit 2; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "tracked tree is not clean" >&2; exit 2; }

echo "Migration body will be added in the next update."
