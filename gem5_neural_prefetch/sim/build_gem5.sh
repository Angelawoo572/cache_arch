#!/bin/bash
# Build gem5 with this study's extension directory. Run on Sacramento.
# Usage: build_gem5.sh [jobs] [extra scons arguments]
#        default 12 jobs, at the lowest CPU priority
set -eu
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
jobs=${1:-12}
shift || true
extras="$GEM5_STUDY/sim/np_ext"
args=(build/X86/gem5.opt -j "$jobs" --linker=gold --ignore-style --no-colors)
if [ -f "$extras/SConscript" ]; then
  args+=("EXTRAS=$extras")
fi
cd "$GEM5_ROOT"
exec nice -n 19 scons "${args[@]}" "$@"
