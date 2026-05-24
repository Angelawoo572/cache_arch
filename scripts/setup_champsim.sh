#!/usr/bin/env bash
# setup_champsim.sh -- VERSION 2 (fixes vcpkg bootstrap)
#
# What was wrong before:
#   - We did 'git submodule update --init --recursive' but never bootstrapped vcpkg
#     or ran 'vcpkg install'. Modern ChampSim needs fmt, nlohmann-json, CLI11
#     installed via vcpkg before config.sh / make work. Without them, you get
#     'fmt/core.h: No such file or directory' and 'core_inst.inc: No such file'.
#   - The 'cc1plus: error: to generate dependencies you must specify either -M
#     or -MM' lines are a known harmless make warning during parallel dep tracking;
#     ignore them. The real errors are the 'No such file' ones at the end.
#
# This script does:
#   1. Install OS packages (g++, cmake, curl, etc.)
#   2. Clone ChampSim (main fork)
#   3. Bootstrap vcpkg and install fmt + nlohmann-json + CLI11 + lzma + zlib + catch2
#   4. ./config.sh + make
#   5. Also clone Quangmire/ChampSim-ML (for the Colab -> ChampSim prefetch path)
#   6. Download 5 DPC-3 traces
#   7. Smoke test
#
# Run from any empty working directory. Creates ./ChampSim, ./ChampSim-ML, ./traces.

set -euo pipefail

WORKDIR="$(pwd)"
echo "[setup] working directory: $WORKDIR"

# ---------- 1. OS packages ----------
echo "[setup] installing system packages"
if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then SUDO="" ; else SUDO="sudo"; fi
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config zip unzip curl wget \
    git python3 python3-pip xz-utils zstd ca-certificates tar
fi

# ---------- 2. ChampSim (main, current master) ----------
if [ ! -d "$WORKDIR/ChampSim" ]; then
  echo "[setup] cloning ChampSim"
  git clone https://github.com/ChampSim/ChampSim.git
fi
cd "$WORKDIR/ChampSim"
git submodule update --init --recursive

# ---------- 3. Bootstrap + install C++ dependencies via vcpkg ----------
# THIS WAS THE STEP MISSING IN V1.
echo "[setup] bootstrapping vcpkg (one-time, ~1 minute)"
if [ ! -x vcpkg/vcpkg ]; then
  # Primary path: download pre-built binary from GitHub releases.
  if ! ./vcpkg/bootstrap-vcpkg.sh -disableMetrics 2>&1; then
    echo "[warn] vcpkg pre-built download blocked. Trying source build..."
    # If GitHub release CDN is blocked (corporate firewall etc.), build from source.
    pushd vcpkg
    rm -f vcpkg vcpkg.part
    # The bootstrap script also supports building from source by removing pre-built tool.
    VCPKG_USE_SYSTEM_BINARIES=1 ./bootstrap-vcpkg.sh -disableMetrics || {
      echo "[error] vcpkg bootstrap failed both ways."
      echo "[hint] If you're on a restricted network, ask sysadmin to allow:"
      echo "       github.com/microsoft/vcpkg-tool/releases/"
      echo "[hint] Or manually install fmt, nlohmann-json, cli11 system-wide and skip vcpkg."
      exit 4
    }
    popd
  fi
fi

echo "[setup] installing C++ deps via vcpkg (~5--10 minutes the first time)"
# ChampSim's vcpkg.json declares: fmt, nlohmann-json, cli11, catch2, liblzma, zlib.
# `vcpkg install` with no args reads vcpkg.json automatically (manifest mode).
./vcpkg/vcpkg install --x-manifest-root=. --x-install-root=./vcpkg_installed

# ---------- 4. configure + build ----------
echo "[setup] configure + make (this takes a few minutes)"
./config.sh champsim_config.json
# -j8 because ChampSim's Makefile occasionally races at -j=ncores;
# the harmless 'cc1plus: error: to generate dependencies' lines are still expected.
make -j8

if [ ! -x bin/champsim ]; then
  echo "[ERROR] ChampSim build failed -- bin/champsim missing. See log above."
  exit 2
fi
echo "[setup] ChampSim built: $WORKDIR/ChampSim/bin/champsim"

# ---------- 5. Also clone Quangmire ML-ChampSim (for the NN prefetcher demo) ----------
cd "$WORKDIR"
if [ ! -d "$WORKDIR/ChampSim-ML" ]; then
  echo "[setup] cloning Quangmire/ChampSim-ML (for the Colab MLP demo path)"
  git clone https://github.com/Quangmire/ChampSim.git ChampSim-ML
fi
# Note: We do NOT build ChampSim-ML now. It has its own (older) build flow.
# When you want to use it for the MLP demo:
#   cd ChampSim-ML
#   ./ml_prefetch_sim.py run path/to/trace.xz --prefetch prefetch_list.txt
# We test that path at the very end (Section 7).

# ---------- 6. Download DPC-3 traces ----------
TRACE_DIR="$WORKDIR/traces"
mkdir -p "$TRACE_DIR"
cd "$TRACE_DIR"

DPC3_BASE="https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu"
declare -a TRACES=(
  "619.lbm_s-4268B"        # streaming
  "605.mcf_s-994B"         # pointer chase
  "620.omnetpp_s-874B"     # indirect
  "602.gcc_s-734B"         # branch / control
  "623.xalancbmk_s-700B"   # sparse C++
)
declare -A STATUS

for t in "${TRACES[@]}"; do
  if [ -f "$TRACE_DIR/${t}.champsimtrace.xz" ]; then
    echo "[trace] $t already cached"; STATUS[$t]="cached"; continue
  fi
  echo "[trace] downloading $t (typical size ~150 MB)"
  if wget --quiet --show-progress \
       "${DPC3_BASE}/${t}.champsimtrace.xz" \
       -O "${t}.champsimtrace.xz"; then
    STATUS[$t]="downloaded"
  else
    echo "[warn] DPC-3 mirror failed for $t. Manual fallback:"
    echo "       Zenodo: https://zenodo.org/records/10960004"
    STATUS[$t]="MISSING"
    rm -f "${t}.champsimtrace.xz"
  fi
done

# ---------- 7. Smoke test ----------
cd "$WORKDIR/ChampSim"
echo
echo "[smoke-test] running baseline LRU + no prefetch on the first available trace"
FIRST_OK=""
for t in "${TRACES[@]}"; do
  if [ -f "$TRACE_DIR/${t}.champsimtrace.xz" ]; then FIRST_OK="$t"; break; fi
done

if [ -n "$FIRST_OK" ]; then
  echo "[smoke-test] trace: $FIRST_OK"
  bin/champsim \
    --warmup-instructions 1000000 \
    --simulation-instructions 5000000 \
    "$TRACE_DIR/${FIRST_OK}.champsimtrace.xz" \
    > "$WORKDIR/smoke_test.log" 2>&1 || {
      echo "[smoke-test] FAILED. Last 30 lines of log:"
      tail -30 "$WORKDIR/smoke_test.log"
      exit 3
    }
  echo "[smoke-test] PASS. Key lines:"
  grep -E "cumulative IPC|LLC TOTAL" "$WORKDIR/smoke_test.log" || tail -10 "$WORKDIR/smoke_test.log"
else
  echo "[smoke-test] SKIPPED: no trace was downloaded successfully."
fi

# ---------- 8. report ----------
echo
echo "================================================================"
echo "[setup] DONE."
echo
echo "  ChampSim binary  : $WORKDIR/ChampSim/bin/champsim"
echo "  ChampSim-ML repo : $WORKDIR/ChampSim-ML/  (for the MLP demo path)"
echo "  Traces dir       : $TRACE_DIR"
echo
echo "[setup] trace status:"
for t in "${TRACES[@]}"; do
  echo "  $t : ${STATUS[$t]:-unknown}"
done
echo
echo "Next steps:"
echo "  bash run_baseline.sh     # baseline IPC on all 5 traces (~10 min)"
echo "  bash run_upper_bound.sh  # 4 configs x 5 traces, upper-bound chart data"
echo "  bash run_mlp_demo.sh     # feed Colab-generated prefetch_list.txt to ChampSim"
echo "================================================================"