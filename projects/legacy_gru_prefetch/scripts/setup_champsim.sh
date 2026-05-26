#!/usr/bin/env bash
# setup_champsim.sh
# Prepare the repo layout used by this project:
#   external/ChampSim
#   external/ChampSim-ML
#   traces/
#
# Run from the cache_arch repo root:
#   bash projects/legacy_gru_prefetch/scripts/setup_champsim.sh

set -euo pipefail

WORKDIR="$(pwd)"
EXT_DIR="$WORKDIR/external"
CHAMP="$EXT_DIR/ChampSim"
ML_DIR="$EXT_DIR/ChampSim-ML"
TRACE_DIR="$WORKDIR/traces"

echo "[setup] working directory: $WORKDIR"
mkdir -p "$EXT_DIR" "$TRACE_DIR"

# ---------- 1. OS packages, when apt is available ----------
if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] apt-get detected; installing build packages if needed"
  if [ "$(id -u)" -eq 0 ]; then SUDO="" ; else SUDO="sudo"; fi
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config zip unzip curl wget \
    git python3 python3-pip xz-utils zstd ca-certificates tar
else
  echo "[setup] apt-get not available; assuming compiler/cmake/git/wget are already installed"
fi

# ---------- 2. Initialize external repositories ----------
# Preferred path: use git submodules recorded in .gitmodules.
if [ -f "$WORKDIR/.gitmodules" ]; then
  echo "[setup] initializing git submodules under external/"
  git submodule update --init --recursive external/ChampSim external/ChampSim-ML || true
fi

# Fallback path for machines where submodules were not added yet.
if [ ! -d "$CHAMP/.git" ]; then
  echo "[setup] external/ChampSim missing; cloning official ChampSim fallback"
  rm -rf "$CHAMP"
  git clone --recursive https://github.com/ChampSim/ChampSim.git "$CHAMP"
fi

if [ ! -d "$ML_DIR/.git" ]; then
  echo "[setup] external/ChampSim-ML missing; cloning Quangmire fallback"
  rm -rf "$ML_DIR"
  git clone --recursive https://github.com/Quangmire/ChampSim.git "$ML_DIR"
fi

# Make sure nested submodules such as vcpkg exist.
cd "$CHAMP"
git submodule update --init --recursive

# ---------- 3. Bootstrap + install C++ dependencies via vcpkg ----------
if [ -d "$CHAMP/vcpkg" ]; then
  echo "[setup] bootstrapping vcpkg"
  if [ ! -x "$CHAMP/vcpkg/vcpkg" ]; then
    "$CHAMP/vcpkg/bootstrap-vcpkg.sh" -disableMetrics || {
      echo "[warn] normal vcpkg bootstrap failed; trying source/system-binaries path"
      (cd "$CHAMP/vcpkg" && VCPKG_USE_SYSTEM_BINARIES=1 ./bootstrap-vcpkg.sh -disableMetrics)
    }
  fi

  echo "[setup] installing ChampSim C++ dependencies through vcpkg"
  "$CHAMP/vcpkg/vcpkg" install --x-manifest-root="$CHAMP" --x-install-root="$CHAMP/vcpkg_installed"
else
  echo "[warn] $CHAMP/vcpkg not found; skipping vcpkg bootstrap"
fi

# ---------- 4. Configure + build default ChampSim binary ----------
echo "[setup] configure + make default ChampSim binary"
cd "$CHAMP"
python3 ./config.sh champsim_config.json > /tmp/config_champsim_default.log 2>&1 || {
  echo "[error] config.sh failed. See /tmp/config_champsim_default.log"
  tail -40 /tmp/config_champsim_default.log
  exit 2
}

if ! make -j8 > /tmp/build_champsim_default.log 2>&1; then
  echo "[error] make failed. Last 80 lines:"
  tail -80 /tmp/build_champsim_default.log
  echo "[hint] full log: /tmp/build_champsim_default.log"
  exit 3
fi

if [ ! -x "$CHAMP/bin/champsim" ]; then
  echo "[error] $CHAMP/bin/champsim missing after build"
  exit 4
fi

echo "[setup] default ChampSim built: $CHAMP/bin/champsim"

# ---------- 5. Download DPC-3 traces if missing ----------
cd "$TRACE_DIR"
DPC3_BASE="https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu"
declare -a TRACES=(
  "619.lbm_s-4268B"
  "605.mcf_s-994B"
  "620.omnetpp_s-874B"
  "602.gcc_s-734B"
  "623.xalancbmk_s-700B"
)
declare -A STATUS

for t in "${TRACES[@]}"; do
  if [ -f "$TRACE_DIR/${t}.champsimtrace.xz" ]; then
    echo "[trace] $t already cached"
    STATUS[$t]="cached"
    continue
  fi

  echo "[trace] downloading $t"
  if wget --quiet --show-progress "${DPC3_BASE}/${t}.champsimtrace.xz" -O "${t}.champsimtrace.xz"; then
    STATUS[$t]="downloaded"
  else
    echo "[warn] failed to download $t from DPC-3 mirror"
    echo "       Manual fallback: https://zenodo.org/records/10960004"
    STATUS[$t]="MISSING"
    rm -f "${t}.champsimtrace.xz"
  fi
done

# ---------- 6. Smoke test first available trace ----------
FIRST_OK=""
for t in "${TRACES[@]}"; do
  if [ -f "$TRACE_DIR/${t}.champsimtrace.xz" ]; then
    FIRST_OK="$t"
    break
  fi
done

if [ -n "$FIRST_OK" ]; then
  echo "[smoke-test] running default ChampSim on $FIRST_OK"
  "$CHAMP/bin/champsim" \
    --warmup-instructions 1000000 \
    --simulation-instructions 5000000 \
    "$TRACE_DIR/${FIRST_OK}.champsimtrace.xz" \
    > "$WORKDIR/smoke_test.log" 2>&1 || {
      echo "[smoke-test] FAILED. Last 40 lines:"
      tail -40 "$WORKDIR/smoke_test.log"
      exit 5
    }
  echo "[smoke-test] PASS"
  grep -E "cumulative IPC|LLC TOTAL" "$WORKDIR/smoke_test.log" || tail -10 "$WORKDIR/smoke_test.log"
else
  echo "[smoke-test] skipped because no trace is available"
fi

echo
echo "================================================================"
echo "[setup] DONE"
echo "  ChampSim      : $CHAMP"
echo "  ChampSim-ML   : $ML_DIR"
echo "  Traces        : $TRACE_DIR"
echo
echo "[setup] trace status:"
for t in "${TRACES[@]}"; do
  echo "  $t : ${STATUS[$t]:-unknown}"
done
echo
echo "Next commands:"
echo "  bash projects/legacy_gru_prefetch/scripts/install_and_build.sh"
echo "  bash projects/legacy_gru_prefetch/scripts/run_baseline.sh"
echo "  bash projects/legacy_gru_prefetch/scripts/install_bypass.sh"
echo "  bash projects/legacy_gru_prefetch/scripts/run_bypass.sh"
echo "================================================================"
