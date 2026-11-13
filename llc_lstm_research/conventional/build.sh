#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cores="${1:-}"
case "$cores" in
    1|2|4|8) ;;
    *) echo "Usage: $0 {1|2|4|8} [build directory]" >&2; exit 2 ;;
esac

build_dir="${2:-$here/runs/build-$cores}"
mkdir -p "$build_dir"
build_dir="$(cd "$build_dir" && pwd)"
pythia="$build_dir/pythia"

if [[ -z "${CXX:-}" ]]; then
    if [[ "$(uname -s)" == Darwin && -x /opt/homebrew/bin/g++-15 ]]; then
        CXX=/opt/homebrew/bin/g++-15
    else
        CXX=g++
    fi
fi
export CXX

# A source download is staged separately so a failed download can be retried.
log="$build_dir/build.log"
trap 'tail -80 "$log" >&2' ERR
(
    if [[ ! -d "$pythia" ]]; then
        stage="$(mktemp -d "$build_dir/source.XXXXXX")"
        trap 'rm -rf "$stage"' EXIT
        git -c advice.detachedHead=false clone --quiet --depth 1 --branch v1.3 \
            https://github.com/CMU-SAFARI/Pythia.git "$stage/pythia"
        git -c advice.detachedHead=false clone --quiet --depth 1 --branch v1.0.0 \
            https://github.com/mavam/libbf.git "$stage/pythia/libbf"
        mv "$stage/pythia" "$pythia"
    fi
    python3 "$here/configure.py" "$pythia" "$cores"
    cmake -S "$pythia/libbf" -B "$pythia/libbf/build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build "$pythia/libbf/build" --parallel "${JOBS:-4}"
    make -C "$pythia" -j"${JOBS:-4}" CXX="$CXX" buildrepo
    make -C "$pythia" -j"${JOBS:-4}" CXX="$CXX"
) >"$log" 2>&1
trap - ERR
echo "$pythia/bin/champsim"
