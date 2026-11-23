#!/bin/bash
# Fetch, patch and build the GAP benchmark suite on Sacramento, then generate
# the graph inputs. Usage: setup.sh [scale ...]   default scale: 20
set -eu
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../sim/env.sh"

gap="$CACHE_ROOT/external/gapbs"
pin=941462c7c42e786e4e006a9211b0576ef78d23f7
graphs="$GEM5_STUDY/raw/workloads/graphs"
mkdir -p "$graphs"

if [ ! -d "$gap/.git" ]; then
  git clone --quiet https://github.com/sbeamer/gapbs.git "$gap"
fi
cd "$gap"
git checkout --quiet "$pin" -- src
python "$here/np_patch.py" "$gap"

# The system compiler and -O2 without vectorisation: dynamically linked x86-64
# baseline code that gem5's syscall emulation runs with the host C library.
make --quiet SERIAL=1 CXX=/usr/bin/g++ \
  CXX_FLAGS="-std=c++11 -O2 -march=x86-64 -mtune=generic -fno-tree-vectorize -Wall" \
  bfs pr cc bc sssp converter

for scale in "${@:-20}"; do
  [ -f "$graphs/kron$scale.sg" ]   || ./converter -g "$scale" -k 16 -b "$graphs/kron$scale.sg"
  [ -f "$graphs/urand$scale.sg" ]  || ./converter -u "$scale" -k 16 -b "$graphs/urand$scale.sg"
  [ -f "$graphs/kron$scale.wsg" ]  || ./converter -g "$scale" -k 16 -w -b "$graphs/kron$scale.wsg"
  [ -f "$graphs/urand$scale.wsg" ] || ./converter -u "$scale" -k 16 -w -b "$graphs/urand$scale.wsg"
done
ls -l "$graphs"
echo "gapbs $(git rev-parse HEAD), patched benchmark.h, built with $(/usr/bin/g++ --version | head -1)"
