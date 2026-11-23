#!/bin/bash
# Build the validation microbenchmarks on Sacramento.
# The system compiler is used on purpose: the binaries are dynamically linked
# against the host C library, which is what gem5's syscall emulation loads.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../../raw/workloads/micro"
mkdir -p "$out"
flags="-O2 -march=x86-64 -mtune=generic -fno-tree-vectorize -Wall -Wextra"
for name in stride list tree csr hash; do
  /usr/bin/gcc $flags -o "$out/$name" "$here/$name.c"
done
ls -l "$out"
