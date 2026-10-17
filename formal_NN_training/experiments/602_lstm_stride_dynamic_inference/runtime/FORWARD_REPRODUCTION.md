# Forward-source reproduction and dependency recovery

The canonical arithmetic is in
`602_lstm_stride_live_inference/runtime/stride_lstm_{runtime,model_loader}.{h,cc}`.
The dynamic engine needs `InferState`, an output limit and an emit margin.
`prepare_forward.py` generates its four C++ files in an isolated ChampSim tree;
do not maintain hand-edited generated copies. Generation validates each source
anchor exactly once before writing any artifact. It recognizes the existing
canonical rounded-count int64-domain guard explicitly; the previous permissive
replacement silently matched zero times on that already-fixed source.

The emitted four files remain byte-identical to both completed historical
builds. The change tightens generation checks, without changing their C++
arithmetic. A source marker records protocol, file paths and byte lengths;
exact byte comparison, rather than a digest, is the validity check. A legacy
completed build may lack the new marker; read-only verification reports that
absence explicitly. Every newly prepared tree requires its marker to match.

## Checks from the repository root

All paths below stay inside the user's cache. On Sacramento replace the local
cache prefix with `/home/qianruw/cache`; do not change authentication/security
settings. Select a fresh runtime/output location for each new study.

```sh
export TMPDIR=/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/runtime_v2/tmp
export PYTHONPYCACHEPREFIX=/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/runtime_v2/pycache
export MPLCONFIGDIR=/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/runtime_v2/mpl
python3 formal_NN_training/experiments/602_lstm_stride_dynamic_inference/validation/test_prepare_forward.py
python3 formal_NN_training/experiments/602_lstm_stride_dynamic_inference/validation/verify_forward_reproduction.py \
  --build /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261016/research_output/raw_v2/builds/build_20261016_185845 \
  --build /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261016/gate_output/raw/builds/build_20261016_213702 \
  --out formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/formal_20261016_v2/forward_source_verification.json
```

The first check has six tests for exact output, anchors, supported count guards,
marker validation and tamper detection. The second compares each actual built
forward to both the current and archived canonical sources, checks the four
canonical source bytes, and checks the actual executable against its immutable
binary snapshot. It does not claim equality of different observer/runner
protocols, identical binaries from a new compiler build, or new ROI results.
The completed main and gate builds keep separate hook protocols.

For a fresh isolated simulator, `prepare_forward.py DESTINATION --repo REPO`
writes then checks all four files and marker. For a completed simulator use
`--verify-only` to avoid writing. Add `--require-marker` for a new tree. Changed
source/installer/measurement inputs require a new immutable build and raw root;
the formal driver's existing source-set/byte checks intentionally reject stale
reuse. Do not overwrite the historical source snapshot to make checks pass.

## Recover the consistent ChampSim dependency

The complete first-formal evidence root contains:

- `ChampSim_dependency.bundle`: the local Git dependency history containing base
  `838be17e`, used by the completed study; this base identifier is provenance,
  not a new hash ledger.
- `recovery_dependency_838be17e.tar.gz`: actual patched simulator sources,
  consistent `simulator/libbf` sources/library, archived source inputs and
  `build.log`/`build.json`. It contains no new clean Git checkout.
- `builds/build_20261016_185845/simulator`: the actual complete old simulator;
  `simulator_binary_snapshot` is its immutable executable copy.

For an exact old result audit use the saved simulator/source and the read-only
checks above. The recovery archive already contains installed hooks: do not
apply the installer or inherited patch to that patched tree again.

For a new build, recover a **fresh cache-owned dependency directory** from the
bundle, then supply the matching libbf source. The following Sacramento example
creates a new directory and does not alter the completed source:

```sh
study_root=/home/qianruw/cache/formal_study_20261016
recovery_root=/home/qianruw/cache/formal_study_20261017/runtime_v2/dependency_recovery
mkdir -p "$recovery_root"
git clone --no-checkout "$study_root/research_output/raw_v2/ChampSim_dependency.bundle" "$recovery_root/ChampSim"
git -C "$recovery_root/ChampSim" checkout --detach 838be17e
tar -xzf "$study_root/research_output/raw_v2/recovery_dependency_838be17e.tar.gz" \
  -C "$recovery_root/ChampSim" --strip-components=1 \
  --exclude='simulator/libbf/build' simulator/libbf
cmake -S "$recovery_root/ChampSim/libbf" -B "$recovery_root/ChampSim/libbf/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$recovery_root/ChampSim/libbf/build" --target libbf_static --parallel 2
```

Use the Sacramento CMake/compiler environment recorded by the original build
(CMake 3.26.5 was recorded by the saved libbf cache), or a deliberately new
protocol if toolchain behavior changes. The Makefile links
`libbf/build/lib/libbf.a`; a different library release or incompatible host
binary is not a substitute. The current formal builder archives the clean
base, applies `inherited_cache.patch` once, copies this dependency's libbf tree,
generates the forward once, and runs the matching installer once. Point the
new study's configuration at that dependency and a new raw directory. This
recovery guide does not rewrite the archived historical configuration or
attempt to reuse its completed row certificates for a new build.

## Re-create the lightweight archive

```sh
python3 formal_NN_training/experiments/602_lstm_stride_dynamic_inference/validation/archive_formal_results.py \
  --study-root /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261016 \
  --out formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/formal_20261016_v2
```

This copies all 11 CSV files, 25 PNG/SVG figure pairs, metadata and certificates,
verifies each copied byte, and refuses to replace a different destination file.
It validates the single protocol and 89-test/12-validation row split. Models,
traces, raw streams, binaries, large logs and redundant nested `results.json`
remain at the authoritative full-result root and are excluded from Git output.

## Actual-build certificate scope

The historical exact-byte evidence is
`../results/formal_20261016_v2/forward_source_verification.json` relative to this
runtime directory. It compares four generated files in each of the actual main
and gate builds, their four canonical source inputs, and each immutable binary
snapshot. It is not a certificate for an unbuilt newer protocol. New v2 builds
record their own `raw_v2/build.json`, actual `compiled_source` byte copies and
fresh `raw_v2/certifications/<id>` parity/flush evidence. The isolated builder
retains actual selected modules instead of restoring NoPF after compilation;
its simulator and probe use `-O3 -ffp-contract=off`. Dependency recovery itself
does not establish numerical parity or a completed formal ROI.
