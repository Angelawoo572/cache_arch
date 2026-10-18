# Canonical layout and recovery

- Read [RESULTS_INDEX](../../RESULTS_INDEX.md) first. The shared reading branch is `main`; branch/path strings in migration records remain historical evidence.
- All local study activity stays under `/Users/angelawoo/Documents/CMU/ece/cache`. Sacramento uses `/home/qianruw/cache`, with the same root-relative layout.

| Purpose | Canonical path |
| --- | --- |
| Active checkout / GitHub repository root | `cache` root (`.`) |
| Shared lightweight results | `formal_study/results/602_width_templates` |
| First formal, gate study and local recovery donor | `formal_study/archive/20261016` |
| Four-route v1/v2, separately preserved | `formal_study/archive/20261017` |
| Original AMPM/SPP H16 and capped H8-template raw | `formal_study/runs/602_width_templates_20261017` |
| Current Stride/Streamer H16 completion raw | `formal_study/runs/602_width_templates_20261017/stride_streamer_h16` |
| Rename evidence and restoration maps | `formal_study/migration` |
| Original offline training experiments | `formal_NN_training/experiments` |
| Traces and matching simulator dependencies | `traces`, `external/ChampSim` |

## Current H16 completion

- Active config: [complete_h16.json](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/complete_h16.json); [run/recovery commands](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/README.md#current-stridestreamer-h16-completion). Earlier `width_templates/config.json` and parent H8 `config.json` remain historical, with their original protocol identities.
- The completed, audited 30-condition extension uses recovered Stride/Streamer H16 weights, unchanged certified binary, two complete 25M ROIs, 5M cache warmup, MAC0 and 8 jobs. Native/NoPF and compatible controls are reused; validation alone freezes new templates.
- Restore the new raw child with its `protocol_source/`, model/template bindings and both accepted certificates: `correctness/stride/compatibility_and_parity.json` and `correctness/streamer/parity/parity_report.json`. `prepare` binds them before validation/select/test.
- Inspect `controller.json` and per-run `run.json`; only the analyzer's audit establishes verification. Do not infer completion from the expected matrix or launch a second active controller. Preserve existing analysis outputs and choose a fresh child directory for recovery analysis.
- Model-size calculation is consolidated in [Model size and inference state](../../formal_study/results/602_width_templates/README.md#model-size-and-inference-state): actual tensor parameters → FP32 bytes, separate global/per-PC h/c and host overhead. SVG/PNG share sources; storage is not synthesized area.

## Retained paths and launcher

- Removed visible aliases `formal_study_20261016` and `formal_study_20261017` must not be recreated.
- The former `cache/docs/formal_study_20261016` report now lives in `formal_study/archive/20261016/report`: **334 files / 25,950,182 bytes**, with every file/inode preserved.
- `formal_study/work/cache_gate_mechanism -> ../..` points to the cache root and contains no second checkout. Retain/restore it for frozen source bindings that still use that path.
- The existing internal `archive/20261016/work/cache_gate_mechanism` pointer also resolves to the active checkout locally. Sacramento's historical source-only location remains independent.
- Frozen JSON keeps its original strings. The [canonical-path launcher](canonical_paths.py) translates decoded date paths only in memory; model, protocol, controls and historical bytes are unchanged.
- Use the launcher for retained drivers reading old build/binding JSON, including `profile_cost.py`. Direct historical commands may reference removed date names.
- The complete **24-row** local/Sacramento binding checks passed without the removed date-root aliases. The work compatibility pointer above is a separate requirement for the recorded source paths.

- Historical 24-row status check, through the existing Sacramento CPU environment:

```sh
study_python=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
"$study_python" docs/602_width_templates/canonical_paths.py \
  formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/driver.py status
```

- For a read-only local source/model/build binding check, place `--local --check` before the script:

```sh
python3 -B docs/602_width_templates/canonical_paths.py --local --check \
  formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/driver.py
```

## Git and migration evidence

- The active `.git` is a real main Git directory at the cache root, not a linked worktree. Run Git commands there.
- The current local Git alternate requires `formal_study/archive/20261016/cache_recovered/.git/objects`; do not remove that donor. Sacramento has independent healthy objects, not an alternate to this local donor.
- No external worktree pointer needs repair. Connectivity and `experiment/602-gate-mechanism` were checked before/after relocation; `main` is now the reading branch.
- Original root Git directories remain under `formal_study/archive/root_checkout_before_unification_20261017/`, with independent local/Sacramento metadata.
- Same-filesystem renames preserved per-file device/inode/size. The earlier consolidation checked **15,314 local files / 7,651,782,754 bytes** and **14,141 Sacramento files / 4,419,228,854 bytes**; their source/donor layouts differ.
- Original JSON, commands, source snapshots, raw logs, builds and v1/v2 remain unchanged. Recorded absolute paths are provenance, not new destination settings.
- Full machine-specific evidence stays in `formal_study/migration`; lightweight maps remain in this documentation. No hash ledger is required.

## Restore or resume

- Check out the required code as the cache root: `main` for shared reading, or the retained experiment branch required for an exact historical reproduction.
- Restore `formal_study/archive/`, `formal_study/runs/` and `formal_study/migration/` underneath it. Keep the Git version of `formal_study/results/`.
- Restore original offline training runs, traces, and the matching ChampSim base/dependencies in the [forward/dependency guide](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/FORWARD_REPRODUCTION.md).
- Restore `formal_study/work/cache_gate_mechanism -> ../..` and the historical internal archived-checkout pointer. Use the launcher for old date paths; do not recreate date-root aliases.
- If the cache root changes, update only the active Git alternate to the retained donor, then run `git fsck --connectivity-only` in the healthy checkout. Preserve archived metadata.
- For a genuinely new experiment, change only its new active destination settings; preserve completed config/source and archived metadata under their existing raw identity.
- Active configuration is `formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/complete_h16.json`. The earlier width `config.json` and parent H8 `config.json` retain frozen protocols/recorded paths; recover them through the launcher without rerunning into completed raw roots.
- Current runtime/temp/matplotlib/Python-cache files belong under `formal_study/runs/602_width_templates_20261017/stride_streamer_h16/runtime`; older runtime and original offline sibling paths stay unchanged.
- The new driver reuses the archived actual binary only after checking executable, compiled source, controls and reference bytes. Its parity probe compiles from that actual core; old builds and row identities are not overwritten.
- `consolidate_remote.py` records the guarded one-time Sacramento rename and refuses to overwrite an existing canonical archive. Bulky per-file evidence is local/server only.

## Root unification: 2026-10-17

- Local, Sacramento and GitHub use the cache repository root and `formal_study/results/602_width_templates/` as the real results directory.
- The root merge moved **84 original light-archive files / 15,496,846 bytes** intact. Old result/checkout locations are compatibility links. The merge checkpoint verified all **82 originally copied artifacts** against their sources; maintained reading-document edits are separate from that historical copy check and do not alter measurements.
- The local merge checked **18,869 protected original files / 89,942,853,626 bytes** by device/inode/size, including offline runs, raw, models, traces, builds and recovery objects. Sacramento has its own equivalent record.
- Divergent source versions and original root Git were preserved. Evidence and the guarded one-time tool remain under `formal_study/migration/unify_cache_root_20261017/`, local/server only.
- Frozen source/model/build gates and all **24** completed run records were rechecked after consolidation.
- Do not recreate a dated top-level `formal_study_2026...` directory. Internal `archive/20261016` and `archive/20261017` distinguish historical protocols, not active parallel projects.

## Reading assets

- `summary.csv` and method SVG/PNG live in `formal_study/results/602_width_templates/`; public Markdown points directly to real files.
- Former `docs/602_width_templates/{summary.csv,method.svg,method.png}` locations are ignored, local-only compatibility symlinks for retained provenance.
- When restoring from Git plus raw archives, point each alias at `../../formal_study/results/602_width_templates/<filename>`. They are excluded from Git and the docs-only Pages checkout; the original Pages workflow remains intact.

## Repository and documentation preservation

- `main` normally merged `experiment/602-gate-mechanism`; the earlier live → dynamic → gate experiment ancestry and all three branch tips remain intact. The archive `archive/summer-2026-offline` was created only when absent and preserves pre-integration remote main, including its Pages update. No force-push or history rewrite was used.
- Both checkouts were clean before integration. Local sparse-checkout removals were restored from retained Git objects without overwriting surviving files, then sparse mode was disabled. Sacramento's shallow history was completed; it has independent healthy Git objects.
- The integration checkpoint verified all 82 manifest-bound copies against their sources, actual raw/build/model paths, branch ancestry and Pages links. The donor pack passed integrity verification; Git had refreshed only its timestamp. Machine-specific checks remain under `formal_study/migration/main_integration/`.
- Redundant reading pointers and repeated paper stories have been consolidated into the root overview, current results, this recovery guide and each historical protocol's findings. Retained experimental reports remain separate from current H8/H16 comparisons.

## Reader documentation

- Maintained Markdown is the English reader layer. Original reports remain in their raw analysis/audit locations and Git history; copy manifests retain their original historical provenance.
- Translated or consolidated Markdown is not claimed byte-identical to its initial archived copy. Original numerical CSV/JSON, trained models, source/config snapshots, old figures and raw evidence are preserved.
- Current reader CSVs/figures are derived from completed runs. Quality denominators, original replay table cells and storage assumptions remain explicit; no training or simulation is needed for these documentation updates.
