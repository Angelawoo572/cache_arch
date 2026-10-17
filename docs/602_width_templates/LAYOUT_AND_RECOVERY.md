# Canonical study layout and recovery

All local study activity stays below `/Users/angelawoo/Documents/CMU/ece/cache`.
Sacramento uses `/home/qianruw/cache`. The root-relative layout is the same:

| Purpose | Canonical path |
| --- | --- |
| Healthy active checkout | `cache` root (`.`), same as GitHub root |
| Shared lightweight result folder | `formal_study/results/602_width_templates` |
| First formal, gate mechanism, recovery donor | `formal_study/archive/20261016` |
| Four-route v1 and v2, independently preserved | `formal_study/archive/20261017` |
| New H16 and capped-H8-template results | `formal_study/runs/602_width_templates_20261017` |
| Rename evidence and restoration map | `formal_study/migration` |
| Original offline training experiments | `formal_NN_training/experiments` (unchanged) |
| Trace and simulator dependencies | `traces`, `external/ChampSim` (unchanged) |

The visible date-root aliases `formal_study_20261016` and
`formal_study_20261017` were removed after consolidation. The former
`cache/docs/formal_study_20261016` report (334 files /25,950,182 bytes) now lives
at `formal_study/archive/20261016/report`, with every file/inode preserved.
Do not recreate these dated top-level names. The existing internal
`archive/20261016/work/cache_gate_mechanism` pointer still resolves to the healthy
checkout; Sacramento's same historical source-only location remains independent.

Frozen JSON records retain original path strings. The
[canonical-path launcher](canonical_paths.py) translates decoded paths only in
memory before running retained scripts. It changes neither historical bytes nor
model/protocol/control values. On Sacramento, prefix retained commands with:

```sh
study_python=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
"$study_python" docs/602_width_templates/canonical_paths.py \
  formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/driver.py status
```

For a read-only local binding check, use `--local --check` before the script.
The launcher is required for retained drivers that read historical build/binding
JSON, including `profile_cost.py`; direct historical commands may contain removed
names. The complete24-row local and Sacramento bindings passed without aliases.

The local active `.git` is a real main Git directory, not a linked worktree.
Its object alternate is the retained recovery donor at
`formal_study/archive/20261016/cache_recovered/.git/objects`; do not remove that
donor. No external Git worktree pointer needs repair. Git connectivity and the
branch `experiment/602-gate-mechanism` were checked before and after relocation.
The original cache-root Git directories are preserved under
`formal_study/archive/root_checkout_before_unification_20261017/` (local and
Sacramento have their own retained metadata). The healthy main `.git` is now at
the cache root. `formal_study/work/cache_gate_mechanism` is a compatibility link
to `../..`; it contains no second checkout. Run Git commands from cache.

Migration used same-filesystem directory renames, with per-file size/device/inode
checks, rather than copying or deleting historical data. Local verification
covered 15,314 files / 7,651,782,754 bytes; Sacramento covered 14,141 files /
4,419,228,854 bytes (its source-only layout and donor differ). Historical JSON,
commands, source snapshots, raw logs, builds, v1 and v2 remain unchanged. Their
old absolute paths are provenance, resolved by the canonical-path launcher; they
are not active destination settings. Full per-file rename evidence remains in
`formal_study/migration`; lightweight maps are retained here. No hash ledger is
required.

The active new configuration is
`formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/config.json`.
The parent `config.json` remains the archived H8 protocol's recoverable settings,
with its original recorded paths preserved. Use the canonical-path launcher for recovery. Do not rerun that protocol into its old raw root.
New runtime, temporary, matplotlib and Python cache files belong under the new
`runs/602_width_templates_20261017/runtime` directory. The offline sibling layout
is unchanged.

To recover, check out this study branch directly as the cache root, then restore
`formal_study/archive/`, `formal_study/runs/` and `formal_study/migration/` beneath
it. Keep `formal_study/results/` from Git and recreate the optional
`formal_study/work/cache_gate_mechanism -> ../..` compatibility link. Use the canonical-path launcher for recorded date paths; do not recreate the removed top-level aliases. Restore the internal archived checkout pointer if needed. If the cache root changes, update the active Git alternates
path to the retained donor and run `git fsck --connectivity-only` in the healthy
checkout. Update only the new active config; preserve archived metadata. Restore
traces and the matching ChampSim base/dependencies described by the existing
[forward/dependency guide](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/FORWARD_REPRODUCTION.md).
The new driver reuses the archived actual binary only after byte checks of its
compiled source, executable, controls and reference metadata, and compiles its
parity probe from that actual core. It neither overwrites an old build nor
relabels old rows as new measurements.

`consolidate_remote.py` documents and checks the one-time Sacramento rename. It
is intentionally guarded against overwriting an existing canonical archive.
The file-level evidence is not committed because it is bulky and machine-specific.


## Root unification (2026-10-17)

Local, Sacramento and GitHub now use the same repository root and real results
folder: `formal_study/results/602_width_templates/`. All 84 original light-archive
files moved intact (15,496,846 bytes); old result and checkout locations are only
compatibility links, not duplicate datasets. The 82 manifest-bound files and all
historical provenance remain byte-identical. The added root navigation does not
change a model, protocol or measurement.

Local merge verified 18,869 protected original files (89,942,853,626 bytes) by
size/device/inode, including offline runs, raw, models, traces, builds and Git
recovery objects. Sacramento has its own equivalent merge record. Divergent
root source versions and the original root Git were preserved rather than
replaced irrecoverably. Detailed evidence and the guarded one-time merge tool
are under `formal_study/migration/unify_cache_root_20261017/`. These machine
records are intentionally local/server only. Both frozen source/model/build
gates and the complete 24-run status are rechecked after consolidation.

Do not copy or recreate a dated top-level `formal_study_2026...` directory. The
old `archive/20261016` and `archive/20261017` names distinguish historical
protocols inside the common archive; they are not parallel active projects.
