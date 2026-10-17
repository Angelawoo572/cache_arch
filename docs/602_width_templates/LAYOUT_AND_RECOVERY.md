# Canonical study layout and recovery

All local study activity stays below `/Users/angelawoo/Documents/CMU/ece/cache`.
Sacramento uses `/home/qianruw/cache`. The root-relative layout is the same:

| Purpose | Canonical path |
| --- | --- |
| Healthy active checkout | `formal_study/work/cache_gate_mechanism` |
| First formal, gate mechanism, recovery donor | `formal_study/archive/20261016` |
| Four-route v1 and v2, independently preserved | `formal_study/archive/20261017` |
| New H16 and capped-H8-template results | `formal_study/runs/602_width_templates_20261017` |
| Rename evidence and restoration map | `formal_study/migration` |
| Original offline training experiments | `formal_NN_training/experiments` (unchanged) |
| Trace and simulator dependencies | `traces`, `external/ChampSim` (unchanged) |

The old date-root names are relative compatibility symlinks to `archive/`.
Local `archive/20261016/work/cache_gate_mechanism` additionally points to the
active checkout. Sacramento's historical source-only directory at that old
location stays an independent archive: it was never a Git worktree. The new
Sacramento checkout is independent. Never synchronize the old compatibility
path onto that preserved historical source directory.

The local active `.git` is a real main Git directory, not a linked worktree.
Its object alternate is the retained recovery donor at
`formal_study/archive/20261016/cache_recovered/.git/objects`; do not remove that
donor. No external Git worktree pointer needs repair. Git connectivity and the
branch `experiment/602-gate-mechanism` were checked before and after relocation.
The original cache-root damaged `.git` was preserved; use only the healthy
checkout for Git commands.

Migration used same-filesystem directory renames, with per-file size/device/inode
checks, rather than copying or deleting historical data. Local verification
covered 15,314 files / 7,651,782,754 bytes; Sacramento covered 14,141 files /
4,419,228,854 bytes (its source-only layout and donor differ). Historical JSON,
commands, source snapshots, raw logs, builds, v1 and v2 remain unchanged. Their
old absolute paths are provenance, resolved by the compatibility links; they
are not active destination settings. Full per-file rename evidence remains in
`formal_study/migration`; lightweight maps are retained here. No hash ledger is
required.

The active new configuration is
`formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/config.json`.
The parent `config.json` remains the archived H8 protocol's recoverable settings,
with old path aliases working. Do not rerun that protocol into its old raw root.
New runtime, temporary, matplotlib and Python cache files belong under the new
`runs/602_width_templates_20261017/runtime` directory. The offline sibling layout
is unchanged.

To recover, restore `archive/`, `work/`, `runs/` and `migration/` under the same
cache root. Restore the relative date aliases and, on the local machine, the
old checkout alias. If the cache root changes, update the active Git alternates
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
