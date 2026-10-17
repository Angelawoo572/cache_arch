# Formal studies: one layout

**Current results: [602 H16 / capped H8 templates](results/602_width_templates/README.md).**
All 24 new full runs are complete, with 9,748 checks passed. Findings, per-run CSV,
metrics, action distributions, progress curves and all 18 figure pairs are in
that real directory. Earlier independent studies remain linked from
[RESULTS_INDEX](../RESULTS_INDEX.md).

| Purpose | Path relative to the cache / GitHub root | On GitHub |
| --- | --- | --- |
| Current lightweight results | `formal_study/results/602_width_templates/` | Yes |
| Source, frozen models, resumable study | `formal_NN_training/experiments/` | Yes, curated files |
| Complete new raw, logs, checkpoints and snapshots | `formal_study/runs/602_width_templates_20261017/` | Local and Sacramento |
| Historical source/raw/model/build/report rounds | `formal_study/archive/20261016/`, `archive/20261017/` | Local and Sacramento; light summaries linked separately |
| Original offline training runs | `formal_NN_training/experiments/*/runs/` | Local and Sacramento |
| Merge and recovery evidence | `formal_study/migration/` | Local and Sacramento |

The active healthy Git checkout is **the cache root**, on
`experiment/602-gate-mechanism`. Local root is
`/Users/angelawoo/Documents/CMU/ece/cache/`; Sacramento root is
`/home/qianruw/cache/`. The same relative results path is used on GitHub.
Sacramento is the server name, not a separate project named “sacramental”.

`work/cache_gate_mechanism` now points to the cache root for retained frozen
source bindings. The former experiment result path points here for compatibility;
there is one real current result directory. Dated top-level aliases stay removed.
Historical metadata and v1/v2 variants are retained independently.

[Layout, original Git preservation and recovery](../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).
