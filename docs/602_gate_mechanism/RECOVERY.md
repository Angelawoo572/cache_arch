# Mechanism-study files and recovery

- [Canonical stage report](REPORT.md) · [shared project overview](../../RESULTS_INDEX.md) · [current cache layout](../602_width_templates/LAYOUT_AND_RECOVERY.md).
- Current maintained source is the **cache root on main**, locally and on Sacramento. The original experiment branch remains historical evidence; it is not a separate current checkout.
- Machine roots: local `/Users/angelawoo/Documents/CMU/ece/cache/`; Sacramento `/home/qianruw/cache/`. The paths below are relative to those roots. Large raw/build/model archives are local/Sacramento only.

| Preserved material | Canonical cache-relative path |
| --- | --- |
| This stage's complete raw/logs/models/tables/builds | `formal_study/archive/20261016/gate_output/raw/` |
| Original-v2 compatible references/checkpoints | `formal_study/archive/20261016/research_output/raw_v2/` |
| Fitted PC tables | `pc_gate/seed{7,17,27}/pc_gate.tsv` beneath this stage's raw root |
| Original balanced NN exports | `models/seed{7,17,27}/balanced/export/model.bin` beneath the reference raw root |
| Actual build/source/binary snapshots | `builds/build_20261016_213702/` beneath this stage's raw root |
| Local Git recovery object donor | `formal_study/archive/20261016/cache_recovered/.git/objects` |

## What is preserved

- Protocol `gate_mechanism_20261016_v1`; isolated build `build_20261016_213702`; ChampSim base `838be17e`.
- Starting source is the actual completed `frozen_formal_20261016_v2` implementation: 21 active files were compared byte-for-byte before validated fixes were overlaid.
- Retained controls include independent seed loading, FIFO/nonlinear configuration, common cache warmup, completion-only h/c publication, callback-independent cycle service and arrival-time shadow labels. The separate canonical h/c-history supplement is not transplanted.
- The runner checks source/config/binary/scorer/model/table equality before reuse and preserves incompatible attempts. A completed process is not an independent validation result: all 71 logical rows also require [completion](completion_certificate.json), [audit](results/audit.json) and [reference compatibility](compatible_reference_manifest.json).
- PC-fit provenance and loader verification are committed in [pc_gate_fit](pc_gate_fit/). The dynamic [installer](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/patches/install.py) and [retained run guide](../../formal_NN_training/experiments/602_gate_mechanism/RUN_GUIDE.md) describe the original implementation.
- All older source/raw/model/build/log variants and damaged Git metadata remain archived. **Do not remove the local Git object donor.**
- Frozen JSON preserves original absolute paths, including old `cache_gate_mechanism_20261016*`, `cache_formal_20261016_runs_v2` and application-workspace locations. These strings are historical provenance, **not active destinations**.
- The general [canonical-path launcher](../602_width_templates/canonical_paths.py) translates the dated paths it supports in memory. It does not by itself map every old gate-root spelling; do not assume an unchanged historical pipeline command resumes at the canonical archive.
- A fresh execution must explicitly bind canonical source/raw/reference paths in an isolated recovery setup and pass the original equality gates; do not edit frozen metadata or silently reuse incompatible outputs. No new simulation is requested by this documentation update.

## Reanalyze preserved outputs

- From the cache root, use **unused output directories**. These commands read the complete canonical raw roots and regenerate tables; they do not train or run ChampSim.

```sh
source .runtime/project_env.sh
EXP=formal_NN_training/experiments/602_gate_mechanism
RAW=formal_study/archive/20261016/gate_output/raw
PRIOR=formal_study/archive/20261016/research_output/raw_v2
OUT=formal_study/migration/gate_reanalysis
SUMMARY=formal_study/migration/gate_resummary
python3 "$EXP/analysis_gate.py" --raw "$RAW" --prior-raw "$PRIOR" \
  --out "$OUT" --no-plots
python3 "$EXP/summarize_gate.py" --raw "$RAW" --analysis "$OUT" \
  --out "$SUMMARY"
```

- Plotting additionally needs Matplotlib. Retain each training-seed row; three seeds are not independent workload samples or confidence intervals.
- Startup plots use complete saved snapshots; mechanism histograms accumulate the full ROI. No prefix log substitutes for either.
- Reuse the existing Sacramento authentication/socket and host verification. No credential or security change is part of recovery.
