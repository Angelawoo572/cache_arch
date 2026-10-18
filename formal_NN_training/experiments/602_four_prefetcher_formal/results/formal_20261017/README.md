# Complete four-route formal v2 archive

- **Current reader comparison:** [four native prefetchers versus H8 NN, with separate H16 results](../../../../../formal_study/results/602_width_templates/README.md). This archive retains its original protocol and row values.

- **Reader note:** Markdown has English translations and concise formatting; original reports and numeric artifacts remain preserved. [What changed](../../../../../formal_study/INTEGRATION.md#reader-documentation).

- Protocol: `four602_frozen_live_20261017_v2`;64 completed full runs (56 test +8 validation), strict callback checks, generated PNG/SVG figures, one actual build and a fresh real-checkpoint correctness gate.

- [Paper report](report.md), [per-run metrics](per_run.csv), [paired contrasts](paired_comparisons.csv), [Reviewed findings](FINDINGS.md), [concise paper story](../../report/PAPER_STORY.md), [audit](audit.json), [metric definitions](metric_definitions.json), [callback verification](callback_verification.json), [cost fields](cost_fields.json), [storage/cost table](storage_cost.csv), [source metadata](metadata/), [copied-byte and subset certificate](archive_verification.json).

- Numerical CSV/JSON, figures and original raw analysis artifacts are preserved. Published Markdown is the English reader layer. The large redundant `raw_snapshots.json`, raw event streams, models, binaries, traces and large logs remain at the full raw source. No old protocol is pooled into these tables.

- **Historically recorded roots (superseded):** local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2`; Sacramento `/home/qianruw/cache/formal_study_20261017/raw_v2`. Frozen metadata keeps these original strings.

- **Current full raw roots:** local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2`; Sacramento `/home/qianruw/cache/formal_study/archive/20261017/raw_v2`. Complete analysis is under `analysis_final/`; individual run evidence is under `runs/<run_id>/`.

## Policy subsets

- [stride](../../../602_lstm_stride_dynamic_inference/results/formal_20261017/README.md).
- [streamer](../../../602_lstm_streamer_live_inference/results/formal_20261017/README.md).
- [ampm](../../../602_lstm_ampm_live_inference/results/formal_20261017/README.md).
- [spp](../../../602_lstm_spp_live_inference/results/formal_20261017/README.md).

- ROI0 was previously observed; ROI1 is future relative to the original training/evaluation. Equal ceilings do not imply equal actual traffic. Native/student SPP effective fill inputs differ. The three global students are complete MAC0 forwards; Stride keeps its per-PC/decoder controls. Negative and mixed findings are retained. No finite-cost IPC or novelty claim is established by this archive.
