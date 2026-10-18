# stride formal v2 result subset

- **Current reader comparison:** [four native prefetchers versus all H8/H16 NNs](../../../../../formal_study/results/602_width_templates/README.md). This archive retains its original protocol and row values.

- Protocol: `four602_frozen_live_20261017_v2`. This is an exact row/field subset of the complete64-row archive, containing this policy and its same-build, same-split, same-ROI matched NoPF controls. It does not import any old replay, old live, first-formal or superseded-v1 row.

- Own policy rows: 13; matched NoPF rows: 3; total: 16.

- [Full results, figures, definitions and certificates](../../../602_four_prefetcher_formal/results/formal_20261017/README.md).

- Tables preserve original fields and values without recomputing averages or hiding negative results.

- [per_run.csv](per_run.csv).
- [action_histograms.csv](action_histograms.csv).
- [action_histograms_full_roi.csv](action_histograms_full_roi.csv).
- [phase_progress.csv](phase_progress.csv).
- [progress_snapshots.csv](progress_snapshots.csv).
- [classified_requests.csv](classified_requests.csv).

- Sparse action/classification tables need not contain NoPF rows when no matching event exists. L2/LLC cohorts and their denominators remain separate; MAC0 is synchronous, cap is a ceiling, and SPP effective fill-update inputs differ from its native PF-only feedback.

- **Historically recorded root (superseded):** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2`. The subset manifest preserves this original source string.

- **Current full raw roots:** local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2`; Sacramento `/home/qianruw/cache/formal_study/archive/20261017/raw_v2`.

- **Find this subset in raw evidence:** [subset_manifest.json](subset_manifest.json) records its run IDs; complete source tables are under `analysis_final/` and individual run evidence under `runs/<run_id>/`.
