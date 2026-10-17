# spp formal v2 result subset

Protocol: `four602_frozen_live_20261017_v2`. This is an exact row/field subset of the complete64-row archive,
containing this policy and its same-build, same-split, same-ROI matched NoPF controls.
It does not import any old replay, old live, first-formal or superseded-v1 row.

Own policy rows: 16; matched NoPF rows: 3; total: 19.

[Full results, figures, definitions and certificates](../../../602_four_prefetcher_formal/results/formal_20261017/README.md).

Tables preserve original fields and values without recomputing averages or hiding negative results.

- [per_run.csv](per_run.csv).
- [action_histograms.csv](action_histograms.csv).
- [action_histograms_full_roi.csv](action_histograms_full_roi.csv).
- [phase_progress.csv](phase_progress.csv).
- [progress_snapshots.csv](progress_snapshots.csv).
- [classified_requests.csv](classified_requests.csv).

Sparse action/classification tables need not contain NoPF rows when no matching event exists.
L2/LLC cohorts and their denominators remain separate; MAC0 is synchronous, cap is a ceiling,
and SPP effective fill-update inputs differ from its native PF-only feedback.

Complete raw result root: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2`.
