# SPP width/template formal subset

- **Current H8/H16 results and controls:** [shared reader](../../../../../formal_study/results/602_width_templates/README.md). This subset preserves the earlier 88-row study; the additional 30-run completion is linked from that reader.
- Complete policy rows plus three exact matched NoPF references: **30 rows**. Original protocols and result identities are retained; references are not new runs.

- [Aggregate findings](../../../../../formal_study/results/602_width_templates/FINDINGS.md), [one-page story](../../../../../formal_study/results/602_width_templates/PAPER_STORY.md), [all88 labeled rows](../../../../../formal_study/results/602_width_templates/per_run.csv), [strict audit](../../../../../formal_study/results/602_width_templates/audit.json), [method](../../../../../formal_study/results/602_width_templates/method.svg).

- [This policy + matched NoPF](per_run.csv), [paired contrasts](paired_comparisons.csv), [entire-ROI action histograms](action_histograms_full_roi.csv), [phase histograms](action_histograms.csv), [classified request outcomes](classified_requests.csv), [actual startup snapshots](progress_snapshots.csv), [phase progress](phase_progress.csv), [storage/cost](storage_cost.csv).

- [Retained real H16 checkpoint](../../models/h16/original_checkpoint.pt) is the source model asset; no online training or new seed is implied.

- CSV profiles join the unmodified original64-source tables and supplemental24-source tables, filter actual run_ids, and add protocol/origin/result labels plus source_table. This is a transparent subset, not a new measurement or an average. All rows use actual complete ROIs; no prefixes are extrapolated.

- MAC0 synchronous inference only. Same ceiling does not imply equal realized requests/DRAM. L2/LLC target cohorts remain separate; unused is not pollution. SPP native keeps prefetch-fill FILTER feedback while the NN sees all raw L2 fills. Exact actual binary/configured Sacramento controls are reused; independent historical hardware telemetry is absent.
