# Reproduce the new four-route analysis

- Run analysis only against this new protocol's raw root. The default matrix is all 8 full validation rows and 56 full test rows. It independently recomputes the validation-only request ceiling and joint per-slot delta/fill templates. An incomplete matrix is explicitly INCOMPLETE; no prefix result becomes a full-ROI estimate. Failures return status1, incomplete returns2 unless --partial is supplied.

- Local command from the selected source checkout:

    /opt/anaconda3/bin/python -B formal_NN_training/experiments/602_four_prefetcher_formal/analyze.py \
      --raw /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2 \
      --out /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2/analysis_final \
      --stages test validation \
      --route-manifest formal_NN_training/experiments/602_four_prefetcher_formal/route_manifest.json

- Use a fresh output directory; existing analysis outputs are preserved. --partial is for displaying completed full rows while jobs run, not for shortening simulation. --no-plots omits raster/vector charts. --expected-matrix can supply a list of specs (or an object containing runs/specs). --route-manifest supplies evidence for offline/live/old-formal status; absent evidence stays not_supplied.
- Its shape is {"routes":{"streamer":{"offline":{"status":"completed","evidence":"path","details":"scope"}}}}. --cost-fields defaults to the source report/cost_fields.json and is verified against actual exported weight/hc payload before being attached.

- Outputs:
- per_run.csv: each actual completed run, including negative results, raw scalar counters and all derived denominators.
- raw_flat.csv and raw_snapshots.json: all source fields, including nested arrays and exact distributions.
- action_histograms.csv / action_histograms_full_roi.csv: phase and exact whole-ROI distributions; SPP fill callbacks are excluded from demand-action denominator.
- classified_requests.csv: origin slot/current/fill scope, actual allowed phases and independent target lifecycle.
- progress_snapshots.csv / phase_progress.csv: actual cumulative and interval coordinates; instruction, demand callback and completed-update axes are distinct.
- paired_comparisons.csv: actual native/student, cap and mechanism interventions, with realized request differences.
- storage_cost.csv / cost_fields.json: FP32/hc payload, separate host requested-memory observations and actual counted operations; never a complete SRAM/PPA/RSS claim.
- callback_verification.json: full recorded callback sequence, D/F totals and pre-callback completed-update axis.
- metric_definitions.json, selection_verification.json, audit.json, route_status.csv, index.html/report.md and PNG/SVG figures.

- LLC tracking availability false produces null LLC accuracy/timeliness/coverage. L2 target accuracy remains separate. Neither target's usefulness ratio is net miss reduction; the two levels are not pooled. MAC0 is synchronous. MAC4 aggregate cycle estimates are analytical work, not a finite-cost IPC result.

- Training budgets differ: retained Stride seed7 balanced 12epochs0..1M, three global students seed7 8epochs0..20M. Compare each native/student route; do not rank four learning algorithms as matched retraining. SPP native and student retain their distinct effective fill-feedback contracts. Test25..50M is previously inspected replication;75..100M is future ROI. Training amount is not live h/c history.

- Focused semantic checks, without Torch or simulation:

    /opt/anaconda3/bin/python -B formal_NN_training/experiments/602_four_prefetcher_formal/report/test_analysis.py

- All runtime/config/log/temp outputs stay inside cache/formal_study_20261017. The analyzer reads raw data without modification and creates its runtime cache directories inside that study.

- The active protocol is four602_frozen_live_20261017_v2. Earlier raw/ contains the preserved v1 attempt with truncated callback tails; never substitute it for raw_v2/. The corrected v2 study reruns all64 rows and strictly requires complete callback logs. Remote CPU Torch has no Matplotlib: analysis records pending_local_render_matplotlib_unavailable and retains data; run the same final command locally for figures.

- Full-ROI mechanism panel script (requires the complete64/PASS v2 audit and verified callback logs):

    /opt/anaconda3/bin/python -B formal_NN_training/experiments/602_four_prefetcher_formal/report/plot_mechanisms.py \
      --analysis /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2/analysis_final \
      --out /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/mechanism_figures_final

- Outputs mechanisms_roi0_seed7.png/svg and mechanisms_roi1_seed7.png/svg, exact plotted24-row mechanism_plot_data.csv, and figure_manifest.json. Full/fixed_payload/unconditional_template/reset_before_each_callback remain separate; actual requests/M, completed DRAM MiB and ΔIPC versus full are annotated. No negative result is removed. A given callback's complete forward/hc is preserved under fixed payload, while changed cache feedback can change later callbacks and the full h/c trajectory. SPP canonical target/action labels and native raw source calls have distinct K scopes.

- Progress axes explicitly use ROI-reset retired instructions, L2 LOAD callbacks, and aggregate completed updates. The full students retain warmed h/c from5M cache warmup; warmup_completed_updates and warmup demand/fill counts are separate per-run fields. Aggregate updates are not per-PC visible history, a zero-history experiment, training samples, or a minimum-history estimate.
