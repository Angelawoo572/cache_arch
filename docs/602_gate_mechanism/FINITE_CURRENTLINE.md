# Finite engine: current-line filtering

The four rows below are the seed 7 balanced-checkpoint subset. Each uses a full 25M measured-instruction ROI, 5M common cache warmup/history, MAC4, resident-weight bandwidth 16B/cycle, one nonlinear unit at 4 cycles/operation, FIFO input16/output32, state64, learned payload and the validation-fixed cap of 2363 requests/million instructions. They retain the original v2 state and service policy. Finite-cost filtering for seeds 17/27 is outside this subset.

| ROI | Arm | IPC | Cycles | Allowed requests | Current suppressed | Budget suppressed | Input dropped | State evictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0: 25–50M | original | 0.36791013 | 67,951,385 | 59,073 | 0 | 31,122 | 79.5299% | 0 |
| 0: 25–50M | filtered | 0.36802691 | 67,929,824 | 44,239 | 45,604 | 0 | 79.5170% | 0 |
| 1: 75–100M | original | 0.28614638 | 87,367,878 | 59,074 | 0 | 66,363 | 80.5773% | 128 |
| 1: 75–100M | filtered | 0.28591890 | 87,437,388 | 56,676 | 63,749 | 4,989 | 80.5612% | 125 |

| ROI | Arm | Queue wait mean | Completed service mean | Completion age mean | Queue wait max | Completion age max | Completed-service busy lower bound |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | original | 13022.558 | 821.342 | 13843.895 | 15,517 | 16,487 | 99.9986% |
| 0 | filtered | 13010.542 | 820.573 | 13831.113 | 15,498 | 16,468 | 99.9995% |
| 1 | original | 13364.412 | 843.511 | 14207.925 | 15,568 | 16,541 | 99.9049% |
| 1 | filtered | 13364.173 | 843.483 | 14207.659 | 15,568 | 16,541 | 99.9052% |

Times are simulated cycles. Means, maxima and complete log2 histogram counts use all started/completed events. Quantile intervals are bucket bounds, not exact percentile points. The denominator is `nn_started` for queue wait/scheduled service, and `nn_completed` for completed service/completion age. The scheduled-service sum may exceed ROI cycles because it includes the final unfinished inference; the busy fraction above excludes service already spent on that final inference.

ROI 0: the filter changes IPC by +0.03174% and requests by -25.111%. NoPF IPC is 0.36786129; original/filtered changes against NoPF are +0.01328%/+0.04502%. Eligible LOAD counts are 404,238/404,238; inference starts are 82,732/82,784, completions 82,731/82,783. Both finish with 16/16 queued inputs and 1/1 active inference(s).

ROI 1: the filter changes IPC by -0.07950% and requests by -4.059%. NoPF IPC is 0.28614732; original/filtered changes against NoPF are -0.00033%/-0.07983%. Eligible LOAD counts are 532,856/532,856; inference starts are 103,479/103,565, completions 103,478/103,564. Both finish with 16/16 queued inputs and 1/1 active inference(s).

The [source certificate](cost_invariants.json) verifies that `latency` and `begin` are byte-identical to v2. Removing only the diagnostic request-context call makes `service` identical too. The original learned decoder count after the existing cap32 determines MAC/nonlinear/scratch/service cost, before final payload selection. One active inference holds h/c and output until completion; saturated initiation interval equals service time. Unconditional outer-cycle service continues without new LOAD callbacks.

Current-line filtering runs after output-queue removal and before page/budget credits. The pop and output initiation interval still occur for suppressed calls, so filtering directly saves requests and credits, rather than original forward computation or output-drain cycles. Both within-ROI pairs have identical simulator commands and functional environments except the filter0/1 flag; warmup completion, eviction, pending-work cleanup and per-PC warmup-update evidence match. No canonical h/c initialization was transplanted. Differences in aggregate work arise from simulation feedback, not a changed service formula.

The present finite engine remains almost continuously busy, drops most eligible inputs and incurs long queue delays. These four rows show request reductions with very small IPC changes under the stated finite model. They do not establish a robust finite-cost performance benefit across training seeds. MAC×4 weight-read accounting omits bias traffic by assumption; resident weights produce no modeled DRAM transactions, while actual cache/DRAM bus transfers are measured separately. No lookup/energy/area synthesis is claimed.

The [combined JSON](finite_currentline.json) preserves all four raw summaries, control checks, timing histograms, means, maxima and denominators. The per-ROI JSON files are retained as subsets. Rebuild this report with `python build_finite_cost_report.py --raw /path/to/gate_output/raw`.
