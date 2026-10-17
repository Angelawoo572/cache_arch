# Finite engine: current-line filtering

- [Stage report](REPORT.md) · [shared project overview](../../RESULTS_INDEX.md).
- **Scope:** four full 25M rows, **seed 7 balanced only**, ROIs 25–50M /75–100M; 5M common cache warmup/history, learned payload and validation-fixed cap **2,363 requests/M**. Seeds 17/27 are not covered.
- **Engine:** MAC4, resident-weight bandwidth 16 B/cycle, one nonlinear unit at 4 cycles/operation, input FIFO 16/output FIFO 32, decoder limit 32 and state 64. Original v2 state/service policy is retained.

| ROI | Arm | IPC | Cycles | Allowed requests | Current suppressed | Budget suppressed | Input dropped | State evictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0: 25–50M | Original | 0.36791013 | 67,951,385 | 59,073 | 0 | 31,122 | 79.5299% | 0 |
| 0: 25–50M | Filtered | 0.36802691 | 67,929,824 | 44,239 | 45,604 | 0 | 79.5170% | 0 |
| 1: 75–100M | Original | 0.28614638 | 87,367,878 | 59,074 | 0 | 66,363 | 80.5773% | 128 |
| 1: 75–100M | Filtered | 0.28591890 | 87,437,388 | 56,676 | 63,749 | 4,989 | 80.5612% | 125 |

| ROI | Arm | Queue wait mean | Completed service mean | Completion age mean | Queue wait max | Completion age max | Completed-service busy lower bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Original | 13022.558 | 821.342 | 13843.895 | 15,517 | 16,487 | 99.9986% |
| 0 | Filtered | 13010.542 | 820.573 | 13831.113 | 15,498 | 16,468 | 99.9995% |
| 1 | Original | 13364.412 | 843.511 | 14207.925 | 15,568 | 16,541 | 99.9049% |
| 1 | Filtered | 13364.173 | 843.483 | 14207.659 | 15,568 | 16,541 | 99.9052% |

## Result and denominators

- **No consistent finite-cost gain:** filtered versus NoPF is **+0.04502% /−0.07983%** on ROI0/1. Both filtered runs fall below the allowed credit ceiling. The engine remains almost continuously busy and drops about 80% of eligible inputs.
- ROI0: filtering changes IPC **+0.03174%**, requests **−25.111%**. NoPF IPC 0.36786129; original/filtered versus NoPF **+0.01328% /+0.04502%**. Eligible LOADs 404,238/404,238; starts 82,732/82,784; completions 82,731/82,783.
- ROI1: filtering changes IPC **−0.07950%**, requests **−4.059%**. NoPF IPC 0.28614732; original/filtered versus NoPF **−0.00033% /−0.07983%**. Eligible LOADs 532,856/532,856; starts 103,479/103,565; completions 103,478/103,564.
- Every row ends with **16 queued inputs and 1 active inference**. Pending work is not counted as completed.
- Times are simulated cycles; histograms cover all started/completed events. Queue wait and scheduled service divide by `nn_started`; completed service and completion age divide by `nn_completed`.
- Quantile intervals are log2 histogram bucket bounds, not precise percentiles. Scheduled service includes the final unfinished inference and can sum beyond ROI cycles; completed-service/ROI-cycles is only a busy-fraction lower bound.

## Preserved cost policy

- [Source certificate](cost_invariants.json): `latency` and `begin` are byte-identical to v2; `service` is identical after removing the diagnostic request-context call.
- Raw learned K, after the existing decoder cap 32, determines MAC/nonlinear/scratch/service cost **before final payload selection**. One active inference publishes h/c and output only at completion; saturated **II = service time**. Outer-cycle service continues without new callbacks.
- Current-line filtering runs after output-queue pop and before page/budget credits. Suppressed calls still incur the pop/output initiation interval; the filter does not directly save forward work or output-drain cycles.
- Within each ROI, commands are identical and filter 0/1 is the only functional environment difference. Warmup updates, evictions, pending-work cleanup and per-PC warmup evidence match. The separate canonical h/c-history initialization is not transplanted here.
- Aggregate work may differ through simulation feedback; the per-event service formula does not change.
- Dense `MAC × 4` read accounting excludes bias reads by assumption; resident weights create zero modeled DRAM traffic. Actual cache/DRAM bus transfers are separate measurements. No lookup/area/energy synthesis or multi-seed finite-cost claim is made.

## Evidence and regeneration

- [Combined four-row JSON](finite_currentline.json), [ROI0 subset](finite_currentline_roi0.json), [ROI1 subset](finite_currentline_roi1.json) retain raw summaries, timing histograms, means/maxima, denominators and pair-control checks.
- Full raw is local/Sacramento only at `formal_study/archive/20261016/gate_output/raw/` beneath either cache root; see [recovery](RECOVERY.md). Historical absolute paths inside JSON remain provenance.
- To regenerate numerical reports without overwriting these maintained docs or certificates, use an unused output directory from the cache root:

```sh
source .runtime/project_env.sh
python3 docs/602_gate_mechanism/build_finite_cost_report.py \
  --raw formal_study/archive/20261016/gate_output/raw \
  --output formal_study/migration/gate_finite_reanalysis
```
