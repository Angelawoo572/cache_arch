# Four-route paper story

- **Question:** Can small recurrent students trained on traditional prefetcher actions improve IPC, accuracy, coverage and timeliness—and the IPC/traffic Pareto trade-off—against their own native teachers?
- **Scope:** Stride, Streamer, AMPM and SPP; retained seed7 weights; two complete 25M ROIs, each with 5M cache/state warmup; synchronous MAC0 inference. ROI0 was previously inspected. Weights remain frozen while h/c advances online.
- **Measured result:** H8 wins **9/16** native comparisons, including **6/8 capped pairs**. Yet **all eight capped NN cases use more DRAM** than their native baseline. Stride/Streamer/AMPM trade higher capped IPC for more DRAM; native SPP dominates its H8 NN on both axes. No capped H8 NN dominates its own native on IPC and DRAM. [Exact pairs](reader_pairs.csv).
- **Quality is not performance:** higher accuracy or timeliness does not guarantee higher IPC. Report L2/LLC first-enqueue-intent cohorts separately. Coverage is useful requests/NoPF misses, an opportunity proxy; net miss reduction is separate. Tiny/empty SPP L2 cohorts cannot stand for LLC benefit. [Metrics and Pareto definitions](README.md).
- **Secondary mechanism result:** H16 AMPM/SPP wins **8/8** native IPC pairs, but simple validation-frozen templates reproduce nearly all gain. SPP matches four cases exactly; AMPM has small uncapped residuals and loses slightly capped. [Detailed interventions](FINDINGS.md).

| Claim | Status and limit |
| --- | --- |
| Independent frozen live inference and mixed quality/IPC/traffic outcomes | **Supported** for these complete runs; replay is separate. |
| Uniform NN superiority over native or necessary learned payload/state | **Not supported**; retain losses and template explanations. |
| Broad or finite-hardware advantage | **Unproven**: one seed/workload, different route training/state budgets, and different native/student SPP fill inputs. MAC0 gives no compute/area/energy Pareto result. |

- [Project overview](../../../RESULTS_INDEX.md) · [four-route results](README.md) · [cost boundaries](storage_cost.csv).
