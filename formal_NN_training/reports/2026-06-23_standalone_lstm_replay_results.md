# Standalone LSTM keyed replay results — 2026-06-23

This note records frozen-list offline replays using the PC-line-occurrence keyed ListReplayer, not in-simulator PyTorch execution.

## Build / replay configuration

```text
Branch Predictor: perceptron
L1D Prefetcher: no
L2C Prefetcher: standalone_nn_replayer
LLC Prefetcher: no
LLC Replacement: ship
Cores: 1
Binary: bin/perceptron-no-standalone_nn_replayer-no-ship-1core
```

Normal prefetchers are comparison baselines only. The standalone LSTM exports are replayed at L2C through the keyed list replayer.

## v3.1 hybrid balanced replay

Source:

```text
formal_NN_training/results/standalone_lstm_replay/v3_1_hybrid_balanced_all5_20260623_194953/summary.csv
```

| trace | IPC | no-pref IPC | best normal | best normal IPC | speedup vs no-pref | speedup vs best normal | selected accuracy | timeliness | pf issued | pf useful | pf useless | pf late | transport ok |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 602.gcc_s-734B | 0.42863 | 0.36800 | sandbox | 0.43628 | 1.16476 | 0.98247 | 0.97375 | 0.99859 | 187803 | 182874 | 4068 | 258 | 1 |
| 619.lbm_s-4268B | 0.38492 | 0.32200 | sms | 0.38105 | 1.19540 | 1.01016 | 0.97826 | 0.98418 | 293605 | 287223 | 1732 | 4618 | 1 |
| 605.mcf_s-994B | 0.18862 | 0.18318 | ampm | 0.18874 | 1.02970 | 0.99936 | 0.52606 | 0.80115 | 104121 | 54774 | 5304 | 13595 | 1 |
| 620.omnetpp_s-874B | 0.24503 | 0.23750 | sms | 0.24695 | 1.03171 | 0.99223 | 0.45482 | 0.99082 | 169940 | 77286 | 63543 | 716 | 1 |
| 623.xalancbmk_s-700B | 0.36407 | 0.35321 | spp | 0.35391 | 1.03075 | 1.02871 | 0.68316 | 0.98718 | 317571 | 216910 | 21844 | 2817 | 1 |

## v3.3 context-hierarchical balanced replay

Source:

```text
formal_NN_training/results/standalone_lstm_replay/v3_3_context_balanced_all5_20260623_194504/summary.csv
```

| trace | IPC | no-pref IPC | best normal | best normal IPC | speedup vs no-pref | speedup vs best normal | selected accuracy | timeliness | pf issued | pf useful | pf useless | pf late | transport ok |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 602.gcc_s-734B | 0.42887 | 0.36800 | sandbox | 0.43628 | 1.16541 | 0.98302 | 0.98091 | 0.99900 | 186723 | 183158 | 3056 | 183 | 1 |
| 619.lbm_s-4268B | 0.38434 | 0.32200 | sms | 0.38105 | 1.19360 | 1.00863 | 0.98564 | 0.99291 | 293442 | 289227 | 2115 | 2064 | 1 |
| 605.mcf_s-994B | 0.18729 | 0.18318 | ampm | 0.18874 | 1.02244 | 0.99232 | 0.43148 | 0.73834 | 118930 | 51316 | 5380 | 18186 | 1 |
| 620.omnetpp_s-874B | 0.24559 | 0.23750 | sms | 0.24695 | 1.03406 | 0.99449 | 0.44437 | 0.99257 | 187366 | 83252 | 74440 | 623 | 1 |
| 623.xalancbmk_s-700B | 0.37893 | 0.35321 | spp | 0.35391 | 1.07282 | 1.07070 | 0.75104 | 0.99700 | 570297 | 428219 | 25092 | 1288 | 1 |

## Chosen standalone winner versus best normal — IPC, accuracy, timeliness, and coverage

For every trace, this table uses the better IPC of v3.1 and v3.3. `selected accuracy = useful / (issued - PQ-merged duplicates)`. `timeliness = useful / (useful + late)`. `coverage vs no-pref miss pool = useful / no-prefetch L2 demand misses`; it is a useful-prefetch count ratio, not the unique demand-miss reduction metric. It can exceed 1.0 for an aggressive prefetcher because useful-prefetch counts may include repeated uses.

| trace | chosen standalone | best normal | IPC delta | standalone accuracy | normal accuracy | accuracy delta | standalone timeliness | normal timeliness | timeliness delta | standalone coverage | normal coverage | coverage delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 602.gcc_s-734B | v3.3 context | sandbox | -0.00741 | 0.98091 | 0.10184 | +0.87907 | 0.99900 | 0.99985 | -0.00085 | 0.90166 | 1.92045 | -1.01879 |
| 619.lbm_s-4268B | v3.1 hybrid | SMS | +0.00387 | 0.97826 | 0.94210 | +0.03616 | 0.98418 | 0.95350 | +0.03068 | 0.97483 | 1.81622 | -0.84139 |
| 605.mcf_s-994B | v3.1 hybrid | AMPM | -0.00012 | 0.52606 | 0.17787 | +0.34819 | 0.80115 | 0.98578 | -0.18463 | 0.07399 | 0.12225 | -0.04826 |
| 620.omnetpp_s-874B | v3.3 context | SMS | -0.00136 | 0.44437 | 0.24470 | +0.19967 | 0.99257 | 0.98185 | +0.01072 | 0.26299 | 0.62465 | -0.36166 |
| 623.xalancbmk_s-700B | v3.3 context | SPP | +0.02502 | 0.75104 | 0.00485 | +0.74619 | 0.99700 | 0.80694 | +0.19006 | 0.95086 | 0.01736 | +0.93350 |

### How to read this table

- **Accuracy and timeliness:** the standalone policy is generally much cleaner than the best normal policy because it emits a selectively filtered list. This does not by itself prove a performance win; IPC remains the final metric.
- **Coverage:** the normal prefetcher can obtain a larger useful-count coverage ratio by issuing far more requests. Therefore coverage should be read together with issued traffic and IPC, not as a standalone winner criterion.
- **602:** the standalone policy is much cleaner but covers fewer demand-miss opportunities than sandbox. This matches the current candidate-reachability ceiling.
- **619:** standalone beats SMS on IPC, accuracy, and timeliness while covering fewer useful-prefetch events; this is a traffic/policy-quality win.
- **605:** standalone has higher selected accuracy but lower coverage and substantially worse timeliness; the candidate representation remains the limiting factor.
- **620:** standalone is cleaner and more timely than SMS but does not yet cover enough useful demand-miss opportunities to overcome the IPC gap.
- **623:** v3.3 is the strongest result: it beats SPP on IPC, accuracy, timeliness, and useful-count coverage.

## Best-normal traffic context

| trace | chosen standalone | standalone IPC | best normal IPC | standalone issued | best-normal issued | standalone / best-normal issued |
|---|---|---:|---:|---:|---:|---:|
| 602.gcc_s-734B | v3.3 context | 0.42887 | 0.43628 | 186723 | 3839474 | 4.86% |
| 619.lbm_s-4268B | v3.1 hybrid | 0.38492 | 0.38105 | 293605 | 568017 | 51.69% |
| 605.mcf_s-994B | v3.1 hybrid | 0.18862 | 0.18874 | 104121 | 509840 | 20.42% |
| 620.omnetpp_s-874B | v3.3 context | 0.24559 | 0.24695 | 187366 | 808086 | 23.19% |
| 623.xalancbmk_s-700B | v3.3 context | 0.37893 | 0.35391 | 570297 | 1670201 | 34.15% |

## Immediate observations

- v3.3 is the strongest current result on `623.xalancbmk_s-700B`: `1.07282x` over no-prefetch and `1.07070x` over the best normal baseline.
- v3.1 is slightly stronger than v3.3 on `619.lbm_s-4268B` IPC, although v3.3 has higher offline selected accuracy and timeliness.
- `602.gcc_s-734B` remains below sandbox. Its selected accuracy and timeliness are already very high, so the main gap is candidate coverage / policy capacity rather than simple address ranking.
- `605.mcf_s-994B` is representation-limited. v3.1 is close to AMPM; v3.3 regresses.
- `620.omnetpp_s-874B` improves over no-prefetch but remains slightly below SMS.

## Per-trace current winner from these two replays

| trace | current best standalone result | reason |
|---|---|---|
| 602.gcc_s-734B | v3.3 balanced | slightly higher IPC and cleaner than v3.1, but still below sandbox |
| 619.lbm_s-4268B | v3.1 balanced | highest IPC |
| 605.mcf_s-994B | v3.1 balanced | closest to AMPM; v3.3 worsens |
| 620.omnetpp_s-874B | v3.3 balanced | slightly higher IPC than v3.1, still below SMS |
| 623.xalancbmk_s-700B | v3.3 balanced | clear win over no-prefetch and best normal |

## Next notebook direction

1. Preserve v3.1 hybrid and v3.3 context-hierarchical exports as two candidate families.
2. Use replay results, not only offline recall, to choose candidate-family and policy winners.
3. Treat `605.mcf_s-994B` as representation-limited unless a new candidate action space raises held-out reachability.
4. Treat `619.lbm_s-4268B` as timing-policy-limited rather than coverage-limited.
5. Treat `623.xalancbmk_s-700B` as the strongest current context-candidate showcase.
