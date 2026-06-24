# Standalone LSTM keyed replay results — 2026-06-23

This note records the raw keyed-replay results for the standalone NN prefetcher experiments. These are frozen-list offline replays using the PC-line-occurrence keyed ListReplayer, not in-simulator PyTorch execution.

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

Source summary:

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

Source summary:

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

## Best-normal comparison with traffic context

The baseline rows below come from `formal_NN_training/results/prefetcher_baselines/summary.csv`. The standalone row is the better of v3.1 and v3.3 balanced replay for each trace. `standalone / best-normal issued` is not a universal efficiency metric, but it makes the traffic scale visible.

| trace | chosen standalone | standalone IPC | best normal IPC | IPC delta vs best normal | standalone issued | best-normal issued | standalone / best-normal issued | comparison |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 602.gcc_s-734B | v3.3 context | 0.42887 | sandbox: 0.43628 | -0.00741 | 186723 | 3839474 | 4.86% | Near sandbox IPC with far less prefetch traffic; coverage is the remaining bottleneck. |
| 619.lbm_s-4268B | v3.1 hybrid | 0.38492 | SMS: 0.38105 | +0.00387 | 293605 | 568017 | 51.69% | Standalone wins IPC while using about half of SMS issued traffic. |
| 605.mcf_s-994B | v3.1 hybrid | 0.18862 | AMPM: 0.18874 | -0.00012 | 104121 | 509840 | 20.42% | Essentially tied IPC, but low candidate reachability prevents a reliable standalone win. SPP also ties AMPM at 0.18874 IPC. |
| 620.omnetpp_s-874B | v3.3 context | 0.24559 | SMS: 0.24695 | -0.00136 | 187366 | 808086 | 23.19% | Near SMS IPC with much lower traffic, but low prediction quality remains the limit. |
| 623.xalancbmk_s-700B | v3.3 context | 0.37893 | SPP: 0.35391 | +0.02502 | 570297 | 1670201 | 34.15% | Strongest result: standalone wins IPC by 7.07% over the best normal baseline with about one-third of its issued traffic. |

### Best-normal baseline behavior

| trace | best normal | IPC | issued | useful | useless | late | selected accuracy | timeliness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 602.gcc_s-734B | sandbox | 0.43628 | 3839474 | 390102 | 32524 | 59 | 0.10184 | 0.99985 |
| 619.lbm_s-4268B | SMS | 0.38105 | 568017 | 535130 | 84 | 26099 | 0.94210 | 0.95350 |
| 605.mcf_s-994B | AMPM | 0.18874 | 509840 | 90495 | 58586 | 1305 | 0.17787 | 0.98578 |
| 620.omnetpp_s-874B | SMS | 0.24695 | 808086 | 197737 | 394038 | 3655 | 0.24470 | 0.98185 |
| 623.xalancbmk_s-700B | SPP | 0.35391 | 1670201 | 7816 | 23108 | 1870 | 0.00485 | 0.80694 |

## Immediate observations

- v3.3 is the strongest current result on `623.xalancbmk_s-700B`: it reaches `1.07282x` over no-prefetch and `1.07070x` over the best normal prefetcher.
- v3.1 is slightly stronger than v3.3 on `619.lbm_s-4268B` IPC, although v3.3 improves accuracy and timeliness. Both beat the best normal SMS baseline on replay.
- `602.gcc_s-734B` remains below sandbox. Both v3.1 and v3.3 have very high selected accuracy and timeliness, so the remaining gap is likely candidate coverage / traffic-policy limited rather than address-quality limited.
- `605.mcf_s-994B` is still representation-limited. v3.1 is close to AMPM, while v3.3 regresses.
- `620.omnetpp_s-874B` improves over no-prefetch but remains slightly below SMS.

## Per-trace current winner from these two replays

| trace | current best standalone result | reason |
|---|---|---|
| 602.gcc_s-734B | v3.3 balanced | slightly higher IPC and better accuracy/timeliness than v3.1, but still below sandbox |
| 619.lbm_s-4268B | v3.1 balanced | highest IPC, despite v3.3 having cleaner accuracy/timeliness |
| 605.mcf_s-994B | v3.1 balanced | closest to AMPM; v3.3 worsens due to representation limitation |
| 620.omnetpp_s-874B | v3.3 balanced | slightly higher IPC than v3.1, still below SMS |
| 623.xalancbmk_s-700B | v3.3 balanced | clear win over no-prefetch and best normal |

## Next notebook direction

The next notebook should be replay-driven rather than only offline-recall-driven:

1. Preserve v3.1 hybrid and v3.3 context-hierarchical exports as two candidate families.
2. Add a per-trace policy selector that chooses between exported families using held-out validation plus replay evidence.
3. Treat `605.mcf_s-994B` as representation-limited unless a new candidate action space raises reachability.
4. Treat `619.lbm_s-4268B` as timing-policy-limited rather than coverage-limited.
5. Treat `623.xalancbmk_s-700B` as the strongest current showcase for context-hierarchical candidates.

## Open design question: cache level placement

Current replay injects standalone NN prefetches only at L2C:

```text
L1D Prefetcher: no
L2C Prefetcher: standalone_nn_replayer
LLC Prefetcher: no
```

Future experiments should consider a cache-level sweep, but keep it separate from model correctness:

- L1D prefetching may help latency more but is more sensitive to pollution and timing.
- L2C is the current safe default and aligns with the existing logger/replayer path.
- LLC prefetching may reduce lower-level miss latency but can hide less latency than L1/L2 and can increase bandwidth pressure.

Recommended next step is not to mix all three levels at once. First finish the L2C standalone matrix; then add an explicit cache-level sweep with the same frozen prefetch lists.
