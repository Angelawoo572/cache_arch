# Direct-position LLC LSTM

For the current two-version report, see [V1 vs. Newest](v5_results/v1_comparison/README.md). The sections below retain the earlier baseline comparisons and timing-provenance correction.

## Correction and equal-request coverage

- The archived V5 label uses NoPF for the future miss address and its lead. Its time weight came from 746 completed fills in a previous LSTM run at 130M. Those samples measure ingress to fill, including the earlier 16-cycle predictor. The first report incorrectly attributed the time weights to 20,361 NoPF demand misses.
- With the same local simulator, segments, degree and 32-cycle predictor, V2 and V5 issued 14,072 and 149,449 requests at their frozen cutoffs on 310M–390M. Coverage rose from 2.728% to 21.395%. At a common cap of ten per core per 20K retired instructions, coverage fell from 0.678% to 0.553% and timely coverage fell from 0.676% to 0.361%. Realized sends were 4,024 and 4,513. [Matched rows](v5_results/coverage_matched_by_region.csv) · [Pooled](v5_results/coverage_matched_pooled.csv)
- At 390M under that cap, both models issued exactly 2,000 requests. V2 covered 1,824 NoPF misses, including 1,821 timely, and reached 0.193028 IPC. V5 covered 968, including 650 timely, and reached 0.192104 IPC. The pure NoPF retrain covered 1,010, including 707 timely, and reached 0.192123 IPC with the same 2,000 sends.
- The original Colab V1 is a separate comparison. With 2,000 requests per policy and region, V1 to V5 coverage rose from 0.754% to 2.489% at 90M and from 0.666% to 2.843% at 110M. [Matched Colab rows](v5_results/coverage_matched_colab.csv)
- On 310M–390M, V5 and SPP covered 48,455 of the same NoPF misses. V5 had 38,709 exclusive targets, of which 21,376 were timely. SPP had 39,677 exclusive targets, of which 3,808 were timely. The better pooled timely coverage is mostly from different misses. [Coverage overlap and timing changes](v5_results/coverage_overlap.csv)

## Pure NoPF retrain

- Measure 369,307 completed NoPF demand services at LLC ingress on 10M, 30M and 50M. The median, p90 and p95 are 210, 263 and 340 cycles. Apply the CDF at lead minus 32 modeled inference cycles. Model features, architecture, 4,000-cycle horizon and region split remain the same. [Calibration](v5_results/clean32/calibration_summary.json)
- Fit for up to 12 epochs. Validation selects LSTM epoch 8 / Q10 −950 and Event-only epoch 5 / Q10 −650 on 70M and 130M. The LSTM threshold is the lower edge of the declared grid. [Development sweep](v5_results/clean32/development32.csv) · [Frozen setup](v5_results/clean32/frozen32.json)
- On the original five segments, the pure NoPF LSTM gets 21.437% direct and 13.677% timely coverage at its selected cutoff. Archived V5 got 21.395% and 13.667%. At the archived cutoff of 50, pure NoPF direct coverage is 20.691%. The source correction alone made little difference to pooled coverage.
- On three newly opened regions at 410M, 430M and 450M, the corrected LSTM gets 0.863222, 0.215747 and 0.663696 IPC. The fastest natives get 0.858428, 0.207555 and 0.651891. The LSTM takes 7.300 ms in total versus 7.517 ms, a 2.88% time saving. It uses 3.15% more DRAM bytes than those three native winners. [Full per-phase rows](v5_results/clean32/per_phase.csv) · [Pooled rows](v5_results/clean32/pooled.csv)
- At the shared 256-per-20K cap on the original five phases, the corrected LSTM wins 5/5 and saves 1.08% summed time. At the old ten-per-20K cap, the 390M regression remains. A fixed-cutoff hidden-state ablation in the fresh phases sent 350,387 versus 153,362 requests. Its 26.64% coverage accompanied a slowdown from 7.300 to 7.613 ms. The unequal sends leave the contribution of recurrence unresolved.

## Archived V5 test

- The frozen LSTM won all five fresh 605 regions against every conventional arm. It cut summed completion time by **2.23%** versus the fastest native policy in each region and by **2.42%** versus Streamer, the best fixed native policy.
- The runs used two independent copies of the 605 trace, 200K warmup and 2M ROI instructions per core, 4 GHz, and a 4 MiB shared LLC. Private cache prefetchers were off. The shared issue filter allowed one request per accepted L2D load miss.
- Train on 10M/30M/50M NoPF and select on local 70M/130M. The frozen regions are 310M/330M/350M/370M/390M. Each row uses actual retirement counts, including up to five instructions of retirement-width overshoot between arms at 390M.

| 605 region | NoPF IPC | Next-Line | Stride | Streamer | AMPM | SPP | Event-only | LSTM | Time saved vs fastest native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 310M | 0.511919 | 0.512422 | 0.512686 | 0.519580 | 0.519003 | 0.515182 | 0.522174 | 0.541265 | 4.01% vs Streamer |
| 330M | 0.502957 | 0.502451 | 0.503716 | 0.514108 | 0.513304 | 0.518255 | 0.523816 | 0.529101 | 2.05% vs SPP |
| 350M | 0.802433 | 0.810514 | 0.804128 | 0.814804 | 0.810709 | 0.809784 | 0.814678 | 0.817193 | 0.29% vs Streamer |
| 370M | 0.716974 | 0.722735 | 0.721503 | 0.726882 | 0.725623 | 0.730049 | 0.732798 | 0.733940 | 0.53% vs SPP |
| 390M | 0.191812 | 0.194343 | 0.194750 | 0.203269 | 0.201111 | 0.202981 | 0.203814 | 0.208630 | 2.57% vs Streamer |

## Pooled metrics

- Sum the five ROI times and raw metric counts before dividing. The rows below are portfolios of one fixed policy. [Definitions](../PERFORMANCE.md)

| Policy | IPC | Time (ms) | LLC hit | LLC miss | MPKI | Accuracy | Timeliness | Direct coverage | Timely coverage | Accepted | DRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.449765 | 11.117 | 32.61% | 67.39% | 17.404 | 58.81% | 63.63% | 21.40% | 13.67% | 149,449 | 29.94 |
| Event-only | 0.441591 | 11.323 | 26.46% | 73.54% | 18.925 | 99.55% | 100.00% | 6.48% | 6.48% | 26,515 | 26.00 |
| Streamer | 0.438890 | 11.392 | 28.85% | 71.15% | 18.369 | 43.11% | 62.70% | 15.40% | 9.70% | 147,609 | 31.58 |
| SPP | 0.438526 | 11.402 | 28.21% | 71.79% | 18.480 | 49.59% | 44.29% | 21.63% | 9.65% | 180,331 | 32.13 |
| AMPM | 0.436341 | 11.459 | 27.20% | 72.80% | 18.783 | 43.99% | 94.37% | 7.98% | 7.54% | 74,365 | 28.81 |
| Stride | 0.427346 | 11.700 | 22.17% | 77.83% | 20.115 | 96.10% | 99.59% | 1.37% | 1.37% | 5,822 | 26.26 |
| Next-Line | 0.427178 | 11.705 | 27.21% | 72.79% | 18.729 | 40.29% | 26.48% | 34.16% | 9.10% | 351,980 | 40.09 |
| NoPF | 0.423870 | 11.796 | 21.12% | 78.88% | 20.370 | — | — | 0.00% | 0.00% | 0 | 26.27 |

- Versus the fastest conventional per region, the LSTM increased pooled accuracy from 49.93% to 58.81%, timeliness from 56.08% to 63.63%, direct coverage from 17.36% to 21.40%, timely coverage from 9.84% to 13.67%, and LLC hit rate from 29.00% to 32.61%. DRAM traffic fell 2.65%. Streamer used less traffic on 310M and 390M individually.
- Stride and AMPM achieved high timeliness at lower coverage. SPP and Next-Line reached more NoPF misses than LSTM, with lower timely coverage and longer completion time. The 390M LSTM accuracy was 48.43%, versus 49.15% for Streamer.
- The separately trained Event-only policy issued 26,515 requests versus 149,449 for LSTM. The LSTM used 15.16% more DRAM bytes and saved 1.82% completion time. These frozen policy results use different realized traffic.
- With the LSTM scoring weights and cutoff held fixed, setting the recurrent input to the output head to zero gave 44,442,963 cycles and 149,770 requests. That control finished 0.06% faster than the full LSTM and won three of five regions. The final experiment supports the direct address representation and issue decisions. It leaves a benefit from recurrent state unestablished in this configuration.

## Cost and deployment checks

- A common ramped allowance of 256 requests per core per 20K retired instructions kept four wins and a 0.68% reduction in total time against the fastest capped native per region. At 64 requests, the LSTM won one region and took 0.37% longer overall.
- At 64-cycle decision latency and four in-flight decisions per core, the frozen LSTM still beat the native arms in all five regions. The changed latency also limits launch throughput. At 32 cycles no measured learned run spilled the waiting FIFO or stalled for in-flight capacity.
- With 602 GCC on the other core at its start and 605 at 310M or 390M, the same model won both segments and saved 2.56% summed time versus the fastest native per segment. These are two contention checks with a model fit on 605. [Sensitivity rows](v5_results/sensitivity.csv)
- C++ and PyTorch agreed on all selected top addresses and top-16 membership across 129,925 development triggers. Four scored lists changed internal order at one Q10 unit. The network has 2,183 FP32 coefficients (8,732 B), up to 1,816 matrix MACs per trigger, 64 B recurrent state per core, and a 512 B physical-line directory per core. Queues, score storage, exclusion masks, arbitration, and gate approximations add hardware cost.
- The input includes PC, translated address and cycle from accepted LLC ingress. All models use the same raw L2D miss stream and shared issue filter. Native policies may observe warmup. The learned policy starts its ingress history in ROI and waits for 128 observations before scoring addresses.

## Artifacts

- [Chinese LaTeX report source](../report/llc_lstm_report.tex). The compiled PDF stays with the local and Sacramento workspace copies. [All 40 frozen run rows](v5_results/per_phase.csv), [pooled counts](v5_results/policy_totals.csv), [timing and budget checks](v5_results/sensitivity.csv), [run signatures](v5_results/run_signatures.csv), and [launch/lifecycle counts](v5_results/learned_lifecycle.csv).
- [Pure NoPF simulator weights](v5_results/clean32/lstm_model.txt), [PyTorch checkpoint](v5_results/clean32/lstm_model.pt), and [Event-only weights](v5_results/clean32/event_model.txt). Raw training, calibration and replay outputs are under `lstm/runs/v5_clean/` and `conventional/runs/v5_clean/`. Matched older-model replays are under `conventional/runs/v5_coverage_replay/`.
- [Frozen model](v5_results/model.txt), [PyTorch checkpoint](v5_results/model.pt), [Event-only model](v5_results/event_model.txt), and [checkpoint selection](v5_results/best.json). The frozen runtime thresholds are Q10 50 and 912, respectively. The 4K offline decision cutoff in the checkpoint metadata uses a different selection metric.
- Raw demand, prefetch lifecycle, and simulator logs stay in `conventional/runs/v5_comparison/`. Training arrays, checkpoints, and per-epoch logs stay in `lstm/runs/v5/`. [Earlier negative controls](V4_DEVELOPMENT.md)

## Reproduce the frozen comparison

- From the research folder, build with `./conventional/build.sh 2 conventional/runs/build-v5-reproduce`. For each trace region, point the runner to the two identical slices of the 605 trace. Keep the trace basename `605.mcf_s-994B.champsimtrace.gz` to match the simulator seed.

```sh
for region in 310 330 350 370 390; do
  trace="lstm/runs/v4/trace_slices/${region}m/605.mcf_s-994B.champsimtrace.gz"
  python3 conventional/run.py \
    --binary conventional/runs/build-v5-reproduce/pythia/bin/champsim \
    --output "conventional/runs/v5_comparison/reproduced_${region}m" \
    --warmup 200000 --instructions 2000000 --degree 1 --events --prefetch-events \
    --policy nopf --policy next_line --policy stride --policy streamer --policy ampm --policy spp \
    --policy event --policy lstm --position-scorer \
    --event-model lstm/v5_results/event_model.txt --event-threshold 912 \
    --lstm-model lstm/v5_results/model.txt --lstm-threshold 50 \
    --prediction-latency 32 --prediction-interval 8 --prediction-depth 4 \
    --prediction-state-cycles 8 "$trace" "$trace"
done
```

- `lstm/v5_slots.py export` derives page slots and NoPF address labels. Train with `fit --slots 2 --relative --discrete-pc --unroll 8 --epochs 12 --latency 32`. Build the time distribution with `lstm/nopf_service.py` from the three training NoPF demand-service logs. Run a separate `--event-only` fit on the same arrays. The archived fit is available by explicitly passing `--historical-prefetch-calibration` and `lstm/runs/v2/service_130m_matched.json`.
- To run the pure NoPF frozen test, use the same runner arguments above with clean simulator weights, cutoffs −950/−650, the 410M/430M/450M trace slices from `lstm/runs/v6/trace_slices/`, and a new output path.
- The current V1 vs. Newest report draws its numeric plots from `report/v1_newest_data.tex`. Compile from `report/` with `tectonic -X compile llc_lstm_report.tex`. The source uses Helvetica Neue and PingFang SC on macOS.
