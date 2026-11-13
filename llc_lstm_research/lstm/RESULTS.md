# 605 MCF LSTM results

## Training and offline test

- **Data.** 438,087 NoPF training events from 10M, 30M, and 50M, with 122,525 positive labels among 6,995,808 valid candidate labels. The 70M region selected the checkpoint and thresholds. The 90M and 110M regions were held out. [Setup](README.md)
- **Training.** Ran PyTorch CPU in Colab for four epochs per model. A local replication gave the same offline target counts. Validation selected the eight-event training unroll. All three LSTMs carried state at inference. The unroll controls gradient length. The selected checkpoints reached the last of the four epochs, so these results describe this training budget.
- **Metric.** Distinct prospective NoPF misses reached at 2,000 or 4,000 offline decisions. All arms had the same candidates and budget. Decisions were ranked by integer logits and were counted before simulator timing and queueing.

| Scorer | 70M validation, 2k / 4k | 90M test, 2k / 4k | 110M test, 2k / 4k |
| --- | ---: | ---: | ---: |
| Page-PC | 55 / 85 | 82 / 139 | 202 / 350 |
| Event-only | 82 / 121 | 74 / 123 | 225 / 380 |
| LSTM, 8-event unroll | 94 / 130 | 72 / 116 | 137 / 225 |
| LSTM, 16-event unroll | 94 / 128 | 70 / 116 | 136 / 224 |
| LSTM, 32-event unroll | 94 / 128 | 70 / 116 | 136 / 224 |

- **State helped this checkpoint.** Zeroing `h` under the same weights reduced 2,000-decision hits from 72 to 34 at 90M and from 137 to 58 at 110M. Resetting to a fresh eight-event window or shuffling its oldest six events yielded almost the same counts as carried state. The separately trained event-only head reached more misses than the LSTM on both held-out regions.
- **Score resolution.** Q10 changed the winning candidate for 249 of 89,962 eligible 90M triggers and 305 of 104,298 eligible 110M triggers. No scores saturated. The 2,000-decision counts differed from float by two and one misses, respectively. The selected LSTM's deficit persisted with float scores.
- **Distribution shift.** About 0.33% of candidates were positive in validation, compared with 0.89% at 90M and 1.45% at 110M. Global offline rankings hold the decision count fixed. A validation-derived runtime threshold can accept a different number of requests on a new region.

## ChampSim replay

- Replays use the same trace regions, cache geometry and instruction counts for all policies. The two learned policies and Page-PC share the candidate directory, eligibility checks, 1,000-request per-core cap and 16-cycle inference latency and initiation interval. Their thresholds are fixed using 70M validation. Native conventional baselines use their existing timing and the shared degree-one limit. Coverage counts direct first use matched to a NoPF miss.

### 90M held-out region

| Policy | Issued | Accuracy % | Timeliness % | Coverage % | Timely coverage % | LLC hit % | LLC miss % | MPKI | Aggregate IPC | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPF | 0 | – | – | 0.00 | 0.00 | 27.42 | 72.58 | 16.373 | 0.54761 | 1.8261 |
| Next-Line | 59,873 | 44.69 | 33.90 | 38.94 | 13.58 | 36.58 | 63.42 | 14.305 | 0.54208 | 1.8447 |
| Stride | 907 | 98.24 | 100.00 | 1.36 | 1.36 | 28.60 | 71.40 | 16.143 | 0.55109 | 1.8146 |
| Streamer | 16,462 | 43.01 | 74.08 | 10.68 | 7.96 | 33.49 | 66.51 | 15.055 | 0.55906 | 1.7887 |
| AMPM | 7,572 | 47.99 | 95.29 | 5.50 | 5.25 | 30.95 | 69.05 | 15.593 | 0.55772 | 1.7930 |
| SPP | 25,086 | 61.05 | 48.94 | 22.85 | 11.37 | 36.27 | 63.73 | 14.363 | 0.56319 | 1.7756 |
| Page-PC | 369 | 99.46 | 99.46 | 0.56 | 0.56 | 28.39 | 71.61 | 16.165 | 0.54995 | 1.8184 |
| Event-only | 1,389 | 40.17 | 99.82 | 0.85 | 0.85 | 28.23 | 71.77 | 16.218 | 0.54821 | 1.8241 |
| LSTM | 2,000 | 24.40 | 98.57 | 0.75 | 0.73 | 27.56 | 72.44 | 16.305 | 0.54417 | 1.8377 |

### 110M held-out region

| Policy | Issued | Accuracy % | Timeliness % | Coverage % | Timely coverage % | LLC hit % | LLC miss % | MPKI | Aggregate IPC | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPF | 0 | – | – | 0.00 | 0.00 | 37.16 | 62.84 | 16.431 | 0.48121 | 2.0781 |
| Next-Line | 64,522 | 43.72 | 24.53 | 39.70 | 9.85 | 42.58 | 57.42 | 14.858 | 0.48399 | 2.0662 |
| Stride | 264 | 98.48 | 100.00 | 0.39 | 0.39 | 36.25 | 63.75 | 16.510 | 0.48130 | 2.0777 |
| Streamer | 13,244 | 40.00 | 73.31 | 7.99 | 5.88 | 40.91 | 59.09 | 15.387 | 0.49412 | 2.0238 |
| AMPM | 5,789 | 51.11 | 95.34 | 4.49 | 4.28 | 40.21 | 59.79 | 15.518 | 0.49277 | 2.0293 |
| SPP | 19,997 | 57.35 | 44.01 | 17.07 | 7.57 | 43.22 | 56.78 | 14.708 | 0.49625 | 2.0151 |
| Page-PC | 444 | 97.30 | 92.13 | 0.66 | 0.61 | 38.58 | 61.42 | 16.066 | 0.48429 | 2.0649 |
| Event-only | 1,607 | 51.09 | 95.25 | 1.25 | 1.19 | 40.72 | 59.28 | 15.431 | 0.48924 | 2.0440 |
| LSTM | 2,000 | 21.85 | 90.62 | 0.66 | 0.60 | 38.35 | 61.65 | 16.051 | 0.48372 | 2.0673 |

### What changed in replay

- **Accuracy and coverage.** Most used LSTM requests arrived on time. Its 21.85–24.40% accuracy was lower than both matched scorer controls. It covered under 0.75% of NoPF misses on each test. Its 300–2,000 cycle label and 16-candidate directory together could reach 5.32% and 5.74% of baseline misses even with perfect scoring. The offline labels and actual timely fills measure different events.
- **Conventional misses.** On 90M and 110M respectively, the same candidates and label could reach 144 of 1,786 and 49 of 1,388 NoPF misses that Streamer covered late. For SPP the counts were 323 of 7,517 and 200 of 6,242. These are candidate reach counts from NoPF. Conventional outcomes only identify the diagnostic subset.
- **Label sensitivity.** A NoPF-only diagnostic lowered the minimum lead from 300 to 150 cycles. Reach rose from 5.32% to 10.90% at 90M and from 5.74% to 8.61% at 110M. Raising the maximum lead to 4,000 cycles reached 6.12% and 6.02%. Raising the candidate cap to 64 with the original lead window reached 12.00% and 9.80%. These counts isolate opportunities. The shorter lead still has to cover inference, queueing, and fill latency, and a wider head costs more work per event.
- **Completion time.** LSTM was 0.77% slower than NoPF at 90M and 0.52% faster at 110M. Event-only finished faster than LSTM on both regions. SPP was the fastest policy in both full replays. At 90M, LSTM reduced MPKI slightly but transferred 2.67% more DRAM bytes and took longer than NoPF. LLC hit rate alone missed that outcome.
- **Equal request counts.** A second replay used the same thresholds and 150 accepted requests per core. All three scorers issued exactly 300 per region. Page-PC had the highest accuracy, coverage, and IPC in both regions. LSTM covered 79 and 42 baseline misses at 90M and 110M. Event-only covered 166 and 109. LSTM and Event-only finished within 0.03% of each other on both tests. LSTM finished 0.35% and 0.10% slower than Page-PC.
- **Latency sensitivity.** Moving from 16 to 64 cycles lowered LSTM timeliness from 98.57% to 97.31% at 90M and from 90.62% to 87.56% at 110M. The predictor dropped 268 and 115 events at 64 cycles, compared with zero at 16. LSTM still finished behind NoPF at 90M and both scorer controls at 110M. The 90M LSTM time moved from 1.8377 to 1.8327 ms as queue and cache traffic also changed.
- **Scope.** Both cores run independent copies of the same MCF region. The experiment measures shared LLC contention for that pair. Shared-state features, mixed workloads, and core-count scaling remain open.

- **Artifacts.** The Colab checkpoint, training metrics, and logs are in `lstm/runs/experiments/mcf_605_colab/` on both the local and Sacramento copies. The corresponding ChampSim logs and demand traces are in `conventional/runs/lstm_605/colab_*`.
