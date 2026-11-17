# 605 LLC LSTM v2

- **Training data.** Fit the address table and model on 10M, 30M, and 50M NoPF. Select checkpoints and score cutoffs on 70M. Use 70M and 130M for fill-time calibration. Final tests are fresh 150M, 170M, and 190M regions of `605.mcf_s-994B`. The two cores replay independent address spaces. All runs use 200,000 warmup and 2 million measured instructions per core.
- **Address proposals.** For each trigger, use the current page and previously seen physical pages. Fit offsets from the later first same-core NoPF misses observed for combinations of current PC, source PC, line position, and source age. Choose at most four offsets per context and 16 candidates per trigger. No conventional policy's output is used as supervision. The learned offsets span −11 to +10. Forty-five of 256 stored offsets are outside v1's ±6 range.
- **Time as input.** Replace the redundant same-line bit with log-bucketed ingress gap. Replace candidate age in events with log-bucketed elapsed cycles. Keep seven event features, six candidate features, an eight-unit LSTM, and a scalar output.
- **Time as supervision.** Compare the first future NoPF miss within 4,000 cycles, the original 300–2,000-cycle binary label, and a soft label. The soft label is zero for no future miss and otherwise uses `P(measured trigger-to-fill time ≤ NoPF lead)`. It represents readiness under the development service distribution. Cache residency and exposed CPU stall need closed-loop measurement.

## What the data measured

- **Same candidate cap, same label.** At 300–2,000 cycles, candidate reach on the three tests grew from **7.37%, 6.23%, 10.35%** to **10.88%, 12.72%, 13.48%**. In the 70M/130M development control, using learned offsets on old pages alone increased reachable counts from 738/896 to 1,273/1,376. Allowing the current page brought them to 1,511/1,609.
- **Service time.** Of 1,738 development prefetches sampled at a modeled 16-cycle predictor latency, 1,735 filled. Trigger-to-fill p50/p90/p95 were 226/318/390 cycles. The empirical fraction filled by 150/300/500 cycles was 11.93%/89.22%/99.08%. Among used live requests that matched the trigger and same-core NoPF miss in the three tests, timely fractions by NoPF lead were 4.23% at 0–149 cycles, 45.54% at 150–299, 79.75% at 300–500, and 97.84% at 501–2,000. The latter is conditional on first use in the replay.
- **Observation horizon.** At the same candidates and trigger cutoff, the 4,000-cycle horizon covers over 99% of the distinct misses reachable by 32,000 cycles on both development regions. The fill-weighted oracle reaches 93.7% and 92.7% of its 32,000-cycle count. Use 4,000 as a finite labeling horizon for this experiment. Longer-lead opportunities still exist.
- **Training longer.** Allow 25 epochs with a patience of five and keep epoch 4 for reference. With the common future-miss ranking metric, the unweighted binary-label arms peaked at epoch 1. The calibrated LSTM peaked at **epoch 7** and ran through epoch 12. Training length and BPTT horizon add no inference state or coefficients.
- **Repeated targets.** In the three training regions, 552,156 candidate positives under the future-miss label map to 159,340 distinct miss events. Within each split, an inverse-frequency positive weight improves distinct misses in the top 4,000 LSTM decisions from **1,593/1,255/385** to **3,307/2,323/1,123**. The calibrated label gives a separate on-time signal.
- **Recurrence.** With a future-only label, increasing the training unroll from 8 to 32 barely changes the 2,000-decision results. Reset/shuffle controls generally track one another. The calibrated epoch-7 checkpoint has a carried-state benefit over eight events at 170M, while zeroing its state wins in the 150M and 190M proxy comparison. There is no consistent benefit from older event order.

## Hardware accounting

- **Scorer.** Keep 611 fused float32 coefficients (2,444 bytes), 64 bytes of `h,c` per core, and at most 16 scored candidates. Fold the candidate's affine key and bias into six per-event weights and one bias. The same real-valued function costs at most **664 matrix MACs per event** in C++, down from **1,084**. Float operation ordering can slightly change Q10 rounding. At 70M, 344 of 1,317,472 valid scores differ by one unit and no chosen candidates change.
- **Generator.** Each of the 32 history entries remains 16 bytes. It stores a 64-bit line, a 32-bit PC, and a 32-bit cycle value. The 64-context table adds **782 packed data bytes**, plus implementation-dependent lookup/ranking logic. A 32-context, 388-byte table differs by four and three targets on development at the same label. Its trained IPC has yet to be evaluated.
- **Timing.** Model an asynchronous 16-cycle decision and use 64 cycles for sensitivity with a 16-cycle configured minimum interval. The current lane permits one in-flight decision, so a longer decision can also lower effective throughput. The table lookup and selection need a real implementation to establish area and cycle cost. Neither measured simulation time nor MAC count establishes equal total hardware cost.

## Replay setup

- **Budget.** Allow up to ten accepted sends per core per 20,000 retired instructions, with no carry between windows. Learned model thresholds come from the 4,000-decision validation point. This evens out the opportunity to send and can produce different accepted counts. SPP is a reference policy and supplies no training data.
- **Trace equivalence.** Replay pre-extracted four-million-record slices with the original trace basename to preserve Pythia's address seed. In all three final regions, the NoPF logs match the original uncut replay byte-for-byte and have identical completion cycles.

| Policy, 16 cycles for neural scores | 150M IPC | 170M IPC | 190M IPC |
| --- | ---: | ---: | ---: |
| NoPF | .35124 | .55945 | .56457 |
| SPP, native degree-one | .36307 | .57391 | .56485 |
| SPP, paced | .35198 | .55943 | .56584 |
| v1 LSTM, paced | .35222 | .56008 | .56646 |
| v2 LSTM, binary timed | .35109 | .55976 | .56286 |
| v2 LSTM, future miss | .35147 | .55966 | .56420 |
| v2 LSTM, future miss weighted by target | .35128 | .55989 | .56357 |
| v2 Event-only, calibrated | .35127 | .55992 | .56423 |
| v2 LSTM, calibrated | .35222 | .56123 | .55771 |
| v2 LSTM, calibrated and weighted by target | .35121 | .55957 | .56229 |

- **Actual timely fills.** The calibrated LSTM issued 1,228/388/723 requests, with 845/311/213 timely uses. The future-miss model weighted by target recorded 209/62/103 timely uses. Calibrated LSTM direct coverage was 0.52%/0.76%/0.46%, of which 0.52%/0.75%/0.45% was timely.
- **Completion time.** The calibrated LSTM was 0.28% and 0.32% faster than NoPF at 150M and 170M, and about 1.23% slower at 190M. The 190M run raised demand MPKI from 10.486 to 10.691 and transferred 3.69% more DRAM bytes. These counters show a performance regression despite timely use. Individual pollution and stall costs were not attributed.
- **Latency sensitivity.** At 64 cycles, the same calibrated LSTM reached IPC .35300/.56108/.55956. There is still a 190M regression. SPP with its native budget remains faster than the calibrated LSTM in each region. The comparison does not establish a general v2 replacement for the incumbent.

## Reproduce

- Run from `llc_lstm_research`. Collect the NoPF logs with `conventional/run.py --events --policy nopf --instructions 2000000` and the skip values listed above. Fit `python3 lstm/v2_data.py fit TABLE TRAIN_RUN...` on the three training run directories. Export each split with `python3 lstm/v2_data.py export RUN NPZ --table TABLE`.
- Use `v2_train.py --train ... --validation ... --assessment ... --test ... --output ...`. For binary objectives use `--labels timed future`. For measured timing use `--labels soft --calibration SERVICE_JSON --selection calibrated`. Add `--state-ablation` for the fixed-weight controls. Select the 4,000-decision cutoff in validation when loading the corresponding `.txt` in ChampSim.
- In `conventional/run.py`, select `--correlation-table TABLE` with the v2 models, `--pace-instructions 20000 --pace-requests 10`, and `--prefetch-events` for lifecycle events. Use `v2_timing.py LOGS --output SERVICE_JSON` to derive the development CDF, or add `--baseline-demand` to join first uses to a matched NoPF log.
- Local runs, packed-table estimates, calibration data, held-out metrics, and checkpoints are under `lstm/runs/v2/`. ChampSim demand and prefetch lifecycle logs are under `conventional/runs/{lstm_v2,v2_comparison}/`. Training and replay ran locally.

## Sources

- PC and page/offset association motivates the proposal representation. [Voyager](https://research.google/pubs/a-hierarchical-neural-model-of-data-prefetching/) · [Twilight](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)
- Memory-level parallelism explains why latency saved on one miss can have a different performance effect from another. [MLP-aware cache replacement](https://hps.ece.utexas.edu/pub/qureshi_isca06.pdf)
