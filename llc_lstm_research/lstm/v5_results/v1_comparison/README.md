# Original Colab V1 vs newest pure-NoPF LSTM

## Comparison protocol

- Replay the original exported Colab V1 and the newest pure-NoPF checkpoint with the same ChampSim binary, two identical 605 MCF trace slices, 200,000 warmup instructions per core and 2,000,000 measured instructions per core. Both use a 32-cycle prediction latency, 8-cycle start and state intervals, depth 4, and at most one accepted request per prediction opportunity. The original publication used 16-cycle V1 inference. The V1 numbers here come from a new replay with the common 32-cycle timing.
- The checkpoint and Q10 cutoff stay fixed across budget conditions. V1 uses [the original Colab weights](v1_model.txt) at −1037. Its four-epoch training chose the eight-event unroll and used hard 300–2000-cycle NoPF labels. The [newest weights](../clean32/lstm_model.txt) use −950. The eight-epoch checkpoint uses future NoPF misses within 4,000 cycles with a soft timing weight from 369,307 completed NoPF demand services and 32 cycles reserved for prediction. See the [frozen setup](../clean32/frozen32.json).
- **No extra ROI cap** is recorded as `native` in the CSVs. It leaves the model's score threshold, one-request degree, cache checks, duplicate filter, and queue limits active. **1,000/core** and **150/core** use the same weights and threshold and stop accepting requests as each core reaches its cap. These are replay settings.
- The 90M and 110M slices are the original V1 test regions. The 410M, 430M and 450M slices are the newest model's frozen test regions. Coverage is relative to the same NoPF replay for both versions in every region.
- **Coverage** counts the fraction of NoPF LLC demand misses directly served by a prefetch, including a late first merge. **Timely coverage** counts only timely first use. **Accuracy** is the fraction of accepted requests that have a first use, timely or late. **Timeliness** is the timely share of those useful requests. LLC hit and miss rates divide the corresponding event counts by demand LLC accesses. MPKI divides demand LLC misses by retired instructions and multiplies by 1,000. Pooled ratios sum counts or cycles before dividing. Times in the CSV are simulated seconds.

## What the comparison shows

- **90M and 110M, exactly 2,000 requests per model per region.** Pooled coverage rose from 0.710% to 2.641% and timely coverage from 0.669% to 1.331%. The newest model's IPC was slightly lower at 90M and higher at 110M. Pooled completion time differed by 0.007%. The newest model served more NoPF misses timely while a smaller share of all useful requests was timely.
- **All four equal-volume regions at 1,000/core.** Across 90M, 110M, 430M and 450M, pooled coverage rose from 0.578% to 1.403%. The newest model took 0.126% longer. The 410M pair has a common cap and different accepted counts, so it is excluded from that row.
- **90M and 110M, no extra cap.** Pooled coverage rose from 0.769% to 25.938% and time fell 4.58%. The newest model sent 43,014 requests versus 5,607 for V1.
- **410M, 430M and 450M, no extra cap.** Coverage rose from 1.269% to 24.308% and pooled completion time fell 6.60%. The newest model sent 153,362 requests versus 12,097 and used 24.196 MB of DRAM traffic versus 19.582 MB.
- **Fresh test runs with an issue cap.** At 1,000/core, V1 sent 1,571 requests at 410M and the newest model sent 2,000. The actual count was 2,000 for both versions at 430M and 450M. At 150/core, both versions reached exactly 300 requests in every region. On the three fresh regions at that cap, matched coverage rose from 288 to 613 of 294,753 NoPF misses, while pooled time rose 0.080%. The newest model covered 112 misses against V1's 125 at 430M.

## Newest vs. conventional prefetchers

- The same 410M, 430M and 450M slices include NoPF, Next-Line, Stride, Streamer, AMPM and SPP. All use the released issue rules with no extra total-count cap. Newest beats the fastest conventional policy on each region, taking 7.300 ms total versus 7.517 ms and using 3.15% more DRAM bytes. Pooled coverage is 24.31% for Newest and 26.37% for Next-Line. Newest reaches 17.97% timely coverage versus Next-Line's 10.56%. [Per-region measurements](../clean32/per_phase.csv) · [Pooled measurements](../clean32/pooled.csv)

## Data and reproduction

- [All per-region measurements](per_region.csv) and [pooled counts and metrics](pooled.csv). Each per-region row includes the actual number accepted by each core, the NoPF denominator, direct and timely coverage numerators, and paths to the raw simulator runs. Use the `exact_count_regions` pooled cohort when matching actual acceptance counts at 1,000/core across both test sets.
- [One live address and LSTM example](example_410m.json) lists the PC, line positions, candidate scores, one gate-state update, actual issue and fill cycles, and the matching label from a separate NoPF replay.
- Run `python3 lstm/v5_results/v1_comparison/replay.py` from the `llc_lstm_research` folder with the existing local simulator and trace slices. Then run `python3 lstm/v5_results/v1_comparison/collect.py` to verify the configuration and join the logs. Commands and raw logs are stored below `conventional/runs/v1_newest_replay/`. Joining uses the run's existing same-host NoPF reference checks.
- Generate the two-version figure data with `python3 report/make_v1_newest_figures.py`. The [current LaTeX report](../../../report/llc_lstm_report.tex) presents Newest and the conventional results. The compiled PDF stays on the local and Sacramento copies.
