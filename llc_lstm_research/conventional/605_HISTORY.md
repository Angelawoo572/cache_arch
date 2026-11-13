# Does intervening history help on 605 MCF?

- **Test region.** Used `605.mcf_s-994B` after skipping 70 million trace records for the first time. Both cores ran NoPF for 2 million measured instructions each. The test had 82,603 LLC demand events and 20,036 LLC misses. Counts came from the 10M region and were frozen before scoring 70M.
- **Fair comparison.** Every lookup used the same 20,586 eligible triggers, the same candidate addresses and page-PC fields, and at most one candidate per trigger. The candidate directory kept pages from the last 32 events in every arm. Only the scoring context changed. All history updates used the full stream, with one in four triggers sampled for scoring in each region.
- **What changed.** The page-PC lookup uses the current PC and offset, the candidate's source PC and offset, the line displacement, and age. The history lookups add a compact summary of events strictly between the source visit and trigger. An 8-, 16-, or 32-event window includes the trigger. The shuffled controls reorder the older intervening tokens and preserve the most recent token and the page-PC fields. Each shuffle uses its own development counts. There were three seeds.
- **What counts.** A decision reaches a target if the candidate's first demand in the measured region is a NoPF LLC miss with 300–2,000 cycles of ingress lead. Counts deduplicate target misses. The 1,000 and 2,000 budgets count offline decisions, before actual cache and queue filtering.

| Scoring context | Distinct misses at 1,000 decisions | Distinct misses at 2,000 decisions |
| --- | ---: | ---: |
| Page-PC table | 213 | 308 |
| Ordered 8 | 226 | 334 |
| Shuffled 8 | 213–222 | 328–345 |
| Ordered 16 | 208 | 337 |
| Shuffled 16 | 202–222 | 324–334 |
| Ordered 32 | 203 | 328 |
| Shuffled 32 | 211–217 | 320–330 |

- **What the result supports.** At 1,000 decisions, ordered 8 reaches 60 new targets and loses 47 the table found. At 2,000 decisions, shuffled 8 reaches as many as 345 misses. The extra context helps some decisions. The advantage from preserving order varies with window and budget, and the 32-event scorer has no consistent advantage.
- **Older page information still matters.** On this same region, keeping the candidate pool fixed and exposing source PCs up to 32 events back gives 213 misses at 1,000 decisions, up from 38 when source PCs are limited to eight events. This is a reason to keep the 32-event directory. It leaves open how much sequence state is useful.
- **Lead sensitivity.** With a 500-cycle minimum, ordered 8 reaches 224 versus 202 for page-PC at 1,000 decisions, and 365 versus 309 at 2,000. Shuffled 8 reaches 197–210 and 352–354. This supports starting at eight events. Window and timing choices still need a separate final region.
- **Design consequence.** Keep the page-PC table and the short history lookup as controls. A future LSTM must improve on both with the same directory and issue budget. Use a small candidate cap and one candidate-dependent score. Start with the shortest tested sequence context that showed a gain. [LSTM design](../DESIGN.md)
- **Scope.** The test used frequency counts and a compact history summary. The lead window is a screening assumption. Actual fills, interference, and IPC require closed-loop replay. No LSTM was implemented or trained.

## Reproduce

- Run from `llc_lstm_research`. The CSVs and 70M simulator log are under `conventional/runs/long_memory/`. Pass `--lead 500 2000` for the timing sensitivity.

```sh
python3 conventional/intervening_probe.py \
  conventional/runs/long_memory/605_nopf_10m \
  conventional/runs/long_memory/605_nopf_70m \
  --output conventional/runs/long_memory/605_intervening_70m.csv
```
