# 620 OMNeT++ history check

- **A short cue repeats.** On core 0, the same two load PCs (`5152793`, `5152797`) with deltas `+1,+1` sometimes precede a useful `+2` prefetch. When the preceding event had delta `+2`, the candidate reached a NoPF miss in 21/21 anchors at 10M and 15/15 at 30M. Both Streamer and SPP were late on 18/21 and 14/15, respectively. When that preceding event was a page change, the candidate had zero usable misses in 24 anchors in each region.
- **Two requests with the same recent history.** At 30M, core 0 instructions `502305` and `1154595` had the same last two PCs, page offsets (`52,53`), and deltas (`+1,+1`). The preceding PC and page offset also matched. Its delta was `+2` for the first request and a page change for the second. From the first request, `+2` reached instruction `502394` with 584 cycles of NoPF lead. The actual Streamer and SPP runs still had 584 and 576 cycles from the matching anchor to demand ingress, respectively. Both marked the demand late. The second request had no later `+2` demand in the region. This cue fits in three events.

## Equal-budget comparison

- Counted candidate outcomes at 10M and froze the lookup before scoring 30M. Each variant saw the same 24,023 eligible trigger events and legal same-page offsets `+1,+2,+3,+4,+6`. Inputs were load PC and physical line, encoded as current page offset and clipped deltas with a page-change token. Past events used PC and delta. Cycle counts defined the outcome only. A candidate was positive if its next same-line demand missed in NoPF with 300–2,000 cycles of ingress lead.
- Ranked exactly 1,000 decisions per variant, with at most one candidate per trigger. Ties used a common seed. The shuffled control keeps the last two events in place and shuffles older tokens as units. It averages five seeds. The sorted control uses the same older tokens in sorted order. Windows of one to three events have nothing to reorder. Counts are distinct demands matched by core, instruction, and load operand.

| History length | Ordered, Streamer late | Ordered, SPP late | Ordered, either late | Shuffled, either late | Sorted, either late |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 186 | 212 | 257 | — | — |
| 2 | 153 | 165 | 206 | — | — |
| 3 | 149 | 165 | 206 | — | — |
| 4 | 139 | 157 | 196 | 202.2 | 196 |
| 8 | 137 | 161 | 198 | 202.2 | 199 |
| 16 | 137 | 161 | 198 | 202.2 | 199 |
| 32 | 137 | 161 | 198 | 202.2 | 199 |

- **The short cue did not become an overall gain.** With the ordinary NoPF label, the one-event lookup reached the most baseline-late misses at 1,000 decisions and at 2,000 decisions. Counting only baseline-late positives at 10M gave a useful short-history control. At 1,000 decisions, two events reached 348 distinct late misses versus 312 with one. At 2,000 decisions, four ordered events reached 539 versus 504 with one and 539 with four sorted events.
- **Long order remains unproven.** Only 17 of 24,023 evaluation decisions had at least five development matches for their chosen candidate's exact ordered eight-event key. None did at 16 or 32 events. The lookup backed off to shorter keys. The 500-cycle minimum and the delta-only history check also showed no consistent advantage for order. The strongest evidence so far supports testing a compact two- or three-event rule before committing to a longer recurrent state.
- **Timing check.** At 30M there were 1,035 Streamer-late and 1,605 SPP-late NoPF misses. The five offsets could reach 798 and 1,060 of them before any issue budget. The same anchor and offset also met the lead window on each baseline's timeline for 782 and 1,051 targets, respectively.
- **Scope.** This offline ranking measures lead time and possible targets. Prefetch issue timing, cache and queue eligibility, and actual fills would require closed-loop replay. Both cores replay the same trace. Both regions have been used for development. No LSTM was implemented or trained.

## Reproduce

- Run from `llc_lstm_research` with `conventional/history_probe.py`, passing the 10M and 30M coverage run directories and `--output` for the CSV. Use `--late-only`, `--lead 500 2000`, or `--deltas-only` for the checks above. Full 1,000- and 2,000-decision counts, per-policy timing rechecks and key support are under `conventional/runs/spec_screen/620_history/`.

```sh
python3 conventional/history_probe.py \
  conventional/runs/spec_screen/620_coverage_10m \
  conventional/runs/spec_screen/620_coverage_30m \
  --output conventional/runs/spec_screen/620_history/pc_delta_300.csv
```
