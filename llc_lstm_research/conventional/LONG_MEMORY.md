# Selecting a trace for longer history

- **Use `605.mcf_s-994B` first.** Its conventional prefetchers leave substantial LLC misses. Its NoPF stream also has a repeatable signal from older page visits. Use 620 as the short-context check. [SPEC trace archive](https://zenodo.org/records/10960004) · [Earlier screen](SPEC_SCREEN.md) · [620 history check](620_HISTORY.md)

## Conventional performance

- All six policies ran on the same dual-core trace regions with a 200,000-instruction warmup, 2 million measured instructions per core, private-cache prefetching disabled, and one issue per trigger at most.
- See [605 baseline results](605_RESULTS.md) for accuracy, timeliness, coverage, and IPC for every policy in all three regions.

| MCF region | NoPF LLC MPKI | Fastest issuing policy | Its LLC MPKI | ROI time saved |
| --- | ---: | --- | ---: | ---: |
| 10M | 32.742 | Streamer | 30.858 | 3.46% |
| 30M | 7.926 | AMPM | 7.582 | 1.72% |
| 50M | 52.188 | SPP | 46.880 | 4.04% |

- SPP saved 12.23–14.37% on `605.mcf_s-1554B` and 21.77–22.68% on `623.xalancbmk_s-165B`. On 619, the same older-page PC check at 250 decisions reached 132 misses with either an eight- or 32-event horizon.

## What older history contributes

- Run two copies of MCF with prefetching disabled. Count candidate outcomes from the 10M region and freeze the lookup before scoring the 30M and 50M regions. Each region uses a 200,000-instruction warmup and 2 million measured instructions per core.
- Every variant sees the same physical-page candidates, current PC, page offsets, candidate ages, and address rules. The pool uses the most recent visit to each other physical page in the previous 32 same-core events. Candidates are nearby lines within those pages that have yet to appear in the measured region. A positive reaches a first NoPF LLC miss with 300–2,000 cycles of ingress lead. One in four triggers is sampled at 10M and 50M. All events still update history.
- The table counts distinct future NoPF misses reached at **1,000 offline decisions**, with one candidate per trigger. Only the visibility of the older page's load PC changes. Shuffling reassigns PCs older than eight events among eligible pages at evaluation, while preserving their addresses and ages. This tests whether the correct page and PC association matters.

| Evaluation region | No source PC | Last 8 | Last 16 | Last 32 | Older PCs shuffled, range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30M | 63 | 115 | 201 | 210 | 33–38 |
| 50M | 12 | 88 | 166 | 353 | 27–36 |

- **Lead sensitivity.** With a 500-cycle minimum, last-eight and last-32 context reached 102 and 211 misses at 30M, and 78 and 337 at 50M. The cycle bounds are screening settings.
- **One concrete case.** In 50M on core 1, the last visit to a candidate page was 32 events before the trigger. The older load's PC was `4203584` at page offset 39. The trigger's PC was `4203599` on another page. The full-history lookup chose line offset 40 of the earlier page with 1,816 cycles of lead before the first NoPF demand missed. The last-eight lookup did not reach that demand in its top 1,000 decisions.
- **An ordinary table explains this gain.** All full-context keys selected at the 1,000-decision budget had already appeared in development. A separate check added the order of earlier visits to each candidate page and found no gain over sorting those visits. Whether recurrent state generalizes better than a page-PC table remains open.
- These are offline opportunities. They measure neither actual fill time nor IPC. The tested conventional LLC ports issue within the current page. A matched non-neural cross-page control is essential because the candidate space itself can improve coverage. [Proposed model and controls](../DESIGN.md)

## Reproduce the lookup

- Run from `llc_lstm_research`. Use `--eval-stride 4` for 50M and `--lead 500 2000` for the timing sensitivity. The full logs, CSVs, and exploratory `page_rank.py` order check are under `conventional/runs/long_memory/`. The `605_nopf_*` run directories retain their initial names and contain all six policies.

```sh
python3 conventional/long_memory_probe.py \
  conventional/runs/long_memory/605_nopf_10m \
  conventional/runs/long_memory/605_nopf_30m \
  --output conventional/runs/long_memory/605_binding_30m.csv
```
