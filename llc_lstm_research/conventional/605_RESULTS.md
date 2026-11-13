# Two-core 605 MCF baselines

- Both cores replayed `605.mcf_s-994B` with private-cache prefetchers disabled. Every policy used 200,000 warmup instructions and 2 million measured instructions per core. The shared LLC filter allowed at most one prefetch per eligible demand event.
- **Accuracy** is used / sent. **Timeliness** is on-time / used. **Coverage** is the share of NoPF LLC demand misses directly served by a prefetch. A late first merge counts toward coverage, and **timely coverage** counts only timely first hits. **Aggregate IPC** divides total retired instructions by the last core's finish cycle. [Metric definitions](../PERFORMANCE.md)

## Skip 10 million trace records

| Policy | Accuracy | Timeliness | Coverage | Timely coverage | Aggregate IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | — | — | 0.00% | 0.00% | 0.3397 |
| Next-Line | 46.54% | 21.06% | 41.01% | 8.73% | 0.3418 |
| Stride | 96.21% | 100.00% | 0.15% | 0.15% | 0.3405 |
| Streamer | 49.87% | 63.35% | 9.66% | 6.14% | 0.3519 |
| AMPM | 57.13% | 92.47% | 4.59% | 4.24% | 0.3497 |
| SPP | 65.62% | 25.06% | 38.29% | 9.69% | 0.3515 |

## Skip 30 million trace records

| Policy | Accuracy | Timeliness | Coverage | Timely coverage | Aggregate IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | — | — | 0.00% | 0.00% | 0.7000 |
| Next-Line | 46.43% | 29.88% | 40.83% | 12.22% | 0.7061 |
| Stride | 96.30% | 100.00% | 0.66% | 0.66% | 0.7011 |
| Streamer | 40.68% | 59.62% | 12.56% | 7.49% | 0.7119 |
| AMPM | 55.84% | 93.81% | 4.74% | 4.44% | 0.7123 |
| SPP | 56.79% | 46.37% | 20.27% | 9.40% | 0.7122 |

## Skip 50 million trace records

| Policy | Accuracy | Timeliness | Coverage | Timely coverage | Aggregate IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | — | — | 0.00% | 0.00% | 0.2091 |
| Next-Line | 26.87% | 23.79% | 20.33% | 5.17% | 0.2054 |
| Stride | 98.86% | 100.00% | 0.90% | 0.90% | 0.2101 |
| Streamer | 47.09% | 74.43% | 10.57% | 8.04% | 0.2157 |
| AMPM | 56.64% | 94.49% | 5.32% | 5.04% | 0.2144 |
| SPP | 57.90% | 43.54% | 22.19% | 10.50% | 0.2179 |

## What the baselines leave open

- **Stride** was on time for all used requests, with under 1% coverage in every region. **AMPM** also arrived on time often, with 4.59–5.32% coverage.
- **SPP** had the highest coverage among the four, but arrived late on more than half of its used requests in every region. **Streamer** used fewer requests and arrived on time more often. The highest IPC came from Streamer at 10M, AMPM at 30M, and SPP at 50M.
- Coverage matched requests by core, instruction, and load operand. At least 99.49% of NoPF misses matched in every arm. Counts credit direct first use. Raw summaries, per-core IPC, and `coverage.csv` are under `conventional/runs/long_memory/605_nopf_{10,30,50}m/`.
