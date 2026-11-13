# Two-core 620 OMNeT++ baselines

- Both cores replayed `620.omnetpp_s-874B` with private-cache prefetchers disabled. Each arm used the same trace start, 200,000 warmup instructions and 1,000,000 measured instructions per core. The shared LLC filter allowed one issue per eligible demand event.
- **Accuracy** = used / sent. **Timeliness** = on-time / used. **Coverage** = NoPF misses directly served by a prefetch / all NoPF misses. A late first merge counts as use and coverage. **Timely coverage** counts timely first hits only. IPC is aggregate instructions divided by the last core's finish cycle. [CMU 18-447 metric definitions](../PERFORMANCE.md)

## Skip 10 million trace records

| Policy | Accuracy | Timeliness | Coverage | Timely coverage | Aggregate IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | — | — | 0.00% | 0.00% | 0.4709 |
| Next-Line | 36.11% | 60.11% | 30.57% | 18.38% | 0.4870 |
| Stride | 77.05% | 100.00% | 0.95% | 0.95% | 0.4725 |
| Streamer | 50.30% | 54.59% | 11.41% | 6.23% | 0.4757 |
| AMPM | 39.62% | 75.58% | 6.58% | 4.97% | 0.4733 |
| SPP | 49.39% | 55.16% | 17.41% | 9.61% | 0.4778 |

## Skip 30 million trace records

| Policy | Accuracy | Timeliness | Coverage | Timely coverage | Aggregate IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | — | — | 0.00% | 0.00% | 0.4672 |
| Next-Line | 35.28% | 58.72% | 30.32% | 17.81% | 0.4835 |
| Stride | 76.52% | 100.00% | 0.88% | 0.88% | 0.4687 |
| Streamer | 49.69% | 53.59% | 11.15% | 5.97% | 0.4720 |
| AMPM | 37.25% | 74.28% | 6.63% | 4.92% | 0.4697 |
| SPP | 46.29% | 53.22% | 17.15% | 9.13% | 0.4737 |

## Where each policy falls short

- **Stride** issued only 244 requests in the first region. All 188 used requests were timely, and coverage stayed below 1% in both regions.
- **Streamer and SPP** arrived late on about 45% of used requests. SPP had the highest IPC of the four in both regions.
- **AMPM** had no qualifying use for 1,984 of its 3,286 issued requests by the end of the first region. Those lines had already filled. The run may end before a later use.
- **Next-Line** had the highest IPC of all policies shown and transferred about 44% more DRAM bytes than NoPF. Match actual traffic as well as degree in a future model comparison.
- Coverage matches the same demand by core, instruction, and load operand. At least 99.98% of NoPF misses matched a lookup in each arm. The measure gives credit for direct first use on the matched demand. Indirect cache effects and later consumers are outside its scope.
- Both runs retain per-core IPC, raw counts and logs under `conventional/runs/spec_screen/620_coverage_10m/` and `620_coverage_30m/`. [Reproduce the runs](README.md)

## Does order help?

- The [late-miss history check](620_HISTORY.md) matches Streamer and SPP late misses and compares 1–32 events at a fixed budget. The earlier check below used all future misses.
- A frequency lookup built from 10M NoPF events ranked `+1,+2,+4` at 30M. It used the current PC, page offset, and line delta. Two variants added the preceding three deltas in their original order or sorted them. Deltas were clipped to ±8 with a separate page-change value. Each rate was smoothed with 20 pseudo-counts from the simpler key's rate.
- At the same 1,000-decision budget, the three versions reached **371** distinct prospective misses with the current input, **448** with ordered history, and **455** with sorted history. The check uses the illustrative 300–2,000 cycle lead range and at most one candidate per event. It is an offline lookup without service, cache, or inference timing. Extra context helped this lookup. The ordered version showed no gain over sorting, so a benefit from recurrence is still an open question.
