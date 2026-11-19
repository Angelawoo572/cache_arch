# Does two-stride lookahead transfer to a full evaluation interval?

An earlier 500K measured window in both GAP PageRank and SPEC06 Soplex favored a two-stride lookahead. This follow-up fixes those two previously withheld programs, the existing native `look2` policy and the primary 50M warmup / 200M measured interval. Selecting these programs after the short comparison makes this a test of transfer to a different interval of each recording. Their short-window outcomes were already known.

## Measured transfer

Two-stride lookahead increased timely coverage in both long intervals. Its earlier time advantage persisted in Soplex and reversed in GAP PageRank. Both long `look2` runs took more cycles than NoPF. Positive time saved in this table means fewer cycles than the reference.

| Program | Short 500K time saved vs Stride | Full 200M time saved vs Stride | Full time saved vs structural rule | Full time saved vs NoPF | Original timely coverage, Stride → look2 |
|---|---:|---:|---:|---:|---:|
| GAP PageRank-3 | +2.444% | −0.235% | −0.236% | −1.253% | 12.237% → 15.104% |
| SPEC06 Soplex-92B | +3.016% | +0.497% | +0.647% | −1.206% | 21.089% → 26.147% |

PageRank gave the clearest counterexample to using timely hits as a performance proxy. Of 15,678,522 original misses, 520,359 operands gained a direct timely use relative to Stride, including 486,166 that had merged late. Another 70,870 lost a direct timely use. Despite this net gain of 449,489, all four 50M quarters took more cycles. Total DRAM read traffic rose from 1,057,446,400 to 1,060,243,328 bytes and unique ROB-head wait increased by 10,278,968 cycles. These are measured associations in the two executions. They do not identify the marginal cost of a particular prefetch.

For Soplex, 182,432 operands gained timely uses and 81,113 lost them against Stride, from 2,003,166 original misses. Two-stride lookahead saved time in each quarter, between 0.329% and 0.597%. The 200M result supports keeping this inexpensive lookahead as a control for Soplex. A claim of complementary learned value still needs a candidate that improves completion time relative to the best policy for the tested interval, including NoPF.

| Full comparison | Direct timely gained | Direct timely lost | Net timely change | Reference late → look2 timely |
|---|---:|---:|---:|---:|
| PageRank vs Stride | 520,359 | 70,870 | +449,489 | 486,166 |
| PageRank vs structural | 557,197 | 70,920 | +486,277 | 460,840 |
| Soplex vs Stride | 182,432 | 81,113 | +101,319 | 120,476 |
| Soplex vs structural | 184,456 | 81,988 | +102,468 | 114,628 |

Each comparison covers the same retired dynamic operands within a complete 200M ROI. Retiring the entire 500,000,001-instruction PageRank recording is a separate experiment on the main panel. [Full fixed-work measurements](performance.csv) · [Paired counts and quarter results](paired.csv) · [Prior 500K diagnostic](../discovery/diagnostic_performance.csv)

## Method and reproduction

After a native PC-based Stride match, `look2` proposes the current physical cache line plus twice the signed stride. It makes one offer. The issuer applies the existing same-page, duplicate, queue and bandwidth controls. All arms run one active workload in the two-core machine at placement slot zero, starting from source ordinal one. The 200M measured instructions are also broken into four successive quarters of 50M with continuous cache and predictor state.

The original-miss denominator contains every dynamic load operand with a NoPF LLC tag miss. Each operand is joined by retired instruction ordinal and memory slot across NoPF, Stride, the frozen structural rule operating alongside Stride, and `look2`. Physical page mappings must match across all four arms. A gained timely use is an attributed prefetch first use in `look2` on an original miss where the reference had no direct timely use. A loss reverses that comparison. An ordinary residency change can improve performance without entering either count. Percent time saved uses the named reference's fixed-work cycles as the denominator. Useful accuracy counts timely first uses and late merges per accepted prefetch. Timely accuracy uses timely first uses alone.

The calculation reads whole-interval outcome words. It checks native tag-miss and first-use counters separately in every quarter. Run the study from the cache root after preparing the primary sources, building the native simulator and completing the matched full-interval NoPF, Stride and structural runs.

```sh
source .runtime/project_env.sh
python3 -B llc_hybrid_prefetch/full_workload/long_lookahead/run_lookahead.py
python3 -B llc_hybrid_prefetch/full_workload/long_lookahead/score.py
```
