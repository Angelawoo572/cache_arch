# Whole LLC arrival order versus instruction order

On each complete 200M-instruction ROI, record the actual sequence presented to Stride. Compare the same dynamic operands in arrival order and source instruction/operand order. A timely-to-miss adjacency is an observation to explain. A transition can change when outstanding instructions arrive in a different order.

| Recording | Stride demand admissions | Admissions in four 250K-instruction islands | Original NoPF misses present / total | Original misses absent at Stride ingress |
|---|---:|---:|---:|---:|
| GAP BFS-3 | 5,072,474 | 35,011 | 2,244,882 / 2,244,917 | 35 |
| SPEC06 Soplex-92B | 5,246,784 | 20,034 | 2,000,588 / 2,003,166 | 2,578 |

## What followed an attributed timely hit

Each row conditions on adjacent targets that both entered Stride and were LLC misses in NoPF. The denominator counts eligible previous direct hits with an observed next target in that stream. The failure numerator combines the next plain tag miss and next late first merge. An ordinary tag hit has its own outcome. The island rate pools four independent slices.

| Recording and scope | Arrival late / plain after direct | Program order late / plain after direct | Whole arrival failure rate | Island arrival failure rate | Arrival denominator, whole / islands |
|---|---:|---:|---:|---:|---:|
| GAP BFS-3, core | 71,667 / 351,925 | 71,808 / 352,226 | 42.87% | 63.02% | 988,157 / 4,218 |
| GAP BFS-3, PC | 69,280 / 162,693 | 69,280 / 162,694 | 23.48% | 25.86% | 988,149 / 4,215 |
| GAP BFS-3, 256 KiB physical region | 70,728 / 195,400 | 70,772 / 195,425 | 26.94% | 33.20% | 988,009 / 4,196 |
| SPEC06 Soplex-92B, core | 85,233 / 141,365 | 81,807 / 138,747 | 53.64% | 28.92% | 422,441 / 2,016 |
| SPEC06 Soplex-92B, PC | 99,631 / 69,073 | 99,711 / 68,677 | 39.94% | 22.67% | 422,413 / 2,007 |
| SPEC06 Soplex-92B, 256 KiB physical region | 87,091 / 112,391 | 84,966 / 108,247 | 47.23% | 25.63% | 422,396 / 2,013 |

## Streams with no direct timely conversion

| Recording and scope | Whole groups with 32+ original misses | Never timely groups | Original misses in those groups | Island instances with 32+ misses | Never timely island instances |
|---|---:|---:|---:|---:|
| GAP BFS-3, PC | 18 | 2 | 2,865 | 11 | 0 |
| GAP BFS-3, 4 KiB physical page | 13,746 | 1,338 | 52,144 | 101 | 0 |
| GAP BFS-3, 256 KiB physical region | 670 | 21 | 816 | 26 | 0 |
| SPEC06 Soplex-92B, PC | 155 | 18 | 2,197 | 17 | 0 |
| SPEC06 Soplex-92B, 4 KiB physical page | 9,400 | 3,697 | 369,918 | 69 | 4 |
| SPEC06 Soplex-92B, 256 KiB physical region | 340 | 53 | 6,336 | 20 | 0 |

Ordering preserves whether a fixed group ever had a direct timely hit. Short logging windows can omit its success or exclude the group through insufficient support. In the island columns one PC or physical region can contribute a separate instance to each quarter.

Gap distributions retain the elapsed cycles and the distance in the full admission sequence from the last direct hit to the first late/plain miss, allowing ordinary tag hits in between. A no-tag event censors an open run. Signed negative gaps in program order expose reordered demand arrivals. An arrival rank difference of one means adjacent admissions. These outcome-to-outcome gaps describe the sequence and do not establish when a prefetch address was predictable.

See [all and matched stream counts](counts.csv), [all predecessor outcomes](predecessors.csv), [signed cycle and admission-rank gaps](gaps.csv), and [native parity and denominators](validation.csv). The separate [retirement-order study](../streams/README.md) also retains original misses absent at Stride ingress.
