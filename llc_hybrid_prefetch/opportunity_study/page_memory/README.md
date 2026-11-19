# Does remembering the last offset on the current page save processor time?

On the fixed later BFS-3 interval, remembering the last offset on the current physical page saved **21,596 more cycles** than scores from only the current PC and offset. A same-page next-line prefetch at exactly the same grants saved another **7,037 cycles** over the history rule. This replay supports a causal value for the previous offset relative to this current-context predictor. The simpler next-line control performed best among the equally scheduled arms. [Equal-grant native results](results.csv)

| Auxiliary rule at 1,460 held-out grants | Correct next qualifying offset on Stride stream | Fixed-work cycles saved vs Stride | Time saved | Auxiliary accepted | Auxiliary timely | Matched Stride misses becoming hits / hits becoming misses |
|---|---:|---:|---:|---:|---:|---:|
| Stride shadow | — | 0 | 0 | 0 | 0 | 0 / 0 |
| Next line within the page | 828 | 55,552 | 0.403% | 751 | 440 | 438 / 43 |
| Repeat last same-page delta | 726 | 43,333 | 0.314% | 597 | 355 | 352 / 43 |
| Current PC and current offset | 511 | 26,919 | 0.195% | 730 | 290 | 254 / 86 |
| Previous same-page offset | 759 | 48,515 | 0.352% | 750 | 379 | 375 / 51 |
| Previous same-page offset and PC | 759 | 48,382 | 0.351% | 750 | 379 | 375 / 51 |
| Previous offset, 16-entry direct map | — | 46,243 | 0.335% | 742 | 374 | 366 / 61 |
| Previous offset and PC, 16-entry direct map | — | 46,243 | 0.335% | 742 | 374 | 366 / 61 |

- Native Stride took 13,798,947 cycles at fixed 2M retired instructions, and NoPF took 15,514,952. The signed percentages use the whole 2M-instruction ROI. All auxiliary rules first activated at zero-based LLC ingress 46,965. They stopped at the fixed exclusive ingress 58,643. Earlier data keep the predictor state current. Each live arm produced 58,707 LLC lookups on the same dynamic operands as Stride. The response to an intervention can change eligibility, order and deadlines later in the run. The counts in the second column are an offline test on the unchanged Stride stream. [Score breakdown](offline.csv) · [Native measurements](results.csv)
- Native Stride transferred 3,213,824 DRAM bytes. Current-context scores added 26,176 bytes, page-offset scores added 23,936, the 16-entry rule added 23,232 and next-line added 21,440. The page-offset rule agreed on 381 grants and supplied a replacement on 1,079. Of those replacements 329 were already resident. A native agreement keeps the existing candidate and its source attribution. The table counts auxiliary timely first uses of replacements; the last column includes all paired outcome changes after the policy changes timing. [Traffic and lifecycle counts](results.csv)
- On the original schedule, a qualifying future access existed on 1,087 grants. Of 759 correctly named by the previous-offset rule, **389 were already directly timely under Stride** and 370 still missed. Next-line correctly named 438 persistent misses and 390 already covered accesses. Each correct immediate-next-on-page match in this test resolves to one distinct dynamic operand. Address quality and net coverage measure different effects. [Offline counts](offline.csv)
- A 64-entry history of recent ingresses can recover every predecessor allowed by the 64-ingress/4,096-cycle cutoff. There was an eligible prior-page record at 1,104 of 1,460 grants in the live full-window offset arm. Using 16 direct-mapped tagged page records left 974, with 2,272 fewer cycles saved. Adding the previous PC had no measurable advantage on this case. The frozen table exporter retains 128 current, 417 offset-context and 542 offset-plus-PC context histograms with at least eight training observations. The replay interprets these offline-fitted histograms, each with 64 outcome counts and a support count, at zero modeled inference latency; their table size, port timing and energy have not been validated in hardware. The 16-entry change bounds predecessor lookup, with the scoring tables retained. [Protocol](protocol.md) · [Frozen prefix export](prepare.py)
- A baseline witness starts with physical line 191187, followed two other ingresses later by line 191188 at instruction 51,608,215. Native Stride was silent. Current context chose line 191185 and the hardware found it resident. Page-offset history chose line 191189, filled it at cycle 12,761,893 and its first use by instruction 51,608,294 occurred at 12,762,387. Next-line and repeating the prior page delta also filled the correct line for the same demand, through their own replay timelines. [Native witness](witness.csv) · [Cue and competing forecasts](../history/page_memory_witnesses.csv)
- The fitted rule used the first 60% of the existing later BFS-3 LLC ingress stream. This follow-up was motivated by a previously measured within-recording result, and each table shares the same prefix. It tests a mechanism on this single previously examined suffix. The existing later BFS-3 → later BFS-8 transfer had only 2 matches in 98 grants for each of these three rules. SPP run over the complete later BFS-3 window saved 1,793,065 cycles relative to Stride under different trigger and issue schedules. That separate result shows how competitive a stronger conventional policy can be. [Transfer results](../history/page_association.csv) · [SPP and other controls](../conventional/policy_results.csv)

## Reproduction

Run in the opportunity study directory after extracting the later BFS-3 Stride/NoPF arrays. NumPy is required by the prefix exporter and offline scoring.

```sh
python3 -B page_memory/prepare.py
python3 -B page_memory/build.py
python3 -B page_memory/run.py --jobs 4
python3 -B page_memory/summarize.py --jobs 4
python3 -B page_memory/offline.py
python3 -B page_memory/witness.py
```

The fitted histogram export contains only prefix information. A separate reference file includes original Stride trigger IDs and predicted addresses exclusively for the shadow check. The simulator reads only the fitted tables and the fixed gate. The Stride shadow reproduced the original cycle count and every stats row. Its trigger identity, physical base and all three chosen addresses matched the frozen Python reference on every one of the 1,460 grants.
