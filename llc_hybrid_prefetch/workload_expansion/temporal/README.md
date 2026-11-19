# Can an old physical address predict a useful distant successor?

Yes for address recurrence. In the seven deeper Google windows, an online table with unlimited capacity reproduced **2,264 next physical lines**, of which 331 were more than 512 lines away and outside the three most recently touched physical pages. Every one of the 2,264 targets was already an LLC demand hit in NoPF. Of the 4,897 eligible next demands on which NoPF, Stride, multi-delta and SPP all missed with sufficient reference lead, **zero** had been seen by physical line at an earlier observed Stride ingress in that ROI. Increasing the exact-target table capacity alone cannot bridge that gap under this trigger and reset convention. [Predictions, offers and PC cohorts](forecasts.csv) · [Causal address availability](reach.csv)

For example, Arizona’s physical line 35,711 was followed by line 1,026,944 at Stride ingress indices 4,038/4,039. At indices 16,105/16,106 this same physical handoff recurred from a different cue PC. The demand at original instruction 4,583,443 was already an LLC hit both in Stride and NoPF. This is a real cross-region association with an absolute displacement of 991,233 lines. A 256-entry direct-mapped line table could not preserve most such infrequent associations in this panel; the unlimited comparator recovered the example. [Dynamic example with both cue PCs, original operands and cycles](examples.csv)

## Distinguishing a correct address from an incremental miss

Each row pools disjoint discovery objects within one panel and one horizon. Offers use every eligible causal trigger in that panel’s chronological final fifth, while state continues to observe earlier arrivals. At a fixed horizon each cue maps to one unique target ID; horizon-one and horizon-four targets can overlap. The reference uses the median training-prefix demand dispatch-to-fill time (128 cycles when no completed training service exists). An eligible first use means that the target differs from the cue and no intervening ingress touched it. Some other earlier cue or a different action could change the outcome. [Protocol](protocol.md)

| Panel | Horizon | All triggers | Unlimited line table offers / correct first uses | 256-entry line table offers / correct | Original misses with sufficient lead and tag misses in all three conventional runs | Of those, already observed physical line / already observed page |
|---|---:|---:|---:|---:|---:|---:|
| 16 discovery objects, first short interval | 1 | 12,678 | 1,229 / 660 | 11 / 4 | 1,563 | 0 / 995 |
| Same objects, second short interval | 1 | 12,850 | 1,133 / 841 | 25 / 1 | 1,205 | 0 / 689 |
| 7 Google objects, disjoint deeper interval | 1 | 18,703 | 4,483 / 2,264 | 15 / 2 | 4,897 | 0 / 2,477 |
| 16 discovery objects, first short interval | 4 | 12,633 | 1,225 / 356 | 11 / 1 | 5,651 | 0 / 4,514 |
| Same objects, second short interval | 4 | 12,805 | 1,130 / 540 | 25 / 0 | 6,056 | 0 / 4,820 |
| 7 Google objects, disjoint deeper interval | 4 | 18,682 | 4,474 / 1,543 | 15 / 0 | 9,035 | 0 / 4,450 |

A PC-only table offered a recent absolute line 12,631 times at horizon one in deeper Google and matched 13 future first uses. Next-line produced 272 correct early choices on the 4,897 additional-miss targets, and 46 on the 9,035 fourth-event targets. The latter control generates a new address from the current physical page. Its additional-miss count is a reference-trajectory address/lead result; native usefulness still depends on admission and service. Of the 4,897 early next-event opportunities, 853 have a target on the cue’s page, and 1,033 target a known physical page outside the three recent pages. The remaining 2,420 had no earlier observed target-page handle in the ROI. These are opportunity counts with future labels, separate from the conditional probability of identifying the correct page and offset. The four-way miss totals include target PCs for which Stride never achieved an attributed T anywhere in the ROI (4,358 / 4,897 next-event and 6,906 / 9,035 fourth-event). [Full census by target-PC cohort](reach.csv)

All tables started at the first logged ROI ingress. Earlier machine warmup and unobserved private-cache events may have touched these lines. The sources provide a finite sample with a particular cache hierarchy, physical mapping and exposure window; this evidence cannot rank lifetime temporal prefetchers or prove that previously unseen lines are intrinsically unpredictable. The finite controls allocate at most 256 tagged entries, updated after observing the successor. At minimum each valid entry stores its trigger tag and physical successor; PC-conditioned variants add a PC tag. H=4 requires a four-event delayed update queue. The unbounded dictionary uses one record per distinct trigger and supplies a capacity diagnostic. Measured data do not include a finite-table pipeline, lookup energy, request admission or performance replay for these added proposals.

## Reproduction

Produce both short-window and deeper Stride/NoPF/multi-delta/SPP history directories with the screen commands. From this directory run

```sh
python3 -B study.py --jobs 4
python3 -B verify.py
```

Validation compares independent table implementations at both horizons, tests that suffix removal preserves earlier predictions, checks unique policy-aligned demand IDs and physical lines, and reconciles all 1,170 score rows with observed-address ceilings. [Predictor and scoring](study.py) · [Checks](verify.py)
