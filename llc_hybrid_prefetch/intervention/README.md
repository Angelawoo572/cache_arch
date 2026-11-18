# Earlier-issue intervention

- This diagnostic uses hindsight to ask whether selected demand misses are worth anticipating. It does not train or evaluate an address predictor.
- Start with loads that miss in both runs and block the retirement head in the Stride run. Measure the median dispatch-to-fill latency on actual demand traffic. For each target, look for a prior LLC trigger on the same core and 4-KiB page between one and four such service times earlier. Avoid crossing a prior demand for the target line.
- Rank targets by exposed head wait, prefer an anchor about two service times earlier, and allow at most one address per anchor and 256 targets per replay. Hindsight identifies the target, the anchor, and the rank. The ordinary prefetch queue, real cache, degree cap, duplicate handling and bandwidth still decide what happens.
- Compare the original Stride execution and the modified Stride execution on identical fixed work. Report plan triggers that still occur, actual issues, the outcomes on the specified dynamic demands, both per-core times and the mix completion time. Retiming can move or eliminate a trigger. Changes in counterfactual cycles cannot be inferred by adding the original head waits.

Run from the study directory after the paired runs complete. The two replays are diagnostics on the first MCF+CactuBSSN point and the tight same-page nested loop.

```sh
for case_name in mcf-cactus-r1 syn-nested_samepage; do
  python3 intervention/earlier_issue.py "raw/runs/$case_name-nopf" "raw/runs/$case_name-stride" "raw/diagnostics/$case_name"
  python3 experiments/run.py --out raw/intervention --run "$case_name-stride" --env "HYBRID_PLAN=$PWD/raw/diagnostics/$case_name/plan.csv"
done
python3 intervention/summarize.py raw results mcf-cactus-r1 syn-nested_samepage
```
