# Does a recent different PC's increment improve full-window MCF prefetching?

**A different PC's increment produced timely prefetches for 23 of the 418 sampled far-MCF targets.** The local rule produced zero timely hits on those same operands. Across the full 200M measured instructions, the shared rule saved 0.076% of cycles relative to the matched local rule. It took 10.85% more cycles than SPP. The address relationship is observable and can sometimes be used in time. Its net value depends on the region and the other requests admitted.

## Full fixed-work comparisons

| MCF17 view | Time saved vs Stride | Time saved vs matched local increment | Strongest measured conventional arm and shared-rule time difference | Original timely gained / lost vs Stride | Original timely gained / lost vs local |
|---|---:|---:|---|---:|---:|
| 472B start | +0.761% | +0.151% | SPP, shared took 10.01% more time | 26,833 / 8,070 | 26,542 / 29,872 |
| 472B far | +0.797% | +0.076% | SPP, shared took 10.85% more time | 25,108 / 6,978 | 23,847 / 22,588 |
| 1644B start | −0.367% | −0.623% | Multi-delta, shared took 3.86% more time | 167,838 / 172,814 | 190,098 / 266,397 |

The shared rule saved 433,730 cycles relative to local in the 472B far view. It displaced 68,714 Stride proposals and added an offer to 368,025 otherwise silent slots. The comparable local rule displaced zero Stride offers and added 437,954 offers to silent slots. These counts precede shared issuance checks. Under the shared rule, 18,765 original misses became directly timely despite lacking a directly timely first use under any of the four conventional arms. Conversely, 384,207 original misses directly timely in at least one of those four executions lacked such a hit under the shared rule. This union uses four different executions and serves only as an offline reference. [Performance and resource counters](performance.csv) · [Full and quarter paired operands](paired.csv) · [Four-arm union](conventional_union.csv)

## What happened to the selected target addresses?

| Region | Frozen offline target operands | Shared hybrid timely / late | Timely supplied by the arithmetic path | Matched local hybrid timely / late |
|---|---:|---:|---:|---:|
| 472B start | 51 | 3 / 2 | 3 | 0 / 1 |
| 472B far | 418 | 24 / 4 | 23 | 0 / 2 |
| 1644B start | 1 | 1 / 0 | 1 | 0 / 1 |

In the 472B far view the remaining timely hit came from ordinary Stride in the changed execution. For 390 of the original 418 candidate operands the shared run had neither a timely nor a late attributed first use. At the original offline cue, the live shared rule had a same-line offer accepted in 20 cases and rejected in 5. It offered a different line in 85, was silent in 288 and had no logged decision for 20. The offline screen tried each legal arithmetic action at every evaluated anchor. The live system has a sparse grant, competes for resources, and sees a changed ingress stream. The 418 count describes reach under the offline assumptions. [Prespecified target identifiers](../mcf_arithmetic_targets.csv) · [Observed first uses](target_outcomes.csv) · [Per-operand lifecycle and address](target_details.csv)

For one concrete far-window example, PC 4,206,027 accessed physical cache lines 57,454 and 57,451 at source ordinals 1,800,071,758 and 1,800,071,804. At ordinal 1,800,071,809, a different PC (4,206,075) accessed line 53,187. The rule applied the earlier −3 increment and generated line 53,184 at cycle 28,669,229. The request dispatched at 28,669,249 and filled at 28,669,452. Operand 0 of instruction 1,800,071,866 reached the LLC at cycle 28,669,479, leaving 27 cycles of fill-to-demand slack. This verifies a causal address calculation using two PCs in the actual replay. The source-level arrays and the marginal cycles saved by this single request are unknown. The useful increment here is a small constant. The observed successes warrant a comparison against cheap negative-delta choices before attributing them to any learned representation.

## Experimental method

The far region of SPEC17 MCF-472B produced 418 sampled original misses that a conditional arithmetic rule could identify with at least 130 cycles of apparent lead under Stride. A second PC had recently changed its own physical line by a signed increment. Applying that increment to the current PC's line sometimes anticipated another demand. Those targets survived the local arithmetic control, six fixed increments (±1, ±2, ±4), and direct first-use coverage under the four conventional arms. Other simple alternatives, including fixed increments −3, −5 and −7 and a per-PC increment table, could still explain those targets. The observation prompted this replay and is part of its discovery evidence.

The cohort covers all three registered MCF17 views: the first and last 250M instructions of the MCF-472B recording and the first 250M instructions of the 1644B recording. Each view retires 50M warmup plus 200M measured instructions. The far view seeds the simulator's page allocator from the skipped source prefix and retains the absolute source ordinals. These are cross-region tests within MCF. Results from this series cannot establish transfer to a separate program.

Each new low-cost arm uses the existing [arithmetic source](../arithmetic.inc) and an identical one-in-eight grant on measured, admitted LLC demand ingresses. Both may replace a Stride offer or supply a candidate at a silent Stride slot. The 64-entry tagged table stores the last physical line and delta per tracked PC. The local rule adds the trigger PC's own latest increment. The shared rule selects the most recently observed different PC within an eight-PC history and adds that PC's last increment to the current physical line. Both require a nonzero delta, a distinct physical target in the trigger page, and the normal shared-issue checks. The arithmetic state and grant counter begin empty at ROI start. The processor, caches and native Stride state receive the full warmup. The shared rule needs history storage and another tagged table read to locate the second PC. The simulator charges no added inference latency. No threshold, table size or grant changed after identifying the 418 targets.

The analysis compares execution cycles and lifecycle-attributed first uses across NoPF, Stride, Multi-delta, SPP, temporal, local and shared increments. All masks are joined by original instruction and operand; physical mappings and quarter boundaries must agree. A first use under the hybrid can come from the arithmetic path or ordinary Stride. Attribution of the sampled cohort also checks the native producer and physical line.

Run from the cache checkout after intake, the normal baseline runs and the existing arithmetic build.

```sh
source .runtime/project_env.sh
p=llc_hybrid_prefetch/full_workload
python3 -B "$p/streams/replay.py" --build
python3 -B "$p/streams/mcf_transfer/run_mcf.py"
python3 -B "$p/streams/mcf_transfer/score.py"
python3 -B "$p/streams/mcf_transfer/targets.py"
```
