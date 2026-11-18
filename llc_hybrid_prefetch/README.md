# What can a shared LLC prefetcher see early enough to matter?

**[Read the measured outcomes, paired load examples, and feature tests](MEASUREMENTS.md).**

## A. Research question

- Take a load that misses the last-level cache (LLC) with prefetching disabled. Find the earliest demand event at which an LLC predictor could have named its line. Did the PC-based Stride prefetcher name it? Was there time to bring it in? Would an earlier fill have advanced retirement or the end of the run?
- Keep the full demand stream. What Stride gets right matters as much as what it misses. Two predictors could agree, and the agreement might change a queued request's priority. Counting two useful prefetches for that one line would exaggerate the value.
- Run two identical out-of-order cores with separate address spaces and a shared LLC and DRAM channel. One program can occupy a queue or evict a line that matters to the other. Run each program alone with the same shared resources so the cost of a co-runner can be measured.
- The synthetic loop supplies ground truth about source-level boundaries. A jump between two addresses in SPEC is evidence of a delta change. A loop-boundary interpretation needs independent evidence.

## B. NoPF and Stride

### Why these workloads?

The sample was fixed before looking at the new policy results. Each program uses its two highest-weight publicly available SPECspeed 2017 windows. The numbers below are parts of the public filenames. A short interval sampled inside one of those files measures that interval. The SimPoint weight describes the original point and cannot turn a short prefix into a whole-program result. [Public files and weights](https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/) · [DPC4 trace collection](https://github.com/CMU-SAFARI/DPC4#workload-traces)

| Program | Public points | What to look for at the LLC |
|---|---|---|
| 605.mcf_s | 665B, 994B | The network simplex algorithm updates trees and works with pointers and integers. Look for an earlier PC or access that narrows the next target. A serial pointer load can also leave too little usable lead time. [SPEC](https://www.spec.org/cpu2017/Docs/benchmarks/605.mcf_s.html) |
| 607.cactuBSSN_s | 2421B, 3477B | Finite differencing across a grid can interleave regular streams. Check whether per-PC separation or more tracker entries explains a difficult-looking global history. Berti studied local deltas at L1. The LLC sees a different stream after private-cache hits are removed. [SPEC](https://www.spec.org/cpu2017/Docs/benchmarks/607.cactuBSSN_s.html) · [Berti, MICRO 2022](https://webdiis.unizar.es/~chus/pubs/2022_MICRO.pdf) |
| 623.xalancbmk_s | 700B, 592B | An XSLT transformation maps a tree of input nodes to another tree. Check whether successive visits share useful earlier context, and whether enough of those visits survive L1/L2 filtering. [SPEC](https://www.spec.org/cpu2017/Docs/benchmarks/623.xalancbmk_s.html) |
| 619.lbm_s | 4268B, 3766B | A three-dimensional lattice update provides a structured-access control. Look for regular streams and measure the bandwidth another program loses when more of those streams are prefetched. [SPEC](https://www.spec.org/cpu2017/Docs/benchmarks/619.lbm_s.html) |

- The main matrix has four solo workloads and three mixes, MCF with CactuBSSN, MCF with Xalancbmk, and LBM with Xalancbmk. Pair the first point with the first and the second with the second. Run NoPF and LLC Stride for each of the fourteen cases.
- Feed the same input instructions and operands to both policies. Use the same address placement, cache and memory configuration, warmup interval and measured work. For the main matrix the target is ten million warmup instructions and five million measured instructions per active core. Test warmup sensitivity at a common measured start.
- Leave private-cache prefetchers off. Both arms use a demand-only cache warmup. Stride learns its PC state during warmup and begins issuing at the measurement boundary. This controls initial residency and measures startup as well as later behavior. Stride observes demand loads admitted from L2 to the shared LLC and keeps separate PC history per core. Its baseline has 64 trackers, requires two equal nonzero line deltas, and can issue one next-stride line on the current page. Mark page boundaries inside the generator before the shared issuer filters requests.
- Align arms by core, trace instruction ordinal and load operand. Verify the PC and virtual line. Within an arm, use physical line and event order to follow service, fills and replacements. A private-cache hit, a merge and an independent LLC lookup are distinct paths. Keep unmatched paths visible in coverage denominators.

### Which score answers which question?

- The CMU 18-447 processor project's `perf_counters.sv` counts elapsed cycles and retired instructions in simulation. It also counts fetched and decoded instructions separately. This study uses **retirement** to define work, and marks simulator-only observations such as a future head wait explicitly. The course's prefetching lecture ties coverage to baseline misses and separates timeliness from address accuracy. [18-447 prefetching lecture](https://course.ece.cmu.edu/~ece447/s15/lib/exe/fetch.php?media=onur-447-spring14-lecture27-prefetching-beforelecture.pdf#page=35)
- The project reading by Chen, Penney and Jiménez frames learned cache and prefetch decisions around what a predictor observes and what action it takes. Here the issue cycle and destination line are both essential parts of that action. [AI for Computer Architecture, §3.3.1](https://link.springer.com/book/10.1007/978-3-031-01770-4)

| Measurement | Definition and use |
|---|---|
| Fixed-work completion time | Per-core cycles to finish its measured instructions, plus the last finishing core's cycle for the mix. Divide cycles by the common frequency for simulated time. This is the main measure of performance. |
| IPC and interference | Report retired instructions per cycle for each active core and total measured instructions divided by the mix's completion time. With identical work on one core, IPC and completion time give equivalent rankings. In a mix, report each program's slowdown versus its solo baseline as well. |
| L2 and LLC demand hit/miss rate | Hits or misses divided by completed demand-load tag lookups at that level, with merged and forwarded accesses separately identified. Show counts, denominator and demand MPKI. The set reaching the LLC can change between policies. |
| Accuracy | Distinct accepted prefetches with a qualifying first demand use divided by accepted prefetches. Show timely uses, late merges, unused evictions and unresolved requests separately. A raw candidate filtered before admission is outside that denominator. |
| Direct coverage and timely coverage | Divide distinct matched NoPF LLC demand misses receiving a direct prefetch use by all NoPF LLC demand misses in the measured cohort. Timely coverage requires that the line was filled, was still resident, and was used on the matched demand. A merge into an in-flight prefetch gets separate late credit. Show ordinary miss-to-hit transitions separately from direct first use. |
| Timeliness | Timely first uses divided by all qualifying first uses. For timely uses show the interval from completed fill to demand lookup. For late requests show the remaining service after the demand arrives where available. Leave zero-denominator ratios undefined. |
| Cost and exposed latency | Report actual DRAM traffic, traffic per thousand instructions, and queue or MSHR pressure. Count unique cycles in which the retirement head waits on a load. Multiple outstanding loads can overlap. Adding their individual service latencies would overstate the CPU time available to save. |

### What happens around a hit?

- Anchor on a dynamic demand and distinguish ordinary hits, first hits on prefetched lines, merges into outstanding prefetches and demand misses. A run of hits can follow private-cache filtering or ordinary temporal reuse. Only a tracked prefetch fill supplies direct evidence of a prefetch hit.
- For a timely hit, read backward from demand to fill, acceptance, candidate generation and trigger. For a late use, include the demand-to-fill gap. Compare earlier events for the same core, favoring controls at the same PC and comparable page position. Report the number of controls available.
- Read forward from the anchor to study consequences such as subsequent hits, eviction or a stall. Only the history up to the prediction time could have been used for prediction. Even an event immediately before the *hit* may have happened after the predictor had to decide.
- Diagnose pollution using matched demand hits that become misses, ordering and fill/eviction evidence. When a shadow-cache comparison is available, label it as an estimate. A hit-to-miss transition alone does not identify the prefetch that caused it.

### Reproduce and inspect

- Build and check the simulator, then run the registered matrix. The public branch includes configurations, scripts, the English report and compact results. Raw traces and event files live under the local and Sacramento study directories. [Simulator](sim/README.md) · [Experiments](experiments/README.md) · [Definitions and event matching](analysis/README.md)
- The 4,096-set LLC has 16 ways, 64-byte lines, 128 MSHRs and a single DRAM channel. Tags and outstanding requests are checked by the simulated issuer at admission. The results describe that simulated mechanism. Real tag/filter port costs and delays are open design questions.
- Each faster core stops taking in new instructions at its fixed work boundary. The run ends when the last core retires its measured work. Traffic refers to transfers in that interval, including work in progress at the boundaries. The shared-memory mixes here replay independent programs; coherence-driven communication within one multithreaded application would require a different trace source.

## C. What would warrant a feature?

| Observation in the experiment | Hardware-visible candidate | Check before exposing the feature |
|---|---|---|
| A stable stride changes after a recurring run length | Last nonzero line delta, count of consecutive equal deltas, PC | Give a counted two-stride table the same legal address range and issue budget. A fixed trip count is a particularly favorable case for this table. [Chen and Baer, IEEE TC 1995](https://homes.cs.washington.edu/~baer/chenieee.pdf) |
| At the same PC and recent deltas, different earlier events precede different useful targets | Short per-core history of PC transitions, page offset, deltas and relative age | Match on current context, hold out a region, then shuffle or remove earlier events. Measure candidate reach at usable lead time. Keep the original ordering if it adds information. |
| The right line was named but arrived after demand | Past service times, local queue pressure, age of competing requests | Compare within the same workload, core and candidate class. Record how many cycles were missing. A younger label can explain a late prefetch without being available when it was issued. |
| Another core changes timeliness or which program slows down | Requesting core and local shared-resource pressure | Compare solo and mixed runs using the same address placement and shared capacity. Keep program-private histories separate. The order between cores changes with scheduling. |
| Some future misses repeatedly block retirement | Delayed core feedback, e.g. an aggregated PC-indexed history of retirement-head waits | Count the transport delay, encoding bits and table cost. Test an earlier-issue intervention under normal queues. Head waits are a proxy, and their duration may change when service moves. |
| A line was already present or pending | Existing tag/MSHR/queue status, sampled or pipelined if necessary | Count checks, ports, retries and duplicated proposals. Charge these checks to hardware. Reserve statistical learning for cases where predicting future utility changes the decision. |

- At the prediction instant, record which inputs have actually arrived. An L2 miss request can carry its PC and core ID, but the number of PC bits transported and how a merge preserves provenance are implementation choices. A future LLC outcome, an exact future demand cycle, a logical loop index from a synthetic generator, and the paired NoPF outcome are offline labels.
- The first review ends with the measured evidence for and against each hypothesis. Model choice and cooperation between predictors are separate conversations after that review.
