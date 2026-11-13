# LLC LSTM design

## What the 620 baselines tell us

- **Stride rarely issues.** Its timely requests cover few misses. First look for recurring address sequences that fall outside a constant stride.
- **Streamer and SPP often reach the right line too late.** About 45% of their used prefetches are late. Look for an earlier cue that would let us choose that line while there is still time.
- **AMPM issues many lines that have no observed use by the end of the run.** Look for context that tells us when to abstain. Keep unresolved requests separate from lines confirmed unused.
- Use the [620 results](conventional/620_RESULTS.md) to test these explanations. The counters tell us where the policies struggle. A trace analysis can tell us whether the relevant sequence is learnable.

## What the model would see and do

```text
L2D demand-load miss admitted to the LLC read queue
  ├─► Demand continues to the LLC tags
  └─► Same load PC and physical line that the conventional policies receive
         │
         ▼
       LSTM ──► scores for eligible offsets ──► issue one or abstain
        ▲ │                                              │
        └─┴── carry this core's h,c to the next event     └─► log use and timing
```

- **Input.** Use the same accepted demand stream, load PC, and translated physical line as the conventional policies. The core selects its own state. Encode the line as page offset and signed delta, with a token for page changes. Keep a bounded PC token so the model can distinguish load instructions. Each matched control gets the same encoding. Keep cycle gaps, MSHR occupancy, and LLC outcomes out of the first input. The native SPP retains its existing delayed use feedback.
- **State.** Start with one causal LSTM layer and a small state per core. Share weights across cores. Update `h,c` once for every admitted event, including events on which the policy abstains. The input gate can write a new cue, the forget gate can preserve an earlier cue through unrelated misses, and the output gate can expose the cue when it matters. For example, the same current PC and recent deltas could call for a farther offset if an earlier cue marks a traversal that recurs. Eight units is a starting budget. [CMU 11-785, LSTM gates](https://deeplearning.cs.cmu.edu/S26/documents/slides/lec14.recurrent.pdf#page=85)
- **Output.** Give each legal signed offset in a small, development-selected set its own score. Use the same set for the LSTM and matched controls. Keep `+1,+2,+4` as a simple reference. Before choosing the set, count distinct NoPF misses it could reach in time. At most one candidate can pass the common LLC filter and issue. The policy abstains when no score passes the threshold.
- **Label.** One bit per candidate. In NoPF, the first later demand to that line must miss and have enough lead time to get the line into LLC. Use zero for a hit, a miss outside the usable window, or no demand in a complete window. Mask illegal and censored cases. Keep lead cycles and the reason in the analysis log. More than one candidate can be positive. Actual late merges, timely uses, evictions and completion time come from simulation. Weights are fixed during the first simulation comparison.

## How to find the value of memory

- **Before training.** The 620 trace shows a three-event cue for misses both Streamer and SPP reached late. Longer ordered histories gave no consistent gain over shuffled histories in the budgeted lookup. Use a short rule as the next control for any future recurrent model. [620 history check](conventional/620_HISTORY.md)
- **Pick offsets from the trace.** A first count on 620 picked `+3,+1,+6` by taking the offset that added the most distinct misses at each step. In the second region this set reached 16.08% of NoPF misses, close to 15.89% for `+1,+2,+4`, using an illustrative 300–2,000 cycle lead. This measures which addresses the output can reach. The next test is whether event order helps choose among them at the same issue budget.
- **Test history and gates separately.** For the proposed streaming model, compare training unrolls of 4, 8, 16, and 32 events at the same width. Keep `h,c` across training chunks and detach the gradient. In evaluation, mask older history or reset state at controlled intervals to measure how much context the model actually uses. Compare with a one-event scorer, an MLP seeing the same ordered suffix, and a small ungated RNN. Check whether any advantage survives when the recent suffix stays fixed and the older order is disturbed.
- **Keep the comparison fair.** Use the same trace regions, warmup, LLC ingress tap, filter, and degree cap. Compare at the same actual issued requests or DRAM traffic. Include inference latency, queueing, and state storage in the performance comparison. Evaluate on a region reserved before tuning. Add shared MSHR occupancy only in a separate multicore comparison with the same signal available to the matched controls. [Metrics and timing budget](PERFORMANCE.md)
- **Hardware boundary.** Target a data line inside the translated physical page. Load PC is instruction metadata. I-fetch and other traffic still share cache and memory resources. Count state, PC encoding, weights, candidate logic, and queues in storage. Model the dependency on the prior update from the same core. [CMU 18-447 virtual memory](https://course.ece.cmu.edu/~ece447/s15/lib/exe/fetch.php?media=onur-447-spring15-lecture20-virtual-memory-afterlecture.pdf#page=26)
