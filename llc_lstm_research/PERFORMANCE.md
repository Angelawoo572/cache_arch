# How we measure prefetching

- **Accuracy:** `(timely uses + late demand merges) / resolved independent prefetches`. Also report timely uses alone. Count attempts, queue merges, and unresolved requests separately.
- **Timeliness:** `timely / (timely + late)`. A timely request fills a line before demand tag lookup and the line is still resident. A late request merges with the demand before fill. Report fill-to-use slack and unused evictions. [Pythia](https://arxiv.org/abs/2109.12021) · [Feedback-Directed Prefetching](https://users.ece.cmu.edu/~omutlu/pub/srinath_hpca07.pdf)
- **Coverage:** Of the NoPF demand misses, how many become timely hits? Report it when stable demand IDs and request lifecycle records support the match. `1 − MPKI_policy / MPKI_NoPF` is a net miss reduction estimate. It includes pollution and access-stream changes.
- **Hit rate, miss rate, and MPKI:** Use the same LLC demand data-load lookup cohort for all policies. Report access counts. MPKI divides LLC demand data-load misses by retired instructions and multiplies by 1,000. Sum counts before computing pooled rates.
- **IPC and runtime:** Report each core's retired instructions, cycles, IPC, and speedup against the matching NoPF workload. For a shared start and fixed work per core, use the last core's finish cycle for makespan. At fixed frequency, time is cycles divided by frequency. Report simulator wall time separately.
- **Traffic:** Count actual DRAM transfers, including reads and writebacks, when the counters support it. Compare strategies at both the same maximum degree and comparable actual traffic. Report queue pressure and the timing distribution of memory stalls.

## Roofline and inference cost

- Roofline bounds throughput by `min(compute ceiling, sustained bandwidth × operational intensity)`. For integer-heavy 602, use a clearly labeled instructions-per-DRAM-byte variant with instructions/cycle and bytes/cycle. Measure exposed stalls, MSHR concurrency, and demand latency separately to see whether memory latency was hidden. [Roofline](https://escholarship.org/uc/item/5tz795vq) · [Cimple §3.2](https://commit.csail.mit.edu/papers/2018/kiriansky-pact18-cimple.pdf)
- The proposed 3-input, 8-unit, 3-output LSTM needs about **376 matrix MACs per streaming update**. With two extra features it needs **440**. Replaying a window of length `W` costs roughly `352W + 24` or `416W + 24` MACs. Gate activations, state access, and queueing add cost.
- Sweep positive model delays and initiation intervals. Count prediction queue drops. The demand path keeps running while prediction, prefetch queueing, and memory service take place.
