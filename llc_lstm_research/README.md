# LSTM prefetching at the shared LLC

Can a small LSTM cover more LLC misses in time than conventional prefetchers at comparable DRAM traffic?

- Input is a **demand data load that missed L2**. Observe successful new entries to the shared LLC read queue before tag lookup. Count merged, forwarded, and rejected requests separately. NoPF supplies the reference labels and every policy uses the same LLC entry point.
- The demand keeps moving. The predictor can issue a separate request that fills the LLC. Current LLC hit/miss is available after lookup.
- This directory holds the experiment. Sacramento uses the same path under `~/cache`. [Simulator and baselines](PYTHIA.md) · [Sacramento results](conventional/RESULTS.md)

## What the papers suggest

- **Hashemi et al. (2018)** used PC and miss deltas to predict the next miss delta. Relative addresses and ordered history are good starting features. A correct next delta can still arrive too late to save a stall. [Delta-LSTM](https://proceedings.mlr.press/v80/hashemi18a.html)
- **Voyager (2021)** used LLC accesses, split addresses into page and offset, and trained on multiple future targets. Several offsets can be useful from one trigger, so give each candidate its own score. Keep the first model small and charge its simulated inference time. [Voyager](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)
- **PPF and Pythia** make the decision and the outcome explicit. A candidate can be correct but late, useless, or harmful under contention. Keep conventional candidates and record which requests actually fill and help a demand. [PPF](https://people.engr.tamu.edu/djimenez/pdfs/ppf_isca2019.pdf) · [Pythia](https://arxiv.org/abs/2109.12021)
- **Augur** found little extra benefit from longer context in its single-core reservoir sweep. That is a reason to measure the window here. Its L2 predictor and our shared LLC LSTM see different streams. [Augur §VI-E](https://arxiv.org/html/2611.10929v1#S6.SS5)

## The label comes first

For each input event, score three same-page candidates, `+1`, `+2`, and `+4`. Find the first later accepted demand to each physical line in the **same NoPF run**, across all cores, using ingress order. Store its LLC hit/miss and the cycles between the two ingress events. Target ingress is a conservative deadline that avoids giving the model credit for time the demand spends stalled in the LLC queue.

- **1:** That demand misses in the LLC, and its lead time falls in the window calibrated on development data.
- **0:** The first demand hits, falls outside the window, or is absent from a complete window.
- **Mask:** The offset crosses a page, the observation ends too soon, or the line is already resident or in flight at the trigger when that state is recorded. Keep targets and follow-up within their own split.
- Use three independent sigmoid scores and masked binary cross-entropy. Calibrate the lower timing bound from measured issue-to-fill latency and the inference budget. Choose the upper bound on development data to limit how long a prefetched line must stay resident. Freeze both before testing. Two candidates can be positive on the same event. A longer useful offset may give more lead time even when the next miss is `+1`. Actual timely hits, evictions, and IPC come from the closed-loop run.
- Before training, measure positives, lead-time distributions, and the fraction of unique NoPF misses that any of the three offsets could reach. Count each miss once. Check how much `+2` and `+4` add beyond `+1`.
- Teacher actions are useful for an imitation baseline or candidate generation. The main label uses future demand. If `+1` dominates the label, try a separate hybrid with Next-Line always on. Label the extra candidates by the misses that **Next-Line still leaves**, and count the extra traffic.

## Start small, then tune the window

- One LSTM layer, **8 hidden units**, separate history per core and shared weights. Start with signed line delta, page offset, and cycles since the previous same-core arrival. Use clipped or bucketed values. Add a same-PC bit after confirming PC survives L2 forwarding, then try shared LLC MSHR occupancy as a separate ablation. If recent hit/miss behavior helps identify phases, test a rolling fraction built from outcomes that have already arrived at prediction time.
- Begin with **8 events**, then compare **4, 8, 16, and 32** on the same chronological train/validation/test splits. Use the same eligible anchors across window lengths, with history drawn only from the current split or its own warmup. Record the actual cycles spanned by each window. A short window may be enough for a stream and too short for recurring irregular patterns.
- Compare a current-event MLP and a small MLP that sees the same window. Choose the shortest window that retains recall at the same issue rate and performs best on the validation trade-off between timely coverage and runtime. Evaluate the chosen settings on the held-out runs. Try width 16 only if the window results suggest that extra capacity could help.
- If window replay costs too much, test carrying `h,c` per core and updating once per event. Retune the training unroll (TBPTT) separately. That setting limits gradient reach while deployment can retain state across chunks. [TBPTT](https://proceedings.mlr.press/v115/aicher20a.html)

## Multicore and success criteria

- First use two copies of 602 to check the path. Each core gets its own address space. Then compare 1/2/4/8 cores with same-phase, staggered, and held-out mixed workloads. Fix training event counts and shared LLC/DRAM resources when isolating the core-count effect. Compare with and without the shared occupancy input at each core count.
- More cores can provide new patterns, repeated samples, or extra contention. Compare the label distribution for similar per-core histories, ingress gaps, and queue occupancy. A more varied stream helps learning when those new cases have patterns that still predict useful prefetches on held-out workloads.
- Pick thresholds on validation data under a request budget. On held-out runs report accuracy, timeliness, coverage, LLC hit/miss rate, MPKI, DRAM bytes, and per-core IPC. Judge the design by fixed-work completion cycles with inference delay enabled. [Metric definitions and Roofline](PERFORMANCE.md) · [Paper index](PAPERS.md)
