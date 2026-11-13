# LSTM prefetching at the shared LLC

**Research question.** When does a core's earlier L2D miss history tell us which previously seen page and line to prefetch?

- The [605 baselines](conventional/605_RESULTS.md) report accuracy, timeliness, coverage, and IPC. The [70M history check](conventional/605_HISTORY.md) tests what ordered events add beyond the page-PC table.
- The [LSTM design](DESIGN.md) gets its offline inputs and labels from NoPF demand events. Conventional runs show where prefetching falls short. Model work is at the design stage.
- [Measurement definitions](PERFORMANCE.md) follow CMU 18-447. [Conventional baselines](conventional/README.md) explains how to reproduce the screen. [Paper index](PAPERS.md) records related designs.

## What we would test

- Check whether an earlier cue predicts a timely candidate or a reason to abstain when current inputs look alike. Keep the label binary per candidate. Measure candidate reach before choosing the output offsets.
- Test the effect of carrying, resetting, and disturbing state with one recurrent layer. Use identical encoded inputs and candidates in the matched controls.
- Compare OMNeT++ with a quiet peer, a second OMNeT++ stream, and an MCF peer on the same simulated hardware. Hold shared cache capacity, queue sizes, DRAM channels, and training event counts fixed. Compare the same per-core trace region with and without shared LLC MSHR occupancy at the scoring head.
- Start with private-cache prefetching disabled to isolate the LLC path. A later run with Berti at L1D and Pythia at L2 can measure what is left for the LLC. Generate labels again under that upper-cache configuration. [DPC4 infrastructure](https://sites.google.com/view/dpc4-2026/submit-work/infrastructure)
