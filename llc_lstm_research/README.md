# LSTM prefetching at the shared LLC

**Research question.** When can an earlier pattern in a core's L2D miss stream tell us to prefetch a farther offset, or to skip a prefetch altogether?

- The [SPEC trace screen](conventional/SPEC_SCREEN.md) identifies 620 OMNeT++ as the first workload. The [620 baseline comparison](conventional/620_RESULTS.md) shows where each conventional policy falls short.
- The [LSTM design](DESIGN.md) uses the same demand stream and fields as those policies. It lays out how to test whether learned state can remember useful cues across intervening misses. Model work is at the design stage.
- [Measurement definitions](PERFORMANCE.md) follow CMU 18-447. [Conventional baselines](conventional/README.md) explains how to reproduce the screen. [Paper index](PAPERS.md) records related designs.

## What we would test

- Check whether an earlier cue predicts a timely candidate or a reason to abstain when current inputs look alike. Keep the label binary per candidate. Measure candidate reach before choosing the output offsets.
- Test the effect of carrying, resetting, and disturbing state with one recurrent layer. Use identical encoded inputs and candidates in the matched controls.
- Compare OMNeT++ with a quiet peer, a second OMNeT++ stream, and an MCF peer on the same simulated hardware. Hold shared cache capacity, queue sizes, DRAM channels, and training event counts fixed. Compare the same per-core trace region with and without shared LLC MSHR occupancy at the scoring head.
- Start with private-cache prefetching disabled to isolate the LLC path. A later run with Berti at L1D and Pythia at L2 can measure what is left for the LLC. Generate labels again under that upper-cache configuration. [DPC4 infrastructure](https://sites.google.com/view/dpc4-2026/submit-work/infrastructure)
