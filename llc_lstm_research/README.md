# LSTM prefetching at the shared LLC

**Research question.** When can an earlier pattern in a core's L2D miss stream tell us to prefetch a farther offset, or to skip a prefetch altogether?

- The first 602 segment strongly favors Next-Line. The [SPEC trace screen](conventional/SPEC_SCREEN.md) looks for a development region where the fastest conventional policy saves little time and still leaves many LLC misses.
- [Proposed LSTM architecture](DESIGN.md) gives the input, cell, output, label, and feedback path. Model work is at the design stage.
- [Measurement definitions](PERFORMANCE.md) follow CMU 18-447. [Conventional baselines](conventional/README.md) explains how to reproduce the screen. [Paper index](PAPERS.md) records related designs.

## What we would test

- Keep the first label binary per candidate. Measure how many distinct future misses the candidate set can reach at a useful lead time. A better scorer has little room to help if the candidate source rarely proposes the right line.
- Test whether longer ordered history improves decisions for requests with the same recent history. Use a fixed one-layer model for the window sweep. Try a wider model only if the results point to limited capacity.
- Compare OMNeT++ with a quiet peer, a second OMNeT++ stream, and an MCF peer on the same simulated hardware. Hold shared cache capacity, queue sizes, DRAM channels, and training event counts fixed. Compare the same per-core trace region with and without shared LLC MSHR occupancy at the scoring head.
- Start with private-cache prefetching disabled to isolate the LLC path. A later run with Berti at L1D and Pythia at L2 can measure what is left for the LLC. Generate labels again under that upper-cache configuration. [DPC4 infrastructure](https://sites.google.com/view/dpc4-2026/submit-work/infrastructure)
