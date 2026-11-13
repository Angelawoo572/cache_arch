# LSTM prefetching at the shared LLC

**Research question.** When does a core's earlier L2D miss history tell us which previously seen page and line to prefetch?

- The [605 baselines](conventional/605_RESULTS.md) report accuracy, timeliness, coverage, and IPC. The [70M history check](conventional/605_HISTORY.md) tests what ordered events add beyond the page-PC table.
- The [LSTM design](DESIGN.md) gets its offline inputs and labels from NoPF demand events. The [605 training and replay results](lstm/RESULTS.md) report what the recurrent state learned and how it performed in ChampSim.
- [Measurement definitions](PERFORMANCE.md) follow CMU 18-447. [Conventional baselines](conventional/README.md) explains how to reproduce the screen. [Paper index](PAPERS.md) records related designs.

## Open questions

- Test whether earlier cues change the right candidate when recent events and the candidate page look alike. The 605 results measure reset and shuffle effects under one candidate set and one lead window.
- Compare OMNeT++ with a quiet peer, a second OMNeT++ stream, and an MCF peer on the same simulated hardware. Hold shared cache capacity, queue sizes, DRAM channels, and training event counts fixed. Compare the same per-core trace region with and without shared LLC MSHR occupancy at the scoring head.
- Start with private-cache prefetching disabled to isolate the LLC path. A later run with Berti at L1D and Pythia at L2 can measure what is left for the LLC. Generate labels again under that upper-cache configuration. [DPC4 infrastructure](https://sites.google.com/view/dpc4-2026/submit-work/infrastructure)
