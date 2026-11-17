# LSTM prefetching at the shared LLC

**Research question.** When does a core's earlier L2D miss history tell us which previously seen page and line to prefetch?

- The [V1 vs. Newest comparison](lstm/v5_results/v1_comparison/README.md) replays both frozen LSTMs on five identical 605 regions. On the two original V1 test regions, exactly 2,000 requests per model per region raise pooled coverage from 0.710% to 2.641%, with nearly equal completion time. The report starts with one real PC and cache-line decision. [Report source](report/llc_lstm_report.tex)

## Earlier experiments

- The [NoPF-only retrain and prior comparisons](lstm/V5_RESULTS.md) record the timing-provenance correction and the conventional baseline experiments.
- The [development experiments](lstm/V4_DEVELOPMENT.md) separate candidate reach, exposed retirement stalls, timing features, and recurrent scoring.
- The [605 baselines](conventional/605_RESULTS.md) report accuracy, timeliness, coverage, and IPC. The [70M history check](conventional/605_HISTORY.md) tests what ordered events add beyond the page-PC table.
- The [LSTM design](DESIGN.md) gets its offline inputs and labels from NoPF demand events. The [605 training and replay results](lstm/RESULTS.md) report what the recurrent state learned and how it performed in ChampSim.
- The [v2 results](lstm/V2_RESULTS.md) compare learned address proposals, measured fill-time supervision, and paced LLC replay on three new 605 regions.
- The [v3 scorer experiment](lstm/V3_RESULTS.md) holds those candidates fixed and measures nonlinear scoring, recurrence, and losses with NoPF supervision.
- [Measurement definitions](PERFORMANCE.md) follow CMU 18-447. [Conventional baselines](conventional/README.md) explains how to reproduce the screen. [Paper index](PAPERS.md) records related designs.

## Scope

- Test whether earlier cues change the right candidate when recent events and the candidate page look alike. The 605 results measure reset and shuffle effects under one candidate set and one lead window.
- Run the same frozen model with one 605 and one 602 stream to vary interference. The two mixed runs are in the [v5 results](lstm/V5_RESULTS.md). The development MSHR experiment isolates a shared-input signal. A distinct core-count comparison would need fixed shared resources and training event counts.
- Start with private-cache prefetching disabled to isolate the LLC path. A later run with Berti at L1D and Pythia at L2 can measure what is left for the LLC. Generate labels again under that upper-cache configuration. [DPC4 infrastructure](https://sites.google.com/view/dpc4-2026/submit-work/infrastructure)
