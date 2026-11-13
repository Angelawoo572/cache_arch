# Moving conventional prefetchers to the LLC

- **Trigger:** L2 demand data-load misses that enter the LLC read queue. Preserve the requesting core and physical line. Use the post-tag callback for outcomes. Pythia v1.3 ships an empty LLC module and conventional algorithms behind the L2 module. [LLC queue and callbacks](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc) · [L2 module](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/multi.l2c_pref)
- **Stride:** Use each core's PC and recurring delta. The stock generator includes the current line in slot zero. Skip it before applying a future-candidate limit. [Source](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/stride.cc)
- **Streamer:** Reuse the per-page direction tracker. It can run without PC. [Source](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/streamer.cc)
- **AMPM:** Reuse page access maps and their spatial patterns. Per-core tables keep unrelated streams separate. [Source](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/ampm.cc) · [Paper](https://jilp.org/vol13/v13paper3.pdf)
- **SPP:** Keep page signatures, delta confidence, and lookahead. Route candidates through the common LLC issuer and adapt the original split-level feedback to an LLC-only target. [Source](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/spp_dev2.cc)

The [conventional baselines](conventional/README.md) run NoPF, Next-Line, Stride, Streamer, AMPM, and SPP with matching cores, traces, cache/DRAM geometry, and request limits. A restricted `+1/+2/+4` offset table is a useful later control for the LSTM's candidate set. [Best-Offset](https://comparch-conf.gatech.edu/dpc2/resource/dpc2_michaud.pdf)

This build keeps one DRAM channel and uses the upstream LLC capacity, queue sizes, and read width, which scale with core count. The upstream build script also adds a DRAM channel above two cores. Hold shared resources fixed when isolating the effect of more input streams. [Upstream build](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/build_champsim.sh#L64-L77) · [Cache geometry](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/inc/cache.h#L67-L79)
