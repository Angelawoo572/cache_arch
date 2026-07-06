# V4.8 validation record

## Review basis

The design was cross-checked against the supplied V3.3–V3.9 notebooks, original V4.0 notebook and replay evidence, V4.1 ledger diagnosis, V4.7 notebook and replay-analysis bundle, the existing script README, and `lec1.intro.pdf.pdf`.

The decisive replay evidence used in the V4.8 design is:

- 602: full V4 association route was accurate/timely but stayed below sandbox; residual coverage remains the bottleneck.
- 605: V4.7 support16 + fixed next-line route beat AMPM, so Stage B freezes that route and changes only widths.
- 619: current full route remained below SMS and showed a degree/duplicate/coverage issue; V4.8 runs three seeds and degree variants before any size sweep.
- 620: residual-pair route improved coverage and misses but remained below SMS; V4.8 adds general causal profiled PC strides before capacity changes.
- 623: context route beat SPP strongly; Stage B freezes it and changes only widths.

## Checks completed

1. All notebook code cells were parsed with Python `ast.parse`; no syntax errors remained.
2. The full V4.0 model configuration instantiated at exactly **4,147,959 trainable parameters**.
3. The fixed-architecture size ladder instantiated at:
   - full: 4,147,959
   - approximate 911K: 910,879
   - approximate 383K: 382,889
   - nearest feasible ~100K preserving all hash-table feature groups: 152,891
4. The generic `profiled_pc_stride` source passed a synthetic signed-delta test containing both `+3` and `-5`; it does not hardcode a PC, address, or stride.
5. The standard-library post-replay analysis script compiled with `py_compile` and ran successfully on synthetic replay/normal/outcome/resource CSVs. It produced the all-candidate table, route seed-variance table, Stage-B gate table, smallest accepted model table, and all-normal IPC comparison table.
6. The notebook uses `int(len(frame))` for exported-row accounting; it does not cast a DataFrame directly to `int`.
7. Every V4.8 rich-list tag includes trace, route, seed, size, and policy tag, preventing the V4.2/V4.7 tag and list-count collisions.
8. The notebook does not use pandas on Sacramento post-analysis; the generated server script uses only existing replay scripts plus a Python-standard-library analysis file.

## What cannot be claimed yet

No local static check can prove that a full Colab training run will complete on the actual oracle files or that a candidate exceeds a normal prefetcher. Those claims require the generated keyed ChampSim replay. The notebook and README state this explicitly and record all needed replay gates.
