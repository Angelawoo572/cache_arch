# Reproduce the paired measurements

Start in this project's llc_hybrid_prefetch directory. On Sacramento use python3.11 in the commands below. The simulator build instructions are in the parent project README.

```sh
python3 experiments/prepare_traces.py
python3 experiments/synthetic.py
python3 experiments/run.py --list
python3 experiments/run.py --group spec --group warmup --jobs 12
python3 experiments/run.py --group synthetic --jobs 4
```

The 28 main runs come from two policies, two public regions, four solo programs, and three two-program mixes. Solo and mixed runs use the identical two-core binary. The peer in a solo run has no trace and issues no requests. Each workload keeps its mixed run's physical placement slot when measured solo. CactuBSSN and Xalancbmk retain slot 1. For shorter warmup the skipped prefix registers the same virtual pages as the main arm without populating cache contents. The initially cold warmup sensitivity replays the MCF–CactuBSSN first-point pair starting at record 5,000,001 with 5M warmup instructions. Both versions then measure original per-core IDs 10,000,001 through 15,000,000. Each input file provides 250,000 records of lookahead beyond that interval. The simulator fixes each core's work count and prohibits wrapping a short trace.

The two highest publicly available source-weighted regions of each program were selected before measurement. The point identifiers and original weights are in matrix.py. The files come from [DPC4](https://github.com/CMU-SAFARI/DPC4#workload-traces). DPC3 supplies the corresponding [SimPoints and weights](https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/weights-and-simpoints-speccpu.tar.gz). The weights pertain to the original phases. Each measurement uses a short prefix of a phase. Report each region separately.

prepare_traces.py reads any existing matching archive in the parent cache/traces directory. Otherwise it requests leading compressed bytes in 4 MiB ranges and decodes a new valid uncompressed prefix. Trace records are 64 bytes. Each generated prefix is checked for exact size and its first records for valid instruction-pointer and branch encodings. raw/transport holds re-downloadable leading compressed byte ranges for reuse across machines. These ranges deliberately do not contain the archive's trailing index. Use the uncompressed files under raw/traces as simulator inputs.

## Run metadata

matrix.json lists the planned arms in machine-readable form. raw/runs/<run_id>/run.json records original trace order, per-core skip, warmup and ROI, policy, exact invocation and simulator environment. complete.json records the simulator exit status and duration after successful execution. The analysis also validates fixed retired counts, demand identities, attribution and other semantic conditions. The runner skips completed arms and reruns previously failed arms. Each independent replay uses a new output directory, including single-arm diagnostic interventions with additional simulator settings.

```sh
python3 experiments/run.py --run mcf-cactus-r1-stride --out raw/intervention/example --env NAME=VALUE
```

## Synthetic controls

These are explicitly generated instruction traces. Each loop emits four instructions with a fixed load PC. A trace contains arithmetic, the load, a dependent consumer and a taken branch. Source-level labels are supplied separately for retrospective analysis. All controls use 100,000 instructions of warmup and 500,000 measured instructions. The tight controls emit one load every four instructions. The paced control emits one load every 260 instructions. Virtual addresses are 64-byte aligned. Inspect forwarded LLC triggers before assuming every programmed load reached the predictor.

| Case | Programmed memory access |
|---|---|
| steady | Successive cache lines with no padding |
| nested | 16 contiguous lines per row, then 16 unused lines |
| nested_samepage and nested_crosspage | Identical PC and line delta history. Two 16-line rows per trial, starts separated by 32 lines. Adding a 32-line displacement to the entire trace causes the second row to cross the 4 KiB page boundary. Compare matched paired_second_row events within each trial. |
| paced_nested_samepage | The same virtual addresses as nested_samepage with a 256-instruction dependent arithmetic chain from the preceding load to the next address generation. The load arrival rate and the count of measured loads differ. |
| nested_variable | Fixed 32-line row pitch and a repeating sequence of 8, 20, 12 and 24 accessed lines per row |
| interleaved | A single load PC alternates between two disjoint unit-stride streams |
| phase | 32 unit-stride lines alternate with 32 stride-three lines in repeated blocks |
| recurring | The same eight irregular offsets recur in successive disjoint 32-line regions |

Each raw/traces/synthetic-*.source.json has the instruction-to-load mapping and event counts by category, warmup/ROI, and virtual page crossing. The matching boundaries.csv.gz provides the dynamic load instruction ID, ordinal, previous and current virtual line, and the ground-truth event label. Boundary annotations are offline evaluation data. The finite sequence of synthetic addresses encodes an access pattern and a small generic instruction stream. It does not model an actual compiled program, branch-prediction behavior for different source loops or real-application prevalence.

Use `python3 experiments/summarize_boundaries.py` to count paired outcomes by ground-truth category from completed synthetic arms. Its output counts each original load once and separates late demand merges from filled-before-first-use hits.
