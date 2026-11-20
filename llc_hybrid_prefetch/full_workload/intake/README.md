# Full instruction recordings and evaluation windows

[Exact archive and simulator input paths](../data/trace_files.csv) · [Where the data live](../data/README.md)

The early roster pairs two generations of MCF with graph traversal and PageRank. It tests changes of program generation, recording, and implementation with affordable two-core simulations. Each row in [selection.csv](selection.csv) identifies a complete, downloadable compressed object, its published byte count, source URL, selection rule and evaluation role. The seven files cover five named program/suite combinations across three suites. There are no workload weights that justify estimating a whole-suite average.

## How the recordings were selected

The roster preceded inspection of simulation gains. For SPEC17 MCF and GAP BFS, take the numerically lowest and highest recording labels in the DPC4 collection. For SPEC06 MCF and GAP PageRank, take the lowest numeric recording label in the named public release. A second, prospectively specified program holdout takes the lowest numeric SPEC06 Soplex region. Voyager identifies an earlier array load whose index is reused by a later vector load reached through either of two branch-dependent PCs in Soplex. This adds a sparse solver with independent program structure. The choice used that published rationale before download or observation of any Soplex gains. This preserves the two existing MCF/BFS research questions and adds a new program generation and a separate graph executable. Changes to the region number establish a different published recording. The GAP filenames do not document which input graph was used. A file can contain a complete recording and still represent only a portion of the original program execution.

| Source and selected recording | Use in this study | Relationship to published research |
|---|---|---|
| SPEC CPU2006, 429.mcf-22B | New generation/binary for the existing MCF question | Voyager's SPEC06 MCF program. Moirai lists MCF regions 51B and 192B. Our selected 22B is a distinct public recording. |
| SPEC CPU2006, 450.soplex-92B | Independent program holdout for the sparse solver, excluded from rule and threshold selection | Voyager figure 16 analyzes the shared index across multiple loads in Soplex. TransFetch also supplies other Soplex recordings. |
| SPEC CPU2017, 605.mcf_s-472B and 605.mcf_s-1644B | First and last numeric MCF recordings, plus a distant window of 472B | Both filenames occur in Moirai's evaluation list. TransFetch supplies SPEC17 MCF under numbered SimPoint aliases. Exact trace identity across differently named releases has not been verified. |
| GAP BFS-3 and BFS-14 | Same GAP traversal algorithm, different recordings. The start of BFS-3 has been examined in the project's short screening runs | Both filenames appear in the TransFetch distribution and Moirai's evaluated list. A2P and Voyager report BFS. The DPC4 release credits ML-DPC for GAP. |
| GAP PageRank-3 | Program/recording withheld from the present fit and in-study choices | TransFetch and Moirai list this filename, and A2P and Voyager report PageRank. Earlier project work examined **Ligra** PageRank on com-lj. This GAP implementation provides an independent executable and trace. The underlying algorithm has been studied before. |

SPEC17 recordings come from DPC3 through the DPC4 release, and GAP recordings from the ML-based Data Prefetching Competition through DPC4. The SPEC06 recordings are from Dimitrios Chasapis's April 2024 deposit, whose metadata gives CC BY 4.0. Public access to a trace archive and a simulator's code license are separate matters. The repository distributes this small registry, validation counters, and scripts. The original compressed recordings and native views stay in the study's raw data directory. [DPC4 provenance](https://github.com/CMU-SAFARI/DPC4#workload-traces) · [DPC4 object index](https://traces.rbera.com/dpc4/manifest.txt) · [SPEC06 deposit](https://zenodo.org/records/10959705) · [Prior project registry](../../workload_expansion/intake/registry.csv)

## Publication precedent and deferred workloads

* **Voyager**. *A Hierarchical Neural Model of Data Prefetching*, ASPLOS 2021, evaluates an LLC prefetcher on irregular SPEC06 and GAP. Its reported set includes SPEC06 MCF, Soplex, OMNeT++, Astar, Xalancbmk and GAP BFS, CC, PageRank. Its 250M-instruction SimPoints, reference SPEC06 input, and stated graph size are specifics of that paper. It selected programs for which its oracle had room to help, so its reported gains cannot set an expectation for this roster. Its memory-only commercial workload experiment had no simulated IPC. [Authors’ paper, §5.1 and figures 5–8](https://www.cs.utexas.edu/~akanksha/asplos21.pdf)
* **TransFetch**. *Fine-Grained Address Segmentation for Attention-Based Variable-Degree Prefetching*, ACM Computing Frontiers 2022, evaluates SPEC06, SPEC17 and GAP with 1M warmup, then 40M training, 10M validation and 50M testing. The authors provide separate instruction and load-only files. Only the former retain the non-load instructions and register/memory fields needed for this fixed-retirement experiment. A filename in the download manifest establishes availability and recording overlap. The paper's group-level reporting does not give a per-file performance result for every manifest entry. [Published paper](https://doi.org/10.1145/3528416.3530236) · [Authors’ full paper](https://arxiv.org/pdf/2205.02269) · [Authors’ file manifest](https://github.com/pengmiao-usc/TransFetch/blob/main/ChampSim/download_links)
* **A2P**. *Accelerating Graph Analytics Using Attention-Based Data Prefetcher*, SN Computer Science, 2024, evaluates BC, BFS, CC, PR and SSSP on ML-DPC data. Its graph-workload precedent and LLC evaluation link directly to our question. A shared upstream collection does not establish that a specific simulation window or graph input matches ours. [Published paper and data-availability statement](https://doi.org/10.1007/s42979-024-02989-w)
* **Moirai**. *From Memorization to Generalization: A Practical Neural Network Prefetching Framework*, ISCA 2026, evaluates an L1D prefetcher. Its released list identifies exact SPEC and GAP filenames and its separate trace archive has five parts. Comparability here concerns workload choice. Our shared LLC sees a filtered demand stream and implements different timing. [Published paper](https://doi.org/10.1109/ISCA66397.2026.00161) · [Authors’ evaluated trace list](https://github.com/comp-arch-research/Moirai_ISCA2026/blob/main/ISCA26_AE.tlist) · [Authors’ trace deposit](https://zenodo.org/records/19447159)

| Requested family | Candidate programs and disposition |
|---|---|
| SPEC06 | **429.mcf and 450.soplex selected**. 471.omnetpp, 473.astar, and 483.xalancbmk are available in the Chasapis recording deposit and were evaluated by Voyager. They remain expansion candidates. The Moirai evaluation list includes OMNeT++, Astar and Xalancbmk among these three. |
| SPEC17 | **605.mcf selected**. 602.gcc, 607.cactuBSSN, 620.omnetpp and 623.xalancbmk have DPC4 instruction recordings and TransFetch-distributed examples. The Moirai list includes CactuBSSN, OMNeT++, and Xalancbmk. All four are deferred in this initial full-scale comparison. |
| GAP | **BFS and PageRank selected**. Connected Components, Betweenness Centrality and SSSP are available in DPC4 and represented in A2P. The preceding project reserved BC, which remains available for a future independent algorithm test. |

## Validation, phase identity and scope of measurement

All seven objects passed end-to-end decoding and record checks. Each of the four SPEC recordings contains exactly 2,000,000,000 instructions. Each of the three GAP recordings contains 500,000,001. The seven sources total 9,500,000,003 instructions and 4,408,779,124 compressed bytes. Each object is fetched through the end of its compressed stream. Intake checks the published byte count, codec termination and integrity status, complete 64-byte little-endian records, nonzero instruction PC and legal branch flags, then counts active source/destination memory and register operands across the **entire** stream. The record has an eight-byte PC, two one-byte branch flags, two destination and four source register bytes, followed by two destination and four source 64-bit memory slots. A zero operand slot is inactive. Nonzero operand counts establish that the corresponding fields are populated. They do not establish instruction-by-instruction agreement with the original program or real hardware physical placement. The earlier native header check independently specifies these offsets. [Existing simulator reader check](../../workload_expansion/intake/check_reader.cc) · [ChampSim source record declaration](https://github.com/CMU-SAFARI/ChampSim/blob/master/inc/trace_instruction.h)

The seven original archives remain complete. For replay, [views.csv](views.csv) fixes the raw record ranges. Each primary view starts at zero and includes 250M records plus 1,024 unused guard records. The last 250M records in BFS-3 and MCF-472B form two additional, nonoverlapping windows. The simulator can end at the last record. A fresh 50M warmup precedes each 200M measured retirement interval. The two MCF region files and two BFS recordings add recording-level comparisons. For a separate start-through-EOF comparison, `pr_heldout_entire` materializes all 500,000,001 GAP PageRank records. Its first 250M instructions overlap the primary `pr_heldout` view. They cannot be counted as independent samples. Retaining complete compressed archives supports revisiting other ranges without storing multiple 128 GB decoded recordings.

The far views preserve simulated physical identity by seeding the page allocator from all preceding source records. The validator records unique 4 KiB virtual page numbers in encounter order, first the instruction PC, then destination memory slots, then source memory slots. The seed file uses one unsigned 64-bit little-endian VPN per first observation. Seeding restores prior address allocation. Caches, predictor state and timing start at the view's own warmup. The seed enters the offline setup for every baseline equally. Each `<view>.validation.csv` reports full-source counters. The duplicate counters for the far view of the same source must be counted only once.

## Reproduce

Run from the cache checkout with C++17, Python 3, curl, gzip and xz. Allow approximately 181 GB of local storage for the frozen view set and archives. The first intake check verifies seven publisher lengths and exercises the record/seed validator on controlled fixtures. The final command also checks every saved archive size, view length and page-seed cardinality.

```sh
source .runtime/project_env.sh
cd llc_hybrid_prefetch/full_workload
mkdir -p raw/traces
c++ -std=c++17 -O3 intake/verify_records.cc -o raw/traces/verify_records
python3 -B intake/check_intake.py --online
python3 -B intake/fetch.py --jobs 3
for view in soplex06_heldout mcf06_discovery mcf17_discovery mcf17_transfer bfs_discovery bfs_transfer pr_heldout bfs_discovery_far mcf17_discovery_far pr_heldout_entire
do
  python3 -B intake/prepare.py "$view" || exit 1
done
python3 -B intake/check_intake.py --online --artifacts
```

A restarted download resumes its partial archive. An incomplete or corrupt source leaves no newly approved view. For replay use only the final `.champsimtrace` file with its corresponding validation row. The published trace length, materialized size and simulated instruction count answer three different questions.
