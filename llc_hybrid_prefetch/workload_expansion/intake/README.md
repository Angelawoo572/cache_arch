# DPC4 workload intake and compatibility

The discovery roster contains seven Google service labels, four graph workloads, three SPEC anchors and two AI/ML programs. Six named future-test objects remain unopened. An additional 74 objects share reserved family labels and are also excluded from discovery. Seven Google labels are separate workload families as named by the provider. Their application implementations and degree of independence cannot be established from the catalog names. [Registered inputs](registry.csv) · [Complete catalog with reserved-family flags](manifest_index.csv)

## Source and selection

The official index generated on August 20, 2026 lists 610 trace objects and a four-core mix description. It enumerates 359 Google, 92 SPEC17, 79 Graph and 80 AI/ML recordings. Publication of the Google originals and conversion to ChampSim are distinct steps. DPC4 credits Google for the originals and Matthew Giordano and Akanksha Jain for conversion. The DPC4 repository links its simulator and index. The original Google distribution supplies usage terms in its own license file. Other collections have their respective provenance in the DPC4 acknowledgements. The public research artifacts store identifiers, metadata and measurements; prefixes reside in the local and Sacramento data directories. [DPC4 source and acknowledgements](https://github.com/CMU-SAFARI/DPC4) · [Official manifest](https://traces.rbera.com/dpc4/manifest.txt) · [Google V2 license](https://storage.googleapis.com/external-traces-v2/LICENSE.txt)

Selection was frozen after reading the filenames and before reading any newly selected trace body. Choose the lower median file in numeric filename order within each declared family. The four familiar anchors use their exact pre-existing DPC4 objects. No IPC, MPKI, Stride success or learning score entered this decision. There was no resampling after measuring. The catalog labels align with actual collection capitalization.

| Discovery panel | Rationale and contrast |
|---|---|
| Google Arizona, Charlie, Delta, Merced, Tahoe, Tango, Yankee | One object per different directory label. Keep Bravo, Whiskey and the full Sierra family outside discovery. All three Sierra directory labels are reserved together. |
| GAP BFS-3, GAP SSSP-5, GMS clique count, Ligra PageRank | Retain one familiar traversal control, compare another traversal, set-based clique processing and a graph iteration. Reserve all GAP betweenness-centrality objects. Different graph algorithms may share inputs or libraries. |
| SPEC MCF-665B, Xalancbmk-700B, CactuBSSN-2421B | Previously characterized reference objects for comparing the short-screen methodology against established code paths. Reserve SPEC gcc. The previous LBM findings remain available as motivating evidence. |
| CLIP, Llama2.c stories15M | Contrasting application labels for model-related CPU traces. Reserve all Whisper objects. |

Six named test files are metadata reservations. The full reserved bands include all Bravo and Whiskey objects, all three Sierra directories, all GAP bc, all Whisper and all 602.gcc regions. Other unused source objects remain unselected. This split demonstrates procedural separation in this study; it gives no evidence that all reserved applications or shared libraries have statistically independent behavior. [Selection code](register.py) · [Registry](registry.csv)

## What the bounded sample contains

Each discovery object yielded 801,024 complete 64-byte instruction records. Four archived prefixes carry the exact same official URL and were reused. Twelve new inputs required 30,408,704 compressed bytes in total. Each HTTP range was bounded to 1 MiB and checked against the exact manifest object length before decoding; decoding stops after reaching the registered record count. Partial prefixes cannot validate a full compressed stream's trailer. The raw manifest, range evidence and source metadata remain with the prefixes. A local source can be absent on a new machine; the script then fetches the registered URL. [Per-input acquisition](acquisition.csv)

Each fixed ROI has 250,000 instructions. In original zero-based record positions, the two ROIs are [100,000, 350,000) and [550,000, 800,000). The second run skips 450,000 instructions and warms independently for 100,000. The 1,024 extra records serve as a reader guard. These are contiguous-in-time regions of a selected object. They do not establish generalization to other regions, applications or host machines. Because DPC4 sources can be captured slices, record zero denotes the start of the downloaded object.

## Record format and limits on interpretation

The DPC4 simulator's ordinary trace record and this project's Pythia reader agree on the 64-byte layout. The branch flags follow the 8-byte PC. Two destination and four source register slots follow, then two destination and four source 64-bit memory-address slots. Nonzero addresses occupy active slots. The Pythia reader attaches zero-based source slot IDs to load operands and maps the input addresses through the project's simulated physical-page allocator. The original Google workloads offer a separate plausible 2 MiB mapping; this project's 4 KiB placement belongs to the experiment. The fixed 64-byte record lacks an explicit process ID, original timestamp, operand size, instruction category, real ISA opcode and a full multithread schedule. [DPC4 trace record](https://github.com/CMU-SAFARI/ChampSim/blob/master/inc/trace_instruction.h) · [Google V2 format](https://dynamorio.org/google_workload_traces.html)

The independent NumPy decoder validated all 12,816,384 prefix records for nonzero PCs, binary branch/taken flags and taken-branch consistency. A second program compiled directly against the simulator's input header checked the record size and every field offset, then independently confirmed branch count, taken count, load and store operand counts, load instruction count and ROI length in all 32 windows. It also measured register uses whose most recent register-writing instruction in the preceding 256 input records had loaded memory. Within the 14 Google ROIs, 20–38 distinct nonzero source register IDs occurred and 1,283–38,120 loads per window had such a preceding load writer. A register match counts a possible trace-level dependence; it does not distinguish address calculation, value use, an overwritten subregister or the critical path. When the source supplies more operands than the format's fixed slots can encode, the converted file alone cannot reveal which ones were omitted. [Slot-level audit](format_audit.csv) · [Native decoder cross-check](reader_compatibility.csv)

Google V2's synthetic ISA retains register-dependency information. The well-known concerns about missing nonbranch dependencies in a 2023 converter discussion addressed V1; attributing them automatically to the V2 Google originals or this conversion would be misleading. For these converted records we can establish that register fields are populated and reach Pythia's register-dependency logic. No original-to-converted instruction-by-instruction equivalence or original core schedule was demonstrated. Google recommends scheduling all active application threads simultaneously and warns that isolated cores can misrepresent its services. Treat simulated IPC and ROB-head exposure as paired results for the specified single-core windows. [Google V2 scheduler guidance](https://dynamorio.org/google_workload_traces.html) · [Earlier converter review](https://github.com/ChampSim/ChampSim/pull/379)

The same reader checks explain a prespecified negative case. Each SSSP-5 ROI contains five distinct instruction PCs, 50,000 store operands and zero load operands. An empty load-triggered LLC stream is an expected observation. PageRank has 83,333 loads in each ROI; private-cache filtering determines how many reach the LLC. Compare the input counts above with the simulator's aligned trace, cache lookups and page maps in the screening results. [Screening protocol](../screen/protocol.md)

## Reproduction

From this directory, use Python with NumPy and a C++ compiler. The header from the existing parent simulator build belongs under `../../raw/build-2/pythia/inc`. The committed registry can drive fetch directly. To reconstruct the selection from the official object index and reproduce the checks, run

```sh
python3 -B register.py --refresh-manifest
python3 -B fetch.py --jobs 4
python3 -B audit.py
python3 -B verify_reader.py
```

The fetcher accepts the discovery roster only. Future test names have no local payload paths. The audit verifies that the set of downloaded trace files exactly matches the discovery roster.

The subsequent [Google support-sensitivity follow-up](long_protocol.md) uses the same seven discovery objects, an independently warmed deeper interval and separate prefix paths. Both reader implementations also validate those extended inputs.
