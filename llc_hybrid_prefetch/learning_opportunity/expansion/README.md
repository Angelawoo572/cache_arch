# External recordings and physical placement

The source register was fixed before collecting predictor outcomes. GCC-734B, OMNeT++-874B and Xalancbmk-700B reuse complete archives already in the cache. CactuBSSN-2421B connects to an earlier short experiment. CC uses the lowest numbered available GAP CC recording. BC-3 was reserved in the previous study. PageRank-14 supplies a second recording. For SPEC06 we use the lowest numbered OMNeT++ and Astar recordings, the next Soplex recording after 92B, and MCF-51B, which Moirai includes. These new runs are independent of fitting on the original discovery workloads. Exact graph identities for numbered GAP recordings are unpublished. The named SNAP input comparison has its own protocol. [Source files](sources.csv) · [Publication precedents](../../full_workload/intake/README.md) · [DPC4](https://github.com/CMU-SAFARI/DPC4) · [SPEC06 source deposit](https://zenodo.org/records/10959705)

The SPEC06 Astar, OMNeT++, MCF and Soplex programs and the GAP BFS, CC and PageRank algorithms appear in Voyager's evaluation. A2P evaluates GAP graph analytics on ML prefetching competition traces, including CC and BC. TransFetch publishes its trace preparation and predictor code. The SPEC17 additions cover a compiler, grid solver, event simulation and tree processing. Paper precedent is at the program or algorithm level unless the exact recording is identified in the source register. Hardware configurations and reporting windows differ across these papers. [Voyager, ASPLOS 2021](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf) · [TransFetch authors' artifact](https://github.com/pgroupATusc/TransFetch) · [A2P, SN Computer Science 2024](https://doi.org/10.1007/s42979-024-02989-w) · [Moirai, ISCA 2026](https://ieeexplore.ieee.org/document/11617757)

Each archive is validated through decompressor EOF. The simulator reads the first 50M warmup and 200M measured instructions. Counts in the validation records refer to the entire decoded source and the guarded native view respectively. Each source is run with NoPF, Stride, Multi-delta and single-candidate SPP. All use the existing shared two-core geometry with core 0 active. A second four-policy run of GCC and CC uses address placement slot 1, moving physical frames by 64 pages. That comparison preserves relative offsets between physical frames. All performance comparisons use simulated completion cycles for identical retired work. The wall-time field records how long a host took to run the simulator.

From the cache root, build the previous full-workload native simulator, then run the expansion. Raw archives, derived native trace views and one complete recorded LLC arrival sequence per run are under `llc_hybrid_prefetch/learning_opportunity/raw/expansion/`. The compact data and measurement files live on the research branch.

```sh
p=llc_hybrid_prefetch/learning_opportunity
python3 -B llc_hybrid_prefetch/full_workload/continuous/build.py
python3 -B "$p/expansion/run.py" --decoders 3 --jobs 14
python3 -B "$p/expansion/summarize.py"
```

The eleven-recording panel covers SPEC and graph programs. The [cross-domain check](../domains/README.md) applies the same protocol to two Google and AI/ML discovery traces. The reserved files retain their prior assignments. [Google and AI/ML registrations](../../workload_expansion/intake/registry.csv)

For a workspace replica, the first host validates the compressed source through EOF. The [replica reader](replicate_input.py) checks the second host's entire compressed object against that source, decodes the required native view, and verifies its byte count and content against the first host. This avoids decoding billions of trailing instructions twice. Its checks are for accidental corruption. The primary source register retains the independent full-record validation.
