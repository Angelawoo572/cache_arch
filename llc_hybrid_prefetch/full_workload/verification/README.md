# Repeating a full interval on Linux

The cross-host control uses GAP PageRank-3, the same complete source object and placement slot 0. Build the existing conventional simulator and the passive native instrument on each host using that host's compiler. Validate/decompress the complete PageRank object on Linux. Run the first 50M warmup and following 200M measured instructions under NoPF, Stride, Multi-delta and SPP independently, with two-core geometry and one active core. The PageRank interval that continues to EOF is a separate result.

From the repository root on Linux, after the original conventional simulator build:

```sh
p=llc_hybrid_prefetch/full_workload
mkdir -p "$p/raw/traces"
g++ -std=c++17 -O3 "$p/intake/verify_records.cc" -o "$p/raw/traces/verify_records"
python3.11 -B "$p/intake/fetch.py" pr_heldout
python3.11 -B "$p/intake/prepare.py" pr_heldout
python3.11 -B "$p/native/build.py" --jobs 4
python3.11 -B "$p/native/batch.py" --jobs 4 --cases pr_heldout --policies nopf stride multi spp
```

Bring the four independently completed folders to a separate comparison directory. Each needs its completion marker, core/phase/statistics tables, page map and compressed outcome stream. Run `python3 -B llc_hybrid_prefetch/full_workload/verification/cross_host.py PATH_TO_INDEPENDENT_RUNS`. It compares configurations and physical page mappings by value, and compares counter files and every decompressed dynamic outcome word byte for byte. The order of hash-table rows in the page-map export carries no timing meaning. The native metrics encompass the prescribed simulated execution and cache model. Reproducing those metrics on another host cannot establish representativeness of the underlying program inputs.
