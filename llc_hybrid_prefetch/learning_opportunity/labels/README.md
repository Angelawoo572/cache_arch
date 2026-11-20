# Continuous LLC histories and future targets

Each dataset contains **every admitted demand in one measured policy execution**, in the order seen by the LLC predictor. Select an anchor by its zero-based arrival rank. Read physical line, PC and simulator cycle at or before the anchor. Query later addresses, their arrival times and aligned outcomes through the separate label interface. [Native capture and cache feedback](../capture/README.md) · [Dataset inventory](datasets.csv) · [Reach and lead findings](RESULTS.md) · [Underlying counts](results/)

## A query at a proposed prediction point

From the cache root, after preparing an indexed dataset:

```sh
cd llc_hybrid_prefetch/learning_opportunity/labels
python3 -B - <<'PY'
from sequence import ROOT, Sequence

s = Sequence(ROOT / "bfs_discovery__stride")
anchor = s.size // 2
history = s.context(anchor, 32, scope="global")
pc_history = s.context(anchor, 16, scope="same_pc")
print(history["pc_sequence"], history["line_deltas"], history["cycle_gaps"])

# Bound targets to the current evaluation partition.
stop = 3 * s.size // 4
target = s.future(anchor, scope="other_pc", stop=stop)
print(target["ranks"], target["valid"], target["lead_cycles"], target["outcomes"])
window = s.time_window(anchor, 128, stop=stop)
print(window["complete"], window["lines"])
PY
```

[Midpoint anchor histories](sample_anchors.csv) · [Matched future target rows](sample_targets.csv) · [Actual future line sets](sample_footprints.csv). These examples fix the anchor at the median arrival rank before checking any future address.

For sequential training, `s.batches(start, stop, history=32, horizons=(1,8,64))` yields separate observation and label tensors in memory-bounded batches. Masks identify an incomplete prefix or censored target. By default both histories and labels stay inside the partition. Use `history_start=0` to carry observable past context into a chronological test segment.

**Causal observations.** Each record stores source instruction ordinal and load operand packed as `ordinal << 2 | operand`, PC, physical cache-line address, and cycle. Each study execution uses one active core in a two-core geometry. Rank orders admissions in the same cycle. Key order can differ from arrival order on an out-of-order core. The page has 64 lines, the region has 4,096 lines (256 KiB), and each line is 64 bytes. The optional `s.pressure(anchor, run=...)` reads counters latched at that ingress in a richer capture and checks the actual key, PC, address and cycle before returning them. A caller can join core-side control or dependency fields using the [source-record reader](../capture/access.py). Transporting such information to the LLC would require an explicit interface.

**Retrospective targets.** Supported scopes are global, same/other PC, same/other page, and same/other region relative to the anchor. Default offsets are 1, 2, 4, 8, 16, 32 and 64 qualifying arrivals. Event gaps count *all* intervening admitted observations. Cycle windows include tied-cycle events later in observer order, and expose both the ordered accesses and their distinct physical lines. A horizon beyond an evaluation boundary has rank −1. Valid target arrays omit these missing ranks. A time window ending at or beyond the last observed cycle is marked incomplete. Require `complete` when evaluating a full footprint.

**Outcome alignment.** Each target has four 4-bit masks in fixed NoPF, Stride, Multi-delta and SPP order. They are matched by absolute source instruction and operand across the four executions. Codes are 0 for no LLC tag/claim on this operand, 1 for an ordinary tag hit, 2 for a plain miss, 6 for a miss with a late merge and 9 for a timely direct first use. Code 0 alone does not identify which private cache supplied an access. Earlier access outcomes may also be unresolved at a new anchor. The history API exposes no mask. To use past feedback as an input, join its completion event only if its observer order is strictly before the current anchor. Labels provide the target's cycle in the selected policy execution. Comparing issue/fill timing across policies requires each policy's own arrival clock.

**Seen and legally expressible.** Exact-line, page, region and PC first observations refer to the measured LLC stream through and including the anchor. The first-touch index tests first rank ≤ anchor. A line absent from this history may have been referenced by the program, private caches, warmup or another core. We report same-page reach, reach using pages in the last 64 ingress records, and an unbounded previously seen-page reference. A generator could select an offset within one of those observed physical pages. The measurements grant hindsight about the offset and page choice and therefore bound vocabulary reach. A future target on another page could be supported by a different physical-construction mechanism. Full-run mappings validate labels offline.

## What was measured

The compressed events and per-chunk outcome labels under `raw/labels/<case>__<policy>/` cover all arrivals. Each independently readable archive holds up to 65,536 observations. The index records chunk rank/cycle bounds and original capture paths. Sorted physical keys and rank postings support arbitrary same-group and other-group history queries without scanning to the end of the trace. Separating observation and outcome files also allows a causal reader to work without opening the labels. [Source and alignment checks](datasets.csv)

For each reported Stride dataset we preselected a uniform sample without replacement of up to 32,768 arrival ranks using seed 448. Sampling uses the full unfiltered sequence. All numerical horizons for a case share these anchors. Counts refer to anchor–target pairs. The same target can appear in multiple scopes or horizons. We applied no address-pattern, hit, miss or criticality filter when sampling.

- `*_horizons.csv` partitions observed targets into previously seen exact lines, new lines on known pages, new pages in known regions and new regions. It counts retained-page membership and joint address/lead/cohort ceilings. The conservative gap cohort requires a NoPF tag miss and a plain miss or a late first use in **all three** comparators. Absent tag lookups and ordinary hits are outside that cohort.
- `*_novelty.csv` splits each global horizon by exact line, page and region novelty. For each reachable target the vocabulary contains 64 offsets per distinct retained physical page. The table counts how many of the conservative gap targets with sufficient lead are new lines and how many address alternatives shared their trigger. A stricter count excludes both a target equal to the trigger line and a line already demanded between trigger and target. Earlier residency, outstanding requests, and the service deadline still matter.
- `*_windows.csv` reports ordered access counts, distinct lines and empty footprints for 32, 128 and 512 cycles after each anchor. The table gives the empirical distribution only over complete windows.
- `*_deadlines.csv` treats each sampled rank as a target, then searches the preceding 64 observations for the earliest legal construction under each vocabulary. A trigger can retain the physical pages of its latest 64 arrivals, including itself. The 130-cycle threshold comes from earlier training-source service diagnostics. The tables also give 64, 134, 146, 194 and 256 cycles for sensitivity. Extra headroom of 4, 16 or 64 cycles measures how many retrospective opportunities tolerate additional decision latency at the fixed 130-cycle reference.
- `*_visibility.csv` joins **all** original NoPF LLC misses from the retirement stream to observed arrivals and outcomes in each execution, including original misses absent at an arm's ingress. This gives the population omitted by an ingress-only cohort.
- `placements.csv` (when produced) pairs physical-placement runs by a unique dynamic anchor operand. It tests whether the two streams meet the same target at the same horizon, whether the signed displacement persists, and whether lead crosses 130 cycles. Changed arrival order and changes in physical line number are counted separately. The full pair of virtual-to-physical maps distinguishes uniform frame shifts from nonuniform placements.

Changing the selector changes which future access counts as the target. To assess the information contributed by historical features, the [held-out probes](../probes/) keep the target definition fixed.

A deadline requires an actual fill under queue and DRAM contention. These lead-time counts use observed time between admissions. The separate [diagnostic replay](../trials/) measures selected actual interventions.

## Reproduce

The following commands use Python 3.10+ and NumPy. The plots also use Matplotlib. Run from this directory. The original complete Stride recordings of BFS-3 and Soplex can be read directly from the previous continuous study. Additional cases use the current capture and expansion directories.

```sh
python3 -B check.py
python3 -B process.py bfs_discovery soplex06_heldout
python3 -B index_all.py bfs_discovery soplex06_heldout --arms nopf multi spp
python3 -B postprocess.py bfs_discovery soplex06_heldout
```

To regenerate the complete published matrix, use the case names in the verified inventory.

```sh
set -- $(python3 -c 'import csv; print(" ".join(sorted({r["case"] for r in csv.DictReader(open("datasets.csv"))})))')
python3 -B process.py "$@"
python3 -B index_all.py "$@" --arms nopf multi spp
python3 -B postprocess.py "$@"
```

Pass additional case names to process or index_all. The `--wait` option follows registered runs that are still in progress. The ingestion check requires a complete capture and all four complete outcome references. It checks event counts, cycle order, retired instruction limits and mask validity. The audit separately checks 65 equispaced source PC/physical-line pairs per stream, all four full page maps and their source intervals, and 260 independently decoded retirement labels per stream. The directed test covers scopes, history truncation, tied timestamps, new regions, an out-of-order source pair, chunk boundaries and censoring.


On machines with slow scratch disks, `index_all.py --memory --mask-cache-mib 2048 --wait -- case1 case2` keeps temporary event and outcome arrays in RAM. The cache limit covers four retired-operand masks for one case. A 200M-instruction case uses 1.60 GB for the four masks. Event and joined outcome arrays require another 34 bytes per admitted demand before NumPy's sort workspaces. A case that exceeds the cache budget reads one mask at a time. The default uses disk-backed scratch and releases each mask after joining it. Both modes write identical logical columns. [Real-recording parity checks](build_parity.csv)

```sh
python3 -B build_parity.py snap_stanford snap_google
python3 -B build_parity.py bfs_discovery --arms stride
```
