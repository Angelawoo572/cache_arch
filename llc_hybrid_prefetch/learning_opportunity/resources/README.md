# LLC decision bandwidth

Across the eleven external Stride runs, mean trigger rates ranged from 2.98 arrivals per thousand cycles in OMNeT++-874B to 22.50 in CactuBSSN-2421B. Short bursts varied even more. Only 0.46% of adjacent GCC-734B arrivals were less than four cycles apart, compared with 45.28% in BC-3. A four-cycle initiation interval would encounter very different buffering demands in those programs. [Per-workload counts and gap distributions](external_at_ingress.csv)

The [original collection](at_ingress.csv) and [external Stride collection](external_at_ingress.csv) count every LLC demand admission in each audited run. Adjacent gaps include simultaneous arrivals and span compression boundaries. A gap under four cycles describes a burst that a serial unit with a four-cycle initiation interval would face. Queues, arbitration and backpressure determine whether such a unit actually drops a prediction. Its result latency sets a separate deadline. Occupancies are sampled at demand admission. Cycle-weighted occupancy and byte traffic also appear in the [original performance table](../../full_workload/native/performance.csv) and the [expansion table](../expansion/results/performance.csv).

MSHR occupancy includes all outstanding LLC misses. DRAM queues belong to the memory controller. Transporting that state to the LLC, or transporting retired load stall history from a core, would have a latency and storage cost. A replacement victim may have had no future use. The [timing replays](../trials/README.md) count original timely first uses both gained and displaced. Those net effects include changes to traffic, execution order and cache residency.

The long sequence runs activate one core in the shared two-core geometry. The matched active-pair experiment joins operands on core and dynamic ID and reports elapsed time per core. For BFS under SPP, 17,474 loads went from timely alone to late with MCF and 2,184 changed the other way. [Paired and solo comparison](../../full_workload/interference/README.md) · [Per-core measurements](../../full_workload/interference/per_core.csv) · [Shared traffic](../../full_workload/interference/shared_system.csv)

From the cache root:

```sh
python3 -B llc_hybrid_prefetch/learning_opportunity/resources/measure.py
```

## Service time of a correctly predicted line

The [request timing table](request_service.csv) samples 8,192 evenly spaced LLC admissions per arm in MCF-472B, BFS-3 and Soplex-92B. It keeps original NoPF misses that the arm actually claims. Each target is joined to a physical request ID and its proposal, acceptance, dispatch, fill and first-use timestamps. A late claim can occur at tag lookup after demand admission. The [individual examples](request_examples.csv) preserve that distinction. Every sampled joined request is retained under `raw/service/<case>__<policy>.csv`.

For each target, find the earliest same-page observation within the preceding 64 admissions, counting from oldest to newest. Its lead minus the *observed* accept-to-fill latency is a descriptive margin. Service could change under an earlier request. Only the separate [replay](../trials/README.md) changes service order and measures the resulting time. Run `python3 -B llc_hybrid_prefetch/learning_opportunity/resources/service.py` after the original captures and label indices are ready.
