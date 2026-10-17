# Four-route measurement and input scope

- This protocol adds its observer through `install.py` only in a fresh isolated build. The historical dynamic observer, completed binaries and results are unchanged. MAC0 is synchronous functional inference: it does not establish a finite hardware cost or give native SPP an equivalent compute model.

## Separate target cohorts

- Requests enter a lifecycle only at their first actual L2 PQ insertion, after common output gates and queue duplicate checks.
- Low-half IDs identify requested L2 fills; high-bit IDs identify requested LLC fills.
- Packet copies preserve the ID through PQ/MSHR forwarding; the LLC does not allocate a replacement ID.
- L2 fill/hit/late/invalidate events affect the L2 observer, and LLC events affect the LLC observer.
- Source-L2 cache/MSHR redundancy resolves the corresponding requested-target observer, including an LLC-target source request that never needs to reach LLC.
- ID-domain exhaustion is rejected explicitly.

- At LLC `add_pq`, a source request forwarded by the write queue or merged into an existing prefetch queue entry is separately counted and resolved as **redundant-inflight**.
- It creates no new resident-prefetch lifecycle.
- The write-forward count is not LOAD usefulness and is not a new DRAM bus transfer.
- The underlying CACHE's original return-data path, fill-level behavior and queue scheduling are preserved.
- LLC hits resolve redundancy; a demand merged into a prefetch MSHR resolves late usefulness before the native code replaces the MSHR packet.
- A matching fill becomes resident-unused until first demand or eviction.
- Pending and resident-unused at the boundary remain censored.

- A downstream PQ rejection or shortcut with a missing pending ID is explicitly counted. Either disables `cohort_all_enqueued_requests_fully_tracked` and `llc_only_lifecycle_accuracy_available`. Analysis must suppress complete-cohort accuracy/timeliness when those flags are false. Such a path is not filled in as a success or a useful request. `l2_cohort_llc_target_excluded` means exclusion from the L2 cohort, while LLC targets have a separate reported cohort.

- The two levels' useful consumers have different scopes: LLC LOAD consumers include all LOAD packets reaching the LLC, while the L2 cohort follows L2 LOADs. Report separate denominators and scopes. Do not add both cohorts into one asserted L2 accuracy/coverage or reinterpret the overall IPC/DRAM counters. Real DRAM bytes continue to use controller events, independently of these measurement-only lifecycle IDs.

## Native SPP versus the trained SPP student

- The historical training logger records both raw CACHE L2-fill call sites, including demand/writeback fills.
- The trained SPP student therefore processes all those raw fill events as state updates (using the evicted address), with no fill-event output.
- Native SPP's original registry forwards its FILTER feedback only when `prefetch==1`.
- The Four adapter deliberately preserves that native prefetch-only feedback.
- Both see the actual cache callback opportunities, but their **effective input/update semantics differ**.
- This is not strict equality of effective state-update inputs.
- The result report must retain this limitation when interpreting native-versus-student comparisons.
- The native fill policy and confidence-based L2/LLC destinations are preserved.

## Correctness evidence without a simulator trial

- `validation/test_installer.py` generates the full observer and rewrites the actual archived CACHE source, compiles both and the Four adapter with actual ChampSim headers, then runs a standalone C++ measurement unit.
- It checks six routing cases, eight lifecycle cases and five shortcut cases, including LOAD usefulness, eviction, source redundancy, queue forwarding, queue merging, rejection and missing IDs.
- Every cohort must conserve enqueued outcomes.
- It also compiles and executes the exact generated LLC JSON helper and parses the output to check escaping and reported counter meanings.
- No simulator ROI, training prefix or pilot is run by this correctness check.

- From the checkout root with cache-owned `TMPDIR`, `PYTHONPYCACHEPREFIX`, and `MPLCONFIGDIR` already exported:

```sh
python3 formal_NN_training/experiments/602_four_prefetcher_formal/validation/test_installer.py \
  --simulator /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261016/gate_output/raw/builds/build_20261016_213702/simulator \
  --out /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/runtime/four602_measurement_fixture
```

- On Sacramento replace the local cache prefix with `/home/qianruw/cache`. The fixture contains the exact generated `.cc` files, compiler output, standalone test binary, parsed JSON and `verification.json`. These temporary compiler artifacts stay in cache; they are not committed as run data. A new formal build still requires its normal whole-simulator compilation before execution.
