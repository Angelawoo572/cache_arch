# Method and interpretation limits

- This is the separate `four602_frozen_live_20261017_v2` protocol. It preserves the actual four students rather than making their architectures or training budgets identical. Weights are frozen; no teacher action enters the three global students' online forward pass.

| Student | Runtime input and state | Training and decoder |
|---|---|---|
| Retained Stride | PC bits + aligned byte-address bits; input projection; bounded per-PC h/c table | Existing seed-7 balanced checkpoint trained on records 0–1M; Hurdle emit/count and scalar-delta decoder; its existing runtime retains state/FIFO/output capacities and decoder limit |
| Streamer and AMPM | 64 aligned byte-address bits; one global chronological h/c pair; H8 | Actual seed-7 v7 checkpoints trained on records 0–20M; Poisson-mode count, initial decoder `tanh(linear(h))`, mixture4 delta head and GRU feedback; default L2 target |
| SPP | 64 aligned byte-address bits plus demand/fill kind; one global chronological h/c pair; H8 | Actual seed-7 empirical-prior v2 checkpoint trained on records 0–20M; categorical emit, positive exponential count, decoder starts from h, mixture4 delta head and learned L2/LLC fill head |

- Thus this comparison does not isolate teacher choice while holding training data, model size, feature set, loss and deployment state organization fixed.
- The three global models have no per-PC state table, FIFO or imposed decoder32 truncation.
- The retained Stride student has an explicit current-line output filter; the three global students and native legacy arms keep their original current-line outputs.
- Native forward1/forward2 remain distinct controls.
- A single seed for those retained global checkpoints does not establish their training-seed robustness.
- The new validation is records 20–25M.
- The first evaluation ROI, 25–50M, was already observed in historical model work; the later 75–100M ROI is reported separately.
- Existing replay IPC and the earlier Stride protocols are not interchangeable result rows.

- The [adapter](../runtime/four602_engine.cc#L118) receives SPP fill events from both actual raw L2 fill call sites, including demand and writeback fills.
- The model sees the **evicted aligned address**, not the incoming line, and a fill-kind bit; it updates encoder h/c and emits nothing.
- Native SPP keeps its original prefetch-only `cache_fill` forwarding and `FILTER` eviction update.
- Its effective state updates therefore differ from the student's raw fill updates. `fill_callbacks` counts callback opportunities, not native FILTER updates.
- The native confidence thresholds remain prefetch 40 and L2-fill 90, and its L2/LLC destinations are preserved.

- SPP training [canonicalizes repeated teacher targets](../../602_offline_lstm_spp/python/normalize_events.py#L40), retaining first-target order and minimum numeric fill (L2 takes priority over LLC), to represent the queue merge effect. Native SPP still generates its raw source calls. Generated call-list K distributions consequently need not share the training-label denominator. The common external page gate permits cross-page requests, while native SPP retains its own internal same-4KiB-page test; equal external configuration does not remove algorithm-internal page restrictions.

- All methods warm caches without issuing prefetches, while their eligible policy state updates still run.
- Global full-history models carry the actual warmup h/c into ROI; [ROI initialization](../runtime/four602_engine.cc#L59) resets measurement counters, not that h/c.
- Native tables and internal policy counters also persist.
- Prefetches from warmup cannot remain pending because the shared request gate rejects them before queue mutation.
- These global MAC0 models have no asynchronous inference work to drain.
- Ordinary cache/DRAM demand work at the boundary remains part of the common warm-cache simulator state and is recorded separately.

- The [mechanism modes](../runtime/four602_engine.cc#L104) have specific meanings:

- `fixed_payload` executes the complete learned forward pass and updates h/c, preserving learned K, then replaces each final delta/fill slot with its frozen validation modal template. A count exceeding observed template length repeats the final slot and is explicitly counted. This probes conditional payload value while retaining conditional emit/count and full decoder work.
- `template` uses only frozen modal count and per-slot delta/fill, relative to the current address. It executes no NN and has no recurrent history. It is an unconditional template control, not a learned gate.
- `reset_state` resets h/c before **every** demand or fill callback, then runs the ordinary complete model. It removes carried history while retaining the current callback input and the model's feed-forward computation.

- For an identical current input/state, fixed-payload replacement occurs after forward computation and cannot change that step's h/c or recorded operations. Different final requests can nevertheless change later cache contents, timing, eligible L2 LOADs and fill callbacks in the closed loop. The experiment does not assert equal whole-run callback or h/c trajectories between arms.

- Budgeted rows share the validation-selected cumulative ceiling, one request credit per validation NoPF eligible L2 LOAD rate expressed per million retired instructions.
- This gates generated request calls before cache acceptance; it does not force equal realized requests, PQ insertions, DRAM transactions or usefulness.
- Actual gaps and both uncapped/capped results must remain visible.
- L2-target and LLC-target usefulness use separate lifecycle cohorts and consumer scopes, as specified in [MEASUREMENT_SCOPE](../MEASUREMENT_SCOPE.md); they cannot be summed into a claimed L2 accuracy or coverage.

- The LLC-target cohort's `enqueued` denominator is the first **source L2 PQ insertion with an intended LLC fill**, not the number of actual downstream LLC PQ insertions.
- It therefore includes source requests later resolved as L2 cache/MSHR redundancy or LLC WQ-forward/PQ-merge redundancy.
- Where tracking is available, `LOAD_accuracy_enqueued` means `(timely_LOAD + late_LOAD observed by ROI end) / source-cohort enqueued`.
- It is an ROI-window observed-usefulness ratio, not fully resolved eventual lifetime accuracy.
- Pending and resident-unused requests at ROI end are right-censored; they are neither assigned future usefulness nor treated as proven useless/pollution.
- Downstream rejection or a missing lifecycle ID disables complete-cohort accuracy/timeliness rather than inventing an outcome.
- The L2-target cohort uses its own source insertion denominator and L2 LOAD consumers; LLC LOAD consumers include all LOAD packets reaching LLC.

- Target groups are fixed by **first source insertion intent**, not eventual minimum fill level.
- Native PQ/MSHR merges can upgrade an initial LLC-intent packet to L2 while retaining its original high-bit measurement ID.
- A later call merged into the same source PQ does not create a second L2 cohort entry.
- The upgraded packet's L2 copy carries no L2 observer ID; its original ID continues to track the LLC-level lifecycle.
- A later L2 hit on that upgraded copy is therefore outside both the initial-L2 subset and the LLC LOAD-usefulness numerator.
- If a later L2 request instead receives its own source insertion and L2 MSHR, its own low-bit ID remains attached to that L2 MSHR when data returns; `return_data` does not replace it with a lower-level ID.
- These are source-intent, target-level cohort metrics, not a partition of every final fill or all cross-level benefits.
- The historical field name `llc_only_lifecycle_accuracy_available` must not be read as proof that every initially LLC-targeted request remained LLC-only.
- Upgrades alone do not invalidate the narrow initial-intent/LLC-level lifecycle metric; an eventual-fill grouping or request-wide any-level accuracy is unmeasured.

- The current global-model integration is synchronous **MAC0**.
- Its [MAC and nonlinear counters](../../../common/live602_runtime/live602_runtime.cc#L37) count actual dense model operations, including full raw decoder work before payload replacement or output gates.
- They do not model service time, initiation interval, queue admission, weight bandwidth, nonlinear throughput, energy or native-predictor compute cost.
- Weights are resident; there is no modeled NN weight DRAM transfer. [Storage measurements](COST.md) distinguish payload, C++ requested heap and explicit exclusions.
- These runs can establish live functional/closed-loop behavior, but cannot establish that the three global students retain benefits under finite hardware cost.

- Static review found no blocking mismatch in the stated D/F routing, fill destinations, ROI state retention, output-only fixed-payload ordering or MAC0 accounting.
- The installer adds bounds guards and defined unsigned SPP address arithmetic; these are correctness safeguards, not a new predictor algorithm or a general static proof of all SPP lookahead bounds.
- Numerical certification compares the actual built scalar source and frozen checkpoints on contiguous real callbacks and boundary/reset fixtures.
- Formal results still require complete runs and lifecycle/data checks; a correctness fixture is not performance evidence.
