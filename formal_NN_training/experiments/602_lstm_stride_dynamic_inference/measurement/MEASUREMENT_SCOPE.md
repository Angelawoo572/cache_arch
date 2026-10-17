# Gate-mechanism measurement scope

- Native SPP is unchanged. In the inspected Sacramento ChampSim source, `prefetcher/spp_dev2.cc:113` selects `FILL_L2` or `FILL_LLC` using its native confidence threshold. `src/cache.cc:1147` and `:1176` allocate an MSHR only when `packet.fill_level <= cache.fill_level`. Thus a low-confidence LLC-only request can pass through the L2 PQ without an L2 MSHR, L2 fill, or L2 first-demand use. The existing observer follows L2 only; it does not follow LLC-only usefulness.

- Previous formal-v2 measurement assigned an L2 cohort identity to every L2 PQ insertion regardless of requested fill level.
- It therefore mixed unobserved LLC-only lifecycle outcomes into the L2 denominator and classified many such identities as pending at ROI end.
- For example, preserved 602 ROI0 uncapped SPP has 3,316,942 enqueued cohort identities, 132,456 timely LOAD uses, 406 late LOAD uses, and 1,827,439 unresolved pending identities.
- These old counters do not identify exactly which pending identities were LLC-only.
- Their whole-SPP accuracy/timeliness/coverage claims are unavailable and must not be compared with the corrected L2-target subset.
- This warning does not invalidate directly measured old IPC, cycles, LOAD misses, or actual DRAM events.
- Old files are kept.

- The new protocol assigns a lifecycle identity only when requested fill level is at or above L2 (`fill_level <= FILL_L2` in this source).
- LLC-only PQ insertions are counted in `cohort_llc_only_excluded` and receive identity zero.
- Their classified `l2_cohort`, accuracy and timeliness are JSON null.
- Native SPP's LLC fill behavior and 90 confidence fill threshold are preserved.
- This is a scope fix, not a new SPP fill policy or an LLC lifecycle implementation.

- Use `cohort_enqueued` for accuracy of accepted **fully tracked L2-target PQ requests**.
- LOAD usefulness is `cohort_timely_load + cohort_late_load`; timeliness is timely LOAD / that LOAD usefulness.
- Tracked LOAD uses per L2-target submission can use `request_l2_target_allowed`, not all L2/LLC submissions.
- This is a submission-yield measure, not fully tracked cohort accuracy: the submitted denominator includes cache merges/rejections and possible WQ forwards whose independent usefulness is not followed.
- Useful / matched NoPF LOAD misses describes a **partial L2-target subset** if LLC-only requests exist; it is not overall SPP coverage. `request_cache_enqueued` preserves its historical API-result meaning: add_pq returned -1 (accepted nonmerged), which also includes a possible WQ forward. `request_cache_pq_inserted` is the actual PQ insertion count; it is separate from API acceptance, merge, and rejection.

- `dynamic602_label_request` is a diagnostic ABI.
- The engine labels final cache attempts before vector insertion or its direct cache call.
- The observer consumes labels FIFO by (CPU, IP, base byte address, target byte address), before shared current/page/budget gates.
- Current-line identity uses cache-line units, including an unaligned base.
- Native SPP labels have slot/duplicate/teacher = unknown (-1), rather than invented slot zero or teacher overlap. `DYNAMIC602_ARM` supplies the actual arm.
- This interface does not supply information to NN inference.

- `mechanism_request_profile` from the engine counts generated payload requests before engine output filters. `mechanism_classified_requests` from measurement starts at actual cache calls, after those engine filters. Do not equate these denominators. Each classified row contains arm, fill scope, source phase, payload slot, current-line flag, within-payload duplicate flag, and teacher category (0 overlap / 1 active-extra / 2 silent-extra; unknown = -1).

- Default phase width is 5M retired ROI instructions.
- With the planned 25M ROI, phases are 0..4; the last phase absorbs retirement overshoot.
- Origin phase is payload generation/completion time from the engine label, not its input arrival. `actual_attempt_phase_counts` and `actual_allowed_phase_counts` preserve the gate time separately.
- L2 redundant/cache-hit, redundant/inflight, timely, late, useful, evicted-unused, resident-unused, and pending outcomes remain attributed to the original labeled generation phase even when they finish later.
- Native unlabeled SPP uses its immediate cache-call phase as the origin.
- The older 100k `cohort` issue buckets remain PQ-enqueue-time buckets and are a distinct view.

- The common current-line filter precedes page and cumulative request-budget checks and consumes no budget credit. Cache acceptance does not imply eventual usefulness. Pending and resident-unused are censored; unused is not pollution. All diagnostic labels/maps/profiles are measurement overhead, excluded from deployment storage. Actual IPC and LLC/DRAM transaction instrumentation remain separate from lifecycle attribution.

- Correctness fixtures cover mixed native fill levels, exact zero LLC-only L2 identity, lifecycle conservation, cross-phase ownership, RFO/LOAD distinction, current suppression before credits, final-phase overshoot, actual simulator trace layouts, unchanged conventional Stride, and compilation against the actual simulator headers. They are correctness tests, not workload pilots.
