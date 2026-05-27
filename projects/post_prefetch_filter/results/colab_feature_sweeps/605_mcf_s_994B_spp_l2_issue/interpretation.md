# Interpretation

[605.mcf_s-994B]

candidate useful rate=0.0339, MSHR avg=0.664, MSHR max=11

F0 issued_ratio=0.0301, accuracy=0.7733, useful_kept=0.9641, bad_suppressed=0.9930

Behavior class: FILTER_BAD_GENERATOR. Candidate identity is enough to suppress most bad SPP candidates while keeping most useful ones.

F1_candidate_mshr_pq is identical to F0_candidate in this run, so this run does not prove MSHR/PQ-aware filtering. MSHR exists but does not add decision power beyond PC/delta/confidence under this scope.

F2_add_recent_accuracy and F4_add_cache_pressure reduce estimated reward relative to F0. Do not claim those features help yet.

Research meaning: 605.mcf is the low-trust SPP case. The controller should aggressively filter or disable/switch the prefetch generator, rather than blindly trusting all SPP L2 candidates.
