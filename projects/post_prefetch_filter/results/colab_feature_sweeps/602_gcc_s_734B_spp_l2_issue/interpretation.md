# Interpretation

[602.gcc_s-734B]

candidate useful rate=0.9297, MSHR avg=0.027, MSHR max=3

F0 issued_ratio=0.9805, accuracy=0.9467, useful_kept=0.9972, bad_suppressed=0.2439

Behavior class: TRUST_SPP / DO-NO-HARM. SPP L2-scope candidates are already highly useful, and resource pressure is very low. The filter should mostly admit candidates rather than aggressively suppress them.

F1_candidate_mshr_pq is identical to F0_candidate in this run, which is expected because MSHR pressure is almost absent.

F2_add_recent_accuracy and F3_add_bandwidth do not materially improve the policy. F4_add_cache_pressure changes estimated reward only by a tiny amount.

Research meaning: 602.gcc is the trust-baseline sanity case. It shows the controller needs a mode that recognizes when the baseline prefetcher is already good and should not be disturbed.
