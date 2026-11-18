# Does older context help on another trace point?

- This descriptive test uses solo NoPF traces. Estimate same-page displacement frequencies in the first point; evaluate on the second point of the same program. No parameters or thresholds are tuned on that second point.
- The anchor is an L2 demand load admitted to the LLC. For each of twelve signed offsets (1, 2, 4, 8, 16 and 32 lines in either direction), label the next ingress to that target positive if it produces an LLC miss and it arrives between one and four observed median service times later. The median is measured independently per run and used for offline labels. Drop the final, right-censored anchors.
- Compare three frequency summaries: current PC, exact page offset and current per-core delta; those fields plus the preceding four delta tokens in order; and those fields plus the very same tokens sorted. Within-page deltas are clipped to ±32. A crossing has its own token.
- Each more detailed key needs eight training observations and is smoothed with sixteen counts from the less detailed distribution. Unsupported contexts fall back. Report how often the more detailed key has support on the second point.
- Give all versions the same retrospective issue budget of 2% of eligible anchors and one same-page candidate per selected anchor. Compare distinct future misses reached, which avoids counting the same target repeatedly. This score assumes an offline budget allocation and a service-time window. It cannot establish actual timeliness, cycle savings, or the merit of a particular network.

Run from the study directory after the base matrix completes:

```sh
python3 features/history.py raw/runs results/history_associations.csv
case_name=syn-paced_nested_samepage-stride
python3 features/counted_boundary.py raw/runs/$case_name raw/diagnostics/counted-paced
python3 experiments/run.py --out raw/counted --run "$case_name" --env "HYBRID_PLAN=$PWD/raw/diagnostics/counted-paced/plan.csv"
python3 features/counted_boundary.py raw/runs/$case_name raw/diagnostics/counted-paced --verify-replay raw/counted/$case_name --boundaries raw/traces/synthetic-paced_nested_samepage.boundaries.csv.gz --summary results/counted_boundary.csv
```
