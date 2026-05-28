# Formal NN Training: SPP Output Schema Draft

This folder is a planning/specification folder for SPP-based NN training. It does **not** define the final project direction yet. It only lists all SPP outputs that can be logged from ChampSim `spp_dev` and explains what each field means.

The purpose is to decide later which fields are good NN inputs, which fields are labels/rewards, and which fields are only for debugging.

---

## 0. Big picture

SPP should be viewed as an observable memory-behavior sensor, not only as a prefetch-address generator.

```text
Demand access: addr / ip
        |
        v
ST: page -> signature
        |
        v
PT: signature -> delta candidates + confidence
        |
        v
Candidate generation: base_addr + delta -> pf_addr
        |
        v
SPP decision path: threshold / same-page / filter / prefetch queue
        |
        v
Cache behavior: issued / filled / used / evicted
```

SPP internal structures:

```text
ST      Signature Table: tracks page-local access history as signatures
PT      Pattern Table: maps signatures to likely next deltas
FILTER  Prefetch Filter: tracks duplicate/useful prefetches
GHR     Global History Register: handles cross-page lookahead state
```

---

# 1. SPP_FULL_LOG

`SPP_FULL_LOG` is the event-level log. It records what SPP sees and decides during each access / candidate-generation step.

Useful event types:

```text
ACCESS    one demand access entering spp_dev::prefetcher_cache_operate
PT_READ   one Pattern Table lookup / lookahead step
CAND      one prefetch candidate produced by SPP
FILL      one cache-fill / eviction callback
```

For NN training, the most important rows are usually `CAND` rows, but `ACCESS`, `PT_READ`, and `FILL` help build phase summaries and labels.

---

## 1.1 Event identity fields

### `event`

Type of row.

Examples:

```text
ACCESS
PT_READ
CAND
FILL
```

Use:

```text
Useful for filtering rows.
Not a model input by itself.
```

### `event_id`

Monotonically increasing event number.

Use:

```text
Debug / ordering only.
Do not use as NN input.
```

### `cand_id`

Monotonically increasing candidate id for candidate rows.

Use:

```text
Debug / join key only.
Do not use as NN input.
```

---

## 1.2 Current demand-access fields

### `addr`

Raw demand access address.

Meaning:

```text
The real address currently accessed by the program.
```

Use:

```text
Debug only, or label matching.
Do not feed raw address directly to NN because it overfits trace layout.
Prefer page_offset, observed_delta, or hashed/low-bit address features.
```

### `addr_line`

Demand access cache-line number.

Meaning:

```text
addr_line = addr >> LOG2_BLOCK_SIZE
```

Use:

```text
Useful for computing relative distance.
Do not feed raw addr_line directly.
```

Better derived feature:

```text
relative_distance = pf_line - addr_line
```

### `page`

Page number of current demand access.

Use:

```text
Debug / grouping / page transition analysis.
Do not feed raw page directly.
```

Better derived features:

```text
same_page
page_crossing_rate
page_reuse statistics
```

### `page_offset`

Offset of current demand access inside the page, usually in cache-line units.

Use:

```text
Strong NN input.
SPP is page-local, so page_offset is semantically meaningful and more general than raw address.
```

### `ip`

Instruction pointer / program counter of the memory access.

Use:

```text
Strong NN input if hashed or embedded.
Do not use raw full PC naively unless model/table is intentionally PC-indexed.
```

Suggested representation:

```text
ip_hash
ip_low_bits
PC embedding index
```

### `cache_hit`

Whether the current demand access hit in this cache level.

Use:

```text
Good online state feature.
Can help summarize locality and whether cache currently covers this stream.
```

### `useful_prefetch`

Whether this demand access used a previous prefetch.

Important distinction:

```text
This is feedback about an earlier prefetch.
It is not the future label for the current candidate.
```

Use:

```text
Good recent-feedback feature.
Useful for phase summaries, recent usefulness rate, and SPP health.
```

### `access_type`

Access type enum from ChampSim.

Meaning:

```text
Load / RFO / prefetch / writeback / translated access type depending on ChampSim enum.
```

Use:

```text
Potentially useful NN input.
Different access types may have different prefetch utility.
```

### `metadata_in`

Metadata entering the prefetcher hook.

Use:

```text
Mostly debug for current SPP, because SPP usually passes prefetch metadata as 0.
Can become useful if later we add custom metadata.
```

---

## 1.3 Signature and global-feedback fields

### `last_sig`

Previous signature for the current page before the new access updates ST.

Meaning:

```text
Used with observed_delta to update PT:
(last_sig, observed_delta) -> PT counter update
```

Use:

```text
Potential NN input if hashed/table-indexed.
Useful for understanding what pattern SPP just learned from.
```

### `curr_sig`

Current signature after ST update.

Meaning:

```text
Used to read PT and generate candidate deltas.
```

Use:

```text
Good input if hashed/table-indexed.
Very central to SPP behavior.
```

### `observed_delta`

Actual delta between current page offset and previous page offset for this page.

Meaning:

```text
observed_delta = current_page_offset - previous_page_offset
```

Use:

```text
Very strong NN input.
This is the direct observed access-pattern feature.
```

### `global_accuracy`

SPP's global useful/issued estimate.

Meaning:

```text
global_accuracy ~= 100 * pf_useful_ctr / pf_issued_ctr
```

Use:

```text
Strong input for adaptive SPP, phase detection, and confidence calibration.
```

### `pf_issued_ctr`

SPP global issued-prefetch counter.

Use:

```text
Better used in ratios or log-scaled form.
Raw counter can overfit time.
```

Better derived features:

```text
log2(pf_issued_ctr + 1)
pf_useful_ctr / max(pf_issued_ctr, 1)
```

### `pf_useful_ctr`

SPP global useful-prefetch counter.

Use:

```text
Useful with pf_issued_ctr to compute SPP reliability.
Do not use raw alone as main input.
```

---

## 1.4 Hardware resource pressure fields

These are important if the model/control policy cares about resource pressure, timeliness, bandwidth, or memory-system congestion.

### `mshr_occ`

Current MSHR occupancy.

Meaning:

```text
Number of outstanding cache misses tracked by this cache.
```

Use:

```text
Very strong input for resource-aware control.
```

### `mshr_size`

Total MSHR capacity.

Recommended derived feature:

```text
mshr_ratio = mshr_occ / mshr_size
```

### `rq_occ`

Read queue occupancy.

Use:

```text
Useful for lower-level read pressure.
```

### `rq_size`

Read queue capacity.

Recommended derived feature:

```text
rq_ratio = rq_occ / rq_size
```

### `wq_occ`

Write queue occupancy.

Use:

```text
Useful for writeback / memory pressure.
```

### `wq_size`

Write queue capacity.

Recommended derived feature:

```text
wq_ratio = wq_occ / wq_size
```

### `pq_occ`

Prefetch queue occupancy.

Use:

```text
Very strong input.
If PQ is full, even good prefetches may be delayed or rejected.
```

### `pq_size`

Prefetch queue capacity.

Recommended derived feature:

```text
pq_ratio = pq_occ / pq_size
```

---

## 1.5 Pattern Table lookup fields

These describe why PT produced a candidate.

### `pt_set`

Pattern Table set index.

Meaning:

```text
pt_set = hash(curr_sig) % PT_SET
```

Use:

```text
Good for debugging/table analysis.
Can be used as table-indexed feature, but may overfit if raw.
```

### `pt_way`

Pattern Table way index.

Meaning:

```text
Which way in the PT set supplied the delta/counter.
```

Use:

```text
Potentially useful small categorical input.
```

### `pt_delta`

Delta stored in the selected PT entry.

Meaning:

```text
PT predicts that this signature may be followed by this delta.
```

Use:

```text
Strong input.
Usually same semantic information as cand_delta for candidate rows.
```

### `pt_c_delta`

Counter for this delta under this signature.

Meaning:

```text
How often this delta has been observed for this signature.
```

Use:

```text
Strong input.
Represents evidence strength.
```

### `pt_c_sig`

Total counter for this signature's PT set.

Meaning:

```text
Denominator for local confidence.
```

Use:

```text
Strong input.
Represents sample size / confidence reliability.
```

### `local_conf`

Local confidence from PT only.

Formula:

```text
local_conf = 100 * pt_c_delta / pt_c_sig
```

Use:

```text
Very strong input.
This is SPP's local pattern confidence before deeper lookahead correction.
```

### `pf_conf`

Final prefetch confidence for this candidate.

Meaning:

```text
At depth 0, approximately local_conf.
At deeper lookahead, adjusted by global_accuracy and lookahead_conf.
```

Use:

```text
Very strong input.
This is SPP's own confidence score for the candidate.
```

---

## 1.6 Lookahead and candidate-queue fields

### `lookahead_way`

PT way selected for continuing the lookahead path.

Use:

```text
Useful for understanding SPP's chosen future path.
Potential categorical input for adaptive lookahead/depth modeling.
```

### `lookahead_conf`

Confidence of the selected lookahead path.

Use:

```text
Strong input for timeliness/depth control.
Low lookahead_conf means deep candidates are less reliable.
```

### `pf_q_head`

Head index of SPP's internal prefetch candidate queue.

Use:

```text
Mostly debug.
Not a strong direct NN input.
```

### `pf_q_tail`

Tail index of SPP's internal prefetch candidate queue.

Use:

```text
Mostly debug.
Useful for derived num_candidates.
```

Better derived feature:

```text
num_candidates = pf_q_tail - pf_q_head
```

### `depth`

Lookahead depth.

Meaning:

```text
0 = near candidate after current demand access
1 = one step deeper in predicted future
2+ = further lookahead
```

Use:

```text
Very strong input for timeliness, aggressiveness, and confidence calibration.
```

### `cand_index`

Candidate index in current candidate stream.

Use:

```text
Potentially useful order/priority feature.
Not as fundamental as cand_delta/cand_conf/depth.
```

---

## 1.7 Candidate address fields

### `base_addr`

Base address used to generate the current candidate.

Meaning:

```text
pf_addr = block_number(base_addr) + cand_delta
```

Use:

```text
Raw base_addr is not good input.
Use relative features derived from it.
```

### `pf_addr`

Prefetch address candidate.

Use:

```text
Debug / label matching.
Do not feed raw pf_addr directly to NN.
```

### `pf_line`

Prefetch cache-line number.

Use:

```text
Use for relative distance calculation.
Do not feed raw pf_line directly.
```

Better derived feature:

```text
pf_line - addr_line
```

### `pf_page`

Page of prefetch candidate.

Use:

```text
Debug / same-page analysis.
Raw pf_page should not be direct input.
```

### `pf_page_offset`

Page offset of prefetch candidate.

Use:

```text
Good input.
SPP is page-local, so candidate page offset is meaningful.
```

### `cand_delta`

Candidate delta in cache-line units.

Use:

```text
Very strong input.
This is the core identity of the candidate prediction.
```

### `cand_conf`

Candidate confidence.

Use:

```text
Very strong input.
Often same as pf_conf on candidate rows.
```

---

## 1.8 Candidate decision-path fields

These describe the path from candidate generation to prefetch issue.

### `threshold_pass`

Whether candidate confidence passed SPP's prefetch threshold.

Meaning:

```text
cand_conf >= PF_THRESHOLD
```

Use:

```text
Useful but mostly derived from cand_conf.
```

### `fill_l2`

Whether SPP chooses to fill this level, usually L2.

Meaning:

```text
fill_l2 = cand_conf >= FILL_THRESHOLD
```

Use:

```text
Very useful if studying fill-level control or SPP aggressiveness.
Can also be a target/action in adaptive SPP.
```

### `same_page`

Whether candidate prefetch address is in the same page as current demand address.

Use:

```text
Very strong input.
SPP normally issues only same-page candidates; cross-page cases go through GHR.
```

### `filter_pass`

Whether SPP's internal filter allowed this prefetch candidate.

Meaning:

```text
False usually means duplicate/already-tracked line.
```

Use:

```text
Useful for diagnosis.
Use carefully as model input because it may encode a decision already made by SPP.
```

### `issued`

Whether ChampSim accepted the prefetch request from `prefetch_line`.

Use:

```text
Do not use as online NN input.
It is an outcome after the decision.
Useful for labels, statistics, and queue rejection analysis.
```

### `ghr_update`

Whether a cross-page candidate was stored into GHR.

Use:

```text
Useful for page-boundary behavior and GHR effectiveness analysis.
Potential input for cross-page / adaptive GHR modeling.
```

---

## 1.9 Fill / eviction fields

These mainly matter on `FILL` rows.

### `evicted_addr`

Address evicted during a cache fill.

Use:

```text
Do not use as online input for candidate generation.
Useful for constructing pollution labels.
```

### `set`

Cache set index for the fill/eviction.

Use:

```text
Debug or set-pressure analysis.
Raw set can overfit; use derived set-pressure features if needed.
```

### `way`

Cache way index.

Use:

```text
Mostly debug / replacement analysis.
Usually not a direct NN input.
```

### `prefetch`

Whether the fill was caused by prefetch.

Use:

```text
Useful for label construction and separating demand fills from prefetch fills.
Not an input for a decision that happens before fill.
```

### `metadata_out`

Metadata returned from prefetcher/cache hook.

Use:

```text
Mostly debug in current SPP.
May become useful if custom metadata is added later.
```

---

# 2. SPP_TABLE_LOG

`SPP_TABLE_LOG` is snapshot-level. It dumps the whole internal state of ST / PT / FILTER / GHR at selected times.

It is not meant to be directly fed to a neural network as one giant vector. It is mainly for:

```text
debugging
SPP behavior analysis
feature engineering
table compression / behavior-class discovery
```

---

## 2.1 Common snapshot fields

### `snapshot_id`

Id of this snapshot.

Use:

```text
Debug / grouping only.
Do not use as NN input.
```

### `reason`

Why this snapshot was taken.

Examples:

```text
ACCESS_PERIODIC
FINAL
MANUAL_DEBUG
```

Use:

```text
Debug only.
```

### `access_count`

Number of accesses seen by the SPP logger so far.

Use:

```text
Timeline/debug only.
Avoid direct NN input.
```

### `addr`, `ip`, `last_sig`, `curr_sig`, `observed_delta`, `depth`

Context of the access that triggered the snapshot.

Use:

```text
Same meanings as in SPP_FULL_LOG.
Useful to join table state back to the triggering access.
```

---

# 3. ST snapshot: `ST_ENTRY`

ST maps recent page behavior to a signature.

```text
page identity + last page offset -> current signature
```

Fields:

```text
event = ST_ENTRY
snapshot_id
reason
access_count
addr
ip
last_sig
curr_sig
observed_delta
depth
st_set
st_way
st_valid
st_tag
st_last_offset
st_sig
st_lru
```

### `st_set`

Signature-table set.

Use:

```text
Debug/table analysis.
```

### `st_way`

Way inside the ST set.

Use:

```text
Debug/table analysis.
```

### `st_valid`

Whether the ST entry is valid.

Use:

```text
Good for table occupancy analysis.
```

### `st_tag`

Partial page tag stored in ST.

Use:

```text
Debug only.
Do not use raw tag as NN input.
```

### `st_last_offset`

Last offset seen for this page.

Use:

```text
Potentially useful feature if using ST-entry summaries.
```

### `st_sig`

Signature stored for this page.

Use:

```text
Potentially useful if hashed/table-indexed.
Useful for behavior-class discovery.
```

### `st_lru`

LRU state of the ST entry.

Use:

```text
Mostly debug / replacement behavior.
Usually not direct NN input.
```

---

# 4. PT snapshot: `PT_ENTRY`

PT maps signatures to candidate deltas.

```text
curr_sig -> {delta, counter} entries
```

Fields:

```text
event = PT_ENTRY
snapshot_id
reason
access_count
addr
ip
last_sig
curr_sig
observed_delta
depth
pt_set
pt_way
pt_delta
pt_c_delta
pt_c_sig
pt_local_conf
```

### `pt_set`

Pattern-table set.

Use:

```text
Debug/table index.
Maybe useful for table-indexed models.
```

### `pt_way`

Way in the PT set.

Use:

```text
Small categorical feature if looking at selected PT entry.
```

### `pt_delta`

Stored delta candidate.

Use:

```text
Strong feature when this entry is relevant to current curr_sig.
```

### `pt_c_delta`

Counter for this delta.

Use:

```text
Strong evidence-strength feature.
```

### `pt_c_sig`

Total counter for the signature set.

Use:

```text
Strong sample-size feature.
```

### `pt_local_conf`

Local confidence for this PT entry.

Formula:

```text
pt_local_conf = 100 * pt_c_delta / pt_c_sig
```

Use:

```text
Strong confidence feature.
```

Good derived PT summary features:

```text
top1_delta
top1_conf
top2_delta
top2_conf
confidence_gap = top1_conf - top2_conf
num_nonzero_entries
PT entropy / uncertainty
```

---

# 5. FILTER snapshot: `FILTER_ENTRY`

FILTER tracks duplicate and useful prefetch state.

Fields:

```text
event = FILTER_ENTRY
snapshot_id
reason
access_count
addr
ip
filter_set
filter_valid
filter_useful
filter_remainder_tag
```

### `filter_set`

Filter index, derived from hash quotient of cache line.

Use:

```text
Debug/table analysis.
```

### `filter_valid`

Whether this filter entry currently tracks a prefetched line.

Use:

```text
Good for duplicate-prefetch/usefulness analysis.
```

### `filter_useful`

Whether the tracked prefetched line has been used.

Use:

```text
Good for feedback analysis.
```

### `filter_remainder_tag`

Remainder tag to reduce aliasing in the filter.

Use:

```text
Debug / aliasing analysis.
Usually not direct NN input.
```

Good derived FILTER features:

```text
filter_occupancy_rate
filter_useful_rate
candidate_filter_match
candidate_filter_valid
candidate_filter_useful
```

---

# 6. GHR snapshot: `GHR_ENTRY`

GHR stores cross-page lookahead state.

Fields:

```text
event = GHR_ENTRY
snapshot_id
reason
access_count
addr
ip
ghr_index
ghr_valid
ghr_sig
ghr_confidence
ghr_offset
ghr_delta
ghr_global_accuracy
ghr_pf_issued
ghr_pf_useful
```

### `ghr_index`

GHR entry index.

Use:

```text
Debug / small categorical feature if using GHR directly.
```

### `ghr_valid`

Whether this GHR entry is valid.

Use:

```text
Useful for cross-page state analysis.
```

### `ghr_sig`

Signature stored for cross-page bootstrap.

Use:

```text
Potential input if modeling cross-page behavior.
```

### `ghr_confidence`

Confidence of the cross-page GHR entry.

Use:

```text
Strong feature for cross-page / GHR usefulness modeling.
```

### `ghr_offset`

Page offset stored in GHR.

Use:

```text
Useful for matching new-page page_offset to old cross-page prediction.
```

### `ghr_delta`

Delta stored in GHR.

Use:

```text
Useful for cross-page delta prediction.
```

### `ghr_global_accuracy`

Global SPP accuracy at snapshot time.

Use:

```text
Strong phase-level reliability feature.
```

### `ghr_pf_issued`

Global issued counter.

Use:

```text
Use as ratio/log feature, not raw time-like counter.
```

### `ghr_pf_useful`

Global useful counter.

Use:

```text
Use with ghr_pf_issued to estimate reliability.
```

Good derived GHR features:

```text
num_valid_ghr_entries
max_ghr_confidence
best_matching_ghr_confidence
page_crossing_rate
ghr_update_rate
ghr_match_rate
```

---

# 7. Summary rows: `SNAPSHOT_BEGIN` / `SNAPSHOT_END`

These mark the boundary of one full table snapshot.

Fields:

```text
event = SNAPSHOT_BEGIN / SNAPSHOT_END
snapshot_id
reason
access_count
addr
ip
last_sig
curr_sig
observed_delta
depth
```

Use:

```text
Grouping/debug only.
Do not use as direct NN input.
```

---

# 8. Recommended feature groups by possible research direction

## 8.1 SPP behavior classification

Goal:

```text
Classify what kind of behavior phase SPP is currently in.
```

Good window-level inputs:

```text
avg(cand_conf)
max(cand_conf)
confidence entropy
num_candidates_per_access
avg(depth)
max(depth)
same_page_rate
page_crossing_rate
ghr_update_rate
issued_rate
useful_prefetch_rate
global_accuracy
mshr_ratio_avg
pq_ratio_avg
cache_hit_rate
PT confidence distribution
PT nonzero-entry count
```

Possible outputs:

```text
TRUST_SPP
NOISY_PT
LOW_CONFIDENCE
DEEP_LOOKAHEAD_RISK
PAGE_BOUNDARY_LIMITED
RESOURCE_PRESSURED
TIMELINESS_RISK
```

## 8.2 Adaptive SPP control

Goal:

```text
Choose SPP operating mode per phase/window.
```

Good inputs:

```text
recent global_accuracy
recent useful / issued ratio
recent candidate count
recent high-confidence candidate ratio
recent depth distribution
recent page-crossing rate
recent MSHR ratio
recent PQ ratio
recent hit/miss trend
```

Possible outputs/actions:

```text
conservative SPP mode
default SPP mode
aggressive SPP mode
LLC-only mode
shallow-lookahead mode
GHR-enhanced mode
```

## 8.3 Learned PT / delta prediction

Goal:

```text
Predict next delta candidates or a delta distribution.
```

Good inputs:

```text
ip_hash
page_offset
observed_delta history
curr_sig / curr_sig_hash
last_sig / last_sig_hash
PT top-k deltas
PT top-k confidence
GHR matching entry
recent delta histogram
```

Possible outputs:

```text
next_delta
top-k deltas
next_page_offset
delta distribution
```

## 8.4 Confidence calibration

Goal:

```text
Estimate whether SPP confidence is actually reliable.
```

Good inputs:

```text
cand_conf
local_conf
pf_conf
pt_c_delta
pt_c_sig
depth
lookahead_conf
global_accuracy
ip_hash
page_offset
observed_delta
```

Possible outputs:

```text
calibrated usefulness probability
calibrated timeliness probability
calibrated pollution risk
```

## 8.5 Timeliness / distance prediction

Goal:

```text
Predict when the candidate will be used or whether it is too early/late.
```

Good inputs:

```text
cand_delta
cand_conf
depth
lookahead_conf
page_offset
curr_sig_hash
ip_hash
recent reuse distance
mshr_ratio
pq_ratio
cache_hit
global_accuracy
```

Possible outputs:

```text
time_to_use bucket
reuse_distance bucket
recommended fill level
recommended lookahead depth
```

---

# 9. Fields that are usually good NN inputs

Good candidate/event-level inputs:

```text
ip_hash
page_offset
observed_delta
curr_sig_hash
last_sig_hash
cand_delta
cand_conf
local_conf
pf_conf
pt_c_delta
pt_c_sig
depth
cand_index
lookahead_conf
fill_l2
same_page
global_accuracy
mshr_ratio
pq_ratio
rq_ratio
wq_ratio
cache_hit
recent useful_prefetch rate
```

Good phase/window-level inputs:

```text
candidate_count_per_access
avg_confidence
high_confidence_ratio
confidence_entropy
avg_depth
max_depth
same_page_rate
page_crossing_rate
ghr_update_rate
issued_rate
useful_prefetch_rate
cache_hit_rate
mshr_ratio_avg
pq_ratio_avg
PT top-k confidence summary
FILTER useful/valid ratio
GHR valid/confidence summary
```

---

# 10. Fields usually not good direct NN inputs

Avoid raw ids:

```text
event_id
cand_id
snapshot_id
access_count
```

Avoid raw addresses unless intentionally doing trace-specific analysis:

```text
addr
addr_line
page
pf_addr
pf_line
pf_page
st_tag
filter_remainder_tag
```

Avoid future/outcome leakage as input:

```text
issued
evicted_addr
prefetch on fill rows
metadata_out after decision
```

These are better for labels, debugging, or analysis.

---

# 11. Important warning about labels

`useful_prefetch` is not automatically the label for the current candidate.

It means:

```text
The current demand access used some previous prefetch.
```

A candidate-level label must be constructed by matching:

```text
CAND pf_addr
    -> future demand access to same line
    -> before eviction?
    -> before too much delay?
    -> did it hide latency?
```

Possible candidate labels:

```text
future_used
future_used_before_eviction
late_prefetch
early_evicted_unused
pollution_caused
useful_but_late
useful_and_timely
```

---

# 12. First clean schema to inspect manually

Before choosing the final research direction, inspect these columns first:

```text
event
addr
ip
page_offset
cache_hit
useful_prefetch
last_sig
curr_sig
observed_delta
global_accuracy
mshr_occ
mshr_size
pq_occ
pq_size
pt_set
pt_way
pt_delta
pt_c_delta
pt_c_sig
local_conf
pf_conf
lookahead_way
lookahead_conf
depth
cand_index
base_addr
pf_addr
pf_page_offset
cand_delta
cand_conf
threshold_pass
fill_l2
same_page
filter_pass
issued
ghr_update
```

This is enough to understand what SPP is doing before deciding whether the final direction is adaptive control, learned PT, confidence calibration, timeliness modeling, or something else.
