# Experiment Plan

## Phase 0: confirm local prefetcher options

From the ChampSim checkout:

```bash
find prefetcher -maxdepth 2 -type f | sort
find prefetcher -maxdepth 2 -type d | sort
```

Look for:

```text
no
next_line
ip_stride
spp_dev
```

Then build two baselines:

```bash
# no prefetch baseline
./config.sh champsim_config_no_prefetch.json
make -j

# candidate-generator baseline
./config.sh champsim_config_spp.json
make -j
```

Exact config names depend on the local ChampSim version.

## Phase 1: aggregate SPP baseline

Before ML/RL, record simulator-level behavior for each trace.

Required table per workload:

```text
trace
IPC
L1D/L2C/LLC access, hit, miss
L1D/L2C/LLC hit rate
L1D/L2C/LLC miss rate
L1D/L2C/LLC MPKI
prefetch issued/useful/useless/accuracy
```

This is the latency/performance layer:

```text
IPC up/down       = latency/performance effect
hit rate up/down  = cache effectiveness
miss rate/MPKI    = pressure on lower levels
prefetch accuracy = useful / issued
coverage          = useful / demand misses, when available
```

Do not use final IPC/hit-rate/miss-rate as online input features. They are evaluation metrics. Online features must come from candidate-time state or previous-window counters.

## Phase 2: shadow candidate logging

Goal: run an existing prefetcher, generate candidates, but log candidate outcomes.

For each candidate, log:

```text
cycle
cpu
level
trigger_pc
trigger_addr
candidate_addr
candidate_delta
prefetcher_confidence if available
whether SPP would fill L2 / issue high-confidence request
cache_hit_state at generation time
mshr_occupancy
pq_occupancy
recent_bw_usage
set_occupancy / set_pressure if easy
```

Then log delayed outcome:

```text
filled_cycle
first_demand_use_cycle
was_used_before_eviction
was_late
was_duplicate
was_evicted_unused
```

### Important controlled-variable lesson

The first event logger records many SPP lookahead/candidate attempts, not only the high-confidence candidates counted by `SPP_FINAL`.

This caused the first Colab plot to collapse:

```text
all SPP candidate attempts
  -> extremely low useful-label rate
  -> bandit learns suppress almost everything
  -> issued_ratio ~ 0 and accuracy ~ 0
```

That result is a diagnostic, not the real filter experiment.

The controlled first experiment must fix the candidate scope:

```text
CANDIDATE_SCOPE = spp_l2_issue
condition       = spp_fill_l2 == 1 or spp_confidence >= 90
```

This asks the cleaner question:

```text
Among candidates SPP itself would issue to L2,
can a tiny filter suppress the bad/resource-risky ones?
```

Only after this controlled scope works should we expand the action space to all lookahead candidates, LLC-only candidates, fill-level control, or degree control.

## Phase 3: notebook feature sweep

The notebook is not final performance evaluation. It is a controlled feature-ablation tool.

Hold fixed:

```text
workload
candidate scope
train/eval split
reward definition
model family
```

Then change exactly one feature group at a time:

```text
F0: candidate identity only
F1: + MSHR/PQ pressure
F2: + recent usefulness feedback
F3: + bandwidth pressure
F4: + cache context / set pressure
```

For every feature set, record:

```text
issued_ratio
accuracy
useful_kept_ratio
bad_suppressed_ratio
estimated_reward
num_states
```

Interpretation pattern:

```text
accuracy improves but useful_kept_ratio drops a lot
  -> filter is over-suppressing; may hurt IPC

issued_ratio drops and accuracy stays high, useful_kept remains high
  -> promising traffic reduction

F1 changes result only when MSHR pressure exists
  -> resource-aware filter is workload/phase dependent

F3 changes nothing if bandwidth_bucket is still constant 0
  -> need real bandwidth logging before making claims
```

## Phase 4: oracle filter upper bound

Before trusting any NN/RL, compute an oracle filter from logged outcomes within the fixed candidate scope:

```text
admit only candidates that were useful and timely
```

This gives the maximum possible value of filtering on each trace. If oracle filtering barely helps, ML will not help either.

## Phase 5: simple supervised utility filter

First real model:

```text
features -> logistic regression / perceptron / small table -> admit/suppress
```

Start with hashed features:

```text
PC hash
candidate delta bucket
page offset
cache level
prefetcher confidence bucket
MSHR occupancy bucket
prefetch queue occupancy bucket
bandwidth bucket
recent useful-prefetch EWMA for this PC
recent pollution EWMA for this PC
```

Labels:

```text
1 = useful and timely
0 = unused / evicted unused / duplicate / late
```

Run:

```text
baseline prefetcher alone
vs.
baseline prefetcher + filter threshold sweep
```

## Phase 6: replay selected policies in ChampSim

This is where hit-rate/miss-rate/IPC are answered.

For each workload and selected policy:

```text
no prefetch
SPP baseline
SPP + filter F0
SPP + filter F1
SPP + filter best
```

Record:

```text
IPC and speedup vs SPP baseline
L1D/L2C/LLC hit-rate delta
L1D/L2C/LLC miss-rate delta
MPKI delta
prefetch issued/useful/useless delta
prefetch accuracy delta
coverage delta
MSHR/PQ pressure delta
```

This turns notebook insight into architecture evidence.

## Phase 7: RL policy

Only after Phase 5/6 works.

State:

```text
same features as supervised filter + resource pressure
```

Actions:

```text
0 suppress
1 admit normal priority
2 admit only if resource pressure low
3 admit to lower cache level / low priority if simulator supports it
```

Reward:

```text
+ useful_timely_bonus
- unused_prefetch_cost
- late_prefetch_cost
- bandwidth_cost
- pollution_cost
- MSHR_pressure_cost
```

Start with tabular Q-learning over bucketized features or contextual bandit. Avoid deep RL until a simple policy shows promise.

## Phase 8: compare GRU

GRU should be tested only as an offline feature extractor or confidence predictor:

```text
recent access/candidate sequence -> predicted usefulness probability
```

Do not put GRU in the first hardware path. If GRU improves only slightly over perceptron/logistic but costs much more, it is not the right design.

## Stop/go criteria

Continue if at least one trace shows:

```text
same or higher IPC
higher prefetch accuracy
lower prefetch traffic
no obvious MPKI regression
```

Strong result:

```text
higher IPC and lower traffic on bandwidth-sensitive traces
```

Weak but still publishable project-result:

```text
same IPC with much less traffic and higher accuracy
```
