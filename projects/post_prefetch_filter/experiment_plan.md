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

## Phase 1: shadow candidate logging

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

## Phase 2: oracle filter upper bound

Before training any NN/RL, compute an oracle filter from logged outcomes within the fixed candidate scope:

```text
admit only candidates that were useful and timely
```

This gives the maximum possible value of filtering on each trace. If oracle filtering barely helps, ML will not help either.

## Phase 3: simple supervised utility filter

First real model:

```text
features → logistic regression / perceptron → admit/suppress
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

## Phase 4: RL policy

Only after Phase 3 works.

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

## Phase 5: compare GRU

GRU should be tested only as an offline feature extractor or confidence predictor:

```text
recent access/candidate sequence → predicted usefulness probability
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
