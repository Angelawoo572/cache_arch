# Oracle-replacer replay workflow

## Why the first replay issued zero prefetches

The notebook's rich export has this column order:

```text
order,pc,line,issue_prob,addr_conf,...,prefetch_addr
```

For the first generated row, `order` was a **cycle** and the second field was a
**PC**. The old list replayer expected only:

```text
idx,0xprefetch_addr
```

It therefore interpreted the cycle as an index and the PC as a hexadecimal
address. No index matched its access counter, so the simulator reproduced the
no-prefetch IPC with zero requests.

## Correct replay contract

The replay has three independently checked pieces:

1. `10_prepare_oracle_replacer_replay_input.py` maps a rich `(cycle, pc, line)`
   trigger to its no-prefetch oracle `demand_idx` and creates a sparse list:

   ```text
   idx,prefetch_addr
   ```

2. The same converter creates a dense reference for **every** no-prefetch oracle
   row:

   ```text
   idx,pc,line
   ```

3. The Pythia-native L2 `list_replayer` starts after warmup, counts only L2
   LOADs, and compares every runtime callback's `(pc,line)` to that reference
   before it emits a list entry. A mismatch suppresses emission and makes the
   run invalid.

A final counter mismatch by itself is therefore not enough to call a replay
wrong: a few terminal callbacks can differ as the simulator drains/stops. The
runner accepts at most `MAX_TAIL_SLACK` terminal callbacks in either direction
only when all observed callbacks have zero signature mismatches and every list
entry in the observed prefix replayed exactly.

## One-time install/build

```bash
cd ~/cache

git pull
# Preserve the local residual-audit patch if it is uncommitted.
git -C external/ChampSim stash push -u -m "local patch before oracle replayer update"
git -C external/ChampSim pull --ff-only
git -C external/ChampSim stash pop

bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh
```

This produces:

```text
external/ChampSim/bin/champsim.oracle_l2_replayer
```

## Validated replay

Run 619 first:

```bash
cd ~/cache
TRACES="619.lbm_s-4268B" \
MAX_JOBS=1 WARMUP=25000000 SIM=25000000 \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh
```

A valid log must contain all of these:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] loaded ... dense ROI L2 LOAD signatures ...
[list_replayer] emitted ... candidates over ... ROI L2 LOAD accesses (...; 0 signature mismatches; ...; reference enabled)
```

The runner rejects both the original format bug and an early stream shift.

## Validated parallel replay

```bash
cd ~/cache
MAX_JOBS=3 WARMUP=25000000 SIM=25000000 \
nohup bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh \
  > formal_NN_training/results/oracle_replacer_replay/driver.log 2>&1 &
```

## Outputs

```text
formal_NN_training/results/oracle_replacer_replay/replay_inputs/
  *.l2roi.idx_addr.csv       # sparse replay list
  *.l2roi.reference.csv      # dense PC/line validation stream
formal_NN_training/results/oracle_replacer_replay/logs/
formal_NN_training/results/oracle_replacer_replay/driver.log
```

The original `prefetch_list_*_fair_dedup_*.csv` files remain useful diagnostic
exports. They are not directly replayable input.
