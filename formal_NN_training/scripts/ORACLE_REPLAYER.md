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

There was a second domain mismatch: the old binary's counter observed a
non-oracle stream (L1D and/or warmup/RFO accesses), while the neural oracle's
`demand_idx` refers to **post-warmup L2 LOAD** demand accesses.

## Fixed contract

1. `10_prepare_oracle_replacer_replay_input.py` converts an existing rich
   notebook CSV into a strict two-column input. It maps `(cycle, pc, line)` to
   the matching oracle `demand_idx` and emits `idx,0xprefetch_addr`.
2. `11_install_oracle_l2_replayer.sh` installs a Pythia-native `list_replayer`
   at L2. Its counter starts after warmup and advances only for L2 LOADs.
3. `09_run_oracle_replacer_replay_parallel.sh` invokes that L2 prefetcher,
   checks its final counter against the oracle row count, and fails on any
   access-domain mismatch or zero matched trigger.

## One-time install/build

```bash
cd ~/cache

git -C external/ChampSim pull
# cache_arch must contain scripts 09--11 from the same revision.
git pull

bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh
```

This produces:

```text
external/ChampSim/bin/champsim.oracle_l2_replayer
```

## Validated parallel replay

Start with two traces:

```bash
cd ~/cache
TRACES="619.lbm_s-4268B 602.gcc_s-734B" \
MAX_JOBS=2 WARMUP=25000000 SIM=25000000 \
nohup bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh \
  > formal_NN_training/results/oracle_replacer_replay/driver.log 2>&1 &
```

Then run all traces only after the two-trace logs end with both:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] emitted ... over <oracle-row-count> ROI L2 LOAD accesses (... matched access indices)
```

The runner's validation rejects the old incorrect case:

```text
issued 0 prefetches ... (0 attempted, 0 matched access indices)
```

## Outputs

```text
formal_NN_training/results/oracle_replacer_replay/replay_inputs/
formal_NN_training/results/oracle_replacer_replay/logs/
formal_NN_training/results/oracle_replacer_replay/driver.log
```

The original `prefetch_list_*_fair_dedup_*.csv` files remain useful diagnostic
exports. They are not directly replayable input.
