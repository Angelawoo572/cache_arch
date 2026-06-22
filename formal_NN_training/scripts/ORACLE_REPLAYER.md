# Base-independent LSTM keyed replay workflow

This is the one replay path for `LSTM_base_independent_oracle_prefetcher.ipynb`.

The notebook exports a rich CSV:

```text
order,pc,line,issue_prob,addr_conf,...,prefetch_addr
```

`order` is a simulator cycle. It must not be treated as a runtime L2 callback index.

## Why the old global-index replay was invalid

A no-prefetch global L2-load ordinal does **not** stay fixed after a useful prefetch: changing memory latency can reorder independent out-of-order L2 callbacks. The previous `idx,prefetch_addr` plus dense global PC/line signature design therefore correctly detected divergence, but it could not be the final replay mechanism: the intervention itself caused the mismatch.

The active workflow instead maps each rich event to:

```text
pc,line,occ,prefetch_addr
```

`occ` is the zero-based occurrence number of that `(pc,line)` pair in the no-prefetch oracle. At runtime ListReplayer maintains the same local per-`(pc,line)` occurrence counter after warmup and triggers the corresponding candidate. This survives reordering between unrelated PC/line pairs.

This is an **offline-policy keyed replay**, not embedded PyTorch inference. Report it as such. It is appropriate for the first question, “what happens when this frozen LSTM policy is applied to the corresponding dynamic demand events?”, but it is not the final hardware implementation claim.

## Current lead-1 artifact snapshot

```text
formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048/
  prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv
  oracle_replacer_sweep_lead1_addrconf_lru2048.csv
```

Replay only the `fair_dedup_lru2048` CSV. The undeduplicated files are diagnostics.

## Update and rebuild

```bash
cd ~/cache

git pull --ff-only
git -C external/ChampSim pull --ff-only

bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh
```

The build script creates a temporary L2 frontend from Pythia's tracked `multi.l2c_pref`, adds ListReplayer only to that temporary file, builds it, and cleans up the generated frontend. It should not permanently modify `prefetcher/multi.l2c_pref`.

Check provenance:

```bash
cd ~/cache
cat external/ChampSim/bin/champsim.oracle_l2_replayer.build_info.txt
```

The file must contain:

```text
replay_key=pc_line_occ
```

## First keyed replay: 619

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048
RUN_TAG=base_lstm_lead1_thr010_619_keyed

rm -rf "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}"

RUN_TAG="$RUN_TAG" \
TRACES="619.lbm_s-4268B" \
MAX_JOBS=1 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR="$ART_DIR" \
OFFLINE_SUMMARY="$ART_DIR/oracle_replacer_sweep_lead1_addrconf_lru2048.csv" \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh
```

A successful run contains:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] loaded ... PC-line-occ triggers
[list_replayer] emitted ... runtime ROI L2 LOAD accesses (... matched PC-line-occ triggers; ... key=pc_line_occ)
[oracle-replay-validation] status=keyed_transport_pass
```

Inspect its result:

```bash
cd ~/cache
RUN_TAG=base_lstm_lead1_thr010_619_keyed

column -s, -t \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/summary.csv" \
  | less -S
```

The essential fields are:

1. `replay_transport_ok=1`: conversion, build, table loading, and keyed runtime counters are consistent.
2. `keyed_trigger_coverage`: matched trigger keys / converted trigger keys. Report it; do not silently assume all decisions fired.
3. `speedup_vs_no_pref`, then `speedup_vs_best_normal`.
4. `pf_issued`, `pf_dropped`, `selected_accuracy`, `timeliness`, `pf_late`, and `pf_useless`.

`offline_*` columns remain model/export diagnostics only.

## Run the primary three-trace batch after 619

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048
RUN_TAG=base_lstm_lead1_thr010_primary3_keyed
OUT_ROOT="formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}"

RUN_TAG="$RUN_TAG" \
TRACES="602.gcc_s-734B 605.mcf_s-994B 620.omnetpp_s-874B" \
MAX_JOBS=2 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR="$ART_DIR" \
OFFLINE_SUMMARY="$ART_DIR/oracle_replacer_sweep_lead1_addrconf_lru2048.csv" \
nohup bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh \
  > "${OUT_ROOT}.driver.log" 2>&1 &

echo $! > "${OUT_ROOT}.driver.pid"
```

Run `623` separately because its current offline LRU dedup rate is 97.9%, which is a model/policy collapse diagnostic rather than a clean main-table point.
