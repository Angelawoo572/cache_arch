# Standalone base-independent LSTM replay workflow

This is the only replay path for `LSTM_base_independent_oracle_prefetcher.ipynb`.

The notebook exports a rich CSV:

```text
order,pc,line,issue_prob,addr_conf,...,prefetch_addr
```

`order` is a simulator cycle, not a dynamic L2 callback index. Never give this rich file directly to Pythia.

Script `10_prepare_oracle_replacer_replay_input.py` maps each rich trigger to the matching no-prefetch post-warmup ROI L2-load ordinal and creates:

```text
idx,prefetch_addr
```

It also writes a dense callback reference:

```text
idx,pc,line
```

At runtime, `list_replayer` verifies the PC/line signature before emitting a list entry. Script 09 rejects a replay if signatures diverge, strict-list replay differs from the observed prefix, or tail drift is above the safety bound.

## Current lead-1 artifact snapshot

The current exported lists use this naming:

```text
prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv
oracle_replacer_sweep_lead1_addrconf_lru2048.csv
```

The replay input is the `fair_dedup_lru2048` CSV only. The undeduplicated CSV is diagnostic-only.

Freeze a copied notebook export under a distinct directory, for example:

```text
formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048/
```

Never overwrite a directory after replaying it.

## Update repositories

```bash
cd ~/cache
git pull --ff-only

git -C external/ChampSim status --short
git -C external/ChampSim pull --ff-only
```

A dirty `prefetcher/multi.l2c_pref` from an older local ListReplayer patch does not block the current build. Script 11 now generates its temporary replay frontend from `HEAD:prefetcher/multi.l2c_pref`, not from that local file. Do not run `git restore` on a dirty Pythia file unless you have inspected its diff and know it is disposable.

## Build ListReplayer

```bash
cd ~/cache
bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh

ls -lh external/ChampSim/bin/champsim.oracle_l2_replayer
cat external/ChampSim/bin/champsim.oracle_l2_replayer.build_info.txt
```

Script 11 verifies the generated `oracle_replayer.l2c_pref` source **before** compilation. It intentionally does not use a `strings` test as a build gate; optimized binaries do not reliably preserve the exact diagnostic literal. Script 09 performs the real runtime check when it requires:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] loaded ... dense ROI L2 LOAD signatures
[list_replayer] emitted ... 0 signature mismatches ... reference enabled
[oracle-replay-validation] status=pass
```

## First replay: 619 only

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048
RUN_TAG=base_lstm_lead1_thr010_619

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

Script 09 automatically runs Script 10 to create strict replay inputs, runs Pythia, validates the full callback stream, and calls Script 12 to write `summary.csv`. Do not run a second legacy replay summary script.

Inspect:

```bash
cd ~/cache
RUN_TAG=base_lstm_lead1_thr010_619

column -s, -t \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/summary.csv" \
  | less -S

tail -n 45 \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/logs/619.lbm_s-4268B.oracle_replacer.log"
```

## Primary batch after 619 validates

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048
RUN_TAG=base_lstm_lead1_thr010_primary3
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

Monitor:

```bash
cd ~/cache
RUN_TAG=base_lstm_lead1_thr010_primary3

tail -f "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}.driver.log"
pgrep -af "09_run_oracle_replacer_replay_parallel|champsim.oracle_l2_replayer"
```

## 623 separately

The lead-1 623 export had a 97.9% offline LRU dedup drop rate. Run it separately, after the primary table, as a repeated-address / policy-collapse diagnostic:

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048
RUN_TAG=base_lstm_lead1_thr010_623

RUN_TAG="$RUN_TAG" \
TRACES="623.xalancbmk_s-700B" \
MAX_JOBS=1 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR="$ART_DIR" \
OFFLINE_SUMMARY="$ART_DIR/oracle_replacer_sweep_lead1_addrconf_lru2048.csv" \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh
```

## Interpret each row in order

1. `replay_validated=1` and `signature_mismatches=0`.
2. `speedup_vs_no_pref`.
3. `speedup_vs_best_normal`.
4. `pf_issued`, `pf_dropped`, `selected_accuracy`, `timeliness`, `pf_late`, and `pf_useless`.
5. `offline_*` fields only as training/export diagnostics.

For this lead-1 export, a high offline address score can still be late in the real cache hierarchy. Replay is the required timeliness test.