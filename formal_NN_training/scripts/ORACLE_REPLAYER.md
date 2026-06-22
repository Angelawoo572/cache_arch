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
git -C external/ChampSim remote -v
git -C external/ChampSim branch -vv
```

### Required recovery when Script 09 says reference was not loaded

A log that contains only:

```text
[list_replayer] emitted ... matched access indices
```

but lacks `dense ROI L2 LOAD signatures`, `signature mismatches`, and `reference enabled` is the old idx-only ListReplayer. Its results are invalid for the formal replay path.

Preserve the local residual-audit patch, discard only the obsolete local multi-registry patch, and reset the tracked Pythia tree to the current GitHub master:

```bash
cd ~/cache

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="formal_NN_training/backups/pythia_${STAMP}"
mkdir -p "$BACKUP_DIR"

git -C external/ChampSim status --short > "$BACKUP_DIR/status_before.txt"
git -C external/ChampSim diff -- src/cache.cc > "$BACKUP_DIR/src_cache_residual_audit.patch"
git -C external/ChampSim diff -- prefetcher/multi.l2c_pref > "$BACKUP_DIR/multi_registry_obsolete.patch"
git -C external/ChampSim branch "backup/pre_signature_replayer_${STAMP}" HEAD

# Script 11 generates its own temporary list-replayer registry. The old local
# modification to multi.l2c_pref is no longer needed.
git -C external/ChampSim restore --source=HEAD -- prefetcher/multi.l2c_pref

# Preserve the residual audit logger through the source update. vcpkg/ stays
# untouched because this stash intentionally does not use -u.
git -C external/ChampSim stash push -m "preserve residual audit logger ${STAMP}" -- src/cache.cc

git -C external/ChampSim remote set-url origin https://github.com/Angelawoo572/ChampSim.git
git -C external/ChampSim fetch origin master
git -C external/ChampSim switch master
git -C external/ChampSim reset --hard origin/master
git -C external/ChampSim stash pop

grep -nE "PFETCH_REF_PATH|dense ROI L2 LOAD signatures|signature mismatches|reference enabled" \
  external/ChampSim/prefetcher/list_replayer.cc
grep -n "ReferenceSignature" external/ChampSim/inc/list_replayer.h
```

The last two commands must print matching source lines before you build. The backup branch and patches remain available under `formal_NN_training/backups/`.

## Build ListReplayer

```bash
cd ~/cache
git pull --ff-only
bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh

ls -lh external/ChampSim/bin/champsim.oracle_l2_replayer
cat external/ChampSim/bin/champsim.oracle_l2_replayer.build_info.txt
```

Script 11 verifies both the signature-validating ListReplayer source ABI and the generated `oracle_replayer.l2c_pref` frontend before compilation. It intentionally does not use a `strings` test as a build gate; optimized binaries do not reliably preserve an exact diagnostic literal. Script 09 performs the real runtime check when it requires:

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

## 623 diagnostic replay

Run this separately because the current 623 export dropped about 97.9% of undeduplicated candidate rows under LRU dedup:

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

## Read every result in this order

1. `replay_validated=1` and `signature_mismatches=0`.
2. `speedup_vs_no_pref`.
3. `speedup_vs_best_normal`.
4. `pf_issued`, `pf_dropped`, `selected_accuracy`, `timeliness`, `pf_useless`, and `pf_late`.
5. `offline_*` values only as export/training diagnostics, never as simulator results.