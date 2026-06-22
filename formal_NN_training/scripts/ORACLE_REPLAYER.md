# Standalone base-independent LSTM replay workflow

This is the one replay path for `LSTM_base_independent_oracle_prefetcher.ipynb`.

The notebook produces a **rich** CSV such as:

```text
order,pc,line,issue_prob,addr_conf,...,prefetch_addr
```

`order` is a simulator cycle, not a dynamic callback index. Never feed this CSV directly to Pythia.

Script `10_prepare_oracle_replacer_replay_input.py` maps each rich trigger to the matching no-prefetch post-warmup ROI L2-load ordinal and produces:

```text
idx,prefetch_addr
```

It also creates a dense reference stream:

```text
idx,pc,line
```

The Pythia `list_replayer` checks every runtime L2-load callback against that reference before it emits an entry. Script 09 rejects a run if signatures diverge, the strict-list prefix does not replay exactly, or final callback drift exceeds the safety bound.

## Current Colab export

The lead-1 address-confidence export currently copied to Sacramento is:

```text
formal_NN_training/artifacts/oracle_replacer/
  oracle_replacer_sweep_lead1_addrconf_lru2048.csv
  prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv
  prefetch_list_<trace>_cl128_fair_undedup.csv
```

The replay uses only the `fair_dedup_lru2048` files. The undeduplicated files are diagnostics, not replay inputs.

## 0. Update both repositories

```bash
cd ~/cache

git pull --ff-only

git -C external/ChampSim status --short
git -C external/ChampSim pull --ff-only
```

`cache_arch` contains the replay driver, converter, and the only supported summary parser:

```text
09_run_oracle_replacer_replay_parallel.sh
10_prepare_oracle_replacer_replay_input.py
11_install_oracle_l2_replayer.sh
12_parse_oracle_replacer_replay.py
```

There is no separate replay-summary command after Script 09: Script 09 invokes Script 12 automatically.

If `git -C external/ChampSim status --short` shows only old generated build files or an old temporary ListReplayer registry patch, reset those generated files before pulling:

```bash
cd ~/cache

git -C external/ChampSim restore \
  prefetcher/multi.l2c_pref \
  prefetcher/l2c_prefetcher.cc

git -C external/ChampSim pull --ff-only
```

Do not run that restore command when those files contain deliberate work you still need.

## 1. Freeze the exact Colab export

Keep the copied root artifacts untouched. Make one immutable directory for this lead-1, confidence-0.10 export:

```bash
cd ~/cache

SRC=formal_NN_training/artifacts/oracle_replacer
ART_TAG=lead1_thr010_addrconf_lru2048
ART_DIR="$SRC/$ART_TAG"

mkdir -p "$ART_DIR"
cp -n "$SRC"/prefetch_list_*_cl128_fair_dedup_lru2048.csv "$ART_DIR"/
cp -n "$SRC"/prefetch_list_*_cl128_fair_undedup.csv "$ART_DIR"/
cp -n "$SRC"/oracle_replacer_sweep_lead1_addrconf_lru2048.csv "$ART_DIR"/

sha256sum "$ART_DIR"/*.csv > "$ART_DIR/SHA256SUMS"
ls -lh "$ART_DIR"
```

Preflight every rich export before simulation:

```bash
cd ~/cache

ART_DIR=formal_NN_training/artifacts/oracle_replacer/lead1_thr010_addrconf_lru2048

python3 - "$ART_DIR" <<'PY'
import csv
import sys
from pathlib import Path

art = Path(sys.argv[1])
traces = [
    "602.gcc_s-734B", "619.lbm_s-4268B", "605.mcf_s-994B",
    "620.omnetpp_s-874B", "623.xalancbmk_s-700B",
]
required = {"order", "pc", "line", "prefetch_addr"}
for trace in traces:
    p = art / f"prefetch_list_{trace}_cl128_fair_dedup_lru2048.csv"
    if not p.is_file() or p.stat().st_size == 0:
        raise SystemExit(f"missing or empty: {p}")
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"{p}: missing columns {sorted(missing)}")
        n = sum(1 for _ in r)
    print(f"[ok] {trace}: {n} deduplicated rich rows")

summary = art / "oracle_replacer_sweep_lead1_addrconf_lru2048.csv"
if not summary.is_file():
    raise SystemExit(f"missing offline summary: {summary}")
print(f"[ok] offline summary: {summary}")
PY
```

## 2. Build the signature-validating ListReplayer once

```bash
cd ~/cache

bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh

ls -lh external/ChampSim/bin/champsim.oracle_l2_replayer
strings external/ChampSim/bin/champsim.oracle_l2_replayer \
  | grep -F "adding L2C_PREFETCHER: list_replayer"
```

The build is exactly `no / multi / no / 1 core`. Do not use a plain `make`, because that may compile the inactive `no.l2c_pref` frontend instead of Pythia's multi-prefetcher registry.

## 3. First end-to-end replay: 619 only

`619` is intentionally the first transport test because the offline address metric is high. This test validates the data path; it does not prove that the prefetch arrives early enough to improve IPC.

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

A valid log must contain all four facts:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] loaded ... dense ROI L2 LOAD signatures
[list_replayer] emitted ... 0 signature mismatches ... reference enabled
[oracle-replay-validation] status=pass
```

Inspect the simulator result:

```bash
cd ~/cache

RUN_TAG=base_lstm_lead1_thr010_619
column -s, -t \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/summary.csv" \
  | less -S

tail -n 45 \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/logs/619.lbm_s-4268B.oracle_replacer.log"
```

## 4. Primary replay batch: 602, 605, and 620

Run these after 619 validates. Do not rerun 619 in this batch.

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
echo "driver PID: $(cat "${OUT_ROOT}.driver.pid")"
```

Monitor it:

```bash
cd ~/cache

RUN_TAG=base_lstm_lead1_thr010_primary3
tail -f "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}.driver.log"

pgrep -af "09_run_oracle_replacer_replay_parallel|champsim.oracle_l2_replayer"
```

After it finishes:

```bash
cd ~/cache

RUN_TAG=base_lstm_lead1_thr010_primary3
SUMMARY="formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/summary.csv"

column -s, -t "$SUMMARY" | less -S

python3 - "$SUMMARY" <<'PY'
import csv
import sys

for r in csv.DictReader(open(sys.argv[1])):
    print(
        "{trace:20s} valid={valid} IPC={ipc:.6f} "
        "vs_no_pref={s0:.4f} vs_best({best})={sb:.4f} "
        "issued={issued} useful={useful} acc={acc:.4f} "
        "time={time:.4f} dropped={dropped}".format(
            trace=r["trace"], valid=r["replay_validated"],
            ipc=float(r["ipc"]), s0=float(r["speedup_vs_no_pref"]),
            best=r["best_normal"] or "<unknown>",
            sb=float(r["speedup_vs_best_normal"]),
            issued=int(float(r["pf_issued"])),
            useful=int(float(r["pf_useful"])),
            acc=float(r["selected_accuracy"]),
            time=float(r["timeliness"]),
            dropped=int(float(r["pf_dropped"])),
        )
    )
PY
```

## 5. 623 diagnostic replay

`623` has a 97.9% offline LRU-dedup drop rate for this export. Replay it separately. It is a diagnosis of collapsed/repeated address predictions and issue aggressiveness, not part of the first primary result table.

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

1. `replay_validated=1` and `signature_mismatches=0`. Otherwise the result is invalid.
2. `speedup_vs_no_pref`. This is the first performance hurdle.
3. `speedup_vs_best_normal`. This is the replacement-baseline hurdle.
4. `pf_issued`, `pf_dropped`, `selected_accuracy`, `timeliness`, and `pf_useless`. These explain IPC movement.
5. `offline_*` values. These are training/export diagnostics only, not simulator performance.

## Important interpretation of the current lead-1 export

The label horizon has median 1 for 602 and 619 and remains short for the other traces. The very high offline address scores prove that the revised notebook is no longer dead-gated and can learn the immediate next miss/address pattern. They do **not** prove a useful prefetch: a demand one L2 callback ahead can still be late. This replay is therefore the required timeliness test.

The current export also runs model inference across the entire source trace, including its training prefix. Treat it as a first end-to-end feasibility result. A later paper-quality result should hold out a future phase or trace segment for final reported replay.

## Sweep policy after this first replay

Do not overwrite this artifact directory. For each new lead/threshold experiment, make a new directory, for example:

```text
formal_NN_training/artifacts/oracle_replacer/lead4_thr030_addrconf_lru256/
```

Then change only `ART_DIR`, `RUN_TAG`, and the matching `OFFLINE_SUMMARY` filename. Rebuild the binary only when `external/ChampSim` changes.