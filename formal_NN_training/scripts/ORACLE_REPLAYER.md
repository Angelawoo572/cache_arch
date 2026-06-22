# Standalone oracle-LSTM replay workflow

This is the only replay path for the base-independent LSTM notebook. It is a **simulator experiment**, not an offline metric: IPC, useful/useless prefetches, queue drops, and timeliness come only from the Pythia run.

## Replay contract

The notebook export is a rich diagnostic CSV:

```text
order,pc,line,issue_prob,addr_conf,...,prefetch_addr
```

`order` is a cycle, not a dynamic L2 callback index. Pythia must not consume this file directly.

`10_prepare_oracle_replacer_replay_input.py` maps every rich trigger back to the matching no-prefetch oracle row and writes:

```text
idx,prefetch_addr
```

where `idx` is the post-warmup ROI **L2 LOAD** ordinal. It also writes a dense signature stream:

```text
idx,pc,line
```

The Pythia `list_replayer` checks every runtime callback against that signature before it emits a candidate. Script 09 rejects a run if the stream shifts, a strict-list entry is emitted at the wrong callback, or the terminal drift is larger than the safety bound.

## 0. Freeze one notebook export directory

Never overwrite an artifact directory that has already been replayed. For the results you just produced, use `thr010` because the notebook used `EXPORT_THRESHOLD=0.10`.

On the cluster:

```bash
cd ~/cache
mkdir -p formal_NN_training/artifacts/oracle_replacer/thr010
```

From your Mac, after downloading/copying the Colab `oracle_replacer` artifact directory into your local repo:

```bash
scp -r \
  /Users/angelawoo/Documents/CMU/ece/cache/formal_NN_training/artifacts/oracle_replacer/. \
  qianruw@sacramento.ece.local.cmu.edu:~/cache/formal_NN_training/artifacts/oracle_replacer/thr010/
```

Verify all five lists and the offline table arrived:

```bash
cd ~/cache
ls -lh formal_NN_training/artifacts/oracle_replacer/thr010/

for trace in 602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B; do
  test -s "formal_NN_training/artifacts/oracle_replacer/thr010/prefetch_list_${trace}_cl128_fair_dedup_lru2048.csv" \
    && echo "[ok] $trace" \
    || echo "[missing] $trace"
done

test -s formal_NN_training/artifacts/oracle_replacer/thr010/oracle_replacer_sweep.csv \
  && echo "[ok] offline sweep summary" \
  || echo "[warn] offline sweep summary is absent; replay will still work but summary.csv cannot join offline metrics"
```

## 1. Update `cache_arch` and Pythia

```bash
cd ~/cache
git pull --ff-only

git -C external/ChampSim status --short
git -C external/ChampSim pull --ff-only
```

If the Pythia status command prints local work you need to preserve, do this instead of the `pull` line above:

```bash
cd ~/cache
git -C external/ChampSim stash push -u -m "local work before oracle-LSTM replay update"
git -C external/ChampSim pull --ff-only
git -C external/ChampSim stash pop
```

## 2. One-time ListReplayer build

```bash
cd ~/cache
bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh
```

This builds:

```text
external/ChampSim/bin/champsim.oracle_l2_replayer
```

The build script compiles `no / multi / no / 1 core`, then enables `list_replayer` through Pythia's existing multi-L2 registry. No manual Pythia source edit is needed after this script succeeds.

Preflight:

```bash
cd ~/cache
ls -lh external/ChampSim/bin/champsim.oracle_l2_replayer
strings external/ChampSim/bin/champsim.oracle_l2_replayer | grep -F "adding L2C_PREFETCHER: list_replayer"
```

## 3. First replay: 619 only

Start with one trace and one job. This validates the complete training-export -> strict-input -> Pythia path before spending more cluster time.

```bash
cd ~/cache

RUN_TAG=base_lstm_cl128_thr010_619 \
TRACES="619.lbm_s-4268B" \
MAX_JOBS=1 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR=formal_NN_training/artifacts/oracle_replacer/thr010 \
OFFLINE_SUMMARY=formal_NN_training/artifacts/oracle_replacer/thr010/oracle_replacer_sweep.csv \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh
```

A correct run ends with all of these facts:

```text
adding L2C_PREFETCHER: list_replayer
[list_replayer] loaded ... dense ROI L2 LOAD signatures
[list_replayer] emitted ... candidates ... 0 signature mismatches ... reference enabled
[oracle-replay-validation] status=pass
[replay] 619... valid=1 IPC=...
```

Inspect the first result:

```bash
cd ~/cache
column -s, -t \
  formal_NN_training/results/oracle_replacer_replay/base_lstm_cl128_thr010_619/summary.csv \
  | less -S

tail -n 35 \
  formal_NN_training/results/oracle_replacer_replay/base_lstm_cl128_thr010_619/logs/619.lbm_s-4268B.oracle_replacer.log
```

## 4. First useful matrix: 602, 619, 605, 620

Do not include 623 in this first batch: the current export deduplicates 97.9% of its candidates, so inspect it separately rather than mixing its diagnosis with the primary replay table.

```bash
cd ~/cache

RUN_TAG=base_lstm_cl128_thr010_primary4 \
TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B" \
MAX_JOBS=2 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR=formal_NN_training/artifacts/oracle_replacer/thr010 \
OFFLINE_SUMMARY=formal_NN_training/artifacts/oracle_replacer/thr010/oracle_replacer_sweep.csv \
nohup bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh \
  > formal_NN_training/results/oracle_replacer_replay/base_lstm_cl128_thr010_primary4.driver.log 2>&1 &
```

Track it:

```bash
cd ~/cache
RUN_TAG=base_lstm_cl128_thr010_primary4

tail -f "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}.driver.log"

pgrep -af "09_run_oracle_replacer_replay_parallel|champsim.oracle_l2_replayer"
```

After it finishes:

```bash
cd ~/cache
RUN_TAG=base_lstm_cl128_thr010_primary4

column -s, -t \
  "formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}/summary.csv" \
  | less -S

python3 - <<'PY'
import csv
p = 'formal_NN_training/results/oracle_replacer_replay/base_lstm_cl128_thr010_primary4/summary.csv'
for r in csv.DictReader(open(p)):
    print(
        '{trace:20s} valid={valid} IPC={ipc:.6f}  vs_no_pref={s0:.4f}  '
        'vs_best({best})={sb:.4f}  issued={issued} useful={useful}  '
        'acc={acc:.4f} time={time:.4f} drops={drops}'.format(
            trace=r['trace'], valid=r['replay_validated'],
            ipc=float(r['ipc']), s0=float(r['speedup_vs_no_pref']),
            best=r['best_normal'], sb=float(r['speedup_vs_best_normal']),
            issued=int(float(r['pf_issued'])), useful=int(float(r['pf_useful'])),
            acc=float(r['selected_accuracy']), time=float(r['timeliness']),
            drops=int(float(r['pf_dropped'])),
        )
    )
PY
```

## 5. 623 diagnostic replay

Run only after the primary table is complete.

```bash
cd ~/cache

RUN_TAG=base_lstm_cl128_thr010_623 \
TRACES="623.xalancbmk_s-700B" \
MAX_JOBS=1 \
WARMUP=25000000 \
SIM=25000000 \
CHUNK_LEN=128 \
DEDUP_CAPACITY=2048 \
ART_DIR=formal_NN_training/artifacts/oracle_replacer/thr010 \
OFFLINE_SUMMARY=formal_NN_training/artifacts/oracle_replacer/thr010/oracle_replacer_sweep.csv \
bash formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh
```

## Result interpretation

Read these columns in this order:

1. `replay_validated=1` and `signature_mismatches=0`: the dynamic replay alignment is valid.
2. `speedup_vs_no_pref`: does the standalone LSTM beat no prefetching?
3. `speedup_vs_best_normal`: does it beat the strongest normal baseline for that trace?
4. `pf_issued`, `pf_dropped`, `selected_accuracy`, `timeliness`, `pf_useless`: why did IPC move?
5. `offline_policy_precision` and `offline_recall_*`: diagnostics only. They are not performance results.

The expected output layout for a tag is:

```text
formal_NN_training/results/oracle_replacer_replay/<RUN_TAG>/
  logs/<trace>.oracle_replacer.log
  logs/<trace>.prepare.log
  replay_inputs/<trace>.l2roi.idx_addr.csv
  replay_inputs/<trace>.l2roi.idx_addr.csv.meta.json
  replay_inputs/<trace>.l2roi.reference.csv
  summary.csv
```

## Repeat policy sweeps without overwriting results

For a new notebook threshold or lead-window setting, copy its exports into a new immutable artifact directory such as `thr030` or `lead4_thr030`. Then change only `RUN_TAG` and `ART_DIR` in the command. Script 09 derives the exact rich filename from `CHUNK_LEN`, `DEDUP_CAPACITY`, and `RICH_SUFFIX`; current notebook defaults are `128`, `2048`, and `fair_dedup_lru2048`.
