# Four frozen students: 602 live study

- **Read the measured native-versus-NN comparison first:** [four-route reader results](../../../formal_study/results/602_width_templates/README.md) · [RESULTS_INDEX](../../../RESULTS_INDEX.md). The latest [H16/template supplement](width_templates/README.md) is separate from this H8 archive.
- **This protocol is complete:** `four602_frozen_live_20261017_v2`; 64 new rows =8 validation + 56 full tests, seed 7, 602 only, two ROIs, MAC0.
- **Conclusion within that scope:** frozen students help versus NoPF, but do not consistently beat native teachers; validation-frozen templates reproduce much of the gain. Learned-payload necessity, generalization and finite-resource benefit are not established.
- **Evidence:** [H8 archive](results/formal_20261017/README.md), [findings](results/formal_20261017/FINDINGS.md), [per-run data](results/formal_20261017/per_run.csv), [paired interventions](results/formal_20261017/paired_comparisons.csv), [64-row audit](results/formal_20261017/audit.json), [callback certificate](results/formal_20261017/callback_verification.json).
- **Contracts:** [config](config.json), [method limits](report/METHOD_LIMITATIONS.md), [measurement scope](MEASUREMENT_SCOPE.md), [cost definitions](report/COST.md), [archive reproduction](report/ARCHIVING.md).

## Models and execution

| Student | Retained input/state/decoder | Weight training |
| --- | --- | --- |
| Stride | Balanced seed 7; PC/address, per-PC h/c, original 64-entry dynamic engine and decoder/output limits | 0–1M, 12 epochs |
| [Streamer](../602_lstm_streamer_live_inference/README.md) | Address-only v 7 LSTM64/H8; Poisson count, mixture/GRU, fixed L2; global h/c | 0–20M, 8 epochs |
| [AMPM](../602_lstm_ampm_live_inference/README.md) | Address-only v 7 LSTM64/H8; Poisson count, mixture/GRU, fixed L2; global h/c | 0–20M, 8 epochs |
| [SPP](../602_lstm_spp_live_inference/README.md) | Empirical-prior hurdle LSTM65/H8; demand/fill, mixture/GRU/learned fill; global h/c | 0–20M, 8 epochs |

- Teacher actions supervise training only; no teacher/private table/future label is a neural runtime input. Weights stay frozen; h/c advances online.
- Global models execute complete synchronous forwards. No imposed FIFO, state eviction, output queue or decoder 32 truncation is added; SPP fills also run complete state updates. Stride retains its own per-PC limits and explicit current-line filter.
- MAC0 removes modeled inference service latency. Operation counts and analytical MAC4 work are not finite-engine IPC, ports, energy or PPA. These routes have different training budgets and state organizations; they are not a matched learning-algorithm ranking.

## Coordinates and controls

| Split | Cache/state warmup | Measured trace records | Scope |
| --- | --- | --- | --- |
| Validation | 15–20M | 20–25M | Existing guard, outside weight-training coordinates |
| ROI0 | 20–25M | 25–50M | Previously inspected; replication |
| ROI1 | 70–75M | 75–100M | Future interval |

- All use `602.gcc_s-734B`, the same machine/cache organization and LOAD eligibility; test ROI length is 25M. Retain actual retired counts, trace origin and overshoot.
- Each run starts h/c at zero at warmup start. Warmup blocks output before cache/PQ mutation while policy/model state advances; ROI retains cache/h/c and resets counters. MAC0 leaves no pending neural work.
- External page policy is `cross_page`; native internal page rules remain. Native SPP keeps same-page behavior, request threshold 40 and L2-fill threshold 90. Old first-formal threshold 25 SPP is a different protocol.
- **SPP inputs differ:** student sees all raw L2-fill evicted addresses; native FILTER receives prefetch fills only. Shared callback opportunities do not imply identical effective state-update inputs.
- Validation NoPF selects a common ceiling of 16,144 requests/M: one credit per eligible LOAD. It is a cumulative upper limit, not equal realized requests, acceptance or DRAM. Test outcomes select neither cap nor template.
- Per ROI: NoPF1 + native/student 4×2 budgets=16 + three global mechanism controls×3=9 + forward 1/2 Stride 2 =28 tests.

## Mechanism and metric boundaries

- `fixed_payload`: full forward/gate/K/h/c for the current input, but ordered delta/fill is replaced by validation's joint per-slot modes. Overflow slots repeat the last observed slot and are counted.
- `template`: no NN; unconditional validation-modal K plus the same slots. Ties use ascending numeric order; modal K may be zero.
- `reset_state`: reset before every model input, including SPP fills; still run complete forwards. Stored mechanism controls are uncapped.
- Changed prefetches can change future callbacks and h/c: a whole-run fixed-payload trajectory is not guaranteed identical to the original.
- SPP teacher labels canonicalize repeated targets with L2-priority minimum fill; native generated actions count raw source calls. K scopes differ.
- L2/LLC cohorts are separate **first source-enqueue intent** cohorts, not final fill or all-level usefulness. Incomplete tracking means NA; never pool their denominators. Pending/resident-unused is censored; evicted-unused is not pollution.
- Four 602 `teacher_category=3` is unclassified/no global shadow teacher; Stride's older shadow labels have a separate meaning.
- Use actual DRAM enqueue/merge/bus-start/completion bytes, not requests×64. ROI callback/update axes exclude warmup counts even though warm h/c is retained; they are not minimum-history or per-PC-history claims.

## Restore and inspect on Sacramento

- Canonical checkout: `/home/qianruw/cache`; local root: `/Users/angelawoo/Documents/CMU/ece/cache`.
- **Current H8 raw roots:** local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2`; Sacramento `/home/qianruw/cache/formal_study/archive/20261017/raw_v2`. Source/runtime remain under `formal_study/archive/20261017/{source_v2,runtime_v2}`.
- **Historically recorded roots (superseded):** local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2`; Sacramento `/home/qianruw/cache/formal_study_20261017/raw_v2`. Frozen JSON/source bindings retain their original strings. Use the [canonical-path launcher](../../../docs/602_width_templates/canonical_paths.py) to resolve current archives; do not recreate removed date-root aliases.
- Restore the trace, consistent ChampSim base `838be17e` plus matching libbf, archived actual build/sources and exact model bytes. See [layout/recovery](../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md) and [forward/dependency recovery](../602_lstm_stride_dynamic_inference/runtime/FORWARD_REPRODUCTION.md).
- `build` does not export/stage models. This restoration block copies only missing retained global files and rejects conflicting bytes; the original Stride model must also be restored.

```sh
cd /home/qianruw/cache
study_python=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
export TMPDIR=$PWD/formal_study/archive/20261017/runtime_v2/tmp
export PYTHONPYCACHEPREFIX=$PWD/formal_study/archive/20261017/runtime_v2/pycache
export MPLCONFIGDIR=$PWD/formal_study/archive/20261017/runtime_v2/mpl
"$study_python" - <<'MODEL_SETUP'
import json, shutil
from pathlib import Path
experiment=Path('formal_NN_training/experiments/602_four_prefetcher_formal')
text=(experiment/'config.json').read_text()
for day in ('20261016','20261017'):
    text=text.replace('/cache/formal_study_'+day,'/cache/formal_study/archive/'+day)
c=json.loads(text)
def restore(source,target):
    if not source.is_file(): raise RuntimeError('Missing retained source: '+str(source))
    if target.exists():
        if target.read_bytes()!=source.read_bytes(): raise RuntimeError('Preserving conflict: '+str(target))
    else:
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    if target.read_bytes()!=source.read_bytes(): raise RuntimeError('Copy differs: '+str(target))
for policy in ('streamer','ampm','spp'):
    retained=experiment.parent/('602_lstm_'+policy+'_live_inference')/'models'
    for name in ('model.bin','immutable_model.bin'):
        restore(retained/'model.bin',Path(c['raw'])/'models'/policy/name)
    checkpoint=Path(c['model_root'])/('602_offline_lstm_'+policy)/'runs'/c['checkpoint_runs'][policy]
    restore(retained/'original_checkpoint.pt',checkpoint)
    restore(retained/'source_training_metadata.json',checkpoint.parent/'run_metadata.json')
stride=Path(c['stride_model'])
if not stride.is_file() or stride.read_bytes()!=stride.with_name('immutable_model.bin').read_bytes():
    raise RuntimeError('Restore the original Stride model and identical immutable copy')
MODEL_SETUP
study_driver=formal_NN_training/experiments/602_four_prefetcher_formal/formal.py
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" status
```

- Retained stage sequence: `build → certify → validation → select → test`; invoke each stage through the same launcher, replacing `status`. Reuse compatible completed artifacts; do not rebuild/overwrite a completed protocol as a new experiment.
- Fresh `certify` requires resolving the standalone [Streamer legacy path restriction](../602_lstm_streamer_live_inference/README.md#export-and-correctness-recovery). JSON translation alone does not fix that guard. Existing accepted results/certificates remain preserved.
- `certify` binds real-checkpoint parity, all actions/h/c, actual probe/model/C++ bytes, measurement and final-log-flush tests. Simulator/probe flags are `-O3 -ffp-contract=off`; correctness fixtures are not simulator pilots.
- `validation` measures all 8 full 5M rows; `select` freezes cap/templates; `test` resumes all 56 full 25M rows under the configured eight-job resource limit. Never launch a second active controller.
- Changed source/model/selection/protocol must not reuse output by path/run ID alone. Preserve failed attempts; distinguish complete/running/failed/validated in `run.json`, final snapshots and controller logs.

## Raw evidence

- `raw_v2/builds`: actual compiled `.cc`/`.h`, Makefile/scripts, executable snapshots and compiler logs. Reuse requires byte checks of source set, binary and model bindings.
- `raw_v2/models`, `certifications/<id>` and `correctness_gate.json`: frozen models, immutable copies, fresh fixtures, actual probe and accepted build-bound evidence.
- `raw_v2/runs`: commands/environments, callbacks, snapshots, final summaries, timing and model copies. `selection.json` and `templates/` preserve validation decisions.
- V1 callback logs had an unflushed buffered tail; v 2 adds measurement flushing and reruns all 64 rows. V1 remains historical and is excluded from v 2 tables.
- Lightweight delivery: [archive index](results/formal_20261017/README.md), [metric definitions](results/formal_20261017/metric_definitions.json), [method diagram](results/formal_20261017/method.svg). Full raw evidence stays inside cache.
