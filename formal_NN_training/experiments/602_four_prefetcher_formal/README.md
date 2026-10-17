# Four frozen prefetcher students: 602 live formal study

This study tests the retained seed7/h8 Stride, Streamer, AMPM and SPP students
with actual recurrent C++ inference in one fresh ChampSim build. It separates
learned payload, event gating and recurrent history without retraining weights.
It does not assume that a student wins, that a PC-independent template must
fail, or that stateful learning is novel. Every completed negative result stays
in the tables. This round contains **8 validation and 56 full test rows (64
simulator rows total)**; no 605 workload is included.

The source contract is [config.json](config.json). The new protocol and raw
root are `four602_frozen_live_20261017_v2` and
`/home/qianruw/cache/formal_study/archive/20261017/raw_v2`. The healthy local checkout is
inside `cache/formal_study/work/cache_gate_mechanism`; the synchronized
Sacramento source is `/home/qianruw/cache/formal_study/archive/20261017/source_v2`. Existing
first-formal and gate/payload results remain separate protocol archives.

The final canonical output is `raw_v2/analysis_final`. The superseded v1 source,
raw outputs and full ROI attempts are preserved. Its completed-run
summary/IPC/phase/cohort records retain the full measurement, but callback
ofstreams could retain a buffered tail when ChampSim exited without destroying the prefetcher object. Those v1 rows are
excluded from the final v2 main table. V2 explicitly flushes and checks callback
logs during statistics output; prediction, model, selection and full ROI budgets
are unchanged. It reruns the full 64-row matrix under a separate immutable build,
certificates and output root. The launcher obtains raw/runtime/analysis/lock
paths from config and never writes new results into the old v1 root.

## Supplemental H16 and capped-template round

The new active protocol is [width_templates/config.json](width_templates/config.json), with its own validation, raw root, width-specific exports, certificates and result tables. It recovers the original AMPM/SPP H16 seed7 checkpoints and adds capped H8 frozen templates. The H8 protocol and archived v1/v2 metadata remain independent. See [canonical layout and recovery](../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md). Canonical local checkout: `cache/formal_study/work/cache_gate_mechanism`; Sacramento uses the same relative location. Canonical prior raw root: `cache/formal_study/archive/20261017/raw_v2`. Old absolute paths in historical JSON are translated in memory by the canonical-path launcher; removed dated names are not recreated.

## Completed results

The full v2 matrix is complete: 64 full rows passed the analysis and strict
callback checks, with generated figures. The results support useful frozen
students versus NoPF; they do not support a general win over native teachers
or a claim that complex learned payload is necessary. Simple validation-frozen
templates reproduce much of the observed benefit. These conclusions remain
limited to the retained seed7 models, 602, two ROIs and MAC0.

- [Findings and exact numerical comparisons](results/formal_20261017/FINDINGS.md).
- [One-page paper story and supported/unsupported/pending claims](report/PAPER_STORY.md).
- [Complete archive index](results/formal_20261017/README.md),
  [per-run metrics](results/formal_20261017/per_run.csv),
  [paired interventions](results/formal_20261017/paired_comparisons.csv), and
  [metric definitions](results/formal_20261017/metric_definitions.json).
- [Full 64-row audit](results/formal_20261017/audit.json),
  [strict callback certificate](results/formal_20261017/callback_verification.json),
  [actual build/parity metadata](results/formal_20261017/metadata/), and
  [archive copied-byte/subset certificate](results/formal_20261017/archive_verification.json).
- Own-policy plus matched-NoPF subsets:
  [Stride](../602_lstm_stride_dynamic_inference/results/formal_20261017/README.md),
  [Streamer](../602_lstm_streamer_live_inference/results/formal_20261017/README.md),
  [AMPM](../602_lstm_ampm_live_inference/results/formal_20261017/README.md), and
  [SPP](../602_lstm_spp_live_inference/results/formal_20261017/README.md).

| Observed full-ROI evidence | ROI0: 25..50M, previously observed | ROI1: 75..100M, future interval |
| --- | --- | --- |
| IPC and actual DRAM | [Figure](results/formal_20261017/ipc_traffic_roi0.svg) | [Figure](results/formal_20261017/ipc_traffic_roi1.svg) |
| Requests versus IPC | [Uncapped](results/formal_20261017/requests_ipc_roi0_cap0.svg), [cap 16144](results/formal_20261017/requests_ipc_roi0_cap16144.svg) | [Uncapped](results/formal_20261017/requests_ipc_roi1_cap0.svg), [cap 16144](results/formal_20261017/requests_ipc_roi1_cap16144.svg) |
| Whole-ROI K/delta/fill | [Distributions](results/formal_20261017/action_distribution_roi0.svg) | [Distributions](results/formal_20261017/action_distribution_roi1.svg) |
| Instruction/callback/update axes | [Progress](results/formal_20261017/progress_axes_roi0.svg) | [Progress](results/formal_20261017/progress_axes_roi1.svg) |
| Actual phase emit and IPC | [Phase curves](results/formal_20261017/phase_observations_roi0.svg) | [Phase curves](results/formal_20261017/phase_observations_roi1.svg) |
| Payload/template/reset interventions | [Mechanisms](results/formal_20261017/mechanisms/mechanisms_roi0_seed7.svg) | [Mechanisms](results/formal_20261017/mechanisms/mechanisms_roi1_seed7.svg) |

[Cost/throughput figure](results/formal_20261017/cost/cost_effect.svg) and
[its derived table](results/formal_20261017/cost/cost_derived.csv) distinguish
measured MAC0 IPC from analytical work requirements. The
[teacher-supervision/live method diagram (SVG)](results/formal_20261017/method.svg)
and its [rendered PNG](results/formal_20261017/method.png)
show the execution boundary; model, page, fill-input, training-budget and
intent-cohort limits are explicit in [METHOD_LIMITATIONS](report/METHOD_LIMITATIONS.md).
Use [ARCHIVING](report/ARCHIVING.md) to reproduce the fresh lightweight archive.

## Models and interpretation

| Student | Actual retained model/state | Original training coordinates |
| --- | --- | --- |
| Stride | Existing balanced seed7 checkpoint; PC/address features, per-PC h/c, original dynamic engine | 0..1M |
| Streamer | Address-only v7 LSTM64/h8, Poisson count, autoregressive mixture decoder; one global h/c | 0..20M |
| AMPM | Address-only v7 LSTM64/h8, Poisson count, autoregressive mixture decoder; one global h/c | 0..20M |
| SPP | Empirical-prior hurdle LSTM65/h8, demand/fill kind, mixture/GRU/fill decoder; one global h/c | 0..20M |

See the sibling [Streamer](../602_lstm_streamer_live_inference/README.md),
[AMPM](../602_lstm_ampm_live_inference/README.md), and
[SPP](../602_lstm_spp_live_inference/README.md) model contracts. Model paths in
config identify the original checkpoints; exports and certificates reside in
the new raw root. The Stride binary comes from the completed first formal
seed7 model. These are one retained training seed per route, with different
original training budgets; this round is not a matched four-model retraining
comparison or a three-seed robustness claim.

The three global students use complete MAC0 synchronous forwards and preserve
their original state and decoder semantics. They have no imposed input FIFO,
state capacity/eviction, output queue or decoder32 truncation. SPP fill callbacks
also execute their complete state update. Stride retains its existing 64-entry per-PC
state engine and decoder/output limits; its NN current-line filter is explicit
in the recorded environment. MAC0 removes modeled inference service latency;
it does not measure finite hardware cost, ports, energy or PPA. A comparison
must not describe these different state/control organizations as identical.

## Fixed coordinates and controls

| Split | Cache warmup | Measured records | Interpretation |
| --- | --- | --- | --- |
| Validation | 15..20M | 20..25M | Outside original weight-training coordinates; existing guard reused for cap/template selection |
| Test ROI0 | 20..25M | 25..50M | Already observed in the old offline evaluation; replication |
| Test ROI1 | 70..75M | 75..100M | Future ROI relative to the original train/guard/evaluation |

All rows use the 602.gcc_s-734B trace, 5M cache warmup, 25M test ROI, identical
machine/cache organization and L2 LOAD callback eligibility. Cache warmup
issues no prefetches; conventional policies and each student's permitted
state inputs still advance. Actual retired instructions, trace origin and
boundary overshoot are retained. Common external page policy is `cross_page`,
so the shared gate adds no same-page restriction. Native policies retain their
own internal page rules; native SPP remains internally same-page and keeps its
confidence-based L2/LLC fill destinations. Native SPP explicitly uses request
threshold 40 and L2-fill threshold 90, matching the teacher configuration that
produced this SPP student's labels. The old first-formal dependency defaulted
to request threshold 25; its IPC is not a same-protocol comparator for these
new native SPP rows. The student's learned destinations also remain intact.
This is not equality of every policy's effective page rule. SPP candidate
arrays have fail-fast write guards: a guard trip invalidates a row rather than
truncating predictions. Defined uint64 arithmetic replaces the original native
negative-delta left shifts, preserving modulo address arithmetic.

The common cumulative request ceiling is frozen from validation NoPF's actual
eligible L2 LOAD callbacks per million retired instructions: one request credit
per validation callback. It is an external maximum, not a requested degree,
cache acceptance rate or equality of realized traffic. Report actual requests/M,
cache acceptance/merging/rejection and controller-measured DRAM separately.
No test result selects the cap or template. Existing 20..25M guard use and old
25..50M inspection are disclosed rather than described as untouched holdout.

For each ROI the 28 rows are:

- One NoPF; four native policies and four students under uncapped and the
  common ceiling (16 rows).
- For each global student, uncapped learned-gate/K with frozen template
  payload, unconditional modal-count template, and reset-before-each-input
  state ablation (9 rows).
- Existing uncapped forward1/forward2 Stride references (2 rows).

The payload intervention preserves the student's full forward, gate, K and h/c
updates, replacing only the ordered delta/fill payload with validation's joint
per-slot modal template. If a later K exceeds the observed template length,
the final observed slot repeats and extrapolated slots are counted. The
unconditional template uses validation's modal K and the same slots. Ties use
ascending numeric order. This is a bounded frozen address-only template, not a
PC table. Modal K may be zero; preserve that result. The reset ablation resets
before every model input, including SPP fills, while still computing forwards.
These mechanism controls are uncapped in the stored matrix; no unrun capped
mechanism effect is inferred.

## SPP effective inputs and cohort scope

The trained SPP student consumes all raw CACHE L2-fill events, including demand
and writeback fills, with the evicted address and fill-kind feature. Original
native SPP FILTER feedback consumes only prefetch fills. The adapter preserves
both effective input contracts. They share actual cache callbacks but not
identical state-update inputs. Report that limitation in every interpretation
of native-versus-student gains.

The new observer follows source-enqueue requested L2 and requested LLC cohorts
in separate ID domains. These are intent cohorts; they are not a claim about
final physical fill level or all-level usefulness after downstream upgrades.
See [method limitations](report/METHOD_LIMITATIONS.md) for that boundary. Source-L2 redundancy and LLC queue forwarding/merging are explicitly
resolved; write-forwarding is not useful LOAD service or a new DRAM transfer.
Unknown/rejected handoff paths disable complete-cohort accuracy availability.
L2 and LLC useful denominators differ; do not pool them into one asserted L2
accuracy/coverage. Pending/resident-unused remain censored, evicted-unused is
not pollution, and requests issued times 64 is not DRAM traffic. The full
[routing/input-scope note](MEASUREMENT_SCOPE.md) records these definitions.

## Recover and run on Sacramento

Reuse the existing authenticated SSH connection; all paths remain under cache.
From the synchronized source root, select the recorded Torch environment and
keep process temporaries inside the current study:

```sh
cd /home/qianruw/cache/formal_study/archive/20261017/source_v2
export TMPDIR=/home/qianruw/cache/formal_study/archive/20261017/runtime_v2/tmp
export PYTHONPYCACHEPREFIX=/home/qianruw/cache/formal_study/archive/20261017/runtime_v2/pycache
export MPLCONFIGDIR=/home/qianruw/cache/formal_study/archive/20261017/runtime_v2/mpl
study_python=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
study_driver=formal_NN_training/experiments/602_four_prefetcher_formal/formal.py
# Stage exact retained model bytes before build; do not overwrite a model.
"$study_python" - <<'MODEL_SETUP'
import json,shutil
from pathlib import Path
experiment=Path('formal_NN_training/experiments/602_four_prefetcher_formal')
config_text=(experiment/'config.json').read_text()
for day in ('20261016','20261017'):
    config_text=config_text.replace('/cache/formal_study_'+day,'/cache/formal_study/archive/'+day)
config=json.loads(config_text)
raw=Path(config['raw'])
def restore(source,target):
    if not source.is_file():raise RuntimeError('Missing retained source: '+str(source))
    if target.exists():
        if target.read_bytes()!=source.read_bytes():
            raise RuntimeError('Preserving different existing file: '+str(target))
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    if target.read_bytes()!=source.read_bytes():raise RuntimeError('Copy differs: '+str(target))
for policy in ('streamer','ampm','spp'):
    retained=experiment.parent/('602_lstm_'+policy+'_live_inference')/'models'
    for name in ('model.bin','immutable_model.bin'):
        restore(retained/'model.bin',raw/'models'/policy/name)
    # Restore the unchanged original source paths required by config/certify.
    checkpoint=Path(config['model_root'])/('602_offline_lstm_'+policy)/'runs'/config['checkpoint_runs'][policy]
    restore(retained/'original_checkpoint.pt',checkpoint)
    restore(retained/'source_training_metadata.json',checkpoint.parent/'run_metadata.json')
    print('Verified retained model:',policy)
stride=Path(config['stride_model'])
if not stride.is_file():raise RuntimeError('Restore the preserved original Stride raw model: '+str(stride))
if stride.read_bytes()!=stride.with_name('immutable_model.bin').read_bytes():
    raise RuntimeError('Original Stride immutable weights differ')
MODEL_SETUP
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" build
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" certify
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" validation
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" select
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" test
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" status
```

The preparation block can recover an empty `raw_v2/models` from the three
sibling retained models without training or overwriting any different file.
It also restores absent original checkpoint/metadata paths used for parity.
The existing Stride raw model path stays unchanged. `formal.py build` does not
automatically export/stage weights; `certify` needs those files in place. A
standalone exporter may instead reproduce the same portable bytes into a fresh
cache-owned directory, using each sibling's retained checkpoint and metadata;
compare its bytes before use and create a new build/certificate when inputs
change. Never export over an active protocol's model files.

Run `build` only to create a new immutable build; a compatible completed build
can be reused after source/binary/model/certificate checks. `certify` must gate
simulation on real-checkpoint parity for all three global students, generated
source byte equality, and measurement correctness; it is not a simulator pilot.
`validation` completes all 8 full 5M validation measurements. `select` freezes
the common ceiling and per-student templates. `test` runs/resumes the stored 56
full 25M test rows with the configured server resource limit (currently 8 jobs).
A changed source, model, selection or protocol must not reuse an old row solely
because a path or run ID matches. Active jobs and failed attempts remain
recoverable; inspect `run.json`, the final snapshot and completion status.

The clean dependency is the consistent ChampSim base `838be17e` plus its matching
libbf, never a mix of old simulator versions. The builder archives a clean base,
applies the inherited patch once, generates the Stride forward once, and adds
the new four-route installer once. Generated forward files must match the actual
build, and completed snapshots must remain unchanged. Exact source/model bytes
provide identity; no SHA/hash ledger is required. See
[forward/dependency recovery](../602_lstm_stride_dynamic_inference/runtime/FORWARD_REPRODUCTION.md).
That guide also links the separately archived first-formal 101-row tables and
full raw evidence. New IPC rows must use this new build/protocol's controls.

## Output and claims

`raw_v2/builds` contains actual sources, executable snapshots and compiler logs.
Its `compiled_source` copies every actual compiled `.cc`/`.h`, Makefile and
build scripts. The isolated `build_four602.sh` retains the selected modules;
the original script's post-build restoration to NoPF is omitted only in that
new script, and the original build script stays unchanged. Current-build checks
compare the complete actual source set and bytes against this saved copy;
`raw_v2/models` contains real exported models and immutable model copies;
Each `raw_v2/certifications/<id>/` contains fresh real-checkpoint parity fixtures,
reports, a probe compiled from the actual simulator runtime, model/probe/C++
byte snapshots, and the pure measurement test. `raw_v2/correctness_gate.json`
binds the accepted certificate to the formal build. Both actual simulator and
probe use `-O3 -ffp-contract=off` with g++; the old reused PASS reports do not
substitute for this certificate. `raw_v2/runs` contains per-arm commands, environments, immutable model copies, complete
snapshots, callbacks, host timing and final summaries. `raw_v2/selection.json` and
`raw_v2/templates` record validation-only decisions. The local synchronized study
is `cache/formal_study_20261017/`; its raw data and all compiler/runtime artifacts
stay inside cache. Track complete, running, failed and validated states
separately. Source registration or a generated script alone is not a completed
experiment.

Report IPC/cycles, actual request rates, separate L2/LLC lifecycle metrics,
actual DRAM, and concrete storage/cost assumptions. A useful frozen template
weakens a learned-payload explanation; a poor template does not prove a
stateful innovation. A prefix parity fixture says nothing about full-ROI IPC.
The future ROI and actual complete causal interventions decide which proposed
mechanism is supported, unsupported or still unresolved.
