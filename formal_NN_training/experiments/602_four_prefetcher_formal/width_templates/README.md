# Supplementary evidence: Frozen H16 and capped-template supplement

- **Read first:** [project overview](../../../../RESULTS_INDEX.md) · [four-route native-versus-own-NN results](../../../../formal_study/results/602_width_templates/README.md), including IPC, accuracy, coverage, timeliness and Pareto comparisons.
- **This page:** Secondary H16/template protocol, model contracts and recovery. Use the reader report above for the main result comparison.

- **Protocol:** `four602_width_templates_20261017_v1`; 24 new rows =2 validation + 22 tests, plus 64 explicitly bound H8 reference rows. No retraining or shared C++ change.
- **Scope:** original seed 7 AMPM/SPP H16, 602, two ROIs, synchronous MAC0. H16 gains do not establish learned-payload necessity or finite-resource benefit; simple templates explain almost all gain. Keep negative/null results.
- **Contracts:** [config](config.json), [driver](driver.py), [old replay alignment](EVIDENCE_ALIGNMENT.md), [method limits](../report/METHOD_LIMITATIONS.md), [canonical layout/recovery](../../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).

## Matrix and selection

| New rows | Full measurements |
| --- | --- |
| Validation 2 | AMPM/SPP H16,20–25M after 5M warmup |
| Test 16 | AMPM/SPP H16 × two ROIs × student/template × uncapped/cap 16144 |
| Test 6 | Streamer/AMPM/SPP H8 templates × two ROIs, cap 16144 |

- `matrix.json` in the run archive is the exact matrix. Each H16 template uses only that model's complete validation output: unconditional modal K and joint per-slot modal delta/fill, ascending numeric tie break.
- H8 templates bind the original frozen H8 text. The 16,144 requests/M ceiling is inherited from validation NoPF in the completed H8 protocol, not reselected on H16/test results.
- A shared ceiling is an upper bound, not equal realized cache requests or DRAM. Original and reused rows keep their own protocol/source identity.
- H16 versus H8 uses independently trained retained widths, not a new paired-training or multi-seed experiment.

## Exact model and execution boundary

| Original retained seed 7 model | FP32 parameters | Weights | Global h/c |
| --- | ---: | ---: | ---: |
| [AMPM H16](../../602_lstm_ampm_live_inference/README.md) | 6,653;14 tensors, v 7 Poisson/mixture | 26,612 B | 128 B |
| [SPP H16](../../602_lstm_spp_live_inference/README.md) | 6,609;16 tensors, empirical-prior hurdle/mixture/fill v 2 | 26,436 B | 128 B |

- Real `.pt` and training metadata are retained in each sibling's `models/h16/`. The driver validates seed/policy/width/parameter metadata and exports independent `models/{ampm_h16,spp_h16}` binaries. H8 is never substituted.
- The archived v 2 executable and actual compiled scalar C++ read H from `LV602R1`. Prepare verifies executable/source bytes; it does not rebuild or replace the historical simulator.
- No imposed global degree cap, FIFO or finite service model. MAC0 IPC and aggregate analytical MAC4 work ratios do not measure finite-resource IPC.
- Native SPP receives prefetch-fill feedback only; the student receives all raw L2 fills. Its internal same-page/confidence destinations remain native. L2/LLC first-source-intent cohorts are separate, not final physical fill or all-level usefulness.

## Training exposure and live history

- Both H16 models were trained on 0–20M,8 epochs. Old AMPM inference started fresh at guard 20M then continued evaluation; old SPP started fresh before complete train→guard→evaluation chronology. Those old states are not loaded here.
- Every live run starts zero h/c at the beginning of its 5M cache warmup. Validation warms 15–20M/measures 20–25M; ROI0 warms 20–25M/measures 25–50M; ROI1 warms 70–75M/measures 75–100M.
- Warmup output is blocked before cache/PQ mutation while actual eligible state inputs advance, including SPP raw fills. Cache/h/c survive ROI start; counters reset. Synchronous MAC0 has no pending inference work to drain.
- Warmup updates are reported separately;5M instructions is not 5M state updates. ROI callback/completion axes are not training amount, per-PC history or a minimum-observation claim.
- ROI0 was already observed in historical work; ROI1 is future. H8 formal references used the same 5M live warmup rule, while old offline/replay curves used different histories.

## Run, restore and resume

- Canonical source: cache root (`/home/qianruw/cache` on Sacramento); run root: `formal_study/runs/602_width_templates_20261017`. All temporary/compiler/Python/plot caches stay in that run's `runtime/`.
- Use the existing CPU Torch environment and canonical-path launcher; it translates historical decoded path strings in memory, without changing archived JSON or model/protocol controls.

```sh
cd /home/qianruw/cache
study_python=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
study_driver=formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/driver.py
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" status
"$study_python" docs/602_width_templates/canonical_paths.py "$study_driver" all
```

- Stages: `prepare → certify → validation → select → test`; `matrix` prints expected rows, `status` inspects state, `all` resumes compatible rows and writes `pipeline_result.json`. Do not start a second active controller.
- `prepare` snapshots checkpoints/metadata/models, verifies the archived build and enforces exact source bindings. After prepare, changed config/driver/export implementation requires a new protocol/run root. Preserve conflicting output; failed/incomplete attempts move to `preserved_attempts`.
- `certify` builds a probe from the actual archived runtime with `-O3 -ffp-contract=off`; compares real PyTorch h/c/count/all actions across consecutive callbacks and reset/address boundaries; and binds exported model bytes. These are correctness fixtures, not simulator pilots.
- Restore missing original checkpoints from [AMPM H16](../../602_lstm_ampm_live_inference/models/h16/) or [SPP H16](../../602_lstm_spp_live_inference/models/h16/): `original_checkpoint.pt` goes to configured `model.pt`; `source_training_metadata.json` goes to adjacent `run_metadata.json`. Compare bytes with retained and archived copies; do not overwrite differences or substitute another teacher/width.
- The run archive also preserves `checkpoints/*_h16/`, exported `model.bin` and `immutable_model.bin`. Prepare can regenerate a missing export from the original; restore bound files exactly for an existing run.
- Restore a missing actual v 2 build from `formal_study/archive/20261017/raw_v2/builds` and its `build.json`; a similar binary is not a substitute. Reference bindings include the original strict 64-row audit, source/run/summary/model/template/control evidence.
- Per run retain `run.json`, complete summaries/snapshots/logs, frozen student bytes or template text/path; uncapped full students retain callback CSVs. Compatible old rows are reused without relabeling them as new measurements.
- Removed dated top-level aliases must not be recreated. For a local read-only gate add `--local --check` before the driver path; see [layout/recovery](../../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).

## Host storage and analytical work

- [profile_cost.py](profile_cost.py) starts no simulator. Via the same canonical launcher, it compiles the existing storage probe against actual archived runtime/native headers and reuses the accepted AMPM 4104/SPP 4112 correctness fixtures.
- [Cost fields](../../../../formal_study/results/602_width_templates/cost_fields_h16.json) retain headers, C++ `sizeof`, persistent requested `new/new[]` bytes and fixture-observed allocation peaks; source/input/compiler evidence remains in the raw `cost/` directory.

| GCC8.5,64-bit host model | Weights + h/c payload | Object + persistent requested heap | Observed incremental inference heap peak |
| --- | ---: | ---: | ---: |
| AMPM H16 | 26,740 B | 28,596 B | 884 B |
| SPP H16 | 26,564 B | 28,656 B | 1,148 B |

- Runtime object 168 B; persistent requested heap 28,428/28,488 B respectively **already includes weights/h/c**. Do not add payload again. Construction peaks are separate.
- These include scalar-runtime map/string/vector allocations, but exclude outer Four 602 adapter, observer/log/fixture memory, stack, allocator metadata and C-library internals. They are neither RSS nor hardware storage/PPA. The uncapped decoder's finite fixture peak is not a universal bound; H8 measurements cannot stand in for H16.
- H16 actual counters: AMPM demand MAC=`5136+256·1[K>0]+1008K`, nonlinear=`81+16·1[K>0]+53K`; SPP fill 5184 MAC/80 nonlinear; SPP demand MAC=`5216+16·emit+1136K`, nonlinear=`80+emit+53K`.
- K is raw learned degree before gates; emit is SPP's positive class. Formula audit reproduces 30,401,424/34,767,136 MACs and 833,059/953,582 nonlinear counts on the fixtures, including the first callback. It is arithmetic consistency, not another performance experiment.
