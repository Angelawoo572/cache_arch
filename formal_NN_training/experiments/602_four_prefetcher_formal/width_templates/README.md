# Frozen width/template supplement

This separate protocol is four602_width_templates_20261017_v1. It does not overwrite the completed H8 v2 protocol, models, templates or results. It reuses the actual archived v2 executable and compiled C++ source, whose generic LV602R1 runtime reads H from each model header. No common C++ change or retraining is needed.

The new root is /home/qianruw/cache/formal_study/runs/602_width_templates_20261017; the canonical source is /home/qianruw/cache/formal_study/work/cache_gate_mechanism. Local copies use /Users/angelawoo/Documents/CMU/ece/cache/formal_study/. Compilation/Python caches belong under the new run root's runtime/, configured by the driver. The driver has no SSH logic.

The exact 24-row matrix is saved in matrix.json:

- Two full H16 validation runs: retained seed-7 AMPM and SPP students, records 20–25M, each preceded by 5M cache/state warmup.
- Sixteen H16 test runs: AMPM/SPP × ROI 25–50M or 75–100M × full NN or no-NN template × uncapped or common cap 16144 requests per million retired instructions.
- Six H8 template test runs: Streamer/AMPM/SPP × those two ROIs at the same cap, using the existing H8 validation-selected templates.

The cap is fixed from the completed reference protocol, not selected again using H16 validation or test data. Each H16 template uses only that H16 model's full validation output: modal K and joint per-slot modal delta/fill; ties use ascending numeric order. H8 controls bind the original frozen H8 text. A common cumulative ceiling does not guarantee equal realized cache requests or DRAM traffic.

The actual H16 checkpoints remain under cache/formal_NN_training/experiments/602_offline_lstm_*. Prepare snapshots both .pt and run_metadata.json files, validates seed/policy/width/parameter metadata, and exports independent models/ampm_h16 and models/spp_h16 binaries. AMPM is the real 14-tensor Poisson/mixture model (6653 parameters); SPP is the real 16-tensor empirical-prior Hurdle/mixture/fill model (6609 parameters). H8 exports are never substituted.

AMPM H16 has 26612 B of FP32 weights and 128 B of global h/c payload. SPP H16 has 26436 B of weights and 128 B of h/c. Those sums exclude implementation control/scratch/output and are not complete storage or PPA. H8 host heap/scratch values cannot be reused for H16. The global runtime has no imposed degree cap, FIFO or finite service model.

From the canonical checkout:

    /home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/driver.py all

Resumable stages are prepare, certify, validation, select, test. Status reads current states and matrix prints the exact expected rows. All writes pipeline_result.json. Do not edit config.json, driver.py or ampm_h16.py after prepare: their source bytes are bound. A changed implementation requires a new protocol/run root. Failed attempts are preserved before resuming.

Certify compiles the probe from the actual archived simulator's scalar C++ implementation with -O3 -ffp-contract=off. It compares each H16 checkpoint with its real PyTorch class on consecutive real callbacks and address/reset boundaries, including full h/c and complete actions. These are correctness fixtures, not simulator pilots. Full warmup and ROI budgets follow only after the gate passes.

Each run retains run.json and full summary/snapshot/log outputs, actual model bytes for students, and frozen template text/path in template specifications. Full uncapped student runs retain callback CSVs. The unchanged lifecycle scopes and SPP raw-fill limitations remain in [method limits](../report/METHOD_LIMITATIONS.md).

Reference bindings join original rows by explicit source, common controls and byte evidence; they do not relabel their protocol. They bind the original strict 64-row audit, run/summary files, frozen models/templates, executable and compiled source. Result provenance stays distinct even when the executable is shared. Matching NoPF/native/H8 references are reused without extra simulator runs.

These are MAC0 functional live runs. Operation counters and aggregate MAC4 work ratios are analytical cost evidence, not measured finite-resource IPC. Negative/null outcomes remain results. H16 versus H8 compares retained independently trained widths, not a new paired retraining experiment. ROI 25–50M was already observed in historical work; ROI 75–100M is reported separately.

## Restore and resume

Original model sources are config.json's model_root checkpoint paths. The repository also keeps portable, byte-identical originals:

- AMPM: ../../602_lstm_ampm_live_inference/models/h16/original_checkpoint.pt and source_training_metadata.json.
- SPP: ../../602_lstm_spp_live_inference/models/h16/original_checkpoint.pt and source_training_metadata.json.

These paths are relative to this README (both are teacher-specific live siblings under experiments). If the original server model root is lost, restore the .pt to the configured model.pt path and its source_training_metadata.json to the corresponding run_metadata.json; compare the restored bytes with the portable originals and any existing raw checkpoints snapshot. Do not change frozen paths or replace one width/teacher with another. The run archive also holds checkpoints/*_h16/model.pt, run_metadata.json, exported model.bin, and immutable_model.bin. Restore an existing archive's bound files exactly rather than silently generating new identities. Prepare refuses changed source/checkpoint/model bytes. It regenerates a missing export from the original checkpoint; incomplete conflicting output should be preserved and investigated, not overwritten.

The reused executable and actual compiled sources remain under formal_study/archive/20261017/raw_v2/builds, referenced by that archive's build.json. Prepare verifies the executable against its binary snapshot and each stored compiled source against the actual simulator tree. H16 parity compiles a standalone probe from those same actual runtime sources, then binds its behavior to each exported H16 checkpoint. It does not rebuild or replace the historical simulator. A missing archived actual build must be restored from its preserved archive before resuming; another binary is not a substitute merely because its configuration looks similar.

The main controller can be resumed using the all entrypoint: complete compatible formal rows are verified and skipped; failed/incomplete prior row directories are moved under preserved_attempts. Do not start a second controller while one is active. For inspection, use status and the per-run run.json, pipeline_result.json and controller log. Formal stages may also be invoked individually after their gates: prepare → certify → validation → select → test. All phase counters and complete raw outputs remain in the independent new run root.

## Weight-training interval versus live state history

Both retained H16 models were already trained on records 0–20M, but their original inference histories differed. AMPM's old offline report initialized fresh h/c at guard start20M, advanced it through guard20–25M, then continued evaluation25–50M. SPP's old offline report initialized fresh h/c before the complete train0–20M → guard20–25M → evaluation25–50M chronology. Neither old state is loaded into this live experiment; see [evidence alignment](EVIDENCE_ALIGNMENT.md).

Each new live run creates zero h/c at the beginning of its selected 5M cache warmup. Validation warms records 15–20M and measures 20–25M. Test ROI0 warms 20–25M and measures 25–50M; ROI1 warms 70–75M and measures 75–100M. All warmup outputs are rejected before cache/PQ mutation, while the student still processes its actual eligible callbacks (SPP includes raw fill events). At ROI start the cache and actual h/c are retained, measurement counters restart, and no NN work is pending because this engine is synchronous MAC0. Warmup completed-update counts are reported separately; 5M instructions is not the same as 5M recurrent updates.

The H8 reference formal runs already used the same 5M live warmup rule, so they are compatible controls. The old offline/replay curves with AMPM's guard-only or SPP's train-plus-guard initialization are not interchangeable with either live protocol. Source-training exposure, live warmup history and ROI observation axes remain distinct; a cumulative IPC curve's first baseline crossing is not a minimum-observation claim.


## Actual H16 host-memory evidence

The separate profile_cost.py utility never starts ChampSim. Run it with the configured server Python from the canonical checkout after the H16 correctness gate passes. It compiles the existing validation/storage_probe.cc against the actual archived simulator runtime and native headers, checks the core bytes against the canonical shared runtime, and runs exactly the already-passed AMPM4104 / SPP4112 callback correctness fixtures.

The output cost/cost_fields_h16.json retains separate ampm_h16 and spp_h16 entries with actual model headers, C++ sizeof layouts, persistent requested new/new[] bytes, object-plus-persistent bytes, and fixture-observed temporary allocation peaks. The persistent implementation totals already include weights and h/c: do not add the payload sum again. Source/header/model copies, the complete input commands and compiler logs accompany the measurements. Native type sizes describe C++ layouts, not complete baseline constructor heaps.

This is host implementation memory, with ordinary C++ allocation requests tracked and scalar-runtime map/string/vector allocations included. It excludes the outer Four602 adapter, observer/log/input-fixture memory, automatic stack frames, allocator metadata and C-library internals; it is neither RSS nor a synthesized deployment-storage/PPA total. The global decoder remains uncapped, so the observed K and temporary peak on a finite correctness fixture do not bound all inputs. MAC0 operation counts and analytical MAC4 work ratios do not establish finite-resource IPC.


The measured GCC8.5 64-bit host evidence is:

| Actual H16 model | FP32 weights + global h/c | Runtime object + persistent requested heap | Observed incremental heap peak |
|---|---:|---:|---:|
| AMPM | 26,740 B | 28,596 B | 884 B |
| SPP | 26,564 B | 28,656 B | 1,148 B |

Each runtime object is 168 B. Persistent heap is 28,428 B /28,488 B respectively and already contains the weights/state. The peak column is an inference increment above persistent requested bytes on the documented correctness fixture, not a full-program peak or deployment upper bound. Construction peaks are recorded separately in the raw JSON.

For H16, the [actual core counters](../../../common/live602_runtime/live602_runtime.cc#L42) yield AMPM demand MAC=5136+256·1[K>0]+1008K and nonlinear=81+16·1[K>0]+53K. SPP fill uses 5184 MAC/80 nonlinear; SPP demand uses 5216+16·emit+1136K MAC and 80+emit+53K nonlinear. Here emit is the empirical-prior positive class; K is the raw learned degree before outer gates. On the existing fixtures these reproduce all 30,401,424 /34,767,136 MACs and 833,059 /953,582 nonlinear counts exactly, including the first individual callback. The separate cost/operation_formula_audit.json records this arithmetic consistency check; it is not another performance experiment.
