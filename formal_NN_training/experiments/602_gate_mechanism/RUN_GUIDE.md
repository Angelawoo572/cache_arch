# 602 gate / payload mechanism study

This study reuses the three independently trained balanced main-v2 checkpoints.
It adds no weight training, parameter sweep, performance pilot, or test fitting.

The matrix contains **64 fresh full formal runs and 7 compatible forward-Stride
references**. Both future ROIs contain 25M retired instructions, with the same 5M
NoPF cache warmup and 5M visible NN state history. The two fresh NoPF runs are
formal references and establish compatibility with the earlier build. The fixed
2363 requests/M ceiling was selected by the original validation forward-1 run.

| Treatment | Fresh runs |
|---|---:|
| Learned NN, MAC0, original/current-line-filtered, 3 seeds, uncapped/capped, 2 ROIs | 24 |
| Full NN forward/h/c with emit-gated fixed line+2 payload, filtered, 3 seeds, both caps/ROIs | 12 |
| Unconditional line+2, filtered, both caps/ROIs | 4 |
| Frozen PC-gate line+2, filtered, 3 seed tables, both caps/ROIs | 12 |
| Learned NN MAC4, seed7, capped, original/filtered, both ROIs | 4 |
| Original learned NN MAC0 seed7, ROI1, teacher-silent-output suppression | 1 |
| Fresh NoPF | 2 |
| Fresh native SPP, uncapped/capped, both ROIs | 4 |
| Previously missing capped forward-2, ROI1 | 1 |

The 7 earlier forward-1/2 results retain their original record, protocol and build
identity in `compatible_reference_manifest.json`; they are never relabeled as new
executions. Analysis must finish the fresh-NoPF equality audit before using them.

## Offline PC table contract

The scorer consumes **the complete saved NoPF training and validation CSVs**,
not `events.jsonl` or its 4096-event diagnostic prefix. Training covers 0–1M;
validation covers 5–10M. Each split starts with zero h/c, has an exact-PC 64-entry
LRU table, and completes one scalar FP32 forward per event. This deliberately
does not give validation a 4M NN-state warmup. It is an offline fitting contract,
distinct from the 5M-history formal performance runs.

Each seed's original model is scored independently. The combined train+validation
event frequency selects at most 64 PCs, with ascending unsigned PC breaking ties.
For each selected PC, `gate = (2 * positive raw NN emits >= event count)`; a tied
majority emits. The gate uses `result.emit`, even if the learned count is zero.
Unlisted PCs remain silent. No threshold or capacity is selected on test data.
Train, validation and combined agreement, table misses, raw row counts, full
per-PC votes, emit-with-K0 counts and source snapshots are preserved.

ASCII table ABI, consumed by the actual engine:

```text
pc gate count positives
1234 1 20 10
5678 0 17 3
```

The fitter performs a strict write/read roundtrip. The coordinated C++ engine
unit harness must also load all three fitted tables before formal launch.

## Runtime environment ABI

- `DYNAMIC602_METHOD=frozen|fixed_plus2|pc_gate_plus2|none|stride`.
- `DYNAMIC602_PAYLOAD=learned|fixed_plus2`. For `frozen+fixed_plus2`, retain the
  complete original forward, compute/admission and h/c update policy; only the
  completed payload becomes one line+2 request when raw NN emit is true.
- `DYNAMIC602_FILTER_CURRENT=0|1`: common current-line filter before budget credit.
- `DYNAMIC602_GATE_TABLE=/absolute/path/pc_gate.tsv`: required for PC-gate mode.
- `DYNAMIC602_PHASE_INSTRUCTIONS=5000000`: five full-ROI mechanism phase bins.

All standard finite-engine, page, queue, skip/warmup/ROI and snapshot settings are
explicit in each `run.json`. Parent-process `DYNAMIC602_*`/`STRIDE_LSTM_*` values
are removed before those settings are applied.

## Resume and launch

Run from the new source checkout. Configuration records exact remote paths and
the common ChampSim base; all large raw data and builds stay outside Git.

```bash
EXP=formal_NN_training/experiments/602_gate_mechanism
python3 -m unittest discover -s "$EXP/validation" -p 'test_gate_runner.py' -v
python3 "$EXP/run_gate.py" matrix
python3 "$EXP/run_gate.py" build
python3 "$EXP/run_gate.py" fit
```

After coordinated engine, measurement, numerical and actual-table loader checks,
the root controller writes `correctness_ready.json` in the new raw root:

```json
{"status":"PASS","protocol":"gate_mechanism_20261016_v1","build_id":"CURRENT_BUILD_ID","checks":["engine","measurement","runner","numerical parity","three actual PC tables"]}
```

Then launch the **full formal matrix** and resume with the same command:

```bash
python3 "$EXP/run_gate.py" pipeline
python3 "$EXP/run_gate.py" status
```

Build/source/config/model/table equality is checked before output reuse.
Incomplete or incompatible attempts are moved to `preserved_attempts`; active
processes are never treated as completed. Postprocessing and documents are
outside the compiled-source identity and may be edited while jobs run. Existing
authentication and SSH host verification are unchanged.

## Final analysis and compact artifacts

Keep the complete raw root outside the Git checkout. `analysis_gate.py` checks
all 71 planned rows, actual method/model/table identity, retired denominators,
credits, cache/PQ/cohort/DRAM conservation, full distributions, and fresh NoPF,
original-NN and native-SPP functional compatibility with main-v2. It keeps the
old reference protocol/build identities. A partial analysis is explicitly
incomplete and cannot authorize the final figures or summaries.

```bash
python3 "$EXP/analysis_gate.py" --raw NEW_RAW --prior-raw ORIGINAL_V2_RAW --out FRESH_ANALYSIS
python3 "$EXP/summarize_gate.py" --raw NEW_RAW --analysis FRESH_ANALYSIS --out FRESH_SUMMARY
python3 "$EXP/plot_distributions.py" --analysis FRESH_ANALYSIS --summary FRESH_SUMMARY --out FRESH_FIGURES
python3 "$EXP/validation/verify_spp_cohorts.py" --help
```

Matplotlib is required for plots. Final summaries and distribution figures
require a complete PASS audit. The independent SPP audit's `--require-complete`
mode is described in
[COHORT_AUDIT.md](validation/COHORT_AUDIT.md). Definitions and all raw fields
remain in the analysis root; compact committed tables preserve every training
seed, actual request rates and real controller traffic. The example event log
is bounded and is never used to estimate full-ROI distributions or fit a table.

`fixed_plus2` is an explicitly registered dynamic-engine conventional method,
with `engine_method=fixed_plus2`, h=0 and zero NN forwards. It directly emits one
128-byte-forward line-aligned target per eligible LOAD, subject to page and
credit gates. It does **not** invoke native `next_line`; therefore no inherited
native next-line default or runner fallback determines this arm.
