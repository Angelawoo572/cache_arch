# `formal_NN_training/scripts`

Sacramento-side Python scripts in this directory are Python 3.6 compatible and use only the standard library.

## Stable core pipeline

`01` through `09`, then `11` through `15`, are the raw-oracle, keyed-replay, event-evidence, and capacity-control workflow. Keep these paths stable because current notebooks, reports, and server commands reference them.

## Optional feature builders

`10_profile_champsim_trace.py`, `14_build_base_candidate_table.py`, `16_build_trace_dependency_features.py`, and `17_prepare_v3_9_605_dependency_sidecar.sh` are not duplicates. They create different raw-trace or base-aware artifacts and should not be deleted.

## Replay helpers

- `replay/verify_same_binary_no_pref.py` is the canonical same-binary IPC guard.
- `v4/run_oracle_ceiling_replay.sh` is the canonical v4 ceiling replay entrypoint.
- `19_build_oracle_ceiling_lists.py` builds ceiling rich lists.
- `21_join_decision_ledger_attribution.py` joins audit rows to one matching full decision ledger.
- `22_resource_summary.py` summarizes measured PQ/MSHR and request pressure.
- `25_build_v4_1_notebook.py` is Colab-only.

The duplicate `16_verify_same_binary_no_pref.py` was removed. Script `08_run_standalone_lstm_replay.sh` now uses the canonical verifier under `replay/`.

`19` and `20` remain separate because one is a pure list builder while the other owns simulator replay. `21` and `22` consume different evidence types, so merging them would make failures less diagnosable.
