# formal_NN_training

This directory contains the completed 602 matched-input study and the current
623 matched-input LSTM study.  The experiment READMEs, contracts, run metadata,
and TeX reports are the source of truth; older proposal/report PDFs are not used
to choose the current research objective or architecture.

## Completed 602 comparison

Four normal prefetchers were evaluated through matched offline replay:
`stride`, `streamer`, `ampm`, and `spp`.  Each neural track sees only the
external inputs visible to its corresponding normal prefetcher, while captured
normal actions are labels and the offline-normal comparator.  The primary
comparison is offline normal versus offline NN; live normal and no-prefetch are
context.  IPC, miss rate, coverage, selected accuracy, timeliness, and request
pressure remain separate metrics.  There is no composite score and
`1 - selected_accuracy` is not called cache pollution.

## Current 623 work

The active tracks are:

```text
experiments/623_offline_lstm_stride/
experiments/623_offline_lstm_spp/
```

The active Stride v16 experiment keeps the v15 external input contract,
lossless 122-feature PC/cache-line encoder, exact-PC recurrent state, and full
causal chronology.  It replaces the sampled mixture output with a
data-frequency-balanced two-class hurdle, a deterministic positive log-count,
and a deterministic scalar signed-log cache-line delta.  This is a controlled
decoder/objective revision; it does not copy the complete 602 model or add a
teacher-derived runtime input.

The active SPP v16A experiment keeps the v15 chronological source-visible
`DEMAND(addr)`/`CACHE_FILL(evicted_addr)` stream and lossless 59-feature
encoder.  It strictly reloads each preserved v15 checkpoint without retraining,
compares deterministic joint MAP decoder candidates on the guard split, freezes
one decoder per capacity, and uses only that decoder on evaluation.  Parent
checkpoint and metadata hashes make the re-decode lineage explicit.

The SPP input comparison is deliberately described as matched-input offline
replay.  Its recorded fill-callback stream came from the source SPP run, so the
result is not a closed-loop live-NN claim.

Current default runs use new directories and never overwrite v15:

```text
623_offline_lstm_stride_compact_hurdle_v16_seed7
623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7
```

The completed v15 runs remain immutable negative checkpoints:

```text
623_offline_lstm_stride_keyed_crn_v15_seed7
623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7
```

Their replay/accounting checks passed, but neither NN exceeded its matched
offline-normal teacher.  They remain the source of the diagnosis and, for SPP
v16A, the parent weights; they must not be relabelled as v16 or deleted.  Both
active revisions reuse the already collected raw input streams because their
external input revisions are unchanged.  Stride v16 retrains in its new run;
SPP v16A re-decodes the strictly validated v15 checkpoints.

Stale seven-track split helpers live under
`legacy/scripts/direct_action_split_workflow/`; they are provenance only and
are not part of the active two-track workflow.
