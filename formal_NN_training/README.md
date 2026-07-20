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

Both v15 tracks use stateless SHA-256 event-keyed inverse-CDF sampling.  A fixed
decoder seed therefore supplies the same event-local quantiles to every LSTM
capacity without a mutable RNG stream; one callback can no longer shift all
later samples.  Stride uses exact-PC recurrent state and a lossless 122-feature
PC/cache-line encoder.  SPP uses the chronological source-visible
`DEMAND(addr)`/`CACHE_FILL(evicted_addr)` stream, a lossless 59-feature encoder,
and a joint delta-component/fill decoder.

The SPP input comparison is deliberately described as matched-input offline
replay.  Its recorded fill-callback stream came from the source SPP run, so the
result is not a closed-loop live-NN claim.

Current default runs:

```text
623_offline_lstm_stride_keyed_crn_v15_seed7
623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7
```

Stale seven-track split helpers live under
`legacy/scripts/direct_action_split_workflow/`; they are provenance only and
are not part of the active two-track workflow.
