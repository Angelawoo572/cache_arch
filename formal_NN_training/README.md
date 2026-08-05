# formal_NN_training

This directory contains the completed 602 matched-input study and the active
623 Stride/SPP redesign.  Per-track READMEs, `data/stream_contract.json`, the
stable `python/model_contract.py`, immutable run metadata, replay results, and
TeX reports are the source of truth.

## Fairness boundary

For every primary comparison, offline normal and offline NN consume the same
chronological source-visible input stream and are replayed through the same
keyed replayer.  The NN never receives captured normal actions or candidates,
normal private state, a normal-derived request budget, or future labels at
runtime.  Captured actions may be used only as supervised output labels and as
the offline-normal comparison list.

This is **same-input fairness**, not algorithm imitation.  It intentionally
leaves the NN free to learn its own recurrent state and direct-address output
distribution.  Normal output templates, page rules, thresholds, and degree are
not neural inference rules.  A training-derived exact-delta vocabulary is an
output representation, not a normal candidate bank, and its continuous
`OTHER` escape provides broad bounded cache-line-delta coverage. Only the
TRAIN-derived categorical entries are integer-exact; the continuous escape
does not guarantee either domain endpoint or every 58-bit integer delta.

## Completed 602 reference

The 602 study compares `stride`, `streamer`, `ampm`, and `spp` with their
matched offline-normal teachers.  Its reusable experimental principles are:

- preserve the exact public input boundary and causal chronology;
- use normal actions as labels, never as runtime features or decoder feedback;
- decode the NN's own actions at evaluation and replay every policy through the
  same queue/cache path;
- report IPC, miss rate, accuracy, selected accuracy, coverage, timeliness,
  request pressure, queue outcomes, and fill outcomes separately;
- treat model capacity as an experiment rather than assuming that larger is
  better.

The 602 architectures are references, not templates for 623.  In particular,
602 Stride used exact-PC recurrence and a hurdle/count plus signed-log-delta
decoder, while 602 SPP used the chronological demand/fill stream, one causal
recurrent path, count/delta heads, and target fill prediction.  The 602 results
also show that the NN can improve by deviating from its teacher, so those runs
support the fairness protocol; they do not justify copying a normal
prefetcher's internal state machine into the NN.

| Track | Reused from 602 | Changed for 623 v21 | Evidence-based reason |
|---|---|---|---|
| Stride | exact-PC causal LSTM state; sparse-label balancing; signed-line-delta output | raw PC64/line58 encoder; dynamic exact-delta+OTHER head; rankwise STOP/EMIT replaces hurdle and rounded count | 623's prior models produced mismatched/no-fill targets, while 602 capacity results were non-monotone; the primary encoder should not pre-impose a Stride feature hypothesis |
| SPP | one global chronological demand/fill LSTM; target-conditioned fill | unweighted rankwise STOP/EMIT; independent ranks; deterministic prior-corrected fill MAP | SPP's action-token prior is dense, and v16--v18 exposed count/target and rare-fill factorization failures; keyed sampling added variance without solving them |

Thus the two 623 tracks deliberately do not share one loss recipe.  They share
the same direct-action representation and causal/no-feedback discipline, while
the observed TRAIN token priors determine whether STOP/EMIT balancing is
appropriate.

## Audited 623 history

The 623 commit history was reviewed as a sequence of hypotheses rather than as
an architecture to preserve:

| Phase | Tested idea | Evidence retained |
|---|---|---|
| Candidate-gated LSTM/CNN | Suppress or rank captured normal candidates | Too close to the normal action interface for the independent-NN question |
| Free-action mixture decoders | Hurdle/count and continuous delta mixtures | Several runs collapsed to no actions or poorly aligned hard decisions |
| v15 keyed CRN | Event-local stochastic count/delta/fill sampling | Reproducible accounting, but neither LSTM track beat its matched offline normal |
| v16--v18 decoder revisions | Balanced/deterministic gates, MAP or peak decoding, and harder feedback alignment | Repeated count/target/fill failures showed that changing only capacity or decode mode did not solve the objective/action mismatch |
| v19 routed grammar | Global/local recurrent routing with sampled STOP/EMIT and exact LEB128 address generation | Exact addresses required a long fragile token trajectory; the replay pipeline's Python 3.6 issue was fixed, but no completed v19 model-quality replay result exists |

The v19 plumbing fix is therefore not evidence that v19 succeeds or fails.
The active v21 redesign starts from the fairness boundary and the observed
failures, not from the v19 grammar or the v20 gate/count factorization.  v20
made the address support train-derived, but still trained trigger, rounded
positive count, and target decisions as separate heads.  The hard action list
could therefore disagree with every one of those individually reasonable
predictions.  No completed v20 replay establishes that this mismatch was
solved.

## Active 623 design constraints

The active tracks are:

```text
experiments/623_offline_lstm_stride/
experiments/623_offline_lstm_spp/
```

Both are independent direct-action LSTMs.  They share a rank-conditioned
direct-action decoder: at each rank the network predicts `STOP` or `EMIT`, and
an emitted action selects a TRAIN-derived exact signed-line-delta class or a
learned continuous `OTHER` escape.  The terminal `STOP` is supervised directly,
so trigger and request count are one learned sequence decision rather than a
gate plus rounded count template.  Ranks are conditionally independent given
the causal callback state and rank encoding: neither teacher nor predicted
actions are fed back.  Inference uses deterministic argmax decisions and a
fail-closed resource watchdog, not a probability threshold or a policy degree
cap.  Neither decoder copies normal source templates or forces same-page
actions.

Stride retains the useful 602 insight that exact-PC causal recurrence matches
the public PC/address stream, but v21 deliberately returns the primary encoder
to lossless raw PC/address bits.  Its sparse `STOP`/`EMIT` supervision uses only
a TRAIN-derived inverse-frequency class weight; that is a loss statistic, not
an inference threshold.  SPP retains the 602 chronological
`DEMAND(addr)`/`CACHE_FILL(evicted_addr)` input and one global causal LSTM.  Its
denser action-token prior uses unweighted `STOP`/`EMIT` loss, and its learned
fill choice is conditioned on the decoded target and rank and selected by
argmax.  The recorded SPP fill-callback stream is source-conditioned, so the
result is a matched-input offline comparison, not a closed-loop live-NN claim.

Both tracks sweep hidden sizes 8, 16, 32, 64, and 128 because the completed 602
results were strongly non-monotone in capacity.  Guard checkpoint selection is
lexicographic over separately reported action diagnostics; it never averages
unlike metrics into a composite.  Evaluation rows remain excluded from
training and checkpoint selection.  However, prior 623 iterations have already
been inspected on this benchmark region, so a v21 replay is comparative held-out
evidence rather than a pristine once-only confirmatory test.

The exact run IDs, hidden-size points, parameter counts, objectives, and
decoder revisions are generated from each track's
`python/model_contract.py`.  `experiments/validate_direct_action_contracts.py`
loads those contracts instead of embedding active version tokens.

Historical scripts remain under `legacy/scripts/` for provenance only.  The
active two-track workflow keeps track-specific collection, training, replay,
analysis, and report scripts in their existing experiment directories; no
cosmetic folder duplication is required.

Large run data and Colab artifacts remain outside Git.  The single shared
`common/split_colab_archive.py` helper creates or rejoins gzip archives as
numbered parts of at most 90 MiB and verifies whole-archive and per-part SHA-256
values from a JSON manifest.  The active notebooks can recover such parts from
Google Drive or the Colab file chooser and use the same format for large
outputs.
