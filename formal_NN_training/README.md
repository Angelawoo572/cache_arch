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
The active v20 redesign starts from the fairness boundary and the observed
failures, not from the v19 grammar.

## Active 623 design constraints

The active tracks are:

```text
experiments/623_offline_lstm_stride/
experiments/623_offline_lstm_spp/
```

Both are independent direct-action LSTMs.  They share a learned, bounded exact
signed-delta vocabulary derived from training labels and a learned continuous
`OTHER` escape for broad bounded line-delta coverage. They use natural-prior
objectives and learned posterior decisions rather than hand-written
probability thresholds.  Neither decoder copies normal source templates,
forces same-page actions, or inherits the normal degree.

Stride retains the useful 602 insight that exact-PC causal recurrence matches
the public PC/address stream, but its address and count decisions are learned;
the current same-PC delta is a causal input feature, not an output rule.  SPP
retains the 602 chronological `DEMAND(addr)`/`CACHE_FILL(evicted_addr)` input
and learns target-conditioned fill, but its target support is not restricted to
the source SPP's page rule.  The recorded SPP fill-callback stream is
source-conditioned, so the result is a matched-input offline comparison, not a
closed-loop live-NN claim.

The exact run IDs, hidden-size points, parameter counts, objectives, and
decoder revisions are generated from each track's
`python/model_contract.py`.  `experiments/validate_direct_action_contracts.py`
loads those contracts instead of embedding active version tokens.

Historical scripts remain under `legacy/scripts/` for provenance only.  The
active two-track workflow keeps track-specific collection, training, replay,
analysis, and report scripts in their existing experiment directories; no
cosmetic folder duplication is required.
