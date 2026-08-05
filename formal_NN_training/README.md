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
distribution.  A learned zero/positive hurdle followed by a learned positive
count is an evidence-backed factorization of the variable-length output, not a
copy of a normal prefetcher's tracker, confidence counter, request template, or
degree rule.  The hurdle is decoded by categorical argmax; it is not a tuned
probability threshold.  Normal output templates, page rules, thresholds, and
degree are not neural inference rules.  A training-derived exact-delta
vocabulary is an output representation, not a normal candidate bank, and its continuous
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

| Track | Reused from 602 | Active 623 v22 design | Evidence-based reason |
|---|---|---|---|
| Stride | exact-PC causal LSTM state; TRAIN-derived sparse-label balancing; learned hurdle/count; signed-line-delta output | lossless raw PC64/line58 encoder; balanced learned ZERO/POSITIVE hurdle; learned positive log-count; rank-conditioned dynamic exact-delta+OTHER head | 602 shows hurdle/count can learn variable request cardinality without a fixed degree, while 623's earlier failures require raw inputs, aligned hard decoding, and stronger fail-closed accounting |
| SPP | one global chronological demand/fill LSTM; empirical-prior hurdle/count; target-conditioned fill | natural-prior ZERO/POSITIVE hurdle; learned conditional positive log-count; independent rank delta decisions; deterministic prior-corrected fill MAP | SPP's callback-level trigger prior is not extremely sparse; natural-prior hurdle initialization avoids inventing a threshold, while deterministic decoding removes keyed-sampling variance |

Thus the two 623 tracks deliberately do not share one loss recipe.  They share
the learned hurdle/positive-count decomposition and causal/no-feedback
discipline, while the observed TRAIN callback priors determine whether hurdle
balancing is appropriate.

## Audited 623 history

The 623 commit history was reviewed as a sequence of hypotheses rather than as
an architecture to preserve:

| Phase | Tested idea | Evidence retained |
|---|---|---|
| Candidate-gated LSTM/CNN | Suppress or rank captured normal candidates | Too close to the normal action interface for the independent-NN question |
| Free-action mixture decoders | Hurdle/count and continuous delta mixtures | Several runs collapsed to no actions or poorly aligned hard decisions |
| v15 keyed CRN | Event-local stochastic count/delta/fill sampling | **Completed negative:** reproducible accounting, but neither LSTM track beat its matched offline normal |
| v16--v18 decoder revisions | Balanced/deterministic gates, MAP or peak decoding, and harder feedback alignment | **Completed negatives:** replayed runs repeatedly exposed count/target/fill failures; these results justify redesign, not a claim that all hurdle models fail |
| v19 routed grammar | Global/local recurrent routing with sampled STOP/EMIT and exact LEB128 address generation | **Unvalidated:** plumbing was repaired, but no completed model-quality replay establishes success or failure |
| v20 split direct-delta heads | Train-derived address support with separate trigger, rounded positive count, and target decisions | **Unvalidated:** no completed replay establishes success or failure; the possible hard-action mismatch was a design risk, not measured evidence |

The v19 plumbing fix is therefore not evidence that v19 succeeds or fails, and
the same restraint applies to v20.  Only v15--v18 are completed negative
evidence.  The active v22 redesign starts from the fairness boundary, the
completed failures, and the successful 602 hurdle precedent; it does not turn
unvalidated v19/v20 hypotheses into empirical conclusions.

## Active 623 design constraints

The active tracks are:

```text
experiments/623_offline_lstm_stride/
experiments/623_offline_lstm_spp/
```

Both are independent direct-action LSTMs.  A learned categorical hurdle first
chooses `ZERO` versus `POSITIVE`; on positive callbacks a learned log-count head
predicts the request count, and independent rank-conditioned heads choose a
TRAIN-derived exact signed-line-delta class or a learned continuous `OTHER`
escape.  This is a neural likelihood factorization of zero inflation and
positive cardinality.  It does not read or reproduce a normal prefetcher's
confidence, degree, candidates, or private state.  Neither teacher nor decoded
actions are fed back.  Inference uses deterministic argmax/rounded learned
values and fail-closed resource watchdogs, not a probability threshold, forced
count, truncation rule, or policy degree cap.  Neither decoder copies normal
source templates or forces same-page actions.

Stride retains the useful 602 insight that exact-PC causal recurrence matches
the public PC/address stream, but v22 deliberately keeps the primary encoder
at lossless raw PC/address bits.  Its sparse hurdle supervision uses only a
TRAIN-derived inverse-frequency class weight; that is a loss statistic, not an
inference threshold.  SPP retains the 602 chronological
`DEMAND(addr)`/`CACHE_FILL(evicted_addr)` input and one global causal LSTM.  Its
denser callback prior uses unweighted natural-frequency hurdle loss, and its learned
fill choice is conditioned on the decoded target and rank and selected by
argmax.  The recorded SPP fill-callback stream is source-conditioned, so the
result is a matched-input offline comparison, not a closed-loop live-NN claim.

Both tracks sweep hidden sizes 8, 16, 32, 64, and 128 because the completed 602
results were strongly non-monotone in capacity.  Guard checkpoint selection is
lexicographic over separately reported action diagnostics; it never averages
unlike metrics into a composite.  Evaluation rows remain excluded from
training and checkpoint selection.  However, prior 623 iterations have already
been inspected on this benchmark region, so a v22 replay is comparative held-out
evidence rather than a pristine once-only confirmatory test.

The exact run IDs, hidden-size points, parameter counts, objectives, and
decoder revisions are generated from each track's
`python/model_contract.py`.  `experiments/validate_direct_action_contracts.py`
loads those contracts instead of embedding active version tokens.

Historical scripts remain under `legacy/scripts/` for provenance only.  The
active two-track workflow keeps track-specific collection, training, replay,
analysis, and report scripts in their existing experiment directories; no
cosmetic folder duplication is required.

Large run data and Colab artifacts remain outside Git.  The active notebooks
default to one run-specific `.colab_input.tar.gz` cached in Google Drive and one
`.colab_output.tar.gz`.  A newly supplied input may retain an older source-run
filename: the notebook records its original name and SHA-256, validates every
payload against `SHA256SUMS`, runs the track's collected-input validator, then
caches the verified bytes under the current run ID.  The shared
`common/split_colab_archive.py` format remains as an optional compatibility
fallback when a single-file transfer is genuinely impractical; it verifies the
whole archive and every numbered part.
