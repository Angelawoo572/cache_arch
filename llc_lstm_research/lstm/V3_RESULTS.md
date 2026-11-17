# 605 LLC scorer experiment

- Use the same 16 candidates, seven event features, and six candidate features as v2. Fit the candidate table only from 10M, 30M, and 50M NoPF. Use those same regions to fit the neural models and 70M/130M to select checkpoints and thresholds. Freeze the configurations before 210M, 230M, and 250M. Use separate 270M and 290M regions for the hidden-state dropout follow-up.
- Compare the existing affine head against a small ReLU head. Project the current event and optional hidden state once. Project each candidate, add the two vectors, apply ReLU, and reduce to one score. Test widths 4, 8, and 16 with and without recurrence. The LSTM still carries eight hidden units and eight cell values per core.
- Use NoPF for every training target and service sample. The soft target is the probability that NoPF demand service plus 16 modeled cycles fits in the candidate's lead. The empirical 130M service sample contains 20,361 completed demand misses. The proxy p50/p90/p95 is 226/280/354 cycles. The separate future label uses the first same-core NoPF LLC miss within 4,000 cycles.
- Compare candidate-wise loss with a 0.25-weighted same-trigger ranking term. It compares the highest-scoring best-label candidate with its highest-scoring lower-label rival. The common validation metric gives the two development regions equal weight at the 2,000- and 4,000-decision budgets. The runtime thresholds use the pooled validation scores. [Pairwise ranking](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf)
- For the dropout follow-up, independently remove the entire hidden vector from the scorer at 25% or 50% of training positions. Carry the cell state normally and disable dropout at inference. Check both probabilities on 270M and 290M.

## What the NoPF data permits

- Enumerate all 16 legal candidates and match each trigger and each future miss at most once. At 280–4,000 cycles, the three-region ceilings before a decision cap are 2,247 / 8,419 / 3,781 distinct misses. An oracle can therefore use all 2,000 decisions in each region. This is an address-and-lead ceiling before queueing, cache eligibility, and replay feedback.
- At a fixed event, the affine head has a single slope in the signed offset. The fresh traces contain 3,618 / 6,989 / 2,435 triggers with two matched candidate pairs that require opposite offset preferences under the NoPF timing proxy. The ReLU head can represent such a preference, but none of the tested heads correctly ordered both pairs on those triggers. This diagnostic holds the other candidate features fixed within each pair. [Nonlinear scoring](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- The best recurrent future-label checkpoint used epoch 7. Its 4,000-decision expected unique counts at 210M/230M/250M were 413.64/539.40/520.79. Setting its hidden output to zero with the same weights yielded 511.49/590.26/910.28. Ordered history length and shuffling changed some rankings. Short- and long-history controls gave no consistent evidence that the carried state adds predictive value.

## Inference cost at 16 candidates

| Head | Event-only MACs | LSTM MACs | Exported float32 weights |
| --- | ---: | ---: | ---: |
| V2 affine, compiled compact scoring | 184 | 664 | 2,444 B |
| ReLU width 4 | 476 | 988 | 2,420 B |
| ReLU width 8 | 952 | 1,496 | 2,788 B |
| ReLU width 16 | 1,904 | 2,512 | 3,524 B |

- The event-only exports include unused recurrent coefficients for format compatibility. A physical event-only implementation can omit those coefficients. All arms use the same 782-byte packed candidate table. Recurrence adds 64 bytes of state per core. Counts exclude gate activations, candidate lookup and ranking, and hardware implementation overhead.
- Python/C++ score parity checks covered 4,497,600 candidate cells. A total of 47 differed by one Q10 count and no candidate won in a different position. Legacy affine paths matched the earlier C++ implementation exactly. The 210M NoPF demand log matched the original uncut trace replay byte-for-byte.

## ChampSim replay

- Run two independent copies of the same trace for 2M ROI instructions per core after 200K warmup. Give each core up to 10 accepted requests per 20K retired instructions. Keep the same 16-cycle latency and minimum initiation interval in the main comparison and report actual sends. Test a 32-cycle decision separately. A small-budget check disables score thresholds and allows five per 100K retired instructions, at most 100 per core.
- Offline and simulator results measure different outcomes. The oracle and ranking metrics use NoPF traces. IPC and coverage use each policy's closed-loop traffic.

| Policy | 210M IPC | 230M IPC | 250M IPC | 270M IPC | 290M IPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPF | .731683 | .368373 | .761849 | .218620 | .805061 |
| Saved v2 calibrated LSTM | .732619 | .369324 | .763959 | .217882 | .805073 |
| LSTM, width 8, future label | .731942 | .369412 | .761380 | .218392 | .805734 |
| LSTM, width 8, future label, 25% training-only dropout | — | — | — | .218668 | .806196 |
| LSTM, width 8, future label, 50% training-only dropout | — | — | — | .218564 | .806418 |
| LSTM, width 8, calibrated and pairwise | .731895 | .368829 | .763015 | — | — |
| Event-only, affine, calibrated and pairwise | .731522 | .368263 | .762864 | .218745 | .806462 |
| Event-only, width 4, calibrated and pairwise | .732661 | .369169 | .761295 | .218402 | .806533 |
| Event-only, width 8, calibrated and pairwise | .732834 | .369352 | .763947 | .218991 | .807751 |

- **Main recurrent comparison.** With a future label and the same recurrent encoder, width 8 improved on the retrained affine head at 210M and 230M and lost at 250M. Against the saved v2 LSTM it was faster only at 230M. With the calibrated pairwise loss, width 8 Event-only outran the matched LSTM on all three primary regions.
- **Head-only controls.** The supplemental Event-only affine and width-4 pairwise controls were trained on the same development data and replayed on the existing test regions for attribution. Width 8 had higher IPC than affine on all five regions at the tested common pacing. Actual request counts varied. The width-8 Event-only completion-time gains over NoPF were 0.157%, 0.265%, 0.275%, 0.169%, and 0.333%.
- **Exact-send subset.** At 210M, the saved v2 LSTM, the width-8 future LSTM, and the width-8 paired Event-only each sent 170 requests under the small-budget control. At 230M, the saved v2 and future LSTM each sent 186. At 250M, both sent 180. Their completion-time differences in the small-budget control stayed within 0.02%. Event-only paired counts at 230M and 250M differed, so those rows compare the same allowance with different realized counts.
- **Training-only dropout.** On the two follow-up regions, 25% hidden dropout raised the 4,000-decision expected unique counts from 754/429 to 767/500. At 270M the calibrated live thresholds caused the unmasked and dropout LSTMs to send 218 and 2,000 requests respectively. The measured IPC change includes that sending difference. At 290M the unmasked and 25%-dropout LSTMs sent 365 and 324. The width-8 Event-only pairwise variant remained faster in both regions.
- **Latency sensitivity.** The width-8 paired Event-only gained about 0.15%, 0.34%, 0.28%, 0.18%, and 0.37% over NoPF at 32 modeled cycles. These estimates describe five regions of one trace. The single in-flight decision lane also changes effective throughput when latency increases.

## Reproduce

- From `llc_lstm_research`, export the fixed v2 candidate pool on each NoPF region with `lstm/v2_data.py`. Use `lstm/v3_oracle.py --lead 280 4000 --output OUTPUT DATASET...` to enumerate the NoPF ceiling.
- Run `lstm/v3_train.py` with `--train`, `--validation`, and the development NoPF service JSON. Use `--arms` to select head and recurrence variants. Use `--pairwise-arms` or `--unroll32-arms` to add matched controls. Use `--objective future` to isolate the label change and `--state-dropout` for training-only dropout. Use `--frozen` to score new regions after checkpoint selection.
- Pass the matching exported model, correlation table, and validation threshold to `conventional/run.py`. Use `--prefetch-events` and the pacing parameters above. The frozen checkpoints, logs, and joined metrics are under `lstm/runs/v3/`. Raw ChampSim logs are under `conventional/runs/v3_comparison/`.
