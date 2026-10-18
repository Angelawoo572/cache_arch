# Supplementary evidence: Old slides → recovered models → live results

- **Read first:** [project overview](../../../../RESULTS_INDEX.md) · [four-route native-versus-own-NN results](../../../../formal_study/results/602_width_templates/README.md), including IPC, accuracy, coverage, timeliness and Pareto comparisons.
- **This page:** Secondary historical evidence alignment. Keep old keyed replay, newer live inference and the H16/template supplement as distinct protocols.

- **Supplemental H16 result:** the recovered models beat native baselines in the measured live MAC0 matrix. Simple templates reproduce nearly all of that IPC gain. The recovered AMPM/SPP seed7 models passed Python/C++ checks and the 24-run live supplement. See the [current results](../../../../RESULTS_INDEX.md) for the conclusion and [paired data](../../../../formal_study/results/602_width_templates/summary.csv) for exact values.
- **This page aligns evidence; it does not combine protocols.** Old slides report offline inference followed by keyed replay. The newer study runs frozen weights on actual live callbacks. Finite-resource benefit remains unproven.

## What the old replay report actually showed

| Old keyed-replay IPC | Teacher | NN H8 | NN H16 |
| --- | ---: | ---: | ---: |
| AMPM | 0.43054 | 0.42786 | **0.43294** |
| SPP | 0.42834 | 0.42255 | **0.43654** |

- These are the original rounded log values: seed7, 25M cache warmup, evaluation 25–50M; NoPF IPC = 0.36800. Both H16 models beat their own replayed teacher. H8 loses to its teacher in both rows. [Full original counters and denominators](EVIDENCE_ALIGNMENT_APPENDIX.md#7-complete-old-primary-report-numeric-rows).

## Which checkpoints were recovered

- **AMPM H16:** v7 variable direct-delta model, 6,653 parameters. [Original checkpoint](../../602_lstm_ampm_live_inference/models/h16/original_checkpoint.pt) · [training metadata](../../602_lstm_ampm_live_inference/models/h16/source_training_metadata.json).
- **SPP H16:** empirical-prior hurdle v2, 6,609 parameters. [Original checkpoint](../../602_lstm_spp_live_inference/models/h16/original_checkpoint.pt) · [training metadata](../../602_lstm_spp_live_inference/models/h16/source_training_metadata.json).
- These are preserved originals, not retrained substitutes. The other AMPM H16 sweep and balanced SPP v1 are different models. [Identity audit](EVIDENCE_ALIGNMENT_APPENDIX.md#1-which-h16-artifacts-match-the-old-slides).

## Why old replay and new live IPC must stay separate

| Measurement | Old slides / replay | Completed live supplement |
| --- | --- | --- |
| Prediction | Precomputed lists, matched by PC/line/occurrence | Frozen model runs on actual callbacks; h/c updates online |
| Cache warmup | 25M instructions | 5M before each measured interval |
| State history | AMPM: guard only; SPP: train → guard | Fresh h/c at warmup start; keep warmup state |
| Measurement | 25–50M | Validation 20–25M; ROI0 25–50M; ROI1 75–100M |
| Controls / cost | Own replayed teacher; no NN execution cost | NoPF/native/H8/templates; uncapped or 16,144 requests/M; MAC0 |
| Quality / traffic | Legacy aggregate counters; requests are not DRAM bytes | Separate tracked L2/LLC cohorts and actual DRAM counters |

- The latest supplement has **24 new runs** (2 validation + 22 test) and **64 explicitly reused references**. It is one retained seed7, not a new multi-seed experiment. [Matrix and state contract](README.md) · [correctness gate](../../../../formal_study/results/602_width_templates/metadata/correctness_gate.json) · [all labeled rows](../../../../formal_study/results/602_width_templates/per_run.csv).

## Three corrections when reading the old slides

- **Silent → act:** Stride’s 163,567 events are **40.4649% of all callbacks**, or **48.0026% of teacher-silent callbacks**. “40.5% of silent events” uses the wrong denominator.
- **Requests ≠ DRAM:** old requested/issued counters cannot establish bandwidth reduction. Replay matching also drops generated actions: AMPM emits 726,238 of 808,438; SPP emits 1,963,940 of 2,021,095. These are transport counts, not causal IPC contributions.
- **Zero L2 coverage ≠ zero benefit:** old SPP H8 targets LLC and has positive LLC usefulness and higher IPC than NoPF. Its L2-only coverage does not measure all-level benefit.

- [Slide corrections](../../../../docs/602_width_templates/SLIDE_ERRATA.md) · [original slides](../../../../docs/602_deck_v7_15.pdf) · [detailed historical audit, logs and source paths](EVIDENCE_ALIGNMENT_APPENDIX.md).

- Raw histories, generated lists and replay logs are **local/Sacramento only**. The appendix lists their paths beneath `/Users/angelawoo/Documents/CMU/ece/cache/` or `/home/qianruw/cache/`; GitHub includes the source, portable checkpoints and curated results.
