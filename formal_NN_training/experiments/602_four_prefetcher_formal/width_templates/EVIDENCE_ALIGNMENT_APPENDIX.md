# Appendix: detailed historical replay audit

- This retains the initial 2026-10-17 checkpoint/replay audit and all its numerical tables. Statements about work not performed refer to that audit, not the current project status. The recovered H16 models were subsequently parity-checked and evaluated in the completed live study. Read the [concise alignment](EVIDENCE_ALIGNMENT.md) or [current results](../../../../RESULTS_INDEX.md) first.

- Paths marked **local/Sacramento only** are relative to either `/Users/angelawoo/Documents/CMU/ece/cache/` or `/home/qianruw/cache/`. They are raw evidence, not missing GitHub files. Source and portable-checkpoint links are repository-relative.

## 1. Which H16 artifacts match the old slides

- The local deck is [602_deck_v7_15.pdf](../../../../docs/602_deck_v7_15.pdf). The already extracted slide text — `formal_study/archive/20261016/research_audit/literature/local_slides_extracted.txt` (local/Sacramento only), page sections72–74 / logical slide14, reports AMPM H16=6,653 parameters and SPP H16=6,609 parameters, with SPP IPC0.43654. Those values match the following actual artifacts and replay tables.

| Route | Actual checkpoint | File bytes | FP32 parameter payload | Parameters |
|---|---|---:|---:|---:|
| AMPM H16 seed7 | [v7 h16/model.pt](../../602_lstm_ampm_live_inference/models/h16/original_checkpoint.pt) | 31,377 | 26,612 B | 6,653 |
| SPP H16 seed7 | [empirical-prior v2 H16/model.pt](../../602_lstm_spp_live_inference/models/h16/original_checkpoint.pt) | 31,638 | 26,436 B | 6,609 |

- Their original [AMPM metadata](../../602_lstm_ampm_live_inference/models/h16/source_training_metadata.json) and [SPP metadata](../../602_lstm_spp_live_inference/models/h16/source_training_metadata.json) explicitly record seed7. The archive CRCs are complete, physical floating-point storage totals equal parameter_count×4, and all stored FP32 values were finite. This was safe ZIP/float inspection; local Torch deserialization and new H16 Python/C++ parity were not performed by this audit.

- One H16 global h/c pair is128 B. Weights+h/c are therefore26,740 B for AMPM and26,564 B for SPP, excluding decoder scratch, outputs, control, allocator overhead, queues, and all hardware implementation costs. File size is not deployed weight SRAM or host RSS.

- **Do not select a model by width alone.** Two other existing H16 artifacts have different identities:

- SPP compact_hurdle_free_running_v1_seed7 / independent_delta_spp_lstm_h16 is31,574 B, also6,609 parameters, but uses a data-derived balanced hurdle loss. It generated710,774 actions, not2,021,095. The slide point above is empirical_prior_hurdle_free_running_v2, whose gate objective is unweighted empirical-prior categorical NLL.
- AMPM 602_offline_lstm_ampm_sweep_seed7 / h16 is26,097 B and5,713 parameters, not the v7 variable direct-delta6,653-parameter point. The old size sweep is not independent training seeds.
- Stride's slide silent/act example is compact_hurdle_v9 H16, not the v7 Poisson H16 that generated zero actions. Its denominator audit is in section4.

## 2. Original training, state, input, and decoder contracts

- Both selected H16 models used source teacher-action supervision, frozen inference weights, and eight recorded epochs. Their AMPM training history — `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/colab_output/h16/training_history.csv` (local/Sacramento only) and SPP training history — `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/colab_output/independent_delta_spp_lstm_h16/training_history.csv` (local/Sacramento only) each contain epochs1..8.

- The existing collection drivers fix training0..20M instructions, guard20..25M, and evaluation25..50M: [AMPM driver](../../602_offline_lstm_ampm/linux/run_server.sh), [SPP driver](../../602_offline_lstm_spp/linux/run_server.sh). Old full-system replay used25M cache warmup followed by25M measured instructions. These coordinates differ from the later5M-warmup live study; old IPC cannot be transferred to that protocol.

| Property | AMPM H16 v7 | SPP H16 empirical-prior v2 |
|---|---|---|
| Effective NN input | 64 LSB-first bits of aligned byte address, reconstructed as line×64 | 64 callback-address bits plus one DEMAND/FILL-kind bit |
| PC | Replay transport key only; not NN input/state key | Replay transport key only; not NN input/state key |
| Recurrent state | One global chronological h/c | One global chronological h/c |
| Training rows | 325,389 eligible demand callbacks | 489,880 context events:325,389 demand +164,491 raw fills |
| Guard | 80,725 demand callbacks | 120,993 context events:80,725 demand +40,268 fills |
| Evaluation | 404,219 demand callbacks | 607,877 context events:404,219 demand +203,658 fills |
| State training | Chronological stateful TBPTT,1024-event chunks, values carried and graph detached; reset at epoch start | Chronological stateful TBPTT,1024-event chunks, values carried and detached; reset at epoch start |
| Optimizer accumulation | Every32 chunks; final epoch10 recorded optimizer steps | Every16 chunks;30 recorded steps/epoch |
| Inference initialization in old report | Fresh state advanced through guard20..25M, then continuous evaluation | Fresh state advanced through complete train→guard→eval chronology |
| Count decision | floor(softplus(count raw)), Poisson mode, no hurdle head | Two-class argmax; positive class uses max(1,rint(exp(log-count))) |
| Decoder | tanh affine initial state; free-running GRU; four-component mixture-logit argmax chooses mean coordinate | Starts from LSTM context directly; free-running GRU with coordinate + predicted-fill one-hot; four-component mixture and learned2-class fill |
| Address output | Signed rounded expm1 coordinate, cache-line delta relative to original trigger base, modulo58-bit line domain | Same signed delta/base semantics; learned FILL_L2=2 / FILL_LLC=4 |
| Fill semantics | No learned fill head; replayer interface fills L2 | Explicit fill column preserved by fill-aware replayer |
| NN constraints | No neural degree cap, teacher candidates/private tables, or imposed source page rule | Same; no PC, private SPP tables/filter/GHR, threshold, or source candidate bank as NN input |

- Source definitions are [direct_action_lstm.py](../../../common/direct_action_lstm.py), [threshold_free_policy.py](../../../common/threshold_free_policy.py), and [SPP trainer/decoder](../../602_offline_lstm_spp/python/train_and_offline_infer.py). Teacher outputs enter training targets and post-inference audits; they do not enter NN inference. Free-running decoder feedback uses its own predicted coordinate/fill, including the final action update.

- SPP supervision canonicalizes repeated targets using minimum requested fill, so L2 takes priority over LLC, matching source-PQ merge effects. The original teacher configuration records prefetch threshold40 and L2 fill threshold90. Native raw source-call count, canonical supervised action count, replay emitted count, cache acceptance, and DRAM transactions are different quantities.

- SPP's student input includes every logged raw CACHE L2-fill callback with evicted address; native SPP's existing FILTER forwarding is conditional on prefetch fill. The raw cache interface may be shared without identical effective state-update streams. Do not silently remove fill events, force all fills to L2, add PC features, or replace the checkpoint with Stride to obtain a live implementation.

## 3. Generated → loaded → matched → emitted: direct local evidence

- The generated replay CSVs were independently counted in this audit, not inferred from the final request counter.

| H16 route | Generated CSV actions | Generated trigger groups / loaded keys | Loader-reported actions | Runtime ROI callbacks | Matched keys | Replayer emitted / cache-requested | Matched-key fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMPM v7 | 808,438 | 404,219 | 808,438 | 404,219 | 363,119 | 726,238 | 89.8322% |
| SPP empirical-prior v2 | 2,021,095 | 404,219 | 2,021,095 | 404,219 | 392,788 | 1,963,940 | 97.1721% |

- AMPM generated list — `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/colab_output/h16/offline_lstm.replay.csv` (local/Sacramento only) contains two actions per generated trigger. Its actual replay log — `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/logs/602.gcc_s-734B.offline_lstm_h16.log` (local/Sacramento only) says:

    loaded 808438 prefetch entries across 404219 PC-line-occ triggers
    emitted 726238 candidates over 404219 runtime ROI L2 LOAD accesses
    363119 matched PC-line-occ triggers; 404219 loaded trigger keys

- SPP generated list — `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/colab_output/independent_delta_spp_lstm_h16/offline_nn.replay.csv` (local/Sacramento only) has2,021,095 rows with404,219 L2 and1,616,876 LLC actions. Its actual replay log — `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/logs/602.gcc_s-734B.offline_independent_delta_spp_lstm_h16.log` (local/Sacramento only) says:

    loaded 2021095 direct actions across 404219 PC-line-occ triggers
    emitted 1963940 actions over 404219 runtime ROI L2 LOAD accesses
    392788 matched PC-line-occ triggers; 404219 loaded trigger keys
    fill_l2=392788; fill_llc=1571152

- The fill-aware [replayer source](../../602_offline_lstm_spp/cpp/list_replayer_fill.cc) accepts only eligible LOADs after warmup, advances a per-(PC,line) occurrence counter, looks up (PC,line,occ), and emits that stored list on a match. It does not run the LSTM or update model h/c. An unmatched runtime key emits nothing. Equal total callback counts do not mean equal complete callback streams or100% list matching.

- Thus generated NN actions and loaded list entries agree, but neither equals runtime requests. AMPM loses82,200 generated actions to unmatched keys; SPP loses57,155. The H16 SPP fill head is preserved rather than changed by transport. This is offline inference + keyed full-system replay, not independent live inference and not finite-cost inference.

- Old AMPM comparison table — `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/matched_comparison.csv` (local/Sacramento only) reports offline AMPM IPC0.43054 and NN H16 IPC0.43294. Old SPP comparison table — `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/matched_comparison.csv` (local/Sacramento only) reports offline SPP0.42834 and H16 NN0.43654. These are existing rounded log IPC values; use actual instructions/cycles for a new analysis.

- The report's primary comparisons are the offline teacher replayer versus offline NN replayer. Existing live-reference prefetch counters include warmup and are context-only: AMPM explicitly marks simulator_prefetch_counter_scope=warmup_plus_measurement and primary_metric_eligible=False. SPP's live useful/noPF-miss ratio can exceed1 for the same scope reason. Do not treat an old live-reference lifecycle count as a same-window denominator.

## 4. Silent→act denominator correction

- The slide's Stride H16 point is compact_hurdle_v9 metadata — `formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_compact_hurdle_v9_seed7/colab_output/h16/run_metadata.json` (local/Sacramento only). Independent set comparisons used its teacher list — `formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_compact_hurdle_v9_seed7/colab_output/h16/offline_stride.replay.csv` (local/Sacramento only) and student list — `formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_compact_hurdle_v9_seed7/colab_output/h16/offline_lstm.replay.csv` (local/Sacramento only), identifying triggers by the actual (pc,line,occ) CSV columns.

| Original generated-event quantity | Count |
|---|---:|
| All evaluation demand callbacks | 404,219 |
| Teacher active / silent | 63,473 / 340,746 |
| Student active / silent | 220,781 / 183,438 |
| Teacher-silent→student-act | 163,567 |
| Teacher-act→student-silent | 6,259 |

- 163,567 /404,219 = **40.4649% of all callbacks**.
- 163,567 /340,746 = **48.0026% conditional on teacher silence**.
- 6,259 /404,219 = **1.5484% of all callbacks**.
- 6,259 /63,473 = **9.8609% conditional on teacher activity**.

- Therefore the slide wording “acts on40.5% of silent events” uses the wrong denominator;40.5% is joint event mass, while the corresponding conditional rate is48.0%. Its approximate1.6% reverse arrow likewise uses all callbacks. These counts concern the original generated/offline event stream, not the subset of keys matched in a later replay and not an IPC causal ablation.

- For selected SPP H16, metadata separately records teacher zero11,144, predicted positive404,219, and false-positive trigger11,144. The same distinction gives2.7569% mass over all callbacks versus100% conditional on teacher silence. Do not confuse false-positive *trigger* counts with false-positive *address/action* counts.

- Later Four602 native/student/template classification teacher_category=3 means unavailable/unclassified, because no global shadow teacher was advanced there. It cannot reproduce these old conditional statistics or establish overlap/teacher-silent attribution. Stride's separate shadow labels have their own explicitly defined scope.

## 5. Request rate is not DRAM traffic; SPP H8 zero L2 coverage is narrow

- The old requested/issued values count cache prefetch API requests, including cache hits and PQ merges. The old tables use:

    selected_accuracy = L2_prefetch_useful / (L2_prefetch_issued - L2_PQ_merged)
    coverage = L2_prefetch_useful / NoPF_L2_LOAD_misses

- They are not modern first-source-intent lifecycle-cohort metrics. SPP's denominator includes requested L2 and LLC destinations while its numerator is the L2 useful counter. Do not compare this old selected_accuracy directly against the later separately tracked L2/LLC cohort accuracies. “Unused” or1−accuracy is not measured pollution.

- Neither requested×64 nor issued×64 is actual LLC↔DRAM bytes. Most requests can hit in L2/LLC or merge before memory. The original logs have controller row-buffer statistics, but this audit has not established an equivalent completed-bus-byte lifecycle with initial/pending boundary work. Old “half traffic” language supported only by requests/load should be relabeled request reduction, not measured DRAM-bandwidth reduction.

- For the empirical-prior **SPP H8** actually used in the old slide, [metadata](../../602_lstm_spp_live_inference/models/source_training_metadata.json) and replay log — `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/logs/602.gcc_s-734B.offline_independent_delta_spp_lstm_h8.log` (local/Sacramento only) show:

| Counter | Existing value |
|---|---:|
| Generated actions, requested LLC destination | 2,021,095; all LLC, zero L2 |
| Runtime replay emitted/cache-requested | 1,897,395 |
| L2 prefetch filled / useful / late | 0 /0 /0 |
| L2 LOAD misses | 203,123 |
| NoPF L2 LOAD misses | 203,131 |
| LLC prefetch filled / useful / late | 216,298 /198,675 /38 |
| Existing IPC, NN /NoPF | 0.42255 /0.368 |

- Thus old “L2 coverage0” correctly says its L2 useful numerator is zero; it does not say no lines reached LLC, no demand benefited at LLC, no requests were made, or IPC equals NoPF. The positive LLC-usefulness counters and higher IPC make an all-level “useless” interpretation false. These old aggregate LLC counters still do not substitute for the later complete target-level lifecycle accounting or actual DRAM-byte counters.

- For old SPP H16, requested fill actions include one L2 and four LLC slots per generated event; the log reports L2 useful197,057 and L2 misses6,071. Its97.0098% old L2 coverage and IPC0.43654 align with the slide, but retain their replay and measurement-window definitions.

## 6. Limits of the initial audit (later work is linked separately)

- Both required H16 checkpoint files, original metadata, training histories, generated lists, replay logs, and comparison tables are present locally. Retraining is not needed to recover them.
- The initial audit executed no new H16 parity, live ROI, finite-cost ROI, extra seed, or frozen-template experiment. Parity and the 24-run live/template supplement are now complete: [correctness record](../../../../formal_study/results/602_width_templates/metadata/correctness_gate.json), [live results](../../../../formal_study/results/602_width_templates/README.md). Global-model finite-cost IPC, extra seeds and generalization remain open.
- No latest cloud slide was fetched in this audit. Alignment is to the existing local v7_15 deck and its archived extracted text; a newer unseen deck is not asserted.
- Original training0..20M is not live state-history length. AMPM's guard-only initialization and SPP's train+guard initialization differ and must be recorded before using these weights in a new common protocol.
- Seed7 width variants are one seed, not seed robustness. Balanced SPP v1 and unweighted v2 share the6609-parameter architecture but must not be treated as the same checkpoint or an automatically proven matched retraining experiment.
- No claim of innovation, firstness, universal model benefit, or causal contribution is supported by recovery of these artifacts alone. Preserve the AMPM H8 loss and other negative replay/live results.

## 7. Complete old primary-report numeric rows

- All rows below have25,000,003 retired instructions and404,219 eligible L2 LOAD callbacks after25M cache warmup. These are the old keyed-replay protocol rows only. No H16 live or finite-cost result is inferred. IPC(log) preserves the five-decimal source value; IPC(ins/cycles) independently uses the recorded denominator.

| Old row | Cycles | IPC(log) | IPC(ins/cycles) | L2 LOAD misses | Miss rate % | Net miss reduction % | Old L2 useful/M0 coverage % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common NoPF | 67935438 | 0.368 | 0.367996493965 | 203131 | 50.252710 | 0.000000 | 0.000000 |
| AMPM offline teacher | 58067029 | 0.43054 | 0.430536974778 | 17051 | 4.218258 | 91.605909 | 91.606402 |
| AMPM NN H8 | 58429699 | 0.42786 | 0.427864654925 | 23827 | 5.894577 | 88.270131 | 88.275054 |
| AMPM NN H16 | 57744903 | 0.43294 | 0.432938695905 | 12940 | 3.201235 | 93.629727 | 93.644003 |
| SPP offline teacher | 58364896 | 0.42834 | 0.428339716394 | 70874 | 17.533565 | 65.109215 | 65.106754 |
| SPP NN H8 | 59164175 | 0.42255 | 0.422553056812 | 203123 | 50.250730 | 0.003938 | 0.000000 |
| SPP NN H16 | 57267981 | 0.43654 | 0.436544165928 | 6071 | 1.501909 | 97.011288 | 97.009811 |

| Old row | Requested = issued | PQ merges | L2 PF fills | L2 PF useful | L2 PF late | L2 PF useless | Old selected accuracy % | Old timeliness % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common NoPF | 0 | 0 | 0 | 0 | 0 | 0 | NA† | NA† |
| AMPM offline teacher | 1271977 | 809 | 192408 | 186081 | 594 | 6216 | 14.638584 | 99.681800 |
| AMPM NN H8 | 392521 | 105 | 193165 | 179314 | 5 | 13708 | 45.694875 | 99.997212 |
| AMPM NN H16 | 726238 | 758 | 206989 | 190220 | 7 | 16580 | 26.219882 | 99.996320 |
| SPP offline teacher | 2254034 | 1418 | 132785 | 132252 | 384 | 524 | 5.871041 | 99.710486 |
| SPP NN H8 | 1897395 | 3962 | 0 | 0 | 0 | 0 | 0.000000 | NA† |
| SPP NN H16 | 1963940 | 4640 | 201126 | 197057 | 729 | 4017 | 10.057521 | 99.631420 |

- Every selected offline row has dropped=0 and issued=requested. †The original CSV stores0 for zero denominators; NA here makes the undefined ratio explicit. The old useful/late counters are aggregate simulator counters, not asserted LOAD-only request cohorts. Selected accuracy includes all requested fill destinations in its denominator and must retain section5's scope warning.

- Recorded training budgets also differ despite eight epochs: AMPM H16 traversed2,544 chronological chunks and80 optimizer steps; SPP H16 traversed3,832 chunks and240 optimizer steps. SPP additionally has fill-context events. These are actual sums from the two training_history.csv files, not a claim of matched training-compute budgets. AMPM's learning-rate value is not recorded in the inspected run_metadata.json; the current source default is not proof of an actual historical override setting.

- Protocol conclusion: the older evidence supports an H16 replay advantage versus each route's own offline teacher under that old protocol. The later completed live study tests those same frozen H16 weights under its own warmup/state and request-ceiling contract; see [current results](../../../../RESULTS_INDEX.md). Neither the old IPC values nor the old accuracy/coverage denominators can be substituted into the live tables.
