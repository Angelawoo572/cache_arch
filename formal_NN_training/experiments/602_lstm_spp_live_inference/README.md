# Frozen SPP: checkpoint to live inference

- **Current H8 IPC vs native (ROI0 / ROI1):** uncapped **-1.2093% / -3.5020%**; cap 16,144 requests/M **-1.5901% / -2.2422%**.
- **Current H16 IPC vs native (ROI0 / ROI1):** uncapped **+2.0634% / +4.1512%**; cap 16,144 requests/M **+1.4146% / +1.4515%**. H8 and H16 L2/LLC accuracy, coverage proxy and timeliness are in the [combined results](../../../formal_study/results/602_width_templates/README.md) and [per-arm metrics](../../../formal_study/results/602_width_templates/reader_metrics.csv).
- **H8 capped LLC-intent quality (native → NN; ROI0 / ROI1):** the NN shifts almost entirely to LLC-intent requests. LLC accuracy is 26.06% → 47.10% / 28.57% → 48.96%; useful/NoPF LLC-miss ratio is 33.43% → 107.17% / 19.67% → 59.17%; timeliness is 99.4603% → 99.7333% / 98.5491% → 99.2615%. L2 NN cohorts contain 0 / 14 enqueues; do not combine levels or treat the useful/miss ratio as misses eliminated.
- **H8 capped traffic:** both arms have the same measured allowed requests/M; NN actual DRAM bytes change by +0.5000% / +2.4109% versus native. Requests and DRAM are separate measurements.
- **Scope:** 602, retained seed7, frozen H8/H16 weights, global chronological h/c, synchronous MAC0. These comparisons do not establish finite-resource benefit or cross-seed/workload robustness. Against H8, native SPP has higher IPC and lower DRAM in both capped ROIs; H16 is reported separately above.
- **Read the measurements:** [native-versus-NN results](../../../formal_study/results/602_width_templates/README.md) · [per-arm metrics](../../../formal_study/results/602_width_templates/reader_metrics.csv) · [paired comparisons](../../../formal_study/results/602_width_templates/reader_pairs.csv) · [own latest subset](results/width_templates_20261017/README.md) · [H8 archive](results/formal_20261017/README.md).

- **Run/recovery:** [four-route guide](../602_four_prefetcher_formal/README.md), [H16/template guide](../602_four_prefetcher_formal/width_templates/README.md).

## Retained models

| Item | H8 | H16 |
| --- | --- | --- |
| Original checkpoint | [model.pt copy](models/original_checkpoint.pt), 16,790 file bytes | [model.pt copy](models/h16/original_checkpoint.pt), 31,638 file bytes |
| Source metadata | [metadata](models/source_training_metadata.json) | [metadata](models/h16/source_training_metadata.json) |
| Portable export | [model.bin](models/model.bin), 12,024 file bytes | [model.bin](models/h16/model.bin) |
| FP32 parameters / weight bytes | 2,865 / 11,460 B | 6,609 / 26,436 B |
| Global FP32 h/c payload | 64 B | 128 B |

- Source, relative to cache: `formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/colab_output/independent_delta_spp_lstm_{h8,h16}/model.pt`; original metadata is adjacent.
- Required identity: `compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2`, LSTM, 65 features, 16 tensors, four mixture components and fill levels `[2,4]`. The same-sized balanced-v 1 checkpoint is a different model.
- The gate was trained with empirical-prior **unweighted** categorical NLL. H16 is not a fallback to H8; each width has independent exports, checks and bindings.

## Exact input, state and actions

- Input: 64 LSB-first bits of the aligned **byte address** (`line << 6`) plus demand/fill kind. No PC, hit/miss, cycle, teacher action, private table or future event enters the feature vector.
- Demand uses the eligible L2 LOAD address. Fill uses the **evicted address** of every raw CACHE L2-fill callback, including demand/writeback fills. Both advance one chronological global h/c pair; a fill emits no action.
- Demand gate: two-class argmax, ties silent. Positive count: `max(1, rint(exp(log_count)))`, using nearest/ties-even rounding.
- Decoder starts directly from LSTM context. Each slot selects the first-argmax mixture logit and its mean; signed `expm1` and ties-even rounding give a delta relative to the original base line, with `2^57` magnitude guard and modulo-`2^58` wrapping.
- Learned fill is `FILL_L2=2` or `FILL_LLC=4`. The GRU feeds back its own unrounded coordinate and chosen fill one-hot after every slot, including the final one. Invalid domains are rejected, not silently clipped.
- [Shared scalar backend](../../common/live602_runtime/): complete synchronous MAC0 forwards, h/c published on completion; no imposed input FIFO, admission loss, output queue, per-PC capacity, finite service latency or decoder 32 truncation. Stride retains a separate engine and limits.

## Native SPP and measurement boundaries

- **Effective inputs differ:** the student/training logger observes all raw L2 fills; the original native registry sends FILTER feedback only when `prefetch==1`. A native/student IPC difference cannot be attributed solely to payload under an identical-input claim.
- Native internal same-page behavior and confidence-based L2/LLC destinations remain intact. The new protocol uses request threshold 40 and L2-fill threshold 90, matching teacher labels. The first formal study used request threshold 25 and is a different comparator protocol.
- Common external `cross_page` or request gates do not remove native internal rules. The student retains its learned fill destinations and complete action list.
- Teacher targets are canonicalized per target with minimum numeric fill (L2 priority); native generated counts are raw source calls. Their K distributions have different scopes.
- L2 and LLC lifecycle cohorts follow **first source-enqueue intent**, not final physical fill or total all-level usefulness after upgrades. Report separately; unavailable complete tracking means NA. Requests are not DRAM bytes, and evicted-unused is not pollution.
- Four 602 `teacher_category=3` means unclassified: no global shadow teacher was run. It is not teacher overlap or teacher-silent causality. See [measurement scope](../602_four_prefetcher_formal/MEASUREMENT_SCOPE.md) and [method limits](../602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).

## Training and live state history

- Weight training 0–20M: 489,880 events =325,389 demands + 164,491 fills; 8 epochs, chunk 1,024, accumulation 16.
- Old offline inference: fresh state before train 0–20M → guard 20–25M (120,993 events) → evaluation 25–50M (607,877 events). That state is not loaded into new live runs.
- New validation 20–25M follows 15–20M cache/state warmup. Tests 25–50M (previously inspected replication) and 75–100M (future ROI) each have their own 5M warmup.
- Live h/c starts at warmup start, advances with output blocked, and survives the ROI boundary. ROI callback/update counters restart; training amount and live observation history are distinct.

## Export, parity and recovery

- [export_and_validate.py](python/export_and_validate.py): `--checkpoint`, `--model`, optional `--probe`, `--stream`, `--report`, `--events`. It verifies the known model revision, full finite FP32 tensor inventory and LV602R1 schema for H8/H16.
- Parity uses the real PyTorch class, continuous demand/fill inputs, every h/c/count/ordered address/fill action, reset and uint 64 boundaries. Actions must match exactly; h/c maximum error must stay below `2e-5`.
- Formal certification builds a probe from the **actual compiled simulator runtime** with `-O3 -ffp-contract=off`, re-exports only into a fresh certificate directory, and byte-compares that candidate with the formal model. Accepted evidence is build-bound; an old standalone PASS is not a substitute.
- Use the existing Sacramento Torch environment. Keep sources, models, fixtures and temporary files inside cache; preserve conflicting files and restore missing files from retained originals.
- Follow the [formal stage commands](../602_four_prefetcher_formal/README.md) after their correctness gates. Parity is a correctness fixture, not a simulator pilot or full-ROI IPC result.
