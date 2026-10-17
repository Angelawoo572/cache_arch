# Corrections to retained v7_15 slides

- The original deck and logs remain unchanged references. These corrections describe old keyed replay; they do not relabel it as new live inference.
- [Concise alignment](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) links the full historical audit and completed live evidence. Use the [project overview](../../RESULTS_INDEX.md) for current conclusions.

## Event denominators

- Stride H16 compact_hurdle_v9 acts on **163,567 teacher-silent callbacks**: **40.4649% of all 404,219 callbacks**, or **48.0026% of 340,746 teacher-silent callbacks**.
- “40.5% of silent events” mixes those denominators.
- Teacher-active → student-silent is **6,259 / 404,219 = 1.5484%** joint mass, or **6,259 / 63,473 = 9.8609%** conditional rate.
- These are generated-stream event counts, not an IPC causal ablation.

## Requests and DRAM

- Requested/issued and requests/load count cache prefetch calls, not LLC↔DRAM bytes.
- Call a change “prefetch request reduction” unless controller transaction/byte counters establish DRAM reduction.
- New live results report requests and actual DRAM bus activity separately; neither their ratio nor requests × 64 supplies measured DRAM.

## SPP H8 coverage

- Zero L2 useful/coverage describes L2, not all-level benefit.
- Old H8 replay emitted **1,897,395 LLC-target requests**; LLC filled **216,298** and useful **198,675**. IPC was **0.42255 versus NoPF 0.368** under that old protocol.
- These aggregate counters are not the later fully tracked, separate L2/LLC first-intent cohort metrics.

## Model and protocol identity

- Old H16 points are **AMPM v7 seed7, 6,653 parameters**, and **SPP empirical-prior v2 seed7, 6,609**. The smaller AMPM sweep and balanced SPP v1 are different models.
- The old size sweep is one seed, not independent-seed robustness.
- Replay used **25M cache warmup**, versus **5M** in current live runs. Old primary teachers were replayed lists; retain their keyed-match losses and different state initialization.
