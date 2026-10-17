# Corrections to the retained v7_15 slide claims

The original deck and offline logs remain unchanged reference material. These
replacement statements apply to its old replay results; they do not relabel
those results as new live measurements. Full evidence and paths are in
[EVIDENCE_ALIGNMENT](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md).

1. **Silent → act:** Stride H16 compact_hurdle_v9 produced student actions on
   163,567 teacher-silent callbacks. This is **40.4649% of all 404,219 callbacks**,
   or **48.0026% of 340,746 teacher-silent callbacks**. The slide's “40.5% of silent
   events” mixes these denominators. Teacher-active → student-silent is 6,259 /
   404,219 =1.5484% joint mass, or 6,259 /63,473 =9.8609% conditional rate.
   These are generated-stream event counts, not an IPC causal ablation.
2. **Requests and traffic:** the old requested/issued or requests/load columns
   measure cache prefetch calls. They do not measure LLC↔DRAM bytes. Replace
   “DRAM traffic reduction” with “prefetch request reduction” unless actual
   controller transaction/byte counters support the former. The new live study
   reports actual DRAM bus activity and requests separately; their ratio is
   neither assumed constant nor inferred using requests×64.
3. **SPP H8 coverage:** zero L2 useful/coverage refers to L2. The old H8 replay
   emitted 1,897,395 LLC-target requests, with LLC filled216,298/useful198,675;
   its IPC was0.42255 versus NoPF0.368 under the old protocol. It therefore
   does not imply zero all-level benefit. Old aggregate useful counters are
   distinct from new fully tracked L2/LLC first-intent cohort metrics.

The old H16 points are AMPM v7 seed7 (6,653 parameters) and SPP empirical-prior
v2 seed7 (6,609), not the smaller AMPM sweep or balanced SPP v1. The old model
size sweep is one seed. Old replay used25M cache warmup; the new live protocol
uses5M. Old primary teacher comparators are replayed teacher lists. Retain those
labels, their keyed-match losses, and their state-initialization differences.
