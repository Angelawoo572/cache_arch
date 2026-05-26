# Related Work Notes: Post-Prefetch Candidate Utility Filter

## Directly related problem family

This project is closest to **prefetch filtering / prefetch throttling / prefetch-aware cache admission**, not direct neural next-address prediction.

The novelty target is:

```text
Use a strong existing prefetcher only as a candidate generator,
then learn candidate utility/timeliness/pollution from hardware feedback.
```

This means the model is not judged by whether it can invent the next address. It is judged by whether it can reject bad candidates and keep useful candidates.

## Papers already in our reading map

### Hashemi et al., Learning Memory Access Patterns

Useful background because it showed that neural models can learn memory patterns, but it framed prefetching as address/delta prediction. Its limitation is exactly why this project changes the task: direct address prediction is hard to make hardware-realistic.

### Voyager

Useful because it showed page/offset decomposition and stronger neural prediction, but it also supports the conclusion that full neural prefetching is too heavy for the online cache path. For this project, Voyager is not the implementation model; it is evidence that address generation should be separated from small online decisions.

### Twilight / candidate-ranking formulation

Most conceptually aligned. It reframes neural prefetching away from raw address prediction and toward selecting among candidates plus a no-prefetch option. Our project goes one step more hardware-practical: candidates come from an existing prefetcher, and the learned part only performs post-candidate utility filtering.

### APT-GET / timeliness work

Important because a correct prefetch can still be useless if it arrives too late or too early. The filter should therefore not learn only binary accuracy. It must eventually learn whether the candidate becomes a timely hit.

### Limoncello

Important because it shows a prefetcher can become harmful under bandwidth pressure. The filter should include hardware pressure features such as bandwidth occupancy, MSHR occupancy, prefetch queue occupancy, and cache-set pressure.

### RL-CoPref / Pythia-style RL prefetching

Important because RL matches the structure of the problem: prefetch usefulness is delayed, and the action changes shared resources. However, this project should not begin by training a large RL system. First build a tiny supervised/online utility filter, then upgrade the action policy to RL.

## Open-source / practical baselines to investigate

### ChampSim built-ins

Use whatever the local ChampSim checkout already supports first. Typical built-in candidates to check:

```text
no
next_line
ip_stride
spp_dev
```

For this project, `spp_dev` is a good first candidate generator if available because it is stronger than next-line/stride and is already designed around candidate confidence.

### Berti

Berti is a strong open-source DPC3/DPC-style prefetcher and is a good candidate for the stronger baseline if it can be integrated into the local ChampSim version cleanly.

### Pythia

Pythia is very relevant because it uses online reinforcement learning for customizable hardware prefetching. It is a related work baseline and a warning: if our filter uses RL, the novelty must be very clear. Our distinction should be:

```text
Pythia learns prefetch action/policy as the prefetcher.
Our project learns post-candidate admission/filtering after an existing prefetcher.
```

## What would count as novelty

A weak novelty claim:

```text
I used NN/RL for cache prefetching.
```

A stronger novelty claim:

```text
I decouple address generation from admission control.
A conventional or prior prefetcher proposes candidates.
A tiny hardware-feedback learner estimates candidate utility, timeliness, and pollution risk before cache admission.
This improves IPC or reduces traffic over the same prefetcher alone.
```

## First experiment question

The first experiment should answer:

```text
Can a filter improve an already-working prefetcher by issuing fewer, better prefetches?
```

Expected first success case:

```text
IPC roughly same or slightly higher,
prefetches issued much lower,
accuracy higher,
bandwidth lower,
MPKI not worse.
```

That is already meaningful because it shows resource-aware filtering.
