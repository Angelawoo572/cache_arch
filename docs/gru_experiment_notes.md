# GRU Experiment Story: How to Explain What We Tried, What Broke, and Why V9 Exists

This file is written as a **talking note**. The goal is that I can use it to explain the GRU experiment path to someone else, and also make sure I understand it myself.

The main message is:

```text
We are not trying to prove that GRU is the final prefetcher.
We are using GRU as the first NN family to learn what matters:
feature choice, target choice, workload choice, replay alignment, and IPC usefulness.
```

A neural prefetcher has more than one part:

```text
trace data -> feature/label design -> model prediction -> prefetch-list decoding -> ChampSim IPC
```

A notebook accuracy number only checks part of this pipeline. IPC checks the whole pipeline. A model can have good prediction accuracy and still hurt IPC if the prefetches are late, too early, wrong-page, polluting, too aggressive, or not aligned with the simulator's access index.

---

## 0. The one-sentence story

The GRU experiments started with a simple next-offset prediction task. That task was too weak. Then we did a controlled GRU feature sweep and learned that adding features like PC can improve in-trace validation but still hurt cross-trace IPC. V8 changed the target to next-delta prediction and became much more accurate, but IPC still did not improve because the output space and action policy were still not right. V9 therefore changes the formulation again: same-workload evaluation, delta-bitmap output, and PC+Delta context. Before judging V9, we must make sure the prefetch-list indices align with ChampSim replay.

---

## 1. Quick glossary: what each metric actually means

### `val_off_acc`

This is validation accuracy for predicting the next cache-line offset. In the GRU V1--V4 notebooks, validation means:

```text
train on early part of mcf
validate on later part of mcf
```

So `val_off_acc` answers:

```text
Did the model learn patterns that continue later in the same workload?
```

It does **not** prove the prefetcher is useful. It only proves the model can fit that label on that validation slice.

### `test_off_acc`

In V1--V4, test means a different trace:

```text
train on mcf
test on omnetpp
```

So `test_off_acc` answers:

```text
Does what the model learned from mcf transfer to a different binary/workload?
```

This is a much harder question. If a feature helps `val_off_acc` but not `test_off_acc`, that feature may be memorizing workload-specific behavior.

### `IPC`

IPC is the real system-level metric. It answers:

```text
Did the simulated CPU execute more instructions per cycle with this prefetch list?
```

IPC can go down even when accuracy goes up. That is not a contradiction. Accuracy checks whether the model guessed a label. IPC checks whether the issued prefetches helped the cache hierarchy at the right time without hurting bandwidth, MSHRs, or pollution.

### `top1`, `top5`

For V8/V9, `top1` means the correct future delta was the model's best guess. `top5` means the correct future delta was somewhere among the top five guesses.

High `top5` but lower `top1` means:

```text
The model often knows the right neighborhood, but the decoder still needs a good policy for which candidate(s) to issue.
```

### `OOV`

OOV means “out of vocabulary.” In V8, the model only had a fixed top-256 delta vocabulary. If the real next delta was not in that vocabulary, the model could not name the exact target.

High OOV means:

```text
The output space is too small for the long-tail memory-access distribution.
```

### `trigger rate` / `issued prefetches`

`trigger rate` in the notebook means how many prefetch-list entries were exported relative to the number of scored accesses.

`issued prefetches` in ChampSim means how many of those list entries actually matched the replayer's current access counter and were accepted by ChampSim.

These are not always the same. A prefetch list can be loaded but issue zero prefetches if the list index and the simulator index are not aligned.

---

## 2. Early model zoo: what it taught us

Relevant notebooks:

```text
notebook/neural_prefetcher_zoo.ipynb
notebook/neural_prefetcher_zoo_v2.ipynb
notebook/neural_prefetcher_zoo_v3.ipynb
```

The early model zoo compared multiple model families on the same basic task:

```text
input  = PC hash + last 4 deltas
label  = next cache-line offset inside a 4 KiB page
models = Perceptron / MLP / CNN / LSTM / Transformer
```

This was a useful first pass because it tested a simple hypothesis:

```text
If I just use a more powerful neural network, will the prefetch problem become easy?
```

The answer was basically no. The validation accuracies were close:

```text
Perceptron   ~= 0.0895
MLP          ~= 0.0962
CNN          ~= 0.0920
LSTM         ~= 0.0952
Transformer  ~= 0.0934
```

But inference latency was very different:

```text
Perceptron   ~=  51 us
MLP          ~= 100 us
CNN          ~= 206 us
LSTM         ~= 348 us
Transformer  ~= 535 us
```

### How to explain this result

The more complicated models did not clearly dominate. MLP, LSTM, and Transformer were all close in accuracy, but Transformer was much slower. So the early lesson was:

```text
The main bottleneck is probably not “GRU vs Transformer.”
The bigger problems are the target, the features, and whether the predictions become useful prefetches.
```

### What was wrong with this early setup

This setup was not enough for a serious conclusion because the label was too simple:

```text
next offset inside current page
```

That has several problems:

1. If the next access jumps to a different page, predicting the offset alone is not enough.
2. Real prefetchers often need to issue more than one candidate, but the label only asks for one next offset.
3. The notebook exported many predictions as prefetches, which can be too aggressive.
4. Some early settings were closer to smoke tests than final experiments: capped rows, short runs, and weaker target design.

So the model zoo was useful as exploration, but not enough as a controlled research result.

---

## 3. V1--V4: the controlled-variable GRU sweep

Relevant files:

```text
notebook/gru_sweep_cross_trace.ipynb
notebook/gru_sweep_v2.ipynb
results/gru_sweep_summary.csv
results/nn_demo_summary.csv
```

This is the cleanest controlled-variable part of the GRU work.

The setup was:

```text
model family = GRU
train trace  = 605.mcf_s-994B
test trace   = 620.omnetpp_s-874B
output       = page head + offset head
```

The variable we controlled was the input feature set:

```text
V1 = delta history only
V2 = V1 + PC
V3 = V2 + page hash
V4 = V3 + PC stats: miss_rate + log_freq
```

This is a real controlled-variable sweep because each version changes one main thing at a time.

---

## 4. What V1--V4 accuracy tells us

The recorded offline results were:

```text
V1 delta only:
    val_off_acc  = 0.0124
    test_off_acc = 0.0126
    inf_us       ~= 450 us

V2 + PC:
    val_off_acc  = 0.0404
    test_off_acc = 0.0127
    inf_us       ~= 498 us

V3 + page:
    val_off_acc  = 0.0346
    test_off_acc = 0.0150
    test_page_acc = 0.0572
    inf_us       ~= 571 us

V4 + PC stats:
    val_off_acc  = 0.0303
    test_off_acc = 0.0126
    test_page_acc = 0.0441
    inf_us       ~= 574 us
```

### The important comparison: V1 vs V2

When we add PC:

```text
val_off_acc improves: 0.0124 -> 0.0404
test_off_acc stays almost the same: 0.0126 -> 0.0127
```

This means:

```text
PC helped the model fit mcf's validation slice.
But PC did not help it generalize from mcf to omnetpp.
```

A simple way to say this:

```text
PC was useful as an in-workload clue, but not as a cross-workload clue.
```

This is reasonable because PC values are tied to a particular binary. A PC hash in `mcf` does not mean the same thing as a PC hash in `omnetpp`.

### What V3/V4 add

V3 adds page hash. V4 adds simple per-PC statistics. They slightly change page/offset accuracy, but they do not create a strong cross-trace result.

So the lesson is not “page is useless” or “PC stats are useless.” The more careful lesson is:

```text
In this cross-trace setup, adding more context features did not solve the real problem.
The model still did not produce robust prefetches for the held-out workload.
```

---

## 5. What V1--V4 IPC tells us

The ChampSim replay results on `620.omnetpp_s-874B` were:

```text
baseline no prefetch = 0.3157 IPC
GRU_V1 = 0.3038 IPC = 0.9623x
GRU_V2 = 0.2783 IPC = 0.8815x
GRU_V3 = 0.2781 IPC = 0.8809x
GRU_V4 = 0.2807 IPC = 0.8891x
```

The most important point is:

```text
V2 improved validation accuracy but made IPC worse.
```

That is not random noise; it is a very important prefetching lesson.

### Why accuracy can improve while IPC gets worse

The validation label asks:

```text
Did I predict the next offset correctly in this dataset?
```

The simulator asks:

```text
Did my prefetch arrive early enough, not too early, not pollute the cache, not waste bandwidth, and help future demand accesses?
```

These are different questions.

So V1--V4 taught us:

```text
A feature can help the model fit labels while making the prefetch action worse.
```

That is why later experiments must always report both ML metrics and IPC.

---

## 6. Why omnetpp was not a good demo trace

The upper-bound sweep showed that `omnetpp` barely responds to common prefetch/replacement changes:

```text
620.omnetpp_s-874B:
    LRU        = 0.2806 IPC
    LRU+stride = 0.2822 IPC
    LRU+SPP    = 0.2833 IPC
    SRRIP+SPP  = 0.2786 IPC
```

This spread is very small. So if a neural prefetcher fails on this trace, there are two possible interpretations:

```text
1. the neural prefetcher is bad
2. the trace has very little prefetch headroom in this window
```

This is why V9 moved to traces with clearer headroom:

```text
619.lbm_s-4268B:
    LRU 0.4523 -> SPP 0.5321   (~18% headroom)

602.gcc_s-734B:
    LRU 0.5564 -> SPP 1.330    (~139% headroom)
```

The point is not that `omnetpp` is unimportant. The point is that it is not a good first demo if the goal is to see whether V9 can ever improve IPC.

---

## 7. Why V8 changed so much relative to V2/V4

V8 was not a tiny controlled ablation. V8 was a formulation repair.

V1--V4 used a page/offset-style prediction target. That target had several issues:

```text
1. offset alone is not enough when accesses jump pages
2. single next-offset prediction does not match multi-line prefetching
3. cross-trace PC/page features did not transfer well
4. IPC was bad even when validation accuracy improved
```

So V8 changed the target to a delta-token formulation:

```text
input  = longer delta history + PC embedding
label  = next delta token from top-256 delta vocabulary + OOV
filter = L1D demand accesses
policy = confidence-gated prefetch list
```

This is why V8's numbers look very different. It was not just “V4 plus one feature.” It changed what the model was asked to predict.

---

## 8. V8 metrics explained in plain English

V8 recorded:

```text
best_val_top1 = 0.7458
test_top1     = 0.4791
test_top5     = 0.8507
train OOV labels = 66.7%
test OOV labels  = 61.2%
CPU inference = 734 us
trigger rate = 33.4%
```

### `best_val_top1 = 0.7458`

This means that on the validation slice, the model's best delta-token guess was correct about 74.6% of the time.

That is much higher than V1--V4 offset accuracy. So V8 proved:

```text
Delta prediction is much more learnable than the old offset target.
```

### `test_top1 = 0.4791`, `test_top5 = 0.8507`

This was cross-binary testing. The model was less accurate on the different test trace, but top-5 was still high.

A careful interpretation is:

```text
The model often places the correct delta in a small candidate set,
but choosing and timing the final prefetch is still a separate problem.
```

### OOV = 66.7% train, 61.2% test

This is the biggest warning in V8.

The model only has a top-256 delta vocabulary. If the correct next delta is outside that vocabulary, the model cannot name it exactly.

So high OOV means:

```text
The memory-delta distribution has a long tail.
A small fixed vocabulary is too restrictive.
```

This is one major reason V9 moves to a bounded local delta bitmap instead of one fixed delta token.

### CPU inference = 734 us

This is far too slow for real hardware. It means the GRU is currently an offline discovery model, not a final online prefetcher.

So the hardware lesson is:

```text
If GRU discovers useful features/targets, the final design must be smaller:
tables, counters, perceptrons, quantized models, or distilled logic.
```

### trigger rate = 33.4%

V8 emitted prefetches for about one third of scored accesses.

But gated and ungated IPC were almost the same:

```text
GRU_V8 gated   = 0.9604x
GRU_V8 ungated = 0.9610x
```

So the confidence threshold was not the real fix. V8 already had a kind of natural gate because many labels were OOV. This means:

```text
The problem is not solved by simply raising/lowering confidence.
The output/action formulation needs to change.
```

---

## 9. V8's main lesson

V8 improved ML prediction but still did not improve IPC:

```text
baseline = 0.3157 IPC
GRU_V8  = 0.3032 IPC = 0.9604x
```

The right conclusion is not “GRU is useless.”

The right conclusion is:

```text
The model can learn a better target, but the emitted prefetches are not yet system-useful.
```

This pushed the project toward V9.

---

## 10. Why V9 is allowed to change many things

Usually, we want controlled-variable changes. But sometimes a previous experiment proves the formulation is wrong enough that a redesign is justified.

V8 showed several problems at once:

```text
1. fixed delta vocabulary has high OOV
2. single next-delta output does not represent multiple useful future lines
3. confidence gating did not fix IPC
4. cross-binary evaluation mixes “can learn” with “can transfer”
5. omnetpp has very little prefetch headroom
```

Therefore V9 changes the setup to answer a cleaner question:

```text
If we evaluate within the same workload, and use a bitmap target that represents multiple nearby future deltas, can GRU produce a useful prefetch list?
```

V9 changes:

```text
A. in-distribution split
   train = first 70%
   val   = next 15%
   test  = last 15%

B. delta bitmap output
   129 possible nearby deltas: [-64, +64] cache lines
   label bit = 1 if that delta appears in the next 8 accesses

C. PC+Delta hash feature
   captures local interaction between instruction context and recent delta behavior

D. better trace choice
   mcf, lbm, gcc instead of only omnetpp
```

Important: V9 is a redesign checkpoint, not a one-variable ablation.

---

## 11. V9 implementation caveat: replay index alignment

Current V9 prefetch lists have idx ranges like:

```text
mcf first_idx ~= 7,415,896
lbm first_idx ~= 1,350,082
gcc first_idx ~= 1,386,722
```

These indices come from the original dumped CSV's simulation phase.

The `list_replayer` also counts from zero at the beginning of ChampSim's simulation phase. Therefore the replay run must use the same simulation-phase coordinate system as the dump.

If we change warmup/sim to run only the last 15%, the replayer counter restarts at zero, while the prefetch list still contains original full-window indices. The symptom is:

```text
list loaded successfully
but issued prefetches = 0 or only a small fraction
```

So V9 IPC should not be judged until we use the corrected full-index replay or generate a new local-index last-15% prefetch list.

---

## 12. Paper connections without forcing the logic

The papers are not being copied directly. They explain why each revision is reasonable.

### Hashemi et al., Learning Memory Access Patterns

Useful idea:

```text
Treat memory prediction as a classification / sequence problem, not raw 64-bit address regression.
```

Connection to our work:

```text
V8 moves from weak offset labels toward delta-token classification.
```

### Voyager

Useful idea:

```text
Memory addresses need structure such as page/offset decomposition, and evaluation is often per workload/application.
```

Connection to our work:

```text
V1--V4 tried a page/offset-style output.
V9 uses same-workload train/val/test to first answer whether the model can learn the workload.
```

### TransFetch / bitmap-style formulations

Useful idea:

```text
Future memory accesses are better represented as a set/window of possible future lines, not always one next token.
```

Connection to our work:

```text
V9 uses a delta bitmap over the next 8 accesses.
```

### Pythia

Useful idea:

```text
Prefetching should use program context and feedback/state, not only raw address history.
```

Connection to our work:

```text
V9's PC+Delta feature is a small step toward combining instruction context with recent memory behavior.
```

### DART / Net2Tab / efficient neural prefetcher line

Useful idea:

```text
A large NN may be useful for discovery, but practical hardware needs a small/fast representation.
```

Connection to our work:

```text
GRU latency is too high for direct hardware use, so a successful GRU result would later motivate distillation or a cheaper family such as perceptron/CNN/MLP.
```

### APT-GET and timeliness work

Useful idea:

```text
Correct prefetches can still be useless if they arrive too early or too late.
```

Connection to our work:

```text
If V9 has high offline F1 but low IPC after replay alignment is fixed, timeliness becomes a leading explanation.
```

### Limoncello

Useful idea:

```text
Prefetching can hurt when it creates bandwidth/resource pressure.
```

Connection to our work:

```text
If V9 emits many prefetches and IPC drops, the next fix may be throttling, degree control, or utility gating, not a bigger NN.
```

---

## 13. What to say if someone asks “are we controlled-variable?”

The honest answer is:

```text
Partly yes.
```

More precise:

```text
The V1--V4 GRU sweep is controlled-variable:
    same GRU family, same basic target, add one feature at a time.

V8 is not controlled-variable:
    it is a target redesign after V1--V4 showed the old target/setup was weak.

V9 is not a one-variable ablation either:
    it is a cleaner formulation designed after V8 exposed OOV, single-target, confidence-gating, and headroom problems.
```

This is not a weakness as long as we describe it honestly. The research process is:

```text
controlled sweep -> identify failure -> redesign formulation -> then do new controlled ablations inside the new formulation
```

After V9 replay is fixed, the next controlled ablations should be inside the V9 formulation:

```text
same trace, same trained model, vary threshold / max degree
same V9 target, compare GRU vs cheaper model families
same model, remove PC+Delta feature
same model, vary next-window size
```

---

## 14. What to do next

Do not rewrite the notebook yet unless replay still fails after the script fix.

Immediate next step:

```text
Run V9 with corrected full-index replay.
```

Then fill this table:

```text
trace | baseline IPC | SPP IPC | V9 IPC | speedup | top1 | top5 | F1 | precision | recall | issued | list lines | CPU us
```

Then decide:

```text
Case 1: lbm/gcc offline high and IPC improves
    V9 formulation is promising.
    Next: compare cheaper model families using the same V9 bitmap target.

Case 2: lbm/gcc offline high but IPC still drops
    Do not switch family immediately.
    First test threshold, max degree, timeliness, pollution, and replay details.

Case 3: offline low and IPC low
    The feature/target/model is not enough for that workload.

Case 4: mcf fails but lbm/gcc work
    That is not a full failure.
    mcf may be a bypass/replacement/utility-control workload rather than a prefetch showcase.
```

The key mindset is:

```text
Do not ask only “is GRU good?”
Ask “which part of the pipeline failed?”
```
