# GRU Experiment Notes

This file explains the GRU-related experiments in the current cache-prefetching project. The goal is not to claim that GRU is the final hardware design. The goal is to understand what the experiments teach us about:

```text
feature choice -> target choice -> model prediction -> prefetch-list decode policy -> ChampSim IPC
```

The main lesson so far is:

```text
Better ML prediction accuracy is necessary but not sufficient.
A prefetch is only useful if it is issued at the right time, with the right degree, and without creating too much cache pollution or memory-system pressure.
```

---

## 1. Early model zoo: what it was and why it was only a smoke test

Relevant notebooks:

```text
notebook/neural_prefetcher_zoo.ipynb
notebook/neural_prefetcher_zoo_v2.ipynb
notebook/neural_prefetcher_zoo_v3.ipynb
```

The early model-zoo notebooks compared multiple model families under the same simple formulation:

```text
input  = PC hash + last 4 delta tokens
label  = next cache-line offset inside the current 4 KiB page
models = Perceptron / MLP / CNN / LSTM / Transformer
```

The useful question was:

```text
If I only change the NN family, does a more complex model automatically solve prefetching?
```

The answer was no. The validation accuracies were close:

```text
Perceptron   ~= 0.0895
MLP          ~= 0.0962
CNN          ~= 0.0920
LSTM         ~= 0.0952
Transformer  ~= 0.0934
```

But inference cost was very different:

```text
Perceptron   ~=  51 us
MLP          ~= 100 us
CNN          ~= 206 us
LSTM         ~= 348 us
Transformer  ~= 535 us
```

### What this means

The early result says the bottleneck is not simply “use a bigger neural network.” A larger model family does not automatically produce a better prefetcher. The target, feature representation, and decode policy are likely more important than raw model family at this stage.

### Why it was a smoke test, not a final experiment

The early formulation had several limitations:

```text
1. The label was only the next offset inside the current page.
2. It did not fully handle page changes.
3. It asked for one next target, but real prefetchers often need multiple future candidates.
4. It did not explicitly model no-prefetch / suppress decisions.
5. Some runs were designed to make the pipeline work, not to be final research-grade evaluation.
```

So the model zoo was useful for pipeline bring-up and latency comparison, but it was not the final evidence for or against neural prefetching.

---

## 2. `gru_sweep_cross_trace.ipynb` / `gru_sweep_v2.ipynb`: controlled feature sweep

Relevant files:

```text
notebook/gru_sweep_cross_trace.ipynb
notebook/gru_sweep_v2.ipynb
results/gru_sweep_summary.csv
results/nn_demo_summary.csv
```

This is the clearest controlled-variable GRU experiment so far.

The fixed setup was:

```text
model family = GRU
train trace  = 605.mcf_s-994B
val trace    = later slice of 605.mcf_s-994B
test trace   = 620.omnetpp_s-874B
output       = page head + offset head
```

The controlled variable was the input feature set:

```text
V1 = delta history only
V2 = V1 + PC
V3 = V2 + page hash
V4 = V3 + PC stats: miss_rate + log_freq
```

This is a controlled experiment because the model family and basic output formulation stayed the same, while one major feature group was added at a time.

---

## 3. What the V1--V4 accuracy metrics mean

Recorded offline results:

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
    val_off_acc   = 0.0346
    test_off_acc  = 0.0150
    test_page_acc = 0.0572
    inf_us        ~= 571 us

V4 + PC stats:
    val_off_acc   = 0.0303
    test_off_acc  = 0.0126
    test_page_acc = 0.0441
    inf_us        ~= 574 us
```

### `val_off_acc`

`val_off_acc` means offset-prediction accuracy on the same workload as training, but on a later time slice.

It answers:

```text
Can the model learn a pattern that continues within the same workload?
```

The biggest change is V1 to V2:

```text
V1 val_off_acc = 0.0124
V2 val_off_acc = 0.0404
```

Adding PC helped the GRU fit `mcf`'s validation slice.

### `test_off_acc`

`test_off_acc` here means accuracy on a different workload, `omnetpp`.

It answers:

```text
Does the learned behavior transfer from mcf to a different binary/workload?
```

The important comparison is again V1 to V2:

```text
V1 test_off_acc = 0.0126
V2 test_off_acc = 0.0127
```

So PC helped in-trace validation, but did not help cross-trace testing.

### Insight

The careful conclusion is:

```text
PC is useful as an in-workload/local-context feature, but raw PC does not transfer well across binaries.
```

This does not mean PC is useless. It means PC should be used carefully. For per-application training, PC can help. For cross-binary generalization, PC can become memorization.

---

## 4. What the V1--V4 IPC results mean

ChampSim replay on `620.omnetpp_s-874B`:

```text
baseline no prefetch = 0.3157 IPC
GRU_V1 = 0.3038 IPC = 0.9623x
GRU_V2 = 0.2783 IPC = 0.8815x
GRU_V3 = 0.2781 IPC = 0.8809x
GRU_V4 = 0.2807 IPC = 0.8891x
```

The key result is:

```text
V2 improves validation accuracy but hurts IPC more than V1.
```

This is an important prefetching lesson. Accuracy and IPC are not the same objective.

The validation label asks:

```text
Did I predict the next label correctly?
```

The processor asks:

```text
Did the prefetch arrive early enough, not too early, and without wasting cache/bandwidth/MSHR resources?
```

So V1--V4 show:

```text
A feature can improve label fitting but make prefetching worse.
```

This is why later experiments must always report both ML metrics and system metrics.

---

## 5. Why `omnetpp` was not a good first demo trace

Upper-bound sweep:

```text
620.omnetpp_s-874B:
    LRU        = 0.2806 IPC
    LRU+stride = 0.2822 IPC
    LRU+SPP    = 0.2833 IPC
    SRRIP+SPP  = 0.2786 IPC
```

The spread is very small. So if a neural prefetcher does not improve `omnetpp`, there are two possible explanations:

```text
1. the neural prefetcher is weak
2. this trace/window has little prefetch headroom
```

This is why V9 moved to traces with clearer headroom:

```text
619.lbm_s-4268B:
    LRU 0.4523 -> SPP 0.5321   (~18% headroom)

602.gcc_s-734B:
    LRU 0.5564 -> SPP 1.330    (~139% headroom)
```

---

## 6. V8: why it changed so much from V1--V4

V8 was not a small feature ablation. It was a formulation repair.

V1--V4 showed that the old page/offset formulation had problems:

```text
1. offset-only prediction is weak when accesses cross pages
2. one next-offset label does not match multi-line prefetching
3. PC/page features did not solve cross-trace IPC
4. higher validation accuracy did not imply higher IPC
```

So V8 changed the target:

```text
old target = next page/offset-style prediction
new target = next delta token from top-256 delta vocabulary + OOV
```

V8 also changed several settings:

```text
history length: longer delta history
input filtering: L1D demand accesses
output space: top-256 delta vocabulary + OOV
policy: confidence-gated prefetch-list export
model size: about 51,953 parameters
```

This is why the V8 numbers look very different from V1--V4. The model was being asked a different and more learnable question.

---

## 7. V8 metrics explained

Recorded V8 summary:

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

This means the model's best delta-token guess was correct about 74.6% of the time on the validation slice.

Compared with V1--V4 offset accuracy, this is much higher. The insight is:

```text
Delta-token prediction is much more learnable than the old offset target.
```

### `test_top1 = 0.4791`, `test_top5 = 0.8507`

This was cross-binary testing. Top-1 dropped relative to validation, but top-5 stayed high.

This means:

```text
The model often places the right future delta in a small candidate set,
but choosing which candidate to issue is still a separate system policy problem.
```

### `train/test OOV labels = 66.7% / 61.2%`

This is the biggest V8 warning.

V8 can only exactly name deltas inside the top-256 vocabulary. If the correct delta is outside that vocabulary, it becomes OOV.

High OOV means:

```text
The memory-delta distribution has a long tail.
A small fixed delta vocabulary cannot express enough targets.
```

This motivates V9's bitmap target.

### `CPU inference = 734 us`

This is far too slow for real hardware. So GRU should be treated as a discovery model, not the final online prefetcher.

Hardware implication:

```text
If GRU discovers useful signals, the final design should be compressed, tabularized, quantized, or replaced by a smaller scorer.
```

### `trigger rate = 33.4%`

V8 emitted prefetches for about one third of scored accesses.

But gated and ungated IPC were almost identical:

```text
GRU_V8 gated   = 0.9604x
GRU_V8 ungated = 0.9610x
```

So the simple confidence gate did not solve the system problem. V8's issue was not only confidence; it was the output/action formulation.

---

## 8. V8 insight and paper transition

V8 proved two things at the same time:

```text
1. The model can learn a better target than page-offset prediction.
2. Better target accuracy still does not automatically produce better IPC.
```

This connects directly to several papers:

```text
Hashemi:
    memory prediction should be formulated as classification/sequence learning, not raw address regression.

Voyager:
    address structure matters; page/offset and per-application behavior need to be handled carefully.

TransFetch / bitmap-style formulations:
    future memory accesses are often better represented as a set/window of possible future lines, not one next token.

DART / Net2Tab:
    a large NN can be useful for discovery, but the hardware path must eventually be smaller and faster.

APT-GET / timeliness work:
    a correct predicted address can still be useless if the timing is wrong.

Limoncello:
    prefetching can hurt under resource pressure.
```

The direct transition from V8 to V9 is:

```text
V8 fixed learnability, but not system usefulness.
V9 changes the output and evaluation setup to better match prefetching.
```

---

## 9. V9 implementation: why it changed from V8

V9 changes several things because V8 exposed several problems:

```text
V8 problem 1: fixed top-256 delta vocabulary has high OOV
V9 change:   use a bounded 129-bit delta bitmap over [-64, +64] cache-line deltas

V8 problem 2: one next-delta token cannot represent multiple useful future lines
V9 change:   label future deltas that appear in the next 8 accesses

V8 problem 3: cross-binary test mixed two questions: can learn vs can transfer
V9 change:   use in-distribution train/val/test split for each trace

V8 problem 4: confidence gate did not fix IPC
V9 change:   make decode policy explicit through probability threshold and max degree

V8 problem 5: omnetpp had little prefetch headroom
V9 change:   evaluate mcf, lbm, and gcc
```

V9 architecture summary:

```text
input sequence: delta-token history
extra context: PC embedding + PC/Delta hash embedding
sequence model: GRU
output: 129-bit sigmoid bitmap
label: which nearby deltas appear in the next 8 accesses
loss: binary cross entropy over bitmap bits
prefetch decode: choose top bitmap positions above threshold, limited by max degree
```

V9 should be read as a redesign checkpoint, not a one-variable ablation.

---

## 10. V9 current system results

Use the full-index V9 rows:

```text
trace              baseline IPC   V9 IPC    speedup   list lines   issued   accesses seen
602.gcc_s-734B     0.5427         0.5463    1.0066x   82,396       61,287   4,154,684
619.lbm_s-4268B    0.4345         0.4346    1.0002x   180,907      31,944   4,800,801
605.mcf_s-994B     0.1841         0.1812    0.9842x   345,329      213,950  8,454,906
```

### `gcc`: small positive signal

`gcc` improved from `0.5427` to `0.5463`, or `1.0066x`.

This is small, but important. It is the first V9 system result showing that the GRU bitmap prefetch list can improve IPC at all.

The careful interpretation is:

```text
V9 has a positive system signal on gcc, but it captures only a small part of gcc's available prefetch headroom.
```

This makes `gcc` the best trace for the next decode-policy sweep.

### `lbm`: offline learnable, system-neutral

`lbm` improved from `0.4345` to `0.4346`, or `1.0002x`, which is essentially neutral.

Because `lbm` has strong offline V9 prediction quality, this suggests the problem is likely not simply “the model cannot learn lbm.” More likely, the current issuing policy is not strong enough or not timed well enough.

Careful interpretation:

```text
The bitmap target is learnable on lbm, but the current decode policy does not translate that into meaningful IPC improvement.
```

### `mcf`: negative prefetch result

`mcf` dropped from `0.1841` to `0.1812`, or `0.9842x`.

This is consistent with the offline V9 diagnosis that `mcf` had low precision and very high recall. For irregular/pointer-like workloads, many speculative prefetches can hurt by causing pollution or resource pressure.

Careful interpretation:

```text
mcf is not a good first showcase for V9 prefetching.
It may be better suited to bypass, replacement, or utility-gating experiments.
```

---

## 11. What V9 teaches us

V9 gives a more nuanced result than simply “GRU works” or “GRU fails.”

What we know now:

```text
1. V9 gives a small positive IPC signal on gcc.
2. V9 is neutral on lbm.
3. V9 hurts mcf.
4. High offline accuracy/F1 still does not guarantee IPC improvement.
5. The next bottleneck is likely action policy: threshold, degree, timing, and usefulness filtering.
```

This supports the larger research direction:

```text
The problem is not only predicting future addresses.
The problem is deciding which speculative memory actions are worth issuing under hardware constraints.
```

---

## 12. Controlled-variable status

The honest answer is:

```text
Partly controlled.
```

More precise:

```text
V1--V4:
    controlled feature sweep inside the GRU family.

V8:
    target redesign after the old page/offset formulation looked weak.

V9:
    formulation redesign after V8 exposed OOV, single-token output, and action-policy problems.
```

The next controlled experiments should happen inside the V9 formulation:

```text
same trained model, vary threshold / max degree
same V9 target, compare cheaper model families
same V9 model, remove PC+Delta feature
same V9 model, vary next-window size
```

---

## 13. Immediate next step

Do not switch to a new NN family yet.

First run a V9 decode-policy sweep, especially on `gcc`:

```text
trace: 602.gcc_s-734B
model: same V9 GRU
features: same
change only: probability threshold and max_degree
```

Suggested sweep:

```text
threshold: 0.30, 0.50, 0.70, 0.90
max_degree: 1, 2, 4
```

Why this is the correct next step:

```text
It controls the model and features.
It changes only the prefetch action policy.
It directly tests whether V9 is too aggressive, too conservative, or issuing the wrong number of candidates.
```

After the decode sweep:

```text
If gcc improves more:
    keep V9 target and compare cheaper model families.

If gcc stays near 1.00x:
    investigate timeliness/pollution/usefulness counters.

If mcf stays negative:
    move mcf toward bypass/replacement/utility gating rather than using it as the prefetch showcase.
```

---

## 14. One-paragraph explanation for others

```text
The GRU experiments show a progression from weak offset prediction to a more realistic delta-bitmap formulation. The controlled V1--V4 sweep showed that adding features like PC can improve validation accuracy without improving IPC. V8 made the target much more learnable with delta-token prediction, but high OOV and no IPC gain showed that prediction accuracy alone is not enough. V9 uses in-distribution training and a 129-bit future-delta bitmap. Current V9 system results show a small positive signal on gcc, neutral behavior on lbm, and negative behavior on mcf. The next bottleneck is not model capacity; it is utility-aware issuing: threshold, degree, timing, and resource-aware filtering.
```
