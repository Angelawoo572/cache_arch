# GRU Experiment Notes

This file explains the GRU-related experiments in the current cache-prefetching project.

The goal is not to claim that GRU is the final hardware design. The goal is to use GRU as a discovery model to learn what matters in neural prefetching:

```text
feature choice -> target choice -> model prediction -> prefetch-list decode policy -> ChampSim IPC
```

The main lesson so far is:

```text
Better ML prediction accuracy is necessary but not sufficient.
A prefetch is only useful if it is issued at the right time, with the right degree, and without creating too much cache pollution or memory-system pressure.
```

---

## 1. Early model zoo: why model family alone was not enough

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

### Lesson

The bottleneck is not simply “use a bigger neural network.” The target, feature representation, and prefetch action policy matter more than raw model family at this stage.

The early formulation was useful for pipeline bring-up, but it had limitations:

```text
1. The label was only the next offset inside the current page.
2. It did not fully handle page changes.
3. It asked for one next target, but real prefetchers often need multiple future candidates.
4. It did not explicitly model no-prefetch / suppress decisions.
5. It measured model prediction more than system usefulness.
```

---

## 2. V1--V4: controlled GRU feature sweep

Relevant files:

```text
notebook/gru_sweep_cross_trace.ipynb
notebook/gru_sweep_v2.ipynb
results/gru_sweep_summary.csv
results/nn_demo_summary.csv
```

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

### V1--V4 accuracy lesson

The most important comparison is V1 to V2:

```text
V1 val_off_acc = 0.0124
V2 val_off_acc = 0.0404

V1 test_off_acc = 0.0126
V2 test_off_acc = 0.0127
```

Adding PC helped the model fit the same workload, but did not help cross-workload transfer.

Careful conclusion:

```text
PC is useful as an in-workload/local-context feature, but raw PC does not transfer well across binaries.
```

### V1--V4 IPC lesson

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

This is the first strong system lesson:

```text
A feature can improve label fitting but make prefetching worse.
```

Accuracy asks:

```text
Did I predict the next label correctly?
```

IPC asks:

```text
Did the prefetch arrive early enough, not too early, and without wasting cache/bandwidth/MSHR resources?
```

---

## 3. Why `omnetpp` was not a good first demo trace

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

This motivated moving to traces with clearer behavior and headroom.

---

## 4. What the main traces mean

```text
602.gcc_s-734B:
    GCC compiler workload.
    Mixed control/data behavior, many program phases, strong SPP headroom.
    Best current trace for showing whether V9 can produce useful prefetch candidates.

619.lbm_s-4268B:
    Lattice Boltzmann fluid-dynamics workload.
    More regular / streaming / stencil-like memory behavior.
    Offline V9 prediction is highly learnable, but current IPC was neutral.

605.mcf_s-994B:
    Minimum-cost flow / graph-like optimization workload.
    More irregular / pointer-like memory behavior.
    Prefetching can easily hurt; better candidate for bypass, replacement, or utility gating.
```

Current interpretation:

```text
gcc = best prefetch-debug trace
lbm = regularity sanity-check trace
mcf = irregular utility/bypass trace
```

---

## 5. V8: target repair

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

### V8 lesson

V8 proved that delta-token prediction is much more learnable than the old offset target.

But V8 still did not improve IPC:

```text
GRU_V8 gated   = 0.9604x
GRU_V8 ungated = 0.9610x
```

The main warning was high OOV:

```text
The memory-delta distribution has a long tail.
A small fixed delta vocabulary cannot express enough targets.
```

This motivated V9's bounded delta-bitmap target.

---

## 6. V8 to V9: paper-guided transition

The direct transition is:

```text
V8 fixed learnability, but not system usefulness.
V9 changes the output and evaluation setup to better match prefetching.
```

Paper connections:

```text
Hashemi:
    memory prediction should be formulated as classification/sequence learning, not raw address regression.

Voyager:
    address structure matters; per-application behavior needs to be handled carefully.

TransFetch / bitmap-style formulations:
    future memory accesses are often better represented as a set/window of possible future lines, not one next token.

DART / Net2Tab:
    a large NN can be useful for discovery, but the hardware path must eventually be smaller and faster.

APT-GET / timeliness work:
    a correct predicted address can still be useless if the timing is wrong.

Limoncello:
    prefetching can hurt under resource pressure.
```

---

## 7. V9 formulation

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

## 8. V9 first full-index system results

Full-index V9 replay results:

```text
trace              baseline IPC   V9 IPC    speedup   list lines   issued   accesses seen
602.gcc_s-734B     0.5427         0.5463    1.0066x   82,396       61,287   4,154,684
619.lbm_s-4268B    0.4345         0.4346    1.0002x   180,907      31,944   4,800,801
605.mcf_s-994B     0.1841         0.1812    0.9842x   345,329      213,950  8,454,906
```

Interpretation:

```text
gcc:
    small positive signal; V9 can generate some useful candidates.

lbm:
    offline learnable, but current system effect is neutral.

mcf:
    negative prefetch result; likely needs usefulness filtering / bypass / replacement rather than raw prefetching.
```

This already suggests that the next bottleneck is not pure model capacity.

---

## 9. V9 gcc decode-policy sweep

The next controlled experiment kept the same trained V9 GRU and changed only the decode policy:

```text
threshold: 0.30, 0.50, 0.70, 0.90
degree:    1, 2, 4
trace:     602.gcc_s-734B
```

Current corrected gcc results:

```text
setting        issued     IPC      speedup
th030 deg1     22,782     0.5440   1.0024x
th030 deg2     82,393     0.5452   1.0046x
th030 deg4    194,528     0.5446   1.0035x

th050 deg1     22,740     0.5440   1.0024x
th050 deg2     82,293     0.5453   1.0048x
th050 deg4    180,968     0.5450   1.0042x

th070 deg1     22,653     0.5442   1.0028x
th070 deg2     82,063     0.5446   1.0035x
th070 deg4    177,692     0.5444   1.0031x
```

Best current setting:

```text
th050 deg2 = 1.0048x
```

### Decode-sweep lesson

The sweep shows three things:

```text
1. V9 has a stable small positive signal on gcc.
2. Degree 2 is usually better than degree 1, so some second candidates are useful.
3. Degree 4 issues many more prefetches but does not improve IPC, so extra candidates have lower utility and may create pollution/resource pressure.
```

This is the strongest evidence so far that the bottleneck is now utility-aware issuing, not simply GRU capacity.

---

## 10. Do we need to run `619.lbm` before moving on?

Running `lbm` would be useful for completeness, but it is not required to decide the next research direction.

Why we can already move forward:

```text
1. gcc has high prefetch headroom, but V9 only gets about +0.5%.
2. Increasing degree greatly increases issued prefetches, but IPC barely improves.
3. mcf already shows that raw prefetching can hurt irregular workloads.
4. lbm was already neutral in the first full-index run.
```

So the project already has enough evidence to say:

```text
The next stage should be utility / timeliness / filtering, not just a bigger GRU.
```

If time is limited, skipping `lbm` is reasonable. If this becomes a formal report/paper table, running `lbm` later would make the claim stronger because it provides a regular-memory contrast to `gcc` and `mcf`.

---

## 11. Current conclusion: did GRU V9 succeed?

The honest answer is:

```text
GRU V9 succeeded as a discovery stage, but not as a final prefetcher.
```

Succeeded:

```text
1. It found a better target than offset prediction.
2. It produced a real positive IPC signal on gcc.
3. It showed that decode degree matters.
4. It exposed that offline accuracy/F1 is not enough for system usefulness.
```

Not enough:

```text
1. Best gcc speedup is only around +0.5%.
2. V9 is far below SPP headroom on gcc.
3. Extra degree does not scale well.
4. mcf still hurts under raw prefetching.
5. CPU inference is far too slow for direct hardware use.
```

Therefore, the next phase should not be “make GRU bigger.”

The next phase should be:

```text
utility-aware issuing / timeliness / filtering
```

---

## 12. Next stage

The next stage should ask:

```text
Given a set of predicted candidate prefetches, which ones are worth issuing under current hardware conditions?
```

Useful signals to add:

```text
candidate confidence
prefetch degree
recent usefulness
lateness / earliness
MSHR pressure
memory bandwidth pressure
cache-set pressure / pollution risk
PC or behavior-class history
explicit no-prefetch decision
```

This moves the project from:

```text
predict the next address
```

toward:

```text
score the utility of speculative memory actions
```

This matches the larger research direction: a behavior-class / utility controller, not a direct neural next-address predictor.

---

## 13. One-paragraph explanation for others

```text
The GRU experiments show a progression from weak offset prediction to a more realistic delta-bitmap formulation. The controlled V1--V4 sweep showed that adding features like PC can improve validation accuracy without improving IPC. V8 made the target more learnable with delta-token prediction, but high OOV and no IPC gain showed that prediction accuracy alone is not enough. V9 uses in-distribution training and a 129-bit future-delta bitmap. It gives a stable but small positive signal on gcc, neutral behavior on lbm, and negative behavior on mcf. A gcc decode sweep shows that degree 2 is slightly better than degree 1, but degree 4 issues many more prefetches without meaningful extra IPC. Therefore the next bottleneck is not GRU capacity; it is utility-aware issuing: threshold, degree, timeliness, pollution, and resource-aware filtering.
```
