# GRU Experiment Notes: What Each Notebook Proves, What Is Still Wrong, and How It Connects to Papers

This note is intentionally **not** a final research idea. It is a readable experiment log for the current GRU line of the project.

Professor's requested mindset:

```text
try one model family -> control one variable at a time -> measure accuracy / latency / IPC -> observe failure modes -> revise -> extract insight
```

So the role of the GRU experiments is not to prove that GRU is the final architecture. GRU is the first NN family we are using to learn which input features, targets, and workload classes matter.

---

## 0. Current high-level status

The GRU line currently has three stages:

1. **Early model zoo / smoke-style exploration**
   - Files: `neural_prefetcher_zoo*.ipynb`
   - Question: if all models get the same input and target, does model family alone matter?
   - Result: not much. MLP / LSTM / Transformer were close in accuracy, but latency differed a lot.

2. **Cross-trace controlled-variable GRU sweep**
   - Files: `gru_sweep_cross_trace.ipynb`, `gru_sweep_v2.ipynb`
   - Question: within the GRU family, what happens if we add one feature at a time?
   - Controlled variable: input feature set.
   - Result: in-trace validation accuracy can improve while held-out trace accuracy and IPC get worse. PC especially looked like memorization.

3. **V8 / V9 redesign**
   - File: `gru_sweep_v8.ipynb`
   - Question: was the old output target too weak?
   - Result: yes. Delta-target formulation dramatically improved top-1/top-5 accuracy, but IPC still did not improve.
   - File: `gru_sweep_v9.ipynb`
   - Question: after V8 showed the target was better but still system-weak, can we use in-distribution training + delta bitmap output + PC+Delta context to make the prefetch list more useful?
   - V9 results should **not** be over-interpreted until ChampSim IPC for the V9 prefetch lists is available.

---

## 1. Early model zoo: what it was useful for, and what was wrong

Relevant notebooks:

```text
notebook/neural_prefetcher_zoo.ipynb
notebook/neural_prefetcher_zoo_v2.ipynb
notebook/neural_prefetcher_zoo_v3.ipynb
```

The model zoo compared several NN families on the same task:

```text
input  = PC hash + last 4 delta tokens
label  = next cache-line offset within the current 4 KiB page
models = Perceptron / MLP / CNN / LSTM / Transformer
```

This was a good first exploration because it answered a simple question: **does a more complex NN family automatically solve the problem?**

The answer was no. The recorded validation accuracies were all close:

```text
Perceptron  val_acc ~= 0.0895
MLP         val_acc ~= 0.0962
CNN         val_acc ~= 0.0920
LSTM        val_acc ~= 0.0952
Transformer val_acc ~= 0.0934
```

But CPU inference latency was very different:

```text
Perceptron  ~=  51 us
MLP         ~= 100 us
CNN         ~= 206 us
LSTM        ~= 348 us
Transformer ~= 535 us
```

### What these numbers mean

`val_acc` here means: among 64 possible offsets inside the current 4 KiB page, did the model predict the next access's offset? Chance is about `1/64 = 1.56%`. So ~9% is above chance, but not strong enough to trust as a useful prefetcher.

The latency numbers mean: even if LSTM/Transformer were slightly more accurate, they are much slower on CPU inference. For hardware, this matters because a prefetcher that predicts too late is often useless. This already points toward the DART / Net2Tab lesson: large NN quality is not enough; the online path must be tiny or table-like.

### What was wrong with this stage

This stage was useful, but it had methodology issues:

1. **The target was too weak.**
   - Predicting only the next page offset ignores page crossing.
   - A correct offset in the wrong page becomes a wrong prefetch.
   - This is why Voyager-style page/offset decomposition and later delta-based targets became necessary.

2. **The output was single-class.**
   - Real prefetching can issue multiple candidate lines.
   - A single next-offset target does not match variable-degree prefetching.
   - This motivates TransFetch/DART-style delta bitmap output.

3. **Some early runs used capped rows / notebook-smoke settings.**
   - `MAX_ROWS = 1_000_000` made the run easier for Colab but less representative.
   - Random train/val split can leak future temporal information.
   - Later notebooks moved toward full trace loads and time-ordered splits.

4. **Every model exported a prefetch for almost every scored access.**
   - That is too aggressive.
   - If accuracy is mediocre, this creates pollution and bandwidth pressure.

### Insight from this stage

The main insight is:

```text
Model family is not the first bottleneck.
The target, feature representation, prefetch degree, and system-level usefulness matter more.
```

This is why the project moved from “try five models and pick the best validation accuracy” to “within each model family, control the features and target carefully.”

---

## 2. `gru_sweep_cross_trace.ipynb` / `gru_sweep_v2.ipynb`: what was controlled

These notebooks are the clearest controlled-variable GRU experiments so far.

They set up:

```text
train trace = 605.mcf_s-994B
held-out test trace = 620.omnetpp_s-874B
train split = first 70% of train trace
val split   = last 30% of train trace
model family = GRU
output = Voyager-style dual head: page hash + offset
```

The controlled variable was the **input feature set**:

```text
V1 = delta history only
V2 = V1 + PC
V3 = V2 + page hash
V4 = V3 + per-PC stats: miss_rate + log_freq
```

This is a good controlled-variable design because the model family and basic training protocol stay fixed, while the feature set changes one step at a time.

---

## 3. What the V1--V4 offline metrics mean

Recorded summary:

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

### Meaning of `val_off_acc`

`val_off_acc` is measured on the same workload as training (`mcf`), but on a later time slice. It tells us whether the model learned patterns that persist within the same trace.

V2 increasing `val_off_acc` from `0.0124` to `0.0404` means:

```text
Adding PC helped the model fit mcf's local behavior.
```

But this is not enough to claim the feature is useful in hardware or even generally useful for prefetching.

### Meaning of `test_off_acc`

`test_off_acc` is measured on a different trace (`omnetpp`). It tells us whether the learned behavior transfers across workload/binary.

The key observation is:

```text
V1 test_off_acc = 0.0126
V2 test_off_acc = 0.0127
```

So PC helped in-trace validation, but did almost nothing on the held-out trace.

Interpretation:

```text
PC is probably memorizing mcf-specific instruction behavior.
It does not transfer to omnetpp.
```

This is not surprising. A PC hash is binary-specific; the numeric PC value in one program does not mean the same thing in another program.

### Meaning of `val_page_acc` / `test_page_acc`

The page head tries to predict a hashed future page. Higher page accuracy would mean the model is learning long-range page movement, not only offset movement.

V3 improved `test_page_acc` to `0.0572`, but `test_off_acc` remained low. That means the page head may learn some page-level structure, but the final prefetch address is still not reliable enough. In the current export logic, the page head also does not fully solve target reconstruction, so page accuracy alone cannot guarantee IPC gain.

### Meaning of `inf_us`

CPU inference latency rises from about `450 us` to `574 us` as features and heads are added. For a real hardware prefetcher, this is far too slow. In this project, CPU inference is only a proxy to compare relative model cost. The real lesson is:

```text
If a feature gives tiny accuracy benefit but increases latency/params, it is not hardware attractive.
```

---

## 4. What the V1--V4 IPC numbers mean

Recorded ChampSim replay summary on `620.omnetpp_s-874B`:

```text
baseline no prefetch = 0.3157 IPC
GRU_V1 = 0.3038 IPC = 0.962x
GRU_V2 = 0.2783 IPC = 0.881x
GRU_V3 = 0.2781 IPC = 0.881x
GRU_V4 = 0.2807 IPC = 0.889x
```

### Interpretation

This is the first major insight:

```text
Higher in-trace validation accuracy made IPC worse.
```

Specifically:

```text
V1 -> V2:
    val_off_acc improved from 0.0124 to 0.0404
    but IPC dropped from 0.962x to 0.881x
```

This means PC was not a universally useful feature. It made the GRU more confident about mcf-specific behavior, then caused harmful prefetches on omnetpp.

### Why this matters

This shows the project should not optimize only ML accuracy. A prefetch can be harmful if it is:

```text
wrong address
right address but too late
right address but too early and evicted
right address but causes cache pollution
right address but wastes bandwidth / MSHRs
```

So the experimental objective must include IPC, MPKI, trigger rate, and prefetch degree, not just `val_acc`.

---

## 5. Why `omnetpp` became a bad demo trace

The upper-bound sweep showed:

```text
620.omnetpp_s-874B:
    LRU        = 0.2806 IPC
    LRU+stride = 0.2822 IPC
    LRU+SPP    = 0.2833 IPC
    SRRIP+SPP  = 0.2786 IPC
```

The total spread is only about 1%. That means even a strong conventional prefetcher does not improve this trace much in the chosen window.

So if GRU does not improve `omnetpp`, it may be because:

```text
1. GRU is bad, or
2. omnetpp has little prefetch headroom, or
3. the replay window is not exposing the memory behavior where prefetch matters.
```

This is why V9 moves to traces with more headroom:

```text
619.lbm_s-4268B: LRU 0.4523 -> SPP 0.5321  (~+18%)
602.gcc_s-734B: LRU 0.5564 -> SPP 1.330   (~+139%)
```

---

## 6. Transition from V1--V4 to V8

V1--V4 showed two problems:

1. Cross-trace PC/page features do not transfer well.
2. The offset target is too weak.

V8 focused on the second problem: **change the target**.

V8 changed the formulation from:

```text
old target: next offset among 64 offsets in current page
```

to:

```text
new target: next delta token from a learned top-K delta vocabulary
```

V8 also changed several settings:

```text
history length: larger delta history, HIST=16
vocab: top-256 learned deltas + OOV
filter: L1 demand misses only
model: smaller GRU-style sequence model, ~52K params
confidence gate: emit only if confidence >= threshold
```

This is why V8 changed results so much. It was not just a small feature change. It was a target redesign.

---

## 7. V8 data: what each number means

Recorded V8 summary:

```text
best_val_top1 = 0.7458
cross-binary test_top1 = 0.4791
test_top5 = 0.8507
train OOV labels = 66.7%
test OOV labels = 61.2%
CPU inference = 734 us
trigger rate = 33.4%
params = 51,953
```

### `best_val_top1 = 0.7458`

This means the model can predict the next delta token very well within the training workload distribution. Compared with V1--V4 offset accuracy, this is a huge improvement.

Interpretation:

```text
The old page-offset target was a bad target.
Delta-token prediction is much more learnable.
```

### `test_top1 = 0.4791`, `test_top5 = 0.8507`

This is cross-binary testing from mcf to omnetpp. Top-1 is lower than in-trace validation, but still much higher than the old offset accuracy.

Interpretation:

```text
Delta patterns transfer better than raw page/offset labels,
but cross-binary generalization is still weaker than in-distribution.
```

### `train/test OOV labels = 66.7% / 61.2%`

This is the most important V8 diagnostic.

The top-256 delta vocab covers only a minority of label occurrences. Most labels fall into OOV. So even if the model is good on head deltas, it cannot express most tail deltas.

Interpretation:

```text
V8 fixed the old target, but introduced a fixed-vocabulary bottleneck.
The address/delta distribution is heavy-tailed.
A small fixed top-K output cannot cover enough future accesses.
```

This directly motivates V9's delta-bitmap output.

### `CPU inference = 734 us`

This is much too slow for a real CPU prefetcher. It is acceptable only as offline exploration. It means any final hardware idea cannot be “run this PyTorch GRU online.”

Interpretation:

```text
GRU is a discovery tool.
Final hardware must be smaller, table-like, compressed, quantized, or used only to generate policies offline.
```

This matches the DART / Net2Tab lesson.

### `trigger rate = 33.4%`

V8 emitted prefetches for about one third of scored accesses.

At first this looks conservative. But V8's gated and ungated IPC were almost identical:

```text
GRU_V8 gated   = 0.9604x
GRU_V8 ungated = 0.9610x
```

Interpretation:

```text
The softmax confidence gate was not solving the system problem.
The model was already implicitly gated by OOV predictions.
```

So V9 should not simply add another confidence threshold. It needs a target/output that naturally supports multiple future candidates and no-prefetch behavior.

---

## 8. V8 IPC: why high accuracy still failed

Recorded replay summary:

```text
baseline = 0.3157 IPC
GRU_V8 gated = 0.3032 IPC = 0.9604x
GRU_V8 ungated = 0.3034 IPC = 0.9610x
```

V8 improved ML accuracy dramatically but did not improve IPC.

This means the bottleneck is not only “can the model predict something?” The bottleneck is:

```text
Does the emitted prefetch arrive at the right time, with low pollution, under cache/bandwidth constraints?
```

This is the key reason the project must track both ML metrics and system metrics.

---

## 9. Why V9 changes many things relative to V8

V9 should not be described as a small controlled-variable ablation. It is a redesign checkpoint caused by V8's failure modes.

V8 proved:

```text
1. Delta target is much more learnable than offset target.
2. Fixed top-256 vocab has severe OOV bottleneck.
3. Cross-binary evaluation creates a large train/test gap.
4. Confidence gating did not improve IPC.
5. Omnetpp has very little prefetch headroom.
```

Therefore V9 changes several things at once:

### A. In-distribution split

V9 trains, validates, and tests on different time slices of the same trace:

```text
train = first 70%
val   = next 15%
test  = last 15%
```

Reason:

```text
V8 cross-binary testing mixed two questions:
    can the model learn the pattern?
    can the model transfer to another binary?
V9 isolates the first question.
```

This is paper-grounded because Hashemi, Voyager, and Pythia-style evaluations are generally per-application/per-workload rather than one binary training a model for a totally different binary.

### B. Delta-bitmap output

V9 predicts a 129-bit bitmap over nearby deltas:

```text
delta range = [-64, +64] cache lines
label bit = 1 if that delta appears in the next 8 accesses
```

Reason:

```text
V8's single delta-token target cannot represent multiple useful future lines.
V8's fixed top-K output also has OOV bottleneck.
Bitmap output bounds the output space and supports variable-degree prefetching.
```

This is the TransFetch/DART direction.

### C. PC + Delta hash feature

V9 adds a PC+Delta hash embedding.

Reason:

```text
PC alone overfits across binaries,
but PC combined with recent delta can describe local load behavior within one application.
```

This is related to Pythia's state-vector design: program context plus recent memory behavior is more informative than address history alone.

### D. New trace choices

V9 should not use omnetpp as the main demo trace because omnetpp has low prefetch headroom.

Better first-order traces:

```text
605.mcf_s-994B:
    difficult irregular/pointer-like behavior; useful for seeing failure modes and bypass potential.

619.lbm_s-4268B:
    more regular/streaming-like behavior; SPP has meaningful headroom.

602.gcc_s-734B:
    large SPP headroom; useful for seeing whether GRU prefetch can become system-useful.
```

Important: these names are not final taxonomy labels. After results, we should classify traces by observed behavior:

```text
high accuracy + high IPC
high accuracy + low IPC
low accuracy + low IPC
SPP-sensitive
GRU-sensitive
bypass-sensitive
```

---

## 10. Do not over-analyze V9 yet

V9 implementation is ready to answer a better question, but the final conclusion requires ChampSim IPC.

Offline metrics alone are not enough.

For each V9 trace we need:

```text
ML metrics:
    vocab coverage
    top1 / top3 / top5
    precision / recall / F1

Action metrics:
    number of prefetches emitted
    trigger_per_access
    average degree
    confidence distribution

System metrics:
    baseline IPC
    V9 IPC
    speedup
    LLC MPKI
    bandwidth / pollution / useless prefetches if available

Cost metrics:
    params
    CPU inference latency
```

The critical cases are:

```text
high F1 + high IPC:
    target and decode are useful.

high F1 + low IPC:
    model learned pattern, but prefetch action is mistimed / polluting / too aggressive.

low F1 + high IPC:
    ML metric is misaligned with useful prefetching.

low F1 + low IPC:
    feature/target/model is not enough for that workload.
```

---

## 11. Paper connections

### Hashemi et al., Learning Memory Access Patterns

Main lesson used here:

```text
Prefetching should be formulated as classification over sparse memory behavior, not address regression.
```

Project connection:

```text
Early offset prediction was too weak.
V8's delta-token target is closer to this classification framing.
```

### Voyager

Main lesson used here:

```text
Address prediction needs structure: page/offset decomposition and per-application learning.
```

Project connection:

```text
V1--V4 used a Voyager-style page/offset dual head.
V8/V9 learned that cross-binary transfer is not a fair first test.
V9 moves to in-distribution per-trace splits.
```

### TransFetch

Main lesson used here:

```text
Prefetching multiple future lines is not the same as text next-token prediction.
Delta bitmap output better matches unordered future address sets.
```

Project connection:

```text
V9's 129-bit delta bitmap is directly motivated by this idea.
```

### DART / Net2Tab

Main lesson used here:

```text
NN accuracy is not enough. Online prefetch inference latency must be small.
A heavy NN can be used offline, but practical hardware should become table-like or low-cost.
```

Project connection:

```text
GRU is currently a discovery tool.
If GRU finds useful features, the final design should compress or replace the GRU with smaller tables, counters, perceptrons, or distilled logic.
```

### Pythia

Main lesson used here:

```text
Prefetching is a decision problem under system constraints.
Useful state includes PC, deltas/history, and feedback about prefetch usefulness/resources.
```

Project connection:

```text
V9's PC+Delta hash is a first step toward program-context + memory-behavior state.
Future RL-family experiments should use Pythia as the main reference.
```

### APT-GET

Main lesson used here:

```text
Correct prefetches can still be useless if they are too early or too late.
```

Project connection:

```text
If V9 has high F1 but low IPC, timeliness should become a leading hypothesis.
```

### Limoncello

Main lesson used here:

```text
Prefetching can hurt when bandwidth/resource pressure is high.
```

Project connection:

```text
If V9 emits many prefetches and IPC drops, the next step is not necessarily a bigger GRU.
It may require throttling, degree control, or bypass/suppress logic.
```

### TLP / perceptron filtering

Main lesson used here:

```text
A small perceptron-style predictor can be useful for off-chip prediction and prefetch filtering with low storage.
```

Project connection:

```text
After GRU, the next family should probably be Perceptron or CNN using the V9 target, because they are cheaper than GRU/Transformer.
```

---

## 12. When to switch away from GRU

Do not switch just because one trace fails.

Use this decision table:

```text
If V9 improves IPC on lbm and gcc:
    GRU target/features are useful.
    Next step: try cheaper families (Perceptron/CNN/MLP) with the same V9 bitmap target.

If V9 has high F1 but low IPC on lbm/gcc:
    Do not switch family yet.
    First debug decode, threshold, degree, replay alignment, and timeliness.

If V9 has low F1 and low IPC on all traces:
    GRU formulation is not enough.
    Try another family or another target.

If mcf fails but lbm/gcc work:
    This is not a failure.
    It suggests workload-specific policy: mcf may need bypass/replacement/utility gating more than prefetching.
```

---

## 13. Immediate next experiment protocol

For the current V9 run, fill this table before changing code:

```text
trace | baseline IPC | SPP IPC | V9 IPC | speedup | top1 | top5 | F1 | precision | recall | trigger/access | CPU us
```

Then interpret with this order:

1. Does V9 beat baseline IPC?
2. Does V9 approach SPP on traces where SPP has headroom?
3. If not, is offline F1 high or low?
4. If F1 high but IPC low, inspect degree / timing / pollution.
5. If F1 low, inspect vocab coverage / zero-label fraction / output target.
6. Only after that decide whether to tune GRU or move to the next NN family.
