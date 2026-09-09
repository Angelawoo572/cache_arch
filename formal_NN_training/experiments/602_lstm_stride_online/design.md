# Causal online learning for the tiny Stride neural prefetcher

## Question and progression

How many retired instructions, eligible L2 observations, and actual optimizer
updates precede useful performance, including startup cost? What state,
computation, bandwidth, and service time accompany it? Summer work trained on
offline PC streams and replayed actions. The live branch deploys the same frozen
model. This experiment changes weights during a causal, closed-loop CPU run.
It is simulator/CPU co-simulation, not completed on-chip training hardware.

The existing model and loss are authoritative: the shared access wrapper imports
the compact PC-keyed hurdle model from the original offline trainer. Inputs are
only 64 PC bits and 64 aligned byte-address bits. H=8 is primary; H=16 checks
capacity. Learned emission, positive count, direct signed deltas, and predicted
decoder feedback are unchanged. The existing scalar C++ forward is extended in
the isolated build through `runtime/prepare_forward.py`, preserving arithmetic.

## Causal loop

Each eligible LOAD callback is observed in the simulator's actual L2 callback
order. It enters a bounded 16-event FIFO if space exists. Predictions use the
current published weights and last completed h/c, never teacher state or cache
outcomes. A separate instance of the actual conventional Stride produces loss
labels only. Its 64 trackers and degree 2 preserve current-line-first requests
`[current, current + stride]`, zero-stride behavior and page restrictions.

The inference engine is serial. At service start it computes a coherent result
from one weight version; h/c and outputs become visible only at its scheduled
completion. Waiting requests cannot see future recurrent states. The finite
table has 64 exact-PC entries with LRU replacement on admitted inference starts.
Each allocation has a distinct lifetime ID. An unbounded routing control is kept.

Completed predictions provide saved predecision h/c and labels for training.
There is one update in flight and at most 64 additional pending examples. Full
queues drop newest work, with separate inference, supervision, training and
output-drop counters. No teacher action supplies a runtime address, gate or K.

The worker receives chronological batches of 64 already-observed decisions.
It groups same-lifetime subsequences only inside that batch, preserves order,
and starts each subsequence from its saved first h/c. It calls the authoritative
packed encoder and loss, then takes one Adam step at LR=0.002. There is no replay,
burn-in, gradient clipping, regularization or future-PC collection. Maximum
same-lifetime TBPTT length is 64 and its actual peak is measured; 64 global
decisions are not 256 same-PC steps. h/c carries detached across publications.

Gate weights are one until both classes occur among labels available to the
trainer, then cumulative `N/(2*n_class)`. Silent, positive, empty and short batches
are tested. Zero-width silent targets are padded in storage without inventing
an action. Teacher coordinates are used only in loss, including decoder steps.
Random-start arms have no checkpoint or prior trace statistics. Warm-start arms
load seed-7 i1m weights and start a fresh optimizer; their offline pretraining is
reported separately. Frozen arms update only h/c.

The socket is persistent, local and length-framed. No Python process or
checkpoint exchange is launched per callback. A complete FP32 tensor sequence
is validated and published atomically after training and modeled copy delay.
An old-version inference already in flight completes coherently from its saved
result. Finite online hardware reserves a second inference weight bank to retain
that version until completion; its bytes and observed use are counted. Completed
training batches require 64 inference completions, so two banks suffice with
one serial inference and one update in flight. Host waiting for a worker adds
no simulated CPU stall cycles.

## Two matched protocols

Historical compatibility uses the original 25M warmup plus 25M measurement,
zero NN service cost, and unbounded state. It suppresses NN inference and issue
throughout warmup and starts with empty predictor state at measurement. This
matches original parity mode; active conventional Stride throughout warmup is
not an interchangeable reference. Four frozen checks use the actual same
checkpoints and finished historical logs.

Primary cold runs skip 25M opaque trace records using the exact reader's compiled
`input_instr`/`cloudsuite_instr` types. Skipped records are not simulated. No
predictor or trainer sees them. The next 25M records are identical across arms;
the reader does not fetch beyond that endpoint. Caches, predictor histories and
optimizer start fresh, with real cache latencies enabled before cycle one.
This is **cold start at the held-out trace-slice boundary**, not program startup.
The pilot uses skip=20M and 100k instructions in the permitted development gap.

Within each protocol, processor, caches, memory, replacement, load eligibility,
admission and fill paths are identical. Cold IPC is not compared to an old
warm-cache IPC bar. Prefetching may change realized callback timing/order.
No precomputed neural action tape is replayed.

## Predeclared service and storage sensitivity

The functional comparison has zero modeled service cost and unbounded routing.
The h8 seed-7 hardware sensitivity compares frozen-i1m to random-start online
with 64 state entries at zero cost, then at 4 and 16 MACs/cycle. These are
architectural assumptions, not measured silicon timing, area, frequency or power.

Inference and training use separate serial engines, each with matched weight
read bandwidth 16 or 64 B/cycle and a nonlinear unit costing four cycles per
operation. Scratch bandwidth is the same; dense, nonlinear, memory and control
phases serialize conservatively. Inference initiation interval equals its
data-dependent service time. Its dense MACs are
`128H + 8H² + 2H + emit*H + decoded_K*(3H²+4H)`.
Nonlinear work is `6H + emit + decoded_K*(3H+1)`, including the final GRU advance
that the actual forward executes. Control costs `8+ceil(entries/4)+4*decoded_K`
cycles. Additional scratch traffic is `4*(128+4H)+24*decoded_K` bytes.

Decoded K is never clamped to Stride's degree. The cold-study decoder/output
resource permits 32 addresses, and excess decoded addresses, numerical failures,
input drops and output drops are exposed. An output FIFO holds 32 requests and
releases at most one/cycle in finite cases through the original `prefetch_line`
path, with original merging/drop behavior. The main core-cycle hook services
work even during core stalls and without any demand callback.

Training forward MACs use actual batch padding: `padded*128H + N*(8H²+2H) +
positive*H + action_atoms*(3H²+4H)`. The original projection executes on padded
positions; LSTM/head work uses valid positions. A conservative backward bound
adds twice the forward MACs. Nonlinear/control and Adam work are explicit in
`online602_engine.cc`; parameter-state traffic adds 32 bytes/parameter and input/
state traffic. Publication costs `ceil(weight_bytes/bandwidth)+8` cycles. One
training update can be pending; the next begins after publication. The shadow
teacher has a separate latency-24, initiation-1 pipeline (64 tags, four tag
comparisons in each of 16 pipeline stages and eight control stages), with
64 bounded label slots and ideal forwarding of tracker state between overlapping
requests. This assumes 64 comparator lanes distributed across the tag pipeline;
it is not a single four-comparator unit accepting a new lookup every cycle.
The actual build fixes L2 `MAX_READ=1`, resets one read allowance per cache
operation, and operates L2 once per core cycle. Thus at most one eligible LOAD
callback reaches the teacher per cycle. The initialization overrides for other
caches do not change L2. This is a protocol constraint: the present label-ready
schedule relies on it and would require admission scheduling for a wider L2.

All storage is FP32 or explicitly sized integer metadata. Weights have 1,908
parameters (7,632 B) for h8 and 5,220 (20,880 B) for h16. State entries reserve
32 B for exact tag/valid/LRU/lifetime metadata plus 8H B h/c. Configured queue
payloads, active event, arithmetic scratch, staged decoded addresses, shadow
teacher, examples, padded tensors, autograd saved activations, gradients, Adam
moments/steps, training weight copy and publication buffer are reported.

64 nonempty grouped decisions have at most 1,056 padded positions. Saved
activation budgets are explicitly enforced at 1 MiB h8 and 2 MiB h16; observed
unique autograd storage excludes aliases to weights and input tensors. These
reserves are conservative configured bounds, not allocator/RSS measurements.
Host object overhead, JSON transport and measurement logging are separate.
Dedicated predictor storage is compared in size with the actual 256 KiB L2;
it is not asserted to occupy or pollute L2. No hypothetical INT8 result is used.

## Measurements and fixed suite

Snapshots occur at 0, 1k, 10k and every 100k retired instructions. They include
eligible callbacks, admitted/started/completed inference, observed/available
labels, completed updates, exposures and published versions. Reader progress is
measurement-only and is never used to define observations. Occupancy comes from
actual simultaneous valid ways; every sample checks the event-based value
against a scan. Invalid-to-valid fill increases occupancy; replacement does not.
50/90/95/99/100% first attainment is recorded at the fill's retired count/cycle.
Fill types, replacements and useful line lifetimes are distinct.

Original counter/parser meanings are retained: `U/M0`, `(M0-M)/M0`, `U/Ipf`,
`U/(Ipf-Q)`, `U/(U+L)`, `P/NL`, IPC, cycles and L2 MPKI. Undefined ratios are NA.
Legacy useful may include a first RFO, while L2 demands/misses use original LOAD
scope. Supplementary unique-enqueue issue cohorts track first demand and retain
pending/resident-unused work as censored across windows. Unused is not a direct
cache-pollution measurement. Fullness is neither warmed working set nor deadline.

The predeclared descriptive stability rule is three consecutive 1M windows.
Positive cumulative cycle saving must persist at all three endpoints to report
startup recovered; positive window saving is separately named. Frozen attainment
means window IPC >=99% of same-H frozen-i1m for three consecutive windows.
Report the first qualifying endpoint and confirmation endpoint with coverage,
selected accuracy, timeliness, traffic, observations and updates. NOT REACHED is
an outcome; this is no statistical equivalence proof or adaptive controller.

The suite is 14 cold functional runs, six h8 finite-state/service runs, and four
frozen historical checks. Only random initialization has seeds 7,17,27; all
pretraining checkpoints are seed 7. At most two experiment jobs and one library
thread per worker are used. The recipe is locked after the development pilot
and before final results; final observations do not tune it. Weak learning is
reported rather than redesigned. Raw runs remain outside tracked source.
