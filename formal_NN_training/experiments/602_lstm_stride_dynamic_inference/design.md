# Offline Training and Online Inference

## Question and scope

How does an offline-trained tiny Stride neural prefetcher behave during live,
causal inference when weights are fixed but recurrent history and predictions
change? What inference storage, service work and queueing accompany usefulness?
This branch deploys the existing seed-7 h8/h16 i1m/i20m checkpoints. It retains
the summer offline training/Colab code and the frozen-live model architecture.
There is no online optimizer, backward pass, runtime teacher, training buffer,
weight publication, scratch initialization or adaptation arm in this experiment.
No reinforcement learning is used. Offline model preparation is separate from
frozen deployment and its storage.

The model access wrapper and original compact PC-keyed forward remain
authoritative. PC and aligned byte address, 64 bits each, are the only neural
features. The learned hurdle, learned positive K, signed direct-delta decoder
and free-running predicted decoder feedback remain unchanged. H8 is the small
primary design; h16 is the capacity comparison. C++ reuses the frozen forward.

## Dynamic causal inference

An eligible L2 LOAD enters a 16-event FIFO if space exists. The serial inference
engine uses fixed weights and that exact PC's last completed h/c. It computes a
result at service start but exposes h/c and addresses only at scheduled
completion. Delayed same-PC requests cannot see future recurrent states.
Predictions therefore evolve from observed accesses even though parameters do
not change. Each PC carries h/c continuously through its completed inference
events; there is no fixed-window W or zero-state reconstruction of the last W
accesses. Statistics sampling never changes h/c or inference admission. No runtime labels or teacher state enter prediction.

A 64-entry exact-PC table uses LRU replacement on inference starts. Each entry
has its own h/c; eviction discards that history. An unbounded control retains
the original routing behavior. A 32-address output FIFO uses the normal
prefetch_line path, including the original merging/drop behavior. Full queues
drop newest work, and input/output/decoder drops and numerical failures remain
visible. The cold decoder output resource permits 32 addresses, not a silent
clamp to Stride's degree. Historical compatibility preserves the original
uncapped path.

Conventional Stride is a distinct baseline. Its actual 64 trackers, degree two,
current-line first requests [current, current + stride], page restrictions and
zero-stride behavior are preserved. There is no shadow Stride in a neural arm.
A core-cycle hook completes inference and issues pending requests even during
stalls and without a new demand callback. Host time is not modeled CPU delay.

## Matched protocols and retained data

Historical compatibility uses 25M no-prefetch cache warmup and 25M measurement,
zero modeled inference cost, unbounded state and empty predictor state at the
measurement boundary. Original retirement-group overshoots are retained:
25,000,004 warmup instructions and 25,000,003 measured instructions. The old
conventional live Stride run issued throughout warmup and is not an equivalent
matched reference. The four preserved frozen historical runs match all 30
original parser fields each.

Cold runs skip exactly 25M opaque records using the reader's compiled format,
without simulating skipped instructions or warming caches. They execute the
same next 25M records with fresh caches and h/c, loading only the pretrained
weights. This is cold start at the held-out trace-slice boundary, not program
startup. Realized live callback timing/order may change with prefetching.
Processor/cache/replacement/memory/request paths and input eligibility stay
matched within a protocol; old warm-cache IPC is not a cold reference.

The retained matrix has 13 compatible completed runs: NoPF and Stride;
four cold frozen h8/h16 i1m/i20m; h8-i1m with 64 entries at zero cost and at
4/16 MACs per cycle; four historical frozen compatibility checks. Existing
frozen 4/16-MAC negative results are retained. The original run identifiers,
raw source paths and trace origin remain unchanged. Filtering uses actual run
configuration; weight-changing results cannot become frozen rows by renaming.
The current refactor is validated by a small regression, not presented as a
new execution of all preserved scientific runs. Full raw data remain untouched; branch retirement is handled separately from
the scientific filtering. New runs, if requested, use their own raw tree.

## Inference service and storage assumptions

The unbounded zero-cost control is algorithmic, not a hardware result. Finite
cases use 4 or 16 MACs/cycle, 16 or 64 bytes/cycle weight and scratch bandwidth,
and four cycles per nonlinear operation. Dense/nonlinear/memory/control phases
serialize; the initiation interval equals the data-dependent service duration.
For hidden H, learned emission e, and actual decoded count K:

- MACs: `128H + 8H^2 + 2H + eH + K*(3H^2+4H)`.
- Nonlinears: `6H + e + K*(3H+1)`.
- Control: `8 + ceil(entries/4) + 4K` cycles.
- Additional scratch traffic: `4*(128+4H) + 24K` bytes.

The finite output engine issues at most one address per cycle. These costs
are architectural sensitivity assumptions, not measured silicon timing, power,
area or synthesized frequency. There is no simulated training service.

FP32 weights occupy 7,632 bytes (h8, 1,908 parameters) or 20,880 bytes (h16,
5,220 parameters). Each recurrent table entry reserves 32 bytes tag/metadata
plus `8H` bytes for h/c. Active event, arithmetic scratch, staged decoded outputs,
and input/output queues are additional deployment storage. One active weight
copy suffices. The frozen refactor reduces active-event metadata from 56 to 24
bytes and the input queue from 16x56 to 16x24 bytes. Its configured h8/64-entry
layout is 16,808 bytes, versus 17,352 bytes in the retained measured layout.
The 544-byte reduction is a source-derived layout change; original observed
peaks and scientific counters are not retroactively rewritten. Conventional Stride tracker storage belongs only to that baseline.

Configured maxima and observed component peaks are separate. Summed peaks are
an envelope, not a simultaneous allocation measurement. An unbounded state
configuration has no finite maximum. Host allocator/RSS and measurement logs
are separate from predictor storage. No offline training memory is charged to
frozen deployment. Predictor SRAM is compared in size to L2, not asserted to
occupy or pollute it. No hypothetical INT8 storage is reported as measured.

## Measurement meanings and regression

Retired target instructions from the trace-slice origin are the horizontal axis.
Eligible callbacks, admitted/started/completed decisions, queues and drops are
separate counters. Reader read-ahead is never predictor observation. Samples at
0, 1k, 10k and every 100k instructions retain startup costs. One-million-
instruction windows show dynamic performance/quality; cumulative cycles saved
use matched Stride. There are no learning or optimizer-update milestones.

L2 geometry is read from the active build: 512 sets x 8 ways x 64 bytes =
4,096 lines = 256 KiB. Occupancy counts simultaneous valid ways and is checked
against a scan. Invalid-to-valid fills increase occupancy; valid-to-valid
replacement does not. Exact first attainment of 50/90/95/99/100% occupancy has
instruction/cycle counters. Demand, RFO, prefetch and writeback fills,
replacements and useful line lifetimes remain distinct. Fullness is not a
warmed-working-set proof or deadline for prefetch usefulness.

The original parser/formulas remain: useful-prefetch coverage `U/M0`, miss
reduction `(M0-M)/M0`, raw accuracy `U/Ipf`, selected accuracy `U/(Ipf-Q)`, legacy
timeliness `U/(U+L)`, request pressure `P/NL`, IPC, cycles and L2 MPKI. Undefined
ratios are NA. Legacy useful may include a first RFO while NL/M keep LOAD scope.
Issue-cohort/first-demand outcomes supplement interval counter deltas; pending
and resident-unused requests remain censored across boundaries. Unused and
one-minus-accuracy are not direct pollution measurements.

Small regression checks cover changing predictions/h/c with fixed parameter
values, no training process/update path, authoritative frozen Python/C++ parity,
finite state eviction and delayed completion, plus existing occupancy and
metric fixtures. The analyzer rejects non-frozen configurations and retains
only genuine compatible frozen/baseline data. There is no audit framework or
SHA inventory. Commands are in command.md and the scientific report is
602_stride_dynamic_inference.tex/PDF.

The paired Sacramento regression uses skip=20M, length=100k, h8-i1m, 64 state
entries and MAC4. The refactored and preserved frozen binaries match all 30
original parser fields: 325,790 cycles, IPC 0.306946, 1,633 L2 LOADs, 410
requests, 2 useful and 0 late. Six snapshot/cohort rows preserve cache/service
counters aside from explicit zero initialization and the smaller event/queue
layout. Host time 2.96s versus 3.07s is not modeled timing improvement. The
pilot remains software regression evidence, outside final scientific rows.

The report additionally shows existing cumulative snapshots at 100k, 1M and
25M retired instructions for h8-i1m zero-cost inference, with matched same-
instruction NoPF denominators. Those descriptive sample points are not
minimum-history requirements, optimized observation budgets, triggers or new
measurements. They leave the continuous recurrent history untouched.

## Actual input history and strict quality comparisons

Each inference consumes one new eligible L2 LOAD: 64 PC bits and 64 aligned-byte
address bits, shaped `(1,1,128)` in the authoritative Python forward. Continuous
exact-PC h/c each have shape `(1,1,H)`. There is no fixed observation window,
buffering requirement to accumulate history, reset schedule, or model switch.
A measurement-only counter records successful completed h/c updates within each
state lifetime; queued/dropped events do not increment it. This describes history
experienced, not remembered content or a causal minimum-history requirement.

Supplementary runs reuse the original ListReplayer and original action tapes.
The original summer v9 seed-7 h8/h16 models and replay streams are byte-identical
to the corresponding i20m prefix-sweep artifacts. Summer means this fixed same-H
i20m reference. Each checkpoint's own offline replay is a separate comparison.
Historical live Stride starts empty after the same no-prefetch warmup; original
Stride replay is retained separately. Cold/summer comparisons are not mixed.

At exact existing cumulative snapshot points and existing 100k/1M counter
windows, all three of coverage, selected accuracy and legacy timeliness must be
strictly greater than one fixed reference. Integer fractions distinguish ties;
zero denominators are NA. Both-reference claims require the same NN run and
same counter interval against each reference. CSVs retain all metric differences,
raw denominators, maximal satisfying intervals, exits, N, eligible/admitted/
completed D, IPC, cycle savings, request pressure, occupancy and storage.

The first recorded strict point is descriptive, not an exact minimum history.
No extra stability threshold is introduced. Issue-cohort follow-up remains
separate from legacy counter-window ratios; unresolved requests are censored.
Measurement-only arrival/start/completion timestamps expose actual modeled
queue waiting and completion delay. Their storage and first-1,024-decision
h/c logs are excluded from required inference deployment SRAM.
