# 602 Stride: offline training and live inference

This sibling experiment keeps the completed
[602_offline_lstm_stride](../602_offline_lstm_stride/) study unchanged and
adds a new execution mode:

> The conventional Stride policy supplies labels before deployment. The LSTM
> is trained offline. ChampSim loads float32 weights once, keeps them frozen,
> and runs causal inference inside each measured L2 demand callback.

“Online” means only that inference executes in the live callback path. It does
not mean online learning: the live runtime has no gradient, optimizer, teacher,
future field, or action-list lookup.

## Scientific questions

- **h16 performance reference:** find the smallest offline prefix that
  preserves live system performance.
- **h8 compact reference:** find the smallest offline prefix that approaches
  the h16 live result with fewer weights and less recurrent state.

h8 and h16 remain separate learning curves. h8 has 1,908 parameters (7,632
float32 weight bytes); h16 has 5,220 parameters (20,880 bytes). h32/h64/h128
and alternative architectures are outside this fixed-model deployment study.

## K and learned actions

For one event, K is exactly the conventional teacher action-list length.
Thus [104, 106] means K=2. It is not history length, sample count, hidden size,
model capacity, or a fixed degree.

The student retains all three learned outputs:

1. issue or stay silent;
2. a positive slot count K after issuing;
3. one signed delta and reconstructed target address for every slot.

No probability threshold, degree cap, same-page filter, or normal Stride
private state is added at inference.

## Reused authority

The completed offline trainer remains the one authoritative Python
implementation of CompactPCKeyedHurdleStrideLSTM,
CompactDirectDeltaDecoder, feature encoding, hurdle/count decoding,
free-running delta feedback, label conversion, class balancing, and address
reconstruction. formal_NN_training/common/stride_direct_action_model.py
loads that module through a stable wrapper. This minimum-diff approach keeps
existing state_dict names and shapes intact.

The prefix trainer calls the existing CLI with a fresh seed-7 initialization
for every (H, budget). It never continues another budget and never borrows
20M class frequencies.

## Phases

1. Collect the 17 exact trace-start prefixes with warmup 0 and simulation N.
   One stream per N is shared by h8/h16.
2. Train/evaluate every valid h8/h16 point on the fixed held-out stream and
   create the existing keyed offline lists.
3. Validate h16 20M export/parity/live smoke, then repeat with h8 20M.
4. Run all valid keyed offline replays and recommend a deduplicated live set.
5. After manual approval, run selected full live points. All 34 are launched
   only with LIVE_ALL_VALID=1.

Tiny prefixes are diagnostic data. no_callbacks, single-class,
insufficient_rows, training failure, or all-silent collapse remain visible.
Undefined metrics remain NA.

## Frozen export and C++ runtime

model.bin format version 1 is little-endian float32 and contains all 16
state_dict tensors: input projection, all LSTM weights/biases, emit head,
positive-count head, all GRU weights/biases, and delta head.
model_metadata.json binds the binary to checkpoint, streams, Python source,
runtime encoder, model revision, seed, budget, and Git commit SHA.

The dependency-free C++11 runtime implements the PyTorch LSTM i,f,g,o and
GRU r,z,n equations, signed-log decoding, dynamic exact-PC (h,c) state,
58-bit line-address reconstruction, and per-callback host timing. Host
nanoseconds are measurements of the simulator host, not simulated CPU cycles.
The functional first version models zero NN latency.

## Sacramento compatibility

Host-side collection, validation, aggregation, export orchestration, and
Python/C++ parity are kept syntactically compatible with Python 3.6. The
synthetic parity fixture uses the legacy NumPy `RandomState` API, so it also
works with NumPy releases older than 1.17. None of these Sacramento stages
requires pandas. Colab training still requires PyTorch and NumPy, and plot
generation requires matplotlib.

## State boundary

The primary parity mode ignores all warmup callbacks, emits no warmup
prefetches, explicitly resets neural state when measurement begins, and starts
the measured region with an empty PC-state map. This matches the existing
held-out evaluation boundary.

Optional realistic_state_warmup advances frozen state during warmup but emits
nothing there. Its logs are named and aggregated separately; it cannot replace
or share a result row with parity mode.

## Generated outputs

Everything below runs/ is ignored, including prefix streams, event logs,
checkpoints, action lists, model.bin, archives, simulator binaries, CSV/JSON
aggregates, figures, and report inputs. Source, configuration, notebook,
validators, TeX, and this documentation are tracked.

See [COMMANDS.md](COMMANDS.md) for the complete Mac, Sacramento, Colab, SCP,
validation, analysis, and source-only commit workflow.
