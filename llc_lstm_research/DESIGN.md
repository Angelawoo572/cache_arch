# Proposed LLC LSTM

## What the LSTM could learn

- A recurring pattern can depend on an earlier event that looks irrelevant by the time the current load arrives. The cell state can retain that cue across intervening misses and update it at a phase change. Voyager gives a concrete case in which the same graph node has different successors depending on the earlier parent. Check for that pattern in the chosen trace by matching recent suffixes and looking at how the useful candidates differ with older context. [CMU 11-785 slides 85–98](https://deeplearning.cs.cmu.edu/S26/documents/slides/lec14.recurrent.pdf#page=85) · [Voyager §5.3](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf#page=10)
- RAOP combines an LSTM-predicted anchor with offset prefetching. TransforMAP uses a bitmap of future same-page offsets and points out that upper-cache filtering and translation can weaken LLC patterns. Twilight separates candidate generation from ranking. Use those ideas to keep the candidate set small and measure its reach before testing a recurrent scorer. [RAOP §§3–4](https://www.memsys.io/wp-content/uploads/2020/10/p374-zhang.pdf) · [TransforMAP §§II–III](https://arxiv.org/html/2205.14778#S2) · [Twilight §§III–IV](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)

## The data path

```text
L2D demand-load miss ──► shared LLC read queue ──► normal tag lookup ──► DRAM on miss
                                │
                                └─► features x[t] ──► LSTM cell ──► K candidate scores
                                                        ▲    │              │
                                         same core h,c │    └─► save h,c   ▼
                                         from last event            filter + threshold
                                                                            │
                                                    abstain ◄───────────────┴─► issue one
                                                                                  │
                                                     outcome logs ◄── use/merge ◄─ LLC fill

Offline only   NoPF future demand and lead time ──► binary labels ──► loss ──► weights
```

- **Input.** One event per newly admitted L2D demand-load miss before LLC tags. `x[t]` is the signed same-page delta from the previous event on that core (with a page-change token), the current line offset within the page, and a bucketed interarrival time in cycles. The physical line anchors the candidates. The core ID selects state. Add a same-PC bit later to identify repeated load instructions. A separate shared-context test appends current LLC MSHR occupancy at the head.
- **Cell.** One unidirectional layer, eight hidden units, shared weights and separate eight-element hidden and cell vectors per core. Let `z = [x[t], h[t−1]]`. Compute `i = sigmoid(W_i z + b_i)`, `f = sigmoid(W_f z + b_f)`, `o = sigmoid(W_o z + b_o)`, and `g = tanh(W_g z + b_g)`. Update `c[t] = f × c[t−1] + i × g` and `h[t] = o × tanh(c[t])`, with elementwise products. The input gate writes new cues, the forget gate controls retention, and the output gate exposes the state to the head. [D2L §10.1](https://d2l.ai/chapter_recurrent-modern/lstm.html)
- **Output.** A linear head reads `h[t]` and emits `K` logits, one per candidate. Each sigmoid score estimates that candidate's timely-miss opportunity. Start with `K = 3` and offsets `+1,+2,+4` on OMNeT++. If a fixed set has low reach, compare a bounded table of observed successor deltas that proposes up to three same-page lines. Keep candidate ordering deterministic. The same candidate source feeds the LSTM, MLP, and a conventional ranking rule. At issue, take the highest eligible score above threshold and issue at most one prefetch. [SPEC screen](conventional/SPEC_SCREEN.md)
- **Feedback.** The next event on the same core reads the updated `h,c`. Cache and memory outcomes arrive later and log timely first uses, late merges, unused evictions, and queue drops. Weights are trained offline in the proposed first comparison. An optional later throttle can use recent per-core outcomes. A recent hit fraction can use completed lookups only. The current demand's hit/miss becomes known after the prediction tap.

## Labels and history

- **One bit per candidate.** From NoPF, find the candidate's first later admitted demand using physical line and ingress order. Assign `1` if it LLC-misses and its ingress lead time is within the usable window. Assign `0` for a hit, lead time outside the window, or no demand in a complete window. Mask illegal, already resident or in-flight candidates when known, and examples without a complete observation. Store the raw lead time and outcome separately. Multiple candidates can be positive. Use masked binary cross-entropy.
- **Timing.** Target ingress is a conservative deadline. Set the lower bound using measured prediction-to-fill time and the upper bound using observed residency on development data. Freeze both for evaluation. A late merge can still reduce stalls, so measure it in the closed loop. Prefetching can change the timing and cache state that produced the offline labels. Compare the full policy in ChampSim with real issue and fill timing. Conventional actions can supply candidates or serve as an imitation control.
- **Window length.** Compare `W = 1, 4, 8, 16, 32` per-core events at fixed width 8. Start each exact window from zero state. Use the same chronological splits, eligible anchors, candidate list and issue budget. Report the cycle span of each window. The same-window MLP gets the same ordered inputs. Compare the effect of hiding older context on cases with a shared recent suffix. Consider width 16 only if wider capacity is needed after the window comparison.
- **Streaming deployment.** Evaluate carrying `h,c` with one update per event after the window study. A finite training unroll limits the gradient, while streaming state can persist longer. Measure score quality and simulation time again for that version. [TBPTT](https://proceedings.mlr.press/v115/aicher20a.html)

## Hardware boundaries

- **Virtual and physical addresses.** A 4 KiB page holds 64 lines of 64 bytes each. The LLC sees the translated data line, so a same-page offset stays inside the known frame. Crossing a page needs a known translated target or an explicit translation mechanism. Record when same-page filtering limits coverage. Reset or tag state on address-space changes and handle migration. ChampSim replay uses a fixed workload per core. [CMU 18-447 virtual memory, slide 26](https://course.ece.cmu.edu/~ece447/s15/lib/exe/fetch.php?media=onur-447-spring15-lecture20-virtual-memory-afterlecture.pdf#page=26)
- **Instruction and data traffic.** A load's PC identifies the instruction and the physical data line supplies the address. I-fetch, writeback, RFO, and prefetch traffic can compete for shared resources. The input is the admitted demand-data-load stream. Record the exclusions and their traffic separately.
- **Latency and throughput.** Charge feature extraction, time in the prediction queue, recurrent compute, candidate filtering, prefetch queue wait, DRAM service, and LLC fill. Separate end-to-end delay from initiation interval. An event from one core depends on its prior `h,c`, while events from other cores can use the engine in between. Use a bounded prediction queue so demand never waits for the model. Recheck eligibility at issue and record stale and dropped decisions.
- **Storage.** For three inputs, eight hidden units and three outputs, the cell and head use 376 matrix MACs per update and 411 coefficients including fused biases. At 8-bit weights and 32-bit biases that is 516 bytes shared. At 16 bits each for hidden and cell elements that is another 32 bytes/core. Count per-core feature state, candidate tables, FIFOs, tags, activation tables, and accumulators separately. An exact `W`-event replay costs `352W + 24` matrix MACs per decision and needs an event buffer. [Performance budget](PERFORMANCE.md)
