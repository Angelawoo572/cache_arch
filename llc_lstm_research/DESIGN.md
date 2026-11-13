# LLC LSTM design

## What 620 suggests

- **Past context can change the right decision.** In 620 NoPF, two loads can have the same recent PCs and `+1,+1` deltas. An earlier `+2` was followed by a reachable `+2` miss in 15/15 cases at 30M. An earlier page change gave 0/24. Streamer and SPP were both late on 14 of the 15. The difference in the NoPF outcomes is the signal to learn. [620 history check](conventional/620_HISTORY.md)
- **Short history is the control to beat.** At 1,000 decisions, the NoPF-only lookup reached 453 distinct future misses with one event and 470 with four. Sorting the older events also reached 470. At 2,000 decisions, the two-event lookup reached 800 versus 777 with one. These counts show a modest benefit from context. They give little evidence that long ordered history is useful on 620. [620 history check](conventional/620_HISTORY.md)
- **The recurrent hypothesis.** A small LSTM could learn to preserve an earlier cue through unrelated loads and apply it to unfamiliar combinations of PCs and deltas. The known three-event pattern can already be handled by a short rule. Test whether state adds value when the relevant cue appears at a variable distance in the history.

## Where the data comes from

- **Inputs and labels come from NoPF.** Use the demand stream from a run with all prefetchers disabled. The input is every accepted L2D demand-load miss at LLC ingress, before the current LLC tag result. Use only the load PC and physical data line. The core selects its own history.
- **One bit per offset.** In the same NoPF run, find the candidate line's first later demand. Assign `1` when that demand LLC-misses and its ingress lead falls within the chosen timing window. Assign `0` for a hit, an out-of-window demand, or no demand in a complete window. Mask illegal and censored cases. Keep raw lead cycles and choose candidates and timing bounds on development NoPF data. Use the same bounds across the offline controls.
- **Conventional logs are diagnostics.** Stride issues infrequently, Streamer and SPP often arrive late, and many AMPM fills go unused during the measured region. Those failures motivate questions about coverage, earlier cues, and abstaining. Conventional actions and late flags stay out of training examples, targets, weighting, threshold tuning, and model selection. [620 baseline results](conventional/620_RESULTS.md)

## What the model would see and do

```text
L2D demand-load miss admitted to the LLC read queue
  ├─► Demand continues to the LLC tags
  └─► Same load PC and physical line that the conventional policies receive
         │
         ▼
       LSTM ──► adjust current-event scores ──► issue one or abstain
        ▲ │                                              │
        └─┴── carry this core's h,c to the next event     └─► log use and timing
```

- **Features.** Encode the physical line as page offset and signed delta with a page-change token. Use a bounded token for load PC. Give each matched control the same encoding. Cycle gaps, MSHR occupancy, and LLC outcomes stay out of the first predictor input.
- **State.** Start with one causal layer and a small `h,c` state per core, with weights shared across cores. Update once per admitted event, including when the predictor abstains. The gates can preserve a cue or overwrite it as the access pattern changes. Eight units is an initial hardware budget. [CMU 11-785, LSTM gates](https://deeplearning.cs.cmu.edu/S26/documents/slides/lec14.recurrent.pdf#page=85)
- **Output.** Let the current event provide a score per offset and let the recurrent state adjust those scores. This gives a direct check of when history changes the decision. Use the same NoPF-selected candidates for the LSTM and matched controls. Issue the best eligible candidate if it clears the threshold, or abstain.

## How to find the value of memory

- **Start with the short controls.** Compare the current-event scorer, a two- to four-event rule or MLP, and carried LSTM state. On a new NoPF region, hold the current load and recent suffix fixed and group older cues by how many events back they occur. Test whether the score still changes when the cue sits farther back. Reset state or shuffle the older prefix to check that the gain depends on that context. A small ungated RNN can show whether any gain specifically needs the gates.
- **Separate trace comparison from simulation.** Use the same held-out NoPF events and candidates in the offline comparison. In ChampSim, each policy observes the requests that reach its LLC as it runs, so its own prefetches can change later timing. Use the same trace regions, warmup, input tap, filters, and degree cap. Compare at similar actual issue counts or DRAM traffic and charge inference delay and storage. Add shared MSHR occupancy only in a separate multicore experiment with that signal available to the matched controls. [Metrics and timing budget](PERFORMANCE.md)
- **Hardware boundary.** Target a data line inside the translated physical page. Load PC is instruction metadata. I-fetch and other traffic still share cache and memory resources. Count state, PC encoding, weights, candidate logic, and queues in storage. Model the dependency on the prior update from the same core. [CMU 18-447 virtual memory](https://course.ece.cmu.edu/~ece447/s15/lib/exe/fetch.php?media=onur-447-spring15-lecture20-virtual-memory-afterlecture.pdf#page=26)
