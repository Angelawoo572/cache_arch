# LLC LSTM design

## Frozen direct-position model

- Use accepted L2D load misses at the shared LLC ingress. The conventional callback remains `access(physical_line, PC)`, with the core ID selecting its predictor. Newest also reads the arrival cycle. It builds a nine-value event vector from a four-value PC embedding, page position, same-page displacement, page change, PC continuity, and interarrival time. The LSTM updates its eight-value hidden and cell states per core as decisions launch. The two cores share the weights and the LLC.
- Score every eligible position in the current physical page and the most recently observed distinct page in a 32-event directory. The eight-unit head has separate weights for the 127 signed displacements. It retains the top 16 positions in Q10 order for the shared issue filter and can issue one request. It uses no fitted offset shortlist.
- Label each legal address using its first later same-core NoPF access. In the pure NoPF revision, a future LLC miss within 4,000 cycles receives the demand service CDF value at lead minus 32 modeled inference cycles. Everything else receives zero. The archived V5 timing CDF used older LSTM prefetch fills. Hit rate, miss rate and IPC measure the resulting policy.
- Train with eight-event truncated backpropagation, 32 burn-in events and up to 12 epochs. The 70M/130M development split selected epoch 8 and Q10 −950 for the pure NoPF revision, and epoch 3 and Q10 50 for archived V5. Deployment carries state across launches. The 32-cycle replay allows four in-flight decisions per core.
- The current pure-NoPF checkpoint won all three 410M/430M/450M regions against the fastest traditional policy per phase. The NoPF offset analysis shows a repeatable PC and same-page direction association on these traces. A single conditional association does not isolate the contribution of recurrent state. [Results](lstm/v5_results/clean32/pooled.csv) · [Offset association](lstm/v5_results/clean32/offset_correlation.csv) · [Report source and diagrams](report/llc_lstm_report.tex)

## Earlier model experiments

## What the trace tells us

- Start with `605.mcf_s-994B`. The conventional LLC prefetchers leave substantial misses. NoPF shows that a load PC from an earlier page visit can help choose a line in that page, even when the visit was 9–32 L2D misses ago. The 620 result was mainly a short-history cue. [Trace selection and measurements](conventional/LONG_MEMORY.md)
- On the new 70M region, the page-PC table reached 213 misses at 1,000 decisions. Adding eight ordered events reached 226. Sixteen and 32 ordered events reached 208 and 203. At 2,000 decisions the best shuffled eight-event result exceeded the ordered one. Keep the 32-event address directory and use eight events as the first sequence control. A need for longer recurrent state remains unproven. [70M history check](conventional/605_HISTORY.md)

## First model

```text
L2D demand-load miss at LLC ingress ──► normal LLC demand path
                   │
                   ├─► PC + line features ──► per-core LSTM ──► h[t]
                   └─► recently seen physical pages ───────────┬─┘
                                                               ▼
                                            score (page, line offset) pairs
                                                               │
                                              issue one or abstain
```

- **Input.** Use load PC and translated physical line from accepted L2D demand misses. Encode PC, within-page line offset, and relative page/line changes. Keep one causal `h,c` state per core and share weights. Update state on every event, including abstentions. NoPF supplies the offline event stream. The model sees the same fields at runtime as the comparison policies.
- **Address representation.** Keep a bounded buffer of already translated physical pages with the line offset, load PC, and event index of each page's most recent visit. Query the preceding 32 same-core events before inserting the current event. Each candidate points to a known page and a line within it. The same bounded set of within-page displacements gives every scorer the same choices.
- **Score.** One causal layer has eight hidden units per core. Project the seven current input features and `h[t]` to a four-value query `q`. Project each candidate's six features to a four-value key `k` and a scalar bias `b`. The logit is `dot(q, k) + b`. The candidate features encode its source PC, line position, displacement, age, and whether the current PC matches. The runtime issues the highest eligible score above threshold. The separately trained event-only control uses the same candidate fields and a zero recurrent input.
- **Score resolution.** Keep one scalar per candidate. Compare fixed-point logits directly and check top-two ties, saturation, and winner changes after a state reset. The issuer needs a maximum and a threshold check. A sigmoid is used in the training loss only.
- **Label.** Use one bit per legal candidate from NoPF. A positive reaches that line's first later same-core demand, which LLC-misses 300–2,000 cycles after the trigger. Multiple candidates can be positive. Mask illegal and censored cases. Keep raw lead time and target ID in the dataset. The candidate directory skips the current page and lines seen in its last 32 events. The runtime issuer filters resident and in-flight lines.

## What would justify recurrence

- **Separate two types of memory.** The physical-page buffer preserves addresses. The LSTM must add value by preserving or updating useful cues across intervening loads. Give all scorers the same candidate buffer, labels, and issue budget. Compare the explicit page-PC table, a short-history scorer, and the same scorer with carried state. Test resets at 8, 16, and 32 events with the page buffer intact. Shuffling the page-to-PC association tests binding. Shuffling the older sequence while preserving page identity tests order.
- **Choose model size from the result.** A training unroll limits how far gradients travel. The streaming state can survive across chunks. Increase width or unroll only if the NoPF comparisons show benefit from older context after controlling for the page buffer. Confirm on a new trace region. Record the score changes and the extra misses reached, then charge storage, inference delay, and queue contention in ChampSim. Compare at matched actual requests or DRAM traffic. [Metrics and timing budget](PERFORMANCE.md)
- **Budget the head.** The 16-candidate LSTM costs up to 1,084 matrix MACs per event plus gate operations and directory lookups. Its 611 float32 coefficients take 2,444 bytes, and `h,c` take 64 bytes per core. The 32-entry directory, prediction queue, and address filters add storage and lookup costs. The model runs off the demand path. The first replay models 16- and 64-cycle inference without RTL timing or area estimates.
- **Result.** The eight-event training unroll won on validation. With the same weights, removing all state hurt held-out ranking. Resetting state to eight events or disturbing their older order changed little. The separate event-only model beat the LSTM on both held-out regions. [Full results](lstm/RESULTS.md)
