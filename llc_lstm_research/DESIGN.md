# LLC LSTM design

## What the trace tells us

- Start with `605.mcf_s-994B`. The conventional LLC prefetchers leave substantial misses. NoPF shows that a load PC from an earlier page visit can help choose a line in that page, even when the visit was 9–32 L2D misses ago. The 620 result was mainly a short-history cue. [Trace selection and measurements](conventional/LONG_MEMORY.md)
- On the new 70M region, the page-PC table reached 213 misses at 1,000 decisions. Adding eight ordered events reached 226. Sixteen and 32 ordered events reached 208 and 203. At 2,000 decisions the best shuffled eight-event result exceeded the ordered one. Keep the 32-event address directory and use eight events as the first sequence control. A need for longer recurrent state remains unproven. [70M history check](conventional/605_HISTORY.md)

## Proposed first model

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
- **Score.** If the history test supports recurrence, start with one causal layer and eight hidden units per core. Project the current input and `h[t]` once to a four-value query `q`. A candidate uses a small page key `k` derived from its source PC and line position, and an offset vector `a`. Compare raw scores `dot(q ⊙ a, k) + bias`, with age available as a small bias term. This lets state change which page and line win. Issue the highest eligible score above threshold. The matched nonrecurrent head has the same candidate fields and zeroed recurrent input.
- **Score resolution.** Keep one scalar per candidate. Compare fixed-point logits directly and check top-two ties, saturation, and winner changes after a state reset. The issuer needs a maximum and a threshold check. A sigmoid is used in the training loss only.
- **Label.** Use one bit per legal candidate from NoPF. A positive reaches that line's first later demand, which LLC-misses and falls in a calibrated lead window. Multiple candidates can be positive. Mask illegal and censored cases. Keep raw lead time and target ID in the logs. Training examples and scores come from NoPF demand outcomes. The screen covers previously unobserved lines in the measured region. The runtime issuer filters resident and in-flight lines.

## What would justify recurrence

- **Separate two types of memory.** The physical-page buffer preserves addresses. The LSTM must add value by preserving or updating useful cues across intervening loads. Give all scorers the same candidate buffer, labels, and issue budget. Compare the explicit page-PC table, a short-history scorer, and the same scorer with carried state. Test resets at 8, 16, and 32 events with the page buffer intact. Shuffling the page-to-PC association tests binding. Shuffling the older sequence while preserving page identity tests order.
- **Choose model size from the result.** A training unroll limits how far gradients travel. The streaming state can survive across chunks. Increase width or unroll only if the NoPF comparisons show benefit from older context after controlling for the page buffer. Confirm on a new trace region. Record the score changes and the extra misses reached, then charge storage, inference delay, and queue contention in ChampSim. Compare at matched actual requests or DRAM traffic. [Metrics and timing budget](PERFORMANCE.md)
- **Budget the head.** The probe can enumerate up to 320 choices per trigger. Measure reach under a smaller candidate cap before simulating inference. For illustration, with six input values, eight hidden units, a four-value query, and 16 candidates, the recurrent update, query and candidate dot products cost about 568 MACs per event, plus gates, shared offset transforms and directory reads. Those dimensions are provisional. Measure decision latency and storage. Keep the demand path independent of the model.
- **Status.** This is a design with measured controls. The page-buffer probe has run. The LSTM has yet to be implemented or trained.
