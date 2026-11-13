# LLC LSTM design

## What the trace tells us

- Start with `605.mcf_s-994B`. The conventional LLC prefetchers leave substantial misses. NoPF shows that a load PC from an earlier page visit can help choose a line in that page, even when the visit was 9–32 L2D misses ago. The 620 result was mainly a short-history cue. [Trace selection and measurements](conventional/LONG_MEMORY.md)
- The strongest MCF result comes from a small table that binds each earlier PC to its page. Additional ordered visits to that page did not improve a separate lookup. The evidence supports keeping old context available. The value of an LSTM is still a hypothesis.

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
- **Scorer.** Start with one unidirectional LSTM layer and 8 hidden units per core as a provisional budget. A shared head reads `h[t]`, the current input, and each candidate's source PC, age, and line position. It returns one sigmoid score per legal candidate. The issuer takes the best eligible score if it clears the threshold. The main nonrecurrent control receives the same fields and page buffer. A current-only control measures how much the stored source PC contributes.
- **Label.** Use one bit per legal candidate, from NoPF. A positive reaches that line's first later demand, which LLC-misses and falls in a calibrated lead window. Mask illegal and censored cases. Keep raw lead time and target ID. The screen uses previously unobserved lines in the measured region. A deployment experiment should apply the actual resident and in-flight filters and test previously seen lines separately. Use no conventional decisions, confidence, or outcomes to generate training examples or tune scores.

## What would justify recurrence

- **Separate two types of memory.** The physical-page buffer preserves addresses. The LSTM must add value by preserving or updating useful cues across intervening loads. Give all scorers the same candidate buffer, labels, and issue budget. Compare the explicit page-PC table, a short-history scorer, and the same scorer with carried state. Test resets at 8, 16, and 32 events with the page buffer intact. Shuffling the page-to-PC association tests binding. Shuffling the older sequence while preserving page identity tests order.
- **Choose model size from the result.** A training unroll limits how far gradients travel. The streaming state can survive across chunks. Increase width or unroll only if the NoPF comparisons show benefit from older context after controlling for the page buffer. Confirm on a new trace region. Record the score changes and the extra misses reached, then charge storage, inference delay, and queue contention in ChampSim. Compare at matched actual requests or DRAM traffic. [Metrics and timing budget](PERFORMANCE.md)
- **Budget the head.** The probe can enumerate up to 320 choices per trigger. Measure reach under a smaller candidate cap before simulating inference. Charge directory storage and tag checks along with LSTM state and scoring. Keep the demand path independent of the model.
- **Status.** This is a design with measured controls. The page-buffer probe has run. The LSTM has yet to be implemented or trained.
