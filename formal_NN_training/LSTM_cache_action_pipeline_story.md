# LSTM Cache-Action Predictor Pipeline Story

This note explains the current LSTM-based cache-action learning flow as a story. It is meant to be easier to read than the main README and to match the diagrams used when explaining the project.

## 0. One-sentence summary

```text
SPP proposes candidate prefetches; the LSTM reads each candidate plus cache context over time and learns whether to keep, suppress, bypass, or deprioritize that candidate.
```

This is **not** direct next-address prediction as the final story. The useful framing is:

```text
SPP = candidate generator + feature/context provider + supervision source
LSTM = sequential cache-action learner / candidate utility selector
Replay = system-level validation in ChampSim
```

## 1. End-to-end project flow

```mermaid
flowchart LR
    A[ChampSim + SPP_dev trace dump] --> B[lstm_events_TRACE.csv]

    B --> C[Load CSV in notebook]
    C --> D[Clean / select useful columns]
    D --> E[Build sequential training examples]

    E --> F[Features per SPP candidate event]
    F --> F1[PC / IP signal]
    F --> F2[Delta / SPP candidate delta]
    F --> F3[SPP confidence]
    F --> F4[Cache hit / miss context]
    F --> F5[MSHR occupancy]
    F --> F6[L2 occupancy / bandwidth pressure]
    F --> F7[Issued / fill_l2 / semantic class if present]

    E --> G[Labels / targets]
    G --> G1[Good: useful prefetch?]
    G --> G2[Duplicate: already useless or redundant?]
    G --> G3[Bypass: should avoid cache pollution?]
    G --> G4[Timing: early / on-time / late bucket]

    F --> H[LSTM cache-action model]
    G --> H

    H --> I[Multi-task outputs]
    I --> I1[Good probability]
    I --> I2[Duplicate probability]
    I --> I3[Bypass probability]
    I --> I4[Timing prediction]

    I --> J[Training loop]
    J --> K[Validation threshold sweep]

    K --> L[Selected good threshold]
    L --> M[Export full_lstm_cache_actions.csv]

    M --> N[Convert to list_replayer prefetch list]
    N --> O[Replay in ChampSim]
    O --> P[Compare against baseline SPP / no-prefetch]

    P --> Q[Metrics]
    Q --> Q1[Useful prefetch rate]
    Q --> Q2[Precision / recall / F1]
    Q --> Q3[Duplicate rate]
    Q --> Q4[Emit count]
    Q --> Q5[IPC / performance impact]
```

## 2. What one row means

Each row in `lstm_events_TRACE.csv` represents one SPP-related candidate event. It is not just an address. It includes the current demand context, the candidate prefetch address, and outcome labels.

```mermaid
flowchart TD
    A[lstm_events_602.gcc_s-734B.csv or other trace] --> B[Each row = one SPP-related event / candidate]

    B --> C[Raw columns]
    C --> C1[trace / event_id / replay_access_idx / cycle]
    C --> C2[ip / pc]
    C --> C3[addr]
    C --> C4[pf_addr]
    C --> C5[delta]
    C --> C6[spp_conf / spp_confidence]
    C --> C7[hit / is_store]
    C --> C8[mshr_occupancy / mshr_size]
    C --> C9[l2_occupancy / bandwidth_pressure]
    C --> C10[spp_issued / fill_l2]
    C --> C11[outcome_useful]
    C --> C12[outcome_duplicate]

    C --> D[Feature engineering]
    D --> D1[Hash or encode PC/IP]
    D --> D2[Quantize / bucket delta]
    D --> D3[Normalize numeric pressure features]
    D --> D4[Keep SPP confidence as signal]
    D --> D5[Keep chronological event order]

    D --> E[Build sequence]
    E --> E1[Sequence = chronological SPP candidate events]
    E --> E2[No sliding-window as the core story]
    E --> E3[LSTM hidden state carries history forward]
    E --> E4[At each event, model updates h_t and c_t]

    E --> F[Target at same event]
    F --> F1[good label = useful and non-duplicate candidate]
    F --> F2[duplicate label = repeated / already covered]
    F --> F3[bypass label = avoid inserting / avoid pollution]
    F --> F4[timing label = when this prefetch would be useful]

    F --> G[Batch for training]
    G --> G1[X sequence tensor]
    G --> G2[y_good]
    G --> G3[y_duplicate]
    G --> G4[y_bypass]
    G --> G5[y_timing]
```

The replay-critical field is `replay_access_idx`, not `event_id`. The prefetch list must use:

```text
replay_access_idx 0xprefetch_byte_addr
```

This matters because `list_replayer` advances an internal counter only on L2 demand accesses, and it parses the second column as hexadecimal.

## 3. Model architecture

At every time step `t`, the model reads one SPP candidate event.

```mermaid
flowchart LR
    A[One SPP candidate event at time t] --> B[Feature vector x_t]

    B --> B1[PC/IP embedding]
    B --> B2[Delta embedding]
    B --> B3[SPP confidence numeric]
    B --> B4[Cache pressure features]
    B --> B5[Hit/miss + issued/fill_l2 signals]

    B1 --> C[Concatenate]
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C

    C --> D[LSTM cell]

    D --> D1[Forget gate f_t]
    D --> D2[Input gate i_t]
    D --> D3[Candidate memory g_t]
    D --> D4[Output gate o_t]

    D1 --> E[Cell memory c_t]
    D2 --> E
    D3 --> E
    E --> F[Hidden state h_t]
    D4 --> F

    F --> G[Shared representation for current candidate]

    G --> H1[Good head]
    G --> H2[Duplicate head]
    G --> H3[Bypass head]
    G --> H4[Timing head]

    H1 --> I1[p_good = should keep / issue candidate]
    H2 --> I2[p_duplicate = likely redundant]
    H3 --> I3[p_bypass = should avoid pollution]
    H4 --> I4[timing bucket / usefulness timing]
```

## 4. What the LSTM gates mean for cache behavior

```mermaid
flowchart TD
    A[Current candidate x_t] --> B[LSTM asks 4 questions]

    B --> C[Forget gate]
    C --> C1[Old cache pattern still important?]
    C1 --> C2[Example: previous stride phase ended, forget it]

    B --> D[Input gate]
    D --> D1[Current event worth writing into memory?]
    D1 --> D2[Example: new PC starts repeating useful deltas]

    B --> E[Candidate memory]
    E --> E1[If writing, what pattern should be written?]
    E1 --> E2[Example: this PC + delta + confidence suggests useful prefetch]

    B --> F[Output gate]
    F --> F1[How much memory should affect current decision?]
    F1 --> F2[Example: use hidden state to judge good / duplicate / bypass]

    C --> G[c_t long-term memory]
    D --> G
    E --> G
    G --> H[h_t current decision state]
    F --> H
```

In cache terms, the LSTM is learning phase-sensitive behavior:

```text
same PC + same delta may be useful in one phase but harmful in another phase
cache pressure changes the value of issuing a candidate
recent duplicate/useful history changes whether a new candidate should be trusted
```

## 5. Training and export

```mermaid
flowchart TD
    A[Train split] --> B[Forward pass]
    B --> C[Predictions]
    C --> C1[p_good]
    C --> C2[p_duplicate]
    C --> C3[p_bypass]
    C --> C4[p_timing]

    C --> D[Loss]
    D --> D1[loss_good]
    D --> D2[loss_duplicate]
    D --> D3[loss_bypass]
    D --> D4[loss_timing]

    D1 --> E[total loss]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[Backprop through time]
    F --> G[Update embeddings + LSTM + heads]

    G --> H[Validation]
    H --> I[Threshold sweep for good head]

    I --> J[selected_good_threshold]
    J --> K[Choose threshold that balances precision / recall / emit count]

    K --> L[Export table]
    L --> L1[full_lstm_cache_actions.csv]
    L1 --> L2[For each candidate: keep / suppress / bypass / timing action]

    L2 --> M[Convert to list_replayer list]
    M --> N[Replay]
    N --> O[Compare]

    O --> O1[SPP baseline]
    O --> O2[LSTM-gated SPP]
    O --> O3[No-prefetch baseline]

    O1 --> P[Final metrics]
    O2 --> P
    O3 --> P

    P --> P1[good_precision]
    P --> P2[good_recall]
    P --> P3[good_f1]
    P --> P4[emit count]
    P --> P5[duplicate_rate]
    P --> P6[useful / useless]
    P --> P7[IPC change]
```

## 6. Runtime decision logic

```mermaid
flowchart LR
    A[SPP proposes candidate prefetch] --> B[LSTM reads candidate context]

    B --> C{Is p_good high enough?}

    C -- No --> D[Suppress candidate]
    D --> D1[Goal: reduce useless prefetches]
    D --> D2[Expected: lower emit count]
    D --> D3[Expected: lower duplicate / pollution]

    C -- Yes --> E{Is duplicate high?}

    E -- Yes --> F[Drop or deprioritize]
    F --> F1[Goal: avoid repeated same-line prefetch]

    E -- No --> G{Bypass predicted?}

    G -- Yes --> H[Bypass / avoid cache insertion]
    H --> H1[Goal: reduce cache pollution]

    G -- No --> I[Issue / keep SPP candidate]

    I --> J{Timing prediction}
    J -- Useful soon --> K[Normal prefetch]
    J -- Too late / too early --> L[Adjust priority or suppress if policy supports it]

    K --> M[Replay outcome]
    L --> M
    H --> M
    F --> M
    D --> M

    M --> N[Measure: useful, duplicate, hit rate, IPC]
```

Current implementation is replay-based, not yet an online hardware deployment. That means the IPC result validates the learned policy, but it does not include online neural inference latency.

## 7. Baseline SPP vs LSTM-gated SPP

```mermaid
flowchart TD
    A[Baseline SPP] --> A1[Many candidates]
    A1 --> A2[Some useful]
    A1 --> A3[Many duplicate / useless candidates]
    A1 --> A4[Possible pollution]

    B[LSTM-gated SPP] --> B1[Fewer but cleaner candidates]
    B1 --> B2[Higher offline good-candidate precision]
    B1 --> B3[Lower duplicate rate]
    B1 --> B4[Less pollution if threshold is right]
    B1 --> B5[Potential IPC improvement]

    C[Metrics expectation] --> C1[Emit count may decrease]
    C --> C2[Offline good_precision should increase]
    C --> C3[duplicate_rate should decrease]
    C --> C4[good_recall may drop if threshold too strict]
    C --> C5[IPC improves only if useful prefetches are preserved]
    C --> C6[ChampSim accuracy can still be lower than SPP]
```

Important distinction:

```text
Offline candidate-selection accuracy:
  LSTM can be much better than raw SPP candidate emission.

True ChampSim prefetch accuracy and IPC:
  SPP is still stronger on 602.gcc_s-734B.
```

## 8. Current 602.gcc_s-734B checkpoint

Warmup / simulation:

```text
25M warmup / 25M simulation
```

Fixed replay requirements:

```text
Use replay_access_idx, not event_id/cycle.
Write prefetch addresses as hex, not decimal.
Use L2 list_replayer binary.
```

Observed fixed replay results:

| Method | IPC | Speedup vs no-prefetch | L2 useful | L2 useless | Interpretation |
|---|---:|---:|---:|---:|---|
| no-prefetch | 0.5427 | 1.0000x | N/A | N/A | baseline |
| SPP baseline | 1.4440 | 2.6608x | 140,717 | 652 | strongest current baseline |
| LSTM fixed replay th0.20 | 0.7175 | 1.3221x | 64,473 | 48,530 | best observed LSTM result so far |
| LSTM fixed replay th0.25 | 0.7172 | 1.3215x | 64,420 | 48,533 | essentially tied with th0.20 |
| LSTM fixed replay th0.35 | 0.6371 | 1.1739x | 16,348 | 70,171 | lower recall / less effective |
| LSTM fixed replay th0.40 | 0.5521 | 1.0173x | 3,933 | 4,142 | too conservative |

Current conclusion:

```text
The fixed LSTM replay is valid and positive: it improves gcc IPC by about 32% over no-prefetch.
However, SPP still wins in both IPC and true ChampSim prefetch accuracy.
```

## 9. Final overview

```mermaid
flowchart LR
    subgraph S1[1. Trace generation]
        A1[Run ChampSim with SPP_dev]
        A2[Dump SPP candidate events]
        A3[lstm_events_TRACE.csv]
        A1 --> A2 --> A3
    end

    subgraph S2[2. Dataset construction]
        B1[Read chronological events]
        B2[Encode PC/IP, delta, confidence]
        B3[Add cache pressure context]
        B4[Create labels: good, duplicate, bypass, timing]
        B1 --> B2 --> B3 --> B4
    end

    subgraph S3[3. LSTM learner]
        C1[x_t candidate feature]
        C2[LSTM memory: h_t, c_t]
        C3[Gate old/new cache patterns]
        C4[Multi-task prediction heads]
        C1 --> C2 --> C3 --> C4
    end

    subgraph S4[4. Training]
        D1[loss_good]
        D2[loss_duplicate]
        D3[loss_bypass]
        D4[loss_timing]
        D5[Backprop through time]
        D1 --> D5
        D2 --> D5
        D3 --> D5
        D4 --> D5
    end

    subgraph S5[5. Policy export]
        E1[Validation threshold sweep]
        E2[selected_good_threshold]
        E3[full_lstm_cache_actions.csv]
        E4[list_replayer prefetch list]
        E1 --> E2 --> E3 --> E4
    end

    subgraph S6[6. Replay + evaluation]
        F1[Replay LSTM actions]
        F2[Compare with SPP / no-prefetch]
        F3[Metrics: useful, accuracy, precision, recall, duplicate, IPC]
        F1 --> F2 --> F3
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

## 10. What to say in a meeting

A concise version:

```text
I used SPP as a candidate generator and trained a stateful LSTM to learn which SPP candidates are useful, duplicate, bypass-worthy, or timing-sensitive. The key bug was in replay alignment: the model output originally used event_id/cycle and decimal addresses, but ChampSim list_replayer needs L2 demand-access indices and hexadecimal byte addresses. After fixing that, the LSTM replay became valid: on gcc, it improves IPC from 0.5427 to 0.7175, about 1.32x over no-prefetch. SPP is still much stronger at 1.444 IPC, so the current result is not beating SPP yet, but it proves the learned action policy can produce useful prefetches and real IPC gain once replay is correct.
```
