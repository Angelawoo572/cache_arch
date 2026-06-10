# LSTM Cache-Action Predictor Pipeline Story

This note explains the current LSTM-based cache-action learning flow as a story. It is meant to be easier to read than the main README and to match the diagrams used when explaining the project.

For exact numbers and the SPP-vs-LSTM metric table, use:

```text
formal_NN_training/README_LSTM_cache_action_predictor.md
```

## 0. One-sentence summary

```text
SPP proposes many candidate prefetches; the LSTM reads each candidate plus cache context over time and learns a cleaner keep/drop policy.
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

    B --> C[Load CSV in notebook / Colab]
    C --> D[Train outcome-aware LSTM]
    D --> E[Export full_lstm_cache_actions.csv]

    E --> F[Prepare actions for replay]
    F --> F1[Restore packed Colab output]
    F --> F2[Validate trace name]
    F --> F3[Merge replay_access_idx if missing]

    F --> G[Convert to list_replayer prefetch list]
    G --> H[Replay in ChampSim]

    H --> I[Compare no-prefetch / SPP / LSTM]
    I --> J[Metrics]
    J --> J1[Issued-prefetch precision = USEFUL / ISSUED]
    J --> J2[Coverage = total USEFUL]
    J --> J3[IPC]
    J --> J4[Latency / deployability]
```

## 2. What one row means

Each row in `lstm_events_TRACE.csv` represents one SPP-related candidate event. It is not just an address. It includes the current demand context, the candidate prefetch address, and outcome labels.

```mermaid
flowchart TD
    A[lstm_events_TRACE.csv] --> B[Each row = one SPP-related event / candidate]

    B --> C[Raw columns]
    C --> C1[trace / event_id / replay_access_idx / cycle]
    C --> C2[pc / ip]
    C --> C3[addr]
    C --> C4[pf_addr]
    C --> C5[delta]
    C --> C6[spp_conf / spp_confidence]
    C --> C7[hit / is_store]
    C --> C8[mshr_occupancy / cache pressure]
    C --> C9[spp_issued / fill_l2]
    C --> C10[outcome_useful]
    C --> C11[outcome_duplicate]

    C --> D[Feature engineering]
    D --> D1[Encode PC/IP]
    D --> D2[Bucket delta]
    D --> D3[Normalize pressure features]
    D --> D4[Keep chronological order]

    D --> E[LSTM sequence]
    E --> E1[Hidden state carries phase history]
    E --> E2[At each event, model updates h_t and c_t]

    E --> F[Targets]
    F --> F1[good = useful and non-duplicate]
    F --> F2[duplicate]
    F --> F3[suppress / bypass-worthy]
    F --> F4[timing]
```

The replay-critical field is `replay_access_idx`, not `event_id`. The prefetch list must use:

```text
replay_access_idx 0xprefetch_byte_addr
```

If conversion prints `[idx_col] event_id`, the replay is invalid.

## 3. Model architecture

At every time step `t`, the model reads one SPP candidate event.

```mermaid
flowchart LR
    A[One SPP candidate event at time t] --> B[Feature vector x_t]

    B --> B1[PC/IP embedding]
    B --> B2[Delta embedding]
    B --> B3[SPP confidence]
    B --> B4[Cache pressure features]
    B --> B5[Hit/miss + issued/fill_l2 signals]

    B1 --> C[Concatenate]
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C

    C --> D[LSTM cell]
    D --> E[Hidden state h_t / memory c_t]
    E --> F[Shared representation]

    F --> G1[Good head]
    F --> G2[Duplicate head]
    F --> G3[Suppress / bypass-worthy head]
    F --> G4[Timing head]

    G1 --> H1[p_good = should keep / issue]
    G2 --> H2[p_duplicate = redundant]
    G3 --> H3[p_bypass = pollution risk]
    G4 --> H4[timing bucket]
```

## 4. What the LSTM memory means for cache behavior

The LSTM is learning phase-sensitive behavior:

```text
same PC + same delta may be useful in one phase but harmful in another phase
cache pressure changes the value of issuing a candidate
recent duplicate/useful history changes whether a new candidate should be trusted
```

In cache terms, the gates mean:

```text
forget gate: old access phase may no longer matter
input gate: current candidate may define a new useful pattern
output gate: memory decides how much to trust the current candidate
```

## 5. Runtime decision logic

```mermaid
flowchart LR
    A[SPP proposes candidate prefetch] --> B[LSTM reads candidate context]

    B --> C{p_good high enough?}
    C -- No --> D[Suppress candidate]
    C -- Yes --> E{Duplicate high?}

    E -- Yes --> F[Drop / deprioritize]
    E -- No --> G{Suppress / bypass-worthy?}

    G -- Yes --> H[Suppress or mark bypass-worthy]
    G -- No --> I[Issue / keep SPP candidate]

    I --> J[Replay outcome]
    H --> J
    F --> J
    D --> J

    J --> K[Measure useful, precision, coverage, IPC]
```

Current implementation is replay-based, not yet an online hardware deployment. Current replay mainly validates keep/drop prefetch decisions; bypass and timing are auxiliary signals for later online policies.

## 6. Baseline SPP vs LSTM-gated SPP

```mermaid
flowchart TD
    A[Baseline SPP] --> A1[Many issued prefetches]
    A1 --> A2[Higher coverage]
    A1 --> A3[Higher IPC so far]
    A1 --> A4[Low USEFUL / ISSUED precision]

    B[LSTM-gated SPP] --> B1[Fewer selected candidates]
    B1 --> B2[Much higher USEFUL / ISSUED precision]
    B1 --> B3[Positive IPC gain over no-prefetch]
    B1 --> B4[Lower coverage than SPP so far]

    C[Current interpretation] --> C1[Precision: LSTM wins]
    C --> C2[Coverage / IPC: SPP wins]
    C --> C3[Latency today: SPP wins]
```

## 7. Current results: 602 and 619

All results use 25M warmup / 25M simulation.

| Trace | Method | IPC | Useful / Issued | Total Useful | Conclusion |
|---|---|---:|---:|---:|---|
| 602.gcc_s-734B | SPP | 1.4440 | 5.14% | 140,717 | best IPC / coverage |
| 602.gcc_s-734B | LSTM th0.20 | 0.7175 | 57.34% | 64,473 | much cleaner selector |
| 619.lbm_s-4268B | SPP | 0.5077 | 3.42% | 88,474 | best IPC |
| 619.lbm_s-4268B | LSTM th0.10-th0.35 | 0.4568 | 94.79% | 157,564 | extremely high precision, positive IPC over no-prefetch |

Current checkpoint:

```text
Across both traces, LSTM wins issued-prefetch precision.
SPP still wins final IPC.
The research opportunity is to keep LSTM precision while increasing coverage, or distill the policy into a low-latency hardware-feasible gate.
```

## 8. Final overview

```mermaid
flowchart LR
    subgraph S1[1. Trace generation]
        A1[Run ChampSim with SPP_dev]
        A2[Dump candidate events]
        A3[lstm_events_TRACE.csv]
        A1 --> A2 --> A3
    end

    subgraph S2[2. Training]
        B1[Colab / notebook]
        B2[Outcome-aware LSTM]
        B3[full_lstm_cache_actions.csv]
        B1 --> B2 --> B3
    end

    subgraph S3[3. Replay prep]
        C1[07_prepare_actions_for_replay.py]
        C2[Validate trace]
        C3[Merge replay_access_idx]
        C1 --> C2 --> C3
    end

    subgraph S4[4. Replay]
        D1[02 actions -> prefetch list]
        D2[03 no-prefetch / SPP / LSTM replay]
        D3[09 compare SPP vs LSTM metrics]
        D1 --> D2 --> D3
    end

    S1 --> S2 --> S3 --> S4
```

## 9. What to say in a meeting

```text
I used SPP as a candidate generator and trained a stateful LSTM to learn which SPP candidates are useful, duplicate, bypass-worthy, or timing-sensitive. The key replay bug was alignment: ChampSim list_replayer needs L2 demand-access indices and hexadecimal byte addresses. After fixing that, LSTM replay is valid on both gcc and lbm. On 602, LSTM improves IPC from 0.5427 to 0.7175 and raises useful/issued precision from SPP's 5.14% to 57.34%. On 619, LSTM reaches 94.79% useful/issued precision and improves over no-prefetch, but SPP still has higher final IPC. So the current story is: LSTM is a much cleaner selector, while SPP still wins coverage and performance.
```
