# H8 paper story: frozen live inference and benefit mechanisms

- This page preserves the original **64-row H8 protocol**. The later **24-run H16/capped-H8 supplement** is separate: [current paper story](../../../../docs/602_width_templates/PAPER_STORY.md), [results](../../../../formal_study/results/602_width_templates/README.md). Do not relabel their rows as one protocol.
- **Finding:** retained teacher-action students improve on NoPF, but do not consistently beat native teachers; simple validation templates explain most observed gain.
- Scope is existing seed7 weights, not a new algorithm, online weight training or a four-model ranking under equal training budgets.
- Evidence covers **64 full runs, 8 validation + 56 test**, with two **25M** ROIs and **5M** cache warmup each. ROI **25–50M** repeats an observed interval; **75–100M** is a future interval relative to training.
- Three actual global-state models execute full decoders; SPP sees live demand/raw-fill callbacks and preserves predicted fill targets. Stride keeps its per-PC deployment. Source/numerical checks preceded formal runs.

| Claim | Evidence status | Support and limit |
|---|---|---|
| Independent frozen C++ live inference beyond offline/replay | Supported | Three real models pass same-sequence Python/C++ h/c/action checks; [full rows](../results/formal_20261017/per_run.csv), [actual-core certification](../results/formal_20261017/metadata/correctness_gate.json). Implementation/validation does not establish algorithmic novelty. |
| Students uniformly beat native teachers | Not supported | 16 pairs: 9 wins / 7 losses; uncapped: 3 wins / 5 losses. All three new students lose to their uncapped native teacher in the future ROI. All routes still beat NoPF. [Pairs](../results/formal_20261017/paired_comparisons.csv) |
| Complex learned payload is necessary | Not supported for these weights/ROIs | Streamer templates are equal or better; AMPM fixed payload/unconditional tables are better; SPP template IPC differs by at most 0.003%. Tables are frozen on validation, without PC/test fitting. [Interventions](../results/formal_20261017/mechanisms/mechanism_plot_data.csv) |
| Recurrent state provides an irreplaceable independent advantage | Pending | Reset lowers these NN IPCs, but unconditional templates still reproduce most gain. Reset changes K, requests and future cache callbacks; neither it nor a cumulative crossing establishes stateful novelty or minimum history. |
| Four MACs/cycle can sustain the same full callback stream | Not supported on this MAC0 timeline | Dense work/4 divided by measured MAC0 cycles is 4.77–11.51×. This is a necessary throughput check, not finite IPC. Weights are about 11.3–11.5 KB; one global h/c is 64 B. [Cost](../results/formal_20261017/cost/cost_derived.csv) |
| Finite-cost IPC, cross-program and independent-seed robustness | Pending | Formal IPC here is MAC0/602/retained seed7; 605 is deferred in this protocol. Earlier finite-cost/other-protocol rows are not pooled. |

- Report actual requests/M and DRAM controller bytes separately; the common ceiling does not equalize realized traffic.
- L2/LLC accuracy, timeliness and coverage use separate first source-enqueue intent cohorts. Preserve boundary pending/resident-unused; intent is not final fill level or all-level usefulness.
- Native/student SPP fill inputs and internal page rules differ. [Method limits](METHOD_LIMITATIONS.md).
- **Supported story:** action supervision can yield useful frozen students; closed-loop template interventions explain most measured benefit, while counted compute remains a deployment obstacle.
- Preserve losses and uncertainty. The evidence does not establish that learned payload, recurrent state or teacher choice is an innovation.
