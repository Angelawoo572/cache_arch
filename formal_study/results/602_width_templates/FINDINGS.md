# 602 mechanisms: H8 and H16

- **Main comparison:** [IPC, accuracy, coverage, timeliness and traffic Pareto](README.md). This page isolates width and mechanism controls; it does not replace those metrics.
- **Completed extension:** 30/30 new conditions: 2 validation + 28 full test ROIs. [Audit](audit_h16_completion.json) · [new rows](per_run_h16_completion.csv) · [28 paired controls](paired_controls_h16_completion.csv). Earlier 64 H8 + 24 width/template rows retain their original identities.
- **Contract:** seed7; frozen weights, live h/c; 5M cache/state warmup; ROI0 25–50M, ROI1 75–100M; MAC0; uncapped or cap16144. Templates are frozen using each model’s validation 20–25M only. No checkpoint was retrained.

## Width: performance improves in most measured pairs

- H16 beats its own native in **16/16 IPC pairs** and H8 in **14/16**. [Native pairs](reader_pairs.csv) · [complete run metrics](reader_metrics.csv).
- **Negative width result:** capped Streamer H16 is below H8 by 0.0104% / 0.0743% in ROI0 / ROI1. Wider is not uniformly better.
- **Traffic cost:** 15/16 H16/native pairs increase actual DRAM. Only capped ROI1 SPP lowers it. Request and DRAM Pareto conclusions differ; [exact classifications](reader_pairs.csv).

## H16: can a frozen NN-free template reproduce it?

- Below, **template IPC change = 100 × (template/full NN − 1)**. Positive favors the template. Each cell lists ROI0 / ROI1. [Exact IPC, actual requests/M and DRAM](paired_controls_h16_completion.csv).
- Fixed-payload keeps full NN gate/forward/state but replaces the final list. NN-free template executes no NN. Reset-state clears h/c before each callback; all other rules stay fixed. These are distinct interventions.

| H16 route | Own validation-frozen list | Uncapped template IPC change | Capped template IPC change |
| --- | --- | --- | --- |
| Stride | K2: (0,L2), (+2,L2); filter 0 before credit | +6.2326% / +5.9652% | +6.1272% / -7.0656% |
| Streamer | K3: (+1,+2,+3), L2 | +0.0000% / +0.0000% | +0.0000% / +0.0000% |
| AMPM | K2: (+9,+8), L2 | -0.0351% / -0.1247% | +0.0858% / +0.0154% |
| SPP | K5: (+1,L2), (+2..+5,LLC) | +0.0000% / +0.0000% | +0.0000% / +0.0000% |

- **Streamer and SPP:** templates exactly match cycles, instructions, allowed requests, DRAM and LOAD misses in all four cases. This supports replication of the measured outcomes, not policy identity on unseen inputs.
- **AMPM:** template residuals are below 0.125% IPC; the NN wins slightly uncapped and loses slightly capped. These small residuals do not isolate state or payload value.
- **Stride:** the effective +2 template wins 3/4 IPC cases while issuing more requests. Full NN wins capped ROI1, with fewer requests and less DRAM. This does not establish recurrent necessity; a matched static PC gate is not evaluated in this protocol.

## H16: fixed payload versus reset state

- **Uncapped fixed payload:** Streamer/SPP exactly reproduce the full NN in both ROIs; AMPM matches its template outcomes above. [All interventions](paired_controls_h16_completion.csv).
- **Reset-state IPC change versus full NN, ROI0 / ROI1:** Streamer -0.2054% / -6.8501%; AMPM -1.6262% / -3.8000%; SPP -1.7398% / -6.2723%.
- Reset-state reduces requests and DRAM as well as IPC. A recurrence-sensitive output is not evidence that recurrence is necessary for the gain: the NN-free templates still match or nearly match those outcomes.

## H8 under the same cap

- **Streamer:** template/full IPC +0.0004% / −0.0051%; **AMPM:** +0.8939% / +0.9542%; **SPP:** 0.0000% / −0.0006% (ROI0 / ROI1). [Original exact pairs](paired_comparisons.csv).
- Streamer/SPP outcomes are nearly reproduced; AMPM’s template improves IPC with fewer requests. Stride’s earlier gate controls remain a [separate protocol](../../../docs/602_gate_mechanism/REPORT.md). Native-relative H8 losses remain visible in the main reader.

## Size, cost and limits

- [Model size and inference state](README.md#model-size-and-inference-state): all eight checkpoints and tensor exports verified; FP32 weights = 4 × parameters; h/c storage is separate. [Layer counts](model_size_layers.csv) · [tensor counts](model_size_tensors.csv).
- MAC0 IPC does not establish finite-compute benefit. [Counted full-ROI work](reader_cost.csv) and host allocation probes are neither synthesized area nor measured energy.
- One workload/seed, independently trained widths, differing route training/state budgets, previously inspected ROI0 and different SPP fill feedback limit generalization. A failed template is not proof of stateful novelty.
- Historical 24-row report/figures are preserved under `formal_study/runs/602_width_templates_20261017/analysis_final/`; original H8 evidence is under `formal_study/archive/20261017/raw_v2/`. Old replay remains [separate](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md).
- New full raw on local: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; Sacramento: `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`.
