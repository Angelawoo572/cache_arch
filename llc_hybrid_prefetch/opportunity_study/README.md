# Architectural opportunities for a learned–conventional LLC prefetcher

项目关注 ML 能否更早、更准确地生成有用的预取地址，并在有限资源下缩短执行时间。[当前研究](../transition_study/README.md) 从完整的 NoPF/Stride 对齐序列出发，比较条件模式、预测距离和跨 context 泛化。以下是已完成的一组架构实验。

这组实验找到了一个真实、可检验的历史线索。在 BFS-3 的后段，一条访问回到某个物理页时，取回**上一次访问该页的行偏移**，比只看当前访问更容易提前选中目标。其中两个实际 load 的页内偏移分别走了 `50 → 51 → 55` 和 `9 → 10 → 12`。Stride、next-line、多步长表和 SPP 都 miss，16 项的页历史在同一触发点及时预取成功。CactuBSSN 暴露了另一个问题：目标有时早已可预测，但同页 Stride 无权构造它；放宽物理页范围后，一个固定偏移已能获得显著部分收益。实际重放中，回查页偏移比只看当前访问多节省 21,596 个周期，但同发出机会的 next-line 又比它多节省了 7,037 个周期。两个例子共同要求先把「地址能否构造、能否提前识别、能否及时到达、程序能否更早结束」逐一验证。

| Report | What to read it for |
|---|---|
| [Architectural opportunities](reports/architectural_opportunities.md) | Real cue/target examples, timing and head waits, reach ceilings, fixed-work effects, and the claims the data support |
| [Experiment methodology](reports/experiment_methodology.md) | Baseline audit, predictor-visible state, fixed-work definitions, implementation, tests and commands |
| [History and feature study](history/history_feature_study.md) | Ordered vs shuffled 4–64-event summaries, transfers, support, physical address vocabularies and target-level samples |
| [Conventional baselines](conventional/conventional_baselines.md) | Strong non-neural controls, matched degree and issue budgets, duplicates, real queueing and criticality interventions |
| [ML mechanism exploration](reports/learned_memory_mechanisms.md) | What different forms of memory could add, relevant computer architecture and ML papers, hardware implications and falsifiers |
| [Prioritized next experiments](reports/next_experiments.md) | The evidence table and the smallest experiments that would justify learned retention or retrieval |

[原生页历史重放](page_memory/README.md) · [31 paired workload cases](workloads/README.md) · [Native selective-cue results](deadlines/replay_results.csv) · [CactuBSSN physical-layout replay](physical_replay/placement_results.csv) · [Original 48-run baseline study](../MEASUREMENTS.md)

The reference simulator, prior measurements and analyses remain in the parent project. This study's sources, configuration, tests and compact data products are here. Local and Sacramento raw traces and logs are under this study's own raw directory. The research branch contains the material needed to inspect and reproduce the comparisons. No neural model is part of these runs.
