# Paper story：先排除预算与简单载荷解释

问题是：传统预取器动作监督得到的 frozen recurrent student，收益究竟来自 learned
address payload、动态 gate，还是更简单的固定 lookahead？本轮只研究 602 的两个未来
25M ROI，沿用三个真实训练 seeds，不预设 recurrent 方案应当获胜。

| 陈述 | 证据状态 | 结果与边界 |
|---|---|---|
| Current-line 会严重扭曲共同请求 cap | **数据支持** | 原 capped NN 的 98.17%–99.92% credit 被 current-line 消耗。只在扣 credit 前过滤，MAC0 相对 NoPF 恢复 ROI0 +3.20%、ROI1 +2.77%–2.82%；仍低于 uncapped。 |
| 主要收益需要 learned address payload | **未支持** | 保留全 NN gate/forward/h/c，最终改为一个 +2，配对 IPC 差 −0.00418% 至 +0.00990%。全 ROI raw slot0 总是 0、slot1 几乎总是 +2。 |
| 达到这些收益必须使用 recurrent gate | **未支持** | 55-entry frozen PC-gate/+2 的 cap 配对差仅 −0.05151% 至 +0.00893%。Unconditional +2 在两个 uncapped ROI 均超过 NN，但请求更多；同 cap 下动态 gate 的 IPC 优势仅约 0.06%–0.11%。 |
| 有限 serial NN 已取得稳健收益 | **未支持** | seed7 MAC4 过滤后相对 NoPF：ROI0 +0.0450%、ROI1 −0.0798%；约八成输入仍丢弃，成本下方向不一致。此结论仅覆盖本成本模型和单 seed 子集。 |
| Teacher-silent 输出有整段净价值 | **数据支持，有限范围** | 原始 ROI1 seed7 MAC0 uncapped 的实际抑制干预使 IPC 下降 9.409%。不能按标签给单个请求分摊因果 IPC，也不能由此证明 payload/recurrence 必要。 |
| 跨程序价值、stateful 必要性或硅上收益 | **待验证** | 本轮没有跨程序重训、line-feature/hc 独立消融、online weight update 或综合硬件；PC 表未完全追平不代表 stateful 创新。 |

可报告的方法贡献是清楚区分生成、credit、cache、L2/LLC cohort、末端删失和实际 DRAM，
并使用保持完整计算的 payload 干预、冻结 PC table 和实际 silent-output 消融识别解释。
这是一份受边界约束的机制结果，不主张新架构或首次性。三 seeds 是训练稳健性，
不是 workload 分布的置信区间，也未作统计等效性检验。

[结果与协议](REPORT.md) · [逐 seed](results/metrics_per_seed.csv) ·
[实际配对](results/paired_comparisons.csv) · [全 ROI payload](results/learned_payload_full_roi.csv) ·
[完整 phase](results/phase_distributions.csv) · [成本](FINITE_CURRENTLINE.md) ·
[可恢复运行](../../formal_NN_training/experiments/602_gate_mechanism/RUN_GUIDE.md) · [diagram](diagram.svg)
