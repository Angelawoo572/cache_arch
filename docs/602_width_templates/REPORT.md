# 602 H16 与 capped H8 模板：正式结果

**24/24 完整运行完成并验证：2 validation +16 H16 test +6 H8 capped-template test；9,748 项审计检查零失败。** 真实原始 seed7 H16 checkpoint 已找回；未重训、未改变公共 C++ 引擎。原 64 行 H8/native/NoPF 数据按兼容控制显式复用，保留原协议身份。

**结论：H16 保留了相对 native 的优势，但复杂 learned payload/循环状态不是这些收益的必要解释。** SPP H16 的简单模板在四个完整 ROI/cap 组合精确复现 IPC、cycles、请求和 DRAM；AMPM H16 仅在无 cap 下有很小残差，加 cap 后模板略好。H8 的 cap 下模板也几乎复现或超过 NN。

模型、teacher 和模板都在相同模拟机器、trace、L2 LOAD、外部 page/cap、warmup 与 ROI 合同下比较。SPP native 的内在同页规则及 prefetch-fill-only FILTER 反馈与 student 的全部 raw fill 输入不同，这些原语义被保留，不能声称其有效输入完全相同。见[方法边界](../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md)。

## H16 是否保留旧报告优势

**是，限定本轮 MAC0、602、原 seed7：两模型相对 native 共8/8胜，也在8/8组中高于各自 H8 NN。** 这里的结论来自新 live 配对，不是将旧 replay IPC 横比。AMPM H16/native 提升0.5320%–2.2054%；SPP 为1.4146%–4.1512%。

| 模型 | ROI | cap | native IPC | H16 NN IPC | 自己的模板 IPC | NN/native | NN/模板 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMPM | 0 | 0 | 0.430021 | 0.432309 | 0.432157 | +0.5320% | +0.0351% |
| AMPM | 0 | 16144 | 0.418974 | 0.428215 | 0.428582 | +2.2054% | -0.0857% |
| AMPM | 1 | 0 | 0.392795 | 0.399042 | 0.398544 | +1.5903% | +0.1248% |
| AMPM | 1 | 16144 | 0.327222 | 0.331705 | 0.331756 | +1.3698% | -0.0154% |
| SPP | 0 | 0 | 0.427730 | 0.436556 | 0.436556 | +2.0634% | +0.0000% |
| SPP | 0 | 16144 | 0.423904 | 0.429901 | 0.429901 | +1.4146% | +0.0000% |
| SPP | 1 | 0 | 0.387512 | 0.403598 | 0.403598 | +4.1512% | +0.0000% |
| SPP | 1 | 16144 | 0.332809 | 0.337639 | 0.337639 | +1.4515% | +0.0000% |

**是否超过自己的简单模板：不稳定。** SPP 四组完全相同；AMPM 无 cap 时 NN 仅高0.0351%／0.1248%，cap 时低0.0857%／0.0154%。这些小差异不能独立归因于循环状态。AMPM 的 H8 模板在 cap 下还高于 H16 模板，两种宽度并非对所有部署方案都更大更好。

## H8 在 cap 下是否仍被模板复现

模板只有 address-relative K/delta/fill，不加入 PC；三个 H8 模板逐字复用已冻结 validation 选择。以下均为完整25M ROI，正号表示模板更好。

| 模型 | ROI | NN IPC | 模板 IPC | 模板/NN | 实际 requests/M：NN /模板 | DRAM MiB：NN /模板 |
|---|---:|---:|---:|---:|---:|---:|
| STREAMER | 0 | 0.431659 | 0.431661 | +0.0004% | 16143.96 /16143.96 | 11.028 /11.028 |
| STREAMER | 1 | 0.337604 | 0.337587 | -0.0051% | 16143.96 /16143.96 | 21.646 /21.649 |
| AMPM | 0 | 0.426299 | 0.430109 | +0.8939% | 16143.96 /16130.24 | 11.466 /11.570 |
| AMPM | 1 | 0.331607 | 0.334771 | +0.9542% | 16143.64 /16101.84 | 22.412 /22.447 |
| SPP | 0 | 0.417164 | 0.417164 | +0.0000% | 16143.96 /16143.96 | 12.661 /12.661 |
| SPP | 1 | 0.325346 | 0.325345 | -0.0006% | 16143.96 /16143.96 | 21.916 /21.915 |

**是，限这些模型和区间。** Streamer 的差不超过0.0052%，SPP 不超过0.0006%；AMPM 模板反而高0.8939%／0.9542%，且请求更少。这是无 NN 机制对照结果，不是“PC表失败所以有 stateful 创新”。原 H8 相对 native 的失败结果保留。

## 实际流量、动作与指标

| H16 NN | ROI | cap | 实际 requests/M：NN /模板 | 实际 DRAM MiB：NN /模板 |
|---|---:|---:|---:|---:|
| AMPM | 0 | 0 | 32339.04 /32339.04 | 11.845 /11.834 |
| AMPM | 0 | 16144 | 16143.96 /16143.96 | 11.474 /11.476 |
| AMPM | 1 | 0 | 42628.64 /42628.64 | 23.806 /23.764 |
| AMPM | 1 | 16144 | 16143.96 /16143.96 | 22.427 /22.423 |
| SPP | 0 | 0 | 80847.59 /80847.59 | 13.537 /13.537 |
| SPP | 0 | 16144 | 16143.96 /16143.96 | 12.663 /12.663 |
| SPP | 1 | 0 | 106570.99 /106570.99 | 25.258 /25.258 |
| SPP | 1 | 16144 | 16143.96 /16143.96 | 21.326 /21.326 |

共同 cap 是上限，**不保证同实际请求、接受或 DRAM**。H8 AMPM 模板实际请求低于上限；即使请求数相同，DRAM 也可能不同。SPP H16 无 cap 相对 native 请求更少，但实际 DRAM 反而更多（ROI0 +4.74%，ROI1 +11.57%）。这不是流量节省。cap 也降低了 H16 未来 ROI 的绝对 IPC：AMPM −16.87%、SPP −16.34%，保留这一负面结果。

H16 全 validation 冻结模板：AMPM K2，slots(+9,L2),(+8,L2)；SPP K5，(+1,L2),(+2..+5,LLC)。SPP H16 在 ROI0 的404,238与 ROI1 的532,855个完整 demand 事件上，emit/K/各 slot delta/fill 均等于该模板；这是完整 ROI 的经验事实，不是所有输入上的数学保证。完整 K 支持、罕见 delta/fill、current-line、duplicate、slot、allowed/redundant/useful、完整与分阶段计数均保留于 CSV。

IPC=实际 retired/cycles；requests/M=实际 credit-allowed cache 调用/retired×10⁶。DRAM 使用控制器实际完成的 read/write bus bytes，并保存 started/completed、LLC↔DRAM 事务、结束未决；不使用 issued×64。

L2/LLC 指标分别按首次 source-enqueue 意图 cohort 报告：accuracy=LOAD-useful/enqueued，timeliness=timely/(timely+late)，coverage=LOAD-useful/相同 ROI NoPF 对应层 LOAD misses。后续 merge 可能升级 fill，故意图 cohort 不等于最终 fill 层级；不完整追踪范围标 NA，不混成全层 accuracy。ROI 结束 resident-unused/pending 被保留为 censored，evicted-unused 不称 pollution。coverage 也不是因果上的净 miss 消除；另存实际 miss/MPKI。

启动与阶段曲线来自完整 snapshots，分别用 retired instructions、实际 callback 和 completed inference update 作为横轴；训练量、warmup 历史和测量进度分开。首次超过 NoPF 的累计点不称最少观察量。

## 代价与存储

| 模型 | FP32 weights | 全局 h/c | 对象+持久堆（包含前两项） | fixture 临时堆峰值 |
|---|---:|---:|---:|---:|
| AMPM H16 | 26,612 B | 128 B | 28,596 B | 884 B |
| SPP H16 | 26,436 B | 128 B | 28,656 B | 1,148 B |

这是实际编译 ABI 与 C++ new/new[] requested payload，排除 outer adapter、观察器、日志、栈、allocator metadata；不是完整硬件 SRAM/PPA 或 RSS。权重与 h/c 已包含在持久堆内，不能重复相加。临时峰值只覆盖已通过的正确性 fixture，原 decoder 无统一 degree 上限，不能当成最坏情况界限。模板逻辑字段为 uint64 K 与每 slot int64 delta/int32 fill（AMPM32B、SPP68B）；实际 Action ABI 为32B/slot，容器容量、descriptor/control 另计，逻辑字段不是完整实现 footprint。

本轮 IPC 是同步 **MAC0**：完整 forward 后发布 h/c 与输出，没有有限队列/service/bandwidth 迟延模型；权重驻留，NN weight DRAM=0 是模型假设，单列记录，不声称实际硬件免费。按完整 ROI 实测运算，H16 的 ceil(MAC/4)/MAC0 cycles 为 AMPM12.95／15.75、SPP23.95／31.15；即使忽略所有其他开销，也不足以在相同 MAC0 callback 时间窗服务全量工作。这仅为吞吐必要条件，不能推断有限资源 IPC。非线性、bias/control、state 与存储范围均明列。

## 旧 slides / replay 与 live 对齐

原 seed7 AMPM v7 H16=6,653参数；SPP empirical-prior v2 H16=6,609参数。二者均8 epochs但训练算力和 event/optimizer 数不同，不能称等计算量重训。旧 AMPM 推理从 guard20M fresh state，SPP 从 train0M起；旧 replay cache warmup25M。本轮各 run fresh h/c 后5M NoPF cache/state warmup，保留 h/c 进入测量：validation20–25M，ROI025–50M，ROI175–100M。

旧 replay 数量链已核实：AMPM 808,438 generated=loaded →363,119/404,219 matched keys →726,238 emitted；SPP2,021,095 generated=loaded →392,788/404,219 matched →1,963,940 emitted。生成列表与 runtime 请求差异来自 keyed match，不能把 replay 称独立在线推理。旧动作/指标/原始日志链接见[证据对齐](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md)及[7行原始口径表](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/old_replay_alignment.csv)。

[Slides 勘误](SLIDE_ERRATA.md)明确：Stride163,567 silent→act是全部404,219事件的40.4649%，或 teacher-silent340,746事件的48.0026%；请求不是 DRAM；SPP H8零 L2 coverage 不等于零 LLC/性能收益。原 deck、v1/v2、source 与所有旧结果不改写。仅对本地保留 v7_15 deck 做对齐，未声称已获取新版云 slides。

## 可复核交付与剩余问题

| 路线 | 本轮状态 | 保留历史 |
|---|---|---|
| Stride | 不新增run；原 per-PC frozen 参考保留 | offline / live / formal / gate，旧13行参考 |
| Streamer H8 | 新增2个 capped-template test，完成且验证 | 原H8模型、NoPF/native/live与模板 |
| AMPM H8/H16 | 新增11个完整run，完成且验证 | 原v7 H8/H16 checkpoint 与 replay |
| SPP H8/H16 | 新增11个完整run，完成且验证 | 原empirical-prior H8/H16、fill-aware replay及v1/v2 |

仅有602、一个真实训练 seed7 和两个已定义区间；ROI0已观察，ROI1是独立未来区间但本轮续研时也已有H8结果，不称完全盲测。未检验有限引擎 IPC、额外 seeds 或跨 workload 泛化；未训练新权重。模板不能复现的少量 AMPM 残差仍需独立机制消融，不能归为 stateful 创新。

- [简表 CSV](summary.csv)；[全部24新增行](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/per_run_new.csv)；[88行来源标注总表](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/per_run.csv)；[配对比较](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/paired_comparisons.csv)。
- [正式审计](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/audit.json)、[指标定义](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/metric_definitions.json)、[成本](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/storage_cost.csv)、[一页 story](PAPER_STORY.md)。
- [H8/H16 IPC–请求–DRAM：ROI0 cap](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/widths_roi0_cap16144.svg)、[ROI1无cap](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/widths_roi1_cap0.svg)、[H8 cap](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/h8_template_cap.svg)、[成本](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/width_cost.svg)。
- [完整动作分布](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/distributions_spp_roi1.svg)、[阶段](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/phases_spp_roi1.svg)、[完整图表索引](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/README.md)；[方法图](method.svg)。
- [配置与恢复运行](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/README.md)、[路径映射与恢复](LAYOUT_AND_RECOVERY.md)。

本地完整新 raw：`/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`。Sacramento：`/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`。新目录保留 checkpoints/models、actual binary snapshot、完整 run logs/callbacks/snapshots、source bindings、cost probes 与 analysis_final；复用的完整实际构建在各自 `formal_study/archive/20261017/raw_v2/builds/`。历史轮次在 `archive/20261016` 与 `archive/20261017`；健康 checkout 在 `work/cache_gate_mechanism`。迁移保留旧路径兼容链接，历史元数据原样保留。

新 raw 同步核对345个服务器文件、176,314,065字节，路径和大小全部一致；另有本地新增分析、图和算术核对将同步回服务器。无 SHA 台账。构建、traces 和大日志保留本地/服务器，不纳入 Git。
