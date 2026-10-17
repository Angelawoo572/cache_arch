# Paper story：恢复模型优势，再验证是否需要神经机制

**保留的 H16 动作监督 student 在本轮 live MAC0 中恢复了相对 native 的优势；简单 validation 模板解释了几乎全部收益。** 这支持一项可复核的经验与部署机制研究，不自动构成新预取算法或 stateful 创新。

从真实旧 checkpoint 出发，核对实际编译 Python/C++ 状态与动作，完成2个完整 validation +22个完整 test。原64行参考通过相同实际模拟器、warmup/ROI/page/native控制核对后显式复用，旧 replay 单列。所有权重冻结；运行时只有 h/c 更新。

| 主张 | 状态 | 对应证据 |
|---|---|---|
| 真实 SPP/AMPM H16 可恢复并独立 live 推理 | **数据支持** | [模型身份与旧日志](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md)，实际构建 parity 与[9748项正式审计](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/audit.json)；未以重训替代旧权重。 |
| H16 保留 native 优势 | **数据支持，限定本矩阵** | [8组 H16 NN/native](summary.csv)全胜：AMPM+0.53%–2.21%，SPP+1.41%–4.15%。H16 NN也高于各自H8 NN；这不是跨协议复用旧IPC。 |
| 收益必须依靠 learned payload / 循环状态 | **未支持** | [完整机制对照](REPORT.md)：SPP H16模板4组精确复现；AMPM无cap仅0.035%/0.125%残差，cap反输模板。不能将微小残差直接归因于状态。 |
| H8 cap 排除了无 NN 模板解释 | **未支持** | [6组 capped H8 模板](summary.csv)：Streamer/SPP近乎复现，AMPM模板高0.89%/0.95%且请求更少。原失败结果保留。 |
| 共同cap保证同流量、更少请求保证省DRAM | **未支持** | [实际请求与DRAM](REPORT.md)：cap只是上限；SPP H16无cap请求少于native，DRAM反增4.74%/11.57%。不使用请求×64。 |
| 有限计算资源下仍获益、跨seed/程序泛化 | **待验证** | 本轮仅MAC0、602、seed7。H16 MAC4聚合工作/窗口约12.95–31.15倍，不是有限成本IPC预测；[实际存储与假设](REPORT.md)。 |

本轮保留：AMPM有小残差；cap降低未来区间绝对IPC；SPP/H8仍输native；简单模板也有不能逐cycle完全复现的情况。模板失败或累计IPC首次过线都不能证明状态创新或最少观察量。旧replay的匹配损失、25M缓存warmup与不同state初始化保留在[slides勘误](SLIDE_ERRATA.md)，与新5M warmup/live分开。

[完整报告](REPORT.md) · [方法图](method.svg) · [图表与原始口径](../../formal_NN_training/experiments/602_four_prefetcher_formal/results/width_templates_20261017/README.md) · [原始数据和恢复路径](LAYOUT_AND_RECOVERY.md)
