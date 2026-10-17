# 602 paper story：先验证可部署语义，再检验收益机制

**现有 teacher-action student 可以改善 NoPF，但本轮不支持普遍优于 teacher，也不支持把主要收益归因于复杂 learned payload。** 研究对象是保留的真实 seed7 checkpoint；不是新算法、在线训练或四模型等训练预算排名。

证据来自新协议的 **64 个完整运行**：8 validation、两个各25M的正式 ROI（每次5M缓存 warmup）。25–50M 是已观察区间的复现，75–100M 是未来区间。三个新 student 使用各自真实权重、全局 h/c 和完整 decoder；SPP 接收实际 demand/raw-fill 回调并保留预测目标。Stride 保留原有 per-PC 部署。代码一致性与测量核对先于正式执行。

| 可写的贡献／主张 | 状态 | 本轮证据与边界 |
|---|---|---|
| 从 offline/replay 推进到真实冻结 C++ live inference | **数据支持** | 三条独立真实模型的同序列 Python/C++ 状态、动作核对通过；完整运行与原始日志在[结果表](../results/formal_20261017/per_run.csv)和[认证](../results/formal_20261017/metadata/correctness_gate.json)。这是实现与验证贡献，不自动构成算法创新。 |
| Student 普遍优于传统 teacher | **未支持** | 同协议16组配对9胜7负；无 cap 为3胜5负。未来 ROI 的三个新 student 均输给各自无 cap teacher；见[配对结果](../results/formal_20261017/paired_comparisons.csv)。所有路线仍相对 NoPF 有收益。 |
| 复杂 learned payload 是收益必要条件 | **未支持（限定这些权重和两个 ROI）** | Streamer 固定模板相同或更好；AMPM 固定 payload/无条件模板更好；SPP 模板与 full 的 IPC 差不超过0.003%。模板只由 validation 冻结，无 PC 或 test 拟合。见[实际机制消融](../results/formal_20261017/mechanisms/mechanism_plot_data.csv)。 |
| 跨回调状态具有不可替代的独立收益 | **待验证；不能据此称为 stateful 创新** | reset 降低这些 NN 的 IPC，但无条件模板仍可复现主要收益。reset 同时改变输出 K、请求量及未来 cache 回调；累计曲线不提供最少历史量结论。 |
| 小模型在4 MAC/CPU-cycle下可维持本轮完整输入流 | **未支持（相同 MAC0 到达时序）** | 三个新模型的 dense-MAC4 聚合工作/实测 MAC0 窗口为4.77–11.51倍；这是吞吐必要条件检查，不是有限成本 IPC 预测。权重约11.3–11.5KB、全局h/c各64B；完整存储与主机开销另列。见[成本](../results/formal_20261017/cost/cost_derived.csv)。 |
| 有限成本 IPC 收益、跨程序与独立训练 seed 稳健性 | **待验证** | 本轮正式 IPC 仅为 MAC0、保留 seed7、602；605暂缓。不能把旧协议的有限成本或其他结果拼接成统一证据。 |

比较报告实际 requests/M 与实际 DRAM controller bytes；共同 cap 不是同实际流量。L2/LLC accuracy、timeliness、coverage 按**首次 source-enqueue 意图目标**分别报告，保留 ROI 结束未决和 resident-unused，不解释为最终填充层级或全层 usefulness。SPP 的 native/student 有效 fill 输入及内在 page policy 也不同，详见[方法边界](METHOD_LIMITATIONS.md)。

可讲的故事是：**动作监督可得到有用的冻结 student；在闭环测量中，简单的 validation 模板解释了大部分观察收益，而真实计算吞吐仍是部署障碍。** 保留这些负面结果，比预设 learned payload、循环状态或教师选择一定带来创新更有依据。
