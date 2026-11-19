# 从新的真实 workload 中寻找可互补的 LLC 预取规律

**Google trace 里已经看到相同物理行可以由不同访问流提前预知。** 在 Delta，两个 load PC 各自按 +2 行前进，交替后整条流按 +1 前进。逐条对齐请求，能解释为什么 Stride 在不少需求上及时命中，而另一个同样给出正确地址的预测器迟到。**目前更接近 ML 研究动机的现象出现在 SPEC MCF**：当前 PC、页内偏移和最近两个本地步长都一样，较早的 PC 序列却能把两个后续目标区别开。其中一个真实目标在三种 conventional 策略中都没有被提出，但该分支的训练样本很少。现有证据支持继续研究条件相依的访存规律；尚未证明神经网络在未见 workload 上具有稳定的性能优势。[模式与支持度](patterns/README.md) · [跨时间段验证](transfer/README.md)

## 实验范围

从官方 manifest 核对 610 个对象：359 Google gtrace_v2、79 Graph、92 SPEC17、80 AI/ML。先按工作负载名称和用途登记样本，再读取 discovery 数据。16 条记录涵盖 7 个 Google 服务标签、4 个 Graph、3 个已有 SPEC 对照和 2 个 AI/ML 程序。保留组含 6 条指定测试记录和同组另外 74 条；只读取目录元数据。之前的 SPEC/GAP、LBM、BFS、synthetic 和编译循环结果继续作为参考。[官方 DPC4 数据说明](https://github.com/CMU-SAFARI/DPC4#workload-traces) · [登记表](intake/registry.csv) · [清单、输入及兼容性](intake/README.md)

| 窗口 | 工作负载 | 原始指令范围（测量） | 选择依据 |
|---|---:|---|---|
| A | 16 | 100,001–350,000 | 全部登记记录，100k warmup；发现候选规律，按时间切分训练、开发、测试。 |
| B | 16 | 550,001–800,000 | 开始时预先固定的第二窗口，重新 warmup 100k；检验 A 的规律。 |
| C | 7 | 3,000,001–5,000,000 | 因 Google 短窗口的条件分支支持稀少，追加全部七条 discovery 记录，1M warmup；同一记录内更深、更长的敏感性试验。 |

39 个窗口分别运行 NoPF、PC-based Stride、256 项 multi-delta 和 degree-one SPP，共 **156 个原生配置**。固定双核几何、一核 active、同一页放置、共享 LLC/DRAM 时序，逐 load 校验原始指令、operand、PC、虚拟行和 LLC 可见的物理行。每个策略有真实的去重、队列、驻留和服务时机；主比较每触发点最多发一项，实际接受数与流量分别报告。这是单核窗口重放，Google 原机的多线程调度、原始物理映射和转换前的所有依赖细节无法从 DPC4 文件恢复。Google V2 确有寄存器依赖编码，不能直接把 V1 的问题套在 V2 上。IPC、head stall 和关键性据此解释。[机器与归因](screen/README.md) · [Google 原始格式及调度说明](https://dynamorio.org/google_workload_traces.html)

![Google 七个服务标签三段的固定工作量时间](figures/google_execution.png)

## 哪些规律值得继续研究？

表中按现有证据的强弱及能否指导下一次架构实验排序。**原生**指真正发出请求并执行完整工作量；**离线**指地址正确且有参考提前量。离线候选的效用不折算为 IPC。

| 优先级、规律 | 发生频率与可预测性 | 性能证据及 conventional 成本对照 | 能得到的结论 |
|---|---|---|---|
| **1. 不同 PC 从不同位置预测同一行**，Google Delta | A 段两个 PC 的交替 +2 模式在训练出现 420 次、测试尾段 112 次。在 A/B/C 的同一类 demand 上，Stride 及时而 SPP 迟到的次数分别为 872/875/952。 | 三段中该组 proposal→demand ingress 中位数均为 Stride 122 周期、SPP 2 周期。A 段 SPP 直接及时覆盖 1,821 个原始 miss，比 Stride 的 1,786 个更多，整段却多用 72,535 周期。反向也有 713 个从 Stride late 变成 SPP timely。两条普通访问流加固定预测距离能解释许多正确目标。 | 最强的可复现原生现象。要同时看两种策略已取得的成功和丢掉的成功。是否需要学习型机制仍开放。[请求生产者与需求](screen/producer_pairs.csv) · [实际完成时间](screen/performance.csv) |
| **2. 相同局部状态，未来有两个不同物理目标**，SPEC MCF | B 段四组局部上下文/距离在 A 的训练前缀已有两个分支，两种未来都在 B 重现。例：同 PC、offset 19、最近两个本地 delta 均 +1。先前全是本 PC 时得到 +2；另外两条 PC 介入后得到 +6。 | +2 已由三种 conventional 及时覆盖。+6 在三条执行轨迹上均可提早 1,354 周期构造，三者都未提出且目标都 miss，head wait 为 238 周期。但 A 中 +6 仅出现 1/5 次；同一测量中，在 horizon 16 增加三个 PC 还失去 47 个原本由局部表识别的目标，多认出 10 个。两项都只统计三个 conventional 在相同需求上均 miss 的情况。 | 这是值得独立复现的候选。要验证输出发生率和可泛化的区分信息，再判断表项是否有共享的必要。[两个未来的原始 load](transfer/paired_futures.csv) · [各策略的真实发出时机](screen/transfer_witnesses.csv) |
| **3. 不固定“第几次未来访问”的跨 PC 关联**，Google Charlie | C 段训练中，先前 PC 与当前 PC/offset 组合 8 次有 7 次支持 +5。测试时指令 4,658,501 提出物理行 370,565，5 个 ingress 后需求到来。相同预算下，前一 PC 的规则在测试尾段比局部表多覆盖 9 个、失去 6 个；只有一个新增目标也超出四种简单对照的并集。 | 候选领先 104 周期，训练参考为 99，仅剩 5 周期。原生 Stride 沉默，multi-delta/SPP 虽命名正确，但 dispatch→fill 分别为 190/229 周期。整段尾部一个高频固定偏移覆盖 808 个参考及时的原始 miss，前一 PC 规则覆盖 220 个。 | 有弱证据支持条件关联；这个具体候选能否真正及时仍未证实。精确序号任务和灵活访问窗给出了不同的观察。[灵活窗口实例](patterns/long_google/window_examples.csv) · [原生生命周期](screen/charlie_probe.csv) |
| **4. 可迁移的简单规律与容易误判的“泛化”**，PageRank、CLIP、Llama 和 Google Tahoe | PageRank 的两个 +1 流交织成不同地址序列，最高支持的四事件组合在测试中出现 1,255 次。对第 16 个未来事件，固定 +8 覆盖 2,359 个有提前量的原始 miss，偏移表 2,380 个。CLIP/Llama 在同一任务上与固定 +16 打平。 | 将原有 LBM R1 的冻结表直接送到 Google，A 段全部七条合计的合格目标与无条件先验完全相同。Tahoe B 一度出现 217 个目标；对 PC 标签做保持先后与相等关系的重编号后，15 种替代标号都只剩 74 个。相同发出位置的固定 +16 达到 358 个。 | 当前结果可由简单步长或数值索引解释。不能从单一相位或表的偶然命中推导通用表示。[跨 workload 冻结测试](transfer/README.md) |
| **5. 可预测的远距离复现与真正的新 miss**，Google 深区间 | 无容量上限的物理行后继表命中 2,264 个“下一个地址”，其中 331 个跨越 512 行以上且超出最近三页；2,264 个在 NoPF 中原本都 hit。 | 在四种配置均 miss 且有参考提前量的 4,897 个 next-event 样本中，先前 ROI 的同核 LLC ingress 从未观察过任何一个精确目标行，但已观察过其中 2,477 个目标页。 | 只检索见过的精确地址在这个集合上没有 reach。已见页内生成新行是另一个可检验假设。尚无证据证明这些所有新行都可预测或都值得取。[时序关联、容量和 reach](temporal/README.md) |

### 为什么要看 conventional 成功的整个片段？

以 Delta 的需求 100,121 为例。Stride 从同一 PC 先前访问的行 58,156 走 +2，到达 58,158。SPP 后来根据另一个 PC 的行 58,157 走 +1，也提出 58,158。前者的 fill 领先 demand ingress 62 周期，后者落后 208 周期；同一 load 的 ROB-head wait 从 0 变为 187。A 段 872 个 Stride timely → SPP late 中，866 个具有这个“同 PC +2 / 另一 PC +1”的组合。该组合在反方向也发生，选择路径还要考虑实际触发时间与内存服务，不能只按哪个 PC 提出来做决策。[真实请求记录](screen/producer_examples.csv)

![同一物理地址的两个实际请求](figures/same_target.png)

始终没有产生 timely hit 的流保留在分母中。Google Tango A 的 243 个 Stride PC 中有 239 个没有可归因的及时首次使用，涵盖 1,333 个原始 misses。CactuBSSN A 中 324/324 个 PC 都没有 Stride 的直接及时命中，但 SPP 能将 1,787 个原始 miss 中的 1,293 个变成及时 hit。另一个边界条件是 GAP SSSP：两个测量区间都解出 50,000 个 store、零个 load，四种策略所得时间相同。它说明该窗口没有 demand-load 预取机会，不能据此评价整个应用的复杂度。[完整序列分母](patterns/sequence_populations.csv) · [原始记录复核](intake/format_audit.csv)

在 Sacramento 上独立编译了相同的模拟器，另开输出目录重跑全部 156 个配置。与本地逐项对照，退休指令数、完成周期和每个已记录的系统计数器均一致。[跨平台校验](screen/verify_repeat.py)

## 证据边界和下次讨论

- **得到支持**：不同流的关系、访问间隔、条件上下文会改变可提前生成的具体地址，有时会改变同一 demand 的及时性。直接 timely coverage、accuracy、miss rate 和 MPKI 各自回答不同问题。在固定退休工作量下，最终以完成周期和 IPC 解释哪种原生策略更快。所有策略的真实接受数、DRAM 流量和 head wait 均已列出。[完整指标](screen/performance.csv) · [第二窗口](transfer/performance.csv) · [较深 Google 窗口](transfer/long_google/performance.csv)
- **缺少支持**：Google 的七个短区间及七个较深区间，在规定的精确局部键、词表、时间预算下，都没有得到满足“双分支在训练及测试均复现”的正例。许多测试键此前未见过；改变观察窗能看到 Charlie 的线索。这里不能得出 Google workload 简单或任何序列模型都无效的结论。[训练支持与容量敏感性](patterns/long_google/key_support.csv)
- **待判定**：是否存在可跨新程序稳定复用、机会数足够多、且同成本 conventional table 难以同时识别的条件关系？最小的下一步是从本文具体的 cue/target 对出发，冻结候选词表、容量、grant 率和推理延迟，在新的独立记录上核实两种未来是否都出现，再对幸存目标进行真实发出重放，记录它挤掉的旧命中以及程序完成时间。独立保留组可以用于这次检验。模型类型、记忆表示、输出距离、空间 footprint 和 hybrid 仲裁方式留待这一步之后讨论。

[复现步骤](REPRODUCE.md) · [上一轮 LBM、BFS、嵌套循环和干扰研究](../transition_study/README.md)
