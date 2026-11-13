# LLC LSTM 的性能评估

首轮让两个核各自运行一份 602 trace。固定指令区段、预热、共享 LLC 与 DRAM 配置和随机种子，并在相同的请求预算下比较策略。这个设置用于检查采集、推理和计数能否对齐。要判断共享状态是否有效，再用不同 602 区段或不同程序的组合改变两个核的流量和干扰程度。

## Accuracy、timeliness 和 coverage

给每个实际产生独立 LLC 预取请求的候选记录发出周期、填充周期、第一次 demand data LOAD 的入口与实际 tag lookup 周期、以及是否逐出。用物理行和请求 ID 关联记录，任一核可消费共享 LLC 中的预取行。分别记录尝试、排队、合并、丢弃与真正产生的请求。队列结束时未观察到结局的请求单独计数。分类可参考 Pythia 对 demand 与 fill 顺序的区分，以及 FDP 对 MSHR 合并和污染的追踪。 [Pythia §3–4](https://people.inf.ethz.ch/omutlu/pub/Pythia-customizable-hardware-prefetcher-using-reinforcement-learning_micro21.pdf) · [Feedback Directed Prefetching §2–3](https://users.ece.cmu.edu/~omutlu/pub/srinath_hpca07.pdf)

| 指标 | 定义 |
| --- | --- |
| Prefetch accuracy | `(及时 + 迟到) / 已有结局的独立预取请求数`。同时报告未决数和及时命中占独立请求的比例。 |
| Timeliness | `及时 / (及时 + 迟到)`，以及及时样本的 `demand tag lookup 周期 − fill 周期` 分布。及时要求 demand 命中由该请求填入且仍驻留的行。迟到要求 demand 合并到该预取尚在途的请求。填充后未使用就逐出另行计数。 |
| Coverage | 可对齐 demand ID 时，用 `由预取转为及时命中的基线 LLC miss 数 / 基线 LLC demand miss 数`。需记录能跨策略匹配的 demand 标识，按核统计匹配率。 |
| 净 miss 变化 | `(基线 demand misses − 策略 demand misses) / 基线 demand misses`。这个比率同时受命中收益、污染和跨层请求流变化影响，可为负。 |
| LLC hit rate / miss rate | 分母为实际到达 LLC 的 demand data LOAD 请求数，命中率和缺失率按同一套 LLC tag 检查口径计算。并列报告访问数，解释分母随策略的变化。 |
| LLC load MPKI | `1000 × LLC demand load misses / 退休指令数`。整组用两核 miss 和退休指令数分别求和再相除。 |

报告各核 IPC 和窗口周期数 `IPC_i = I_i / C_i`。固定模拟频率 `f` 时，各核所需时间为 `C_i/f`。两核共享测量起点时，整组完成时间由最晚完成的核决定。合计 IPC 可作为吞吐的补充，同时报告各核相对同工作负载基线的速度比和最慢核速度比，便于识别某一核承受的额外竞争。若单独运行过每个程序，可另算以单独运行 IPC 为参照的 weighted speedup。记录模拟器是否让先完成的核继续产生请求，并在各策略使用同一设置。模拟器在 Sacramento 的运行时间与 Colab 训练时间分别记录。

## Roofline 与访存延迟

原始 Roofline 的上界是 `P ≤ min(P_peak, B_sustained × operational intensity)`，强度以每 DRAM 字节的有效浮点运算数表示。602 以整数工作负载为主，实验可画清楚标明单位的吞吐图，横轴用共同时间窗口内的退休指令/DRAM 字节，纵轴用退休指令/模拟周期。用同一模拟配置下测得的可持续 DRAM 带宽（字节/周期）与指令吞吐上界画线。字节由控制器实际传输的 demand read、prefetch read、writeback 等数据量累加，并分项报告。缓存命中、队列合并与取消会使提交的预取请求数和实际 DRAM 传输数不同。 [Williams、Waterman、Patterson，Roofline](https://escholarship.org/uc/item/5tz795vq)

延迟单独分析。记录 LLC demand miss 到返回或填充的 p50/p95、等待最老 load 的退休停顿周期、demand MSHR 并发数和 LLC/DRAM 队列占用。稳定窗口中，用同一个请求集合计算 `完成请求/周期 ≈ 平均在途请求数 / 平均请求驻留周期`，可检查吞吐是否受访存并发度限制。结合停顿和带宽利用率，可区分访存依赖造成的低并发与共享内存排队拥堵。 [Cimple §3](https://commit.csail.mit.edu/papers/2018/kiriansky-pact18-cimple.pdf)

## LSTM 在预取路径上的时间成本

第一版单层 LSTM 以 `D=5` 个输入、`H=8` 个隐藏单元和 `K=3` 个输出计算，每次状态更新的矩阵部分是 `4H(D+H)+HK = 440` MAC。门的激活、逐元素状态更新、偏置、状态读写和特征计算另计。每个核保留自己的 `h,c` 共 `2H=16` 个状态值并共用权重。每核的下一次递推依赖前一次状态，多个核可交错执行。若每次到达都重新计算长度 8 的窗口，矩阵部分约为 `8×416+24 = 3,352` MAC。 [Learning to Forget: Continual Prediction with LSTM](https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf)

预取与 demand 并行。模型从 L2 demand miss 到达 LLC 时起算，等模型排队、执行和预取队列处理完成后才能向内存发出请求。目标数据需要在对应 demand 查询 tag 时已填充且仍驻留。先做零周期理想对照，再以有界模型队列和正推理延迟扫描 IPC、及时率与丢弃率。可用 `0、8、16、32、64` 模拟周期作初始敏感性点，并单独改变推理启动间隔。周期点是实验参数，实际硬件估计需要量化格式、每周期 MAC 数、门函数实现和状态读写时序。例如 32 MAC/周期的计算单元，矩阵运算本身每事件至少需要 14 周期，未计门之间的依赖和状态开销。
