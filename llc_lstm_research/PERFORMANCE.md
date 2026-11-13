# 怎么判断预取是否有用

- **最后看执行时间。** 固定测量指令量和频率，逐核报告 `IPC = 指令数 / 周期`、完成周期和相对同配置基线的速度比。多核同时开始时，另报最晚完成的周期与最慢核速度比。Colab 训练时间和 Sacramento 模拟用时各自记录。
- **Accuracy**。对已结束的独立预取请求，统计 `(及时使用 + 迟到合并) / 已结束请求`。同时报只算及时命中的比例。候选数、排队数、独立请求数和运行尾部未决数分别记录。
- **Timeliness**。统计 `及时 / (及时 + 迟到)`，并报告 `demand 查 tag 周期 − fill 周期` 的分布。及时要求 demand 查 tag 时该行已被这次预取填入且仍驻留。迟到指 demand 与在途预取合并。提前填入后未使用就逐出的请求单独计。 [Pythia §3–4](https://arxiv.org/pdf/2109.12021) · [Feedback Directed Prefetching](https://users.ece.cmu.edu/~omutlu/pub/srinath_hpca07.pdf)
- **Coverage 和净效果**。跨策略能配对同一 demand 时，coverage 是基线 miss 中由预取转成及时 hit 的比例，并报告配对率。用 hit rate、miss rate 和 `LLC data-load MPKI = 1000 × LLC data-load misses / 退休指令数` 看整段净变化。整组按总计数算，另报逐核值。预取也可能带来新 miss。
- **流量按实际 DRAM bytes 算**。分别数 demand read、prefetch read 和 writeback。先用相同发出上限做机制对照，再在相近流量下看谁的 IPC 更高。每次候选数相同也可能产生不同的实际 DRAM 流量。

## Roofline、等待和 LSTM 成本

- **Roofline 看吞吐。** 原式为 `P ≤ min(P_peak, B_sustained × intensity)`。原论文用 FLOP/DRAM byte。对以整数为主的 602，可另画单位清楚的指令吞吐图，用共同时间窗内的退休指令/DRAM byte 和退休指令/cycle，以实测持续带宽画上界。 [Williams 等，Roofline](https://escholarship.org/uc/item/5tz795vq)
- **访存等待单独看。** 记 LLC miss 延迟 p50/p95、最老 load 等待造成的退休停顿、MSHR 并发和 DRAM 队列占用。稳定窗口内，同一请求组的完成速率约等于平均在途数除以平均驻留时间。这能区分并发不足和带宽排队。 [Cimple §3.2](https://commit.csail.mit.edu/papers/2018/kiriansky-pact18-cimple.pdf)
- **推理时间算进预取路径。** demand 继续执行，候选经历模型排队、推理、预取排队和内存返回。单层宽度 8、3 个输入和 3 个输出的 stateful LSTM 矩阵部分约 **376 MAC/事件**，两个额外特征都启用时为 **440 MAC/事件**。重放长度为 `W` 的窗口分别约为 `352W+24` 和 `416W+24` MAC。门函数、状态读写和排队另计。零时延作为理想对照，实际对照扫正时延和启动间隔，并报告预测队列丢弃率。
