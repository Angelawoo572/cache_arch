# 从 L2 迁到共享 LLC

- **输入固定为 L2D demand-load miss。** 在 `CACHE::add_rq` 成功新入队时在 L2→LLC 路径核实来源后，旁路读取 `PACKET.cpu`、物理行、周期，以及经核实可用的 load PC。当前请求继续查 LLC tag，命中结果随后记录。按包类型与指令标记排除取指、RFO、writeback 和预取。若上游把指令与数据合并，先核对并单列这种包。单独记录合并、转发和拒绝，避免重试重复更新历史。 [缓存连接](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/main.cc#L775-L789) · [读队列入口](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1261-L1389)
- **四种传统预取器都可以移植。** 官方 Pythia v1.3 的 LLC 只提供空模块，传统实现由 L2 的 `multi.l2c_pref` 调用。沿用它们的候选生成逻辑，把触发统一到 LLC 入口、把候选交给同一个发送器并填入 LLC。默认 `llc_prefetcher_operate` 在 LLC 查 tag 之后运行，可用来记需求结果。 [LLC 空模块](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/no.llc_pref) · [L2 调用层](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/multi.l2c_pref) · [LLC 回调](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L818-L841)

## 四种方法怎么移

- **Stride**。沿用按 PC 学习稳定地址差的逻辑，每核一份表。先核实 LLC 入口的 load PC；保留同页检查，重新调提前距离和度数。原实现从 `deg=0` 开始，会把当前行列为第一个候选；先排除当前行，再用第一个未来候选做 degree-1 对照。它适合做第一条能预测非 `+1` 的传统对照。 [Stride 源码](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/stride.cc)
- **Streamer**。沿用物理页内的正向、反向流追踪；每核独立，比较固定方向和可调提前距离。相较 Stride，对 PC 的依赖更少。 [Streamer 源码](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/streamer.cc)
- **AMPM**。沿用物理页内访问位图，保留其对访问顺序和 PC 的低依赖。先用每核独立的页表，记录每核容量和总容量；之后可单独比较共享页表与分区页表。 [AMPM 源码](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/ampm.cc) · [原论文](https://jilp.org/vol13/v13paper3.pdf)
- **SPP**。先保留签名和多步 delta 的预测逻辑。原实现直接从父缓存发请求，按置信度选择 L2 或 LLC，lookahead 数组使用 `L2C_MSHR_SIZE`，反馈还区分 L2 demand、逐出和两种填充等级。移植时要把候选发出、容量上限、重复过滤及 demand/fill/evict 反馈一起接到 LLC。单改 `FILL_LLC` 会改变 SPP 的有效性统计。 [SPP 源码](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/spp_dev2.cc) · [SPP 反馈](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/spp_dev2_helper.cc)

## 第一批对照

1. 双核 602 先跑 **NoPF、同页 Next-Line、Stride**。Next-Line 用于核对路径，Stride 检查同一输入上能否学到更长距离。先关闭上游 L1D/L2 预取。
2. 加一个只在 `+1/+2/+4` 中选的**偏移计数表**，与 LSTM 共用候选集合和发出预算。这项对照能检验 RNN 是否比简单的偏移统计带来更多收益。可借用 Best-Offset 的时序评分思路。 [Best-Offset](https://comparch-conf.gatech.edu/dpc2/resource/dpc2_michaud.pdf)
3. 接入 **Streamer、AMPM、SPP**，记录各自的候选度数、状态表容量、预取提前量和实际 DRAM 字节。在等请求预算下比较，再单列各策略经验证集调参后的结果。

## 采集口径

- demand 日志记入口和 tag lookup 的周期、物理行、来源核与合并状态。跨策略追踪同一 demand 时带原 trace 的动态指令序号和 load 操作数位置。
- 预取日志用候选 ID 串联申请、唯一入队、合并、发往 DRAM、填充、消费、迟到和逐出。发出前检查目标是否驻留或在途。`prefetch_line` 的成功返回包含部分队列合并，实际新请求以入队和下发事件计数。共享 LLC 允许任一核消费，按请求 ID 和物理行归因。 [预取发出](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1442-L1475) · [填充、迟到及逐出](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc)
- 这些是移植方案。四种 LLC 版本需要各自通过双核 smoke test，再进入性能比较。
