# ChampSim-Pythia 的接入位置

## 已核实的源码

本研究以官方 Pythia v1.3 为参考。仓库中保留的历史 simulator 快照具有相同的基本连接，但其中含有旧实验的测量和预取改动。后续实现从官方版本单独构建，固定版本和配置。构建脚本的参数顺序是 `L1D L2 LLC 核数`，例如 `./build_champsim.sh no no no 2` 代表双核且关闭 L1D、L2 和 LLC 预取。官方 `prefetcher/` 中唯一的 `.llc_pref` 文件是 `no.llc_pref`，它提供空回调。`multi.l2c_pref` 才有可选的 Next-Line、BOP、Stride、SPP 和 Scooby（Pythia）等实现。在 LLC 使用传统策略时，要把它接到同一个 LLC 输入与填充路径，并用同一请求预算比较。 [构建脚本](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/build_champsim.sh) · [官方目录清单](https://api.github.com/repos/CMU-SAFARI/Pythia/contents/prefetcher?ref=v1.3) · [LLC 空实现](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/no.llc_pref) · [L2 实现](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/multi.l2c_pref)

| 位置 | v1.3 的行为 | 在实验中的用途 |
| --- | --- | --- |
| [`src/main.cc` 的缓存连接](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/main.cc#L775-L789) | 每核的私有 L2 向同一个 `uncore.LLC` 转发请求。 | 使用请求自带的 `PACKET.cpu` 选择 per-core LSTM 状态。跨核调度可以共用模型参数。 |
| [`CACHE::add_rq`](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1261-L1389) | 读请求入口。先检查写队列转发，再判断读队列合并与容量，最后入队。LLC tag lookup 随后在读队列处理时发生。 | 在 LLC 成功接收一条新的 L2 数据 demand LOAD 后旁路采样。给入队事件分配稳定 ID，保留对转发、合并和拒绝的独立计数。 |
| [LLC 需求回调](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L610-L626) | `llc_prefetcher_operate` 在 hit 时已得到 LLC 查找结果，miss 时要等到 [`miss_handled`](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L817-L841)。更上级发来的预取也可能走 [`PREFETCH` 调用分支](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L914-L927)。 | 作为需求结果和测量钩子。仅替换 `.llc_pref` 的默认 demand 回调会把预测放在 lookup 之后。 |
| [`CACHE::prefetch_line`](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1442-L1475) | 构造带 `cpu`、物理目标地址和 `pf_metadata` 的 prefetch 包，再调用 `add_pq`。队列有空间时其返回值为 1，底层 `add_pq` 还会做写队列转发和重复合并。 | 以 LLC 为发起及填充层，使用 `FILL_LLC`。区分调用、唯一 PQ 入队、合并和实际 DRAM 请求。 |
| [填充](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L214-L229) · [首次命中](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L642-L657) · [迟到](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L787-L806) · [逐出](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1124-L1155) | 原生路径已有填充、预取块被访问、需求与 prefetch MSHR 合并、未使用逐出的观测点。 | 记录实际请求结果。需要额外维护候选 ID，并记录需求类型，才能把某次 LLC data-load hit 归给某个预取候选。 |

## 输入与推理时序

首轮固定两核各执行一份 602 trace，并关闭 L1D 和 L2 prefetch。每个 L2 已翻译的 demand data LOAD 在成功进入 LLC read queue 时，预测器旁路取得来源核、PC、物理行与到达时刻。用 `type == LOAD && !instruction` 排除指令读取，同时排除 RFO、writeback 和硬件 prefetch。初版历史按新入队的物理请求更新一次。被 RQ 合并的请求单独记录，以便核对同一物理请求有多个消费者的情况。拒绝后重试的请求在成功入队时才进入历史。两份独立进程 trace 即使访问同一虚拟地址，也不能据此假定它们访问同一物理页。 [物理 LOAD 包来源](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/ooo_cpu.cc#L1394-L1412) · [LLC RQ 路径](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1261-L1389)

LSTM 在 LLC ingress 旁路更新该核状态，当前需求继续查 tag。推理完成后，候选请求等到其模拟 ready cycle，再经过物理页边界、重复、驻留、在途和队列容量检查。预测时当前需求的 LLC hit/miss 还未知。延迟模型同时记录预测器的服务时间与启动间隔。同一核的下一次状态更新等待上一状态就绪，预测队列溢出时记录丢弃的预测，需求继续执行。共享版使用入口当时已经可读的 MSHR 占用率。每核消融版将该输入置零。原生需求回调临时设置 `CACHE::cpu = read_cpu`，延迟发行不能依赖该字段仍保留原值。候选需要保存发起核，并在发行时按实际 API 恢复请求的归属。实际消费事件可来自任何核，归因依据是同一物理行和预取请求 ID。 [需求回调与核归属](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L818-L841) · [预取请求构造](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/src/cache.cc#L1442-L1475)

## 实现后要验证的事件

- 入口事件保存 `event_id, cycle, cpu, pc, physical_line, rq_result`。另以 `event_id` 关联同一运行的 LLC 查找结果和时间。跨策略配对另带 core、原 trace 的动态指令序号和 load 操作数位置，并记录匹配率。按实际进入顺序处理同 cycle 事件。预热可更新 LSTM 状态，ROI 前后分别标记，避免窗口末尾尚未观察到的目标被标为无用。
- 每个候选保存 `candidate_id, anchor_event_id, source_cpu, target_line, ready_cycle`，以及被阈值或队列抑制、合并、进入 PQ、发往 DRAM、填入、首次有效需求命中、迟到合并、逐出等结果。多个候选指向同一在途目标时，只给实际促成填入的请求计一次有效预取。
- 度量 LLC data LOAD 命中率与 MPKI 时固定计数口径。保留原始 miss 数，并用相对 NoPF 的 MPKI 降幅报告净效果。逐请求覆盖率需要额外核对跨运行的需求标识。与 NoPF、传统候选策略和 LSTM 比较时，输入位置、请求限制和 LLC 填充级别需要一致。初次联调分别检查各核训练和模拟事件、候选的模拟 ready cycle、fill-before-demand 与 run-end 未完成项。

本文件记录接入设计和源码核对结果。代码实现与模拟数据采集属于后续工作。
