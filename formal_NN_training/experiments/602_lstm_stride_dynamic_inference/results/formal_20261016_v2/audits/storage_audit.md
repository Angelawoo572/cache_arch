# 存储、流量与 native SPP 资源核对

对应正式协议 `frozen_formal_20261016_v2`，构建 `build_20261016_185845`。本核对读取实际 Sacramento 源码和正式行参数；未修改活动构建或实验。结构化逐项数据见 [storage_audit.json](storage_audit.json)，服务器实际 `sizeof` 输出和可复现核对程序在 [storage_source/](storage_source/)。

## 可直接报告的数字

| 方法 | 固定设计源字段预算 | 加共同 cap 控制 16B | 边界 |
|---|---:|---:|---|
| NoPF | 0B 额外预测器状态 | 无输出请求 | 共享 CPU/cache/DRAM 不在此表 |
| NN H8/state64 | **16,920B** | **16,936B** | 16,808B 核心数组/event/scratch + 112B 显式 controller schema |
| legacy / remove-current / forward1 / forward2 Stride | **2,136B** | **2,152B** | 2,064B 既有 table/control + 56B scalar scratch + 16B 输出地址 |
| native SPP | **5,938B** | **5,954B** | 5,554B 持久表 + 256B 原生候选数组 + 128B scalar/control schema；所列8个正式情形已通过guard，非一般静态证明 |
| native SPP C++ 对象 `sizeof` | **33,432B** | 不适用 | 宿主对象表示，含诊断控制；与硬件源字段预算不可互换 |

这些边界应并列披露。不能把 SPP 的 33,432B C++ 对象与 NN 的源字段架构预留直接当成相同实现层次比较。每个正式行的 `host_time.txt` 给出整个模拟器 RSS，含代码、trace reader、共享缓存、allocator 和诊断数据；也不能当预测器 SRAM。

NN 的逐项计算为：7,632B 权重 + 6,144B 状态表 + 1,344B scratch + 24B active metadata + 512B decoded delta/address + 384B input FIFO + 768B output FIFO = 16,808B。状态项是每 PC 32B tag/controller/lifetime 预留及 64B FP32 h/c。active metadata 的旧 h/c 和完成前 staged h/c 已在 scratch 预留里，避免重复收费。未配置容量的 unbounded control 没有有限最大值；若相加各项峰值，只能称峰值包络。

Stride 源码每个 `Tracker` 实为 PC uint64、line uint64、stride int64，服务器 `sizeof(Tracker)=24B`；产生的 stride 先在 int32 运算中截断。服务器 `sizeof(StridePrefetcher)=192B`，其中 deque 控制 80B、诊断统计 72B；另有 tracker 对象、deque 指针块和 allocator 开销。64-entry tracker payload 为 1,536B，不能漏掉 LRU/valid/controller 后把它称完整硬件预算。四种动作政策不需要不同数量的算法 tracker。forward helper 的重复 tracking 是当前宿主实现表示；最小架构可从同一 tracker 取得 stride，宿主 RSS 则包含实际重复表示。

## SPP 的完整持久字段

| 组件 | 容量和保留位宽 | 打包字节 | C++ 字节 |
|---|---|---:|---:|
| ST | 256 × (valid1 + tag16 + last offset6 + signature12 + LRU8) | 1,376 | 4,352 |
| PT | 512 × 4 × (signed delta7 + count4) + 512 × signature count4 | 3,072 | 18,432 |
| Filter | 1,024 × (remainder6 + valid1 + useful1) | 1,024 | 10,240 |
| GHR entries | 8 × (valid1 + signature12 + confidence32 + offset6 + delta7) | 58 | 136 payload |
| Global feedback | pf_useful64 + pf_issued64 + global_accuracy64 | 24 | 24 |
| 合计 | 所有持久决策字段 | **5,554** | **33,184** |

有源码 mask 或页内 delta 范围证明的字段按对应位宽打包；confidence 没有显式 clamp，故保留 uint32。三个 global feedback 值会影响预测，属于决策状态，不能当诊断计数器排除。现成 `estimate_prefetcher_storage.py` 算出 5,508B，但漏了这三个计数器，把 GHR confidence 假设为 10-bit，并漏掉候选数组；因此不能当完整存储结果。

`confidence_q[32]` 和 `delta_q[32]` 原生 payload 共 256B。若仅按精确值压缩 delta 为 7-bit 而保留 confidence32，则为 156B，但本轮没有实现这种压缩。算法没有新的 SPP finite-cost 服务模型，hash/表访问/控制端口、标量寄存器、SRAM macro padding/ECC 等无法从 C++ 类大小得到，仍须标为未建模。

## 诊断排除和共同预算控制

NN 主设计的部署数排除 shadow teacher、teacher 地址快照、overlap categories、时间戳、全事件 histogram、PC 更新/eviction 统计、CSV occurrence map、cohort bookkeeping、文件与 allocator，以及离线训练/optimizer。native SPP 的 delta/depth histogram 同属诊断。silent-only 消融的 teacher 输出 gate 是研究中的因果对照控制，不是免费加入部署主设计的功能。

共同 request-credit gate 会影响实际输出，不能把它的全部状态称诊断：它至少需要一个 uint64 allowed-request counter，额外 **8B**；若 cap 通过寄存器配置再加 8B，retired count 复用已有 CPU 计数。该公共开销尚未计入上方各预测器数，应对所有预算匹配方法统一另列。仅记录、不参与输出选择的计数器仍排除。上方汇总采用两项合计16B，以保守覆盖共同可编程预算 gate。

## 权重流量的准确含义

当前 `inference_weight_read_bytes = 4 × inference_macs` 是**分析性的 FP32 dense matrix 权重操作数读取量**，并非实际 SRAM 事务追踪，也不含 bias 读取。成功 forward 若逐次计算源代码 bias 加载，还会有 `4 × [9H + 2 + e + K(6H + 1)]` 字节。当前服务模型假设 dense arithmetic 与 dense-weight 读取按较慢者重叠，随后串行计入非线性、control 和 scratch；bias 的端口、能耗和传输没有另外收费。必须把有限引擎结论限定在这些明确假设下。

FP32 权重有一个 resident predictor 存储副本，本轮模型的 NN 权重 DRAM 字节为 0；不能将上述分析读取量加到实际 LLC↔DRAM 流量，也不能用 issued×64 替代 DRAM 数据。真实 DRAM 计数另由 controller enqueue/merge、bus start、带完成时间的 bus completion 得出，末端在途与初始队列明确保留。

## SPP 的输入、阈值和 lookahead 边界

正式 SPP 命令加载 `dynamic602`（method none，负责共同观测）及 `spp_dev2`。实际日志与已编译 knobs 均为 fill threshold **90**、prefetch threshold **25**；没有额外偷偷调参。ST=256、PT=512×4、filter=1024、GHR=8。

L2 demand operate 回调只来自 LOAD hit/miss；RFO 不会触发该 operate。PREFETCH operate 只在请求来自上一级 cache 时发生，相同 L2 的自产 SPP 请求不会递归触发；本正式构建 L1 prefetcher=none。除此以外，native SPP 还收到 MSHR/writeback 的 cache-fill 回调，含 demand/RFO/PF/writeback；实现使用 evicted address 更新 filter 和 global feedback。因此 native SPP 获得比 NN 的 PC/address + h/c 更丰富的 fill/eviction 信息，不能宣称所有算法输入完全相同。公平要求在本轮是共同机器、数据与 cache warmup、ROI、page policy 和请求预算，并披露各原生 baseline 的反馈边界。

原生 `read_pattern` 对两个32-entry候选数组的pf_q_tail写入没有源级边界判断，lookahead do/while没有显式深度cap。depth/breadth统计自身不能证明数组安全，因为same-page breadth不含跨页GHR候选。

已完成独立guarded simulator的**8/8个完整正式重复**：602两个独立ROI、605与623，各含原策略及共同cap。每项均为5M cache warmup +25M ROI。guard在confidence append、delta append、terminal confidence write三个写入点的非法索引使用前抛错；合法范围内不改变行为。八项均**0次guard trip**，且在原矩阵全部完成后重新比较，每项与对应原始结果的**210个summary字段完全一致**，总计1,680项精确字段比较。此前605原策略的`original_not_yet_complete`已刷新为实际完成后的精确比较。

该证据是对**所列完整测试情形**的数组写入边界复核，不是所有trace/state/configuration的通用静态证明；原生源码中未设tail guard和未显式限制lookahead的事实仍须披露。它也没有建立nativeSPP的有限计算吞吐或RTL实现。原始矩阵、活动引擎、配置及数据均未修改，guarded build未复制进交付证据目录。完整证书见[spp_bounds_certificate.json](spp_bounds_certificate.json)，逐项proof/log位于`research_output/raw_v2/correctness_repetitions/spp_bounds/`。JSON旧字段`array_safety_verified=false`明确仅指未获得一般源码证明；另列`array_safety_verified_for_listed_formal_runs=true`与8/8 PASS，不能把它误读为未做过正式数据复核。


## 补齐 controller/ring primitive 的源字段预算

活动引擎原始计数的 **16,808B** 应准确标为 NN 的核心数组、事件 payload 和 forward scratch 预留。活动引擎原始计数器的历史名字是 `deployment_bytes_configured`，但它没有另列 ring head/tail/count 与串行引擎控制字；保留原数据，并用下表补齐边界，不改变正在运行的模型。

采用保守 **64-bit word** 分配，而不猜测优化 RTL 的最小位数：

| NN controller 项 | 64-bit words |
|---|---:|
| service due/remaining | 1 |
| next output release | 1 |
| state allocation lifetime | 1 |
| global LRU timestamp | 1 |
| state occupancy | 1 |
| input FIFO head/tail/occupancy | 3 |
| output FIFO head/tail/occupancy | 3 |
| predicted K / decoded valid count | 2 |
| active/failed/page/queue flags 的预留字 | 1 |
| 合计 | **14 words = 112B** |

因此，固定资源参数硬连线时，NN 的完整**源字段预留模型**为 **16,920B**。预算匹配行统一再列 allowed counter64 + programmable cap64 的 **16B**，得到 **16,936B**。若还要求动态配置 MAC、state/decoder/input/output 容量、nonlinear 周期和 bandwidth，这七个 64-bit 配置字另需 **56B**；本固定设计表不默认收取可编程资源功能。warmup/history、teacher 消融与仅诊断的计数不当部署状态。

同一层次补齐 conventional 方法：Stride 既有 16B controller 可分配 matched-slot64、degree32、emitted-count32；再列输入 PC/address、当前 line、stride、offset、candidate offset、page 和 target 共 52B scalar payload，按字取整预留 **56B**，即时双地址结果 **16B**。完整固定设计源字段预算为 **2,136B**，预算匹配为 **2,152B**。

SPP 在 5,554B 持久表及 256B 两个候选数组之外，保守列 **128B scalar/control reservation**：caller 88B，最大嵌套 helper 40B。caller 包括当前 input/page/base、signature/delta/offset、候选 head/tail、depth、lookahead confidence/way、循环下标、目标地址及 flag；helper 最大值覆盖 ST 28B + GHR 12B，或 filter 32B + hash key8B，PT 小于此包络。诊断 breadth/depth histogram 不在其中。故固定设计源字段预算为 **5,938B**，共同预算 gate 后 **5,954B**；若两个 threshold32 也需可编程再加 **8B**。所列8个完整正式情形已通过guard复核；不扩张为通用源码安全证明。

这些数字给出完整、可审查的**源字段存储 schema**，并明确复用与 reserve；它们不是新的 RTL、SRAM macro 面积、管线端口或时序测量。native SPP/Stride 仍是功能 baseline，未突然获得已验证的有限计算引擎。

服务器对 NN 宿主对象另实测 `sizeof(Dynamic602)=2,152B`、Event136B、Entry72B、Output32B、RecurrentState48B、InferenceResult136B。这些值含 vector/map/stream 控制及 diagnostic 字段，且排除其动态 allocation；与上述硬件源字段预算没有可直接互换的关系。
