# ML 数据预取论文

- 检索到 2026-11-12，沿 Google 论文引用、体系结构会议、MLArchSys workshop、期刊和 arXiv 补查。索引收录 40 个论文条目，含 workshop 和预印本，覆盖主要路线。范围限 CPU 硬件数据预取和相关控制。
- 我们最先借用三件事。用未来 demand 给少量候选打标签，用传统算法生成候选，用模拟器记录候选何时填充、何时被使用。 [Voyager §4.4](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf) · [PPF §3](https://people.engr.tamu.edu/djimenez/pdfs/ppf_isca2019.pdf) · [Pythia §4](https://arxiv.org/abs/2109.12021)

## Google 两篇

- **Learning Memory Access Patterns · ICML 2018**。把 load PC 和 cache-miss 地址差做 embedding，送进双层 LSTM，分类预测下一条 miss 的 delta。离线看 top-10 precision/recall。我们可以先借用相对地址表示和历史序列，另用模拟器测请求时序与 IPC。 [论文 §3–5](https://proceedings.mlr.press/v80/hashemi18a/hashemi18a.pdf)
- **Voyager · ASPLOS 2021**。PC、页号和页内 offset 进入层次模型，两个单层 LSTM 分别预测页和 offset，历史长度 16、每层 256 个隐藏单元。标签来自未来访问，包括全局下一次、同 PC 下一次、同 basic block 下一次、空间邻近和未来十次中最常出现的地址。用 BCE 训练多个正例，默认发最高分的候选。 [论文 §4、表 1](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)
- Voyager 在模拟中由 LLC 访问触发并填充 LLC，报告 accuracy、coverage 和 IPC。地址生成延迟和存储开销未计入 IPC 实验，Google search 和 ads 做了轨迹预测评估。复用结果时要对齐输入事件和预测成本。 [论文 §5.1、§5.4](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)

## 直接预测地址或 delta

- **Neural Network Prefetcher · TACO 2019**。从程序与机器上下文学习语义局部性，用在线前馈神经网络回归未来地址信息。 [论文](https://doi.org/10.1145/3345000)
- **Understanding Memory Access Patterns · AIDArc 2019 workshop**。用合成微基准比较不同历史长度和并发访问流，帮助拆开“历史不够长”和“流混在一起”两种影响。 [论文](https://par.nsf.gov/servlets/purl/10187649)
- **Hierarchical Sequence Model · NeurIPS 2019 workshop**。Voyager 的早期版本，用层次模型处理不规则地址空间。 [论文](https://www.cs.utexas.edu/~lin/papers/mlsys19.pdf)
- **Compact ML Prefetcher · MEMSYS 2019**。用二进制输出编码缩小 LSTM 的输出空间。 [论文](https://alazaris.github.io/assets/memsys_2019.pdf)
- **MemMAP · PAKDD 2020 / C-MemMAP · 2022**。先在相关应用上学紧凑的 LSTM，再为新应用微调。C-MemMAP 用 trace 聚类找可迁移的模型。 [MemMAP](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206302/) · [C-MemMAP](https://doi.org/10.1007/s41060-021-00268-y)
- **TransforMAP · MLArchSys 2021 workshop / 2022 预印本**。Transformer 输入二进制地址，标签是未来会访问的页内 offset bitmap，可以一次预测多条线。 [Workshop](https://sites.google.com/view/mlarchsys/isca-2021/accepted-papers) · [论文](https://arxiv.org/abs/2205.14778)
- **TransFetch · ACM CF 2022**。把地址切成较细的段，用 attention 预测未来 delta bitmap，按分数决定发多少条。 [论文](https://par.nsf.gov/servlets/purl/10376312)
- **Drishyam · PACT 2023**。把 page/offset 序列编码成图像式输入，用 Transformer 预测页和页内线。 [论文](https://www.cse.iitb.ac.in/~biswa/PACT23.pdf)
- **MPGraph · SC 2023 / GraFetch · TPDS 2025**。为图应用的不同阶段选择不同的地址或 delta 预测方式，常规访问模式还可交给规则。 [MPGraph](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap202.html) · [GraFetch](https://doi.org/10.1109/TPDS.2025.3575106)
- **A2P · SN Computer Science 2024**。用 attention 学 LLC 的未来 delta bitmap，也测试与传统 ISB 的组合。 [论文](https://doi.org/10.1007/s42979-024-02989-w)

## 与传统预取器结合，降低推理开销

- **PPF · ISCA 2019**。SPP 出候选，感知机按未来 demand 是否使用训练过滤器，并按置信度选择填 L2 或 LLC。 [论文](https://people.engr.tamu.edu/djimenez/pdfs/ppf_isca2019.pdf)
- **RAOP · MEMSYS 2020**。紧凑 LSTM 预测一个未来地址，Next-Line 或 Best-Offset 可从当前地址和预测地址出发继续生成候选。 [论文](https://par.nsf.gov/servlets/purl/10194054)
- **MPMLP · MLArchSys 2021 workshop**。在 LLC 用小 MLP 补 Best-Offset 难以覆盖的模式。 [论文](https://people.ece.ubc.ca/mieszko/ml-prefetcher-mlarchsys2021.pdf)
- **Twilight / T-LITE · ISCA 2024**。先用在线频率统计产生候选，再用神经网络打分，允许选择不发。T-LITE 用聚类和量化降低代价，Twilight 的论文还测试了 4 核与 8 核的混合负载。 [论文](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)
- **DART · IPDPS 2024 / Net2Tab · JPDC 2026**。把训练过的 attention 或其他神经模型蒸馏、转换成在线查表，缩短推理时间。 [DART](https://arxiv.org/abs/2401.06362) · [Net2Tab](https://www.sciencedirect.com/science/article/pii/S0743731526000432)
- **TLP · HPCA 2024**。在 L1D 用两级感知机结合 off-chip 预测与预取过滤。 [论文](https://arxiv.org/abs/2403.15181)
- **BTCP · IEEE Access 2025 / Moirai · ISCA 2026**。探索二值化的时序卷积和二值神经网络，分别用于不规则访问预测及低开销 L1D 预取。 [BTCP](https://ieeexplore.ieee.org/document/11062847/) · [Moirai](https://ieeexplore.ieee.org/document/11617757/)
- **PF-LLM · ASPLOS 2026**。离线从 load 的汇编上下文学预取提示，再由硬件执行轻量策略。 [论文](https://fact-lab.hkust.edu.hk/publications/conference-paper/2025/xu-2025-pf-llm/3779212.3790202.pdf)

## 在线学动作、配置和多核协调

- **Hiebel 等 · ICPP 2019**。用 contextual bandit 根据应用状态决定哪些硬件预取器开关。 [论文](https://digitalcommons.mtu.edu/michigantech-p/567/)
- **Pythia · MICRO 2021**。表式 RL 在候选 delta 和不发之间选动作，用 demand 是否晚于 fill 区分及时和迟到，并考虑带宽。论文主要用 L1 miss 训练、填 L2 或 LLC。 [论文 §3–5](https://arxiv.org/abs/2109.12021)
- **Puppeteer · TACO 2022 / Managing Prefetchers With Deep RL · IEEE CAL 2022 / ReSemble · SC 2022**。分别用随机森林、深度 RL 和基于 MLP 的 RL 选预取器配置或组合。 [Puppeteer](https://bu-icsg.github.io/publications/2022/puppeteer_taco_2022.pdf) · [Deep RL](https://www.computer.org/csdl/journal/ca/2022/02/09904841/1H2kY3Sjt0A) · [ReSemble](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap334.html)
- **Micro-Armed Bandit · MICRO 2023 / Micro-MAMA · MICRO 2025**。低成本 bandit 选 L2 传统预取器组合，后者加入跨核协调，处理带宽竞争和公平性。 [Micro-Armed Bandit](https://iacoma.cs.uiuc.edu/iacoma-papers/micro23.pdf) · [Micro-MAMA](https://iacoma.cs.uiuc.edu/iacoma-papers/micro25_2.pdf)
- **Alcorta 等 · IEEE CAL 2024**。在实际 many-core 机器上研究按应用阶段配置硬件预取器，用决策树做选择。 [论文](https://ut-sysml.ece.utexas.edu/publications/prints/cal2024_alcorta.pdf)
- **RL-CoPref · Journal of Supercomputing 2024**。用 prefetch hit/miss 与内存带宽作为反馈，动态调传统预取器的开关和 degree。 [论文](https://link.springer.com/article/10.1007/s11227-024-05938-9)
- **CRL-Pythia · 2025 预印本 / Athena · HPCA 2026**。前者扩展 Pythia 的跨核协调，后者用 RL 协调预取器和 off-chip predictor。 [CRL-Pythia](https://arxiv.org/abs/2509.10719) · [Athena](https://arxiv.org/abs/2601.17615)
- **Performance-Driven Composite Prefetching with Bandits · DPC4 2026 workshop**。在线 bandit 选择 L2 传统组合，也测试了上层预取器活跃时 LLC 的静态移植。在该单核设置下，移植的几种 LLC 策略都略微降低了性能。 [论文 §3.1](https://chaseblock.com/assets/pdf/dpc4.pdf)

## 其他学习方法

- **Pathfinder · ASPLOS 2024 / SNAP · 2026 期刊在线版**。用脉冲神经网络和在线学习捕捉不规则访问时序。 [Pathfinder](https://users.cs.utah.edu/~rajeev/pubs/asplos24.pdf) · [SNAP](https://www.sciencedirect.com/science/article/pii/S138376212600370X)
- **A Joint Learning Approach to Hardware Caching and Prefetching · NeurIPS 2025 workshop**。试验 Glider 和 Voyager 共享 encoder 或用对比学习对齐 embedding，结果是预取和替换联合学习的初步证据。 [论文](https://mlforsystems.org/assets/papers/neurips2025/paper55.pdf)
- **Augur · 2026-11 预印本**。用短历史 delta 的哈希特征做 reservoir，一次点积预测下一条 delta，在线 SGD 更新。报告了 1、4、8 核评估。单核窗口从 4 扫到 64，长窗口的边际收益有限。 [论文](https://arxiv.org/abs/2611.10929)

## 用在这次 LLC 实验

- **标签**从同一次 NoPF 运行取候选行的第一次后续 demand，要求它在参考中 LLC miss 且提前量合适。闭环用 issue、fill、demand 和 eviction 检查是否及时命中。teacher imitation 可以作为对照。 [Voyager §4.4](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf) · [Pythia §4](https://arxiv.org/abs/2109.12021)
- **模型**先比较传统打分、小 MLP 和单层 LSTM。先用每核 delta、页内 offset 和到达间隔，逐项加同 PC 标记和共享 LLC MSHR 占用率。历史长度和未来标签窗口分别调。 [历史窗口实验](https://par.nsf.gov/servlets/purl/10187649) · [T-LITE](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)
- **多核**把更多训练样本、访问流交错和资源竞争分开测。固定每组训练事件数，两版均保留每核历史、共用权重，对比加入共享占用率前后的差异。在 held-out 上同时看预取准确率、覆盖率、迟到率、LLC demand hit/miss、MPKI 和每核 IPC。 [Micro-MAMA](https://iacoma.cs.uiuc.edu/iacoma-papers/micro25_2.pdf)
