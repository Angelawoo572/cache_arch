# 论文笔记

## 2018 · Learning Memory Access Patterns

Milad Hashemi 等，ICML 2018。论文先用三级缓存模型从程序轨迹抽出 cache miss，再把下一次 miss 的行地址差作为离散分类目标。Embedding LSTM 分别嵌入当前 load PC 与地址差，将两者拼接输入双层 LSTM，输出 delta 的概率分布。作者还尝试先对地址分区，再用共享权重的 LSTM 学各区内的 delta。 [Google Research](https://research.google/pubs/learning-memory-access-patterns/) · [论文 §3–5](https://proceedings.mlr.press/v80/hashemi18a/hashemi18a.pdf)

论文离线报告 top-10 precision 和 recall。一次预测只要十个候选里有正确的下一条 delta 就可以计中。这个评估没有测出预取能否在 demand 前填充、会占用多少带宽，或能提高多少 IPC。我们可以借用相对地址和循环状态，把输出范围限制在少数可发出的候选，并用模拟器单独测请求效果。 [论文 §4.1、§5.3](https://proceedings.mlr.press/v80/hashemi18a/hashemi18a.pdf)

## 2021 · A Hierarchical Neural Model of Data Prefetching（Voyager）

Zhan Shi、Akanksha Jain、Kevin Swersky、Milad Hashemi、Parthasarathy Ranganathan、Calvin Lin，ASPLOS 2021。Voyager 用 PC、页号和页内偏移表示访问，先计算有页上下文的偏移 embedding，再用两个单层 LSTM 分别预测页与偏移。论文在线训练，配置的历史长度为 16，每个 LSTM 有 256 个隐藏单元。常见地址可以直接预测，低频地址可以用 delta 表示。 [Google Research](https://research.google/pubs/a-hierarchical-neural-model-of-data-prefetching/) · [论文 §4、表 1](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)

它从同一轨迹取多种未来地址作为候选标签，包括全局下一个地址、同 PC 的下一个地址、同 basic block 的下一个地址、空间邻近访问，以及接下来十次访问中最常见的地址。多标签训练使用二元交叉熵，默认推理选概率最高的候选。这个做法提示我们可以让几个可执行候选各自有标签。第一版可先只判断少数同页候选是否值得发。 [论文 §4.4](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)

论文的 ChampSim 比较把预取器放在 LLC，以 LLC 访问作为输入并填充 LLC。它模拟了请求时序，报告 accuracy、coverage 和 IPC，但 IPC 比较没有计入生成预取地址的延迟或存储开销。论文报告的软件实现约需 18,000 ns 每次预测，作者认为硬件使用仍需降低延迟。Google search 和 ads 只做了轨迹预测评估，没有报告模拟 IPC。我们的 IPC 实验需要把预测延迟与排队纳入模拟。 [论文 §5.1、§5.4](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)

## 2021 · Pythia

Rahul Bera 等，MICRO 2021。这是另一篇论文，采用表式在线强化学习。每次访问从候选 delta 或不预取中选一个动作。Evaluation Queue 按候选记录动作和填充状态，demand 在填充前到达得到迟到反馈，填充后到达得到及时反馈。候选跨页、在观察期内无人访问、不发请求，各有对应奖励。无人访问和不发请求的奖励还受带宽使用水平影响。论文比较主要用 L1 miss 训练预取器，填充 L2 或 LLC。 [论文 §3、§4、§5](https://arxiv.org/pdf/2109.12021)

借鉴 Pythia 的请求结果记录方式，将发出、填充、首次 demand、逐出与队列合并对应到具体候选。及时性可由事件先后计算。要认定预取真正避免了 LLC miss，还需要基线访问结果与因果归因。训练先用可复核的候选标签，传统预取器可作为对照、候选来源或模仿基线。 [Pythia §4](https://arxiv.org/pdf/2109.12021) · [Perceptron-Based Prefetch Filtering §3](https://people.engr.tamu.edu/djimenez/pdfs/ppf_isca2019.pdf)
