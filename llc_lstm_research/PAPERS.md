# ML data prefetching papers

This index covers the CPU hardware data prefetching and control papers collected through November 12, 2026. It includes workshop papers and preprints. The papers try several different things: predict a line, score candidates, coordinate policies, or make inference cheap enough to use.

## The two Google papers

- **Delta-LSTM, ICML 2018.** Embeds load PC and miss delta, then predicts the next miss delta with a two-layer LSTM. The paper evaluates offline top-10 precision and recall. [Learning Memory Access Patterns](https://proceedings.mlr.press/v80/hashemi18a.html)
- **Voyager, ASPLOS 2021.** Predicts pages and offsets with two LSTMs. Uses a 16-event history and several future targets, including the next access, next same-PC access, and nearby lines. It observes and fills the LLC in simulation. Its IPC results omit prediction latency. [Paper, §§4–5](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)

## Predict an address or offset

- **Semantic locality, TACO 2019.** Uses a small online neural network to learn from program and machine context. [Peled et al.](https://doi.org/10.1145/3345000)
- **Shorter and more transferable LSTMs, 2019–22.** Compact ML Prefetcher compresses the output vocabulary. MemMAP and C-MemMAP adapt compact models to related traces. A microbenchmark study varies history length and concurrent streams. [Compact ML Prefetcher](https://alazaris.github.io/assets/memsys_2019.pdf) · [MemMAP](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206302/) · [C-MemMAP](https://doi.org/10.1007/s41060-021-00268-y) · [History study](https://par.nsf.gov/servlets/purl/10187649)
- **Hierarchical Sequence Model, 2019 workshop.** Early version of Voyager's hierarchical address model. [Paper](https://www.cs.utexas.edu/~lin/papers/mlsys19.pdf)
- **Attention and bitmap targets, 2021–24.** TransforMAP represents addresses in binary. TransFetch predicts a bitmap of future deltas. Drishyam encodes page and offset history as images. A2P applies attention and delta bitmaps at the LLC and tests a hybrid with ISB. [TransforMAP](https://arxiv.org/abs/2205.14778) · [TransFetch](https://arxiv.org/abs/2205.02269) · [Drishyam](https://www.cse.iitb.ac.in/~biswa/PACT23.pdf) · [A2P](https://doi.org/10.1007/s42979-024-02989-w)
- **Workload-specific models, 2023–25.** MPGraph and GraFetch specialize in graph workloads, where program phases and access patterns change. [MPGraph](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap202.html) · [GraFetch](https://doi.org/10.1109/TPDS.2025.3575106)

## Use ML with conventional predictors

- **PPF, ISCA 2019.** SPP generates candidates and a perceptron filters them. Demand use and unused eviction provide feedback. [Paper](https://people.engr.tamu.edu/djimenez/pdfs/ppf_isca2019.pdf)
- **RAOP, MEMSYS 2020, and MPMLP, 2021 workshop.** RAOP combines a compact LSTM with Next-Line or Best-Offset. MPMLP uses an LLC MLP alongside Best-Offset. [RAOP](https://www.memsys.io/wp-content/uploads/2020/10/p374-zhang.pdf) · [MPMLP](https://people.ece.ubc.ca/mieszko/ml-prefetcher-mlarchsys2021.pdf)
- **Twilight and T-LITE, ISCA 2024.** Rank a small set of temporal successors. T-LITE uses clustering and quantization to reduce cost. Twilight includes 4- and 8-core mixes. [Paper](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)
- **TLP, 2024.** Uses perceptrons for off-chip prediction and L1D prefetch filtering. [Paper](https://arxiv.org/abs/2403.15181)
- **PF-LLM, ASPLOS 2026.** Generates hints offline from assembly around loads. A lightweight hardware policy consumes them. [Paper](https://doi.org/10.1145/3779212.3790202)
- **Best-Offset, DPC2.** Learns a useful offset by checking whether it could have generated the current line with enough lead time. This is a conventional control for a timing-aware LSTM. [Paper](https://comparch-conf.gatech.edu/dpc2/resource/dpc2_michaud.pdf)

## Choose actions online and coordinate cores

- **Pythia, MICRO 2021.** Learns which offset to issue or when to abstain. Rewards distinguish useful timely and late requests and account for bandwidth. [Paper](https://arxiv.org/abs/2109.12021)
- **Policy selection, 2019–24.** Researchers used contextual bandits, random forests, deep RL, ensembles, and phase-dependent decision trees to select prefetchers and settings. [Hiebel et al.](https://digitalcommons.mtu.edu/michigantech-p/567/) · [Puppeteer](https://bu-icsg.github.io/publications/2022/puppeteer_taco_2022.pdf) · [Managing Prefetchers With Deep RL](https://www.computer.org/csdl/journal/ca/2022/02/09904841/1H2kY3Sjt0A) · [ReSemble](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap334.html) · [Alcorta et al.](https://ut-sysml.ece.utexas.edu/publications/prints/cal2024_alcorta.pdf) · [RL-CoPref](https://link.springer.com/article/10.1007/s11227-024-05938-9)
- **Shared resources, 2023–26.** Micro-Armed Bandit selects L2 configurations. Micro-MAMA and CRL-Pythia coordinate cores. Athena coordinates prefetchers with an off-chip predictor. A DPC4 entry found that its static LLC prefetchers slightly hurt single-core performance when strong upper-level prefetchers were active. [Micro-Armed Bandit](https://iacoma.cs.uiuc.edu/iacoma-papers/micro23.pdf) · [Micro-MAMA](https://iacoma.cs.uiuc.edu/iacoma-papers/micro25_2.pdf) · [CRL-Pythia preprint](https://arxiv.org/abs/2509.10719) · [Athena](https://arxiv.org/abs/2601.17615) · [DPC4 entry](https://chaseblock.com/assets/pdf/dpc4.pdf)
- **CHROME, HPCA 2024.** Uses core and PC context to distinguish mixed streams. It coordinates LLC replacement, bypass, and pattern-based prefetching with online RL. [Paper](https://xiaoyang-lu.github.io/files/HPCA2024/CHROME_paper.pdf)

## Reduce the cost of learning or inference

- **DART and Net2Tab, 2024–26.** Distill or convert neural models to tables for faster lookup. [DART](https://arxiv.org/abs/2401.06362) · [Net2Tab](https://www.sciencedirect.com/science/article/pii/S0743731526000432)
- **BTCP and Moirai, 2025–26.** Use binary temporal convolutions or a compact binarized network. Moirai targets L1D. [BTCP](https://ieeexplore.ieee.org/document/11062847/) · [Moirai](https://ieeexplore.ieee.org/document/11617757/)
- **Pathfinder and SNAP, 2024–26.** Explore spiking networks and low-cost online updates. [Pathfinder](https://users.cs.utah.edu/~rajeev/pubs/asplos24.pdf) · [SNAP](https://www.sciencedirect.com/science/article/pii/S138376212600370X)
- **Joint caching and prefetching, 2025 workshop.** Explores shared representations for replacement and prefetch prediction. [Paper](https://mlforsystems.org/assets/papers/neurips2025/paper55.pdf)
- **Augur, November 2026 preprint.** Uses short delta history, reservoir features, and online updates at L2. Tests 1, 4, and 8 cores. Its single-core context sweep found little extra benefit beyond a short window. [Paper, §VI-E](https://arxiv.org/html/2611.10929v1#S6.SS5)
