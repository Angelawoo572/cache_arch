# 602 结果入口

**本轮结果直接打开 [formal_study/results/602_width_templates](formal_study/results/602_width_templates/README.md)。**
本地、Sacramento、GitHub 的相对目录完全相同。24/24 新运行完成，9,748 项检查通过。

- [中文简洁报告及比较表](docs/602_width_templates/REPORT.md)
- [一页 paper story](docs/602_width_templates/PAPER_STORY.md)
- [结果与全部图](formal_study/results/602_width_templates/FINDINGS.md)
- [逐项完整数据（24 新运行＋64 兼容参考，保留各自协议）](formal_study/results/602_width_templates/per_run.csv)
- [24 项新运行](formal_study/results/602_width_templates/per_run_new.csv) · [配对比较](formal_study/results/602_width_templates/paired_comparisons.csv) · [审计](formal_study/results/602_width_templates/audit.json)
- [配置、模型与恢复运行](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/README.md)
- [旧 slides 勘误](docs/602_width_templates/SLIDE_ERRATA.md) · [旧 replay 对齐](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md)

本轮完整原始数据、日志、checkpoints、回调、snapshots 和模型：
`formal_study/runs/602_width_templates_20261017/`。
本地以 `/Users/angelawoo/Documents/CMU/ece/cache/` 为根；服务器以 `/home/qianruw/cache/` 为根。

## 保留的前轮结果

- [64 项四预取器 H8 live 正式结果](formal_NN_training/experiments/602_four_prefetcher_formal/results/formal_20261017/README.md)
- [current-line / gate / payload 研究](docs/602_gate_mechanism/REPORT.md)
- [首轮正式 Stride 轻量归档](formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/formal_20261016_v2/README.md)
- [原 offline 训练和 replay 各模型目录](formal_NN_training/experiments/)

历史完整 source/raw/models/builds/logs 保留在 `formal_study/archive/20261016/` 和
`formal_study/archive/20261017/`；四路线有缺陷的 v1 和修复后的 v2 保持独立。
大日志、traces、builds 和原始数据保存在本地及 Sacramento 的 cache 内，按研究要求不提交 GitHub。

健康 Git 现在就在 cache 根；分支为 `experiment/602-gate-mechanism`。
旧内层 checkout 和旧结果位置仅为兼容链接，实际数据没有重复保存。
[统一目录说明](formal_study/README.md) · [路径映射与恢复](docs/602_width_templates/LAYOUT_AND_RECOVERY.md)
