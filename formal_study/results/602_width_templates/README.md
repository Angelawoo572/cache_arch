# 602 H16 / capped H8：结果、CSV 和图

**24/24 新运行完成，9,748 项检查通过。**

- [中文结论与简洁比较表](../../../docs/602_width_templates/REPORT.md)
- [一页 paper story](../../../docs/602_width_templates/PAPER_STORY.md)
- [结果与全部图](FINDINGS.md) · [88 行逐项结果](per_run.csv) · [24 项新运行](per_run_new.csv)
- [配对比较](paired_comparisons.csv) · [审计](audit.json) · [指标定义](metric_definitions.json)

本地、Sacramento 和 GitHub 统一使用 `formal_study/results/602_width_templates/`。完整原始数据位于各自 cache 内的 `formal_study/runs/602_width_templates_20261017/`；大日志和 builds 按既定要求只保留本地及服务器。

H16 在本轮 MAC0/602/seed7 保留相对 native 优势；SPP H16 四组可被自己的简单模板精确复现，AMPM H16 无 cap 仅有很小残差、cap 下略输模板。H8 cap 下 Streamer/SPP 近乎复现，AMPM 模板更好。有限资源收益与跨 seed 泛化仍待验证。

## Original archive scope

This archive contains24 new complete runs (22 test +2 validation),64 explicitly labeled compatible references, all18 scientific PNG/SVG figure pairs, full-ROI/phase tables and original metadata. Every copied source is byte-equal; no raw source is edited.

[Findings](FINDINGS.md), [per-run data](per_run.csv), [new24](per_run_new.csv), [unaltered original64](reference_per_run_original.csv), [audit](audit.json), [method](method.svg), [copy manifest](archive_manifest.json).

Large snapshots, simulator logs, traces, checkpoints and portable model binaries remain in the complete cache/formal_study raw/archive trees. Their source paths are retained in metadata and the manifest. No replay/offline IPC is combined with live IPC.

All measured IPC assumes synchronous MAC0. H16 is an existing seed7 model-width comparison, not an additional training seed. The inherited ceiling does not equalize actual requests or DRAM. Old reference protocols remain distinct. Historical hardware telemetry is unavailable beyond the configured Sacramento environment and exact simulator-byte reuse.
