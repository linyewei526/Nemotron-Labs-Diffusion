时间戳：2026-08-24 09:59:54 +0800（CST）

# NLD 可迁移实验结果索引

本目录把原本只存在于 `/data/home/wly/dLLM/NLD_results/` 的关键实验数据、证据和结论压缩为可随 Git 迁移的 Markdown。它不是原始结果备份：完整 `metrics_*.json`、逐请求 trace、生成文本、日志和 artifact 仍以原服务器目录为准。

## 文档索引

|层级|文档|内容|
|:--:|:--:|:--:|
|Basis|[`basis/NLD_experiment_provenance_and_metrics_zh.md`](basis/NLD_experiment_provenance_and_metrics_zh.md)|可信结果源、公共配置、指标和平均口径|
|Observation|[`observations/NLD_reproduction_and_diagnostic_evidence_zh.md`](observations/NLD_reproduction_and_diagnostic_evidence_zh.md)|SGLang/PyTorch 十项复现，confidence/rank/drop 和 draft alignment 证据|
|Method|[`method/NLD_three_overlap_methods_results_zh.md`](method/NLD_three_overlap_methods_results_zh.md)|三套 overlap 优化与 B16/B32 的结果、机制指标和复盘|

表中变量：层级表示 `basis/observations/method` 分类；文档是仓库内可迁移记录；内容是保留的证据范围。例如 `Method` 行指向三套方法的统一结果，而不是复制原始 JSON。

## 收录原则

- 主结果保留逐数据集 TPF、Accuracy 和必要机制指标；不搬运可由这些数字推导的重复字段。
- AIME24 因已知精度问题不进入主表和任何平均；原始结果仍保留在外部结果目录。
- “九集均值”表示 HumanEval、MBPP、LiveCodeBench-C++、GSM8K、MATH-500、AIME25、GPQA、MMLU、IFEval 各占 `1/9`，不按请求数或 token 数加权。
- 历史 pooled-token micro 只用于阈值分类诊断，并显式标为 micro；不能替代九数据集等权结果。
- TPS、时延和显存受 GPU、运行日期及显存预留影响，只保留能解释方法成本的参考值。
- 表格列全部使用居中 Markdown 对齐；短名和最小分隔符用于压缩宽度。每张表后给出全部变量的中文含义和例子。

## 未收录为科研主结果的产物

|产物|处理|原因|
|:--:|:--:|:--:|
|Chat smoke|只保留原手册索引|Arena-Hard、MT-Bench、AlpacaEval 主要验证评测框架和 artifact 收口，不是完整可比实验|
|失败/部分跑|不进数值主表|缺数据不能与十项完整运行做宏平均|
|原始 trace|不复制|体积大；本目录保留由 trace 校验得到的关键统计|
|请求日志|不复制|主要用于运行排障，不改变研究结论|

表中变量：产物是外部结果类型；处理说明在迁移文档中的保留方式；原因说明为何不作为主证据。例如 Chat smoke 仍可按 `configs/observations/NLD_Chat_Benchmarks_eval_framework_changes_zh.md` 复现，但这里不把少量 smoke 分数当正式能力结论。

## 使用顺序

新服务器或新会话应先读 Basis 口径，再读 Observation 证据，最后读 Method 对比；需要复现实验时转到 `configs/observations/` 或 `configs/method/` 的对应手册。外部绝对路径仅作为历史 provenance，不要求在迁移后的机器存在。
