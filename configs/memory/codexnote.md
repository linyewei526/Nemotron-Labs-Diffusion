时间戳：2026-08-17 19:38:02 +0800（CST）

# 新 Codex 会话对齐指南

本文用于让没有历史上下文的新 Codex 会话恢复对 Nemotron-Labs-Diffusion 的理解和当前进度。不要只总结本文；应按下面顺序读取实际文档、代码、运行状态和结果，然后再继续工作。

## 1. 固定路径与环境

- 项目：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion`
- 环境：`conda activate nld_sglang`
- 模型与 remote code：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`
- observation 结果：`/data/home/wly/dLLM/NLD_results/observations`
- method 结果根目录：`/data/home/wly/dLLM/NLD_results`

模型权重目录应默认按只读对象处理。不要因为文档中的历史路径或参数与当前磁盘不一致，就擅自移动、覆盖或删除文件。

## 2. 开始工作前先做状态审计

1. 阅读 `configs/memory/quicknote.md`，了解最近完成事项和生成时的运行快照。
2. 检查 `git status --short`，保留用户及此前会话的未提交改动。
3. 检查 `ps`、`nvidia-smi`、`/data/home/wly/dLLM/NLD_results/*/Settings.json` 和隐藏工作目录，确认是否仍有 benchmark 在运行。
4. `Settings.json` 为 `server_ready`、存在部分 metrics 或存在隐藏工作目录都不代表任务已经完成；应结合父进程、子进程、当前 benchmark 日志和最终状态判断。
5. 活跃 method 任务使用的隐藏工作目录不得删除。observation 迁移时删除的 15 个 `.eval_*_work_*` 是已经明确要求不保留的历史目录，两者不要混淆。

本文生成时有两轮十项 benchmark 在运行：Confidence-Overlap 位于 GPU 1、结果前缀为 `confidence_overlap_linearspec_20260817_033248`；Mask-Redraft 位于 GPU 3、结果前缀为 `confidence_mask_redraft_linearspec_20260817_182905`。新会话必须重新核验，不要假设该快照仍然有效。

## 3. 按顺序恢复项目理解

### 第一步：论文与原生解码

完整阅读：

1. `configs/basis/Nemotron_Diffusion_Tech_Report_v1.pdf`
2. `configs/basis/Nemotron_Labs_Diffusion_decode_code_guide_zh.md`

随后对照项目和模型 remote code，确认自己能说明 AR、block-wise dLLM、Linear Self-Speculation、LinearSpec + LoRA 的 draft/verify 流程，以及 `chat/`、`evaluate.py`、`eval.sh`、GPU-only/NeMo-Skills 服务链之间的调用关系。

### 第二步：运行环境、SGLang 与评测边界

按任务需要阅读：

- `configs/basis/NLD_eval_sh_vs_evaluate_explained_zh.md`
- `configs/basis/NLD_A100_environment_and_runbook_zh.md`
- `configs/basis/NLD_SGLang_zero_to_dev_benchmark_zh.md`
- `configs/basis/NLD_SGLang_serving_config_benchmark_optimization_zh.md`

这里用于对齐原生 Hugging Face、SLURM/NeMo-Skills、SGLang serving 三条路径，不要把 accuracy benchmark、serving benchmark 和离线统计混为一谈。

### 第三步：复现与观察实验

先读两条主评测链：

1. `configs/observations/NLD_SGLang_NeMoSkills_eval_pipeline_zh.md`
2. `configs/observations/NLD_PyTorch_NeMoSkills_eval_pipeline_zh.md`

再读诊断实验：

1. `configs/observations/NLD_SGLang_LinearSpec_confidence_trace_zh.md`
2. `configs/observations/NLD_PyTorch_NeMoSkills_LinearSpec_confidence_trace_zh.md`
3. `configs/observations/NLD_SGLang_LinearSpec_draft_alignment_zh.md`
4. `configs/observations/NLD_SGLang_LinearSpec_low_confidence_rejection_zh.md`
5. `configs/observations/NLD_PyTorch_LinearSpec_low_confidence_offline_zh.md`
6. 如涉及 Arena-Hard、MT-Bench 或 AlpacaEval，再读 `configs/observations/NLD_Chat_Benchmarks_eval_framework_changes_zh.md`。

入口索引为 `observations/README.md`。实际 Shell 入口位于 `observations/`，共享底层实现位于 `xp/`；默认结果已经迁出项目目录。历史结果结构和已知旧 MT-Bench artifact 例外见 `/data/home/wly/dLLM/NLD_results/observations/README.md`。

### 第四步：两套解码优化

按演进顺序阅读：

1. `configs/method/NLD_PyTorch_NeMoSkills_confidence_overlap_linearspec_zh.md`
2. `method/confidence_overlap_linearspec/`
3. `configs/method/NLD_PyTorch_NeMoSkills_confidence_mask_redraft_linearspec_zh.md`
4. `method/confidence_mask_redraft_linearspec/`

阅读 method 代码时至少跟踪入口 Shell、`run_pipeline.sh`、`server.py`、`generation.py`、`hybrid.py`、`segmented_lora.py`、指标合并和 `tests/`。前者是固定第二候选 B 的旧实验，后者是自主全 MASK 重生成并支持完整/部分后缀复用的当前新方案；不要把两者合并或覆盖。

## 4. 代码目录职责

| 路径 | 职责 |
|---|---|
| `chat/`、`evaluate.py` | 原生直接推理与小规模评测 |
| `eval.sh` | 官方/legacy SLURM 入口，仍保留在项目根目录 |
| `observations/` | 本地 SGLang/PyTorch + NeMo-Skills 复现和诊断入口 |
| `xp/` | server、NeMo-Skills adapter、pipeline、trace/summary 等共享实现 |
| `method/` | 隔离的新解码优化实验，不应反向污染旧入口 |
| `sglang_dllm/` | 本地 SGLang fork、缓存和 LoRA 工作区 |
| `configs/basis/` | 论文、解码、环境和 SGLang 基础理解 |
| `configs/observations/` | 复现/观察实验手册 |
| `configs/method/` | 新方法设计、命令、指标和验收记录 |
| `configs/memory/` | 会话进度和新会话对齐入口 |

`configs/NLD_prompt.md` 是历史请求记录，可能含旧入口和旧结果路径，不是当前执行规范。

## 5. 继续修改解码前的最低对齐要求

新会话应先确认并能解释以下边界，具体定义以解码导读和 method 文档为准：

- 输出只由 causal AR verifier 提交，prospective/redraft row 只能提供未提交草稿。
- LinearSpec logits/token 存在一位 shift；接受、修正和 bonus token 的索引必须统一。
- MASK、EOS、thinking budget、生成长度和位置上限必须在候选与复用逻辑中显式处理。
- canonical KV cache 只能来自可信 verifier 路径；row 1 的 proposal 不能污染已提交状态。
- `diffusion_lm`、全局 LoRA 与分段 LoRA 路由必须按 attention 区域恢复，不能泄漏到其他请求或实验。
- TPF/NFE 必须按物理 model forward 记账，同时报告 TPS、processed rows/query tokens 和复用统计，不能只看“省掉几次 draft”。

如果无法清楚说明这些约束，应继续读代码和测试，不要直接修改生成状态机。

## 6. 结果与文档的可信顺序

判断当前状态时按以下优先级：

1. 活跃进程、当前实际代码、当前 `Settings.json`、metrics/error 和日志；
2. `configs/memory/quicknote.md` 的时间戳快照；
3. `configs/method/`、`configs/observations/` 中的验收记录；
4. 历史 prompt 或聊天描述。

method 文档记录的早期验收目录在本文生成时已不位于当前结果根目录；引用具体数字前先验证目录存在。partial metrics 和 `server_ready` 状态不得作为最终十项 benchmark 结论。

## 7. 后续实验规范

- 复现和 observation 实验使用 `observations/` 的现有入口与外部 observation 结果根目录。
- 新解码变体新建 `method/<独立名称>/`、对应 `configs/method/*.md` 和独立时间戳结果；不要改写旧实验来冒充新变体。
- 结果目录创建后立即写 Settings；自动选空闲端口，显式记录 GPU、显存预留、模型、LoRA、benchmark 和超参数。
- 验证顺序应为：静态语法/单元测试 → dry-run → 小模型逻辑或真实模型等价性 → 单样本 NeMo-Skills smoke → 指定完整 benchmark → 多 benchmark。
- HumanEval/MBPP 的 EvalPlus 需要完整题集，不能用 `--max-samples` 伪装完整评分。
- 不要停止、删除或覆盖与当前任务无关的活跃进程、用户改动和历史结果。

## 8. 对齐完成的判据

新 Codex 在开始新工作前，应向用户简要报告：

1. 四类核心解码方式及主要调用链已经理解；
2. observation 与 method 的代码/结果边界已经理解；
3. 两套优化的先后关系和当前方案已经理解；
4. 当前进程、结果完整性和未完成任务已经重新核验；
5. 下一步准备修改的具体独立路径、验证计划和不会影响的旧实验范围。

完成这些对齐后，从 `quicknote.md` 的“后续优先事项”或用户的新指令继续，不要从头重做已经完成的研究。

---

时间戳：2026-08-24 10:21:44 +0800（CST）

# 新会话增量对齐与跨服务器迁移指南

本节追加于 2026-08-17 快照之后，优先级高于前文关于“两套方法仍在运行”和“固定路径”的旧状态描述；不要删除或改写旧快照。

## 9. 先读取可随 Git 迁移的实验档案

在检查外部结果前依次阅读：

1. `configs/experiments/README.md`
2. `configs/experiments/basis/NLD_experiment_provenance_and_metrics_zh.md`
3. `configs/experiments/observations/NLD_reproduction_and_diagnostic_evidence_zh.md`
4. `configs/experiments/method/NLD_three_overlap_methods_results_zh.md`

这些文档保存了必要 TPF、Accuracy、confidence/rank/drop、alignment、方法状态和分析结论。即使 `/data/home/wly/dLLM/NLD_results` 没有迁移，也应先靠它们恢复研究进度；若任务需要逐请求 trace、完整 metrics、生成文本或 artifact，再询问用户原始结果的新位置，不能从摘要反向伪造原始数据。

## 10. 三套方法的最新演进顺序

按以下顺序阅读手册和同名 `method/` 目录：

1. `confidence_overlap_linearspec`：固定第二候选 B，只有 verifier 确认 `C=B` 时复用完整 L。
2. `confidence_mask_redraft_linearspec`：自主全 MASK 重生成，允许直接、下游、bonus 和部分后缀复用。
3. `confidence_direct_mask_redraft_linearspec`：自主全 MASK 重生成，但只在 `m=p,R0=C` 时复用完整 L。

三轮正式十项结果均已完成。第三套方法暴露的关键问题是：`R0=C` 只说明自主 row 1 猜中修正 token，同一次 forward 产生的 `R1...` 并未看到 C 的 token embedding，因此不能认为后缀已条件化在 C 上。具体数据和复盘以 `configs/experiments/method/` 为准。

Strict Direct 还有一个未完成工程项：当前报告只在整个 pipeline 结束后生成。修改时应在 Settings 创建后立即建立完整 `report.md` 框架，每完成/失败一个数据集就原子重建当前报告，全部请求数据集结束前不计算平均，结束后才计算排除 AIME24 的数据集等权平均。

## 11. 旧服务器路径不是不可变常量

当前仓库的文档、命令和部分代码默认基于以下旧服务器映射：

|名称|旧默认值|
|---|---|
|工作区根目录|`/data/home/wly/dLLM`|
|项目目录|`/data/home/wly/dLLM/Nemotron-Labs-Diffusion`|
|结果根目录|`/data/home/wly/dLLM/NLD_results`|
|模型目录|`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`|
|数据集目录|`/data1/linyewei/datasets/NLD`|
|Conda 环境|`nld_sglang`|

名称表示需要确认的迁移参数；旧默认值只是当前服务器布局。例如项目 clone 到 `/workspace/Nemotron-Labs-Diffusion` 时，项目相对路径仍可用，但任何以 `/data/home/wly/dLLM` 开头的运行默认值都必须审计。

在新服务器开始运行前逐项检查实际项目目录、结果目录、模型、数据集和环境。任一项不存在时：

1. 主动告诉用户缺失的旧路径以及它被哪些任务需要。
2. 询问用户对应的新绝对路径或可用 Conda 环境，不猜测、不静默下载、不创建空模型/数据目录。
3. 指导用户通过现有 CLI、环境变量或配置覆盖；确需改代码时，只改当前活跃默认值并保持旧实验隔离。
4. 修改后先做路径存在性、依赖 import、dry-run 和单样本 smoke，再执行正式评测。

## 12. 绝对路径审计与修改边界

迁移后先在实际项目根目录搜索旧前缀和环境名，至少覆盖 `configs/`、`observations/`、`method/`、`xp/`、`chat/`、根目录脚本及模型 remote code 的调用处：

```bash
rg -n '/data/home/wly/dLLM|/data1/linyewei/models/Nemotron-Labs-Diffusion-8B|/data1/linyewei/datasets/NLD|nld_sglang' configs observations method xp chat ./*.sh ./*.py
```

搜索结果分两类处理：

- 活跃入口、默认参数、当前命令、环境激活和运行依赖：更新为用户确认的新映射，优先使用项目相对路径、CLI 参数或明确的任务专用环境变量。
- 历史 `Settings.json`、旧报告、`configs/experiments/` 的来源路径和历史命令：保留为 provenance；必要时追加“迁移后映射”，不要把历史记录改写成从未发生过的新路径。

若 `/data/home/wly/dLLM/NLD_results` 不存在，不应把 `configs/experiments/` 当作原始结果目录；它只能支持结果理解。需要重生成报告、检查逐样本输出或继续离线 trace 分析时，必须先向用户询问原始结果迁移位置或是否允许重新运行。

若模型或数据集缺失，应分别确认新 `MODEL_PATH` 和 NeMo-Skills data root，并检查 Google Research/EvalPlus 等附加资产；只找到同名目录不足以证明内容版本正确。若 `nld_sglang` 不存在，应先询问新环境名或安装方案，核对 Python、PyTorch、Transformers、SGLang、NeMo-Skills 和 CUDA 兼容性后再运行。

## 13. 最新对齐完成判据

新会话应向用户简要确认：

1. 已读取 `configs/experiments/` 并理解三套方法及正式结果；
2. 已区分仓库内可迁移摘要与外部原始结果；
3. 已核验当前是否仍有活跃进程、隐藏工作目录和未提交改动；
4. 已确认项目、结果、模型、数据集和 Conda 环境的实际映射；
5. 已说明 Strict Direct 的后缀条件化瓶颈、报告增量更新待办和下一步独立修改路径。

达到这些判据后再继续开发，不要因为旧绝对路径失效就重做已完成研究，也不要在未经确认时批量替换历史 provenance。

---

时间戳：2026-08-31 10:37:20 +0800（CST）

# 2026-08-31 新增实验对齐指南

本节优先于前文关于“三套 method 均结束”和旧运行快照的描述。新会话先读取 `configs/memory/quicknote.md` 最后一条记录，再按下面入口恢复新增进展；不要删除旧快照，也不要从零重做已有搜索。

## 14. 新增 observation 阅读顺序

1. Block-size shadow：先读 `configs/observations/NLD_PyTorch_LinearSpec_block_size_shadow_zh.md`，再按需检查 `observations/pytorch_linearspec_block_size_shadow/` 和 `/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/`。
2. 自适应首错位置搜索：读 `configs/observations/NLD_PyTorch_LinearSpec_adaptive_failure_locator_search_zh.md`，代码位于 `observations/adaptive_failure_locator_search/`，完整九数据集结果位于 `/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527/`。
3. 历史自适应 margin-risk：读 `configs/observations/NLD_PyTorch_LinearSpec_adaptive_margin_history_search_zh.md`，代码位于 `observations/adaptive_margin_history_search/`，九数据集全数据等权搜索结果位于 `/data/home/wly/dLLM/NLD_results/observations/adaptive_margin_history_search_results/adaptive_margin_history_20260830_153105/`。

这三项分别回答 block 长度与同轮接收、免训练首错定位、历史信息能否自适应 margin-risk。需要具体指标或结论时直接读取对应 `report.md`、`summaries/`、`analysis/` 和 trace，不要根据本文复述推断数值。

## 15. 新增 method 阅读顺序

在前文三套 confidence/drop 方法之后，继续按以下顺序阅读：

1. `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_overlap_linearspec_zh.md`
2. `method/margin_risk_overlap_linearspec/`
3. `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_multi_overlap_linearspec_zh.md`
4. `method/margin_risk_multi_overlap_linearspec/`

单候选版以固定 margin-risk 取代原 drop 阈值；多候选版扩展为最多三个候选并加入整块通过时的 continuation。阅读代码仍须跟踪入口 Shell、pipeline、server、generation、hybrid attention、分段 LoRA、metrics/report 和 tests，且不得反向修改旧 confidence 方法。

两套新入口默认是 efficiency-only，而不是旧的严格 accuracy pipeline：单请求 OOM 记录后跳过，效率只聚合成功请求，报告必须结合 Att/OK/Fail/OOM/Cov；仅在明确传入 `--require-accuracy` 时恢复严格 scorer/OOM 失败行为。省略 `--efficiency-only` 仍是默认效率模式。该口径不会主动缩小数据集；只有 `--max-samples` 或 `--quick-test` 才改变请求范围。

## 16. 新结果与实时状态核验

本文生成时：

- Block-size shadow 的 `block_size_shadow_20260828_153452` 正在 GPU 3 跑 MMLU，已有 6 个非 AIME24 数据集结果；
- 单候选 `margin_risk_overlap_linearspec_20260831_004647` 正在 GPU 1 跑 IFEval，报告为 6/9；
- 多候选 `margin_risk_multi_overlap_20260831_004612` 正在 GPU 3 跑 GPQA，报告为 5/9；
- `margin_risk_overlap_linearspec_20260830_190823` 是不完整旧尝试，不能作为正式九数据集结论；
- 两个自适应定位 observation 已完成，优先复用其现有结果。

以上只是时间戳快照。新会话必须重新检查进程树、GPU、隐藏工作目录、Settings、metrics/error、当前日志和 report 完成数。活跃任务不得停止或清理；`server_ready` 也不能被当作最终完成状态。

开始本次交接更新前 Git HEAD 为 `3359200accd080e3172b36c2aac0e2eedf370f5c`，工作树仅显示用户对 `configs/NLD_prompt.md` 的修改；本次另外只修改 `quicknote.md` 和 `codexnote.md`。后续若状态不同，以实时 Git 和文件内容为准并保留用户改动。

## 17. 新增进度的对齐完成判据

新会话继续研究前应能简要确认：

1. 已区分三项新增 observation 的问题、代码和结果入口；
2. 已理解固定 margin-risk 单候选与多候选方法相对旧 confidence/drop 方法的演进关系；
3. 已理解 efficiency-only 默认值、全数据集范围与 OOM/Cov 统计边界；
4. 已实时核验三轮正式任务是否完成，并排除不完整旧结果；
5. 后续修改会继续使用独立 observation/method 目录，不覆盖既有复现、搜索或正在运行的实验。
