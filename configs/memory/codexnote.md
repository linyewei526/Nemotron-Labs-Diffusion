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
