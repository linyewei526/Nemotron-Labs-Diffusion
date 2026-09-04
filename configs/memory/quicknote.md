时间戳：2026-08-17 19:38:02 +0800（CST）

# Nemotron-Labs-Diffusion 当前会话进展速记

## 已完成工作

1. 项目与解码理解
   - 已阅读技术报告、项目代码、模型 remote code/权重侧解码实现，梳理 AR、block-wise dLLM、Linear Self-Speculation、LinearSpec + LoRA 及服务化调用链。
   - 入口、参数、attention/KV cache/LoRA/一位 shift 等映射见 `configs/basis/Nemotron_Labs_Diffusion_decode_code_guide_zh.md`；论文原文位于 `configs/basis/Nemotron_Diffusion_Tech_Report_v1.pdf`。
   - 环境、原生评测、SGLang 部署与开发背景见 `configs/basis/` 其余文档。

2. 复现与观察实验
   - 已梳理 SGLang + NeMo-Skills、原生 PyTorch + NeMo-Skills 的十项 benchmark 评测链路及结果口径。
   - 已整理 confidence/rank、draft/final alignment、`token_x_drop_abs`、`token_y_drop_pct`、低置信度与 rejection、PyTorch trace 离线阈值分析等实验。
   - 实验说明见 `configs/observations/`；统一入口见 `observations/`；历史与后续结果见 `/data/home/wly/dLLM/NLD_results/observations/`。
   - `offline_low_confidence_20260809_112909/report.md` 等历史结果已迁入上述外部 observation 结果根目录。

3. 两套独立解码优化
   - 已实现固定第二候选 B 的 Confidence-Overlap LinearSpec：`method/confidence_overlap_linearspec/`，说明见 `configs/method/NLD_PyTorch_NeMoSkills_confidence_overlap_linearspec_zh.md`。
   - 已实现自主全 MASK 重生成、支持完整/部分后缀复用的新版方案：`method/confidence_mask_redraft_linearspec/`，说明见 `configs/method/NLD_PyTorch_NeMoSkills_confidence_mask_redraft_linearspec_zh.md`。
   - 两套方法均为隔离实现，未覆盖旧 SGLang/PyTorch 入口；文档记录了单元测试、真实模型等价性和 NeMo-Skills 验收情况。

4. 目录整理
   - `configs/` 现按 `basis/`、`observations/`、`method/`、`memory/` 分类。
   - 7 个 observation Shell 入口已迁入 `observations/`，并拆分 `OBSERVATIONS_DIR`/`PROJECT_DIR`；默认结果统一到 `/data/home/wly/dLLM/NLD_results/observations/`。
   - 原 observation 历史结果约 15 GB 已分类迁移；15 个历史 `.eval_*_work_*` 已按要求删除。迁移说明见 `observations/README.md` 和 `/data/home/wly/dLLM/NLD_results/observations/README.md`。
   - 已通过入口语法/help/dry-run、跨工作目录调用、509 个 JSON 解析和 GSM8K 离线全链路 smoke。

## 生成本文时的运行快照

- Confidence-Overlap 十项 benchmark 仍在 GPU 1 运行：结果目录 `/data/home/wly/dLLM/NLD_results/confidence_overlap_linearspec_20260817_033248/`；已生成 8 项 metrics，正在执行 IFEval，`Settings.json` 仍为 `server_ready`。
- Mask-Redraft 十项 benchmark 仍在 GPU 3 运行：结果目录 `/data/home/wly/dLLM/NLD_results/confidence_mask_redraft_linearspec_20260817_182905/`；已生成 GSM8K、HumanEval、MBPP metrics，正在执行 MATH-500，`Settings.json` 仍为 `server_ready`。
- 对应隐藏工作目录仍被活跃进程使用，不属于已删除的 observation 历史工作目录；进程存活时不要移动或删除。
- 两份 method 文档中的早期验收结果路径是历史记录；本次盘点时这些早期目录已不在当前结果根目录，引用前应先检查实际存在性。

## 后续优先事项

- 先确认上述两轮任务是否结束，再检查最终 `Settings.json`、十项 metrics/error、运行时清理和汇总完整性。
- 在结果完整后比较 baseline、固定 B overlap 与自主 MASK redraft 的准确率、TPF、TPS、物理 NFE、复用率和额外计算量；当前 partial metrics 不能作为最终结论。
- 后续新增解码策略继续新建独立 `method/<新方法>/` 和 `configs/method/<新文档>.md`，不要覆盖已有复现、观察和两套方法代码。

---

时间戳：2026-08-24 10:21:44 +0800（CST）

## 增量交接记录

1. 正式实验与分析
   - Confidence-Overlap 与 Autonomous Mask-Redraft 的十项实验均已完成，各有 10 份 metrics、0 error；结果分别位于 `/data/home/wly/dLLM/NLD_results/confidence_overlap_linearspec_20260817_033248/` 和 `confidence_mask_redraft_linearspec_20260817_182905/`。
   - 两方法与 PyTorch B16/B32 baseline 的完整比较见 `/data/home/wly/dLLM/NLD_results/marks/NLD_PyTorch_LinearSpec_two_methods_vs_baselines_20260821_zh.md`。

2. 第三套独立方法
   - 已实现只接受触发位置直接命中的 Strict Direct MASK-Redraft：代码位于 `method/confidence_direct_mask_redraft_linearspec/`，手册为 `configs/method/NLD_PyTorch_NeMoSkills_confidence_direct_mask_redraft_linearspec_zh.md`。
   - 十项正式实验已完成，10 份 metrics、0 error；结果与统计见 `/data/home/wly/dLLM/NLD_results/confidence_direct_mask_redraft_linearspec_20260821_173928/report.md`。
   - 结果显示直接修正率与固定 B 接近，但 TPF 提升较小；主要诊断是自主 row 1 的后缀在同一次 forward 中没有条件化在预测修正 token 上。当前 verifier shift、row 0 KV cache 和 verifier-only 提交链路未发现明显错位。

3. 可迁移实验档案
   - 已将外部结果中的必要配置、TPF、Accuracy、confidence/rank/drop、alignment、三方法机制指标和结论提炼到 `configs/experiments/`；总索引为 `configs/experiments/README.md`。
   - 该目录可随 Git 迁移，但不是原始 metrics、trace、生成文本和日志的备份。原始结果仍默认位于 `/data/home/wly/dLLM/NLD_results/`。

4. 当前状态与待办
   - 本次盘点时三轮 method 正式实验均已结束，无对应活跃进程或隐藏工作目录。
   - Strict Direct 的 `report.md` 仍在整个 pipeline 结束后才创建；尚需改为运行初始化时创建完整框架、逐数据集原子刷新、全部数据集完成后才计算等权平均。
   - 后续应做同 commit/GPU/输入的 B16/B32/三方法配对重跑与逐 token verifier/KV 等价性检查。

## 迁移路径提示

- 当前文档和代码中的默认旧布局是：工作区根目录 `/data/home/wly/dLLM`，项目目录 `/data/home/wly/dLLM/Nemotron-Labs-Diffusion`，结果根目录 `/data/home/wly/dLLM/NLD_results`。
- 当前模型路径为 `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`，数据集目录为 `/data1/linyewei/datasets/NLD`，Conda 环境为 `nld_sglang`。
- 新开发者或新服务器若找不到任一路径/环境，应先向用户确认新的项目、结果、模型、数据集和环境映射，再指导修改运行默认值、命令或参数；不要静默创建空目录、替换模型/数据或伪造缺失结果。
- 历史 Settings、实验报告和 `configs/experiments/` 中的旧绝对路径属于 provenance，通常保留；真正需要改的是活跃代码默认值和当前执行命令。

---

时间戳：2026-08-31 10:37:20 +0800（CST）

## 增量交接记录

1. 新增 observation
   - 多 block 同状态影子实验：代码 `observations/pytorch_linearspec_block_size_shadow/`，手册 `configs/observations/NLD_PyTorch_LinearSpec_block_size_shadow_zh.md`，结果根目录 `/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/`。用于对照 L=4/8/16/32 的接收及历史特征；正式任务仍在运行。
   - 自适应首错位置免训练搜索：代码 `observations/adaptive_failure_locator_search/`，手册 `configs/observations/NLD_PyTorch_LinearSpec_adaptive_failure_locator_search_zh.md`。九个非 AIME24 数据集已完成，结果与策略统计见 `/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527/`。
   - 历史自适应 margin-risk 搜索：代码 `observations/adaptive_margin_history_search/`，手册 `configs/observations/NLD_PyTorch_LinearSpec_adaptive_margin_history_search_zh.md`。已在九数据集全部有效轮上按数据集等权完成全局搜索，结果和最优策略说明见 `/data/home/wly/dLLM/NLD_results/observations/adaptive_margin_history_search_results/adaptive_margin_history_20260830_153105/report.md`。

2. 新增 method
   - 固定 `margin_risk>0.5` 单候选 overlap：代码 `method/margin_risk_overlap_linearspec/`，手册 `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_overlap_linearspec_zh.md`，结果根目录 `/data/home/wly/dLLM/NLD_results/margin_risk_overlap_linearspec/`。
   - 固定 `margin_risk>0.5` 多候选与 continuation overlap：代码 `method/margin_risk_multi_overlap_linearspec/`，手册 `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_multi_overlap_linearspec_zh.md`，结果根目录 `/data/home/wly/dLLM/NLD_results/margin_risk_multi_overlap_results/`。
   - 两套入口现默认采用 efficiency-only 口径：报告不展示 accuracy，单请求 OOM 会被记录并排除出效率均值，同时以 Att/OK/Fail/OOM/Cov 披露覆盖率；`--require-accuracy` 恢复严格旧行为。具体边界和命令以各自手册为准。

3. 当前运行快照
   - Block-size shadow：`block_size_shadow_20260828_153452` 仍在 GPU 3 运行 MMLU；报告已有 6 个非 AIME24 数据集，IFEval 和 LiveCodeBench 尚待运行。
   - 单候选 margin-risk：`margin_risk_overlap_linearspec_20260831_004647` 仍在 GPU 1 运行，当前报告完成 6/9，正在 IFEval。
   - 多候选 margin-risk：`margin_risk_multi_overlap_20260831_004612` 仍在 GPU 3 运行，当前报告完成 5/9，正在 GPQA。
   - 上述活跃任务的隐藏工作目录不得移动或删除。旧目录 `margin_risk_overlap_linearspec_20260830_190823` 只有 3 份 metrics、5 份 error，不是正式完整结果。
   - 开始本次交接更新前，HEAD 为 `3359200accd080e3172b36c2aac0e2eedf370f5c`，工作树仅有用户的 `configs/NLD_prompt.md` 修改；本次另外只修改这两份 memory 文档。新会话仍须实时复核。

## 后续优先事项

- 等待三轮活跃实验自然完成，再检查最终 Settings、metrics/error、覆盖率、增量报告和隐藏工作目录清理情况。
- 解读效率时以各数据集等权结果为主；若 Cov 小于 100%，明确结果只覆盖成功请求，不能当作全请求均值。
- 不要从零重做上述搜索；优先读取对应 report、trace/summaries 和方法手册后继续分析或设计下一轮独立实验。

---

时间戳：2026-09-04 21:10:30 +0800（CST）

## 增量交接记录

1. SGLang/PyTorch 推理口径修复
   - 已完成 LinearSpec 差异排查，根因、对照 trace、修复与回归记录见 `configs/sglang_pytorch_diff/report.md`；SGLang CUDA Graph 已正确调用 pre-draft hook，启动 warmup trace 已过滤。
   - 当前 PyTorch 复现、相关 observation 和 method 的 TPF 均改为排除 prompt prefill 的 decode-only 口径，同时保留含 prefill 的审计字段；接口变化见两份 NeMo-Skills pipeline 手册。

2. 动态 block size observation
   - 需求与方案沉淀在 `configs/observations/NLD_PyTorch_LinearSpec_dynamic_block_size_history_signal_design_zh.md`。
   - SGLang 真实 trace、九数据集等权离线搜索和冻结验证代码位于 `observations/sglang_dynamic_block_history_signal/`，手册为 `configs/observations/NLD_SGLang_NeMoSkills_dynamic_block_size_history_signal_zh.md`，结果为 `/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/`。
   - 当前探索九集、离线搜索和 S8 九集验证已完成；S16 已完成 GSM8K、HumanEval、MBPP，正在 GPU 1 运行 MATH-500。失败重试和续跑状态以该目录 `report.md` 为准。
   - 旧 block-size shadow `block_size_shadow_20260828_153452` 最终只有 6 个非 AIME24 数据集完成，MMLU/IFEval/LiveCodeBench 失败，不能当作九集完整结果。

3. 新增 margin-risk method
   - 原单候选 `margin_risk_overlap_linearspec_20260831_004647` 和原多候选 `margin_risk_multi_overlap_20260831_004612` 已完成非 AIME24 九数据集；另有修复口径后的多候选结果 `margin_risk_multi_overlap_20260901_150214`。
   - P1/P2 + always-new 方法：代码 `method/margin_risk_two_plus_new_overlap_linearspec/`，手册 `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_two_plus_new_overlap_linearspec_zh.md`；0.5 与 0.45 阈值九集结果分别为 `margin_risk_two_plus_new_overlap_20260831_223138`、`margin_risk_two_plus_new_overlap_20260902_113656`。
   - 条件式 rank 方法：代码 `method/margin_risk_conditional_rank_overlap_linearspec/`，手册 `configs/method/NLD_PyTorch_NeMoSkills_margin_risk_conditional_rank_overlap_linearspec_zh.md`；单样本 smoke 为 `margin_risk_conditional_rank_overlap_20260903_152623`，正式结果 `margin_risk_conditional_rank_overlap_20260903_154307` 当前完成 7/9，正在 GPU 2 运行 LiveCodeBench，之后仍有 MMLU。

4. 实时版本与待办
   - 本次审计 HEAD 为 `9a6c93e0a7962397b88c0951995aebd4305e4eae`；提交后工作树干净，本次只追加两份 memory 文档。
   - 上述 S16 验证和条件式 rank 正式实验仍有活跃进程及隐藏工作目录，不得停止、移动或删除。
   - 待两项任务自然结束后，先核验最终 Settings、metrics/error、覆盖率和报告完整性，再分析冻结动态 block 策略及条件式 rank 方法结果。
