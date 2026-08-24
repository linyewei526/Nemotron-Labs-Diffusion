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
