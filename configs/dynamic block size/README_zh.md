# NLD 动态 block size 研究交接入口

> 状态快照：2026-09-04 21:51（Asia/Shanghai）
> 目的：另一台服务器不迁移 `/data/home/wly/dLLM/NLD_results` 时，仍可仅依靠 Git 仓库理解动态 block size 研究的动机、方法、关键结果、当前进度和下一步实验设计。

## 1. 一页结论

本研究希望在连续 serving 中，为每个 request 的下一轮从 `L ∈ {8,16,32}` 中选择合适的 draft block size。目标不是尽量增大 block，而是在大 batch、容易 compute-bound 的场景下，只在大 block 能显著增加验证接收长度时才承担额外计算。

当前结论分为两层：

1. **缩短 block 基本不会破坏小 block 覆盖范围内的接收潜力。**第一阶段 PyTorch 同状态影子实验表明，L8/L16 相比更大 block 的主要损失来自容量之外的尾部，而不是 lookahead 缩短导致靠前 token 大量失配。因此，动态缩块在算法上是合理的。
2. **历史对下一轮接收能力有可预测性。**连续整块通过轮数、最近接收长度/比例、删失感知统计以及部分 confidence 特征都包含信号。严格 test 上，L16→L32 是否值得的 AUROC 达到 `0.8423`，是当前最强信号。
3. **已有 S8 冻结策略尚不适合直接部署。**九集验证中，升级后达到预设显著收益的比例为 `68.44%`，升级但只多接收不超过 1 token 的浪费率为 `29.55%`。它相对固定 L8 增加 `37.64%` block-token，却只增加 `35.76%` 接收量，compute-bound 效率下降约 `1.36%`。
4. **最有希望的是保守的 L16/L32 两档策略。**离线九集 held-out 上，动态 S16 相对固定 L16 增加 `10.12%` block-token、增加 `16.36%` 接收量，接收/block 提高 `5.67%`；但当前已完成的三集冻结验证只提高约 `0.28%`，且升级误用仍高。必须等待九集冻结验证完成。
5. **尚未找到可靠的 L16→L8 安全下调规则。**当前 `P(safe8) ≥ 0.98` 几乎从不触发，因此 S16 策略实际上是 16/32 两档，而不是完整三档。
6. **现有实验不能给出真实 serving 吞吐结论。**观察链路每轮执行 L8/L16/L32 三个 shadow，并关闭 CUDA Graph。报告中的接收/block 只是 compute-bound 代理。真正收益必须在移除 shadow、按 block size 分桶组 batch 的连续 serving 中验证。

## 2. 文档导航

- [01_实验问题与方法_zh.md](./01_实验问题与方法_zh.md)：完整解释两阶段实验、动态 canonical、同状态 shadow、删失感知历史、四个概率指标、全局搜索与动作规则。
- [02_核心结果与解读_zh.md](./02_核心结果与解读_zh.md)：固化不随原结果目录迁移的关键数值，并解释哪些结果支持动态 block、哪些结果仍不足以部署。
- [03_当前进度与后续方案_zh.md](./03_当前进度与后续方案_zh.md)：记录当前运行状态、代码与配置入口、迁移注意事项和下一步推荐实验。

## 3. 必须统一的术语

|术语|含义|
|:---:|:---|
|`L`|当前轮 draft block size，候选为 8、16、32。|
|`A_L`|同一轮、同一 causal prefix/KV/seed 下使用 block size `L` 得到的验证接收/发出 token 数。|
|canonical|当前策略预先选中并真正提交、用于更新 request 状态的分支。|
|shadow|同一状态下为获得反事实标签而额外运行、但不提交的其他 block 分支。|
|整块通过|当前 block 的 draft 全部通过验证；小 block 整块通过只说明真实能力至少达到该 block 上限。|
|删失|选择 L8 并整块通过时，只观察到“能力 ≥8”，不知道本轮若用 L16/L32 会接收多少。|
|大精|实际升级的轮次中，大 block 达到预设显著收益标签的比例；按数据集条件宏平均。|
|大浪|实际升级的轮次中，相比默认 block 只多接收不超过 1 token 的比例；越低越好。|
|块均|策略平均选择的 block size，是部署 block-token 计算量的一阶代理。|
|接均|策略平均每轮接收/发出的 token，是算法 TPF 潜力的方向指标。|
|TPF代|报告中的 `接均/块均`，用于 compute-bound 搜索；不是实际 serving TPF。|

## 4. 当前推荐判断

如果下一步需要尽快收敛一条可部署路线，优先级应为：

1. 完成 S16 九集冻结验证；
2. 暂时只研究 L16/L32 两档，使用高精度、可拒绝升级的 gate；
3. 用冻结运行产生的 on-policy 动态历史重新校准，解决探索分布与冻结策略分布不一致；
4. 策略通过后，在 SGLang 中移除 shadow，按 L8/L16/L32 分桶组 batch，测真实 TPF、吞吐、TPOT、延迟和 GPU 利用率；
5. S8 三档策略和 L16→L8 安全下调作为后续问题，不应阻塞 16/32 路线。

## 5. 原始资料入口

仓库内原始设计和运行手册仍是实现细节的权威来源：

- `configs/observations/NLD_PyTorch_LinearSpec_block_size_shadow_zh.md`
- `configs/observations/NLD_PyTorch_LinearSpec_dynamic_block_size_history_signal_design_zh.md`
- `configs/observations/NLD_SGLang_NeMoSkills_dynamic_block_size_history_signal_zh.md`
- `observations/pytorch_linearspec_block_size_shadow/`
- `observations/sglang_dynamic_block_history_signal/`

本文档组负责跨服务器的研究状态交接；若它与实时运行产物冲突，应以最新完整 trace 的重新统计和上述源码为准，并更新本目录中的状态快照。
