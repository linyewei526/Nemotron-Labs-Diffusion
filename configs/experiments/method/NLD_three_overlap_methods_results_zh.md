# NLD 三套 Confidence-Overlap 方法结果与复盘

本文压缩保存三套独立 PyTorch+NeMo-Skills 方法的正式十项结果。公共配置、结果路径和指标定义见 `../basis/NLD_experiment_provenance_and_metrics_zh.md`；实现与命令见 `configs/method/`。

## 1. 方法差异

设当前 draft 为 `D`，confidence first-crossing 位置为 `p`，原 token 为 `A=D[p]`，verifier 修正为 `C`，row 1 输出为 `R`，block length `L=16`。

|简称|row 1|复用规则|长度|
|:--:|:--:|:--:|:--:|
|O|`D[:p]+B+MASK×(L-1)`|在 p 拒绝 A 且 `C=B`|完整 L|
|M|`D[:p]+MASK×L`|可信段匹配，含直接/下游/bonus|完整或部分|
|D|`D[:p]+MASK×L`|仅 `m=p` 且 `R0=C`|完整 L|

表中变量：O 是固定第二候选 B 的 Confidence-Overlap；M 是 Autonomous Mask-Redraft；D 是 Strict Direct MASK-Redraft；`m=matched+1` 是当前 verifier 发出的 token 数。例如 D 在 `p=8,m=8,R0=C` 时保留完整 R，其余状态全部丢弃。

三套方法均为 L16、LinearSpec LoRA、`token_y_drop_pct>0.15`、greedy、verifier-only 输出；没有修改或覆盖 baseline 与其他方法目录。

## 2. TPF 主结果

|数据集|B16|B32|O|M|D|
|:--:|:--:|:--:|:--:|:--:|:--:|
|HumanEval|5.0760|6.1590|5.6025|5.5071|5.1447|
|MBPP|3.9731|4.5612|4.4000|4.3503|4.1684|
|LCB-C++|3.7763|4.2704|4.1400|4.0919|4.0263|
|GSM8K|5.0009|6.3515|5.3007|5.2589|5.1891|
|MATH-500|5.4520|7.2771|5.9924|5.9031|5.8154|
|AIME25|5.4505|7.0329|5.8519|5.3485|5.1312|
|GPQA|5.3499|7.2937|5.5437|5.6051|5.5691|
|MMLU|3.8032|4.3721|4.1661|4.0532|3.9455|
|IFEval|4.8354|6.3371|5.1347|4.9536|4.9510|
|九集均值|4.7464|5.9617|5.1258|5.0080|4.8823|

表中变量：B16/B32 是原生 PyTorch LinearSpec baseline；O/M/D 是上一节三套方法；单元为 TPF。例如 HumanEval 上 O 的 `5.6025` 比 B16 的 `5.0760` 高，但仍低于 B32 的 `6.1590`。九集均值排除 AIME24且各数据集等权。

### 2.1 宏平均增益

|方法|Δ16|相对16|Δ32|相对32|
|:--:|:--:|:--:|:--:|:--:|
|O|+0.3794|+7.99%|-0.8359|-14.02%|
|M|+0.2616|+5.51%|-0.9537|-16.00%|
|D|+0.1359|+2.86%|-1.0794|-18.11%|

表中变量：Δ16/Δ32 是方法 TPF 减 B16/B32；相对16/32 再除以对应 baseline。例如 O 的 `+0.3794` 等于 `5.1258-4.7464`。三套 L16 方法都优于 B16，但都没有超过 B32。

## 3. Accuracy 主结果

|数据集|B16|B32|O|M|D|
|:--:|:--:|:--:|:--:|:--:|:--:|
|HumanEval|76.8293%|75.6098%|77.4390%|78.6585%|76.2195%|
|MBPP|68.2540%|67.4603%|66.6667%|67.1958%|66.1376%|
|LCB-C++|29.0749%|29.2952%|30.6167%|29.2952%|29.5154%|
|GSM8K|93.7074%|93.1766%|93.5557%|93.4799%|92.9492%|
|MATH-500|88.4000%|87.8000%|87.2000%|86.6000%|88.4000%|
|AIME25|36.6667%|33.3333%|26.6667%|26.6667%|36.6667%|
|GPQA|37.0558%|38.0711%|40.6091%|39.0863%|34.5178%|
|MMLU|78.9489%|79.0058%|79.1625%|79.0414%|79.1411%|
|IFEval|68.8558%|69.6177%|68.4936%|68.4474%|69.5715%|
|九集均值|64.1992%|63.7078%|63.3789%|63.1635%|63.6799%|

表中变量：B16/B32/O/M/D 与 TPF 表相同；主 Accuracy 字段按 Basis 文档定义。例如 D 的 MATH-500 与 B16 都是 `88.4%`。宏平均仅用于导航；AIME25 30 题造成的离散变化会被赋予与 MMLU 相同权重。

### 3.1 宏平均差值

|方法|ΔAcc16|ΔAcc32|
|:--:|:--:|:--:|
|O|-0.8203 pp|-0.3289 pp|
|M|-1.0357 pp|-0.5443 pp|
|D|-0.5193 pp|-0.0279 pp|

表中变量：ΔAcc16/ΔAcc32 是方法九集 Accuracy 减 B16/B32，单位为百分点 pp。例如 D 相对 B32 的 `-0.0279 pp` 不是相对下降 0.0279%，而是两个百分数直接相减。

不同方法的 completion token 数和逐题输出不完全一致，因此现有结果不能证明 greedy token-level 严格等价，也不能把小幅 Accuracy 波动全部归因于方法质量。

## 4. 命中与实际节省

|数据集|OHit|OSave|MReuse|MSave|DHit|DSave|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HumanEval|22.82%|13.18%|48.60%|29.68%|23.82%|15.45%|
|MBPP|21.41%|17.59%|45.89%|37.84%|23.57%|19.53%|
|LCB-C++|21.77%|18.26%|43.90%|36.99%|24.53%|20.49%|
|GSM8K|23.43%|16.52%|48.32%|33.79%|23.09%|15.79%|
|MATH-500|23.50%|13.47%|49.96%|29.56%|22.63%|13.08%|
|AIME25|22.81%|12.96%|49.41%|34.24%|23.78%|16.78%|
|GPQA|21.60%|12.32%|44.75%|25.30%|23.81%|12.88%|
|MMLU|20.62%|15.87%|40.75%|32.12%|23.74%|18.74%|
|IFEval|16.21%|9.21%|41.82%|25.33%|23.39%|13.90%|
|九集均值|21.58%|14.38%|45.93%|31.65%|23.59%|16.29%|

表中变量：OHit 是固定 B 被 verifier 确认的次数/attempts；MReuse 是 M 最终可复用次数/attempts；DHit 是 `m=p,R0=C` 次数/attempts；OSave/MSave/DSave 都是实际 Saved/Rounds。例如 HumanEval 的 MReuse=`48.60%`，但只有 `29.68%` 的全部 rounds 真正省掉下一轮 draft。

## 5. 为什么更多 Saved 没有形成更高 TPF

|方法|TPF|Hit|Save|Tok/R|TPS|Rows/NFE|QTok/NFE|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|O|5.1258|21.58%|14.38%|9.601|169.74|1.357|26.40|
|M|5.0080|45.93%|31.65%|8.513|138.46|1.409|27.44|
|D|4.8823|23.59%|16.29%|9.052|149.66|1.373|26.75|

表中变量：TPF/Save/Tok/R/TPS/Rows/NFE/QTok/NFE 按 Basis 定义；Hit 对 O/D 是直接触发命中，对 M 是实际复用率。例如 M 的 Save=`31.65%` 远高于 O 的 `14.38%`，但 Tok/R 从 `9.601` 降到 `8.513`，导致更多 verify rounds，最终 TPF 反而更低。TPS 是跨 GPU/日期参考，不是严格硬件对照。

普通路径近似满足：

```text
physical_nfe = requests + 2×rounds - saved
tpf = completion_tokens / physical_nfe
```

一次 Saved 只减少 1 个 draft forward；如果复用后缀使接受量下降并增加一个普通 round，可能新增接近 2 个 forward，足以抵消节省。

## 6. Strict Direct 的六状态证据

|状态|占尝试|复用|本M|下M|ΔM|
|:--:|:--:|:--:|:--:|:--:|:--:|
|m<p|19.41%|否|1.343|5.646|+4.302|
|直中|23.59%|是|3.892|4.067|+0.174|
|重A|5.85%|否|3.586|5.989|+2.402|
|改错|2.26%|否|3.415|4.843|+1.428|
|A对后拒|40.18%|否|6.369|6.150|-0.219|
|Bonus|8.70%|否|15.000|10.153|-4.847|

表中变量：占尝试是状态次数/row 1 attempts；复用表示是否允许 R 成为下一轮 draft；本M/下M 是当前/下一轮平均 matched token；ΔM 对有下一轮事件逐条计算 `下M-本M` 后平均。例如“重A”表示 A 被 verifier 拒绝但 R0 仍是 A，row 1 被丢弃，下一轮重新 normal draft。

六状态严格划分 attempts，只有“直中”可复用。`A对后拒+Bonus=48.88%` 表明约一半尝试中 confidence first-crossing 位置 A 实际正确，因此该信号不能视为精确错误位置。

### 6.1 直中复用后的下一轮质量

|数据集|直中下M|重A下M|
|:--:|:--:|:--:|
|HumanEval|4.629|7.062|
|MBPP|4.287|7.195|
|LCB-C++|3.370|4.939|
|GSM8K|5.434|7.623|
|MATH-500|5.462|7.695|
|AIME25|4.734|6.367|
|GPQA|3.728|5.414|
|MMLU|3.035|4.647|
|IFEval|1.923|2.955|
|九集均值|4.067|5.989|

表中变量：直中下M是 D 复用完整 R 后，下一轮 verifier 平均接受的 draft token 数；重A下M是 R0=A 后丢弃 R、下一轮 normal draft 的平均 matched。例如 HumanEval 直中复用后只通过 `4.629`，而重A分支回到 normal draft 后为 `7.062`。两组事件难度不同，因此这是强相关证据而非严格反事实实验。

## 7. 核心算法诊断

O 的 row 1 在 fused forward 前已显式放入候选 B：

```text
[D[:p],B,MASK×15]
```

当 verifier 证明 `C=B` 时，后续 15 个 token 确实条件化在 B 的 token embedding 上。D 的输入却是：

```text
[D[:p],MASK0,MASK1,...,MASK15]
```

即使输出 `R0=C`，`R1...R15` 也是在 MASK0 而非 C 的输入表示下同一次并行产生；把输出 token 解释为 C 不会追溯性改变后缀 hidden state。因此“D 的直接修正率与 O 接近”不代表命中后的后缀质量接近。这是 D 的 Hit/Save 略高于 O、Tok/R 和 TPF却明显更低的主要解释。

M 还允许下游修正、bonus 和部分后缀复用，Save 大幅增加；但 observation 中首次拒绝后 O1/O4 alignment 只有约 `31.39%/19.69%`，复用靠后后缀会降低下一轮接受量，所以 M 也没有超过 O。

## 8. 当前结论与后续验证

1. 三套方法均证明“verify 时顺便起草下一轮”可以提升 L16 的物理 TPF，但 O 的九集结果最好。
2. 命中率或 Saved 不能单独作为优化目标；至少同时报告 Tok/R、Rows/NFE、QTok/NFE 和 TPS。
3. 若要自主决定 C 后再生成真正条件化后缀，标准 Transformer 通常需要第二次 `[C,MASK×15]` refinement，或并行 top-k 显式 anchor row，或新的两阶段算子。
4. 需要在同 commit、同 GPU、同 prompt/token IDs 下配对重跑 B16/B32/O/M/D，验证 fused row 0 logits、KV cache 和最终 greedy token 的逐步等价性。
5. D 的现有 `report.md` 只在全部 benchmark 完成后生成；尚未实现“初始化空框架、每完成一个数据集原子刷新、全部完成后才计算平均值”。这是下一次修改应处理的工程缺口。

## 9. 代码与原报告

- O：`method/confidence_overlap_linearspec/`；手册 `configs/method/NLD_PyTorch_NeMoSkills_confidence_overlap_linearspec_zh.md`。
- M：`method/confidence_mask_redraft_linearspec/`；手册 `configs/method/NLD_PyTorch_NeMoSkills_confidence_mask_redraft_linearspec_zh.md`。
- D：`method/confidence_direct_mask_redraft_linearspec/`；手册 `configs/method/NLD_PyTorch_NeMoSkills_confidence_direct_mask_redraft_linearspec_zh.md`。
- O/M 原始综合报告：`/data/home/wly/dLLM/NLD_results/marks/NLD_PyTorch_LinearSpec_two_methods_vs_baselines_20260821_zh.md`。
- D 原始报告：`/data/home/wly/dLLM/NLD_results/confidence_direct_mask_redraft_linearspec_20260821_173928/report.md`。
