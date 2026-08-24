# NLD 复现与解码观察关键证据

本文从外部 observation 结果中保留与后续方法设计直接相关的最小证据集。运行来源、完整性和统一口径见 `../basis/NLD_experiment_provenance_and_metrics_zh.md`；复现命令仍以 `configs/observations/` 为准。

## 1. 十项复现：TPF

|数据集|S16|P8|P16|P32|
|:--:|:--:|:--:|:--:|:--:|
|HumanEval|4.0364|3.3121|5.0760|6.1590|
|MBPP|3.6150|2.9554|3.9731|4.5612|
|LCB-C++|3.1692|2.9319|3.7763|4.2704|
|GSM8K|4.3232|3.2977|5.0009|6.3515|
|MATH-500|4.7433|3.4480|5.4520|7.2771|
|AIME25|4.4999|3.4490|5.4505|7.0329|
|GPQA|4.8486|3.4270|5.3499|7.2937|
|MMLU|3.3017|2.8558|3.8032|4.3721|
|IFEval|4.1946|3.2088|4.8354|6.3371|
|九集均值|4.0813|3.2095|4.7464|5.9617|

表中变量：S16 是 SGLang LinearSpec L16；P8/P16/P32 是原生 PyTorch LinearSpec 的 block length 8/16/32；单元均为 TPF。例如 P16 的 GSM8K `5.0009` 表示每次物理 encoder forward 平均得到约 5.00 个 completion token。九集均值排除 AIME24且各数据集等权。

## 2. 十项复现：Accuracy

|数据集|S16|P8|P16|P32|
|:--:|:--:|:--:|:--:|:--:|
|HumanEval|77.4390%|78.0488%|76.8293%|75.6098%|
|MBPP|67.1958%|68.7831%|68.2540%|67.4603%|
|LCB-C++|31.2775%|32.1586%|29.0749%|29.2952%|
|GSM8K|93.4799%|93.1766%|93.7074%|93.1766%|
|MATH-500|87.6000%|87.0000%|88.4000%|87.8000%|
|AIME25|30.0000%|40.0000%|36.6667%|33.3333%|
|GPQA|36.0406%|36.5482%|37.0558%|38.0711%|
|MMLU|78.9560%|79.0201%|78.9489%|79.0058%|
|IFEval|69.0219%|69.2367%|68.8558%|69.6177%|
|九集均值|63.4456%|64.8858%|64.1992%|63.7078%|

表中变量：S16/P8/P16/P32 与上一表相同；Accuracy 口径按 Basis 文档定义。例如 HumanEval 使用 Base pass@1，而 IFEval 使用 average_score。AIME25 只有 30 题，3 题变化就是 10 个百分点，不能单独据其宏平均变化判断算法损害精度。

复现结论：在原生 PyTorch baseline 内，block 从 16 增至 32 后九集等权 TPF 从 `4.7464` 增至 `5.9617`，约提升 `25.60%`；主 Accuracy 没有显示随 block 单调变化。SGLang 与 PyTorch 的执行栈和计时口径不同，S16 主要证明十项链路可运行，不应把其 TPS/TPF差值归结为单一算法因素。

## 3. Confidence 与正确 token rank

### 3.1 两后端九集等权摘要

|后端|AccC|RejC|R2|R2-3|
|:--:|:--:|:--:|:--:|:--:|
|SGLang|0.8354|0.3282|52.02%|67.75%|
|PyTorch|0.9539|0.5688|57.46%|73.33%|

表中变量：AccC/RejC 是 accepted/rejected draft token 的平均 confidence；R2 是 verifier 正确 token 在 draft 分布中恰为第 2 名的比例；R2-3 是 rank 2 或 3 的比例。例如 PyTorch 的 R2=`57.46%` 表示错误 draft token 被拒绝时，正确 token 超过一半恰是第二高概率候选。数值先按数据集统计再做九集等权，后端间绝对 confidence 不宜直接校准。

### 3.2 PyTorch 逐数据集

|数据集|AccC|RejC|R2|R2-3|MedR|
|:--:|:--:|:--:|:--:|:--:|:--:|
|HumanEval|0.9647|0.6096|61.33%|76.35%|2|
|MBPP|0.9355|0.5916|59.19%|76.16%|2|
|LCB-C++|0.9291|0.5645|55.93%|72.58%|2|
|GSM8K|0.9593|0.6108|64.49%|79.27%|2|
|MATH-500|0.9676|0.6068|63.11%|78.43%|2|
|AIME25|0.9679|0.5894|60.41%|76.56%|2|
|GPQA|0.9695|0.5654|56.76%|72.95%|2|
|MMLU|0.9358|0.5376|53.05%|69.27%|2|
|IFEval|0.9558|0.4435|42.88%|58.37%|3|
|九集均值|0.9539|0.5688|57.46%|73.33%|—|

表中变量：AccC、RejC、R2、R2-3 同上一表；MedR 是每个数据集 rejected correct-token rank 的中位数。例如 HumanEval 的 MedR=2，表示至少一半拒绝事件的正确 token 排名不高于 2。九集行的 MedR 不对各中位数再次求平均，因此记为“—”。

该证据支持把第二候选 B 用作高价值分支，但 R2 远低于 100%，所以固定 B 只能覆盖一部分真实修正；IFEval 的 rank 分布尤其弱。

## 4. `token_y_drop_pct>0.15` 的错误覆盖

|数据集|FPR|Recall|Prec|
|:--:|:--:|:--:|:--:|
|HumanEval|6.40%|73.31%|42.43%|
|MBPP|12.07%|76.47%|42.17%|
|LCB-C++|12.74%|78.75%|45.01%|
|GSM8K|7.42%|75.81%|40.94%|
|MATH-500|5.91%|76.30%|41.66%|
|AIME25|5.82%|77.76%|42.15%|
|GPQA|5.15%|77.55%|44.63%|
|MMLU|10.85%|79.09%|46.51%|
|IFEval|6.33%|84.05%|44.17%|
|九集均值|8.08%|77.68%|43.30%|

表中变量：FPR 是被阈值误标的 accepted token/全部可评估 accepted token；Recall 是被阈值覆盖的 rejected token/全部可评估 rejected token；Prec 是所有被标 token 中实际 rejected 的比例。例如 GSM8K 的 Recall=`75.81%`、FPR=`7.42%`，说明阈值覆盖约四分之三错误 token，同时误标约 7.4% 正确 token。

上述统计是 token 级累计阈值，不等同于“每轮第一个越阈位置正好是第一个错误位置”。后续 overlap 方法使用 first-crossing，还需由在线状态计数验证定位质量。

### 4.1 Pooled-token 阈值诊断

|信号|阈值|Prec|Recall|FPR|F1|
|:--:|:--:|:--:|:--:|:--:|:--:|
|drop_pct|0.15|45.58%|78.96%|9.98%|0.5780|
|drop_pct|0.23|50.10%|70.74%|7.46%|0.5866|
|drop_abs|0.19|48.20%|71.19%|8.10%|0.5748|

表中变量：信号是相对下降 `drop_pct=1-C_i/C_imean` 或绝对下降 `drop_abs=C_imean-C_i`；阈值是标记边界；Prec/Recall/FPR 同上一表；F1 是 Prec 与 Recall 的调和平均。例如把 drop_pct 从 0.15 提高到 0.23，会牺牲约 8.22 个百分点 Recall，换取更高 precision 和更低 FPR。

本表是把全部数据集 token 计数合并后的 micro，不是数据集等权。它用于阈值形状诊断，不能代替上一表的九集均值。

## 5. Draft/final alignment

|数据集|All|P1|P8|P15|O1|O4|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HumanEval|58.24%|95.28%|52.58%|37.41%|32.98%|22.30%|
|MBPP|54.25%|95.47%|47.97%|29.36%|32.80%|21.73%|
|LCB-C++|44.40%|93.18%|36.48%|21.32%|29.16%|16.35%|
|GSM8K|61.84%|95.72%|57.97%|40.66%|34.46%|23.41%|
|MATH-500|65.01%|95.91%|61.61%|45.65%|32.55%|22.32%|
|AIME25|61.00%|95.25%|56.79%|41.84%|30.66%|19.32%|
|GPQA|64.59%|95.77%|60.13%|48.86%|31.55%|18.55%|
|MMLU|46.61%|93.38%|38.69%|26.38%|29.47%|16.79%|
|IFEval|53.64%|94.21%|46.69%|37.03%|28.91%|16.41%|
|九集均值|56.62%|94.91%|50.99%|36.50%|31.39%|19.69%|

表中变量：All 是全部可比 draft candidate 与最终输出同位置 token 的 micro alignment；P1/P8/P15 是 block 内第 1/8/15 个 draft candidate 的 alignment；O1/O4 是首次拒绝后第 1/4 个位置仍与最终输出一致的比例。例如 HumanEval 的 P1=`95.28%`，但首次拒绝后的 O1 只有 `32.98%`。

alignment 随 block 位置明显下降，首次拒绝之后更低。这直接解释了为什么“复用靠后的旧/重生成后缀”容易增加下一轮 verify rounds：修正位置之后的 token 本来就处于较弱条件分布中。

## 6. 可迁移结论

1. `drop_pct` 对 rejection 有稳定区分度，但单阈值 precision 约 40%–50%，它是候选信号而不是准确错误定位器。
2. rejected correct token 的 rank 2 概率在两后端九集均值均超过 50%，支持固定第二候选分支；但 IFEval 等数据集表明该规律并不统一。
3. 草稿越靠后越不可靠；首次拒绝后 O1/O4 九集均值只有 `31.39%/19.69%`，任何后缀复用都必须同时考虑“省 forward”和“增加 verify round”。
4. 增大 block 从 P16 到 P32 已提供很强 TPF baseline，新的 L16 方法至少应同时报告相对 P16 和 P32 的差距。

## 7. 原始与复现入口

- SGLang/PyTorch 十项复现：`configs/observations/NLD_SGLang_NeMoSkills_eval_pipeline_zh.md`、`NLD_PyTorch_NeMoSkills_eval_pipeline_zh.md`。
- Confidence/rank：`NLD_SGLang_LinearSpec_confidence_trace_zh.md`、`NLD_PyTorch_NeMoSkills_LinearSpec_confidence_trace_zh.md`。
- Alignment：`NLD_SGLang_LinearSpec_draft_alignment_zh.md`。
- Drop 离线分析：`NLD_PyTorch_LinearSpec_low_confidence_offline_zh.md`。
