# NLD 实验来源与统一统计口径

本文固定本轮迁移所依据的可信结果源和公共指标定义。数字结果分别见 `../observations/` 与 `../method/`。

## 1. 可信结果源

|简称|类型|外部结果目录|状态|
|:--:|:--:|:--:|:--:|
|S16|SGLang L16|`observations/sglang_nemo_eval_results/eval_20260628_160944`|10 metrics，0 error|
|P8|PyTorch L8|`observations/pytorch_nemo_eval_results/eval_20260806_001910`|10 metrics，0 error|
|P16|PyTorch L16|`observations/pytorch_nemo_eval_results/eval_20260804_120138`|10 metrics，0 error|
|P32|PyTorch L32|`observations/pytorch_nemo_eval_results/eval_20260804_114935`|10 metrics，0 error|
|SC|SGLang confidence|`observations/sglang_linearspec_confidence_results/linearspec_confidence_20260628_233332`|10 summaries|
|PC|PyTorch confidence|`observations/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154`|10 summaries|
|DA|SGLang alignment|`observations/sglang_linearspec_draft_alignment_results/linearspec_draft_alignment_20260803_140928`|10 summaries|
|PD|PyTorch drop 离线分析|`observations/pytorch_linearspec_low_confidence_offline_results/offline_low_confidence_20260809_112909`|10集+pooled summary|
|O|Confidence-Overlap|`confidence_overlap_linearspec_20260817_033248`|completed，10 metrics，0 error|
|M|Autonomous Mask-Redraft|`confidence_mask_redraft_linearspec_20260817_182905`|completed，10 metrics，0 error|
|D|Strict Direct MASK-Redraft|`confidence_direct_mask_redraft_linearspec_20260821_173928`|completed，10 metrics，0 error|

表中变量：简称供后续窄表引用；类型说明后端、block 或方法；外部结果目录均相对 `/data/home/wly/dLLM/NLD_results/`；状态是完整性检查。例如 `P16` 表示原生 PyTorch、`block_length=16` 的完整十项 baseline。

## 2. 公共配置

|项|主实验设置|
|:--:|:--:|
|模型|Nemotron-Labs-Diffusion-8B|
|评测|NeMo-Skills，repeat/pass@1=1|
|精度|BF16|
|模式|LinearSpec LoRA|
|解码|greedy，temperature=0|
|预算|context 10240，completion 8192|
|并发|单模型请求串行|
|方法阈值|`token_y_drop_pct>0.15`|

表中变量：项是配置维度；主实验设置是比较时采用的值。例如“解码=greedy”表示每个 verifier logit 取 argmax；HTTP 中记录的 `top_p=0.95` 未被原生模型应用。S16 的服务实现与 PyTorch 不同，TPS 不作严格横比。

## 3. 指标定义

|短名|定义|
|:--:|:--:|
|TPF|`completion_tokens/physical_nfe`|
|NFE|物理 encoder 调用次数；双行 fused batch 仍算 1 次|
|Acc|数据集主 Accuracy，单位 `%`|
|TPS|completion token/模型生成秒数|
|Round|一次逻辑 draft-verify 轮|
|Saved|下一轮消费预取草稿，实际省掉的 normal draft forward|
|Hit%|方法定义的可信命中数/row 1 尝试数|
|Save%|`Saved/Round`|
|Tok/R|`completion_tokens/Round`|
|Rows/NFE|每次物理 forward 平均处理的 batch row 数|
|QTok/NFE|每次物理 forward 平均处理的 `batch×query_length`|

表中变量：短名用于后续表格；定义给出计算口径。例如返回 100 token、模型执行 20 次 encoder，则 `TPF=5`；100 轮中复用 15 次 row 1，则 `Save%=15%`。Rows/NFE 和 QTok/NFE 暴露“双行只算一次 NFE”隐藏的实际计算。

## 4. Accuracy 与平均

|数据集|主 Acc|
|:--:|:--:|
|HumanEval/MBPP|Base pass@1|
|LiveCodeBench-C++|accuracy|
|IFEval|average_score|
|其余|symbolic_correct|

表中变量：数据集表示 scorer；主 Acc 是本文统一采用的字段。例如 HumanEval 的 Plus pass@1 可以补充分析，但主表使用 Base pass@1。

九集均值先在每个数据集内部得到指标，再做九个数的算术平均。MMLU 的 14,042 条请求和 AIME25 的 30 条请求权重相同。AIME24 原始值可用于排障，但不进入本目录主统计。

## 5. 可比性边界

- P16/P32/O/M/D 都是原生 PyTorch、NeMo-Skills、greedy、LinearSpec LoRA；O/M/D 为 L16，最适合直接比较 TPF。
- 各轮运行日期、GPU 和显存预留不完全相同，Accuracy/TPF 可作算法观察，TPS 只作参考。
- verifier-only 设计不自动保证 BF16 下逐 token 位相同；不同 batch shape、attention mask 和 round 切分可能在近似 logits 处改变 argmax。
- PD 的 pooled micro 先合并所有 token 计数，必然偏向 MMLU/LiveCodeBench；只用于说明 token 级分类能力，主结论仍用逐数据集和九集等权结果。
