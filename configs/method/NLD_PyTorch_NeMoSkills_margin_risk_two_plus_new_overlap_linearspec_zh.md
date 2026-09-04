# NLD PyTorch + NeMo-Skills：固定 margin-risk 的 P1/P2 + always-New overlap 实验

> 实验入口：`method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh`  
> 独立代码：`method/margin_risk_two_plus_new_overlap_linearspec/`  
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results/`  
> 正式策略：greedy、LinearSpec LoRA、block size 16、draft threshold 0、固定 `margin_risk_threshold=0.5`  
> 主报告：排除 AIME24 的九数据集，各数据集始终等权，MMLU 默认最后运行
> 默认评测口径：只以效率数据是否可用作为完成标准，不报告 accuracy

## 1. 实验目标与隔离边界

本方法是既有 margin-risk multi-overlap 的独立变体，不修改下列任何既有实现：

- 论文复现与原始 PyTorch/NeMo-Skills pipeline；
- `method/confidence_overlap_linearspec/`；
- `method/margin_risk_overlap_linearspec/`；
- `method/margin_risk_multi_overlap_linearspec/`；
- 此前 observation 实验及其结果。

新目录自带 generation、四行 hybrid attention、分段 LoRA、OpenAI-compatible server、NeMo-Skills pipeline、指标合并、增量报告和测试。并行安全措施包括：

- 默认 `--port 0`，由操作系统原子分配空闲端口；
- 每次运行在结果根目录下原子创建时间戳子目录，同一秒冲突时追加 `_01`、`_02`；
- 内部工作目录含 PID；
- server、显存预留进程和日志均为本次运行私有；
- 退出 trap 只清理本次运行创建的进程和内部目录。

默认结果形态如下。`Settings.json` 和报告模板在评测开始前就创建；每完成一个数据集，`report.md` 立即原子刷新。

```text
/data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results/
└── margin_risk_two_plus_new_overlap_YYYYMMDD_HHMMSS/
    ├── Settings.json
    ├── report.md
    ├── metrics_<dataset>.json
    └── artifacts/<dataset>/
        ├── output-rs0.jsonl
        ├── pytorch_request_stats.jsonl
        ├── pytorch_margin_risk_two_plus_new_overlap_metrics_summary.json
        └── pytorch_benchmark.log
```

### 1.1 默认效率优先与显存不足处理

入口默认启用 `--efficiency-only`。NeMo-Skills 继续发送真实数据集请求，但最终报告不读取或展示 accuracy，只保留 TPF、NFE、TPS、每 forward dense token/padding、各 overlap 状态及跨轮接收等效率指标。

若某个 request 在生成期间发生 CUDA OOM：

- server 写入一条 `ok=false` 记录，保留错误类型、prompt token 数和生成预算；
- 清理可回收 CUDA cache，并返回空占位响应，让同一数据集后续 request 继续；
- OOM request 不进入任何效率均值或 overlap 状态比例；
- 报告通过 `Att`、`OK`、`Fail`、`OOM`、`Cov=OK/Att` 披露覆盖率。例如 Att=164、OK=163、OOM=1 时，TPF/NFE/TPS 和状态表只来自 163 个成功 request。

如果全部样本生成已留下 `output*.jsonl.done` 完成标记、存在成功记录、且 NeMo accuracy/scorer 失败或未产生 `metrics.json`，pipeline 会从独立 request stats 创建 efficiency-only metrics，报告仍可更新。生成中途退出、模型加载就 OOM、整个数据集没有任何成功 request、或出现非 OOM 生成错误时，没有完整可靠的效率结果，仍按失败处理。server 还默认使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解显存碎片，但它不能解决总容量不足。

如需恢复旧的严格 accuracy 口径，传 `--require-accuracy`；此时单 request OOM 或 scorer 失败都会令数据集失败。

## 2. 策略定义

### 2.1 margin-risk 与前两个位置

对当前 draft block 的位置 `1..L-1` 从左向右扫描；位置 0 是已知 seed，不参与定位。每个位置先从 softmax 归一化中排除 MASK，再计算：

```text
margin = 最高概率 - 第二高概率
margin_risk = 1 - margin
```

收集最先出现的两个严格满足下式的位置：

```text
margin_risk > 0.5
```

等于 0.5 不触发。扫描仍继续到 block 末尾，因此除保存 P1/P2 外，也能准确知道整块 crossing 总数是 0、1、2 还是 3 及以上；第三及以后的位置只计数，不再构造纠错分支。

示例：某位置最高和第二高概率为 0.65、0.25，则 margin=0.40、margin-risk=0.60，会触发；0.75、0.15 对应 margin-risk=0.40，不触发。

### 2.2 候选修正分支

对 P1、P2 各自选择替代 token B：排除该位置原 draft token A、MASK 和 EOS 后，取 logit 最大的 token。B 因排除了 EOS，不一定等同于未经约束的 raw top2。

每个可执行位置 p 构造一条 prospective row：

```text
[当前 draft 的 0..p-1 causal prefix] + [B + L-1 个 MASK]
```

prefix 使用 causal attention；从 B 开始的 L-token prospective block 使用 bidirectional attention 和 LinearSpec LoRA。只有 verifier row 能提交当前输出和 canonical KV；候选 row 只能在 verifier 证明“首错位置就是 p 且 B 正确”后成为下一轮 draft。

### 2.3 continuation 分支

每一轮只要剩余生成长度、context 和 thinking 边界允许，无论当前有多少个严格 crossing，都追加一条 continuation row：

```text
[当前完整 L-token draft causal prefix] + [L 个 MASK 的 prospective new]
```

若当前 verifier 的 L-1 个 draft token 全部通过，它最后产生一个 bonus token。仅当：

```text
bonus token == draft sequence new 的第 0 个 token
```

才把完整 new block 复用为下一轮 draft。这样在 block size 16 整块通过时，也有机会掩盖下一轮普通 draft forward。

New 不会因为 crossing 数量达到 3 或更多而被取消。它只可能因为没有下一轮生成预算、`cache + 2L` 超过上下文上限，或仍处于受限 thinking 阶段而不可构造。报告把“已构造但 bonus 不匹配”和“受边界限制没有 New”分别记录。

P1、P2 和 New 产生的完整 draft 都保留 confidence、top1-top2 margin 和新的 crossing 信息。当前 verify 选中唯一有效分支后，该分支会在下一轮继续进入相同的 `Verify + P1′ + P2′ + New′` 状态机；不会在同一个 forward 内递归展开所有分支。

### 2.4 最多四行与 padding

本版保留原始组 batch，不做展平：

|crossing 总数|speculative row|含 verifier 总行数|
|:---:|:---:|:---:|
|0|new|2|
|1|P1 + new|3|
|2|P1 + P2 + new|4|
|3 及以上|P1 + P2 + new|4|

表格描述的是候选均满足边界条件的理想情况。某个 P 或 New 不可执行时，实际 row 数相应减少；每个 forward 始终不超过 4 row。

所有 row pad 到同一个 query length Q。attention mask 能阻止有效 token 看见 padding，却不会跳过 dense QKV/MLP，因此 padding 仍实际占用计算。本实验记录：

```text
FwdTok = row 数 × 公共 Q
有效 token = 各 row 有效长度之和
padding token = FwdTok - 有效 token
```

这些值按 prefill、normal draft、normal verify、multi fused 分开保存，并为全部 forward、解码 forward、仅 fused forward 报告均值、最小值、P50/P90/P95/P99、最大值和 padding 比例。

## 3. 验证、选择与互斥状态

设 q 为 verifier 从左向右第一个不通过的 draft 位置；整块通过时 q 不存在。每次实际 fused 轮只属于以下 11 个状态之一：

|字段|含义|例子|
|:---:|:---:|:---:|
|`miss_no_candidate_error`|没有实际候选，但当前发生首错|仅有 new，q=4|
|`miss_before_first`|q 在首个实际预测位置之前|P1=5，q=3|
|`miss_between_candidates`|q 位于两个预测位置之间|P1=3、P2=7，q=5|
|`miss_after_last`|q 在最后预测位置之后，所有预测位置原 token 均正确|P2=8，q=11|
|`candidate_1_fixed`|q=P1 且 P1 的 B 正确|P1=3，B 命中|
|`candidate_1_wrong`|q=P1 但 P1 的 B 错误|正确 token 是 C|
|`candidate_2_fixed`|q=P2 且 P2 的 B 正确|第二个预测命中|
|`candidate_2_wrong`|q=P2 但 P2 的 B 错误|第二个预测未修正|
|`full_continuation_hit`|整块通过且 bonus=new[0]|new 可验证复用|
|`full_continuation_miss`|整块通过但 bonus≠new[0]|new 不可复用|
|`full_continuation_absent`|整块通过但本轮因边界没有 new|已接近生成终点|

P1/P2 指原始从左向右 crossing rank。即使 P1 因预算边界被跳过、P2 仍可执行，P2 命中仍归入 `candidate_2_fixed`，不会重编号为 P1。

状态次数之和必须等于实际 fused 轮数 `prefetch_attempts`，字段 `outcome_partition_valid` 自动校验。报告另给出：

- 预测未命中实际错误位置的总次数和比例；
- P1/P2 修正正确、错误的独立次数与比例；
- 整块通过且 new 命中；
- 整块通过但 new 未命中或不存在；
- C3+ 轮次的 New 覆盖率、条件验证命中率和实际复用率，用于直接衡量 New 替代 P3；
- 各状态本轮 verify 平均接收 token；
- 各状态同一 request 下一轮 verify 平均接收 token；
- 配对样本逐轮的“下一轮减当前轮”均值。

终止轮没有下一轮时不补成 0，只降低 `NextCov`。

## 4. 推荐单行命令

本节所有命令均为单行形式。

### 4.1 查看帮助

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --help
```

### 4.2 只做参数检查

该命令不创建结果目录、不加载模型、不运行评测。

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --gpu-device 0 --gpu-memory-reserve-gb 0 --dry-run
```

### 4.3 一题全链路 smoke

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --context-length 2112 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --efficiency-only --keep-runtime
```

### 4.4 正式九数据集全量实验

MMLU 明确放在最后；该命令不运行 AIME24。报告在每个数据集完成后增量刷新，最终宏平均为九数据集等权。

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.45 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device 0 --gpu-memory-reserve-gb 30 --require-accuracy --output-path /data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results
```
加 --require-accuracy：才恢复原来 pipeline 的严格行为——请求 OOM 或评分失败都会让数据集失败，并要求 accuracy/scorer 正常完成。

### 4.5 常用十数据集，主报告仍排除 AIME24

这里 AIME24 位于中间，MMLU 仍最后运行。

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --output-path /data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results
```

### 4.6 单数据集和自定义多数据集

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1,aime25:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

`human-eval:1` 和 `mbpp:1` 中的 `:1` 是 pass@1，不是只取 1 个 sample。严格 accuracy 模式下，当前 NeMo-Skills/EvalPlus scorer 要求二者跑完整题集；默认效率模式允许子集生成，若 scorer 拒绝该子集，pipeline 会从成功 request stats 补建效率 metrics。

### 4.7 指定 GPU、自动选择 GPU 和等待 GPU

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 3
```

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device auto --gpu-candidates 1,3,5 --gpu-min-free-gb 28 --gpu-wait-seconds 1800
```

后端为单 GPU；`--gpu-devices` 只是兼容别名，也只允许一个 ID。

### 4.8 预留显存

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 3 --gpu-memory-reserve-gb 20
```

`--gpu-memory-reserve-gb` 会在模型加载前由独立进程真实占用指定显存，用于模拟受限环境；它不是“替模型保留可用空间”。本实验最多四行，显存需求明显高于单行 baseline 和两行 overlap。

### 4.9 显式端口、结果路径和 baseline

```bash
bash method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 2 --tokens 512 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 1 --port 19081 --output-path /data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935 --keep-runtime
```

并行运行应优先保留 `--port 0`。显式端口已占用时，原子 bind 会失败，不会误连到其他实验。

### 4.10 只重建已有结果的报告

```bash
python method/margin_risk_two_plus_new_overlap_linearspec/report.py --result-dir /data/home/wly/dLLM/NLD_results/margin_risk_two_plus_new_overlap_results/margin_risk_two_plus_new_overlap_YYYYMMDD_HHMMSS --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935
```

该命令只读现有 settings/metrics，原子重写 `report.md`，不启动模型。

## 5. 参数详解

### 5.1 模式、模型与数据集

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--mode overlap_lora`|normal draft 与 speculative suffix 使用 LinearSpec LoRA|正式模式|
|`--mode overlap_base`|不加载 LoRA 的消融模式|可选|
|`--model PATH`|模型目录|Nemotron-Labs-Diffusion-8B 本地路径|
|`--served-model-name NAME`|本地 OpenAI API 模型标签|方法专用标签|
|`--lora-path DIR`|LinearSpec LoRA 目录|`<model>/linear_spec_lora`|
|`--benchmarks LIST`|逗号分隔，支持单/多数据集|常用十项，MMLU 最后|
|`--tokens N`|每请求最多返回 completion token|8192|
|`--max-samples N`|每数据集最多取前 N 条|全量；HumanEval/MBPP 禁用|
|`--quick-test`|NeMo-Skills quick test|关闭；HumanEval/MBPP 禁用|
|`--num-chunks N`|客户端 chunk 数|等于 client concurrency|
|`--client-concurrency N`|HTTP 并发请求数|1；server 内模型生成串行|
|`--math-prompt-config NAME`|数学任务 prompt config 覆盖|空|
|`--efficiency-only`|只要求成功 request 的效率统计；OOM request 跳过|默认启用|
|`--require-accuracy`|恢复严格 accuracy/scorer 完成要求|关闭|

### 5.2 解码与 thinking

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--block-length N` / `--block-size N`|draft、verify、prospective block 的 L|16，至少 2|
|`--threshold V`|LinearSpec draft unmask threshold|必须为 0|
|`--margin-risk-threshold V`|严格 crossing 阈值|0.5，范围 0 到 1|
|`--temperature V`|生成温度|必须为 0|
|`--top-p V`|协议对齐并记录；原生方法当前不应用|0.95|
|`--context-length N`|prompt 加生成预算上限|未指定时为最大所需生成预算+2048|
|`--enable-thinking`|chat template 开启 thinking|关闭|
|`--disable-thinking`|显式向 NeMo-Skills 传递关闭 thinking|关闭|
|`--max-thinking-tokens N`|超预算后强制结束 thinking|空|
|`--keep-thinking`|评分输出保留 thinking|关闭|
|`--strip-thinking`|支持的任务剥离 thinking 后评分|关闭|

候选数上限、最大 row 数和 continuation 条件是本实验定义，不提供命令行修改，避免不再对应本实验。`--drop-pct-threshold` 不被接受。

### 5.3 GPU、端口和路径

|参数|含义|默认|
|:---:|:---:|:---:|
|`--gpu-device ID/auto`|指定物理 GPU；auto 先满足空闲显存，再优先低利用率|auto|
|`--gpu-devices ID`|兼容别名，只允许单 ID|auto|
|`--gpu-min-free-gb V`|auto 的最低空闲显存|24|
|`--gpu-candidates LIST`|auto 只在这些 GPU 中选择|全部|
|`--gpu-wait-seconds N`|等待满足条件 GPU 的秒数|0|
|`--gpu-memory-reserve-gb V`|模型加载前额外真实占用显存|0|
|`--dtype DTYPE`|bfloat16、float16、float32 及别名|bfloat16|
|`--port N`|本地 server 端口；0 为原子自动分配|0|
|`--output-path DIR`|时间戳子目录的结果根|专用 results 子文件夹|
|`--baseline-block16-dir DIR`|报告对照的 B16 baseline|20260804 全量结果|
|`--baseline-block32-dir DIR`|报告对照的 B32 baseline|20260804 全量结果|
|`--pytorch-python PATH`|模型 server Python|`nld_sglang` Python|
|`--eval-python PATH`|NeMo-Skills Python|同 PyTorch Python|
|`--nemo-skills-data-dir DIR`|持久数据/cache 根目录|`/data1/linyewei/datasets/NLD`|
|`--google-research-dir DIR`|IFEval scorer checkout|`<data-dir>/google-research`|
|`--keep-runtime`|保留隐藏工作目录和调试日志|关闭|
|`--dry-run`|只解析打印，不写目录、不加载模型|关闭|

### 5.4 Judge benchmark

|参数|含义|默认|
|:---:|:---:|:---:|
|`--judge-model NAME`|覆盖数据集默认 judge|数据集默认|
|`--judge-server-address URL`|OpenAI-compatible judge 地址|OpenAI 默认地址|
|`--judge-server-type TYPE`|judge server 类型|openai-compatible|
|`--judge-concurrency N`|judge 并发|4|
|`--mt-bench-max-tokens N`|MT-Bench 每轮候选预算|1024|
|`--alpaca-eval-max-tokens N`|AlpacaEval 候选预算|2048|
|`--skip-judge-api-key-check`|跳过入口 API key 预检|关闭|

## 6. 报告内容与等权口径

`report.md` 在创建结果目录后立即生成模板，并在每个数据集完成后更新。每张表前都有中文变量解释和例子，列使用居中紧凑 Markdown。报告包括：

1. 新方法配置与既有 B16/B32 greedy baseline 核验；
2. 九数据集逐项请求覆盖率、TPF、NFE、TPS 与等权平均；
3. crossing 数、2/3/4 row 次数、候选/new 分支、验证命中、实际复用漏斗；
4. 漏报、预测后出错、P1/P2 修正正确/错误、new 命中/未用的次数和比例；
5. 每数据集每 forward 的实际 dense query-token 数和分布；
6. C3+ 高风险轮的 New 覆盖、验证命中和实际复用，直接对应替换 P3 的新增收益；
7. 全部 11 个互斥状态的本轮/下一轮接收长度与差值；
8. 分区完整性和统计口径说明。

九数据集等权的计算顺序是：先在每个数据集内部计算比例、均值或分位数，再对已完成数据集做算术平均。MMLU 即使 sample 数最多也只有一个数据集权重。绝对次数另列总计，不用作主要等权比例。AIME24 即使运行也不会进入主表或宏平均。

报告不展示任何 accuracy 列。评分即使存在也不纳入报告；scorer 没完成时，metrics 由成功 request stats 创建。只有 `OK` request 参与效率和状态聚合，`Fail/OOM` 用来解释覆盖率，不能把 `Cov<100%` 的结果当成全样本结果。

当前默认 TPF 排除 prompt prefill，以 `completion_tokens/decode_forward_passes` 计算并与 SGLang 对齐；`total_forward_passes/physical_nfe` 继续保留含 prefill 的总调用用于审计。TPF 的一个 fused forward 仍只计一次物理 encoder 调用，因此不能独立代表算力效率。必须同时查看 `FwdTok均`、`Rows均`、`Q均` 和 `Pad率`；这些指标能反映 dense token 工作量，但仍不是包含 KV 长度影响的完整 FLOPs。

默认 B16/B32 路径是历史结果。如果其中没有 `metric_schema_version=2` 和 `decode_forward_passes`，报告只在内存中用“历史总 forward−成功请求数”扣除每请求一次 LinearSpec prefill，再计算 baseline TPF/NFE；不会修改历史文件。新运行的 schema v2 baseline 会直接使用其 decode-only 字段。

## 7. 自检与真实模型 smoke

静态检查：

```bash
bash -n method/margin_risk_two_plus_new_overlap_linearspec/eval_margin_risk_two_plus_new_overlap.sh method/margin_risk_two_plus_new_overlap_linearspec/run_pipeline.sh
```

```bash
python -m py_compile method/margin_risk_two_plus_new_overlap_linearspec/*.py method/margin_risk_two_plus_new_overlap_linearspec/tests/*.py
```

离线单元测试：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python -m unittest discover -s method/margin_risk_two_plus_new_overlap_linearspec/tests -v
```

真实模型四行融合 smoke：验证 `verifier + P1 + P2 + New`，比较 fused verifier 与普通 causal verifier 的 greedy token，检查 P1/P2 prospective seed、New 分支、padding 计数和 canonical cache 不变。

```bash
CUDA_VISIBLE_DEVICES=3 /data/home/wly/.conda/envs/nld_sglang/bin/python method/margin_risk_two_plus_new_overlap_linearspec/tests/smoke_fused_two_plus_new.py --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --block-size 16 --dtype bfloat16
```
