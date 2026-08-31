# NLD PyTorch + NeMo-Skills：固定 margin-risk 重起草 overlap 实验

> 实验入口：`method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh`  
> 新代码目录：`method/margin_risk_overlap_linearspec/`  
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/`  
> 正式策略：固定 `margin_risk_threshold=0.5`，greedy，draft threshold 为 0  
> 主报告范围：除 AIME24 外的九个数据集，数据集等权宏平均
> 默认评测口径：只以效率数据是否可用作为完成标准，不报告 accuracy

## 1. 实验目标与隔离边界

本实验沿用 `confidence_overlap_linearspec` 的两行融合 verify/draft overlap、第二候选 B、verifier-only 提交、prospective draft 复用和 LoRA 分段路由，仅替换潜在错误位置的定位规则。旧实验目录、模型 remote code、论文复现入口、此前 observation 代码和既有结果均不会被修改。

新方法有自己的 server、generation、hybrid mask、LoRA 路由、指标合并、报告生成和测试文件。每次运行使用：

- `/data/home/wly/dLLM/NLD_results/margin_risk_overlap_linearspec_<时间戳>/` 独立结果目录；
- 带 PID 的隐藏工作目录；
- `--port 0` 时由操作系统原子分配空闲端口；
- 独立 server 和显存预留进程，并在退出时清理；
- 原子创建结果目录；同一秒并行启动会追加 `_01`、`_02`。

因此可以与原始 PyTorch+NeMo-Skills、`confidence_overlap_linearspec` 和其他实验并行运行。GPU generation 仍在每个 server 内串行，`--client-concurrency` 增大只会增加 HTTP 排队，不会把本实现变为 continuous batching。

### 1.1 默认效率优先与显存不足处理

入口默认启用 `--efficiency-only`。NeMo-Skills 仍负责发送真实数据集请求，但最终报告不读取或展示 accuracy，只汇总此前设计的 TPF、NFE、TPS、overlap 状态和跨轮接收等效率指标。

若某个请求生成期间发生 CUDA OOM：

- server 将该请求写成一条 `ok=false` 的失败记录，保留错误类型、prompt 长度和请求预算；
- 清理可回收的 CUDA cache，并向 NeMo-Skills 返回空占位响应，使同一数据集的后续 request 继续；
- 该失败 request 不进入 TPF、NFE、TPS、状态比例或接收长度的分子/分母；
- 报告用 `Att`、`OK`、`Fail`、`OOM` 和 `Cov=OK/Att` 披露覆盖情况，不能把 `Cov<100%` 的均值解释为全样本结果。

若全部样本生成已留下 `output*.jsonl.done` 完成标记、存在成功请求、且 NeMo 的 accuracy/scorer 阶段失败或未生成 `metrics.json`，pipeline 会从独立 request stats 新建仅含效率的 metrics，继续生成增量报告。若生成中途退出、模型加载即 OOM、一个成功请求都没有，或者出现非 OOM 的生成错误，则没有完整可信的效率统计，数据集仍会失败。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 只缓解碎片，不保证总显存不足时成功。

如需恢复旧的严格 accuracy 口径，显式传 `--require-accuracy`；此时 OOM 或 scorer 失败都会令数据集失败。

## 2. 定位指标与调用链

### 2.1 固定 margin-risk

对 draft block 中的位置 `1..L-1` 逐位置计算；位置 0 是已知 seed，不参与候选定位。每个位置先把 MASK 的 logit 设为负无穷，再在剩余词表上计算概率，EOS 保留：

```text
margin = P_top1 - P_top2
margin_risk = 1 - margin
```

从左到右选择第一个严格满足下面条件的位置 `p`：

```text
margin_risk > margin_risk_threshold
```

正式值为 `margin_risk_threshold=0.5`。等于 0.5 不触发。与旧 drop 指标不同，新规则可以选择 `p=1`。

示例：某位置最高和第二高概率分别是 0.65、0.25，则 margin 为 0.40，margin-risk 为 0.60，所以在阈值 0.5 下会触发；若二者分别是 0.75、0.15，则 margin-risk 为 0.40，不触发。

### 2.2 第二候选 B

找到 `p` 后，B 仍沿用旧 overlap 方案：排除 draft 原 token A、MASK 和 EOS 后，选择 logit 最高的可用 token。B 通常是调整后的第二候选，但如果原始概率第二名是 EOS，它不等于未调整词表中的 raw top2。报告和代码均使用“第二候选 B”而不是笼统称为 raw top2。

### 2.3 两行融合与提交规则

设 block size 为 L：

- row 0：原 draft block 的 causal verifier，后部 padding；
- row 1：复制 `[seed, A 之前的 draft prefix]`，在 `p` 放 B，后接 L-1 个 MASK；
- row 1 的 prefix 使用 causal attention，B 开始的完整 L-token prospective block 使用 bidirectional attention；
- `overlap_lora` 只在 row 1 从 B 开始的 prospective suffix 开启 LinearSpec LoRA；prefill、verifier row 和 causal prefix 都关闭 LoRA；
- 只有 row 0 的 verifier token 和 KV cache 可以提交到最终输出；row 1 永远不能自行改变当前输出。

若首个不通过位置恰为 `p`，且 verifier 在一位 shift 后对应的输出等于 B，则 prospective draft 验证命中。满足 EOS、thinking budget、token budget和上下文限制时，它作为下一轮完整 draft，省去一次普通 dLLM draft forward；下一轮对该 prospective draft 继续使用相同的固定 margin-risk 规则。

## 3. 结果目录创建和增量报告

非 dry-run 启动后，入口先原子创建结果目录，立即写入：

```text
/data/home/wly/dLLM/NLD_results/margin_risk_overlap_linearspec_YYYYMMDD_HHMMSS/
├── Settings.json
└── report.md
```

`Settings.json` 初始状态是 `initialized`，记录原始命令、解析后的全部参数、固定 margin-risk 定义、模型/LoRA、GPU、显存预留、端口、数据目录、两个 baseline 路径和运行目录。server 就绪后写入实际端口；最终状态为 `completed`、`completed_with_errors` 或 `failed`。

每完成一个数据集，程序会保存 `metrics_<dataset>.json` 和 `artifacts/<dataset>/`，然后原子重写根目录的 `report.md`。因此无需等待九个数据集全部完成即可查看已完成部分。AIME24 即使被运行，也不会进入主报告或宏平均。

正式完成后的紧凑结构为：

```text
margin_risk_overlap_linearspec_YYYYMMDD_HHMMSS/
├── Settings.json
├── report.md
├── metrics_<dataset>.json
└── artifacts/<dataset>/
    ├── output-rs0.jsonl
    ├── pytorch_request_stats.jsonl
    ├── pytorch_margin_risk_overlap_metrics_summary.json
    └── pytorch_benchmark.log
```

默认 baseline：

- block 16：`/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138`
- block 32：`/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935`

报告会核验 baseline 的 block size、模式及关键生成设置。两个 baseline 的 GPU 或显存预留不同会被明确提示，TPS 因此只作参考。

## 4. 推荐单行命令

本节每个代码块中的命令均为单行。

### 4.1 查看帮助

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --help
```

### 4.2 不创建目录、不加载模型的参数检查

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --gpu-device 0 --gpu-memory-reserve-gb 0 --dry-run
```

### 4.3 最小全链路 smoke test

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --context-length 2112 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --efficiency-only --keep-runtime
```

### 4.4 正式九数据集全量实验（推荐）

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device 1 --gpu-memory-reserve-gb 0 --efficiency-only --output-path /data/home/wly/dLLM/NLD_results/margin_risk_overlap_linearspec
```
加 --require-accuracy：才恢复原来 pipeline 的严格行为——请求 OOM 或评分失败都会让数据集失败，并要求 accuracy/scorer 正常完成。

该命令不运行 AIME24。它和最终报告的九数据集范围完全一致，并保证九个数据集的宏平均等权。

### 4.5 同时运行常用十数据集但报告排除 AIME24

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --output-path /data/home/wly/dLLM/NLD_results
```

### 4.6 单数据集与多数据集

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1,aime25:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

`human-eval:1` 和 `mbpp:1` 中的 `:1` 表示 pass@1，不是只运行一题。严格 accuracy 模式下，当前 NeMo-Skills/EvalPlus scorer 要求这两个数据集使用完整题集，不能和 `--max-samples` 或 `--quick-test` 同时使用；默认效率模式允许子集生成，若 scorer 拒绝子集则保留成功请求的效率 metrics。

### 4.7 指定或自动选择 GPU

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 3
```

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device auto --gpu-candidates 1,3,5 --gpu-min-free-gb 28 --gpu-wait-seconds 1800
```

本后端是单 GPU 实现，`--gpu-devices` 只是 `--gpu-device` 的兼容别名，逗号分隔的多 GPU ID 会被拒绝。

### 4.8 预留显存

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 3 --gpu-memory-reserve-gb 20
```

该参数会在模型加载前由独立进程真实占用指定 GiB，用于模拟受限可用显存；不是为模型保留空间。退出或异常时 trap 会释放。融合 forward 使用两行且 query 比普通 block 更长，所需显存会随候选位置、block 和 context 改变。

### 4.9 显式端口、结果根目录、baseline 和保留日志

```bash
bash method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 2 --tokens 512 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 1 --port 19081 --output-path /data/home/wly/dLLM/NLD_results --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935 --keep-runtime
```

并行运行推荐保留默认 `--port 0`。显式端口已占用时，server 的原子 bind 会失败，不会连接到别的实验实例。

### 4.10 仅重建某个已有结果目录的报告

```bash
python method/margin_risk_overlap_linearspec/report.py --result-dir /data/home/wly/dLLM/NLD_results/margin_risk_overlap_linearspec_YYYYMMDD_HHMMSS --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935
```

该命令只读取现有 `Settings.json` 和 `metrics_*.json`，原子重写 `report.md`，不启动模型。

## 5. 全部参数

### 5.1 模式、模型和数据集

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--mode overlap_lora`|draft 和 prospective suffix 使用 bundled LinearSpec LoRA|正式模式|
|`--mode overlap_base`|全部使用 base 权重|消融模式|
|`--model PATH`|本地/HF 模型目录|`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`|
|`--served-model-name NAME`|OpenAI API 暴露名|方法专用名|
|`--lora-path DIR`|LoRA adapter 目录|`<model>/linear_spec_lora`|
|`--benchmarks LIST`|逗号分隔，支持单/多数据集|常用十项|
|`--tokens N`|每请求最多返回 completion token|8192|
|`--max-samples N`|每数据集只跑前 N 条|全量；HumanEval/MBPP 禁用|
|`--quick-test`|NeMo quick test|关闭；HumanEval/MBPP 禁用|
|`--num-chunks N`|NeMo 客户端 chunk 数|等于 client concurrency|
|`--client-concurrency N`|并发 HTTP 请求数|1；模型执行串行|
|`--math-prompt-config NAME`|数学任务 prompt config 覆盖|空|
|`--efficiency-only`|只要求成功 request 的效率统计；OOM request 跳过|默认启用|
|`--require-accuracy`|恢复严格 accuracy/scorer 完成要求|关闭|

### 5.2 解码、thinking 与输出处理

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--block-length N` / `--block-size N`|normal draft、verify 和 prospective draft 的 L|16，至少 2|
|`--threshold V`|dLLM 单轮 unmask 阈值|必须为 0|
|`--margin-risk-threshold V`|首个严格 `margin_risk > V` 的位置|0.5，范围 `[0,1]`|
|`--temperature V`|生成温度|必须为 0|
|`--top-p V`|协议对齐并记录，原生方法当前不应用|0.95，范围 `[0,1]`|
|`--context-length N`|prompt 加生成预算上限|未指定时为所需生成预算+2048|
|`--enable-thinking`|chat template 开启 thinking|关闭|
|`--disable-thinking`|显式传递 NeMo disable-thinking|关闭|
|`--max-thinking-tokens N`|超预算后强制 `</think>` seed|空|
|`--keep-thinking`|评分输出保留 thinking|关闭|
|`--strip-thinking`|支持的任务中剥离 thinking 后评分|关闭|

旧参数 `--drop-pct-threshold` 不在新入口中接受，以免把旧 confidence-drop 实验和固定 margin-risk 实验混用。

### 5.3 GPU、端口、路径与结果

|参数|含义|默认|
|:---:|:---:|:---:|
|`--gpu-device ID/auto`|指定单张物理 GPU 或自动选择|auto|
|`--gpu-devices ID`|兼容别名，只允许一个 ID|auto|
|`--gpu-min-free-gb V`|auto 要求的最低空闲显存|24|
|`--gpu-candidates LIST`|auto 只考虑指定 GPU|全部|
|`--gpu-wait-seconds N`|等待满足条件 GPU 的秒数|0|
|`--gpu-memory-reserve-gb V`|模型加载前真实占用的显存|0|
|`--dtype DTYPE`|bfloat16/float16/float32 及别名|bfloat16|
|`--port N`|0 为 OS 原子分配，也可显式指定|0|
|`--output-path DIR`|时间戳结果根目录|`/data/home/wly/dLLM/NLD_results`|
|`--baseline-block16-dir DIR`|报告使用的 block-16 baseline|20260804 全量结果|
|`--baseline-block32-dir DIR`|报告使用的 block-32 baseline|20260804 全量结果|
|`--pytorch-python PATH`|模型 server Python|`nld_sglang` Python|
|`--eval-python PATH`|NeMo-Skills Python|同 PyTorch Python|
|`--nemo-skills-data-dir DIR`|持久数据和 cache 根目录|`/data1/linyewei/datasets/NLD`|
|`--google-research-dir DIR`|IFEval google-research checkout|`<data-dir>/google-research`|
|`--keep-runtime`|保留隐藏工作目录和日志|关闭|
|`--dry-run`|只解析打印，不创建目录/加载模型|关闭|

### 5.4 Judge-based benchmark 参数

|参数|含义|默认|
|:---:|:---:|:---:|
|`--judge-model NAME`|覆盖数据集默认 judge|数据集默认|
|`--judge-server-address URL`|judge OpenAI-compatible 地址|OpenAI 默认地址|
|`--judge-server-type TYPE`|judge server 类型|openai-compatible|
|`--judge-concurrency N`|judge 并发|4|
|`--mt-bench-max-tokens N`|MT-Bench 每轮候选预算|1024|
|`--alpaca-eval-max-tokens N`|AlpacaEval 候选预算|2048|
|`--skip-judge-api-key-check`|跳过入口的 API key 预检|关闭|

## 6. 五状态与下一轮指标

设 `p` 为预测位置，`q` 为当前 verifier 从左到右首个不通过位置；整块通过时 q 不存在。每次实际 overlap 尝试只属于以下一类：

|状态字段|中文含义|例子|
|:---:|:---:|:---:|
|`before_candidate_error`|预测位置之前出错，q<p|p=5、q=3|
|`candidate_fixed_by_alternative`|q=p 且 B 是 verifier 正确 token|p=q=5，B 命中|
|`candidate_wrong_alternative`|q=p 但 B 仍不正确|p=q=5，verifier 输出 C|
|`after_candidate_error`|q>p，预测位置 A 实际正确|p=5、q=8|
|`full_block_bonus`|所有 L-1 个 draft token 通过并产生 bonus|q 不存在|

五类的 `count` 之和必须等于 `prefetch_attempts`，`outcome_partition_valid` 自动检查该不变量。

每类还记录：

这里的 verify 接收长度按“从左向右连续匹配的 draft token 数 + verifier 本轮产出的 1 个 token”计算；整块通过时等于 L，并包含 bonus。

|字段|含义|例子|
|:---:|:---:|:---:|
|`current_accept_avg`|该状态当前 verify 的平均接收长度|平均 4.2 token|
|`next_count`|同一 request 确实存在下一 verify 的配对数|10 次中有 8 次进入下一轮|
|`next_coverage`|`next_count/count`|8/10=80%|
|`paired_current_accept_avg`|仅有下一轮的配对样本当前均值|配对当前平均 4|
|`next_accept_avg`|配对样本下一 verify 平均接收长度|下一轮平均 5.5|
|`next_minus_current_avg`|逐对“下一轮−当前轮”后平均|平均 +1.5|

EOS、最大 token 或 context 终止造成“没有下一轮”时，不把下一轮伪造为 0；它只会降低 `next_coverage`。这些跨轮值是同一请求的相邻状态描述，不单独构成 overlap 的因果收益。

## 7. 报告表与等权口径

`report.md` 包含：

1. 新方法、block-16 baseline、block-32 baseline 的配置和来源核验；
2. 每个非 AIME24 数据集的请求覆盖率、TPF、NFE、参考 TPS 对比；
3. 候选发现、实际尝试、B 命中、可复用、真实复用和跳过原因漏斗；
4. 五状态的次数、占实际尝试比例和分区校验；
5. 五状态当前轮与下一轮 verify 接收长度、配对覆盖率和变化量；
6. 九数据集总计和九数据集等权宏平均。

等权宏平均先在每个数据集内部算比例或均值，再对九个数据集做算术平均。绝对事件数另列总计，并同时给出每数据集平均次数。不会把所有 request/round 直接拼接后求一个 micro 指标，因此 MMLU 不会因样本多而盖过 AIME25 等小数据集。运行未结束时宏平均会标为 `已完成数/9`，最终应为 `9/9`。

报告不展示任何 accuracy 列。NeMo 即使生成了评分文件，本报告也不使用；scorer 没完成时，metrics 由成功 request stats 补建。效率表中只有 `OK` 请求参与均值，`Fail/OOM` 只用于覆盖率披露。

## 8. 解码与计算指标

|字段|含义|
|:---:|:---:|
|`physical_nfe` / `forward_passes`|真实 encoder 调用次数；两行 fused batch 仍计一次|
|`tokens_per_forward_pass` / `tpf`|返回 completion token 总数除以 physical NFE|
|`average_forward_passes_per_sample` / `average_nfe`|每请求平均 physical NFE|
|`model_output_tokens_per_s` / `tps`|completion token 除以 CUDA 同步的模型生成时间|
|`processed_rows`|全部 forward 处理的 batch row 总数|
|`processed_query_tokens`|全部 forward 的 batch×query length 总和|
|`normal_draft_forwards`|普通 dLLM draft 次数|
|`normal_verify_forwards`|无可执行候选时的单行 causal verify 次数|
|`fused_verify_draft_forwards`|双行 verify+prospective forward 次数|
|`prefetch_attempts`|实际发起融合 prospective 的次数|
|`prefetch_verified_hits`|q=p 且 B 正确的次数|
|`prefetch_hits`|B 命中且 prospective 可保存给下一轮的次数|
|`prefetch_saved_draft_forwards`|下一轮真实消费 prospective 的次数|
|`average_candidate_position`|候选位置 p 的均值，0 是 seed|

TPF 不能完全代表计算量：fused forward 的 batch 和 query length 更大，因此还应结合 processed rows/query tokens 与参考 TPS。当前用户要求的正式结果报告保留这些指标，但不把显存或端到端时间作为本实验主要结论。

## 9. 自检方法

静态与单元测试命令：

```bash
bash -n method/margin_risk_overlap_linearspec/eval_margin_risk_overlap.sh method/margin_risk_overlap_linearspec/run_pipeline.sh
```

```bash
python -m py_compile method/margin_risk_overlap_linearspec/*.py method/margin_risk_overlap_linearspec/tests/*.py
```

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python -m unittest discover -s method/margin_risk_overlap_linearspec/tests -v
```

测试覆盖 MASK 排除、margin-risk 定义、严格阈值、最左命中、`p=1`、第二候选排除、一位 shift、五状态互斥完备、跨轮配对、无下一轮不补零、状态聚合、初始报告模板、hybrid attention、KV cache 和分段 LoRA。

新规则可首次进入 `p=1`，需要真实模型验证该融合形状。下面的可选命令在调用前用 `CUDA_VISIBLE_DEVICES` 指定一张物理 GPU；它比较 `p=1` fused verifier 与普通 causal verifier 的 greedy token，并检查 canonical cache 不变及 prospective seed 为 B：

```bash
CUDA_VISIBLE_DEVICES=3 /data/home/wly/.conda/envs/nld_sglang/bin/python method/margin_risk_overlap_linearspec/tests/smoke_fused_p1.py --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --block-size 16 --dtype bfloat16
```
