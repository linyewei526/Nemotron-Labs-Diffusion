# NLD PyTorch + NeMo-Skills：固定 margin-risk 条件式 rank 候选 overlap

> 实验入口：`method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh`  
> 独立代码：`method/margin_risk_conditional_rank_overlap_linearspec/`  
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results/`  
> 正式配置：greedy、LinearSpec LoRA、block size 16、draft threshold 0、固定 `margin_risk_threshold=0.5`  
> 主报告：排除 AIME24 的九数据集等权平均；MMLU 默认最后运行  
> 默认完成标准：只要求效率统计可用，不要求 accuracy/scorer 成功

## 1. 目标与隔离边界

本实验在既有 `margin_risk_multi_overlap_linearspec` 思路上实现一套新的、完全独立的候选分配策略：风险位置较少时保留 continuation，风险位置较多时把有限的三个 speculative row 集中分配给 P1 的第二、第三置信 token 和 P2 的第二置信 token。

它不修改或导入下列实验目录：

- 原始 PyTorch/NeMo-Skills 论文复现；
- `method/confidence_overlap_linearspec/`；
- `method/margin_risk_overlap_linearspec/`；
- `method/margin_risk_multi_overlap_linearspec/`；
- `method/margin_risk_two_plus_new_overlap_linearspec/`；
- 既有 observations 及结果。

本目录自带 generation、hybrid attention、分段 LoRA、OpenAI-compatible server、NeMo-Skills pipeline、指标合并、增量报告和测试。并行安全措施如下：

- 默认 `--port 0`，由操作系统原子绑定空闲端口；
- 每次运行建立独立时间戳目录，同一秒重名时追加 `_01`、`_02`；
- 隐藏工作目录带 PID，server、日志、request stats 和显存预留进程均只属于本次运行；
- 退出 trap 只终止本次入口启动的进程；
- 新方法使用独立 backend 名、模型服务名、NeMo expname 和 metrics key。

运行开始时先创建 `Settings.json` 和 `report.md` 模板。每完成一个数据集就原子更新报告，不必等九个数据集全部完成。

```text
/data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results/
└── margin_risk_conditional_rank_overlap_YYYYMMDD_HHMMSS/
    ├── Settings.json
    ├── report.md
    ├── metrics_<dataset>.json
    └── artifacts/<dataset>/
        ├── output-rs0.jsonl
        ├── pytorch_request_stats.jsonl
        ├── pytorch_margin_risk_conditional_rank_overlap_metrics_summary.json
        └── pytorch_benchmark.log
```

## 2. 策略精确定义

### 2.1 风险位置

对当前 draft block 的位置 1 到 L-1 从左向右扫描；位置 0 是已知 seed，不参与定位。先从概率归一化中排除 MASK，然后计算：

```text
margin = 最高概率 - 第二高概率
margin_risk = 1 - margin
```

记录最先出现的三个严格满足下式的位置 P1、P2、P3，同时继续扫描到 block 末尾以获得真实 crossing 总数：

```text
margin_risk > 0.5
```

等于 0.5 不触发。例如最高、第二高概率为 0.65、0.25 时，margin 为 0.40，margin_risk 为 0.60，因此触发。

### 2.2 crossing 不超过 2 个

若整个 draft 中 crossing 总数为 0、1 或 2：

- 对已有的 P1/P2，各取原 draft token 之外的有效第二置信 token，构造纠错 prospective row；
- 无论是否存在 P1/P2，都额外构造一条 continuation，即 `draft sequence new`；
- 若当前 draft 整块通过，只有 verifier bonus token 等于 new 的第一个 token 时，new 才可作为下一轮 draft 复用。

候选位置 p 的 row 为：

```text
当前 draft 的 0..p-1 causal prefix + 替代 token + L-1 个 MASK
```

continuation row 为：

```text
当前完整 L-token draft causal prefix + L 个 MASK
```

### 2.3 crossing 达到 3 个或更多

若整个 draft 中至少存在 P1、P2、P3，则三个 speculative row 固定分配为：

1. P1 的第二置信 token；
2. P1 的第三置信 token；
3. P2 的第二置信 token。

此时 P3 只作为“第三个风险位置/首错定位是否命中”的观测信号，不为 P3 构造候选，也不构造 continuation。若 verifier 首错为 P1，先检查 P1 第二置信分支，再检查第三置信分支；只有被 verifier 证明 token 正确的分支才能复用。若首错为 P2，则只检查 P2 第二置信分支。

这里的第二、第三置信 token 都在排除原 draft token、MASK 和 EOS 后按 logit 排序；原 draft token视为第一置信 token。正常 draft 的扫描位置均来自 argmax，因此这对应所需的 2nd/3rd-confidence 候选。

### 2.4 行数与 padding

策略仍采用原始 padded 组 batch，不做展平，最大为 verifier 加三条 speculative row：

|crossing 数|speculative row|通常总行数|
|:---:|:---:|:---:|
|0|new|2|
|1|P1二选 + new|3|
|2|P1二选 + P2二选 + new|4|
|3 及以上|P1二选 + P1三选 + P2二选|4|

若候选受剩余 token、context 或 thinking 边界限制而不可执行，实际行数可以更少。所有 row pad 到公共 query length Q；attention mask 不会省掉 dense QKV/MLP 对 padding token 的计算，因此报告同时记录：

```text
每次 forward 实际计算 token = row 数 × 公共 Q
padding token = 实际计算 token - 各 row 有效长度之和
```

只有 verifier row 能提交输出与 canonical KV。所有 prospective row 都只是推测；未被当前 verifier 证明的分支一律丢弃。

每条并行生成的纠错 prospective block 和 continuation block 都会在生成后立刻用同一套 `margin_risk>0.5` 规则分析并保存下一层 P1/P2/P3 与 rank 候选。如果本轮 verifier 证明某条分支可复用，它成为下一轮当前 draft 时不需要重新起草，并且已经带有下一轮融合 verify 所需的候选描述。因此连续命中时每一步仍沿用最多 4 row 的递归 overlap，而不是只优化一轮后退回普通逻辑。

## 3. 状态与统计口径

每个实际 fused 轮严格落入 15 个互斥状态之一：

|状态|含义|例子|
|:---:|:---:|:---:|
|`miss_no_risk_position`|没有风险位置但当前仍发生首错|只有 new，首错在位置 4|
|`miss_before_first`|首错在 P1 之前|P1=5，首错=3|
|`miss_between_positions`|首错位于相邻风险位置之间|P1=3、P2=7，首错=5|
|`miss_after_last`|首错在最后一个已记录风险位置之后|P3=8，首错=11|
|`p1_rank2_fixed`|首错=P1，第二置信 token 修正正确|P1 二选命中|
|`p1_rank3_fixed`|首错=P1，第二置信错误但第三置信正确|P1 三选命中|
|`p1_all_candidates_wrong`|首错=P1，本轮为 P1 构造的一条或两条候选均错|正确 token 不在所建候选中|
|`p1_no_executable_candidate`|首错=P1，但 P1 候选受边界限制未执行|剩余预算不足|
|`p2_rank2_fixed`|首错=P2，第二置信 token 正确|P2 二选命中|
|`p2_rank2_wrong`|首错=P2，第二置信 token 错误|正确 token 不是 P2 二选|
|`p2_no_executable_candidate`|首错=P2，但 P2 候选未执行|context 不足|
|`p3_detected_no_candidate`|首错=P3，但策略有意不为 P3 分配 row|成功定位、没有纠错分支|
|`full_continuation_hit`|整块通过且 bonus=new[0]|new 可复用|
|`full_continuation_miss`|整块通过但 bonus 不等于 new[0]|new 丢弃|
|`full_continuation_absent`|整块通过但该轮没有 new|crossing 至少 3 个|

`outcome_partition_valid` 自动检查 15 个状态次数之和是否等于实际 fused 轮数。除此之外，报告单独统计 P1二选、P1三选、P2二选在“首错恰在相应位置”时各自被检查、修正正确、修正错误的次数和条件正确率。这样 P1 的第二与第三置信候选不会被一个互斥状态合并后丢失信息。

每个状态还记录：本轮 verify 平均接收 token、同一 request 确实存在的下一轮 verify 平均接收 token、配对样本的下一轮减当前轮均值。终止轮没有下一轮时不补成 0，只降低 `NextCov`。

TPF 使用更新后的 decode-only 口径：

```text
TPF = 返回的 completion token / decode 阶段物理 encoder forward 次数
```

prompt prefill 不进入 TPF，与当前 SGLang 口径一致；含 prefill 的总 NFE/端到端 TPF仍保留用于审计。一个四行 fused forward 在 TPF 中仍计一次物理 forward，所以必须和 `FwdTok均`、`Rows均`、`Q均`、`Pad率` 一起解释。

默认 B16/B32 是修改 TPF 口径前产生的不可变历史产物，其 `forward_passes` 含每个成功 request 的一次 prompt prefill。报告只在读取时按“旧 forward 总数减成功 request 数”重算 baseline 的 decode NFE 与 TPF，不改写原结果；若以后指定的 baseline 已带 `decode_forward_passes`，则直接采用其显式 decode 字段。

## 4. 推荐单行命令

以下每条命令都是可直接复制的一行。

### 4.1 查看帮助

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --help
```

### 4.2 只检查参数与路径

不创建结果目录、不加载模型、不启动 server。

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --context-length 2112 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --gpu-device 0 --gpu-memory-reserve-gb 0 --dry-run
```

### 4.3 单题全链路 smoke

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --context-length 2112 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --efficiency-only --keep-runtime
```

### 4.4 正式九数据集全量效率实验

该命令排除 AIME24，MMLU 最后运行。每完成一个数据集就更新一次 `report.md`，最终主表为九数据集等权平均。

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device 2 --gpu-memory-reserve-gb 30 --require-accuracy --output-path /data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results
```

### 4.5 常用十数据集，报告仍排除 AIME24

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --efficiency-only --output-path /data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results
```

### 4.6 单数据集与自定义多数据集

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2 --efficiency-only
```

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1,aime25:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2 --efficiency-only
```

`human-eval:1` 和 `mbpp:1` 中的 `:1` 表示 pass@1，不表示只取一个 sample。`--max-samples` 才是样本上限。严格 accuracy 模式下 EvalPlus scorer 通常要求完整题集；默认效率模式允许子集生成，并可在 scorer 不接受子集时从成功 request stats 创建效率 metrics。

### 4.7 自动选 GPU、限定候选与等待

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device auto --gpu-candidates 1,3,5 --gpu-min-free-gb 28 --gpu-wait-seconds 1800 --efficiency-only
```

后端是单 GPU；`--gpu-devices` 只是兼容别名，也只接受一个 ID。

### 4.8 预留显存

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 3 --gpu-memory-reserve-gb 20 --efficiency-only
```

`--gpu-memory-reserve-gb` 会在模型加载前额外真实占用指定显存，用于模拟受限环境；它不是替模型保留可用空间。最多四行的本方法通常比单行 baseline 需要更多峰值显存。

### 4.9 恢复严格 accuracy 行为

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2 --require-accuracy
```

默认已经是 efficiency-only；显式写 `--efficiency-only` 不改变默认值。传 `--require-accuracy` 后，request OOM 或 scorer 失败会令相应数据集失败，并要求 accuracy 正常完成。

### 4.10 显式端口、结果目录与 baseline

```bash
bash method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 2 --tokens 512 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 1 --port 19081 --output-path /data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935 --keep-runtime --efficiency-only
```

并行运行应优先保留 `--port 0`；显式端口已被占用时原子 bind 会直接失败，不会误连到其他实验。

### 4.11 只重建已有结果报告

```bash
python method/margin_risk_conditional_rank_overlap_linearspec/report.py --result-dir /data/home/wly/dLLM/NLD_results/margin_risk_conditional_rank_overlap_results/margin_risk_conditional_rank_overlap_YYYYMMDD_HHMMSS --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935
```

该命令只读取现有 Settings/metrics，并原子重写 `report.md`，不启动模型。

## 5. 参数详解

### 5.1 模式、数据和采样

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--mode overlap_lora`|draft 与 prospective suffix 使用 LinearSpec LoRA|正式模式|
|`--mode overlap_base`|不加载 LoRA 的消融模式|可选|
|`--model PATH`|模型 checkpoint|本地 8B checkpoint|
|`--served-model-name NAME`|本地 OpenAI API 标签|本方法专用标签|
|`--lora-path DIR`|LinearSpec LoRA|`<model>/linear_spec_lora`|
|`--benchmarks LIST`|逗号分隔，支持单/多数据集|常用十项；MMLU 最后|
|`--tokens N`|每 request 最大 completion token|8192|
|`--max-samples N`|每数据集最多取前 N 个 sample|默认全量|
|`--quick-test`|NeMo-Skills quick test|默认关闭|
|`--num-chunks N`|客户端 chunk 数|等于并发数|
|`--client-concurrency N`|HTTP 请求并发；模型执行仍串行|1|
|`--math-prompt-config NAME`|数学数据 prompt config 覆盖|空|
|`--efficiency-only`|OOM 跳过且只要求效率统计|默认启用|
|`--require-accuracy`|要求 scorer/accuracy 完整成功|默认关闭|

### 5.2 解码与 thinking

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--block-length N` / `--block-size N`|draft、verify、prospective block 长度 L|16；至少 2|
|`--threshold V`|LinearSpec draft unmask threshold|必须为 0|
|`--margin-risk-threshold V`|strict crossing 阈值|0.5；范围 0 到 1|
|`--temperature V`|采样温度|必须为 0|
|`--top-p V`|协议对齐并记录；原生方法不应用|0.95|
|`--context-length N`|prompt 加生成预算上限|未显式给出时为所需生成预算+2048|
|`--enable-thinking`|chat template 开启 thinking|关闭|
|`--disable-thinking`|显式告诉 NeMo-Skills 关闭 thinking|关闭|
|`--max-thinking-tokens N`|超预算后强制结束 thinking|空|
|`--keep-thinking`|评分输出保留 thinking|关闭|
|`--strip-thinking`|支持的任务剥离 thinking 后评分|关闭|

候选分配、最多 4 row、P3 只定位以及 crossing≤2 时追加 new 是本实验固定定义，不提供命令行开关。正式复现应保持 `--margin-risk-threshold 0.5`。

### 5.3 GPU、端口和路径

|参数|含义|默认|
|:---:|:---:|:---:|
|`--gpu-device ID/auto`|指定物理 GPU；auto 先满足空闲显存，再优先低利用率|auto|
|`--gpu-devices ID`|兼容别名，只允许单 ID|auto|
|`--gpu-min-free-gb V`|auto 选择要求的最低空闲显存|24|
|`--gpu-candidates LIST`|auto 只在这些 GPU 中选择|全部|
|`--gpu-wait-seconds N`|等待满足条件 GPU 的秒数|0|
|`--gpu-memory-reserve-gb V`|模型前额外真实占用显存|0|
|`--dtype DTYPE`|bfloat16、float16、float32 及别名|bfloat16|
|`--port N`|本地 server 端口；0 为原子自动选择|0|
|`--output-path DIR`|时间戳运行目录的根路径|本方法专用 results 根|
|`--baseline-block16-dir DIR`|报告对照 B16 结果|既有 PyTorch 全量结果|
|`--baseline-block32-dir DIR`|报告对照 B32 结果|既有 PyTorch 全量结果|
|`--pytorch-python PATH`|模型 server Python|`nld_sglang` 环境 Python|
|`--eval-python PATH`|NeMo-Skills Python|同 PyTorch Python|
|`--nemo-skills-data-dir DIR`|持久数据/cache 根目录|本地 NLD 数据目录|
|`--google-research-dir DIR`|IFEval scorer checkout|`<data-dir>/google-research`|
|`--keep-runtime`|保留隐藏工作目录和调试信息|关闭|
|`--dry-run`|只解析和打印，不写文件、不载入模型|关闭|

### 5.4 Judge 数据集

|参数|含义|默认|
|:---:|:---:|:---:|
|`--judge-model NAME`|覆盖数据集默认 judge|数据集默认|
|`--judge-server-address URL`|OpenAI-compatible judge 地址|默认 OpenAI 地址|
|`--judge-server-type TYPE`|judge server 类型|openai-compatible|
|`--judge-concurrency N`|judge 并发|4|
|`--mt-bench-max-tokens N`|MT-Bench 每轮候选预算|1024|
|`--alpaca-eval-max-tokens N`|AlpacaEval 候选预算|2048|
|`--skip-judge-api-key-check`|跳过入口 API key 预检|关闭|

## 6. 增量报告内容与九数据集等权

`report.md` 包含：

1. 配置以及与 PyTorch+NeMo-Skills、greedy、block size 16/32 baseline 的一致性核验；
2. 每数据集请求覆盖、decode-only TPF/NFE/TPS 及九数据集等权平均；
3. crossing 数、少风险/多风险模式、2/3/4 row、三类候选、新分支命中/复用漏斗；
4. 漏报、预测后首错、P1第二/第三置信修正、P2第二置信修正、P3只定位、new 命中的次数和占比；
5. P1二选、P1三选、P2二选的独立检查数、正确/错误数和条件正确率；
6. 全部/解码/融合 forward 的 dense token 均值、Min、P50、P90、P95、P99、Max、有效 token、padding、row 与公共 Q；
7. 全部 15 个互斥状态的本轮接收、下一轮接收和配对差值；
8. 分区完整性、失败覆盖和口径说明。

涉及数据集的比例、均值和分位数都先在每个非 AIME24 数据集内部计算，再对数据集取算术平均。无论 MMLU 样本数多大，它都只有一个数据集权重。绝对次数另列总计，不作为全局比例分母。某条件在某数据集从未发生时，条件均值未定义，不用 0 冒充观测；报告会显示破折号。

准确率即使存在也不进入本报告。默认 efficiency-only 下，单 request CUDA OOM 会写入失败记录并返回空占位，后续 request 继续；失败 request 不进入效率均值，`Att/OK/Fail/OOM/Cov` 显式披露有效覆盖率。若需要完整 accuracy，使用 `--require-accuracy`。

## 7. 自检和真实模型 smoke

静态检查：

```bash
bash -n method/margin_risk_conditional_rank_overlap_linearspec/eval_margin_risk_conditional_rank_overlap.sh method/margin_risk_conditional_rank_overlap_linearspec/run_pipeline.sh
```

```bash
python -m py_compile method/margin_risk_conditional_rank_overlap_linearspec/*.py method/margin_risk_conditional_rank_overlap_linearspec/tests/*.py
```

离线单元测试：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python -m unittest discover -s method/margin_risk_conditional_rank_overlap_linearspec/tests -v
```

真实模型融合 smoke 分别验证“P1二选+P1三选+P2二选”和“P1二选+P2二选+new”两个 4-row 路径；它比较 fused verifier 与普通 causal verifier 的 greedy token，检查每条 prospective seed、continuation、padding 计数和 canonical KV 不变。

```bash
CUDA_VISIBLE_DEVICES=3 /data/home/wly/.conda/envs/nld_sglang/bin/python method/margin_risk_conditional_rank_overlap_linearspec/tests/smoke_fused_multi.py --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --block-size 16 --dtype bfloat16
```
