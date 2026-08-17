# NLD SGLang LinearSpec 低置信度 token 是否对应拒绝位置实验

> 入口：`observations/eval_linearspec_low_confidence_rejection.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results/`

本文档说明新实验入口 `observations/eval_linearspec_low_confidence_rejection.sh` 的用途、统计口径、输出格式和常用命令。

## 1. 实验目的

此前 `sglang_linearspec_confidence` 实验验证了 rejected token 的 draft confidence 往往偏低。本实验反过来验证：

- 对于某一轮 LinearSpec draft round 中相对前缀均值明显偏低的 token，它是否就是 verify round 中被拒绝的位置。
- 对于这些低置信度 token，有多少实际被 verify 接收，有多少实际被 verify 拒绝。

输出会对每个 benchmark 分别统计所有阈值：

- `token_x_drop_abs`：`C_imean - C_i >= x`。
- `token_y_drop_pct`：`1 - C_i / C_imean >= y`。

默认阈值完全按当前需求：

- `x = 0.300, 0.305, 0.310, ..., 0.400`，包含 0.300 和 0.400。
- `y = 0.40, 0.41, 0.42, ..., 0.60`，包含 0.40 和 0.60。

## 2. 统计口径

LinearSpec 每轮 block 中第一个位置是 seed token，不是 draft 生成候选，因此本实验的 candidate index 从 seed 后第一个 draft candidate 开始。

对于同一轮 draft candidate 序列：

- 第 `i` 个 candidate 的 draft confidence 记为 `C_i`。
- `C_imean = mean(C_0, C_1, ..., C_{i-1})`。
- `drop_abs = C_imean - C_i`。
- `drop_pct = 1 - C_i / C_imean`。
- `i = 0` 没有前缀均值，不进入阈值统计。

每个 candidate 的 verify outcome 只有以下几类：

- `accepted`：该 draft token 位于连续验证通过前缀中。
- `rejected`：该 draft token 是本轮第一处验证失败的位置。
- `unverified_after_rejection`：该 token 在第一处 rejected token 之后，不进入接收/拒绝统计。
- `unverified_after_eos`：EOS 截断后不再关心的位置，不进入接收/拒绝统计。

最终每个阈值只统计 `accepted` 和 `rejected` 两类。

## 3. 与旧实验的关系

旧实验目录如：

- `/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_20260628_030905`
- `/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_20260628_125539`

其中 `traces/raw_trace_*.jsonl` 只记录了：

- accepted draft prefix 的 confidence。
- 第一处 rejected draft token 的 confidence。

它没有记录一轮 draft round 中所有 candidate token 的 confidence，因此不能严格完成本实验。新实验会写新的 trace 文件：

```text
traces/raw_low_confidence_trace_<benchmark>.jsonl
```

## 4. 不影响旧实验的保证

新实验只在 LinearSpec YAML 中存在字段 `low_confidence_trace_file` 时启用。

普通 `observations/eval_sglang.sh` 不写这个字段，因此不会：

- clone draft logits；
- 计算 per-token softmax confidence；
- 写低置信度 trace；
- 影响复现论文 accuracy/TPF 实验效率。

旧 `observations/eval_linearspec_confidence.sh` 使用的是 `confidence_trace_file`，仍然走旧 tracer，不会自动启用本实验。

新入口 `observations/eval_linearspec_low_confidence_rejection.sh` 会显式设置：

```text
SGLANG_LOW_CONFIDENCE_TRACE_FILE=<本轮trace路径>
```

并显式清空：

```text
SGLANG_CONFIDENCE_TRACE_FILE=
```

避免两套诊断实验混在同一次运行里。

## 5. 输出目录

默认输出根目录：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

每次运行生成：

```text
sglang_linearspec_low_confidence_results/linearspec_low_confidence_YYYYMMDD_HHMMSS/
```

目录结构：

```text
Settings.json
benchmark_status.jsonl
traces/raw_low_confidence_trace_<benchmark>.jsonl
summaries/low_confidence_rejection_<benchmark>.json
eval_runs/<benchmark>/eval_YYYYMMDD_HHMMSS/metrics_<benchmark>.json
```

其中最重要的是：

```text
summaries/low_confidence_rejection_<benchmark>.json
```

## 6. Summary JSON 字段解释

`threshold_definition` 记录阈值定义和 confidence 定义。

`tokens.candidate_tokens_by_outcome` 记录所有 draft candidate 的 outcome 数量，包括 ignored outcome。

`tokens.countable_tokens_with_prefix` 是真正进入阈值判断的 token 总数，也就是：

```text
accepted/rejected 且不是 candidate index 0 且 confidence 有效
```

`tokens.countable_accepted_tokens_with_prefix` 是进入阈值判断的 accepted token 总数。

`tokens.countable_rejected_tokens_with_prefix` 是进入阈值判断的 rejected token 总数。

`tokens.skipped_no_prefix` 是 candidate index 0，因为没有 `C_imean`，无法计算 drop。

`token_x_drop_abs` 中每个键类似：

```text
token_0.300_drop_abs
```

表示 `C_imean - C_i >= 0.300` 的 token。

`token_y_drop_pct` 中每个键类似：

```text
token_0.40_drop_pct
```

表示 `1 - C_i / C_imean >= 0.40` 的 token。

每个阈值项包含：

- `accepted_count`：满足该阈值且最终被 verify 接收的 token 数。
- `rejected_count`：满足该阈值且是第一处 verify rejected token 的 token 数。
- `accepted_ratio_within_flagged`：`accepted_count / (accepted_count + rejected_count)`，低置信度规则的误判比例。
- `rejected_ratio_within_flagged`：`rejected_count / (accepted_count + rejected_count)`，低置信度规则命中真实拒绝位置的比例。
- `accepted_coverage_of_all_countable_accepted_tokens`：命中的 accepted token 占全部可统计 accepted token 的比例。
- `rejected_coverage_of_all_countable_rejected_tokens`：命中的 rejected token 占全部可统计 rejected token 的比例。

迁移后第一次运行时，可先做无副作用路径自检；该命令不创建结果目录、不启动模型：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks gsm8k:1 --gpu-devices 0 --block-size 16 --dry-run
```

## 7. 最常用 smoke test

单卡、1 条 GSM8K 样本、block size 32：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

单卡、1 条 GPQA 和 1 条 IFEval 样本：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks gpqa:1,ifeval:1 --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

## 8. 全量 benchmark 示例

全量 10 个 benchmark：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

只跑 MMLU 和 IFEval：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks mmlu:1,ifeval:1 --gpu-devices 1 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 40 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

## 9. 自定义阈值

默认不需要传阈值参数。如果要显式指定和当前需求一致的阈值：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --abs-start 0.300 --abs-end 0.400 --abs-step 0.005 --pct-start 0.40 --pct-end 0.60 --pct-step 0.01 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

## 10. 端口冲突时

指定 SGLang server 端口和 timing proxy 端口：

```bash
bash observations/eval_linearspec_low_confidence_rejection.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --port 30020 --proxy-port 31020 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results
```

## 11. 常用参数解释

- `--benchmarks LIST`：benchmark 列表，多个用逗号连接。支持当前 NLD/NeMo-Skills 路线的 `human-eval:1`、`mbpp:1`、`livecodebench-cpp:1`、`gsm8k:1`、`math-500:1`、`aime24:1`、`aime25:1`、`gpqa:1`、`mmlu:1`、`ifeval:1`。
- `--gpu-devices LIST`：指定 GPU，例如 `3` 或 `0,1`。
- `--batch-size N`：传给 SGLang 的 `--max-running-requests`。
- `--client-concurrency N`：NeMo-Skills client 并发和 timing proxy 并发。
- `--gpu-memory-reserve-gb V`：在指定 GPU 上预占显存，避免其他任务误抢占。
- `--block-size N`：LinearSpec block size。
- `--max-samples N`：每个 benchmark 只跑 N 条，用于 smoke test；全量跑时不要传。
- `--tokens N`：最大生成长度，默认 8192。
- `--context-length N`：SGLang context length；不传时沿用 `observations/eval_sglang.sh` 的自动规则。
- `--output-path DIR`：新实验输出根目录。
- `--abs-start/--abs-end/--abs-step`：控制 `token_x_drop_abs` 阈值序列。
- `--pct-start/--pct-end/--pct-step`：控制 `token_y_drop_pct` 阈值序列。
- `--dry-run`：校验迁移后的顶层入口、`observations/eval_sglang.sh`、汇总脚本和输出路径，只打印解析结果，不创建目录、不启动模型。

## 12. 离线重新汇总新 trace

如果已经有新实验生成的 `raw_low_confidence_trace_<benchmark>.jsonl`，可以不重跑模型，只重新汇总：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python /data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/sglang_eval/summarize_linearspec_low_confidence_rejection.py --trace-file /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results/linearspec_low_confidence_YYYYMMDD_HHMMSS/traces/raw_low_confidence_trace_gsm8k.jsonl --output-json /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_low_confidence_results/linearspec_low_confidence_YYYYMMDD_HHMMSS/summaries/low_confidence_rejection_gsm8k.json --benchmark gsm8k --benchmark-spec gsm8k:1
```

注意：这个离线命令只适用于新实验的 `raw_low_confidence_trace_*.jsonl`，不适用于旧 confidence 实验的 `raw_trace_*.jsonl`。
