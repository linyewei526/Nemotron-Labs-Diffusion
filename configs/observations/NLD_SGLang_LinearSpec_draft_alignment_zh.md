# NLD SGLang LinearSpec draft token 与最终输出对齐实验

> 入口：`observations/eval_linearspec_draft_alignment.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results/`

本文档说明新实验入口 `observations/eval_linearspec_draft_alignment.sh` 的用途、统计口径、输出格式和常用命令。

## 1. 实验目的

本实验用于回答：

```text
LinearSpec 某一轮 draft round 中，如果第一处 verify 失败的位置可以被纠正，那么这个错误之后的其他 draft token 是否仍然可能和最终解码结果一致？
```

换言之，本实验不是只看连续前缀接收长度，而是把一轮 draft 生成的所有 candidate token 与该 request 最终完成后的同位置输出 token 对齐比较。

## 2. 统计口径

一轮 LinearSpec block 可理解为：

```text
[seed, draft_1, draft_2, ..., draft_N]
```

其中：

- `seed` 是已有正确 token，不是本轮 draft candidate。
- `draft_1 ... draft_N` 是本轮真正要分析的 draft candidate。
- 当 `block_size=32` 时，通常是 1 个 seed + 31 个 draft candidate。

本实验严格排除 seed，只统计 seed 后面的 draft candidate。

对于每个 draft candidate：

```text
aligned = draft_token_id == final_output_token_id_at_the_same_sequence_position
```

如果最终输出没有覆盖该位置，例如提前 EOS 或长度截断，则该 candidate 记为 `missing_final_token`，不进入对齐率分母。

## 3. 输出指标

每个 benchmark 会生成：

```text
summaries/draft_alignment_<benchmark>.json
```

核心字段：

- `alignment.mean_alignment_count`：对所有可比较 round，逐 round 的 aligned draft candidate 数取平均。
- `alignment.mean_alignment_rate`：对所有可比较 round，逐 round 的 `aligned_count / compared_count` 取平均。
- `alignment.micro_alignment_rate`：全数据集所有 aligned candidate 总数除以所有 compared candidate 总数。

`block_position_alignment` 记录 block 内相对位置的平均对齐率：

```text
position_1 = seed 后第 1 个 draft candidate
position_2 = seed 后第 2 个 draft candidate
...
```

每个位置包含：

- `aligned_count`：该相对位置上 draft token 等于最终同位置 token 的次数。
- `total_count`：该相对位置上可比较次数。
- `alignment_rate`：`aligned_count / total_count`。

`post_rejection_offset_alignment` 记录从第一处 verify rejected token 后开始的 offset 对齐率：

```text
offset_1 = rejected 位置之后第 1 个 draft candidate
offset_2 = rejected 位置之后第 2 个 draft candidate
...
```

注意：`offset_0` 是 rejected 位置本身。按照 LinearSpec verify 定义，该位置 draft token 已经被证明和 AR verify token 不一致，因此本实验不记录 `offset_0`，只从 `offset_1` 开始记录。

每个 offset 同样包含：

- `aligned_count`
- `total_count`
- `alignment_rate`

## 4. 不影响旧实验的保证

新实验只在 LinearSpec YAML 中存在字段 `draft_alignment_trace_file` 时启用。

普通 `observations/eval_sglang.sh` 不写这个字段，因此不会：

- 记录 draft candidate token；
- 记录 request final output；
- 写 draft alignment trace；
- 影响复现论文 accuracy/TPF 实验效率。

旧 `observations/eval_linearspec_confidence.sh` 使用 `confidence_trace_file`，不会启用本实验。

旧 `observations/eval_linearspec_low_confidence_rejection.sh` 使用 `low_confidence_trace_file`，不会启用本实验。

新入口 `observations/eval_linearspec_draft_alignment.sh` 会显式设置：

```text
SGLANG_DRAFT_ALIGNMENT_TRACE_FILE=<本轮trace路径>
```

并显式清空：

```text
SGLANG_CONFIDENCE_TRACE_FILE=
SGLANG_LOW_CONFIDENCE_TRACE_FILE=
```

避免三套诊断实验混在同一次运行里。

此外，新入口会在 SGLang server 健康检查、timing proxy 健康检查和数据准备完成后，进入 NeMo-Skills 正式评测前清空本 benchmark 的 draft-alignment trace 文件。这样 server 启动预热或 `HEALTH_CHECK_*` 请求不会混入最终 summary。

## 5. 输出目录

默认输出根目录：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

每次运行生成：

```text
sglang_linearspec_draft_alignment_results/linearspec_draft_alignment_YYYYMMDD_HHMMSS/
```

目录结构：

```text
Settings.json
benchmark_status.jsonl
traces/raw_draft_alignment_trace_<benchmark>.jsonl
summaries/draft_alignment_<benchmark>.json
eval_runs/<benchmark>/eval_YYYYMMDD_HHMMSS/metrics_<benchmark>.json
```

迁移后第一次运行时，可先做无副作用路径自检；该命令不创建结果目录、不启动模型：

```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks gsm8k:1 --gpu-devices 0 --block-size 16 --dry-run
```

## 6. 最常用 smoke test

单卡、1 条 GSM8K 样本、block size 32：

```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

单卡、1 条 GPQA 和 1 条 IFEval 样本：

```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks gpqa:1,ifeval:1 --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

## 7. 全量 benchmark 示例

全量 10 个 benchmark：
 
```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 1 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 40 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

只跑 MMLU 和 IFEval：

```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks mmlu:1,ifeval:1 --gpu-devices 1 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 40 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

## 8. 端口冲突时

指定 SGLang server 端口和 timing proxy 端口：

```bash
bash observations/eval_linearspec_draft_alignment.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --port 30020 --proxy-port 31020 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results
```

## 9. 常用参数解释

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
- `--dry-run`：校验迁移后的顶层入口、`observations/eval_sglang.sh`、汇总脚本和输出路径，只打印解析结果，不创建目录、不启动模型。

## 10. 离线重新汇总新 trace

如果已经有新实验生成的 `raw_draft_alignment_trace_<benchmark>.jsonl`，可以不重跑模型，只重新汇总：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python /data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/sglang_eval/summarize_linearspec_draft_alignment.py --trace-file /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results/linearspec_draft_alignment_YYYYMMDD_HHMMSS/traces/raw_draft_alignment_trace_gsm8k.jsonl --output-json /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_draft_alignment_results/linearspec_draft_alignment_YYYYMMDD_HHMMSS/summaries/draft_alignment_gsm8k.json --benchmark gsm8k --benchmark-spec gsm8k:1
```

注意：这个离线命令只适用于新实验的 `raw_draft_alignment_trace_*.jsonl`。
