# NLD SGLang LinearSpec LoRA Confidence / Rank 诊断实验指南

> 入口：`observations/eval_linearspec_confidence.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/`

## 1. 这个新实验测什么

这个实验用于分析 `linearspec_lora` 每一轮 draft/verify 中的细粒度行为。它不是论文复现的效率实验，因为它会额外计算 softmax confidence 和 rank，会引入额外开销。论文复现的 accuracy / TPF / TPS 等结果仍然使用 `observations/eval_sglang.sh` 默认路径。

本实验只在显式运行新入口 `observations/eval_linearspec_confidence.sh` 时开启，不会影响默认 `observations/eval_sglang.sh` 复现路径。

## 2. 四类指标定义

所有 confidence 和 rank 都按 LinearSpec 实际 draft 逻辑计算：在 draft forward 后、任何 in-place MASK 屏蔽和 verify forward 前，先保存一份稳定的 draft logits；统计时排除 `MASK` token，再计算 softmax confidence 或排名。

`accepted_draft_confidence`：所有验证通过的 draft token 的 confidence 分布。这里不包含 seed token，因为 seed 来自上一轮 AR 验证，不是本轮 draft token。

`rejected_draft_confidence`：所有存在验证失败的轮中，第一个失败位置的 draft token confidence 分布。

`rejected_correct_token_rank`：所有存在验证失败的轮中，AR 验证给出的正确 token 在该位置 draft 分布里的排名。rank 按排除 `MASK` 后的 draft logits 计算，1 表示正确 token 本来就是 draft 分布第一名；在当前 argmax draft 逻辑下，失败位置一般不应出现 rank=1，除非存在完全并列 top logit 这类极少数情况。

`confidence_drop_abs` 和 `confidence_drop_pct`：如果某轮存在失败位置，且这一轮失败前至少有一个通过的 draft token，则计算失败位置 draft confidence 相比本轮通过 token confidence 均值低多少。绝对值为 `accepted_mean_conf - rejected_conf`，百分比为 `(accepted_mean_conf - rejected_conf) / accepted_mean_conf`。

如果某轮一开始就失败，也就是没有任何通过的 draft token，则第 4 类指标没有本轮均值作为参照，不会强行记录。

## 3. 新增代码路径

入口脚本：

`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/observations/eval_linearspec_confidence.sh`

LinearSpec 采集 helper：

`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/analysis/linearspec_confidence_trace.py`

分布汇总脚本：

`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/sglang_eval/summarize_linearspec_confidence_trace.py`

SGLang LinearSpec 默认逻辑仍在：

`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py`

新增采集通过 YAML 字段 `confidence_trace_file` 控制。默认 `observations/eval_sglang.sh` 不写这个字段，因此不启用采集。

## 4. 输出文件结构

默认输出根目录是：

`/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results`

每次运行会生成一个时间戳目录，例如：

`/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_20260627_203000`

目录内主要文件如下：

`Settings.json`：记录本次实验命令和核心参数。

`benchmark_status.jsonl`：每个 benchmark 的运行状态、trace 文件路径、summary 文件路径。

`traces/raw_trace_<benchmark>.jsonl`：逐轮原始 trace。这里保存完整分布来源，包含每轮通过 token confidence 数组、失败 token confidence、正确 token rank、drop 指标等。

`summaries/confidence_distribution_<benchmark>.json`：汇总后的分布文件。默认只提供 count、mean、std、min、max、quantiles、histogram，不重复保存每个指标的原始 values 数组，防止全量 benchmark summary 过大。rank 分布仍然保留 `exact_rank_counts`。如果确实需要在 summary 中额外保存原始 values，可以显式加 `--include-values`。

`eval_runs/<benchmark>/eval_*/`：该 benchmark 对应的底层 `observations/eval_sglang.sh` 输出，包括 `metrics_<benchmark>.json` 或 `error_<benchmark>.json`。

迁移后第一次运行时，可先做无副作用路径自检；该命令不创建结果目录、不启动模型：

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks gsm8k:1 --gpu-devices 0 --block-size 16 --dry-run
```

## 5. 最小 smoke test

下面命令只跑 GSM8K 1 条样本，用于检查新实验链路是否正常。注意这是 smoke test，不是全量分布。

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --max-samples 1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

## 6. 单 benchmark 全量运行

下面命令对 GSM8K 全量运行，记录完整 confidence / rank 分布。

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

如果希望 context 按输入最多 8192 + 输出最多 8192 的上界设置，可以显式加 `--context-length 16384`：

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --context-length 16384 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

## 7. 多 benchmark 全量运行

下面命令按命令行顺序逐个 benchmark 运行。每个 benchmark 会单独启动一次 SGLang server，并生成独立 trace 和 summary，避免不同 benchmark 的 trace 混在同一个文件里。

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks human-eval:1,mbpp:1,gsm8k:1,math-500:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

如果要对之前 10 个 benchmark 全量运行：

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
bash observations/eval_linearspec_confidence.sh --benchmarks mmlu:1,ifeval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

## 8. 当前复现实验仍在跑时怎么避免冲突

如果论文复现实验正在 GPU 2 上运行，并占用 SGLang 端口 `30000` 和 proxy 端口 `31000`，新实验建议这样做：

使用另一块 GPU，例如 `--gpu-devices 3`。

不要手动指定 `--port` 和 `--proxy-port`，脚本会发现默认端口被占用并自动选择空闲端口。

如果你必须固定端口，请选择当前未占用端口，例如：

```bash
bash observations/eval_linearspec_confidence.sh --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32 --port 30011 --proxy-port 31011 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results
```

## 9. 常用参数解释

`--benchmarks LIST`：要测的 benchmark 列表，逗号分隔。这里的 `:1` 表示每道题 repeat 1 次，不是只跑 1 条样本。全量运行也建议保留 `:1`。

`--gpu-devices LIST`：指定使用哪块或哪几块 GPU，例如 `3` 或 `0,1`。多卡时底层 `observations/eval_sglang.sh` 会默认按 GPU 数推断 tensor parallel size。

`--batch-size N`：传给 SGLang 的 `max-running-requests`，控制 server 同时运行的请求数。

`--client-concurrency N`：NeMo-Skills 客户端并发请求数，同时也是 timing proxy 的最大 in-flight 请求数。

`--gpu-memory-reserve-gb V`：在指定 GPU 上预先空占 V GiB 显存，防止其他进程误抢占显存。诊断实验一般可以继续用 `10`；如果你想更强地锁住 GPU，可以根据剩余显存调大。

`--block-size N`：LinearSpec block size，当前常用值是 `32`。

`--tokens N`：NeMo-Skills 请求的最大生成 token 数，默认 `8192`。

`--context-length N`：SGLang server context length。不传时沿用 `observations/eval_sglang.sh` 的自动规则；如果需要覆盖输入 8192 + 输出 8192 的上界，可以设为 `16384`。

`--max-samples N`：每个 benchmark 只跑 N 条样本，适合 smoke test。全量运行不要加这个参数。

`--port N` 和 `--proxy-port N`：固定 SGLang server 和 timing proxy 端口。不传时如果默认端口被占用，会自动寻找空闲端口。

`--output-path DIR`：新实验输出根目录。

`--include-values`：summary JSON 中额外保存每个指标的原始 values 数组。默认不保存 values，因为全量 benchmark 会很大；完整逐轮信息始终保存在 `traces/raw_trace_<benchmark>.jsonl` 中。

`--no-values`：兼容旧命令的参数。当前已经默认不保存 values，所以通常不用加。

`--dry-run`：校验迁移后的顶层入口、`observations/eval_sglang.sh`、汇总脚本和输出路径，只打印解析结果，不创建目录、不启动模型。

## 10. 怎么看结果

查看某个 benchmark 的分布 summary：

```bash
python -m json.tool /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/summaries/confidence_distribution_gsm8k.json | less
```

只看各分布的样本数：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/summaries/confidence_distribution_gsm8k.json'; d=json.load(open(p)); print(d['rounds']); print({k:v['count'] for k,v in d['distributions'].items()})"
```

直接读取 raw trace 的前几轮：

```bash
head -n 3 /data/home/wly/dLLM/NLD_results/observations/sglang_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/traces/raw_trace_gsm8k.jsonl
```

## 11. 结果解释注意事项

这个实验会额外计算 softmax 和 rank，所以不要用它的 wall time、TPS、TPOT 与论文复现效率直接比较。

`rejected_correct_token_rank=1` 在失败位置通常是不合理信号，因为 draft token 本身就是排除 `MASK` 后的 argmax。若全量实验中大量出现 rank=1，应优先检查是否使用了旧版本 trace 结果，或是否存在 logits 缓冲区被后续 verify forward 复用/覆盖的问题。当前实现已经在 draft pass 后立即 clone logits 来避免这个问题。

`confidence_drop_abs` 或 `confidence_drop_pct` 为负数是可能的，表示失败位置的 draft confidence 反而高于本轮通过 token 的平均 confidence。这类样本值得单独检查。

如果 `confidence_drop_abs` 数量少于 `rejected_draft_confidence` 数量，通常是因为部分轮一开始就失败，没有通过 token 均值可以比较。
