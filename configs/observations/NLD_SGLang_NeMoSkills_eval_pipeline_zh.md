# SGLang + NeMo-Skills 论文式 benchmark 迁移评测路径

> 入口：`observations/eval_sglang.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/`

本文说明本次新增的评测路径：保留 `eval.sh` 使用的 NeMo-Skills benchmark 组织、prompt、dataset、scoring 逻辑，但把实际推理后端换成本地 SGLang engine。

新增入口：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

项目根目录默认是：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

默认模型权重是：

```text
/data1/linyewei/models/Nemotron-Labs-Diffusion-8B
```

默认 Python 是：

```text
/data/home/wly/.conda/envs/nld_sglang/bin/python
```

注意：这条路径需要 `--eval-python` 对应的 Python 同时能导入：

```text
nemo_skills.pipeline.eval
fastapi
httpx
uvicorn
```

检查命令：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python -c "import fastapi, httpx, uvicorn; from nemo_skills.pipeline.eval import eval; print('ok')"
```

当前 `nld_sglang` 环境已经补齐了这条路径所需的 NeMo-Skills 依赖，并且脚本会把 `--eval-python` 所在目录 prepend 到 `PATH`，确保 NeMo-Run 子进程里写死的 `python -m ...` 也使用 `nld_sglang/bin/python`，不会被其他 conda 环境抢先。新脚本会在启动 SGLang 前做这个检查，避免模型已经加载后才失败。

环境说明：官方依赖中 `SGLang`、`LiteLLM`、`LeptonAI` 的 `openai/httpx` 元数据约束互相冲突。本路径实际使用的是 NeMo-Skills/LiteLLM 请求链路可运行的组合，并已通过端到端 smoke test；Lepton 后端不是这条 SGLang 迁移路径的一部分。

## 1. 这条新路径解决什么问题

原来项目里有三条评测/benchmark 路径：

- `evaluate.py`：本地单进程，直接调用 HF model 函数，快速测 `gsm8k`、`math-500`。
- `eval.sh`：SLURM + 容器 + HTTP server + NeMo-Skills，是作者确认的论文复刻 benchmark pipeline。
- SGLang 原 benchmark：跑在 SGLang engine 上，但 accuracy 和 serving efficiency 分开，prompt / dataset / scoring 不是严格沿用 `eval.sh` 的 NeMo-Skills 路径。

本次新增的是第四条“迁移后”的路径：

```text
NeMo-Skills benchmark/prompt/dataset/scoring + SGLang inference backend + 单次运行同时记录 accuracy 与效率指标
```

调用链是：

```text
observations/eval_sglang.sh
  -> xp/examples/run_sglang_eval_pipeline_gpu_only.sh
     -> 启动 sglang.launch_server
     -> 启动 xp/sglang_eval/openai_timing_proxy.py
     -> 调 xp/nemo-skills/eval_dlm.py --no-extra-body
        -> NeMo-Skills prepare/eval/scoring
        -> HTTP 请求发到 timing proxy
        -> proxy 流式转发给 SGLang server
     -> xp/sglang_eval/add_sglang_metrics_to_metrics.py 合并 SGLang 指标到 metrics.json
```

这里继承的是 `eval.sh` 的 benchmark 组织层：

- benchmark 名称和 `name:reps` 写法。
- NeMo-Skills 的数据准备、prompt config、输出 JSONL 和 scorer。
- `eval_dlm.py` 里原有的 math prompt override、thinking strip/rescore 等逻辑。

这里不继承的是 legacy 非 SGLang 推理层：

- 不启动 `xp/dlm_api/dlm_batch_server.py`。
- 不启动 legacy load balancer。
- 不使用 SLURM / container / sbatch。
- 不把旧 HTTP server 专属的 `extra_body.steps/block_length/threshold/generation_algorithm` 传给 SGLang。

按照你的要求，SGLang 推理参数暂时使用 SGLang 当前默认行为和 SGLang-native 配置；旧 pipeline 的 prompt / dataset / scoring 完整迁移。

## 2. 代码改动位置

本次新增/修改的关键文件：

```text
observations/eval_sglang.sh
xp/examples/run_sglang_eval_pipeline_gpu_only.sh
xp/sglang_eval/openai_timing_proxy.py
xp/sglang_eval/add_sglang_metrics_to_metrics.py
xp/nemo-skills/eval_dlm.py
```

其中：

- `observations/eval_sglang.sh`：用户入口，解析命令行参数，设置输出目录和环境变量。
- `run_sglang_eval_pipeline_gpu_only.sh`：实际 pipeline，启动 SGLang server、timing proxy、NeMo-Skills eval、指标合并。
- `openai_timing_proxy.py`：OpenAI-compatible timing proxy，把 NeMo-Skills 的非流式请求转成 SGLang streaming 请求，用于记录 TTFT / TPOT / latency / tokens/sec。
- `add_sglang_metrics_to_metrics.py`：读取 NeMo-Skills `metrics.json`、SGLang decode stats JSONL、proxy timing JSONL，把 accuracy 与效率指标合并到同一个 `metrics.json`。
- `eval_dlm.py`：新增 `--no-extra-body`、`--num-chunks`、`--max-concurrent-requests`。`--no-extra-body` 会保留 NeMo-Skills benchmark/prompt/scoring，但不发送 legacy DLM server 专属 `extra_body`。

## 3. 最小运行步骤

先进入项目根目录：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

激活环境：

```bash
conda activate nld_sglang
```

先做 dry-run，只解析配置，不启动服务：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --dry-run
```

跑一个很小的 GSM8K smoke test：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --max-samples 10 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

注意：`human-eval` / `mbpp` 这类 `evalplus` 代码 benchmark 不建议用 `--max-samples` 做 smoke，因为 evalplus 默认要求样本覆盖完整题集。代码 benchmark 可以改用较小的 `--tokens`、独立端口或先跑 `--dry-run` 检查配置。

跑完整 GSM8K：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

跑多个 benchmark：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1,math-500:1,aime24:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

不传 `--benchmarks` 时会使用和 `eval.sh` 一致的完整默认 suite：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

指定输出目录：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

## 4. 支持的解码模式

`--mode linearspec_lora`：

```text
SGLang LinearSpec + LoRA draft weights。默认使用 sglang_dllm/linear_spec_lora。
```

命令：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

`--mode linearspec_base`：

```text
SGLang LinearSpec，不加载 LoRA adapter。
```

命令：

```bash
bash observations/eval_sglang.sh --mode linearspec_base --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

`--mode fastdiffuser`：

```text
SGLang FastDiffuser / dLLM block denoising。
```

命令：

```bash
bash observations/eval_sglang.sh --mode fastdiffuser --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

`--mode ar`：

```text
SGLang AR mode，通过 --json-model-override-args '{"ar_mode": true}' 启动。
```

命令：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

## 5. 支持的 benchmark

原则上支持当前 NeMo-Skills 环境能 `prepare_data` 和 `eval` 的 benchmark。与 `eval.sh` 默认套件一致，常用名称包括：

```text
gsm8k
human-eval
mbpp
math-500
aime24
aime25
gpqa
mmlu
ifeval
livecodebench-cpp
```

写法仍然是 NeMo-Skills / `eval.sh` 的 `name:reps`：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1,math-500:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

如果只想快速检查 math pipeline：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --max-samples 5 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

如果检查 HumanEval 路径，建议跑完整题集；例如：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks human-eval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --cuda-graph-bs "1"
```

如果 benchmark 需要额外依赖或数据下载，`run_sglang_eval_pipeline_gpu_only.sh` 会直接调用当前 `--eval-python` 执行 `python -m nemo_skills.dataset.prepare <benchmark> --parallelism 20 --retries 3`。这样不会依赖用户全局 NeMo-Run cluster 配置，也不会误用其他 conda 环境。

## 6. accuracy 与效率指标如何一次跑完

新路径只跑一次 NeMo-Skills eval，同一次请求流中同时得到：

- NeMo-Skills 原始 accuracy / pass@k / benchmark-specific metric。
- SGLang decode stats：`forward_passes`、accepted/generated tokens、TPF。
- Timing proxy stats：wall time、latency、TTFT、TPOT、tokens/sec。

SGLang decode stats 来源：

```text
SGLang DLLM algorithm YAML 里写入 stats_file: <runtime>/sglang_decode_stats.jsonl
```

Timing stats 来源：

```text
xp/sglang_eval/openai_timing_proxy.py 强制用 stream=True 转发给 SGLang，并请求 stream_options.include_usage=True。
```

最终结果位置：

```text
<output-path>/eval_YYYYMMDD_HHMMSS/metrics_<benchmark>.json
```

如果一次运行多个 benchmark，同一个时间戳目录下会生成多个文件，例如：

```text
<output-path>/eval_YYYYMMDD_HHMMSS/metrics_gsm8k.json
<output-path>/eval_YYYYMMDD_HHMMSS/metrics_math-500.json
```

内部仍会临时使用 NeMo-Skills 原生输出结构、SGLang server log、decode stats 和 timing JSONL 来完成评分与指标合并。每个 benchmark 评估和指标合并完成后，会立即把最终结果写到 `metrics_<benchmark>.json`；成功后默认清理内部文件。如果后续 benchmark 中途失败，已经完成并写出的 `metrics_<benchmark>.json` 会保留在时间戳目录下。

`metrics_<benchmark>.json` 中新增字段：

```text
sglang.decode.decode_blocks
sglang.decode.decode_tokens
sglang.decode.decode_forward_passes
sglang.decode.tokens_per_forward_pass
sglang.decode.mean_tokens_per_block
sglang.decode.mean_forward_passes_per_block
sglang.decode.weighted_acceptance_rate
sglang.decode.mean_acceptance_rate
sglang.serving.request_count
sglang.serving.failed_request_count
sglang.serving.prompt_tokens
sglang.serving.completion_tokens
sglang.serving.wall_time_s
sglang.serving.benchmark_wall_time_s
sglang.serving.request_window_time_s
sglang.serving.latency_s.mean/p50/p90/p95/p99
sglang.serving.ttft_s.mean/p50/p90/p95/p99
sglang.serving.tpot_s.mean/p50/p90/p95/p99
sglang.serving.per_request_output_tokens_per_s.mean/p50/p90/p95/p99
sglang.serving.wall_output_tokens_per_s
sglang.serving.wall_requests_per_s
sglang.serving.request_window_output_tokens_per_s
sglang.serving.request_window_requests_per_s
average_nfe
tokens_per_forward_pass
tpf
```

其中：

- `average_nfe`：按 decode forward passes / request_count 聚合。
- `tokens_per_forward_pass` / `tpf`：`sum(decode_tokens) / sum(decode_forward_passes)`。
- `wall_time_s` / `benchmark_wall_time_s`：整个 NeMo-Skills benchmark 子流程耗时，包括 NeMo-Run 启动、数据读取、生成、打分、汇总等开销。
- `wall_output_tokens_per_s`：用整个 benchmark wall time 计算的 output token 吞吐，适合看端到端 pipeline 成本。
- `request_window_time_s`：timing proxy 观察到的第一条请求开始到最后一条请求结束的时间窗口。
- `request_window_output_tokens_per_s`：用 request window 计算的 output token 吞吐，更接近 serving 请求窗口吞吐。
- `per_request_output_tokens_per_s`：每个请求独立的 output token/s 分布。
- `ttft_s`：proxy 收到请求到第一个非空 streamed delta 的时间。
- `tpot_s`：首 token 后平均每个后续 output token 的时间；如果只生成 1 个 token，则为 0。
- AR mode 没有 DLLM stats file 时，合并脚本用 `completion_tokens` 作为 forward passes，因此 AR 的 TPF 记为 1。

## 7. 常用控制参数

Benchmark 相关：

- `--benchmarks LIST`：benchmark 列表，例如 `gsm8k:1,math-500:1`；不传时默认使用 `eval.sh` 的完整 suite。
- `--tokens N`：传给 NeMo-Skills 的最大生成 token 数，对应 `++inference.tokens_to_generate=N`。
- `--temperature V`：采样温度，默认 `0`。
- `--top-p V`：top-p，默认 `0.95`。
- `--num-chunks N`：NeMo-Skills client 侧 chunk 并发/切分控制。默认等于 `--client-concurrency`。
- `--max-samples N`：限制每个 benchmark 样本数，用于 smoke test。
- `--quick-test`：NeMo-Skills quick test 模式。
- `--keep-thinking`：保留 `<think>` 内容，不让 NeMo-Skills parse reasoning。
- `--strip-thinking`：让 NeMo-Skills strip reasoning，并在 eval 后继续调用 strip/rescore 保险逻辑。
- `--disable-thinking`：在 SGLang 迁移路径下不发送 legacy `extra_body.chat_template_kwargs`，该参数只保留配置可读性。
- `--math-prompt-config NAME`：对 `gsm8k/math/math-500/aime24/aime25` 覆盖 NeMo-Skills prompt config。

SGLang server 相关：

- `--model PATH`：模型权重路径，默认 `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`。
- `--served-model-name NAME`：SGLang OpenAI API 暴露和 NeMo-Skills 请求使用的模型名，默认 `nemotron-labs-diffusion-8b`。
- `--gpu-devices LIST`：设置 `CUDA_VISIBLE_DEVICES`，例如 `0`、`1`、`0,1`。
- `--tp-size N`：SGLang tensor parallel size；如果未显式指定，脚本会根据 `--gpu-devices` 的 GPU 数自动推断，例如 `--gpu-devices 0,1` 默认使用 `--tp-size 2`。
- `--port N`：SGLang server 端口，默认 `30000`。如果未显式指定且默认端口被占用，脚本会自动寻找下一个空闲端口；如果显式指定的端口被占用，则直接报错。
- `--proxy-port N`：timing proxy 端口，默认 `31000`。如果未显式指定且默认端口被占用，脚本会自动寻找下一个空闲端口；如果显式指定的端口被占用，则直接报错。
- `--batch-size N`：`--max-running-requests` 的别名，控制 SGLang server 最大运行请求数。
- `--max-running-requests N`：SGLang server 端并发请求上限。
- `--client-concurrency N`：同时控制 NeMo-Skills `max_concurrent_requests` 和 timing proxy 转发到 SGLang 的最大 in-flight 请求数。
- `--gpu-memory-reserve-gb V`：在启动 SGLang server 前，在 `--gpu-devices` 指定的每张 GPU 上各自空占 `V` GiB 显存，默认 `0` 表示关闭。这个参数用于防止其他进程误抢当前评测 GPU 的显存；如果 NLD 实际负载约 36GiB，希望总占用约 70GiB，可先尝试 `--gpu-memory-reserve-gb 34`。如果 GPU 上已有其他进程占用显存，需要相应调低这个值。
- `--context-length N`：SGLang context length，默认按 NeMo-Skills 生成长度自动处理。脚本初始值是 `2048`，如果未显式指定且 `--tokens` 大于当前 context length，会自动调到 `tokens + 2048`；例如默认 `--tokens 8192` 会自动使用 `--context-length 10240`。
- `--mem-fraction V`：SGLang `--mem-fraction-static`，默认 `0.55`。
- `--cuda-graph-bs LIST`：SGLang CUDA graph batch sizes，例如 `"1"` 或 `"1 2 4"`。
- `--dtype DTYPE`：默认 `bfloat16`。
- `--quantization NAME`：可选量化，例如 `fp8`，具体取决于本地 SGLang 支持。
- `--block-size N`：写入 SGLang DLLM algorithm YAML 的 `block_size`。
- `--max-steps N`：FastDiffuser 模式写入 YAML 的 `max_steps`。
- `--threshold V`：FastDiffuser 模式写入 YAML 的 `threshold`。
- `--lora-path DIR`：`linearspec_lora` 模式的 LoRA adapter 目录。
- `--lora-mode MODE`：默认 `draft_only`，也可传 SGLang 算法支持的其他值。
- `--extra-server-args "..."`：追加到 `python -m sglang.launch_server` 的原生参数。

输出与环境：

- `--output-path DIR`：最终 compact 输出根目录，默认 `/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results`。
- `--out-dir DIR`：`--output-path` 的兼容别名。
- `--exp-name NAME`：兼容保留参数；当前 compact 输出固定使用 `eval_YYYYMMDD_HHMMSS` 时间戳目录。
- `--sglang-python PATH`：启动 SGLang server、proxy、eval 使用的 Python，默认 `nld_sglang`。
- `--eval-python PATH`：单独指定 NeMo-Skills eval Python；默认等于 `--sglang-python`。
- `--sglang-src DIR`：SGLang source root，默认 `sglang_dllm/src/sglang`。
- `--sglang-work-dir DIR`：SGLang 工作目录，默认 `sglang_dllm`。
- `--hf-home DIR`：HF cache 目录。
- `--sglang-cache-dir DIR`：SGLang cache 目录。
- `--keep-server`：评测结束后不杀 SGLang server 和 timing proxy，方便手工继续请求。
- `--dry-run`：只打印配置，不启动服务。

## 8. 控制 serving 场景的示例

单卡、batch/concurrency 都为 1：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --cuda-graph-bs "1"
```

单卡，提高 SGLang server 运行请求数和 client 并发：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 4 --client-concurrency 4 --num-chunks 4 --cuda-graph-bs "1 2 4"
```

两卡 tensor parallel：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0,1 --batch-size 2 --client-concurrency 2 --num-chunks 2
```

换端口避免冲突：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 1 --port 30010 --proxy-port 31010 --batch-size 1 --client-concurrency 1
```

控制最大生成长度和 context length：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --tokens 4096 --context-length 4096 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

控制 LinearSpec block size：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --block-size 32 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

FastDiffuser 控制 block size、max steps、threshold：

```bash
bash observations/eval_sglang.sh --mode fastdiffuser --benchmarks gsm8k:1 --block-size 8 --max-steps 128 --threshold 0.9 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

保留 server 方便后续手工 curl：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --max-samples 5 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --keep-server
```

## 9. 输出目录结构

默认输出根目录：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results
```

一次运行会创建：

```text
<output-path>/eval_YYYYMMDD_HHMMSS/
  Settings.json
  metrics_<benchmark>.json
  error_<benchmark>.json
```

`Settings.json` 会在本轮实验最开始写入，也就是在显存占位、SGLang server 和 NeMo-Skills benchmark 启动之前。它记录原始命令行参数、自动解析后的最终参数、端口、GPU、TP size、context length、显存空占、输出路径、模型路径和 Python 环境等信息。

成功的 benchmark 会生成 `metrics_<benchmark>.json`。如果某个 benchmark 在 NeMo-Skills eval、metrics 生成或 SGLang 指标合并阶段失败，不会中断后续 benchmark；脚本会为失败项写出 `error_<benchmark>.json`，其中包含失败阶段、退出码、内部日志路径和日志尾部内容。

例如单个 GSM8K：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_20260627_163717/Settings.json
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_20260627_163717/metrics_gsm8k.json
```

例如一次运行 GSM8K 和 MATH-500：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_20260627_163717/Settings.json
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_20260627_163717/metrics_gsm8k.json
/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_20260627_163717/metrics_math-500.json
```

查看 GSM8K metrics：

```bash
python -m json.tool /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/eval_YYYYMMDD_HHMMSS/metrics_gsm8k.json
```

## 10. 与 eval.sh 的对应关系

`eval.sh` 论文式路径：

```text
SLURM/container
  -> xp/examples/run_dlm_eval_pipeline_gpu_only.sh
     -> legacy DLM HTTP server/load balancer
     -> xp/nemo-skills/eval_dlm.py
     -> NeMo-Skills scoring
```

新 SGLang 迁移路径：

```text
local current environment
  -> xp/examples/run_sglang_eval_pipeline_gpu_only.sh
     -> SGLang HTTP server
     -> timing proxy
     -> xp/nemo-skills/eval_dlm.py --no-extra-body
     -> NeMo-Skills scoring
     -> SGLang metrics merge
```

两者共享：

- `xp/nemo-skills/eval_dlm.py` 入口。
- NeMo-Skills benchmark 名称、prompt、dataset、scorer。
- NeMo-Skills benchmark 内部输出和 `metrics.json` 生成逻辑；新入口最终只保留 compact 后的 `metrics_<benchmark>.json`。

两者不同：

- `eval.sh` 依赖 SLURM/container；新路径直接在当前 `nld_sglang` 环境运行。
- `eval.sh` 后端是项目 legacy DLM server；新路径后端是 SGLang server。
- `eval.sh` 通过 legacy `extra_body` 控制 `steps/block_length/threshold/generation_algorithm`；新路径不发送这些旧后端专属参数，SGLang 解码由 server 启动参数和 algorithm YAML 控制。
- 新路径用 timing proxy 在同一次 accuracy eval 中记录 TTFT / TPOT / latency / tokens/sec。

## 11. 与 SGLang 原 benchmark 的关系

原 SGLang benchmark 更适合单独压测 serving workload，例如固定 request rate、随机输入输出长度分布、固定并发下测 p90/p99 latency。

本路径的目标不同：

```text
优先复刻论文 benchmark 的 prompt/dataset/scoring，同时在同一批真实 benchmark 请求上记录 serving-style timing。
```

因此这里记录的 TTFT / TPOT / p90 / p99 是在 NeMo-Skills benchmark 请求分布和你指定 `--client-concurrency/--num-chunks/--batch-size` 下观察到的指标。它不是独立的 synthetic serving benchmark，但它与 accuracy 完全同源、同 prompt、同输出。

如果后续要做纯 serving 压测，仍然可以继续使用 SGLang 原生 `bench_serving.py`；如果要让 accuracy 对齐论文，则优先使用本文这条迁移路径。

## 12. 常见问题

如果端口冲突：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --port 30020 --proxy-port 31020 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

如果显存不够：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --context-length 2048 --mem-fraction 0.5 --batch-size 1 --client-concurrency 1 --gpu-devices 0
```

如果想确认命令解析：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --dry-run
```

如果想看 SGLang server 日志，需要在失败后查看脚本打印的内部工作目录，或者调试时加 `--keep-server` 保留内部目录：

```bash
tail -n 200 /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/.eval_YYYYMMDD_HHMMSS_work_PID/sglang_runtime/sglang_server.log
```

如果想看 proxy 记录的逐请求 timing，同样需要保留内部目录：

```bash
tail -n 20 /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/.eval_YYYYMMDD_HHMMSS_work_PID/results/eval-results/gsm8k/sglang_timing.jsonl
```

如果 `metrics_<benchmark>.json` 中没有 SGLang timing tokens：

```text
检查 sglang_timing.jsonl 是否有 completion_tokens。当前本地 SGLang 支持 stream_options.include_usage=True；如果换到不支持 usage chunk 的 SGLang 版本，TTFT/latency 仍可记录，但 token/sec 需要 usage 才能准确计算。
```

如果 `linearspec_lora` 找不到 LoRA：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --lora-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

如果要改用其他 Python：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --sglang-python /data/home/wly/.conda/envs/nld_sglang/bin/python --eval-python /data/home/wly/.conda/envs/nld_sglang/bin/python --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

如果启动前报 `EVAL_PYTHON cannot import one or more required packages`：

```text
说明评测 Python 缺少 nemo_skills / fastapi / httpx / uvicorn。SGLang server 本身可能能启动，但 NeMo-Skills benchmark/prompt/scoring 无法运行。修复方式是安装 NeMo-Skills 到该 Python，或用 --eval-python 指向已有 NeMo-Skills 的 Python。
```

如果 `ifeval` 在生成完成后报 `/opt/benchmarks/google-research` 找不到，说明本地 NeMo-Skills evaluator 仍在使用容器里的硬编码 scorer 路径。当前脚本只在 `--benchmarks` 包含 `ifeval` 时启用 IFEval scorer 兼容逻辑，默认 scorer checkout 是 `/data1/linyewei/datasets/NLD/google-research`，也可以通过环境变量 `NLD_GOOGLE_RESEARCH_DIR` 覆盖。

```bash
NLD_GOOGLE_RESEARCH_DIR=/data1/linyewei/datasets/NLD/google-research bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks ifeval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 34 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results
```

如果重新创建了 conda 环境或 NeMo-Skills 安装目录，可以手动重新应用本项目里的 evaluator 路径补丁：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python /data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/nemo-skills/patch_ifeval_google_research_path.py
```

## 13. 当前最常用命令

只控制你现在最关心的参数：并发度、batch size、GPU、显存空占、block size、benchmark、output path。

命令模板：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks BENCHMARK_LIST --gpu-devices GPU_LIST --batch-size BATCH_SIZE --client-concurrency CONCURRENCY --gpu-memory-reserve-gb RESERVE_GB --block-size BLOCK_SIZE --output-path OUTPUT_PATH
```

直接可用示例：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks human-eval:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 34 --block-size 32 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results
```

参数可选值：

- `--client-concurrency CONCURRENCY`：正整数，例如 `1`、`2`、`4`、`8`；控制 NeMo-Skills client 和 timing proxy 的并发请求数。
- `--batch-size BATCH_SIZE`：正整数，例如 `1`、`2`、`4`、`8`；对应 SGLang `--max-running-requests`，显存不足时先降这个值。
- `--gpu-devices GPU_LIST`：单卡可用 `0`、`1`、`2`、`3`；多卡可用 `0,1`、`2,3`、`0,1,2,3`。如果不显式传 `--tp-size`，脚本会自动按 GPU 数设置 tensor parallel size。
- `--gpu-memory-reserve-gb RESERVE_GB`：非负数字，例如 `0`、`16`、`32`、`34`。含义是在每张指定 GPU 上先空占 `RESERVE_GB` GiB 显存，再启动 SGLang。默认 `0` 关闭。当前 batch size 1、context 10240 的 NLD 负载大约 36GiB，如果希望每张卡总占用接近 70GiB，可用 `34` 作为起点；如果目标 GPU 已经有其他进程占用显存，需要按 `目标总占用 - 已有占用 - NLD预计占用` 调低。
- `--block-size BLOCK_SIZE`：正整数，例如 `8`、`16`、`32`、`64`；如果不想覆盖 SGLang algorithm 默认值，可以直接省略整个 `--block-size BLOCK_SIZE`。
- `--benchmarks BENCHMARK_LIST`：当前 NLD/eval.sh 论文式默认 suite 的全部 10 个 benchmark 是 `gsm8k:1`、`human-eval:1`、`mbpp:1`、`math-500:1`、`aime24:1`、`aime25:1`、`gpqa:1`、`mmlu:1`、`ifeval:1`、`livecodebench-cpp:1`；多个 benchmark 用逗号连接并按命令行顺序执行，例如 `gsm8k:1,math-500:1,human-eval:1`。每完成一个 benchmark，就会立刻写出对应的 `metrics_<benchmark>.json`。
- `--output-path OUTPUT_PATH`：最终结果根目录；默认是 `/data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results`。

输出结果：

```text
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/Settings.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/metrics_<benchmark>.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/error_<benchmark>.json
```

如果一次测多个 benchmark：

```text
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/Settings.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/metrics_gsm8k.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/metrics_math-500.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/metrics_human-eval.json
```

如果其中某个 benchmark 失败，例如 `math-500` 失败而 `gsm8k` 成功，则同一个目录中会类似：

```text
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/Settings.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/metrics_gsm8k.json
OUTPUT_PATH/eval_YYYYMMDD_HHMMSS/error_math-500.json
```

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 0 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results
```
