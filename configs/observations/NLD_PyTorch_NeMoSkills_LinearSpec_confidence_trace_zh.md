# NLD 原生 PyTorch + NeMo-Skills LinearSpec Confidence / Rank 诊断实验指南

> 记录时间：2026-08-04 16:35 CST（Asia/Shanghai）
>
> 入口：`observations/eval_pytorch_linearspec_confidence.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/`
>
> 目录整理：入口与结果已分别迁入 `observations/` 和外部 observation 结果根目录；底层 `xp/` 实现仍位于项目根目录下。

## 1. 实验用途和边界

这条链路把 `NLD SGLang LinearSpec LoRA Confidence / Rank` 诊断迁移到了原生 Hugging Face remote code + PyTorch 后端，同时保留正式 NeMo-Skills 的 benchmark、数据、prompt 和 scorer 组织。它不启动、不导入 SGLang。

```text
observations/eval_pytorch_linearspec_confidence.sh
  → 独立 GPU 显存占位进程（可选）
  → 独立 native PyTorch 诊断 server
  → NeMo-Skills 正式 prompt / dataset / scorer pipeline
  → accuracy + 原生 NFE/TPF/TPS 合并
  → LinearSpec 逐轮 trace
  → 每个 benchmark 的 confidence/rank 分布汇总与不变量校验
```

本实验的重点是解释 draft/verify 行为。采集会额外保存 logits、计算 full-vocabulary softmax/rank 并写 JSONL，因此其 TPS、wall time、显存峰值不能作为无插桩性能基线。正式原生 PyTorch 性能请使用 `observations/eval_pytorch_nemo.sh`，SGLang 性能请使用 `observations/eval_sglang.sh`。

## 2. 指标定义

所有 confidence 均使用采集位置的 draft logits，转为 FP32 后排除 `MASK` token，再做 softmax；rank 也先排除 `MASK`。

| 指标 | 统计对象与公式 |
|---|---|
| `accepted_draft_confidence` | 所有实际通过 AR verify 并输出的 draft token 的置信度；不含每轮位置 0 的 AR seed token |
| `rejected_draft_confidence` | 每个发生拒绝的轮次中，第一个 mismatch 位置上 draft 所选 token 的置信度 |
| `rejected_correct_token_rank` | 同一 mismatch 位置上，AR verify 给出的正确 token 在 draft logits 中的 1-based competition rank：`1 + count(logit > correct_logit)` |
| `confidence_drop_abs` | 本轮拒绝前 accepted confidence 均值减 rejected confidence |
| `confidence_drop_pct` | `confidence_drop_abs / accepted_mean_confidence` |

重要边界：

- 一轮开头立即 mismatch 时没有 accepted 均值，因此该轮有 rejected confidence/rank，但没有两项 drop。
- `confidence_drop_abs/pct` 可以为负，表示拒绝位置反而比本轮此前通过位置更自信。
- temperature 为 0 时，mismatch 的正确 token 通常不是唯一 rank 1；BF16 完全并列时 competition rank 仍可能为 1，这是合法现象。
- temperature 大于 0 时 draft/verify 都会采样，mismatch 位置出现 rank 1 更不异常；做 SGLang 贪心对照建议显式使用 `--temperature 0 --threshold 0`。
- threshold 大于 0 时，一个 block 可能需要多次 draft forward。实现为每个位置保留“该位置真正被 commit 的那一次 forward”的 logits，而不是最后一次 forward 的共享缓冲区。
- native remote code 可能在最后一轮内部产生略多于 API 返回上限的 token，随后 server 截断到 `--tokens`。trace 记录实际执行过的内部轮次；`pytorch_request_stats.jsonl` 同时记录 `completion_tokens` 和 `raw_generated_tokens`。

## 3. 新增文件与复用关系

| 文件 | 职责 |
|---|---|
| `observations/eval_pytorch_linearspec_confidence.sh` | 独立顶层入口；校验参数、准备数据、管理 reserve/server、逐 benchmark 评测、汇总、失败恢复 |
| `xp/pytorch_linearspec_confidence/native_linearspec_confidence.py` | 与模型 `linear_spec_generate()` 对齐的插桩解码器；保持 adapter、attention、KV crop、EOS、thinking 和 NFE 逻辑 |
| `xp/pytorch_linearspec_confidence/confidence_trace.py` | 逐轮计算 accepted/rejected confidence、rank、drop，并线程安全追加 JSONL |
| `xp/pytorch_linearspec_confidence/pytorch_confidence_openai_server.py` | 诊断专用 OpenAI-compatible 原生 PyTorch server |
| `xp/pytorch_linearspec_confidence/summarize_linearspec_confidence_trace.py` | 生成 count/mean/std/quantile/histogram/rank exact counts，并检查 trace 不变量 |

只读复用：

| 文件/目录 | 复用内容 |
|---|---|
| `xp/pytorch_nemo_eval/pytorch_openai_server.py` | tokenizer、chat template、OpenAI API、原生请求锁、NFE/TPF/TPS stats |
| `xp/pytorch_nemo_eval/add_pytorch_metrics_to_metrics.py` | 将请求统计合并到 NeMo `metrics.json` |
| `xp/pytorch_nemo_eval/gpu_memory_reserver.py` | 在指定 GPU 上真实占位指定 GiB |
| `xp/nemo-skills/eval_dlm.py` | 与 SGLang/正式 PyTorch 链路相同的 NeMo-Skills benchmark、prompt、数据和评分入口 |
| `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B` | 模型 remote code、权重、tokenizer 和 `linear_spec_lora` |

该实现没有修改 `observations/eval_sglang.sh`、`observations/eval_linearspec_confidence.sh`、`observations/eval_pytorch_nemo.sh`、模型目录或 SGLang fork。只有显式运行新入口时才启用诊断。

## 4. 环境与运行前检查

激活环境：

```bash
conda activate nld_sglang
```

检查核心依赖：

```bash
python -c "import torch,transformers,peft,fastapi,uvicorn,nemo_skills; print(torch.__version__,transformers.__version__)"
```

检查模型与 LoRA：

```bash
ls -lh /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/model.safetensors /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors
```

检查 GPU 和正在运行的相关实验：

```bash
nvidia-smi
```

```bash
ps -eo pid,lstart,cmd --sort=start_time | rg 'observations/eval_sglang.sh|observations/eval_pytorch_nemo.sh|observations/eval_pytorch_linearspec_confidence.sh|pytorch_confidence_openai_server|sglang.launch_server'
```

查看入口帮助：

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --help
```

IFEval 需要 `--google-research-dir` 指向含 `instruction_following_eval` 的 checkout；默认位置是 `/data1/linyewei/datasets/NLD/google-research`。

## 5. 推荐运行方式

以下所有命令都是单行形式。

### 5.1 dry run：只检查参数

不会创建结果目录、启动进程、准备数据或加载模型。

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 1 --block-size 16 --tokens 128 --context-length 4096 --disable-thinking --max-samples 1 --dry-run
```

### 5.2 最小真实 smoke test

只测 GSM8K 第一题，并在 summary 中保留原始 values，适合核对字段。

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 1 --block-size 16 --threshold 0 --temperature 0 --tokens 128 --context-length 4096 --disable-thinking --max-samples 1 --include-values --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results
```

### 5.3 单 benchmark 全量诊断

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 30 --block-size 16 --threshold 0 --temperature 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results
```

### 5.4 多 benchmark 顺序运行

脚本按命令行顺序逐项评测。每个 benchmark 都重启独立 server，并写独立 trace/summary，避免不同任务的数据混在一起；某一项失败后会记录错误并继续后续项。

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks human-eval:1,mbpp:1,gsm8k:1,math-500:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 30 --block-size 16 --threshold 0 --temperature 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results
```

### 5.5 不加载 LoRA 的 LinearSpec base 对照

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_base --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --threshold 0 --temperature 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/base
```

### 5.6 threshold 多步 draft 诊断

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 20 --block-size 16 --threshold 0.9 --temperature 0 --tokens 2048 --context-length 4096 --disable-thinking --max-samples 20 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/threshold09
```

### 5.7 thinking 模式

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --threshold 0 --tokens 8192 --context-length 10240 --enable-thinking --keep-thinking --max-thinking-tokens 6000 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/thinking
```

若希望生成时启用 thinking、评分前在支持的任务上剥离 thinking，将 `--keep-thinking` 换为 `--strip-thinking`。

## 6. 全部命令行参数

### 6.1 入口、模式与输出

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--benchmarks LIST` | 逗号分隔的 NeMo benchmark spec；`:1` 是每题 repeat 1 次，不是只取 1 题 | 必填 |
| `--mode` | `linearspec_lora` 或 `linearspec_base` | `linearspec_lora` |
| `--output-path DIR` / `--out-dir DIR` | 结果根目录；每次自动追加时间戳子目录 | `/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results` |
| `--include-values` | summary 额外保留完整原始 values；raw trace 始终保留 | 关闭 |
| `--no-values` | 显式关闭 summary values，兼容参数 | 默认行为 |
| `--bins N` | confidence/drop histogram bin 数；rank 使用 exact counts 并附带 50-bin histogram | 100 |
| `--dry-run` | 仅解析、校验并显示配置 | 关闭 |

### 6.2 模型与解码

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--model PATH` | 本地模型目录或 HF id | 本地 NLD-8B |
| `--served-model-name NAME` | OpenAI API 中的模型标签，不是权重路径 | `nemotron-labs-diffusion-8b` |
| `--lora-path DIR` | `linearspec_lora` adapter；base 模式自动忽略 | `<model>/linear_spec_lora` |
| `--block-size N` / `--block-length N` | 原生 LinearSpec block 长度 | 32 |
| `--threshold V` | draft unmask 阈值；0 表示一次填满 block，正值可能多次 draft forward | 0 |
| `--temperature V` | 原生 draft 和 verify 实际使用的 temperature | 0 |
| `--top-p V` | 传入 API 并记录以对齐协议；当前原生 remote method 不应用 top-p | 0.95 |
| `--tokens N` | 最多返回给 NeMo 的 completion token 数 | 8192 |
| `--context-length N` | prompt token + 内部生成预算的上限 | `tokens + 2048` |
| `--dtype` | `bfloat16/bf16`、`float16/fp16`、`float32/fp32` | bfloat16 |
| `--max-thinking-tokens N` | 超过预算仍无 `</think>` 时注入结束 token | 不启用 |

LinearSpec 仍遵循原模型 LoRA 语义：causal prefill 和 AR verify 关闭 adapter，draft 阶段开启 adapter；`linearspec_base` 全程没有有效 adapter。

### 6.3 GPU、并发与端口

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--gpu-device ID` / `--gpu-devices ID` | 一个物理 GPU ID；不支持逗号多卡或 tensor parallel | 0 |
| `--batch-size N` | 与 SGLang 命令对齐的兼容参数；原生 LinearSpec 只允许 1 | 1 |
| `--client-concurrency N` | NeMo 同时发出的请求数；GPU 模型调用仍由全局锁串行 | 1 |
| `--num-chunks N` | NeMo 客户端 chunk 数 | client concurrency |
| `--gpu-memory-reserve-gb V` | 模型加载前在所选 GPU 上真实占位 V GiB，退出时释放 | 0 |
| `--port N` | 诊断 server 固定端口；显式端口冲突会报错 | 自动从 `33000 + GPU ID` 向上找空闲端口 |

`--gpu-memory-reserve-gb` 是真实 CUDA allocation，不是 PyTorch memory fraction。必须确保 reserve 后剩余显存还能容纳约 16 GiB BF16 权重、LoRA、CUDA context、KV cache、activation 和诊断 logits。并行实验最好使用不同 GPU；不要依赖 reserve 阻止其他进程继续抢占剩余显存。

### 6.4 Benchmark、thinking 与环境

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--max-samples N` | 每个 benchmark 最多跑 N 道题 | 全量 |
| `--quick-test` | NeMo quick test | 关闭 |
| `--enable-thinking` | native chat template 启用 thinking | 关闭 |
| `--disable-thinking` | 显式 non-thinking；与 `--enable-thinking` 互斥 | 关闭，但模型默认即 non-thinking |
| `--keep-thinking` | NeMo 输出/评分流程保留 thinking | 关闭 |
| `--strip-thinking` | NeMo 在支持的任务上剥离 thinking 并重新评分 | 关闭；与 keep 互斥 |
| `--math-prompt-config NAME` | 覆盖数学类任务的 NeMo prompt config | NeMo 默认 |
| `--pytorch-python PATH` | 模型 server Python | `nld_sglang/bin/python` |
| `--eval-python PATH` | NeMo-Skills Python | 同 server Python |
| `--nemo-skills-data-dir DIR` | 持久化准备数据与 cache 根目录 | `/data1/linyewei/datasets/NLD` |
| `--google-research-dir DIR` | IFEval scorer checkout | `<data-dir>/google-research` |

## 7. 正式支持的 benchmark

与 `observations/eval_sglang.sh` / `observations/eval_pytorch_nemo.sh` 相同，可按任意顺序选择：

```text
human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1
```

代码题依赖 NeMo-Skills 当前代码评测环境；IFEval 依赖 google-research checkout 及 `langdetect/immutabledict/nltk`。数据不存在时入口会优先复制持久 cache，否则使用 NeMo dataset prepare；共享准备过程使用 `.prepare.lock` 减少并行竞争。

## 8. 结果目录与字段

每次运行生成：

```text
results/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/
├── Settings.json
├── benchmark_status.jsonl
├── metrics_<benchmark>.json
├── error_<benchmark>.json                    # 仅该项失败时
├── traces/
│   └── raw_trace_<benchmark>.jsonl
├── summaries/
│   └── confidence_distribution_<benchmark>.json
├── eval_runs/<benchmark>/eval-results/<benchmark>/
│   ├── output-rs0.jsonl
│   ├── metrics.json
│   ├── pytorch_request_stats.jsonl
│   └── pytorch_native_metrics_summary.json
└── runtime/<benchmark>/
    ├── pytorch_confidence_server.log
    ├── nemo_skills_benchmark.log
    └── pytorch_request_stats.jsonl
```

如果启用 reserve，`runtime/` 还会有 `gpu_memory_reserver.log` 和 `gpu_memory_reserver_ready.json`。

关键文件：

- `Settings.json`：完整调用参数、解析后配置、端口、模型、LoRA、GPU、thinking 和输出路径。
- `benchmark_status.jsonl`：每个 benchmark 的 `completed/failed`、各阶段退出码、trace 条数及产物路径。
- `metrics_<benchmark>.json`：NeMo accuracy/scorer 字段，以及 `pytorch_native`、NFE、TPF、TPS 等原生统计。
- `raw_trace_*.jsonl`：一行一个 draft/verify round，包含 request/round/position/token id、accepted confidence 数组、首次拒绝 confidence/rank、drop、NFE、draft forward 数。
- `confidence_distribution_*.json`：五项指标的 count/mean/std/min/max/quantiles/histogram；rank 另有 `exact_rank_counts`；`validation` 全为 true 且 `status=ok` 才算汇总成功。

## 9. 查看与检查结果

列出最近一次运行：

```bash
find /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results -maxdepth 1 -type d -name 'linearspec_confidence_*' | sort | tail -n 1
```

查看 benchmark 状态：

```bash
python -m json.tool /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/benchmark_status.jsonl
```

查看某任务的轮次、五项分布 count 和 validation：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/summaries/confidence_distribution_gsm8k.json'; d=json.load(open(p)); print(d['status']); print(d['rounds']); print({k:v['count'] for k,v in d['distributions'].items()}); print(d['validation'])"
```

查看 rejected correct token 的精确 rank 计数：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/summaries/confidence_distribution_gsm8k.json'; d=json.load(open(p)); print(d['distributions']['rejected_correct_token_rank']['exact_rank_counts'])"
```

查看前三轮 raw trace：

```bash
head -n 3 /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/traces/raw_trace_gsm8k.jsonl
```

查看 accuracy + NFE/TPF/TPS：

```bash
python -m json.tool /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/metrics_gsm8k.json | less
```

`benchmark_status.jsonl` 是 JSONL，不是单个 JSON；只有一行时 `python -m json.tool` 可以直接看，多行时推荐逐行解析：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS/benchmark_status.jsonl'; [print(json.dumps(json.loads(x),ensure_ascii=False,indent=2)) for x in open(p) if x.strip()]"
```

## 10. 校验规则、失败语义与排错

汇总器会检查：

- accepted confidence 数量等于 `accepted_draft_tokens`；
- rejected confidence 数量等于 rejected correct rank 数量；
- drop 数量不超过 rejection 数量；
- confidence 全部位于 `[0,1]`；
- rank 全部大于等于 1；
- 至少存在一个有效 trace round。

任一条件失败时 summary 写出 `status=invalid` 并返回非零，顶层入口将该 benchmark 标为 failed。采集内部异常会记录 server traceback、停止后续插桩但继续当前生成；最终因 trace 为空或计数异常而显式失败，不会静默给出伪正常诊断结果。

建议排错顺序：

1. 先用 `--dry-run` 检查 GPU、reserve、block、tokens/context、thinking、端口和输出目录。
2. server 加载/OOM 看 `runtime/<benchmark>/pytorch_confidence_server.log`。
3. 显存占位失败看 `runtime/gpu_memory_reserver.log` 和 ready JSON。
4. NeMo prompt、数据或 scorer 失败看 `runtime/<benchmark>/nemo_skills_benchmark.log` 和 `error_<benchmark>.json`。
5. accuracy 正常但诊断失败时先看 raw trace 最后一行，再看 summary 的 `validation` 与 `decode_errors`。
6. 端口不要与已有实验共用；不传 `--port` 时会自动避让，手动指定的端口被占用则立即报错。

## 11. 隔离与并行保证

- 默认端口段从 `33000 + GPU ID` 开始，与原生正式入口常用 32000、SGLang 30000、timing proxy 31000 分离。
- 每次运行创建唯一时间戳目录；不同 benchmark 使用独立 trace、summary、stats 和 server log。
- 原生生成有全局锁，即使 `--client-concurrency > 1` 也不会并发修改模型的 adapter/attention 状态；并发请求只会排队。
- 请求 id 使用线程局部上下文，trace 追加使用文件锁；不会把排队请求的 id 写串。
- cleanup trap 只结束本入口自己创建的 server/reserver PID，不扫描、不杀其他 SGLang/PyTorch 实验。
- 与其他实验只共享只读模型、LoRA 和已准备数据。新代码不覆盖旧入口或旧结果。

不同 GPU 并行示例：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/parallel_sglang
```

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-devices 1 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 30 --block-size 16 --threshold 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/parallel_pytorch_confidence
```

## 12. 与 SGLang 诊断比较时的口径

可直接比较五项 confidence/rank 分布，但至少应固定：同一权重/LoRA、benchmark 顺序和 repeat、数据与 NeMo prompt、样本数、temperature、threshold、block size、tokens/context、thinking、dtype。

两端都使用如下定义：accepted 不含 seed，只统计首次 mismatch，confidence/rank 排除 `MASK`，rank 使用 competition rank，drop 使用本轮此前 accepted 均值。

仍需注意：SGLang scheduler/KV/cache/停止边界与 native remote code 不同，逐轮数量和 NFE 边界未必相同；本诊断额外开销也与 SGLang 插桩实现不同。因此以分布和 token/round 对齐为主，不用诊断链路 TPS 判断引擎快慢。

## 13. 真实 GPU 2 验证记录

2026-08-04 16:25–16:31 CST 在 GPU 2、`nld_sglang` 环境完成 GSM8K 单样本端到端验证：8B BF16 权重与 bundled LoRA 实际加载，`--gpu-memory-reserve-gb 1` 生效，NeMo prompt/scorer、accuracy、NFE/TPF/TPS 合并、逐轮 trace、五项分布和 cleanup 均通过。

诊断运行结果：

```text
results/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_162508
```

本次得到 15 个 trace round、116 个 accepted confidence、11 个 rejected confidence、11 个 rejected correct rank、11 个 abs/pct drop；所有 validation 为 true。随后用未插桩 `observations/eval_pytorch_nemo.sh` 在同一 GPU、同一 GSM8K 样本及同一解码参数上进行等价对照，生成文本逐字一致，且两边均为 `completion_tokens=128`、`raw_generated_tokens=131`、`NFE=31`、`finish_reason=length`。服务端口 33002/32002 和本次创建的模型/reserve 进程均已释放。

这是短 token smoke，仅验证实现正确性，不代表正式 GSM8K accuracy 或性能。

## 14. 与指定 SGLang 十数据集实验对齐的可直接运行命令

下面命令与指定 SGLang 实验在 benchmark 顺序、单请求 client concurrency、batch size 1、GPU 2、40 GiB 显存占位、LinearSpec + LoRA、block size 16、threshold 0、temperature 0、8192 生成长度、10240 上下文长度和 non-thinking 设置上对齐。替换最后的 `--output-path` 即可改变结果根目录；正式全量运行不要添加 `--max-samples`。

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --mode linearspec_lora --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 0 --block-size 16 --threshold 0 --temperature 0 --top-p 0.95 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results
```
