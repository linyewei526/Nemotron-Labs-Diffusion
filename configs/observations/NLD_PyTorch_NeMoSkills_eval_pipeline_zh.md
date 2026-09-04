# NLD 原生 PyTorch + NeMo-Skills 正式评测使用手册

> 记录时间：2026-08-04 11:42 CST
>
> 入口：`observations/eval_pytorch_nemo.sh`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/`
>
> 目录整理：入口与结果已分别迁入 `observations/` 和外部 observation 结果根目录；底层 `xp/` 实现仍位于项目根目录下。

## 1. 这条评测链路是什么

它使用模型目录中的 Hugging Face remote code 和原生 PyTorch 推理，不启用 SGLang，但复用与 `observations/eval_sglang.sh` 相同的 NeMo-Skills benchmark、数据、prompt 和 scorer 组织。

```text
observations/eval_pytorch_nemo.sh
  → 独立显存占位进程（可选）
  → 独立 native PyTorch OpenAI server
  → xp/nemo-skills/eval_dlm.py --no-extra-body
  → NeMo-Skills 数据、prompt、生成和评分
  → 原生请求 stats + NeMo accuracy 合并
  → metrics_<benchmark>.json / error_<benchmark>.json
```

三条评测路径的区别：

| 入口 | 后端 | Benchmark/prompt/scorer | 用途 |
|---|---|---|---|
| `evaluate.py` | 原生 PyTorch | 脚本内自定义，仅 GSM8K/MATH-500 | 最轻量 smoke/reference |
| `observations/eval_pytorch_nemo.sh` | 原生 PyTorch | NeMo-Skills 正式十项 | 本文的新基线 |
| `observations/eval_sglang.sh` | SGLang | NeMo-Skills 正式十项 | serving/优化正式结果 |

因此，`observations/eval_pytorch_nemo.sh` 与 `observations/eval_sglang.sh` 的准确率协议更接近；两者的推理实现、batching、KV 管理和性能口径仍不同。

## 2. 新增文件及职责

| 文件 | 职责 |
|---|---|
| `observations/eval_pytorch_nemo.sh` | 用户入口、参数校验、时间戳目录、Settings、端口与最终清理 |
| `xp/examples/run_pytorch_nemo_eval_pipeline_gpu_only.sh` | 启动显存占位/server，准备数据，逐任务调用 NeMo-Skills，失败恢复 |
| `xp/pytorch_nemo_eval/pytorch_openai_server.py` | 原生 HF/PyTorch OpenAI-compatible server，调用三种模型生成方法 |
| `xp/pytorch_nemo_eval/add_pytorch_metrics_to_metrics.py` | 将请求级 NFE、时间、token 汇入 NeMo `metrics.json` |
| `xp/pytorch_nemo_eval/gpu_memory_reserver.py` | 在所选 GPU 上真实占用指定 GiB，进程退出时释放 |

这些都是新文件，不修改 `observations/eval_sglang.sh`、旧 `eval.sh`、SGLang fork 或三个 LinearSpec 诊断实验。

## 3. 支持范围

### 3.1 解码模式

| `--mode` | 原生调用 | mode 默认参数 |
|---|---|---|
| `ar` | `model.ar_generate()` | block 1、threshold None |
| `dlm` | `model.generate()` | block 8、threshold 0.9、causal context |
| `linearspec_base` | `model.linear_spec_generate()` | block 32、threshold 0.0 |
| `linearspec_lora` | PEFT adapter + `linear_spec_generate()` | block 32、threshold 0.0、draft-only LoRA |

兼容别名：`fastdiffuser` 会归一化为原生 `dlm`；`linearspec`、`linear_spec` 等会归一化为 `linearspec_base`。这里的 `dlm` 调用模型 remote code，不是 SGLang `FastDiffuser` 类。

### 3.2 默认十项 benchmark

```text
gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1
```

`benchmark:1` 表示每道题生成一次，即 pass@1；它不是只评一题。限制问题数量用 `--max-samples N`。

## 4. 环境和运行前检查

激活环境：

```bash
conda activate nld_sglang
```

检查 Python 与依赖：

```bash
python -c "import torch,transformers,peft,fastapi,uvicorn,nemo_skills; print(torch.__version__,transformers.__version__,nemo_skills.__version__)"
```

检查模型与 LoRA：

```bash
ls -lh /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/model.safetensors /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors
```

检查 GPU 和现有实验：

```bash
nvidia-smi
```

```bash
ps -eo pid,lstart,cmd --sort=start_time | rg 'observations/eval_sglang.sh|observations/eval_pytorch_nemo.sh|eval_linearspec_|sglang.launch_server|pytorch_openai_server'
```

IFEval 还需要：

```text
/data1/linyewei/datasets/NLD/google-research/instruction_following_eval
```

新 pipeline 不会自动 `pip install` 或修改共享 conda 环境；缺少 IFEval 依赖时会明确失败。

## 5. 推荐运行顺序

所有命令均为单行形式。

### 5.1 只解析参数，不加载模型

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --block-length 16 --max-samples 2 --dry-run
```

### 5.2 AR smoke

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 512 --max-samples 5
```

### 5.3 原生 block diffusion smoke

```bash
bash observations/eval_pytorch_nemo.sh --mode dlm --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 512 --block-length 8 --threshold 0.9 --max-samples 5
```

### 5.4 Linear SS base smoke

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_base --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 512 --block-length 16 --max-samples 5
```

### 5.5 Linear SS + LoRA smoke

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 512 --block-length 16 --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --max-samples 5
```

### 5.6 多个 benchmark

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1,math-500:1,human-eval:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 8192 --block-length 16
```

### 5.7 默认完整十项

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 8192 --block-length 16
```

### 5.8 指定结果根目录

```bash
bash observations/eval_pytorch_nemo.sh --mode dlm --benchmarks gsm8k:1 --gpu-device 2 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/my_pytorch_baseline --block-length 8 --threshold 0.9
```

### 5.9 保留全部原始日志

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks gsm8k:1 --gpu-device 2 --max-samples 5 --keep-runtime
```

### 5.10 长 prompt 显式增加 context

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gpqa:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 8192 --context-length 12288 --block-length 16
```

### 5.11 启用 thinking

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks gsm8k:1 --gpu-device 2 --enable-thinking --max-thinking-tokens 6000 --keep-thinking --tokens 8192
```

## 6. `--gpu-memory-reserve-gb` 的准确含义

示例：

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --block-length 16
```

执行顺序：

1. 在物理 GPU 2 上启动独立占位进程；
2. 占位进程真实分配并保持 30 GiB；
3. 同一个 `CUDA_VISIBLE_DEVICES=2` 下启动模型 server；
4. 模型只能使用剩余显存；
5. 成功、失败或 Ctrl-C 时，trap 先关闭模型，再关闭占位进程并释放显存。

用途是复现实验的显存限制或在不同 GPU 上并行运行时预留边界。它不是 PyTorch memory fraction，也不能阻止其他用户随后占用剩余显存。

注意：

- NLD-8B BF16 主权重约 16 GiB，实际还需要 KV cache、activation、CUDA context 和 LoRA 空间；
- reserve 太大时 server 会在加载模型或生成时 OOM；
- 同一张 GPU 同时跑 SGLang 和 PyTorch 仍会竞争显存，最稳妥方案是选择不同 `--gpu-device`；
- 新入口只支持一个 GPU，不接受 `--gpu-devices 0,1`。

## 7. 全部命令行参数

### 7.1 模式与任务

| 参数 | 含义 | 默认 |
|---|---|---|
| `--mode` | `ar`、`dlm`、`linearspec_base`、`linearspec_lora` | 必填 |
| `--benchmarks LIST` | 逗号分隔的 NeMo benchmark spec | 十项 `:1` |
| `--tokens N` | API 返回的最大 completion token | 8192 |
| `--max-samples N` | 每个任务最多评多少道题 | 全量 |
| `--quick-test` | NeMo quick test；未给 max-samples 时通常限制为 10 | 关闭 |
| `--num-chunks N` | NeMo 客户端数据 chunk 数 | client concurrency |
| `--client-concurrency N` | 同时发出的 HTTP 请求数 | 1 |
| `--math-prompt-config NAME` | 覆盖数学任务的 NeMo prompt config | 空 |

原生模型的 attention mode、LoRA adapter 开关会在生成中改变，因此 server 对 GPU 生成加全局锁。即使 `--client-concurrency > 1`，请求也只会排队并串行进入模型；该参数主要研究客户端排队，不会形成 SGLang continuous batching。

### 7.2 解码

| 参数 | 含义 | 默认 |
|---|---|---|
| `--block-length N` / `--block-size N` | dLM/Linear block | dLM 8，Linear 32，AR 1 |
| `--threshold V` | 原生置信阈值；`none`/`null` 可关闭 | dLM 0.9，Linear 0.0 |
| `--temperature V` | 原生方法实际使用的 temperature | 0 |
| `--top-p V` | 为协议对齐传入并记录 | 0.95；当前原生方法不应用 |
| `--causal-context` | dLM 使用 clean-prefix causal KV | 开启 |
| `--no-causal-context` | 关闭上述行为 | — |
| `--context-length N` | prompt + 内部生成预算的安全上限 | tokens + 2048 |
| `--max-thinking-tokens N` | 超预算仍未闭合 thinking 时注入 `</think>` | 关闭 |

dLM remote code 要求生成预算能整除 block；为与项目原生 `evaluate.py` 保持一致，Linear SS 也采用相同取整。server 会向上取整内部预算，然后最多只返回用户请求的 `--tokens`；stats 同时记录 `requested_tokens`、`generation_budget` 和 `raw_generated_tokens`。

### 7.3 模型、GPU 与环境

| 参数 | 含义 | 默认 |
|---|---|---|
| `--model PATH` | 模型目录/HF id | 本地 8B 目录 |
| `--served-model-name NAME` | OpenAI API 标签，不是权重路径 | `nemotron-labs-diffusion-8b` |
| `--lora-path DIR` | LinearSpec adapter | `<model>/linear_spec_lora` |
| `--gpu-device ID` | 一个物理 GPU | 0 |
| `--gpu-devices ID` | 单 GPU 兼容别名 | 0 |
| `--gpu-memory-reserve-gb V` | 模型加载前真实占位 GiB | 0 |
| `--dtype` | bfloat16、float16、float32 | bfloat16 |
| `--pytorch-python PATH` | server Python | `nld_sglang/bin/python` |
| `--eval-python PATH` | NeMo-Skills Python | 同 server Python |
| `--nemo-skills-data-dir DIR` | 持久数据缓存 | `/data1/linyewei/datasets/NLD` |
| `--google-research-dir DIR` | IFEval scorer checkout | 数据目录下 `google-research` |

### 7.4 Thinking 与输出

| 参数 | 含义 |
|---|---|
| `--enable-thinking` | chat template 生成 `<think>` 开头 |
| `--disable-thinking` | 显式采用默认的 non-thinking template |
| `--keep-thinking` | NeMo 输出/评分流程保留 thinking |
| `--strip-thinking` | NeMo 去除 thinking 并在支持时重新评分 |
| `--output-path DIR` | 最终紧凑结果根目录 |
| `--port N` | server 端口；默认端口冲突时自动选择，显式端口冲突时报错 |
| `--keep-runtime` | 不删除隐藏工作目录 |
| `--dry-run` | 不创建目录、不加载模型、不运行评测 |

`--enable-thinking` 与 `--disable-thinking` 互斥；`--keep-thinking` 与 `--strip-thinking` 互斥。

## 8. 结果目录

默认：

```text
/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_YYYYMMDD_HHMMSS/
├── Settings.json
├── metrics_gsm8k.json
├── metrics_human-eval.json
├── ...
└── error_<benchmark>.json            # 仅失败项
```

每个 benchmark 完成后立即写紧凑结果，单项失败不会阻止后续任务。

`--keep-runtime` 或存在失败时保留：

```text
/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/.eval_TIMESTAMP_work_PID/
├── pytorch_runtime/
│   ├── pytorch_server.log
│   ├── pytorch_request_stats.jsonl
│   ├── gpu_memory_reserver.log
│   └── gpu_memory_reserver_ready.json
└── results/eval-results/<benchmark>/
    ├── output*.jsonl
    ├── metrics.json
    ├── pytorch_native_metrics_summary.json
    ├── pytorch_request_stats.jsonl
    └── pytorch_benchmark.log
```

无失败且未传 `--keep-runtime` 时，隐藏工作目录会被删除，只保留最终紧凑文件。

## 9. 指标字段

NeMo-Skills 原有 accuracy/scorer 字段保持不变，新增：

```text
pytorch_native.backend
pytorch_native.benchmark
pytorch_native.decode.*
average_nfe
tokens_per_forward_pass / tpf
model_output_tokens_per_s / tps
```

核心口径：

| 字段 | 公式/含义 |
|---|---|
| `completion_tokens` | 最终返回给 NeMo 的 token 数 |
| `raw_generated_tokens` | 原生方法实际产生、尚未按 API 上限/stop 截断的 token 数 |
| `forward_passes` / `decode_forward_passes` | decode 阶段 NFE 之和；LinearSpec 的 prompt prefill 已排除，与 SGLang 口径一致 |
| `prefill_forward_passes` | 被排除的 LinearSpec prompt prefill 次数 |
| `total_forward_passes` | remote code 返回的总 NFE；保留用于端到端审计 |
| `tokens_per_forward_pass` | completion tokens / decode NFE，即默认 TPF |
| `end_to_end_tokens_per_forward_pass` | completion tokens / total NFE，保留原含 prefill 口径 |
| `model_generation_time_s` | CUDA synchronize 包围的原生生成函数时间之和 |
| `model_output_tokens_per_s` | completion tokens / model generation time；主要 TPS |
| `benchmark_wall_output_tokens_per_s` | completion tokens / 完整 NeMo benchmark 命令时间 |
| `queue_wait_s` | 请求进入 server 到取得模型锁的时间 |
| `request_time_s` | 请求进入 server 到模型生成和文本解码完成的时间 |

本链路不伪造真实流式 TTFT/TPOT：原生 remote code 一次返回完整序列，无法提供 SGLang 式逐 token streaming timing。

## 10. 与 SGLang 做公平比较

至少固定：

- 同一模型和 adapter；
- 同一 NeMo-Skills 版本、数据、prompt config 和 scorer；
- 同一 benchmark spec 与 `--max-samples`；
- 同一 temperature、tokens、thinking 设置；
- dLM/Linear 对齐 block 和 threshold 能力；
- 同一 GPU 型号、dtype 和显存占位；
- client concurrency 1，除非专门研究 serving。

推荐对照命令：

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --tokens 8192 --block-length 16 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/pytorch_compare
```

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 3 --gpu-memory-reserve-gb 30 --tokens 8192 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/sglang_compare
```

准确率可以直接对照；当前 PyTorch 与 SGLang 的 LinearSpec TPF 都排除 prompt prefill，但仍要确认两套 remote/engine 是否同样统计 post-block KV update；TPS 则代表各自真实后端时间，正是本实验希望比较的内容。

## 11. 并行运行与隔离保证

新实验：

- 使用独立顶层入口和 Python server；
- 默认端口 32000，与 SGLang 30000/代理 31000 分离；
- 默认写 `results/pytorch_nemo_eval_results`；
- 每轮使用时间戳最终目录和 PID 隐藏工作目录；
- 不读取或清空 SGLang stats/trace；
- 不杀 SGLang、旧 worker 或诊断进程；
- 仅共享只读模型、LoRA 和已准备的数据；
- 数据未准备时使用 cache/NeMo prepare，使用 `.prepare.lock` 减少本链路间的准备竞争。

并行示例应在两个终端分别运行，以下每条仍是完整单行命令：

```bash
bash observations/eval_sglang.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --gpu-memory-reserve-gb 30 --block-size 16 --output-path /data/home/wly/dLLM/NLD_results/observations/sglang_nemo_eval_results/parallel_sglang
```

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 1 --gpu-memory-reserve-gb 30 --block-length 16 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/parallel_pytorch
```

不要把两条命令指向同一 GPU，也不要人为指定同一个端口。

## 12. 当前限制

- 只支持单 GPU、单模型副本；没有 TP。
- GPU 生成串行，不做 dynamic/continuous batching。
- 当前原生函数只暴露 temperature，`top_p` 会记录但不应用；temperature 0 时不影响贪心对照。
- dLM 是模型原生 `generate()`，不是 SGLang FastDiffuser 的逐请求 scheduler 实现。
- LinearSpec 原生函数要求 batch size 1；server 串行保证这一点。
- 不报告真实 TTFT/TPOT。
- 原生 NFE 和 SGLang forward stats 的边界可能不同，尤其 post-block KV update。
- 新 pipeline 不自动修改 conda site-packages；当前环境需已能运行现有 NeMo-Skills pipeline。

## 13. 排错顺序

1. 先运行 `--dry-run` 检查 mode、GPU、reserve、block、context、端口和输出路径。
2. 检查 `nvidia-smi`，确认“剩余显存”能同时容纳 reserve、模型与 KV。
3. 传 `--keep-runtime --max-samples 2 --tokens 512` 做最小 smoke。
4. server 启动失败看 `pytorch_server.log`；显存占位失败看 `gpu_memory_reserver.log`。
5. NeMo 失败看 `pytorch_benchmark.log` 和最终 `error_<benchmark>.json`。
6. accuracy 有但 TPS/TPF 缺失时，检查 `pytorch_request_stats.jsonl` 和 `pytorch_native_metrics_summary.json`。
7. GPQA context 错误时增大 `--context-length` 或减小 `--tokens`。
8. IFEval 失败时检查 google-research 路径及 `langdetect/immutabledict/nltk`。

## 14. 真实 GPU 验证记录

2026-08-04 11:37–11:42 CST 在 GPU 0（A100 80GB）和 `nld_sglang` 环境完成 GSM8K 单样本端到端验证。四种模式都实际加载了 8B BF16 权重，并依次通过原生生成、NeMo-Skills prompt/scorer、请求统计和指标合并。

| mode | tokens | NFE | TPF | 原生 TPS | 请求失败/错误文件 |
|---|---:|---:|---:|---:|---:|
| `ar` | 32 | 32 | 1.0000 | 17.1233 | 0 / 0 |
| `dlm` | 32 | 27 | 1.1852 | 12.4272 | 0 / 0 |
| `linearspec_base` | 32 | 11 | 2.9091 | 30.2864 | 0 / 0 |
| `linearspec_lora` | 128 | 31 | 4.1290 | 56.7242 | 0 / 0 |

LinearSpec + LoRA 测试同时实际启用了 `--gpu-memory-reserve-gb 1`；ready 文件确认在物理 GPU 0 分配了 1.0 GiB，pipeline 结束后占位进程、模型 server 和端口 32000 均已释放。

结果位置：

```text
/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_113706
results/pytorch_nemo_eval_real_smoke/ar/eval_20260804_113838
results/pytorch_nemo_eval_real_smoke/dlm/eval_20260804_113939
results/pytorch_nemo_eval_real_smoke/linearspec_base/eval_20260804_114039
```

这些 smoke 使用 32/128 token 小预算，只用于验证链路，生成在输出答案前被截断，因此 GSM8K accuracy 为 0；不能把该 accuracy 当作模型正式精度。

## 15. 与指定 SGLang 十数据集实验对齐的可直接运行命令

下面这条命令与以下 SGLang 实验在 benchmark 顺序、单请求并发、显存占位、LinearSpec + LoRA、block size、生成长度、上下文长度和 non-thinking 设置上对齐；输出根目录可通过 `--output-path` 自行替换。

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 0 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 40 --block-size 8 --threshold 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results
```

对应关系：

| SGLang 参数/行为 | PyTorch 命令 | 说明 |
|---|---|---|
| `--mode linearspec_lora` | 相同 | 加载 bundled LoRA，并仅在原生 LinearSpec draft 阶段启用 adapter |
| `--gpu-devices 2` | 相同写法可用 | PyTorch 后端只允许一个 GPU ID；也可写成 `--gpu-device 2` |
| `--batch-size 1` | 原生固定 batch size 1 | PyTorch 入口没有 server batch 参数，模型执行始终串行 |
| `--client-concurrency 1` | 相同 | NeMo-Skills 同时只发送一个请求 |
| — | `--num-chunks 1` | 显式固定 NeMo 客户端 chunk 数，避免随并发默认值变化 |
| `--gpu-memory-reserve-gb 40` | 相同 | 模型加载前在 GPU 2 实际占位 40 GiB |
| `--block-size 16` | 相同写法可用 | `--block-size` 是 PyTorch `--block-length` 的别名 |
| LinearSpec 默认行为 | `--threshold 0` | 显式固定原生 LinearSpec draft 一轮填满 mask 的配置 |
| SGLang 默认 `--tokens 8192` | `--tokens 8192` | 每题最多返回 8192 completion tokens |
| SGLang 自动 context 10240 | `--context-length 10240` | 对齐 `tokens + 2048`；约束 prompt + 内部生成预算 |
| non-thinking | `--disable-thinking` | 明确使用 non-thinking chat template/评测设置 |
| `--output-path ...` | 相同 | 每次仍会在指定根目录下创建独立 `eval_时间戳` 目录 |

如果要启用 thinking，将上面命令中的 `--disable-thinking` 替换成 `--enable-thinking --keep-thinking`。例如：

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_lora --benchmarks human-eval:1,mbpp:1,livecodebench-cpp:1,gsm8k:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1 --gpu-devices 2 --client-concurrency 1 --num-chunks 1 --gpu-memory-reserve-gb 40 --block-size 16 --threshold 0 --tokens 8192 --context-length 10240 --enable-thinking --keep-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/thinking
```

常用的可替换配置只有以下几项：

- 改结果目录：替换 `--output-path PATH`；
- 改生成长度：替换 `--tokens N`，同时保证 `--context-length` 大于最大 prompt token 数与内部生成预算之和；
- 改上下文：替换 `--context-length N`；
- 关闭 thinking：使用 `--disable-thinking`；
- 开启并保留 thinking：使用 `--enable-thinking --keep-thinking`；
- 开启但评分前剥离 thinking：使用 `--enable-thinking --strip-thinking`；
- 限制 smoke 样本数：追加 `--max-samples N`；正式全量评测不要传该参数。

注意：`--gpu-memory-reserve-gb 40` 会真实占用 40 GiB，再加载约 16 GiB 的 BF16 权重及运行时状态。运行前应确认 GPU 2 有足够空闲显存；它与 SGLang 命令中的显存占位语义相同，但 PyTorch 后端不提供 SGLang 的 continuous batching、`--mem-fraction` 或 tensor parallel。
