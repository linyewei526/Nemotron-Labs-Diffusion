# NLD 原生 PyTorch + NeMo-Skills Confidence-Overlap LinearSpec 实验手册

> 实现目录：`method/confidence_overlap_linearspec/`
>
> 入口：`method/confidence_overlap_linearspec/eval_confidence_overlap.sh`
>
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/`
>
> 推荐环境：`conda activate nld_sglang`

## 1. 实验目标

该实验在原生 PyTorch Linear Self-Speculation 上实现 confidence-guided verify/draft overlap。它不修改模型权重目录、模型 remote code、旧 PyTorch/SGLang 入口或旧结果目录，可以和此前实验使用不同 GPU、独立进程、独立临时目录及操作系统原子分配的端口并行运行。

设当前正常 draft block 长度为 `L`，位置 0 是已经由上一轮 AR verifier 给出的 seed，位置 `1..L-1` 是 dLLM 并行预测的 token。对位置 `i` 的已选 token 置信度记为 `C_i`，其左侧可评分 draft token 的平均置信度记为 `C_imean`：

```text
token_y_drop_pct(i) = 1 - C_i / C_imean
```

程序从左向右寻找第一个严格满足 `token_y_drop_pct > drop_pct_threshold` 的位置 `p`，默认阈值是 0.15。对该位置保存 dLLM logits 中除原第一候选 A、MASK 和 EOS 后的最高分 token B。

若存在候选，verify forward 使用一个两行、统一长度为 `p+L` 的 batch：

```text
row 0 = [原 draft 的 L 个 token] + [padding MASK × p]
row 1 = [原 draft 的前 p 个 token] + [B] + [MASK × (L-1)]
```

- row 0 使用 causal attention，完成原 draft 的 AR 验证；尾部 padding 的输出完全丢弃。
- row 1 的前 `p` 个 token 使用 causal attention、关闭 LoRA，用于在旧 canonical cache 上重建命中 B 之前的 prefix。
- row 1 从 B 开始的完整 `L` 个位置使用 bidirectional attention；`overlap_lora` 模式只在这一段打开 LoRA，因此得到下一轮完整的 `[B] + draft(L-1)`，而不是只得到本轮剩余后缀。
- 全模型所有 `diffusion_lm` 开关在 fused forward 中保持关闭，行间差异完全由显式 4D attention mask 表达，避免全局模式开关无法按 batch 行分流的问题。
- 只有 row 0 的 causal verifier 有权提交输出和 canonical KV cache；row 1 永远不能直接改变输出。

命中条件考虑 LinearSpec 的一位 shift：verifier 必须先接受 B 之前的全部 draft token，并且在位置 `p-1` 输出 B。若 verifier 更早拒绝、接受原 A、或在该位置输出既非 A 也非 B 的 token，row 1 全部丢弃。命中后 row 1 作为下一轮完整 draft，省掉一次正常 dLLM draft forward，并可继续在其中寻找下一处 confidence drop。

以下情况恢复普通 draft + verify 或丢弃 prospective 结果：

- 本轮没有严格超过阈值的 token；
- 没有合法第二候选；
- 已经没有下一轮生成预算；
- `p+L` 会超过模型位置上限；
- thinking budget 将强制改变下一轮 seed；
- verifier 未按上述条件命中 B；
- verifier 输出 EOS。

## 2. 隔离性与文件职责

| 文件 | 职责 |
|---|---|
| `method/confidence_overlap_linearspec/eval_confidence_overlap.sh` | 用户入口、参数校验、自动 GPU、时间戳目录、立即写 Settings、启动和收尾 |
| `method/confidence_overlap_linearspec/run_pipeline.sh` | 独立 server、NeMo-Skills 数据与评分、逐 benchmark 合并结果 |
| `method/confidence_overlap_linearspec/server.py` | 独立 OpenAI-compatible 原生 PyTorch 服务；请求级模型执行串行化 |
| `method/confidence_overlap_linearspec/generation.py` | confidence 检测、normal/fused forward、verifier-only 提交状态机、NFE 统计 |
| `method/confidence_overlap_linearspec/hybrid.py` | 两行混合 4D attention mask 和独立 DynamicCache repeat/select/crop |
| `method/confidence_overlap_linearspec/segmented_lora.py` | 从 safetensors 加载 `o_proj` LoRA，并按 batch/token 位置路由 |
| `method/confidence_overlap_linearspec/select_gpu.py` | 按空闲显存和利用率自动选择物理 GPU |
| `method/confidence_overlap_linearspec/merge_metrics.py` | 合并 accuracy、物理 NFE、TPF、TPS 与 overlap 命中统计 |
| `method/confidence_overlap_linearspec/update_settings.py` | 原子更新 Settings 中的实际端口、GPU 和运行状态 |
| `method/confidence_overlap_linearspec/tests/test_core.py` | hybrid mask、分段 LoRA、confidence、shift 和 KV cache 单元测试 |

旧实验不会 import 这些文件。本实验不会改写 `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`，也不调用 PEFT 的全局 adapter 开关。每个任务使用：

- `/data/home/wly/dLLM/NLD_results/confidence_overlap_linearspec_<时间戳>/` 独立结果目录；
- 带 PID 的隐藏工作目录；
- 独立 server/request stats/PID；
- `--port 0` 时由操作系统先 bind 再发布实际端口，不使用存在竞态的“先探测、后监听”；
- 数据准备阶段的共享文件锁。

## 3. 环境和自检

激活环境：

```bash
conda activate nld_sglang
```

检查关键依赖：

```bash
python -c "import torch,transformers,safetensors,fastapi,uvicorn,nemo_skills; print(torch.__version__,transformers.__version__,nemo_skills.__version__)"
```

检查模型和 LoRA：

```bash
ls -lh /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/model.safetensors /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors
```

运行方法级测试：

```bash
python -m unittest method.confidence_overlap_linearspec.tests.test_core -v
```

只解析参数、不选 GPU、不加载模型、不创建目录：

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device auto --block-size 16 --drop-pct-threshold 0.15 --tokens 512 --max-samples 2 --dry-run
```

## 4. 推荐命令

本节所有命令均为单行。

### 4.1 支持子集评分的全链路 smoke

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 512 --max-samples 2 --keep-runtime
```

### 4.2 完整 HumanEval（本实验最终验收命令）

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks human-eval:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

当前环境的 NeMo-Skills/EvalPlus scorer 要求 HumanEval 和 MBPP 输入包含完整题集。因此这两个任务不能和 `--max-samples` 或 `--quick-test` 一起使用，入口会在加载模型前直接报错。`human-eval:1` 中的 `:1` 表示 pass@1，而不是只运行一题。

### 4.3 base 权重消融

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_base --benchmarks human-eval:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 4.4 单数据集与多数据集

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device 1 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1,human-eval:1 --gpu-device 1 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 4.5 默认十项 benchmark

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --gpu-device 1 --gpu-memory-reserve-gb 40 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

默认十项是：

```text
gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1
```

### 4.6 指定 GPU 或限制自动候选

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device 2 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-candidates 1,3 --gpu-min-free-gb 28 --gpu-wait-seconds 1800 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 4.7 预留显存

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device 1 --gpu-memory-reserve-gb 20 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

`--gpu-memory-reserve-gb` 会先在选定 GPU 启动独立进程并真实持有指定 GiB，再加载模型；退出和异常时由 trap 释放。它用于研究受限可用显存，不等于给模型“保留”显存。NLD-8B BF16 与 FP32 LoRA 在最终 HumanEval 中单请求观测最大值约 20.53 GiB；长 context、不同 block 和 fused batch 会改变 KV/activation 开销，因此不要把该观测值当作所有任务的硬上限。

### 4.8 显式端口、结果目录和保留日志

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device 1 --port 19081 --output-path /data/home/wly/dLLM/NLD_results --block-length 16 --drop-pct-threshold 0.15 --tokens 512 --max-samples 2 --keep-runtime
```

并行实验推荐保持默认 `--port 0`。显式端口若已占用，server 的原子 bind 会失败而不会误连到其他实验。

### 4.9 thinking budget

```bash
bash method/confidence_overlap_linearspec/eval_confidence_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --gpu-device auto --enable-thinking --max-thinking-tokens 6000 --keep-thinking --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

若候选 B 所在位置会触发 thinking seed 强制替换，程序不启动该 prospective 分支；若状态在 verify 后才触发，已经计算的 prospective 结果也不会提交。

## 5. 全部入口参数

### 5.1 模式、模型与 benchmark

| 参数 | 含义 | 默认 |
|---|---|---|
| `--mode overlap_lora` | draft 使用 bundled LoRA；prefill/verify 关闭；fused 时只路由 prospective suffix | 必填模式之一 |
| `--mode overlap_base` | 所有阶段只使用 base 权重 | 必填模式之一 |
| `--model PATH` | 本地/HF 模型目录 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B` |
| `--served-model-name NAME` | OpenAI API 暴露的模型名 | 方法专用名字 |
| `--lora-path DIR` | `overlap_lora` 的 adapter 目录 | `<model>/linear_spec_lora` |
| `--benchmarks LIST` | 逗号分隔的 NeMo benchmark spec，支持单项和多项 | 正式十项 |
| `--tokens N` | 每条请求最多返回的 completion token | 8192 |
| `--max-samples N` | 每项只运行前 N 道；HumanEval/MBPP 禁用 | 全量 |
| `--quick-test` | NeMo quick test；HumanEval/MBPP 禁用 | 关闭 |
| `--num-chunks N` | NeMo 客户端数据 chunk 数 | client concurrency |
| `--client-concurrency N` | 并发 HTTP 请求数 | 1 |

模型的 attention/LoRA 路由状态是进程级状态，server 用锁串行执行 GPU generation。`--client-concurrency > 1` 会产生排队，不等于 SGLang continuous batching。

### 5.2 解码参数

| 参数 | 含义 | 默认/限制 |
|---|---|---|
| `--block-length N` / `--block-size N` | 正常 draft、verify 和 prospective draft 的 L | 16，至少 2 |
| `--threshold V` | dLLM 一轮内的 unmask 阈值 | 当前必须为 0.0 |
| `--drop-pct-threshold V` | 第一处 `1-C_i/C_imean > V` 的阈值 | 0.15，范围 `[0,1)` |
| `--temperature V` | greedy/sampling 温度 | 当前必须为 0 |
| `--top-p V` | 为协议对齐而记录；当前原生方法不应用 | 0.95 |
| `--context-length N` | server 接受的 prompt+生成预算上限 | 默认 `tokens+2048` |
| `--max-thinking-tokens N` | 超预算后按模型原链路强制 `</think>` seed | 空 |
| `--enable-thinking` | chat template 开启 thinking | 关闭 |
| `--disable-thinking` | 明确传递 NeMo 的 disable-thinking 行为 | 关闭 |
| `--keep-thinking` | NeMo 输出保留 thinking | 关闭 |
| `--strip-thinking` | 支持的任务中剥离 thinking 后重评分 | 关闭 |

这版实验把 temperature/threshold 固定为 0，是为了保持 deterministic verifier-only 等价性并避免多步 dLLM unmask 时“每个 token 应使用哪一次提交 logits”的额外变量。后续扩展这两个参数应另建实验版本。

### 5.3 GPU、运行时和输出

| 参数 | 含义 | 默认 |
|---|---|---|
| `--gpu-device ID` | 指定一个物理 GPU | `auto` |
| `--gpu-devices ID` | `--gpu-device` 的兼容别名；不接受多 ID | `auto` |
| `--gpu-min-free-gb V` | auto 选择要求的最低空闲显存 | 24 |
| `--gpu-candidates LIST` | auto 仅考虑这些物理 GPU | 所有 GPU |
| `--gpu-wait-seconds N` | 暂无满足条件 GPU 时最长等待时间 | 0 |
| `--gpu-memory-reserve-gb V` | 模型加载前由独立进程真实占用的显存 | 0 |
| `--dtype DTYPE` | `bfloat16`、`float16` 或 `float32` 及别名 | bfloat16 |
| `--port N` | 0 为 OS 原子分配；也可显式指定 | 0 |
| `--output-path DIR` | 时间戳结果根目录 | `/data/home/wly/dLLM/NLD_results` |
| `--pytorch-python PATH` | server Python | `nld_sglang` 的 Python |
| `--eval-python PATH` | NeMo-Skills Python | 与 PyTorch Python 相同 |
| `--nemo-skills-data-dir DIR` | 持久数据和 cache | `/data1/linyewei/datasets/NLD` |
| `--google-research-dir DIR` | IFEval checkout | `<data-dir>/google-research` |
| `--keep-runtime` | 保留隐藏工作目录、server log 和全部中间文件 | 关闭 |
| `--dry-run` | 只解析和展示；不创建目录、不加载模型 | 关闭 |

judge benchmark 的 `--judge-model`、`--judge-server-address`、`--judge-server-type`、`--judge-concurrency`、`--mt-bench-max-tokens`、`--alpaca-eval-max-tokens` 和 `--skip-judge-api-key-check` 与旧 PyTorch+NeMo-Skills 入口含义一致。

## 6. 结果目录与 Settings

成功运行的紧凑目录结构如下：

```text
/data/home/wly/dLLM/NLD_results/confidence_overlap_linearspec_YYYYMMDD_HHMMSS/
├── Settings.json
├── metrics_<benchmark>.json
└── artifacts/<benchmark>/
    ├── output-rs0.jsonl
    ├── pytorch_request_stats.jsonl
    ├── pytorch_confidence_overlap_metrics_summary.json
    └── pytorch_benchmark.log
```

结果子目录用原子 `mkdir` 创建；同一秒并行启动时自动追加 `_01`、`_02`。`Settings.json` 在目录创建后立即写入，初始状态为 `initialized`，随后记录：

- 原始命令行和完整解析参数；
- 模型、LoRA、block、confidence 阈值和 verifier-only 语义；
- 请求端口与 server 实际绑定端口；
- 自动选择后的物理 GPU；
- 数据、工作目录、Python 路径；
- `server_ready`、`completed` 或失败状态。

默认成功后会删除带 PID 的隐藏工作目录，因为必要原始产物已复制到 `artifacts/`；传 `--keep-runtime` 或任务失败时会保留它。

## 7. 指标解释

`metrics_<benchmark>.json` 保留 NeMo-Skills accuracy，并增加 `pytorch_confidence_overlap.decode`。核心字段：

| 字段 | 含义 |
|---|---|
| `forward_passes` / `decode_forward_passes` | decode 阶段真实 encoder 调用次数；排除一次 prompt prefill，两行 fused batch 仍算一次物理 forward |
| `total_forward_passes` / `physical_nfe` | 包含 prompt prefill 的全部真实 encoder 调用次数，用于内部一致性审计 |
| `tokens_per_forward_pass` / `tpf` | API 返回 completion token 总数除以 decode NFE，与 SGLang 口径一致 |
| `model_output_tokens_per_s` / `tps` | completion token 除以 CUDA 同步的模型生成时间；是参考 TPS |
| `processed_rows` | 所有 forward 实际处理的 batch row 总数，暴露 fused batch 的额外计算 |
| `processed_query_tokens` | 所有 forward 的 `batch × query length` 总和，不把 overlap 计算隐藏在 NFE 中 |
| `normal_draft_forwards` | 未复用 prospective 时执行的普通 dLLM draft 次数 |
| `normal_verify_forwards` | 无可执行候选时的单行 causal verify 次数 |
| `fused_verify_draft_forwards` | 两行 verify+prospective 物理 forward 次数 |
| `prefetch_attempts` | 实际构造并计算 prospective 的候选数 |
| `prefetch_verified_hits` | verifier 在识别位置确认 B 的次数 |
| `prefetch_hits` | 确认 B 且 prospective 可交给下一轮的次数 |
| `prefetch_saved_draft_forwards` | 下一轮真实消费 prospective、实际省下普通 draft forward 的次数 |
| `prefetch_discarded_before_candidate` | verifier 在 p 之前已经拒绝 |
| `prefetch_discarded_candidate_accepted` | verifier 接受 A 并继续，或整个 block 无该处拒绝 |
| `prefetch_discarded_wrong_b` | verifier 在 p 拒绝 A，但正确 token 不是 B |
| `prefetch_discarded_eos` | B 被确认但本轮已命中 EOS |
| `prefetch_skipped_thinking_budget` | 已知下一轮 seed 会被 thinking budget 改写，因此不做 prospective |
| `prefetch_hit_rate` | `prefetch_hits / prefetch_attempts` |
| `saved_draft_fraction_of_rounds` | `prefetch_saved_draft_forwards / rounds` |
| `average_candidate_position` | 候选 block 位置 p 的均值；位置 0 是 seed |

TPF 是本实验优先指标，但不能单独代表算力成本：fused forward 的 batch/序列更大，因此应同时报告 `processed_rows`、`processed_query_tokens` 和参考 TPS。

## 8. 已完成的实现级验证

开发时已执行以下检查：

- shell `bash -n` 与全部 Python `py_compile`；
- 6 项方法级单元测试；
- 真实 NLD-8B BF16 + bundled LoRA 的 AR token 等价性检查，包括实际命中并复用 prospective draft 的序列；
- 真实 fused branch 检查；
- GSM8K 单样本完整 NeMo-Skills 链路，包含 6 次 prospective 命中与复用；
- 最终完整 HumanEval 164 题、请求统计、EvalPlus、指标合并与运行时清理。

## 9. 本次完整 HumanEval 验收结果

结果目录：

```text
/data/home/wly/dLLM/NLD_results/confidence_overlap_linearspec_20260816_224201/
```

关键结果：

| 项目 | 数值 |
|---|---:|
| 完成/失败请求 | 164 / 0 |
| HumanEval base pass@1 | 77.44% |
| HumanEval+ pass@1 | 74.39% |
| 返回 completion tokens | 85,936 |
| 物理 NFE | 15,339 |
| TPF | 5.6025 |
| 模型侧参考 TPS | 198.9923 |
| prospective attempts | 4,693 |
| prospective hits / saved draft forwards | 1,071 / 1,071 |
| prospective hit rate | 22.8212% |
| saved draft fraction of rounds | 13.1848% |
| 实际 GPU / 端口 | GPU 1 / 43147 |

该段为旧 smoke 的历史记录；当时使用含 prefill 的 TPF。当前重新运行时验收关系为 `TPF=completion_tokens/decode_forward_passes`，同时保留 `total_forward_passes/physical_nfe` 做总调用次数审计；其余输出、进程和产物检查不变。
