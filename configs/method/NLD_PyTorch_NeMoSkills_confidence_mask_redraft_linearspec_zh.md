# NLD 原生 PyTorch + NeMo-Skills 自主全 MASK Redraft LinearSpec 实验手册

> 实现目录：`method/confidence_mask_redraft_linearspec/`
>
> 实验入口：`method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh`
>
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/`
>
> 推荐环境：`conda activate nld_sglang`

## 1. 实验目标与旧实验的区别

本实验是一个独立的新解码变体，不修改模型权重目录、模型 remote code、旧 PyTorch/SGLang 入口、`method/confidence_overlap_linearspec/` 固定第二候选实验或旧结果。不同实验可通过独立 GPU、独立 PID 工作目录、独立服务进程和操作系统原子分配端口并行运行。

设当前待验证草稿长度为 `Q`，配置的完整 block 长度为 `L`。正常起草时 `Q=L`；成功复用 row 1 后，下一轮可能只保留长度小于 `L` 的后缀，因此实现原生支持 `2 <= Q <= L` 的可变长度草稿。

草稿位置 0 是上一轮 AR verifier 给出的、尚未提交的 seed。对位置 `i` 的已选 token 置信度记为 `C_i`，左侧可评分草稿 token 的均值记为 `C_imean`：

```text
token_y_drop_pct(i) = 1 - C_i / C_imean
```

程序忽略 seed 的置信度，从左向右寻找第一个严格满足 `token_y_drop_pct > drop_pct_threshold` 的位置 `p`。默认阈值为 0.15。MASK 被排除出 softmax 置信度分母；判断不再依赖固定第二候选 B，也不会把 MASK、EOS 或 thinking budget 当作候选修正 token。

若存在 `p`，一次 fused forward 构造两个长度为 `p+L` 的 batch row：

```text
row 0 = [当前 Q-token draft] + [padding MASK × (p+L-Q)]
row 1 = [draft[:p]] + [MASK × L]
```

- row 0 使用 causal attention，完成当前草稿的 AR 验证；`Q` 之后的 padding 输出全部丢弃。
- row 1 的 `draft[:p]` 使用 causal attention，重建触发位置之前的上下文。
- row 1 从位置 `p` 开始的 `L` 个 MASK 使用 bidirectional attention，自主重解码潜在错误位置及其后 `L-1` 个位置。
- `mask_redraft_lora` 只在 row 1 的全 MASK suffix 和普通 dLLM draft 上打开 LinearSpec LoRA；prefill、row 0 verifier、row 1 prefix 均关闭 LoRA。
- fused forward 中全模型 `diffusion_lm` 保持关闭，行内 attention 差异由显式 4D mask 表达，避免无法按 batch 行分流的全局开关。
- 只有 row 0 verifier 可以提交输出和 canonical KV cache；row 1 无论是否复用，都不能自行提交 token。

因此，固定 B 的旧实验：

```text
draft[:p] + [B] + [MASK × (L-1)]
```

在本实验中变为：

```text
draft[:p] + [MASK × L]
```

row 1 必须自行预测触发位置，不再被第二候选先验强制锚定。

## 2. 四种情况的统一复用规则

令 row 1 预测出的完整 `L` token 为 `R`。令 verifier 本轮提交 token 数为 `m=matched+1`：若 row 0 中途不一致，`m` 是第一个修正 token 的位置计数；若当前 `Q` token 草稿全部通过，`m=Q`，最后一个是 verifier bonus token。LinearSpec 的 logits/token 存在一位 shift，所以从触发位置到修正或 bonus 的可信 verifier token 是：

```text
target = ar_tokens[p-1:m]
```

只有同时满足下式才复用 row 1：

```text
m >= p  且  R[:m-p+1] == ar_tokens[p-1:m]
```

命中后保留：

```text
next_draft = R[m-p:]
```

保留后缀的首 token 必然是 verifier 已确认的本轮修正或 bonus token，因此可作为下一轮 seed；其长度为 `L-(m-p)`，可以小于 `L`。新 draft 重新计算 confidence drop，且 confidence 历史从新 seed 后重新开始，不跨旧草稿拼接。

该规则覆盖用户定义的四种情况：

1. verifier 在 `p` 前拒绝，或触发位置 verifier token 与 `R[0]` 不同：丢弃 row 1，下一轮正常起草。
2. verifier 恰在 `p` 修正，且修正 token 等于 `R[0]`：直接命中，完整保留 `R` 作为下一轮长度 `L` 的草稿。
3. 原 token A 在 `p` 正确、但 row 1 在 `p` 预测的不是 A：触发位置不一致，丢弃 row 1。
4. 原 token A 和 row 1 在 `p` 一致：继续比较到 row 0 的真实修正位置或整块 bonus；row 1 若更早不一致，或在真实修正位置不一致，则丢弃；若全段一致，保留从真实修正位置开始的 row 1 后缀。

EOS、generation budget 结束、thinking budget 强制替换 seed、位置上限不足等边界不会错误提交 row 1；相关结果分别记入 discard/skip 统计。

## 3. 文件职责与隔离机制

| 文件 | 职责 |
|---|---|
| `method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh` | 参数校验、自动 GPU、原子时间戳结果目录、立即写 Settings、启动及收尾 |
| `method/confidence_mask_redraft_linearspec/run_pipeline.sh` | 独立服务、NeMo-Skills 数据准备/生成/评分、指标合并和紧凑产物复制 |
| `method/confidence_mask_redraft_linearspec/server.py` | 独立 OpenAI-compatible 原生 PyTorch 服务和请求级统计 |
| `method/confidence_mask_redraft_linearspec/generation.py` | confidence 检测、普通/融合 forward、统一复用判定、可变 draft 状态机和 NFE 统计 |
| `method/confidence_mask_redraft_linearspec/hybrid.py` | 两行混合 4D attention mask、DynamicCache repeat/select/crop |
| `method/confidence_mask_redraft_linearspec/segmented_lora.py` | 加载 `o_proj` LoRA，以 FP32 adapter 数学按 batch/token 路由 |
| `method/confidence_mask_redraft_linearspec/select_gpu.py` | 根据空闲显存和利用率选择物理 GPU |
| `method/confidence_mask_redraft_linearspec/merge_metrics.py` | 合并 accuracy、物理 NFE、TPF、TPS、row/query 工作量与 redraft 统计 |
| `method/confidence_mask_redraft_linearspec/update_settings.py` | 原子更新 Settings 的实际端口、GPU 和状态 |
| `method/confidence_mask_redraft_linearspec/tests/` | attention、LoRA、shift、KV、四分支、可变草稿和指标单元测试 |

每轮使用独立资源：

- 结果目录：`/data/home/wly/dLLM/NLD_results/confidence_mask_redraft_linearspec_<时间戳>/`；
- 工作目录：结果根目录下带 PID 的隐藏目录；
- 独立 server log、request stats、PID 和方法专用 model name；
- `--port 0` 时由 server 先 bind socket 再发布实际端口，避免“探测空闲端口后再监听”的并发竞态；
- 同一秒启动时，结果目录自动追加 `_01`、`_02`；
- 共享数据准备使用文件锁；旧实验不会 import 本方法文件。

## 4. 环境和自检

激活环境：

```bash
conda activate nld_sglang
```

检查依赖：

```bash
python -c "import torch,transformers,safetensors,fastapi,uvicorn,nemo_skills; print(torch.__version__,transformers.__version__,nemo_skills.__version__)"
```

检查模型与 LoRA：

```bash
ls -lh /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/model.safetensors /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors
```

运行全部方法级测试：

```bash
python -m unittest discover -s method/confidence_mask_redraft_linearspec/tests -v
```

只解析参数，不选 GPU、不加载模型、不创建结果目录：

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --block-size 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --dry-run
```

## 5. 推荐命令

本节每条命令均为单行。

### 5.1 GSM8K 单样本全链路 smoke

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --keep-runtime
```

### 5.2 单数据集正式测评

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks human-eval:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

当前 NeMo-Skills/EvalPlus scorer 要求 HumanEval 和 MBPP 使用完整题集，因此不能将 `human-eval` 或 `mbpp` 与 `--max-samples`、`--quick-test` 组合。`human-eval:1` 的 `:1` 是 pass@1，不是只测一题。

### 5.3 多数据集正式测评

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1,math-500:1,human-eval:1 --gpu-device 2 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.4 默认十项 benchmark

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --gpu-device 3 --gpu-memory-reserve-gb 0 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

默认列表为：

```text
gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1
```

### 5.5 base 权重消融

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_base --benchmarks gsm8k:1 --gpu-device auto --gpu-min-free-gb 24 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.6 指定 GPU

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.7 限制自动 GPU 候选并等待

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-candidates 1,2,3 --gpu-min-free-gb 28 --gpu-wait-seconds 1800 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.8 预留显存实验

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 20 --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

`--gpu-memory-reserve-gb` 会在指定 GPU 上由独立进程真实持有显存，再加载模型；退出或异常时由 trap 释放。它用于模拟受限显存，不是替模型预留独占空间。长 context、fused 两行 batch 和 block length 都会影响实际峰值，设置前应保留安全余量。

### 5.9 显式端口、输出根目录和保留运行产物

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --port 19082 --output-path /data/home/wly/dLLM/NLD_results --block-length 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --keep-runtime
```

并行实验推荐使用默认 `--port 0`。显式端口已占用时，新 server 会原子 bind 失败，不会误连到旧实验服务。

### 5.10 thinking budget

```bash
bash method/confidence_mask_redraft_linearspec/eval_confidence_mask_redraft.sh --mode mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --enable-thinking --max-thinking-tokens 6000 --keep-thinking --block-length 16 --drop-pct-threshold 0.15 --tokens 8192
```

如果 verify 后必须强制插入 `</think>` seed，本轮已算出的 row 1 不会复用，确保输出只服从原 verifier/thinking 链路。

## 6. 全部入口参数

### 6.1 模式、模型和数据集

| 参数 | 含义 | 默认/限制 |
|---|---|---|
| `--mode mask_redraft_lora` | 普通 draft 和 row 1 全 MASK suffix 使用 LinearSpec LoRA | 必填模式之一 |
| `--mode mask_redraft_base` | 所有阶段只使用 base 权重 | 必填模式之一 |
| `--model PATH` | 本地或 HF 模型目录 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B` |
| `--served-model-name NAME` | 本地 OpenAI API 暴露的模型标签 | 方法专用标签 |
| `--lora-path DIR` | lora 模式的 adapter 目录 | `<model>/linear_spec_lora` |
| `--benchmarks LIST` | 逗号分隔的 benchmark spec，支持单项/多项 | 正式十项 |
| `--tokens N` | 每条请求最多返回的 completion token 数 | 8192 |
| `--max-samples N` | 每项只运行前 N 道；HumanEval/MBPP 禁用 | 全量 |
| `--quick-test` | NeMo-Skills quick-test；HumanEval/MBPP 禁用 | 关闭 |
| `--num-chunks N` | NeMo 客户端数据 chunk 数 | client concurrency |
| `--client-concurrency N` | 并发 HTTP 请求数 | 1 |
| `--math-prompt-config NAME` | 数学任务的 NeMo prompt_config 覆盖 | 空 |

模型 attention/LoRA 路由状态是进程级状态，server 用锁串行执行 GPU generation。`--client-concurrency > 1` 会产生服务端排队，不等于 SGLang continuous batching。

### 6.2 解码参数

| 参数 | 含义 | 默认/限制 |
|---|---|---|
| `--block-length N` / `--block-size N` | 完整正常 draft 和 row 1 的 `L` | 16，至少 2 |
| `--drop-pct-threshold V` | 第一处严格满足 `1-C_i/C_imean > V` 的触发阈值 | 0.15，范围 `[0,1)` |
| `--threshold V` | dLLM 一轮内 unmask threshold | 当前必须为 0.0 |
| `--temperature V` | greedy/sampling 温度 | 当前必须为 0 |
| `--top-p V` | 为协议对齐记录；原生模型方法当前不应用 | 0.95 |
| `--context-length N` | server 接受的 prompt+completion 上限 | 默认 `tokens+2048` |
| `--max-thinking-tokens N` | 超预算后按原链路强制 `</think>` seed | 空 |
| `--enable-thinking` | chat template 开启 thinking | 关闭 |
| `--disable-thinking` | 明确向 NeMo 传递 disable-thinking | 关闭 |
| `--keep-thinking` | 输出保留 thinking | 关闭 |
| `--strip-thinking` | 支持的任务中剥离 thinking 后重评分 | 关闭 |

本版固定 greedy 和一次性并行 unmask，是为了隔离研究自主重起草变量并保持 verifier-only 等价性。若要支持采样或多步 dLLM unmask，应另建实验变体。

### 6.3 GPU、运行时和输出

| 参数 | 含义 | 默认 |
|---|---|---|
| `--gpu-device ID` | 指定一个物理 GPU | `auto` |
| `--gpu-devices ID` | `--gpu-device` 兼容别名；不接受多个 ID | `auto` |
| `--gpu-min-free-gb V` | auto 选择所需最低空闲显存 | 24 |
| `--gpu-candidates LIST` | auto 仅考虑指定的物理 GPU 列表 | 全部 |
| `--gpu-wait-seconds N` | 无满足条件 GPU 时最长等待秒数 | 0 |
| `--gpu-memory-reserve-gb V` | 加载模型前由独立进程真实占用显存 | 0 |
| `--dtype DTYPE` | `bfloat16`、`float16`、`float32` 及别名 | bfloat16 |
| `--port N` | 0 为 OS 原子分配，也可显式指定 | 0 |
| `--output-path DIR` | 时间戳结果根目录 | `/data/home/wly/dLLM/NLD_results` |
| `--pytorch-python PATH` | 原生 server 使用的 Python | `nld_sglang` Python |
| `--eval-python PATH` | NeMo-Skills 使用的 Python | 与 PyTorch Python 相同 |
| `--nemo-skills-data-dir DIR` | 持久数据/cache 根目录 | `/data1/linyewei/datasets/NLD` |
| `--google-research-dir DIR` | IFEval google-research checkout | `<data-dir>/google-research` |
| `--keep-runtime` | 保留隐藏工作目录、server log 和中间文件 | 关闭 |
| `--dry-run` | 只解析打印，不创建目录或加载模型 | 关闭 |

Judge benchmark 的 `--judge-model`、`--judge-server-address`、`--judge-server-type`、`--judge-concurrency`、`--mt-bench-max-tokens`、`--alpaca-eval-max-tokens` 和 `--skip-judge-api-key-check` 与现有 PyTorch+NeMo-Skills 入口一致。

## 7. 结果目录与 Settings

成功运行的紧凑目录：

```text
/data/home/wly/dLLM/NLD_results/confidence_mask_redraft_linearspec_YYYYMMDD_HHMMSS/
├── Settings.json
├── metrics_<benchmark>.json
└── artifacts/<benchmark>/
    ├── output-rs0.jsonl
    ├── mask_redraft_request_stats.jsonl
    ├── pytorch_confidence_mask_redraft_metrics_summary.json
    └── pytorch_benchmark.log
```

入口在成功创建时间戳目录后立即写 `Settings.json`，初始状态为 `initialized`，其中记录：

- 原始单行命令和全部解析后参数；
- 模型、LoRA、block、confidence 阈值；
- `draft[:p] + MASK*L` 构造、可变长度复用规则和 verifier-only 权威；
- 请求端口、server 实际绑定端口、自动选择后的物理 GPU；
- Python、数据、工作路径；
- `server_ready`、`completed`、`completed_with_errors` 或 `failed` 状态。

默认成功后会删除隐藏工作目录，因为必要原始产物已复制到 `artifacts/`；传入 `--keep-runtime` 或运行失败时保留工作目录。

## 8. 指标解释

`metrics_<benchmark>.json` 保留 NeMo-Skills accuracy，并增加 `pytorch_confidence_mask_redraft.decode`。主要字段：

| 字段 | 含义 |
|---|---|
| `forward_passes` / `decode_forward_passes` | decode 阶段实际 encoder 调用次数；排除 prompt prefill，两行 fused batch 算 1 次物理 forward |
| `total_forward_passes` / `physical_nfe` | 包含 prompt prefill 的总 encoder 调用次数，用于内部一致性审计 |
| `tokens_per_forward_pass` / `tpf` | API completion token 数除以 decode NFE，与 SGLang 口径一致 |
| `model_output_tokens_per_s` / `tps` | completion token 除以 CUDA 同步的模型生成时间，仅作参考 TPS |
| `processed_rows` | 所有 forward 实际处理的 batch row 总数，显示 fused batch 的额外工作 |
| `processed_query_tokens` | 所有 forward 的 `batch × query length` 总和 |
| `rounds` | LinearSpec verify 轮数 |
| `normal_draft_forwards` | 未复用 row 1 时执行的普通 dLLM draft 次数 |
| `normal_verify_forwards` | 没有融合 prospective row 的 verifier 次数 |
| `fused_verify_redraft_forwards` / `redraft_attempts` | 同时执行 verifier 和自主 row 1 的次数 |
| `rounds_without_candidate` | 没有 confidence drop 触发位置的轮数 |
| `redraft_reuse_hits` | 可信 verifier 段完整匹配且实际保留 row 1 后缀的次数 |
| `redraft_hit_rate` | `redraft_reuse_hits / redraft_attempts` |
| `redraft_saved_draft_forwards` | 后续实际消费了 full/partial redraft、因此省掉普通 draft forward 的次数 |
| `redraft_direct_trigger_hits` | verifier 恰在 `p` 修正且 `R[0]` 命中的次数 |
| `redraft_downstream_correction_hits` | `p` 正确，row 1 又命中更靠后修正位置的次数 |
| `redraft_full_block_bonus_hits` | 当前草稿全通过且 row 1 命中 bonus token 的次数 |
| `full_length_reuses` / `partial_length_reuses` | 保留长度等于 `L` / 小于 `L` 的次数 |
| `average_draft_length` | 实际被验证的可变 `Q` 均值 |
| `average_retained_draft_length` | 成功复用时保留后缀长度均值 |
| `redraft_discarded_before_trigger` | verifier 在触发位置前已经修正 |
| `redraft_discarded_trigger_token_mismatch` | row 1 在触发位置预测与 verifier 不同 |
| `redraft_discarded_before_correction` | row 1 在真实修正位置之前先失配 |
| `redraft_discarded_correction_mismatch` | 前文一致，但 row 1 未命中真实修正/bonus token |
| `redraft_discarded_eos` / `redraft_discarded_thinking_budget` / `redraft_discarded_generation_end` | 虽满足匹配，但因边界条件不复用 |
| `redraft_skipped_no_future_round` / `redraft_skipped_context_limit` | 无下一轮预算或位置不足时未启动 row 1 |
| `peak_gpu_memory_gib` | 请求级 CUDA 峰值显存分布 |

`physical_nfe` 的下降不等于 FLOPs 或时延同比下降：fused forward 包含两行且 query 可能更长。因此本实验主要比较 TPF 时，应同时报告 `processed_rows`、`processed_query_tokens`、参考 TPS 和峰值显存。

## 9. 等价性与排错检查

本实验的目标是改变物理 forward 编排，不改变 verifier 决定的 greedy token 序列。推荐先做以下检查：

1. 相同 prompt、thinking 设置和 token budget 下，与 `model.ar_generate` 比较完整 token IDs；
2. 查看 request stats 中 `ok=true`、`nfe == mask_redraft.physical_nfe`；
3. 确认成功复用时 `redraft_saved_draft_forwards` 在下一轮实际消费后才递增；
4. 检查 `Settings.json` 的 `resolved_runtime.port`、`gpu_device` 和最终 `status`；
5. 若任务失败，使用 `--keep-runtime` 查看 `mask_redraft_server.log` 和 `pytorch_benchmark.log`；
6. 若 TPF 上升但 TPS 下降，结合 `processed_query_tokens` 判断是否是两行融合计算量导致，而不是 NFE 记账错误。

由于 canonical KV 只从 row 0 截取，row 1 只产生未提交 proposal；任何 row 1 不一致都会回退到普通 draft。这个约束是本方法保持“只信 verifier”输出等价性的核心。

## 10. 已完成的验收

2026-08-17 使用 8B BF16 模型、bundled LinearSpec LoRA、`L=16`、阈值 0.15 完成了两层验收：

- 真实模型等价性测试：同一个 GSM8K prompt 生成 131 token，与逐 token `ar_generate` 的完整 token IDs 完全一致；自主 redraft 路径实际触发并发生 full/partial 后缀复用。
- NeMo-Skills 全链路 smoke：GSM8K 单样本 `symbolic_correct=100%`，结果目录、Settings、自动 GPU、OS 原子端口、服务、评分、指标合并、紧凑产物和成功清理均通过。

本段 smoke 的物理 NFE=31、TPF=4.2258 是修改前包含 prefill 的历史口径；当前重新运行会另外报告 decode NFE，并用它计算默认 TPF。该次 smoke 的状态结论不变：20 轮中执行 19 次 fused 尝试并实际复用 10 次，包含直接触发命中 5 次、下游修正命中 4 次、整块 bonus 命中 1 次。5 次为完整长度复用，5 次为部分后缀复用，实际保留长度范围为 10 到 16。

验收产物：`/data/home/wly/dLLM/NLD_results/confidence_mask_redraft_linearspec_20260817_175440/`。这些数字只用于证明链路和分支确实运行，不代表完整 benchmark 的准确率或稳定性能结论。
