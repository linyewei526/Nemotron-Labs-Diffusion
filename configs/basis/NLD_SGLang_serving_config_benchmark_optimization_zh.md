# Nemotron-Labs-Diffusion 的 SGLang serving 配置、测试与 Linear SS 优化指南

本文面向的目标不是复现论文离线评测，而是在 SGLang 推理引擎下运行 Nemotron-Labs-Diffusion-8B，理解 `sglang_spark` 目录的服务启动链路，并围绕 Linear SS / LinearSpec 解码做 serving 场景下的效率实验与优化。

本文结合以下本地材料整理：

- 项目 README：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/README.md`
- SGLang Spark 指南：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/README.md`
- SGLang 启动封装脚本：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/launch_server.sh`
- SGLang Web smoke 页面：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/index.html`
- 本地模型权重：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`

路径更新：如果你希望把 SGLang 工作目录放在当前项目内，推荐使用：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion && mkdir -p sglang_dllm/src sglang_dllm/logs sglang_dllm/hf_cache sglang_dllm/bench_results
```

并在每个新终端设置：

```bash
export NLD_ROOT=/data/home/wly/dLLM/Nemotron-Labs-Diffusion && export NLD_SGLANG_WORK_DIR=${NLD_ROOT}/sglang_dllm
```

这时本文中旧式的 `~/sglang_dllm` 可以理解为 `${NLD_SGLANG_WORK_DIR}`。使用 `sglang_spark/launch_server.sh` 时还必须显式传入 `WORK_DIR=${NLD_SGLANG_WORK_DIR}`，因为脚本默认会去 `$HOME/sglang_dllm`。更完整的从零部署命令以 `configs/NLD_SGLang_zero_to_dev_benchmark_zh.md` 为准。

## 1. 先给结论：你的优化主线应该怎么走

如果你的项目需求是“在 SGLang 推理引擎下优化 Linear SS 解码效率，并关注 serving 场景下不同 request、不同并发度的推理效率”，建议主线如下：

1. 先用本项目原生 `evaluate.py` 或 `chat` smoke test 验证模型权重、tokenizer、remote code、LoRA 文件是否完整。
2. 然后进入 SGLang 服务链路，跑通 OpenAI-compatible `/v1/chat/completions` 接口。
3. 在 SGLang 下分别建立 `AR`、`FastDiffuser`、`LinearSpec-base`、`LinearSpec + LoRA` 的性能基线。
4. 只在 SGLang fork 的 DLLM / LinearSpec 代码路径上修改解码逻辑、调度逻辑、验收统计和并发 batching 逻辑。
5. 用 serving benchmark 对不同并发、输入长度、输出长度、请求速率做横向对比。

也就是说，`eval.sh` 那条 NeMo-Skills / SLURM 离线评测路径不是你这个目标的核心路径。它适合做集群批量任务和论文式 accuracy / efficiency 复现，但不能直接回答 SGLang serving 下的 TTFT、ITL、吞吐、排队、并发调度、LinearSpec 接受长度等问题。

## 2. 需要先区分三套运行链路

### 2.1 原生 Hugging Face / evaluate.py 链路

项目 README 里给的三种解码模式本质上是模型 remote code 里的 Python 生成方法：

- `ar`：调用 `model.ar_generate`
- `dlm`：调用 `model.generate`
- `linear_spec`：调用 `model.linear_spec_generate`

这条链路的优点是最接近模型仓库自带实现，适合检查模型文件、tokenizer、generation config、LoRA adapter 和解码语义是否正常。

这条链路的缺点是它不是 SGLang 服务引擎，不能真实反映 serving 里的 scheduler、batching、KV cache 管理、streaming、HTTP 并发、CUDA graph、FlashInfer attention backend 等因素。

所以它适合作为“模型可用性检查”和“语义参考”，不适合作为最终优化目标。

### 2.2 eval.sh / NeMo-Skills / SLURM 链路

`eval.sh` 更像是面向集群批量评测的任务提交封装。它关心的是用容器和 SLURM 把一批离线 benchmark 跑完，得到 accuracy 或离线 efficiency 指标。

这条链路的价值是：

- 批量、可复现地跑公开任务；
- 在多节点或集群环境下统一调度；
- 适合论文表格式对比。

但它不直接服务于你的当前目标，因为它不是 SGLang serving 的常驻服务，不模拟真实请求进入服务端后的排队、动态 batching、流式输出、不同并发度下的调度压力。

### 2.3 SGLang serving 链路

`sglang_spark` 目录封装的是 SGLang 服务端启动方式。服务起来后，客户端通过 OpenAI-compatible API 发请求：

- 健康检查：`GET /health`
- 对话生成：`POST /v1/chat/completions`

这里的 `serving` / `Web` / HTTP API 不表示你在调用远端黑盒服务。推荐理解成：

```text
你本地启动一个 SGLang server 进程
客户端用 curl、benchmark 脚本或浏览器页面给本机端口发请求
server 进程在本机 GPU 上执行你本地 SGLang 源码里的解码逻辑
```

也就是说，SGLang server 只是把模型推理包装成 HTTP 服务。它不是不能修改的封闭引擎。只要你用源码 fork 启动，就可以修改本地 `~/sglang_dllm/src/sglang` 里的 scheduler、DLLM algorithm、LinearSpec decode 逻辑，然后重启服务验证效果。

这条链路才是你应该重点投入的路径，因为它能测到：

- 单请求延迟；
- TTFT，time to first token；
- ITL，inter-token latency；
- 总 tokens/s；
- request/s；
- 不同并发下的吞吐和尾延迟；
- LinearSpec draft + verify 每轮能接受多少 token；
- SGLang scheduler 对不同长度 request 的混合调度效果；
- FP8 / BF16、LoRA draft-only、CUDA graph batch size 等服务端参数的影响。

## 3. `sglang_spark` 目录代码组织

`sglang_spark` 目录主要包含三个重要文件。

### 3.1 `README.md`

这是面向 DGX Spark / aarch64 + Blackwell 的 SGLang 启动说明。它的默认配置是：

- SGLang fork：`hutm/sglang` 的 `upstream/2-dllm-lora-ar` 分支；
- 默认算法：`LinearSpec`；
- 默认 LoRA：LinearSpec draft 阶段使用 LoRA；
- 默认量化：`QUANT=fp8`；
- 默认模型路径：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`；
- 默认端口：`30000`；
- 默认最大运行请求数：`MAX_REQS=2`；
- 默认上下文长度：`CTX_LEN=2048`。

需要注意：这个 README 原始目标机器是 DGX Spark / Blackwell，而你当前机器是 A100 服务器。A100 上不要盲目照搬所有默认项，尤其是 FP8、FlashInfer wheel、Docker 镜像架构这三点，后文会单独说明。

### 3.2 `launch_server.sh`

这是最关键的启动脚本。它把“选择哪种解码算法”和“怎么把算法配置传给 SGLang”封装起来。

核心环境变量如下：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `WORK_DIR` | `$HOME/sglang_dllm` | SGLang fork、cache、log、LoRA 文件的工作目录 |
| `FORK_DIR` | `$WORK_DIR/src/sglang` | SGLang fork 本地路径 |
| `HF_CACHE` | `$WORK_DIR/hf_cache` | Hugging Face cache |
| `LOG_DIR` | `$WORK_DIR/logs` | 服务日志目录 |
| `LORA_HOST_DIR` | `$WORK_DIR/linear_spec_lora` | host 上的 LinearSpec LoRA adapter 目录 |
| `PORT` | `30000` | SGLang 服务端口 |
| `ALGO` | `LinearSpec` | 解码算法选择 |
| `MODEL` | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B` | 模型路径 |
| `CTX_LEN` | `2048` | SGLang context length |
| `MEM_FRAC` | `0.5` | 静态显存占比 |
| `MAX_REQS` | `2` | SGLang 最大同时运行 request 数 |
| `LORA_MODE` | `draft_only` | LoRA 应用范围 |
| `QUANT` | 空 | 量化方式；README 默认示例设置为 `fp8` |

脚本最终在容器里执行的 SGLang 启动命令等价于：

```bash
python3 -m sglang.launch_server --model-path ${MODEL} --trust-remote-code --tp-size 1 --mem-fraction-static ${MEM_FRAC} --max-running-requests ${MAX_REQS} --attention-backend flashinfer --dllm-algorithm ${ALGO} --cuda-graph-bs 1 2 3 4 --context-length ${CTX_LEN} --host 0.0.0.0 --port ${PORT}
```

根据 `ALGO` 和 `QUANT`，脚本还会追加：

- `--quantization fp8`
- `--json-model-override-args '{"ar_mode": true}'`
- `--dllm-config /opt/linearspec_lora.yaml`
- 或 SGLang fork 内置的 DLLM YAML config。

### 3.3 `index.html`

这是一个本地 Web smoke 页面。它不是 benchmark 框架，但对手工验证服务很有用。

它会请求：

- `http://localhost:30000/health`
- `http://localhost:30000/v1/chat/completions`

它还会在 streaming 模式下估算 LinearSpec 的 acceptance length。它的估算方式是：

```text
acceptance length ~= completion_tokens / streaming delta chunk 数
```

这个指标不是严格的内核级统计，但很适合快速观察 LinearSpec 一轮 draft + verify 平均能吐出多少 token。如果你修改 LinearSpec 解码逻辑，可以先用它做人工 smoke test，再用更严格的 benchmark 脚本做量化测评。

## 4. 当前 A100 服务器上最重要的适配点

`sglang_spark` README 是 Spark / Blackwell 方向的指南，而当前服务器是 A100。这里有几个必须提前理解的差异。

### 4.1 A100 不应该默认把 FP8 当成主路径

README 示例默认使用：

```bash
QUANT=fp8 ~/sglang_dllm/launch_server.sh detach
```

但 A100 不是 Hopper / Blackwell，不具备 H100、B200、GB10 这类架构上的原生 FP8 Tensor Core 路径。A100 的主力低精度路径是 FP16 / BF16 / TF32。

因此在 A100 上建议顺序是：

1. 先跑 BF16 参考服务，不加 `QUANT=fp8`。
2. 如果 SGLang 当前 fork 在 A100 上支持某种 FP8 fallback，再把 `QUANT=fp8` 作为实验项，而不是默认项。
3. 如果 FP8 启动失败、速度异常、日志里出现 kernel 不支持或 fallback，A100 主实验就应该回到 BF16。

推荐 A100 首次启动用：

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

而不是一开始就用：

```bash
QUANT=fp8 ~/sglang_dllm/launch_server.sh detach
```

### 4.2 `lmsysorg/sglang:spark` 镜像可能不是 A100/x86_64 的最佳镜像

`sglang_spark/README.md` 使用：

```bash
docker pull lmsysorg/sglang:spark
```

这个镜像名和说明都明显服务于 DGX Spark。若当前 A100 服务器是 x86_64，可能遇到以下问题：

- 镜像架构不匹配；
- 镜像内预装的 CUDA / torch / flashinfer 与 A100 不匹配；
- 容器里的 FlashInfer kernel 主要面向 Blackwell；
- `QUANT=fp8` 在 A100 上没有预期性能。

建议先检查本机架构：

```bash
uname -m
```

如果输出是 `x86_64`，并且 `lmsysorg/sglang:spark` 启动失败，下一步应该改用适配 x86_64 + A100 的 SGLang 环境。两种可选方式：

- 使用官方或项目可用的 x86_64 SGLang Docker 镜像，再挂载 `hutm/sglang` fork；
- 在 conda 环境里从 `hutm/sglang` fork 源码安装 SGLang，并确保 torch / triton / flashinfer 版本支持 A100。

本文后面的命令仍然以 `sglang_spark/launch_server.sh` 为主，因为它最完整地表达了 Nemotron-Labs-Diffusion 的 LinearSpec 配置方式。但你在 A100 上实际执行时，需要先确认容器能否正常启动。

### 4.3 当前脚本没有显式选择单张 GPU

`launch_server.sh` 里 Docker 启动参数是：

```bash
--gpus all
```

SGLang 启动参数是：

```bash
--tp-size 1
```

这表示容器能看到所有 GPU，但 SGLang tensor parallel size 只有 1。一般情况下服务会使用容器内的 `cuda:0`，但为了做严谨 benchmark，最好显式隔离到某一张 GPU。

如果你要长期做实验，建议把脚本里的：

```bash
--gpus all
```

改成可配置形式：

```bash
DOCKER_GPUS=${DOCKER_GPUS:-all}
...
--gpus "${DOCKER_GPUS}"
```

这样启动时可以指定：

```bash
DOCKER_GPUS='"device=0"' MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec MAX_REQS=1 ~/sglang_dllm/launch_server.sh detach
```

如果你暂时不改脚本，也至少要在 benchmark 记录里写清楚：容器可见所有 GPU，SGLang 使用 `--tp-size 1`。

## 5. SGLang 工作目录准备

以下命令都假设你从任意目录执行，工作目录统一放到：

```text
$HOME/sglang_dllm
```

### 5.1 克隆 SGLang fork

```bash
mkdir -p ~/sglang_dllm/src && cd ~/sglang_dllm/src && git clone --depth 1 -b upstream/2-dllm-lora-ar https://github.com/hutm/sglang.git
```

如果目录已经存在，检查分支即可：

```bash
cd ~/sglang_dllm/src/sglang && git branch --show-current && git rev-parse --short HEAD
```

### 5.2 应用 README 里的 one-line patch

`sglang_spark/README.md` 要求对 scheduler 做一个一行 patch：

```bash
sed -i 's|self\.report_prefill_stats(|self.metrics_reporter.report_prefill_stats(|' ~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

这个 patch 的目的不是改变解码算法，而是修正 metrics reporter 调用路径，避免服务启动或统计上报时走到错误对象。

应用后建议检查：

```bash
rg -n "metrics_reporter.report_prefill_stats|report_prefill_stats" ~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

### 5.3 准备 LinearSpec LoRA 文件

README 原始命令从 Hugging Face 下载 LoRA adapter。但你的模型目录本地已经有：

```text
/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora
```

优先使用本地文件：

```bash
mkdir -p ~/sglang_dllm/linear_spec_lora && cp /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors ~/sglang_dllm/linear_spec_lora/
```

检查文件：

```bash
ls -lh ~/sglang_dllm/linear_spec_lora
```

应该至少看到：

```text
adapter_config.json
adapter_model.safetensors
```

### 5.4 复制启动脚本和 Web 页面

```bash
cp /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/launch_server.sh ~/sglang_dllm/launch_server.sh && chmod +x ~/sglang_dllm/launch_server.sh
```

如果想使用浏览器 smoke 页面：

```bash
cp /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/index.html ~/sglang_dllm/index.html
```

## 6. 启动 SGLang 服务

### 6.1 A100 上推荐的首次启动：BF16 LinearSpec + LoRA

首次验证建议不用 FP8：

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

查看日志：

```bash
tail -f ~/sglang_dllm/logs/server.log
```

健康检查：

```bash
curl -fsS http://localhost:30000/health
```

停止服务：

```bash
~/sglang_dllm/launch_server.sh stop
```

### 6.2 README 默认路径：LinearSpec + LoRA + FP8

这是 Spark / Blackwell README 的默认推荐方式：

```bash
QUANT=fp8 MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec MAX_REQS=2 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

A100 上仅把它作为实验项。如果启动失败，先不要修改算法代码，先回到 BF16 验证服务链路。

### 6.3 LinearSpec 不使用 LoRA 的 baseline

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec-base MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

这个配置会使用 SGLang fork 内置的：

```text
/opt/sglang_fork/test/registered/dllm/configs/nemotron_labs_linearspec.yaml
```

它适合回答一个问题：LinearSpec 的加速到底来自算法本身，还是来自 LoRA-enhanced draft。

### 6.4 FastDiffuser baseline

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=FastDiffuser MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

这是更接近纯 dLLM diffusion decoding 的 baseline。

### 6.5 AR baseline

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=AR MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

脚本里 `ALGO=AR` 实际会做两件事：

1. 设置 `MODEL_OVERRIDE='{"ar_mode": true}'`；
2. 把 SGLang DLLM algorithm 仍映射到 `FastDiffuser` 配置路径。

因此这里的 AR 不是 README 表格里原生 HF `model.ar_generate` 的同一条 Python 调用链，而是 SGLang serving 里的 AR mode 启动方式。做性能对比时要把它标记为“SGLang AR serving baseline”。

### 6.6 LoRA 应用范围：`draft_only` 和 `both`

默认：

```bash
LORA_MODE=draft_only
```

含义是 LoRA 只增强 draft 模型，不改变 verifier。这个设置最符合 LinearSpec 的基本意图：draft 更强可以提高候选 token 被 verifier 接受的概率，但 verifier 仍保持原模型语义。

诊断用：

```bash
LORA_MODE=both MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec MAX_REQS=1 ~/sglang_dllm/launch_server.sh detach
```

`both` 会让 draft 和 verify 都使用 LoRA。它可能改善某些表面指标，但语义上不再是“LoRA draft + base verifier”的严格对照。除非你明确要研究这个变量，否则主实验建议使用 `draft_only`。

## 7. Smoke test

### 7.1 健康检查

```bash
curl -fsS http://localhost:30000/health
```

只要它返回正常 JSON 或正常状态码，就说明 HTTP 服务已经起来。但这还不能证明模型生成正常。

### 7.2 单请求 chat completion

```bash
curl -sS http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"/data1/linyewei/models/Nemotron-Labs-Diffusion-8B","messages":[{"role":"user","content":"What is 15% of 240?"}],"max_tokens":256,"temperature":0}'
```

期望输出应该包含 `choices[0].message.content`。如果 health 正常但这个请求失败，优先看：

```bash
tail -n 200 ~/sglang_dllm/logs/server.log
```

常见问题包括：

- 容器内看不到模型路径；
- `trust_remote_code` 加载失败；
- FlashInfer / CUDA kernel 不支持当前 GPU；
- FP8 在 A100 上不可用；
- LoRA adapter 路径没有挂载进容器；
- SGLang fork 分支或 patch 不一致。

### 7.3 streaming smoke test

```bash
curl -N -sS http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"/data1/linyewei/models/Nemotron-Labs-Diffusion-8B","messages":[{"role":"user","content":"Write a short explanation of speculative decoding."}],"max_tokens":256,"temperature":0,"stream":true}'
```

LinearSpec 下 streaming chunk 的数量可以粗略反映 draft + verify 的循环次数。如果 completion token 数相同，但 chunk 数减少，通常说明每轮接受 token 更多，服务端开销更低。

### 7.4 Web 页面 smoke test

先启动一个本地静态页面服务：

```bash
cd ~/sglang_dllm && python3 -m http.server 8000 --bind 127.0.0.1
```

然后在浏览器或 SSH tunnel 中打开：

```text
http://127.0.0.1:8000/index.html
```

这个页面适合快速观察：

- 服务是否在线；
- chat completion 是否能返回；
- streaming 是否顺畅；
- LinearSpec acceptance length 的粗略变化。

它不适合作为正式 benchmark 结论。

## 8. Benchmark 应该测什么

针对 Linear SS serving 优化，至少要记录以下指标。

| 指标 | 含义 | 为什么重要 |
| --- | --- | --- |
| end-to-end latency | 单个请求从发送到完成的总耗时 | 用户实际等待时间 |
| TTFT | time to first token | 交互式 serving 里非常关键 |
| ITL | inter-token latency | 流式输出是否平滑 |
| output tokens/s | 每秒生成 token 数 | 解码吞吐核心指标 |
| request/s | 每秒完成请求数 | serving 承载能力 |
| p50 / p90 / p99 latency | 延迟分位数 | 并发下比均值更重要 |
| GPU utilization | GPU 利用率 | 判断瓶颈是否在 GPU |
| GPU memory | 显存占用 | 判断可支持的并发和上下文长度 |
| accepted tokens per verify | LinearSpec 每轮 verify 平均接受 token 数 | Linear SS 解码效率核心指标 |
| draft forward count | draft 前向次数 | 判断 draft 阶段成本 |
| verify forward count | verify 前向次数 | 判断 verifier 成本 |

如果只能先做最小可行 benchmark，建议先测：

- output tokens/s；
- p50 / p90 end-to-end latency；
- TTFT；
- accepted tokens per verify；
- GPU memory。

## 9. SGLang benchmark 的推荐方式

### 9.1 先确认当前 fork 里 benchmark 工具怎么调用

不同 SGLang 版本 benchmark 入口可能略有变化。不要先假设命令一定完全一致，先在 fork 中查：

```bash
rg -n "bench_serving|benchmark_serving" ~/sglang_dllm/src/sglang
```

如果当前环境已经能 import fork 里的 SGLang，可以看 help：

```bash
PYTHONPATH=~/sglang_dllm/src/sglang/python python3 -m sglang.bench_serving --help
```

如果是在容器内跑 benchmark，则进入容器后看：

```bash
python3 -m sglang.bench_serving --help
```

### 9.2 随机长度 serving benchmark 模板

如果当前 fork 支持 `sglang.bench_serving` 的常见参数，可以从这个模板开始：

```bash
PYTHONPATH=~/sglang_dllm/src/sglang/python python3 -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --num-prompts 100 --random-input-len 256 --random-output-len 256 --request-rate inf
```

如果参数名不匹配，以 `--help` 输出为准。重点不是死记这个命令，而是保证每次实验固定以下变量：

- prompt 数量；
- 输入长度；
- 输出长度；
- request rate 或 concurrency；
- `ALGO`；
- `MAX_REQS`；
- `CTX_LEN`；
- `MEM_FRAC`；
- 是否 `QUANT=fp8`；
- `LORA_MODE`。

### 9.3 简易并发 smoke benchmark

如果正式 benchmark 工具还没跑通，可以先用 `xargs -P` 做粗略并发压测：

```bash
/usr/bin/time -f 'elapsed_seconds=%e' bash -lc 'seq 1 16 | xargs -P 4 -I{} curl -sS http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"/data1/linyewei/models/Nemotron-Labs-Diffusion-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Give a concise explanation of CUDA graphs in LLM serving. Request {}.\"}],\"max_tokens\":256,\"temperature\":0}" >/tmp/nld_sglang_xargs_outputs.jsonl'
```

这里：

- `16` 是总请求数；
- `-P 4` 是并发数；
- `/usr/bin/time` 给出总耗时；
- 输出保存到 `/tmp/nld_sglang_xargs_outputs.jsonl`。

这个方法不提供 TTFT 和分位数，只能作为非常粗的 sanity check。正式实验仍应该使用 SGLang benchmark 或自己写 async HTTP benchmark。

### 9.4 同时监控 GPU

开一个终端监控显存和利用率：

```bash
nvidia-smi dmon -s pucm
```

或者每秒查询一次：

```bash
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv'
```

记录 benchmark 时，不要只看 tokens/s。LinearSpec serving 的瓶颈可能在：

- draft / verify 循环导致 GPU 小 batch 碎片化；
- scheduler 不能有效合并不同阶段请求；
- CPU Python 调度开销；
- LoRA 路径引入额外 kernel；
- KV cache 或 attention backend；
- CUDA graph capture batch size 不覆盖实际 batch；
- 输出长度较短时每轮开销占比过大。

## 10. 推荐实验矩阵

### 10.1 算法 baseline

先固定：

- `CTX_LEN=2048`
- `MEM_FRAC=0.5`
- `MAX_REQS=1`
- 输入 256 token
- 输出 256 token
- `temperature=0`

依次跑：

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=AR MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=FastDiffuser MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec-base MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec LORA_MODE=draft_only MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

如果 A100 上 FP8 能启动，再额外加入：

```bash
QUANT=fp8 MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec LORA_MODE=draft_only MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

### 10.2 并发矩阵

固定算法为主目标：

```text
ALGO=LinearSpec
LORA_MODE=draft_only
```

然后改变：

```text
MAX_REQS = 1, 2, 4, 8, 16
```

benchmark 侧也要同步改变并发或 request rate。例如：

- 低压：单请求或 request rate 很低；
- 中压：GPU 利用率接近 70%；
- 饱和：继续提高并发但吞吐不再增加；
- 过载：p90 / p99 明显恶化。

对 serving 优化来说，饱和点和过载点往往比单请求 tokens/s 更重要。

### 10.3 长度矩阵

至少测：

| 场景 | 输入长度 | 输出长度 | 意义 |
| --- | --- | --- | --- |
| 短问答 | 128 | 128 | 每轮调度开销占比高 |
| 常规聊天 | 512 | 256 | 接近普通 serving |
| 长回复 | 512 | 1024 | 解码阶段占比高 |
| 长上下文 | 2048 | 512 | prefill 和 KV cache 压力高 |

LinearSpec 的优势通常更依赖输出阶段。如果输出很短，draft + verify 的额外机制可能来不及摊薄固定开销。

### 10.4 记录表模板

每次实验至少记录：

```text
date:
server:
gpu:
sglang fork commit:
model path:
algorithm:
quant:
lora_mode:
ctx_len:
mem_frac:
max_reqs:
input_len:
output_len:
num_prompts:
request_rate or concurrency:
throughput output tok/s:
request/s:
p50 latency:
p90 latency:
p99 latency:
TTFT:
ITL:
accepted tokens per verify:
GPU util:
GPU memory:
notes:
```

不要只保存最终数字。SGLang fork commit、启动命令和日志路径也要保留，否则后续很难判断性能变化来自代码修改还是环境变化。

## 11. LinearSpec / Linear SS 在 SGLang 中大致怎么接入

从 `launch_server.sh` 可以看出，SGLang 侧的 LinearSpec 接入不是调用项目根目录里的 `evaluate.py`，而是通过 `sglang.launch_server` 的 DLLM 参数完成：

```bash
--dllm-algorithm LinearSpec
--dllm-config /opt/linearspec_lora.yaml
```

默认 `LinearSpec` 时，脚本会在 host 上生成：

```yaml
algorithm: LinearSpec
causal_context: true
lora_path: /opt/linear_spec_lora
lora_mode: draft_only
```

然后把这个 YAML 挂载进容器：

```text
host:      ~/sglang_dllm/linearspec_lora.yaml
container: /opt/linearspec_lora.yaml
```

这说明 LinearSpec 的 SGLang runtime 主要由两部分决定：

1. SGLang fork 里的 DLLM algorithm 代码；
2. 启动时传入的 DLLM YAML config。

想优化 serving 下的 Linear SS 效率，主要应该改这两处，而不是改 `eval.sh`。

## 12. 应该到哪里找 SGLang LinearSpec 解码代码

克隆 fork 后，用下面命令定位相关代码：

```bash
rg -n "LinearSpec|FastDiffuser|dllm|lora_mode|draft|verify|accept|causal_context" ~/sglang_dllm/src/sglang/python ~/sglang_dllm/src/sglang/test
```

重点目录通常在：

```text
~/sglang_dllm/src/sglang/python/sglang/srt/dllm
~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin
~/sglang_dllm/src/sglang/test/registered/dllm/configs
```

其中 `sglang_spark/README.md` 明确 patch 的文件是：

```text
~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

这说明 SGLang 的 DLLM serving 链路至少涉及 scheduler mixin。LinearSpec 下的 draft、verify、接受 token、请求调度、统计上报，很可能分散在 `dllm` 目录的 algorithm、scheduler、worker 或 mixin 代码中。

建议第一次阅读顺序：

1. 先看 `test/registered/dllm/configs/nemotron_labs_linearspec.yaml` 和 `nemotron_labs_fastdiffuser.yaml`。
2. 再找 `LinearSpec` algorithm 类或注册表。
3. 再看 scheduler 如何根据 `--dllm-algorithm` 和 `--dllm-config` 初始化算法。
4. 最后看每次 decode step 中 draft 和 verify 是如何被排入 batch、如何更新 token、如何停止请求。

## 13. 哪些修改适合先做

### 13.1 配置层优化

优先尝试不用改代码的参数：

- `MAX_REQS`：控制最多同时运行 request 数；
- `MEM_FRAC`：控制 SGLang 静态显存池；
- `CTX_LEN`：控制最大上下文；
- `--cuda-graph-bs 1 2 3 4`：控制捕获哪些 batch size 的 CUDA graph；
- `LORA_MODE=draft_only` / `both`：控制 LoRA 使用范围；
- `QUANT=fp8`：仅在支持 FP8 的 GPU 上作为加速项；
- `--attention-backend flashinfer`：当前脚本默认使用 FlashInfer。

对 A100，建议先调：

```text
MAX_REQS, CTX_LEN, MEM_FRAC, cuda-graph-bs
```

不要把 FP8 当成第一优化手段。

### 13.2 统计与可观测性修改

在真正改算法前，建议先加统计。否则你只能看到总 tokens/s，不知道 LinearSpec 快或慢的原因。

建议至少记录：

- 每个 request 的 draft forward 次数；
- verify forward 次数；
- 每轮 draft 候选 token 数；
- 每轮实际接受 token 数；
- 平均 acceptance length；
- draft 阶段耗时；
- verify 阶段耗时；
- scheduler 排队耗时；
- batch size；
- request 是否因为长度、EOS 或 verifier 拒绝而结束。

这些统计可以先写到 server log。等稳定后再考虑通过 OpenAI response 的 extra 字段或 metrics endpoint 暴露。

### 13.3 算法层优化方向

LinearSpec serving 的效率通常可以从四个方向优化。

第一，提高每轮 verify 接受的 token 数：

- 调整 draft block size；
- 改善 draft token 质量；
- 使用更合适的 LoRA；
- 检查 `draft_only` 是否真的只作用在 draft；
- 分析哪些 prompt 类型 acceptance length 低。

第二，降低每轮 draft + verify 的固定开销：

- 减少 Python scheduler 循环开销；
- 避免每轮重复构造临时对象；
- 复用中间 buffer；
- 扩大 CUDA graph 覆盖的 batch size；
- 减少不必要的同步和 log。

第三，改善并发场景下的 batching：

- 将处于相同阶段的 request 合批；
- 避免长请求阻塞短请求；
- 分析 `MAX_REQS` 增大后 acceptance length 是否下降；
- 避免不同 request 的 draft/verify 阶段频繁打散 batch；
- 观察 p90 / p99 延迟，而不只看平均 tokens/s。

第四，优化 memory 和 attention backend：

- 合理设置 `CTX_LEN`，不要过大导致静态 cache 浪费；
- 调整 `MEM_FRAC`，避免显存池不足或过度预留；
- 在 A100 上确认 FlashInfer kernel 是否适配 sm80；
- 对短输出和长输出分别测量，避免只优化一种场景。

## 14. 修改解码逻辑时的正确性边界

LinearSpec 的核心不是“draft 说什么就提交什么”，而是 draft 提供候选，verifier 决定哪些 token 可以接受。

因此有几条边界不要轻易破坏：

1. verifier 不应无意中被 LoRA 改写，除非你明确在做 `LORA_MODE=both` 实验。
2. 被提交到输出序列的 token 应该经过 verifier 接受逻辑。
3. 改 block size 或接受规则后，要重新跑 correctness smoke。
4. 在 `temperature=0` 下，同一配置应尽量保持输出稳定。
5. 优化吞吐时不要只看平均值，要看 p90 / p99 是否明显恶化。

最小正确性回归可以这样做：

- 固定 20 到 100 个 prompts；
- 分别用 SGLang AR、LinearSpec-base、LinearSpec+LoRA 生成；
- `temperature=0`；
- 比较是否有明显重复、截断、乱码、提前 EOS；
- 对数学题、代码题、长回复题分别抽样人工检查。

如果改动比较大，再回到 `evaluate.py` 或小规模标准任务上做 accuracy 验证。

## 15. 推荐的实际操作顺序

### 15.1 第一阶段：跑通服务

1. 准备 SGLang fork、LoRA、启动脚本。
2. 在 A100 上用 BF16 启动 `ALGO=LinearSpec`。
3. 跑 `/health`。
4. 跑单请求 `/v1/chat/completions`。
5. 跑 streaming smoke。
6. 保存启动命令和 `server.log`。

### 15.2 第二阶段：建立 baseline

固定相同 prompt、输入长度、输出长度和并发，分别测：

- SGLang `AR`
- SGLang `FastDiffuser`
- SGLang `LinearSpec-base`
- SGLang `LinearSpec + LORA_MODE=draft_only`
- 如果 A100 上能正常支持，再测 `QUANT=fp8`

这一阶段不要改代码。先知道当前实现是什么水平。

### 15.3 第三阶段：并发与长度实验

固定主算法：

```text
ALGO=LinearSpec
LORA_MODE=draft_only
```

测：

- `MAX_REQS=1,2,4,8,16`
- 输入输出长度矩阵；
- request rate 从低到高；
- GPU 利用率和显存；
- p50 / p90 / p99 latency；
- TTFT 和 ITL。

这一步会告诉你瓶颈是单请求解码，还是并发调度。

### 15.4 第四阶段：加统计

在 SGLang DLLM 代码里加 LinearSpec 专用统计：

- 每轮 accepted tokens；
- draft / verify 耗时；
- draft / verify forward 次数；
- batch size；
- request 阶段状态。

加统计后重新跑第二、第三阶段，确保统计本身没有显著拖慢服务。

### 15.5 第五阶段：改解码或调度逻辑

每次只改一个变量：

- draft block size；
- verifier 接受逻辑；
- LoRA 使用范围；
- CUDA graph batch size；
- scheduler 合批策略；
- request 阶段切换逻辑；
- cache / buffer 复用。

每次修改后都跑：

1. smoke test；
2. 小 benchmark；
3. 主 benchmark；
4. correctness 抽查。

## 16. 常见问题定位

### 16.1 服务启动失败

先看：

```bash
tail -n 200 ~/sglang_dllm/logs/server.log
```

重点搜索：

```bash
rg -n "error|Error|ERROR|Traceback|CUDA|flashinfer|quant|lora|dllm|LinearSpec" ~/sglang_dllm/logs/server.log
```

### 16.2 A100 上 FP8 失败

处理顺序：

1. 停服务；
2. 去掉 `QUANT=fp8`；
3. 用 BF16 跑通；
4. 确认服务和模型都正常后，再单独研究 FP8 是否有 A100 可用路径。

### 16.3 LoRA 找不到

确认 host 目录：

```bash
ls -lh ~/sglang_dllm/linear_spec_lora
```

确认启动日志里是否出现：

```text
/opt/linear_spec_lora
```

因为容器内看到的路径不是 host 上的 `/data1/.../linear_spec_lora`，而是脚本挂载后的：

```text
/opt/linear_spec_lora
```

### 16.4 health 正常但生成失败

优先怀疑：

- 模型 remote code 加载；
- tokenizer chat template；
- `--dllm-algorithm` 和 YAML 不匹配；
- A100 不支持当前 attention / quant kernel；
- `CTX_LEN` 太大导致 cache 初始化失败；
- `MEM_FRAC` 太高或太低。

### 16.5 单请求快，并发慢

重点看：

- `MAX_REQS` 是否太低；
- batch size 是否落在 `--cuda-graph-bs 1 2 3 4` 覆盖之外；
- request 长度差异是否导致排队；
- draft 和 verify 阶段是否不能有效合批；
- CPU 调度开销是否过高；
- GPU utilization 是否很低但 latency 很高。

## 17. 你是否有办法修改解码逻辑来优化 SGLang serving

有，而且这正是你当前目标应该做的事。但修改位置应该在 SGLang fork，而不是只改 Nemotron-Labs-Diffusion 项目根目录里的 `evaluate.py`。

`sglang_spark/launch_server.sh` 的设计本身就是“本地源码可修改”的方式。它会把 host 上的 SGLang fork：

```text
~/sglang_dllm/src/sglang
```

挂载到容器内：

```text
/opt/sglang_fork
```

并在容器里设置：

```bash
PYTHONPATH=/opt/sglang_fork/python
```

然后用：

```bash
python3 -m sglang.launch_server
```

启动服务。因此，只要没有被别的已安装包覆盖，服务端运行的就是你 host 上这份 fork 里的 Python 源码。你可以在 host 上直接修改：

```text
~/sglang_dllm/src/sglang/python/sglang/srt/dllm
```

然后停止并重启服务：

```bash
~/sglang_dllm/launch_server.sh stop && MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec LORA_MODE=draft_only MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

这就是本地部署、本地改代码、本地 benchmark 的闭环。

推荐修改层次如下：

### 17.1 低风险：改 DLLM YAML 和 launch 参数

适合先做：

- `LORA_MODE`
- `MAX_REQS`
- `CTX_LEN`
- `MEM_FRAC`
- `--cuda-graph-bs`
- `--attention-backend`
- 是否量化

优点是容易回滚，适合建立性能地图。

### 17.2 中风险：加 LinearSpec 统计和日志

适合第二步做。目标是让每次 benchmark 都能解释：

- 为什么 LinearSpec 快；
- 为什么某些并发下不快；
- acceptance length 是否真的提高；
- draft 和 verify 的时间比例是多少。

### 17.3 高价值：改 SGLang DLLM scheduler / LinearSpec algorithm

这是核心优化区域。可能涉及：

- draft 阶段每轮生成多少候选 token；
- verify 阶段如何批量验证；
- 接受 token 后如何更新 request state；
- 不同 request 的 draft/verify 阶段如何合批；
- streaming chunk 何时发给客户端；
- KV cache 如何复用和裁剪；
- LoRA 权重如何只在 draft 路径启用；
- CUDA graph 如何覆盖实际 serving batch。

这类修改最可能带来 serving 性能收益，但也最容易改变语义或引入并发 bug。每次修改后都要同时测 correctness 和 performance。

### 17.4 不建议作为主线：只改原生 HF 解码函数

模型目录里的 remote code 或项目里的 `evaluate.py` 对理解算法很重要，但如果 SGLang 已经在自己的 runtime 中实现 DLLM / LinearSpec，单纯改原生 `model.linear_spec_generate` 不一定影响 SGLang 服务。

除非 SGLang runtime 确实调用了模型 remote code 中的对应生成函数，否则 serving 优化应该以 SGLang fork 为准。

## 18. 一套建议的最小命令序列

下面是一套从零到 smoke 的最小命令序列。A100 上先使用 BF16。

准备 fork：

```bash
mkdir -p ~/sglang_dllm/src && cd ~/sglang_dllm/src && git clone --depth 1 -b upstream/2-dllm-lora-ar https://github.com/hutm/sglang.git
```

打 patch：

```bash
sed -i 's|self\.report_prefill_stats(|self.metrics_reporter.report_prefill_stats(|' ~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

准备 LoRA：

```bash
mkdir -p ~/sglang_dllm/linear_spec_lora && cp /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors ~/sglang_dllm/linear_spec_lora/
```

准备启动脚本：

```bash
cp /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_spark/launch_server.sh ~/sglang_dllm/launch_server.sh && chmod +x ~/sglang_dllm/launch_server.sh
```

启动 BF16 LinearSpec：

```bash
MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B ALGO=LinearSpec LORA_MODE=draft_only MAX_REQS=1 CTX_LEN=2048 MEM_FRAC=0.5 ~/sglang_dllm/launch_server.sh detach
```

看日志：

```bash
tail -f ~/sglang_dllm/logs/server.log
```

健康检查：

```bash
curl -fsS http://localhost:30000/health
```

生成测试：

```bash
curl -sS http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"/data1/linyewei/models/Nemotron-Labs-Diffusion-8B","messages":[{"role":"user","content":"What is 15% of 240?"}],"max_tokens":256,"temperature":0}'
```

停服务：

```bash
~/sglang_dllm/launch_server.sh stop
```

## 19. 最终建议

对你的需求来说，推荐判断如下：

- `evaluate.py`：用来确认模型、权重、tokenizer、原生解码函数和基本 accuracy 没问题。
- `eval.sh`：除非你要复现论文表格或跑 SLURM 集群批量评测，否则不是主线。
- `sglang_spark` / SGLang fork：这是你的主战场。
- Linear SS 优化：优先在 SGLang DLLM algorithm、scheduler、batching、LoRA draft、CUDA graph 和统计指标上做。
- A100 实验：先以 BF16 为主，不要默认采用 Spark README 的 FP8 方案。

一句话概括：先用 `evaluate.py` 做模型验货，然后直接进入 SGLang serving；建立 SGLang 下 AR / FastDiffuser / LinearSpec-base / LinearSpec+LoRA 的基线，再围绕 LinearSpec 的 draft + verify 接受长度、调度开销和并发 batching 做优化。`eval.sh` 可以暂时不投入。
