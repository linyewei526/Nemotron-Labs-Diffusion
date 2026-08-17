# eval.sh 评测链路通俗解释：它在做什么，和 evaluate.py 有什么区别，和 SGLang 有什么关系

本文档面向完全不了解 SLURM、enroot、pyxis、容器评测链路的读者，解释本项目里两条评测路径：

- `evaluate.py`：一个 Python 进程直接加载模型并评测。
- `eval.sh`：SLURM + 容器 + HTTP server + NeMo-Skills 的完整评测流水线。

最后也会回答：如果你关心 SGLang 推理引擎和 serving 场景，是否有必要采用 `eval.sh` 链路。

## 1. 最简结论

如果你只是想确认模型能不能跑、三种解码模式是否正常、在 GSM8K/MATH-500 上大概 accuracy 和 TPF 如何：

```text
优先用 evaluate.py。
```

如果你想复现 README/论文风格的多 benchmark sweep，使用更正式的 NeMo-Skills 任务定义、输出目录、日志、NFE 记录，并且在集群上多 GPU 并发跑评测：

```text
使用 eval.sh。
```

如果你关心 SGLang 的 serving 实现、并发、吞吐、延迟、量化、scheduler、CUDA graph、线上 OpenAI API 服务：

```text
eval.sh 不是 SGLang 链路。你应该看 sglang_spark/ 和 SGLang 自己的 benchmark/serving 工具。
```

可以把三条路径理解成：

```text
evaluate.py      = 本地直接调用模型函数，最简单，最适合调试
eval.sh          = 集群离线评测流水线，适合正式跑 benchmark
sglang_spark/    = SGLang serving 部署路径，适合研究线上推理系统
```

## 2. 三条路径分别回答什么问题

### 2.1 `evaluate.py` 回答的问题

`evaluate.py` 回答的是：

```text
在当前 Python 环境、当前 GPU、当前模型代码下，
直接调用 model.generate / model.ar_generate / model.linear_spec_generate，
能得到什么 accuracy、平均生成 token 数、平均 NFE、TPF？
```

它的特点：

- 不起 server。
- 不用 SLURM。
- 不用容器。
- 一个 Python 进程内完成：加载模型、加载数据集、生成、打分。
- 目前内置任务少，只支持：
  - `gsm8k`
  - `math-500`
- scorer 是项目里手写的轻量 scorer。
- 适合 smoke test 和快速理解模型行为。

### 2.2 `eval.sh` 回答的问题

`eval.sh` 回答的是：

```text
在集群环境里，用一个标准化容器，
按 NeMo-Skills 的 benchmark/scorer/prompt 流程，
通过 OpenAI-compatible HTTP server 调用模型，
在多个 benchmark 上得到正式评测结果和 NFE/TPF 统计。
```

它的特点：

- 用 SLURM 提交作业。
- 用 enroot/pyxis 容器运行环境。
- 每张 GPU 启动一个推理 worker server。
- 前面有一个 load balancer。
- 评测客户端是 `xp/nemo-skills/eval_dlm.py`。
- 通过 HTTP 请求访问模型，而不是在同一个 Python 函数里直接调用模型。
- 默认 benchmark suite 更完整，README 里默认包括：
  - `gsm8k`
  - `human-eval`
  - `mbpp`
  - `math-500`
  - `aime24`
  - `aime25`
  - `gpqa`
  - `mmlu`
  - `ifeval`
  - `livecodebench-cpp`
- 更适合跑正式 benchmark、保存日志、合并 NFE、跨任务批量扫参数。

### 2.3 SGLang 路径回答的问题

SGLang 路径回答的是：

```text
把模型部署成一个真正面向 serving 的推理服务时，
在 SGLang runtime 下并发、吞吐、延迟、量化、调度表现如何？
```

它关注的不是单纯的 benchmark accuracy，而是 serving 系统：

- request batching
- scheduler
- KV cache 管理
- CUDA graph
- FlashInfer / kernel backend
- FP8 / INT4 等量化
- OpenAI-compatible server
- 多用户并发吞吐
- 首 token 延迟和端到端延迟

本项目对应入口在：

```text
sglang_spark/
```

而不是 `eval.sh`。

## 3. 先理解几个基础概念

### 3.1 什么是 SLURM

SLURM 是集群作业调度系统。你可以把它理解成：

```text
一台大服务器或很多服务器上有很多 GPU，
不能每个人直接随便占用，
所以大家向 SLURM 申请资源。
SLURM 根据资源情况安排你的作业什么时候、在哪些 GPU 上运行。
```

常见命令：

- `sbatch`：提交一个后台作业。提交后通常立刻返回一个 job id。
- `srun`：在已经申请到的资源里启动具体命令。
- `squeue`：看作业排队/运行状态。
- `scancel`：取消作业。

本项目 `eval.sh` 的真实运行路径会用到：

```text
sbatch -> srun -> 在容器里执行评测命令
```

### 3.2 什么是 container

container 是一个打包好的运行环境。它通常包含：

- Python
- PyTorch
- Transformers
- NeMo-Skills
- CUDA 相关库
- 各种评测依赖

为什么要用 container？

因为正式评测依赖很多，直接在你的 conda 环境里装很容易乱。container 的意义是：

```text
把“能跑这套评测的环境”固定下来，
换一台机器也尽量得到同样的软件环境。
```

### 3.3 什么是 enroot / pyxis

在这个项目的 README 里写的是：

```text
SLURM + enroot/pyxis container required
```

可以粗略理解为：

- `enroot`：一种在 HPC 集群里常用的用户态容器运行工具。
- `pyxis`：SLURM 的一个插件，让 `srun` 可以直接加 `--container-image=...` 来启动容器。
- `.sqsh`：enroot 常用的容器镜像文件格式。

你不需要先理解它们的全部细节。对跑本项目来说，你只需要知道：

```text
如果集群装好了 pyxis，
srun 可以这样启动容器：

srun --container-image=/path/to/image.sqsh ...
```

本项目 pipeline 里实际就会生成类似命令：

```bash
srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-workdir="$PROJECT_DIR" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-mount-home \
    bash "$CMD_SCRIPT"
```

### 3.4 什么是 ACCOUNT、partition、time、gpus

`eval.sh` 真实跑作业时需要这些参数：

- `--account` / `ACCOUNT`：你的集群计费账号或项目账号。
- `--partition`：提交到哪个队列，例如 `batch,backfill`。
- `--time`：作业最长运行时间。
- `--gpus`：这个作业要几张 GPU。
- `CONTAINER_IMAGE`：容器镜像 `.sqsh` 路径。

如果这些没有配置好，`eval.sh` 会报错。当前环境里 `srun/sbatch` 不在 `PATH`，所以真实 `eval.sh` 路径当前不能直接跑，只能先 `--dry-run`。

## 4. `evaluate.py` 到底怎么跑

`evaluate.py` 的流程非常直接：

```text
python evaluate.py
  -> AutoTokenizer.from_pretrained(model)
  -> AutoModel.from_pretrained(model, trust_remote_code=True)
  -> datasets.load_dataset(...)
  -> 对每条样本：
       tokenizer.apply_chat_template(...)
       根据 --mode 调模型函数：
         ar          -> model.ar_generate(...)
         dlm         -> model.generate(...)
         linear_spec -> model.linear_spec_generate(...)
       decode output
       用本文件里的 scorer 打分
  -> 打印 accuracy / avg_tok / avg_nfe / TPF
```

它的 `generate()` 分发逻辑在 `evaluate.py` 中：

```text
mode=ar          -> model.ar_generate(...)
mode=dlm         -> model.generate(...)
mode=linear_spec -> model.linear_spec_generate(...)
```

这里没有 HTTP，没有 server，没有 worker，没有 load balancer。

可以把它看成：

```text
一个 Python 脚本自己问模型问题，自己判答案。
```

## 5. `eval.sh` 到底怎么跑

`eval.sh` 本身不是评测器，它更像一个总控脚本。它做的是：

```text
把你的命令行参数翻译成一堆环境变量，
然后为每个 benchmark 提交一个 SLURM 作业。
```

### 5.1 第一步：解析 mode

`eval.sh --mode` 支持三种：

| mode | 服务 engine | 生成 algorithm | 最终 native 调用 |
| --- | --- | --- | --- |
| `ar` | `ar_native` | `ar_native` | `model.ar_generate()` |
| `dlm` | `nemotron` | `nemotron` | `model.generate()` |
| `linear_spec` | `nemotron` | `nemotron` + `LINEAR_SPECULATION=true` | `model.linear_spec_generate()` |

也就是说，`eval.sh` 和 `evaluate.py` 最后调用的模型原生方法可以是同一个：

```text
AR         都是 ar_generate()
dLM        都是 generate()
Linear SS  都是 linear_spec_generate()
```

区别在于外面包了什么评测框架和服务框架。

### 5.2 第二步：设置环境变量

`eval.sh` 会把命令行参数变成两类环境变量。

第一类是 server 相关：

```text
SERVER_MODEL_PATH
SERVER_ENGINE
SERVER_BATCH_SIZE
SERVER_MAX_MODEL_LEN
SERVER_MAX_POSITION_EMBEDDINGS
SERVER_LORA_PATH
SERVER_EOS_EARLY_STOP
SERVER_GPUS
```

第二类是 eval client 相关：

```text
SEQ_EVAL_GENERATION_ALGORITHM
SEQ_EVAL_TOKENS_TO_GENERATE
SEQ_EVAL_STEPS
SEQ_EVAL_BLOCK_LENGTH
SEQ_EVAL_THRESHOLD
SEQ_EVAL_TEMPERATURE
SEQ_EVAL_MAX_THINKING_TOKENS
SEQ_EVAL_BENCHMARK
```

你可以理解成：

```text
SERVER_*   告诉推理服务器怎么加载模型、用几张 GPU、开什么模式。
SEQ_EVAL_* 告诉评测客户端要跑哪个 benchmark、生成多少 token、block_length/threshold 是多少。
```

### 5.3 第三步：调用 pipeline

`eval.sh` 实际调用：

```text
xp/examples/run_dlm_eval_pipeline_gpu_only.sh
```

这个 pipeline 会生成一个 SLURM batch script，然后：

```text
sbatch 提交作业
  -> 作业里 srun 启动容器
     -> 容器里执行真正的评测流程
```

### 5.4 第四步：容器里做 7 件事

pipeline 文件开头已经写了 workflow，通俗解释如下：

1. 激活容器里的 Python 环境。
2. 如果需要，安装或补齐评测依赖。
3. 如果传的是训练 DCP checkpoint，先转换成 Hugging Face 格式。
4. 启动推理 worker server。
5. 启动 load balancer。
6. 等 server 健康检查通过。
7. 启动 NeMo-Skills eval client，通过 HTTP 请求模型，完成评测。
8. 清理 server 进程，写 `COMPLETED` 或 `FAILED` 标记。

其中最关键的是：

```text
先起模型服务，再让评测客户端通过 HTTP 调它。
```

这和 `evaluate.py` 的“一个 Python 进程直接调用模型函数”完全不同。

## 6. eval.sh 里的 server 架构

`eval.sh` 路径使用本项目自己的 HTTP server，不是 SGLang。

相关文件：

```text
xp/dlm_api/dlm_batch_server.py
xp/dlm_api/dlm_load_balancer.py
xp/dlm_api/dlm_generate/
xp/nemo-skills/eval_dlm.py
```

### 6.1 worker server

每张 GPU 启一个：

```text
dlm_batch_server.py
```

它负责：

- 加载模型。
- 接收 OpenAI-compatible `/v1/chat/completions` 请求。
- 根据 `generation_algorithm` 选择算法。
- 调用模型原生方法。
- 记录 NFE。
- 返回类似 OpenAI API 的响应。

### 6.2 load balancer

多 GPU 时，前面再启动一个：

```text
dlm_load_balancer.py
```

它负责：

- 接收 eval client 的请求。
- 分发给后面的 worker。
- 多个 worker 可以分别占用不同 GPU。

可以画成：

```text
NeMo-Skills eval client
        |
        v
dlm_load_balancer.py
        |
        +--> GPU0 worker: dlm_batch_server.py
        +--> GPU1 worker: dlm_batch_server.py
        +--> GPU2 worker: dlm_batch_server.py
        +--> GPU3 worker: dlm_batch_server.py
```

### 6.3 generation algorithm registry

worker 内部通过 `xp/dlm_api/dlm_generate/` 分发：

```text
ar_native        -> ArNativeGeneration      -> model.ar_generate()
nemotron         -> NemotronGeneration      -> model.generate()
nemotron + linear_speculation
                 -> NemotronGeneration      -> model.linear_spec_generate()
nemotron_mixed   -> NemotronMixedGeneration -> model.generate_mixed()，需要额外代码支持
```

所以 `eval.sh` 路径虽然复杂，但最后仍然会落到模型目录中的原生方法。

## 7. NeMo-Skills 在这里做什么

NeMo-Skills 是 NVIDIA 的评测框架。你可以把它理解成：

```text
一套 benchmark 数据加载、prompt 构造、模型调用、结果解析、打分、保存结果的框架。
```

本项目的 `xp/nemo-skills/eval_dlm.py` 是对 NeMo-Skills 的包装，额外支持 diffusion 参数。

普通 OpenAI API 没有这些字段：

```text
block_length
threshold
steps
linear_speculation
draft_lora_only
max_thinking_tokens
generation_algorithm
```

所以本项目用 OpenAI request 的 `extra_body` 携带它们：

```text
NeMo-Skills client
  -> extra_body.block_length = 8
  -> extra_body.threshold = 0.9
  -> extra_body.generation_algorithm = nemotron
  -> extra_body.linear_speculation = true
```

server 收到后再分发到对应生成方法。

## 8. accuracy 对比：evaluate.py vs eval.sh

### 8.1 `evaluate.py` 的 accuracy

`evaluate.py` 的 accuracy 是轻量版本：

- 内置任务少。
- scorer 逻辑直接写在 `evaluate.py`。
- GSM8K 主要看最后的 `\boxed{N}` 或最后一个数字。
- MATH-500 主要看最后的 `\boxed{...}`。
- prompt 也由本文件简单构造。

它适合：

- 快速 smoke test。
- 看三种 decoding mode 是否正常。
- 小样本调试。
- 初步比较不同 block_length/threshold。

但它不适合声称“复现论文完整 benchmark”。

### 8.2 `eval.sh` 的 accuracy

`eval.sh` 通过 NeMo-Skills 跑 benchmark：

- 支持更多任务。
- 每个任务有对应的数据准备、prompt、解析、打分流程。
- 输出格式更标准。
- 可以按 benchmark 分目录保存结果。
- 更接近 README/论文里的评测组织方式。

因此如果你想严肃比较：

```text
AR vs dLM vs Linear SS
在多任务 benchmark 上的平均 accuracy
```

更应该用 `eval.sh`。

### 8.3 两者 accuracy 不一定完全一样

即使跑同一个 `gsm8k`，两者也可能不完全一致，原因包括：

- prompt template 不同。
- scorer 细节不同。
- 是否 strip thinking 不同。
- max token 设置不同。
- EOS early stop 设置不同。
- 是否通过 server 传了额外参数不同。
- 数据集版本或 split 处理细节不同。

所以比较时不要混用：

```text
evaluate.py 的结果和 eval.sh 的结果可以互相参考，
但最好不要直接当作同一套严格实验结果。
```

## 9. efficiency 对比：evaluate.py vs eval.sh

本项目里常见 efficiency 指标是：

```text
NFE = number of function evaluations，模型 forward 次数
TPF = tokens per forward = 生成 token 数 / NFE
```

注意：

```text
TPF 不是 tok/sec。
```

TPF 反映的是算法每次 forward 平均产出多少 token。它更像 algorithmic efficiency。

### 9.1 `evaluate.py` 的 efficiency

`evaluate.py` 直接拿模型返回的 `nfe`：

```text
out_ids, nfe = model.generate(...)
```

然后统计：

```text
TPF = total_new_tokens / total_nfe
```

优点：

- 简单直接。
- 没有 HTTP/server/load balancer 干扰。
- 适合看解码算法本身的 NFE/TPF。

缺点：

- 不统计真实 serving overhead。
- 不反映多用户并发。
- 不代表生产推理吞吐。
- 默认只有一个 Python 进程，通常只用一张 GPU。

### 9.2 `eval.sh` 的 efficiency

`eval.sh` 路径会让 worker server 记录 NFE：

```text
nfe_log.jsonl
```

评测结束后再合并进 metrics。

它的 efficiency 更适合描述：

```text
在 NeMo-Skills 多任务评测链路中，
每个样本生成用了多少 NFE，
平均 TPF 是多少。
```

优点：

- 可以覆盖多 benchmark。
- 每个 benchmark 有独立 NFE 记录。
- 多 GPU worker 可以并行处理多个请求。
- 输出日志更完整。

缺点：

- 默认主要仍是 NFE/TPF，不是严格 tok/sec serving benchmark。
- HTTP server 和 load balancer 存在 overhead，但 metrics 重点仍是生成层 NFE。
- `SERVER_BATCH_SIZE` 默认是 1，不等于高并发生产 serving。
- 不是 SGLang runtime，不包含 SGLang scheduler/kernel/quantization 的真实效果。

### 9.3 如果要测 tok/sec 怎么办

如果你关心的是：

```text
每秒输出多少 token
多用户并发下吞吐如何
单请求延迟如何
SGLang 比原生 Python 快多少
FP8/INT4 后速度如何
```

那应该用 serving benchmark，而不是只看 `evaluate.py` 或 `eval.sh` 的 TPF。

SGLang 相关路径在：

```text
sglang_spark/
```

## 10. 为什么 eval.sh 要这么复杂

你可能会问：

```text
既然 evaluate.py 已经能算 accuracy 和 TPF，
为什么还要 eval.sh？
```

主要原因有 5 个。

### 10.1 正式 benchmark 很复杂

HumanEval、MBPP、LiveCodeBench、MMLU、IFEval、AIME、GPQA 等任务，每个都有自己的：

- 数据格式
- prompt
- 输出解析
- scorer
- 依赖包
- 结果保存方式

把这些都写进一个轻量 `evaluate.py` 会很臃肿。NeMo-Skills 已经有这些能力，所以 `eval.sh` 复用它。

### 10.2 集群上要标准化环境

正式评测通常要跑很久，依赖很多。用 container 可以避免：

- 今天环境能跑，明天 pip 版本变了不能跑。
- A 用户环境和 B 用户环境不一致。
- 某个 benchmark 需要额外包但本地没装。

### 10.3 多 GPU 并行跑任务

`evaluate.py` 是单进程，通常一张 GPU。

`eval.sh` 可以：

- 一个 benchmark 一个 SLURM job。
- 一个 job 内多个 worker。
- 多个 benchmark 互相独立提交。

这样适合大规模 sweep。

### 10.4 评测过程需要日志和失败恢复

`eval.sh` 会组织输出目录，包括：

```text
pipeline_group<N>.log
results/eval-results/<task>/
nfe_group<N>/nfe_log.jsonl
server_info_group<N>.env
COMPLETED_group<N> 或 FAILED_group<N>
```

这比一个本地 stdout 更适合长期实验管理。

### 10.5 通过 HTTP 模拟“模型服务被评测客户端调用”

NeMo-Skills 通常通过 OpenAI-compatible API 调模型。`eval.sh` 路径把 NLD 模型包装成一个 OpenAI-compatible server，使它能接入同一套评测框架。

## 11. eval.sh 不是 SGLang

这一点很重要。

`eval.sh` 启动的是：

```text
xp/dlm_api/dlm_batch_server.py
xp/dlm_api/dlm_load_balancer.py
```

不是：

```text
sglang.launch_server
```

所以 `eval.sh` 不会测到这些 SGLang 特性：

- SGLang scheduler。
- SGLang 的 request batching。
- SGLang 的 KV cache 管理。
- FlashInfer attention backend 在 SGLang 中的使用。
- SGLang CUDA graph capture。
- SGLang FP8/INT4 serving 量化路径。
- SGLang 的真实 OpenAI API serving throughput。

`eval.sh` 的 HTTP server 是项目自己的轻量 worker，用于接 NeMo-Skills 评测，不是生产级 SGLang runtime。

## 12. 如果我关心 SGLang，是否需要 eval.sh？

分情况。

### 12.1 不需要 eval.sh 的情况

如果你的目标是：

- 看 SGLang 怎么部署 NLD。
- 看 SGLang 的 LinearSpec / FastDiffuser / AR 怎么配置。
- 看 serving API 怎么调。
- 测 SGLang 的并发吞吐、延迟。
- 测 FP8 或 INT4 量化服务。
- 看 SGLang 分支中的 DLLM scheduler 实现。

那么：

```text
不需要先跑 eval.sh。
```

你应该直接看：

```text
sglang_spark/README.md
sglang_spark/launch_server.sh
```

以及对应的 SGLang fork。

### 12.2 eval.sh 有帮助的情况

如果你的目标是：

- 在上 SGLang 前确认模型权重、tokenizer、chat template 正常。
- 先确认 AR/dLM/Linear SS 三种模式 accuracy 没明显异常。
- 用 NeMo-Skills 在多个任务上跑 baseline。
- 比较不同 decoding mode 的 benchmark accuracy 和 NFE/TPF。

那么：

```text
eval.sh 有帮助，但它不是 SGLang serving benchmark。
```

一个合理流程是：

```text
1. chat smoke test
2. evaluate.py 小样本
3. eval.sh 跑正式 accuracy/NFE benchmark
4. sglang_spark 跑 SGLang serving
5. 用 SGLang 自己的 benchmark 看 tok/sec、latency、concurrency
```

如果你只关心 SGLang serving，可以跳过第 3 步。

## 13. 用一个类比理解三者

可以用“考试/厨房/餐厅”类比：

### `evaluate.py`

像你自己在厨房试菜：

```text
拿一道题 -> 直接问模型 -> 自己看答案 -> 记一下用了几次 forward
```

简单、快、适合调试。

### `eval.sh`

像组织正式考试：

```text
申请考场 -> 统一环境 -> 安排监考 -> 发试卷 -> 统一打分 -> 保存成绩单和日志
```

复杂，但标准化、可批量、适合正式 benchmark。

### SGLang

像开餐厅：

```text
来了很多顾客 -> 排队调度 -> 厨房并行出菜 -> 关心上菜速度、吞吐、延迟、成本
```

它关注的是服务系统，而不仅是题目对错。

## 14. 该怎么选

### 你是第一次跑这个项目

用：

```bash
python evaluate.py --mode dlm --tasks gsm8k --limit 5
```

或者先跑 `chat/`。

### 你要调试某个解码参数

用：

```text
evaluate.py
```

因为它最短路径，报错也最容易定位。

### 你要跑完整 benchmark

用：

```text
eval.sh
```

前提是你有 SLURM、容器镜像、account。

### 你要研究 SGLang serving

用：

```text
sglang_spark/
```

不要把 `eval.sh` 当作 SGLang benchmark。

## 15. eval.sh 的 dry-run 有什么用

当前机器上 `srun/sbatch` 不在 `PATH`，但可以运行：

```bash
bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --dry-run
```

dry-run 会打印：

- 选择的 mode。
- 模型路径。
- engine / generation algorithm。
- block length。
- threshold。
- tokens to generate。
- max model len。
- 是否 Linear Spec。
- 是否 LoRA。
- 输出目录。
- benchmark 列表。

它不会：

- 提交 SLURM 作业。
- 启动容器。
- 加载模型。
- 跑评测。

所以 dry-run 的意义是：

```text
检查 eval.sh 会如何解释你的参数。
```

## 16. eval.sh 真实运行的最小概念命令

真实运行需要把下面两项换成你自己的：

```text
<your_slurm_account>
<path_to_container.sqsh>
```

例如：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

这条命令不是“直接运行评测”，而是：

```text
提交一个 SLURM 作业。
```

提交成功后，你需要看 SLURM job id、pipeline log、输出目录里的 `COMPLETED` 或 `FAILED`。

## 17. eval.sh 结果目录怎么看

默认输出根目录：

```text
eval_suit_results/
```

典型结构：

```text
eval_suit_results/<exp_name>/hf_base/<eval_dir_name>/
  pipeline_group<N>.log
  results/eval-results/<task>/
  nfe_group<N>/nfe_log.jsonl
  server_info_group<N>.env
  COMPLETED_group<N> 或 FAILED_group<N>
```

你优先看：

1. `pipeline_group<N>.log`：完整作业日志。
2. `results/eval-results/<task>/metrics.json`：任务指标。
3. `nfe_group<N>/nfe_log.jsonl`：每批请求的 NFE。
4. `FAILED_group<N>`：如果失败，说明这个 group 没跑完。

## 18. 总结表

| 问题 | evaluate.py | eval.sh | SGLang |
| --- | --- | --- | --- |
| 是否需要 SLURM | 不需要 | 需要 | 不一定，取决于部署方式 |
| 是否需要容器 | 不需要 | 需要 `.sqsh` | SGLang Spark 路径用 Docker |
| 是否直接调用模型函数 | 是 | worker 内部最终会调用，但外层走 HTTP | SGLang runtime 调用 |
| 是否用 NeMo-Skills | 否 | 是 | 否 |
| 是否适合 smoke test | 最适合 | 不适合入门 | 可用于 serving smoke |
| 是否适合完整 benchmark | 不完整 | 适合 | 不是主要用途 |
| 是否测真实 serving 吞吐 | 否 | 不严格 | 是 |
| 是否反映 SGLang 性能 | 否 | 否 | 是 |
| 主要指标 | accuracy、NFE、TPF | benchmark metrics、NFE、TPF、日志 | tok/sec、latency、concurrency、serving throughput |

最终建议：

```text
先用 evaluate.py 学会模型和解码参数。
需要正式多 benchmark 时再理解 eval.sh。
研究 SGLang serving 时，不要把 eval.sh 当成 SGLang 替代品，直接进入 sglang_spark/ 和 SGLang benchmark。
```
