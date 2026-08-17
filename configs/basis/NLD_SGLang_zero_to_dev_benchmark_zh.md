# Nemotron-Labs-Diffusion 本地 SGLang 部署、评测与开发优化指南

本文取代旧版 `NLD_SGLang_zero_to_dev_benchmark_zh.md`。旧版文档里混入了早期排障提案、DGX Spark 容器路径和后来废弃的 `sglang_full` 目录，容易误导。本文只保留当前已经在这台 A100 服务器上走通、并且适合后续修改 SGLang 源码做 Linear SS / LinearSpec 解码效率优化的本地开发路径。

本文默认项目根目录为：

```bash
/data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

模型权重目录为：

```bash
/data1/linyewei/models/Nemotron-Labs-Diffusion-8B
```

当前机器是 4 张 A100 80GB PCIe。最后一次检查时 4 张卡都空闲，但每次启动前仍应先看 `nvidia-smi`，再用 `CUDA_VISIBLE_DEVICES=0`、`CUDA_VISIBLE_DEVICES=1` 等显式指定 GPU。

## 1. 先理解这条路径在做什么

SGLang 在这里不是一个不能修改的远端黑盒。我们本地 clone 了 SGLang fork，并把它以 editable 方式安装到 conda 环境 `nld_sglang` 中。也就是说：

- 运行时的 `import sglang` 指向本地目录 `sglang_dllm/src/sglang/python/sglang`。
- 修改 Python 层 scheduler、DLLM algorithm、Nemotron 模型实现后，重启 SGLang server 即可生效。
- 修改 CUDA / Triton / FlashInfer 相关 kernel 或 CUDA graph 捕获逻辑后，通常也需要重启 server；如果 JIT 缓存影响结果，还要清理对应缓存后再测。
- SGLang server 通过 HTTP / OpenAI-compatible API 对外提供服务，这是 serving 引擎的正常接口，不代表内部逻辑不能改。

这条路径和项目里的 `evaluate.py` 不同：

- `evaluate.py` 是单进程 PyTorch / Transformers 路径：加载 HF 模型，直接调用 `model.ar_generate`、`model.generate`、`model.linear_spec_generate`，适合验证权重、模型文件、reference 解码正确性。
- 本文 SGLang 路径是本地 serving 路径：先启动 `sglang.launch_server`，再用 HTTP client 做 smoke test、accuracy eval 或 serving benchmark。它才是后续研究 serving 场景、不同 request、不同并发度、CUDA graph、attention backend、kernel 优化的主线。
- `eval.sh` 是 README 中面向 SLURM + enroot/pyxis + NeMo-Skills 的大规模集群评测编排，不是你当前做本地 SGLang 引擎开发的必要路径。

## 2. 当前已完成状态

当前已经整理后的目录结构如下：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/
├── sglang_dllm/
│   ├── src/
│   │   └── sglang/                 # 当前唯一保留的 SGLang 本地源码
│   ├── logs/                       # server 日志
│   ├── hf_cache/                   # Hugging Face cache
│   ├── sglang_cache/               # SGLang / JIT cache
│   ├── bench_results/              # smoke、accuracy、serving benchmark 输出
│   ├── linear_spec_lora/           # LinearSpec draft LoRA
│   └── linearspec_lora_host.yaml   # 本机路径版 LinearSpec LoRA 配置
└── configs/
    └── NLD_SGLang_zero_to_dev_benchmark_zh.md
```

注意：此前临时存在过 `sglang_dllm/src/sglang_full` 和一个早期错误安装用的 `sglang`，现在已经清理为只保留：

```bash
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang
```

当前 SGLang 源码状态：

```text
repo:   /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang
branch: upstream/2-dllm-lora-ar
commit: ce4286e
local change:
  python/sglang/srt/dllm/mixin/scheduler.py
  把 self.report_prefill_stats(...) 修为 self.metrics_reporter.report_prefill_stats(...)
```

当前 conda 环境：

```text
conda env:    nld_sglang
python:       3.12.13
torch:        2.11.0+cu130
torch cuda:   13.0
torchvision:  0.26.0+cu130
torchaudio:   2.11.0+cu130
sglang:       0.0.0.dev1+gce4286ed0.d20260606
sglang path:  /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/__init__.py
sgl_kernel:   0.4.2.post2, real site-packages wheel
flashinfer:   0.6.11.post1
transformers: 5.8.1
kernels:      0.14.1
human-eval:   1.0.3
pip check:    No broken requirements found.
```

`human-eval` 是为了运行 HumanEval accuracy benchmark 后补装的轻量依赖。它不是启动 SGLang server 的必要依赖。

## 3. 从零复现当前本地配置

如果你只是继续使用当前机器，通常不需要重新做本节；本节用于说明当前配置是怎么一步步来的，也用于将来重建环境。

### 3.1 建立项目内工作目录

命令：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion && mkdir -p sglang_dllm/src sglang_dllm/logs sglang_dllm/hf_cache sglang_dllm/sglang_cache sglang_dllm/bench_results
```

解释：

- `cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion`：先进入当前项目根目录，保证后面的相对路径建在项目内。
- `sglang_dllm/src`：放 SGLang 源码。
- `sglang_dllm/logs`：放 server 日志。
- `sglang_dllm/hf_cache`：放模型和 tokenizer 的 HF cache。
- `sglang_dllm/sglang_cache`：放 SGLang / JIT 相关 cache。
- `sglang_dllm/bench_results`：放 smoke test、accuracy eval、serving benchmark 输出。

设置两个常用路径变量：

```bash
export NLD_ROOT=/data/home/wly/dLLM/Nemotron-Labs-Diffusion && export NLD_SGLANG_WORK_DIR=${NLD_ROOT}/sglang_dllm && export NLD_MODEL=/data1/linyewei/models/Nemotron-Labs-Diffusion-8B
```

解释：

- `NLD_ROOT`：项目根目录。
- `NLD_SGLANG_WORK_DIR`：本文所有 SGLang 源码、日志、cache、结果所在目录。
- `NLD_MODEL`：Nemotron-Labs-Diffusion-8B 权重路径。

### 3.2 下载 SGLang 本地源码

命令：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src && git clone -b upstream/2-dllm-lora-ar https://github.com/hutm/sglang.git sglang
```

解释：

- `hutm/sglang` 的 `upstream/2-dllm-lora-ar` 分支来自 SGLang Nemotron DLLM onboarding 相关工作，是 README 和 issue `sgl-project/sglang#25802` 指向的开发分支。
- 这个分支包含 Nemotron-Labs-Diffusion 的 SGLang 模型实现、DLLM scheduler、FastDiffuser、LinearSpec、LoRA-aware LinearSpec 等代码。
- 目录命名为 `sglang`，后续 editable install 和源码修改都基于这个目录。

检查当前源码版本：

```bash
git -C /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang branch --show-current && git -C /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang rev-parse --short HEAD && git -C /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang status --short
```

当前预期输出应显示：

```text
upstream/2-dllm-lora-ar
ce4286e
 M python/sglang/srt/dllm/mixin/scheduler.py
```

`M python/sglang/srt/dllm/mixin/scheduler.py` 是当前必要的一行修复，不是错误。

### 3.3 创建 SGLang 专用 conda 环境

当前环境名是 `nld_sglang`。不要在已经跑通 `/chat` 和 `evaluate.py` 的 `nld` 环境里继续折腾 SGLang full 依赖，因为 SGLang 当前分支需要更新的 torch、sglang-kernel、flashinfer、transformers 等组合。

命令：

```bash
conda create -y -n nld_sglang -c conda-forge python=3.12 pip rust ninja cmake
```

解释：

- `python=3.12`：当前已验证版本是 Python 3.12.13。
- `rust`：SGLang 的部分 Python 扩展使用 Rust / PyO3。
- `ninja`、`cmake`：FlashInfer、CUDA / C++ 扩展和 JIT 编译常用工具。

激活环境：

```bash
conda activate nld_sglang
```

升级基础打包工具：

```bash
pip install --upgrade pip setuptools wheel setuptools-rust
```

### 3.4 安装 SGLang full 依赖

先安装 CUDA 13.0 对应的 PyTorch 2.11 组合：

```bash
conda activate nld_sglang && pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

解释：

- 当前 `sglang` 分支的 `python/pyproject.toml` 要求 `torch==2.11.0`、`transformers==5.8.1`、`sglang-kernel==0.4.2.post2`。
- 这里不是为了“强行维持旧 torch”，而是为了匹配当前 SGLang full 分支本身的依赖。
- A100 服务器驱动足够新，可以运行 CUDA 13.0 wheel。

安装 SGLang Python 包和依赖：

```bash
conda activate nld_sglang && cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang && pip install --extra-index-url https://download.pytorch.org/whl/cu130 -e "python"
```

解释：

- `-e "python"`：editable install，本地源码修改后不需要重新打 wheel。
- `--extra-index-url https://download.pytorch.org/whl/cu130`：让 resolver 能找到 CUDA 13.0 的 torch 相关 wheel。
- 如果这一步重新解析依赖导致 torch 版本异常，先停下来检查，不要继续在错误环境里叠包。

强制换成已验证的 CUDA 13.0 `sglang-kernel` wheel：

```bash
conda activate nld_sglang && pip install --force-reinstall --no-deps "https://github.com/sgl-project/whl/releases/download/v0.4.2.post2/sglang_kernel-0.4.2.post2+cu130-cp310-abi3-manylinux2014_x86_64.whl"
```

解释：

- 这个 wheel 名字里有 `cp310-abi3`，但它是 Python stable ABI wheel，当前 Python 3.12 可以使用。
- 这样可以避免装到错误的 CPU / 非 cu130 kernel 包。

固定 `kernels==0.14.1`：

```bash
conda activate nld_sglang && pip install --force-reinstall kernels==0.14.1
```

解释：

- 当前环境验证过 `kernels==0.14.1` 能和 `transformers==5.8.1` 正常配合。
- 更新到更高版本时曾遇到 `LayerRepository` revision/version 相关兼容问题，因此这里保守固定。

安装 SOCKS proxy 支持和 HumanEval 评分依赖：

```bash
conda activate nld_sglang && pip install socksio human-eval
```

解释：

- `socksio`：如果 shell 里配置了 SOCKS proxy，OpenAI-compatible client / benchmark client 需要它，否则可能在 HTTP 请求时报 proxy 相关错误。
- `human-eval`：只用于 SGLang 自带 `simple_eval_humaneval.py` 的代码题评分。

最后重新做一次 editable install，但不让 pip 再改依赖：

```bash
conda activate nld_sglang && cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang && pip install -e "python" --no-deps
```

解释：

- 这一步保证 `import sglang` 指向本地源码。
- `--no-deps` 避免 resolver 再次修改 torch、transformers、sglang-kernel 等已经确认过的版本。

检查依赖一致性：

```bash
conda activate nld_sglang && pip check
```

预期输出：

```text
No broken requirements found.
```

检查关键包实际导入位置：

```bash
conda activate nld_sglang && python -c "import sys, torch, sglang, sgl_kernel, flashinfer, transformers; print('python', sys.version.split()[0]); print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('sglang', getattr(sglang, '__version__', 'NA'), sglang.__file__); print('sgl_kernel', getattr(sgl_kernel, '__version__', 'NA'), sgl_kernel.__file__); print('flashinfer', getattr(flashinfer, '__version__', 'NA')); print('transformers', transformers.__version__)"
```

当前预期关键信息：

```text
python 3.12.13
torch 2.11.0+cu130 13.0 True
sglang ... /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/__init__.py
sgl_kernel ... /data/home/wly/.conda/envs/nld_sglang/lib/python3.12/site-packages/sgl_kernel/__init__.py
flashinfer 0.6.11.post1
transformers 5.8.1
```

### 3.5 确认当前必要源码修复

当前本地源码有一个必要修复，文件是：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

修复点：

```python
self.metrics_reporter.report_prefill_stats(...)
```

而不是：

```python
self.report_prefill_stats(...)
```

检查命令：

```bash
rg -n "report_prefill_stats|metrics_reporter.report_prefill_stats" /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

解释：

- `Scheduler` 本身没有 `report_prefill_stats` 方法。
- 正确调用路径是 `self.metrics_reporter.report_prefill_stats(...)`。
- 如果重新 clone 分支后没有这处修复，LinearSpec server 可能在处理 DLLM prefill stats 时出错。

语法检查命令：

```bash
conda activate nld_sglang && python -m py_compile /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

### 3.6 准备 LinearSpec LoRA

当前 LoRA 文件已经放在：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora/
├── adapter_config.json
└── adapter_model.safetensors
```

如果重建环境，可以从模型目录复制：

```bash
mkdir -p /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora && cp /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_config.json /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora/adapter_model.safetensors /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora/
```

创建本机路径版 LinearSpec 配置：

```bash
printf 'algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\n' > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_host.yaml
```

解释：

- `algorithm: LinearSpec`：启用 Linear self-speculation，也就是 diffusion draft + AR verify。
- `causal_context: true`：Nemotron-Labs-Diffusion 这个模型在 prefix KV context 上需要 causal 语义，和 HF reference 的 `causal_context=True` 对齐。
- `lora_path`：draft LoRA 的本机路径。
- `lora_mode: draft_only`：只在 draft pass 使用 LoRA，verify pass 使用 base 权重。这是当前已验证配置，也是优化 LinearSpec draft 效率时最重要的路径。

## 4. 启动本地 SGLang server

启动前先看 GPU：

```bash
nvidia-smi
```

选择空闲 GPU 后，用 `CUDA_VISIBLE_DEVICES=0` 显式指定。下面命令使用 GPU 0、端口 30000、bf16 权重、LinearSpec + LoRA。

### 4.1 当前已验证的 LinearSpec + LoRA 启动命令

单行命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_host.yaml --cuda-graph-bs 1 2 3 4 --context-length 2048 --host 0.0.0.0 --port 30000
```

关键参数解释：

- `PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH`：确保 server 进程能找到当前 conda 环境里的 `python`、`ninja` 等工具。FlashInfer 首次 JIT 编译时需要 `ninja`。
- `CUDA_VISIBLE_DEVICES=0`：只让 server 看到物理 GPU 0。如果要用 GPU 1，改成 `CUDA_VISIBLE_DEVICES=1`。
- `HF_HOME=.../hf_cache`：把 HF cache 固定在项目内，避免污染默认 home cache。
- `SGLANG_CACHE_DIR=.../sglang_cache`：把 SGLang cache 固定在项目内。
- `python -m sglang.launch_server`：启动 SGLang server。
- `--model-path`：指向本地 Nemotron-Labs-Diffusion-8B 权重。
- `--trust-remote-code`：允许加载模型目录中的自定义 HF 代码；这个模型需要。
- `--dtype bfloat16`：A100 上当前验证使用 bf16。不要把 DGX Spark 文档里的 FP8 当成 A100 必需路径。
- `--tensor-parallel-size 1`：单卡启动。多卡 TP 不是当前第一阶段优化重点。
- `--mem-fraction-static 0.55`：SGLang 预留静态显存比例。上下文长度或并发增大时需要调。
- `--max-running-requests 1`：当前 smoke 已验证的保守值。做 serving 并发实验时要增大到 2、4、8 等。
- `--attention-backend flashinfer`：DLLM 需要 bidirectional / ENCODER_ONLY attention，当前代码明确要求 DLLM 使用 FlashInfer backend。
- `--sampling-backend flashinfer`：采样也使用 FlashInfer。
- `--dllm-algorithm LinearSpec`：进入 LinearSpec 解码路径。
- `--dllm-algorithm-config ...linearspec_lora_host.yaml`：载入 LoRA、`causal_context`、`lora_mode` 等算法配置。
- `--cuda-graph-bs 1 2 3 4`：捕获 batch size 1、2、3、4 的 CUDA graph。后续并发实验如果要测 `max_running_requests=8`，这里也应包含 8。
- `--context-length 2048`：最大上下文长度。先用 2048 做稳定性和效率实验，之后再扩。
- `--host 0.0.0.0 --port 30000`：监听 30000 端口。

首次启动会看到 FlashInfer 为 A100 `sm_80` JIT 编译 kernel，第一次较慢，后续会走 cache。启动成功的日志里应该出现类似信息：

```text
type=NemotronLabsDiffusionModel
attention_backend='flashinfer'
sampling_backend='flashinfer'
LinearSpec LoRA loaded: 34 delta tensors
LinearSpec dual weights built from LoRA
LinearSpec: CUDA graph capture done (LoRA baked)
Uvicorn running on http://0.0.0.0:30000
```

### 4.2 FastDiffuser 启动方式

FastDiffuser 是迭代式 dLLM denoising 路径，不是 LinearSpec。它每个 block 可能做多步 forward，适合作为 dLLM baseline。

先写配置：

```bash
printf 'algorithm: FastDiffuser\ncausal_context: true\ntemperature: 0.0\nthreshold: 0.9\nmax_steps: 32\n' > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_host.yaml
```

启动命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm FastDiffuser --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_host.yaml --cuda-graph-bs 1 2 3 4 --context-length 2048 --host 0.0.0.0 --port 30000
```

解释：

- `threshold: 0.9`：和项目 `evaluate.py --mode dlm` 的默认 dLLM 阈值语义接近。
- `max_steps: 32`：每个 block 的最大 denoising 步数。越大越接近充分迭代，forward 次数也越多。
- FastDiffuser 用于对照 LinearSpec 是否显著减少 forward 次数和提升 serving 指标。

### 4.3 AR 模式启动方式

AR 模式在 SGLang fork 里通过修改 HF config 的 `ar_mode=true` 让 Nemotron attention 变为 causal。真正的 native AR serving 不需要 `--dllm-algorithm`，因为它应该走 SGLang 标准 causal decode scheduler，而不是 DLLM block scheduler。

启动命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --json-model-override-args '{"ar_mode": true}' --cuda-graph-bs 1 2 3 4 --context-length 2048 --host 0.0.0.0 --port 30000
```

解释：

- `--json-model-override-args '{"ar_mode": true}'`：在加载模型 config 时强制 `ar_mode=True`。
- `nemotron_labs_dllm.py` 中 `NemotronLabsDiffusionEncoder` 会读取 `config.ar_mode`，为每层 attention 选择 `AttentionType.DECODER` 或 `AttentionType.ENCODER_ONLY`。
- 这里不要传 `--dllm-algorithm FastDiffuser`。如果传了，它会变成“causal attention + FastDiffuser/DLLM scheduler”的混合路径，stats 也会变成 FastDiffuser block stats，不是纯 AR 逐 token stats。

## 5. Smoke test：确认 server 真能服务

下面所有 smoke test 都要求第 4 节的 server 已经在另一个终端运行。

### 5.1 健康检查

命令：

```bash
curl -fsS http://127.0.0.1:30000/health
```

解释：

- `curl` 请求 SGLang 的 health endpoint。
- `-f` 表示 HTTP 非 2xx 时直接返回失败。
- `-sS` 表示安静模式，但出错仍打印错误。
- 这个 endpoint 可能没有正文输出；只要命令退出码为 0，就是 server 活着。

### 5.2 OpenAI-compatible chat 请求

命令：

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"/data1/linyewei/models/Nemotron-Labs-Diffusion-8B","messages":[{"role":"user","content":"What is 15% of 240? Answer briefly."}],"max_tokens":64,"temperature":0}'
```

解释：

- `/v1/chat/completions` 是 OpenAI-compatible chat endpoint。
- `model` 要和启动 server 的 `--model-path` 保持一致。
- `messages` 是标准 chat 格式。
- `max_tokens=64` 限制最多生成 64 个 token，smoke test 不需要长输出。
- `temperature=0` 做确定性 greedy 输出。

当前已验证返回中包含类似：

```text
15% of 240 = 36
```

### 5.3 最小 serving benchmark

命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --random-input-len 64 --random-output-len 32 --num-prompts 4 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/full_linearspec_smoke_random_4.jsonl --disable-tqdm --seed 1
```

解释：

- `python -m sglang.bench_serving`：SGLang 自带 serving benchmark client。
- `--backend sglang-oai-chat`：通过 OpenAI-compatible chat API 压测本地 SGLang server。
- `--base-url http://127.0.0.1:30000`：注意这里不带 `/v1`，`bench_serving` 会自己拼接 endpoint。
- `--dataset-name random`：使用合成随机 prompt，不涉及任务 accuracy。
- `--random-input-len 64`：每个 prompt 约 64 token。
- `--random-output-len 32`：每个请求要求生成约 32 token。
- `--num-prompts 4`：只发 4 个请求，是 smoke test，不是性能结论。
- `--request-rate inf`：尽快发请求。
- `--max-concurrency 1`：客户端最多同时 1 个未完成请求。
- `--output-file ...jsonl`：保存 benchmark 原始指标。

当前已验证这条 smoke benchmark 可以跑通。之前一次结果大致是 4 个请求全部成功，mean TTFT 约 58 ms，mean TPOT 约 6 ms。这个数只说明链路正常，不能作为最终性能 baseline。

## 6. Accuracy benchmark：GSM8K、MATH-500、HumanEval

Accuracy benchmark 的逻辑是：

1. 先启动本地 SGLang server。
2. 再用 SGLang 自带或自写的 OpenAI-compatible client 请求 server。
3. client 收到输出后做任务评分。

这和 `bench_serving` 不同。`bench_serving` 主要看 latency、throughput、TTFT、TPOT、ITL，不负责判断答案是否正确。

### 6.1 GSM8K accuracy

GSM8K 可以直接用 SGLang 自带 `sglang.test.run_eval`。

小样本 smoke：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.test.run_eval --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --eval-name gsm8k --num-examples 20 --num-threads 8 --max-tokens 512 --temperature 0 --num-shots 5
```

解释：

- `OPENAI_API_KEY=EMPTY`：OpenAI Python client 要求环境变量存在；请求的是本地 SGLang，不会调用 OpenAI。
- `--base-url http://127.0.0.1:30000`：`run_eval.py` 内部会自动拼成 `http://127.0.0.1:30000/v1`。
- `--eval-name gsm8k`：选择 GSM8K eval object。
- `--num-examples 20`：只测 20 题，先确认链路和输出格式。
- `--num-threads 8`：client 并发线程数。server 如果仍是 `--max-running-requests 1`，多线程请求会在 server 侧排队；这不等于 GPU batch 并发。
- `--max-tokens 512`：给数学推理留足生成长度。
- `--num-shots 5`：GSM8K eval 默认 few-shot 数，和 `simple_eval_gsm8k.py` 逻辑一致。

全量 GSM8K：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.test.run_eval --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --eval-name gsm8k --num-threads 32 --max-tokens 512 --temperature 0 --num-shots 5
```

输出位置：

- HTML report：`/tmp/gsm8k_<model_path_rewritten>.html`
- JSON metrics：`/tmp/gsm8k_<model_path_rewritten>.json`

### 6.2 MATH-500 accuracy

SGLang 自带 `run_eval --eval-name math` 默认读的是 OpenAI simple-evals 的 `math_test.csv`，而且默认用 `gpt-4-turbo` 当等价性判分器。这不是你要的本地闭环 MATH-500。

如果你关心和项目 `evaluate.py --tasks math-500` 对齐，应使用 HuggingFace `HuggingFaceH4/MATH-500`，把列转换成 SGLang `MathEval` 需要的 `Question` / `Answer` CSV，然后让本地 SGLang server 同时负责作答和等价性判分。

先准备 MATH-500 CSV：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import pandas as pd; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_sglang_simple_eval.csv'; ds=load_dataset('HuggingFaceH4/MATH-500', split='test'); pd.DataFrame({'Question': ds['problem'], 'Answer': ds['answer']}).to_csv(out, index=False); print(out, len(ds))"
```

解释：

- `load_dataset('HuggingFaceH4/MATH-500', split='test')`：读取和 `evaluate.py` 一致的数据集。
- `Question`：SGLang `MathEval` 的 prompt template 里使用 `{Question}`。
- `Answer`：SGLang `MathEval` 用 `row["Answer"]` 做 gold answer。
- 输出 CSV 放在 `bench_results`，后续可复用。

小样本 MATH-500：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from sglang.test.simple_eval_common import ChatCompletionSampler; from sglang.test.simple_eval_math import MathEval; base='http://127.0.0.1:30000/v1'; model='/data1/linyewei/models/Nemotron-Labs-Diffusion-8B'; answer=ChatCompletionSampler(base_url=base, model=model, temperature=0.0, max_tokens=1024); judge=ChatCompletionSampler(base_url=base, model=model, temperature=0.0, max_tokens=64); result=MathEval('/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_sglang_simple_eval.csv', judge, 20, 4)(answer); print({'score': result.score, **result.metrics})"
```

解释：

- `answer` sampler：向本地 SGLang server 请求完整解题输出。
- `judge` sampler：仍请求本地 SGLang server，让同一个模型判断 gold answer 和 extracted answer 是否等价。
- `MathEval(..., 20, 4)`：随机采样 20 道题，用 4 个 client 线程。
- 这种本地 judge 方式不等价于 `evaluate.py` 的 `\boxed{...}` 字符串匹配，也不等价于远端 GPT-4 judge；它适合做 SGLang 闭环 smoke。正式报告时要明确评分器。

全量 MATH-500：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from sglang.test.simple_eval_common import ChatCompletionSampler; from sglang.test.simple_eval_math import MathEval; base='http://127.0.0.1:30000/v1'; model='/data1/linyewei/models/Nemotron-Labs-Diffusion-8B'; answer=ChatCompletionSampler(base_url=base, model=model, temperature=0.0, max_tokens=1024); judge=ChatCompletionSampler(base_url=base, model=model, temperature=0.0, max_tokens=64); result=MathEval('/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_sglang_simple_eval.csv', judge, None, 16)(answer); print({'score': result.score, **result.metrics})"
```

### 6.3 HumanEval accuracy

HumanEval 会执行模型生成的 Python 代码来跑单元测试。只在隔离的本地评测环境里运行，不要把它接到不可信外部请求。

小样本 smoke：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.test.run_eval --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --eval-name humaneval --num-examples 10 --num-threads 4 --max-tokens 512 --temperature 0.2
```

解释：

- `--eval-name humaneval`：使用 `sglang.test.simple_eval_humaneval.HumanEval`。
- `--num-examples 10`：随机测 10 个任务，先验证生成、抽取代码、执行测试都正常。
- `--temperature 0.2`：HumanEval 代码生成常用非零温度；如果只想确定性 smoke，可改为 0。
- `HumanEval` 默认每个 task 采样 5 次并计算 `pass@1/2/5`，所以 10 个 task 会发约 50 次生成请求。

全量 HumanEval：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.test.run_eval --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --eval-name humaneval --num-threads 16 --max-tokens 512 --temperature 0.2
```

解释：

- 不传 `--num-examples` 时会跑完整 HumanEval。
- HumanEval 全量 164 个 task，默认每题 5 samples，总请求数约 820。
- 如果 server 仍是 `--max-running-requests 1`，client 线程数再高也主要是在排队；要评估并发 serving，需要同时提高 server 端 `--max-running-requests`。

## 7. Serving benchmark：测 latency、throughput、并发

Serving benchmark 的目标不是 accuracy，而是回答这些问题：

- 单请求时 TTFT / TPOT / ITL 是多少？
- 并发 2、4、8 时吞吐提升还是排队恶化？
- LinearSpec + LoRA 相比 FastDiffuser / LinearSpec-base 的 serving 指标如何？
- 修改 scheduler、acceptance、attention backend、CUDA graph 后性能是否真的变好？

### 7.1 server 并发参数要和 client 并发匹配

如果 server 以 `--max-running-requests 1` 启动，即使 client 写 `--max-concurrency 4`，server 也只能一次运行 1 个 request，其余请求排队。要测真正并发，server 应改成：

```text
--max-running-requests 4 --cuda-graph-bs 1 2 3 4
```

如果要测 8 并发，则应至少：

```text
--max-running-requests 8 --cuda-graph-bs 1 2 3 4 8
```

同时观察显存，不够就调小：

- `--context-length`
- `--mem-fraction-static`
- `--max-running-requests`
- prompt / output 长度

### 7.2 合成 random serving benchmark

单并发、输入 256、输出 256：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --num-prompts 128 --random-input-len 256 --random-output-len 256 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_random_c1_i256_o256.jsonl --disable-tqdm --seed 1
```

并发 4、输入 256、输出 256：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --num-prompts 128 --random-input-len 256 --random-output-len 256 --request-rate inf --max-concurrency 4 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_random_c4_i256_o256.jsonl --disable-tqdm --seed 1
```

解释：

- `--num-prompts 128`：发 128 个请求，比 smoke 更能稳定反映吞吐。
- `--request-rate inf`：客户端尽最大速度发送，适合测饱和吞吐。
- `--max-concurrency 4`：客户端最多保持 4 个 in-flight 请求。
- 如果 server 端 `--max-running-requests` 仍是 1，那么这个命令测到的是队列表现，不是真并发执行。

### 7.3 用 GSM8K prompt 做 serving benchmark

先生成 OpenAI JSONL 请求文件：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_128.jsonl'; ds=load_dataset('gsm8k','main',split='test').select(range(128)); f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':'Solve the following math problem. Put the final numerical answer inside \\\\boxed{} at the very end.\\n\\n'+r['question']}], 'max_tokens':512, 'temperature':0})+'\\n') for r in ds]; f.close(); print(out)"
```

解释：

- `dataset-name openai` 要求 JSONL 每行是一个 OpenAI-compatible request。
- 每行包含 `messages`、`max_tokens`、`temperature`。
- 这里不是评分，只是用真实 GSM8K prompt 形状测 serving 延迟和吞吐。

压测命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_128.jsonl --num-prompts 128 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_gsm8k_c1_128.jsonl --disable-tqdm
```

把 `--max-concurrency 1` 改成 2、4、8，可以得到不同并发下的 serving 指标。前提是 server 端也用对应的 `--max-running-requests` 重启。

### 7.4 用 MATH-500 prompt 做 serving benchmark

生成 OpenAI JSONL：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_128.jsonl'; ds=load_dataset('HuggingFaceH4/MATH-500', split='test').select(range(128)); f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':'Solve the following math problem. Put the final answer inside \\\\boxed{} at the very end.\\n\\n'+r['problem']}], 'max_tokens':1024, 'temperature':0})+'\\n') for r in ds]; f.close(); print(out)"
```

压测命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_128.jsonl --num-prompts 128 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_math500_c1_128.jsonl --disable-tqdm
```

### 7.5 用 HumanEval prompt 做 serving benchmark

HumanEval accuracy 需要执行生成代码；serving benchmark 不执行代码，只测 prompt 形状和输出长度下的服务指标。

生成 OpenAI JSONL：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from human_eval.data import read_problems; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_128.jsonl'; problems=list(read_problems().values())[:128]; instruction='Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\\n'; f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':instruction+p['prompt']}], 'max_tokens':512, 'temperature':0.2})+'\\n') for p in problems]; f.close(); print(out)"
```

压测命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_128.jsonl --num-prompts 128 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_humaneval_c1_128.jsonl --disable-tqdm
```

### 7.6 结果怎么看

`bench_serving` 输出里重点看：

- `Successful requests`：成功请求数。必须等于 `--num-prompts`。
- `Benchmark duration`：总耗时。
- `Request throughput`：每秒完成请求数。
- `Output token throughput`：每秒生成 token 数。
- `Total Token throughput`：输入 + 输出总 token 吞吐。
- `Mean TTFT`：time to first token，首 token 延迟。
- `Mean TPOT`：time per output token，后续 token 平均间隔。
- `Mean ITL`：inter-token latency。

优化 LinearSpec serving 时，优先固定一套 benchmark matrix：

```text
算法: LinearSpec+LoRA, LinearSpec-base, FastDiffuser
并发: 1, 2, 4, 8
prompt/output: random i256/o256, GSM8K, MATH-500, HumanEval
server: 相同 dtype、context-length、mem-fraction、cuda-graph-bs
```

每次只改一个变量，否则很难判断性能变化来自哪里。

## 8. 代码结构和可修改点

你的目标是优化 SGLang 推理引擎下 Linear SS / LinearSpec 解码效率，并考虑 serving 场景并发。因此主要关注以下源码。

### 8.1 启动参数和模式选择

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/server_args.py
```

关键点：

- `--dllm-algorithm`：选择 `LinearSpec` 或 `FastDiffuser`。
- `--dllm-algorithm-config`：读取 YAML，比如 `linearspec_lora_host.yaml`。
- `--json-model-override-args '{"ar_mode": true}'`：把模型 config 改成 AR causal attention。
- `--attention-backend flashinfer`：DLLM 模式要求 FlashInfer，因为需要 ENCODER_ONLY bidirectional attention。
- `--sampling-backend flashinfer`：采样 backend。
- `--cuda-graph-bs`：CUDA graph 捕获的 batch size 集合。
- `_handle_dllm_inference()`：DLLM 特殊处理，包括禁用 overlap schedule、LinearSpec partial accept 时把 `page_size` 设为 1。

为什么 LinearSpec 会自动 `page_size=1`：

- LinearSpec 可能一次只接受 block 中前几个 token。
- 未接受 token 对应的 KV slot 需要释放。
- 如果 page size 大于 1，释放单个 slot 可能影响同页中仍在使用的 prefix slot。
- 因此当前代码对 LinearSpec 这类 partial acceptance algorithm 使用 `page_size=1`。

### 8.2 DLLM 配置解析

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/config.py
```

关键类：

```python
class DllmConfig
```

关键逻辑：

- 根据 HF config architecture 识别 DLLM 模型。
- 对 `NemotronLabsDiffusionModel` 设置：
  - `block_size = 32`
  - `mask_id = 100`
- 从 YAML 读取：
  - `block_size`
  - `max_steps`
  - `causal_context`
  - `lora_path`
  - `lora_mode`
  - `profile`
  - `stats_file`

如果你要实验不同 block size，可以在 YAML 中加：

```yaml
block_size: 16
```

但要注意：block size 改变会影响 mask block 构造、position、CUDA graph shape、acceptance 统计和吞吐，不只是一个简单超参。

### 8.3 算法注册

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/__init__.py
```

关键逻辑：

- 自动扫描 `sglang.srt.dllm.algorithm` 包下所有 `.py` 文件。
- 如果模块里有 `Algorithm` 变量，就注册到 `algo_name_to_cls`。

因此新增算法时，一般做法是：

1. 在 `sglang/srt/dllm/algorithm/` 下新增 `my_linear_spec.py`。
2. 定义 `class MyLinearSpec(DllmAlgorithm)`。
3. 文件末尾写 `Algorithm = MyLinearSpec`。
4. 启动时用 `--dllm-algorithm MyLinearSpec`。

### 8.4 LinearSpec 核心解码逻辑

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py
```

这是你最重要的优化入口。

LinearSpec 每个 block 的核心流程：

1. 如果当前 batch 没有 mask token，说明处于 prompt prefill / seed 生成阶段，走一次普通 forward，并记录每个 request 的 seed token。
2. 对每个 request 找到当前 block 中 mask token 开始位置。
3. 把上一个 block 留下的 seed token 注入到第一个 mask 位置。
4. Draft pass：
   - 使用 bidirectional attention。
   - 如果配置了 LoRA，draft pass 使用 base + LoRA 权重。
   - 得到每个 mask 位置的 draft token。
5. Verify pass：
   - 设置 `forward_batch.dllm_causal_kv_update = True`。
   - 使用 causal attention 做 AR verify。
   - 得到 AR token。
6. Accept：
   - 比较 `draft[i] == ar[i-1]`。
   - 只接受连续匹配前缀。
   - 输出 `[seed, ar[0], ..., ar[c-1]]`。
   - 未接受位置释放 KV。
7. 保存 `ar[c]` 作为下个 block 的 seed。

最值得优化的点：

- draft pass 和 verify pass 之间的数据搬运、mask 替换、argmax 是否有额外开销。
- LoRA 权重是否已经 bake 到 CUDA graph，避免每个 block 动态加减 delta。
- acceptance 统计是否能帮助调整 block size 或 draft LoRA 策略。
- `forward_batch.dllm_causal_kv_update` 的切换是否导致图捕获或 attention metadata 重建开销。
- `profile: true` 下 scheduler 与 algorithm 的 wall-clock 统计。

### 8.5 FastDiffuser 核心逻辑

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/fastdiffuser.py
```

FastDiffuser 做的是 iterative denoising：

- 根据 confidence 和 threshold 决定每步提交哪些 mask token。
- `max_steps` 控制每个 block 最多 forward 次数。
- `temperature=0` 时是 greedy argmax。
- `threshold=None` 时使用 HF 风格的 token transfer schedule。
- `causal_context=True` 时，最后 KV update pass 用 causal attention。

FastDiffuser 适合作为对照：

- 如果 LinearSpec 的 acceptance 长度高，理论上每 block 2 次 forward 能比 FastDiffuser 更高效。
- 如果 LinearSpec acceptance 很低，可能吞吐反而被 verify 和 partial accept 开销拖住。

### 8.6 DLLM scheduler 和 request 生命周期

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/req.py
sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
sglang_dllm/src/sglang/python/sglang/srt/managers/scheduler.py
```

request phase：

```text
INCOMING_PREFILL  -> 首次进入，请求需要先缓存 prompt KV
STAGING_PREFILL   -> 已在 DLLM manager 中等待 prefill
INCOMING_DECODE   -> 新进入 decode 的请求
STAGING_DECODE    -> 已在 DLLM manager 中等待 DLLM_EXTEND
```

关键调用链：

```text
HTTP request
  -> SGLang server 接收 OpenAI-compatible 请求
  -> Scheduler waiting_queue
  -> Req.init_diffusion_llm()
  -> Scheduler.init_diffusion_llm()
  -> Scheduler.get_new_batch()
  -> 如果 dllm_config 不为空，走 get_new_batch_dllm()
  -> _process_dllm_batches()
  -> _create_dllm_batch()
  -> ModelRunner forward
  -> DllmAlgorithm.run()
  -> process_batch_result_dllm()
  -> stream_output()
```

`scheduler.py` 中你要重点看：

- `get_new_batch_dllm()`：如何从 waiting / staging queue 组 batch。
- `_prepare_staging_reqs()`：每轮为 staged request 追加 mask block，并设置 prefix KV indices。
- `_process_dllm_batches()`：决定当前 batch 是 prompt cache prefill 还是 DLLM denoising。
- `process_batch_result_dllm()`：把 algorithm 返回的 accepted tokens 写回 request，释放 rejected KV，检查 EOS / stop / max_new_tokens，并流式输出。

如果要优化 serving 并发，scheduler 是核心之一。LinearSpec 的 GPU forward 可能很快，但如果 scheduler 在每个 block 之间有明显 CPU overhead，高并发下就会被放大。

### 8.7 Nemotron 模型实现和 AR / bidirectional attention 切换

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/models/nemotron_labs_dllm.py
```

关键类：

- `NemotronLabsDiffusionModel`
- `NemotronLabsDiffusionEncoder`
- `NemotronLabsDiffusionLayer`
- `NemotronLabsDiffusionAttention`

关键逻辑：

```python
causal = getattr(config, "ar_mode", False)
attn_type = AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
```

含义：

- `ar_mode=False`：使用 `AttentionType.ENCODER_ONLY`，也就是 bidirectional / full attention，适合 dLLM draft / denoise。
- `ar_mode=True`：使用 `AttentionType.DECODER`，也就是 causal attention，适合 AR baseline。
- LinearSpec 不通过全局 `ar_mode=True` 做 verify，而是在 verify pass 设置 `forward_batch.dllm_causal_kv_update=True`，让 DLLM attention backend 在特定 forward 中使用 causal mask。

### 8.8 DLLM attention causal 逻辑

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/attention.py
```

核心函数：

```python
get_dllm_causal_attention(...)
```

逻辑：

- 如果不是 DLLM，保持 backend 默认 causal 设置。
- 如果 layer 不是 `AttentionType.ENCODER_ONLY`，保持默认。
- 如果 `causal_context=False`，不启用 DLLM causal override。
- 如果 `causal_context=True`：
  - 非 `DLLM_EXTEND` pass 使用 causal。
  - 或者 `forward_batch.dllm_causal_kv_update=True` 时使用 causal。

LinearSpec verify pass 就是通过这个布尔开关进入 causal verify。

### 8.9 FlashInfer attention backend

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
```

这是 DLLM 当前最重要的 attention backend。`server_args.py` 明确要求 DLLM 使用 `flashinfer`，否则会报错。

重点看：

- extend / prefill metadata 如何初始化。
- `AttentionType.ENCODER_ONLY` 时 causal 如何设置。
- `get_dllm_causal_attention(...)` 如何影响 FlashInfer wrapper 的 `causal` 参数。
- CUDA graph capture / replay 下 metadata 是否复用。

如果要优化 attention 层效率，优先从这里和 CUDA graph runner 联动看，而不是先改模型数学逻辑。

### 8.10 CUDA graph 捕获和 replay

文件：

```text
sglang_dllm/src/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py
```

关键点：

- 对 DLLM，只有 `DLLM_EXTEND` pass 使用 CUDA graph；prompt cache 的 EXTEND pass 走 eager。
- `--cuda-graph-bs` 控制捕获哪些 batch size。
- LinearSpec + LoRA 会 defer CUDA graph capture，等 LoRA dual weights hooks 设置好后再捕获。
- `linear_spec.py` 中 `_bake_dual_weights_into_graphs()` 会设置：
  - `_dllm_pre_draft_hook`
  - `_dllm_pre_verify_hook`
- draft graph 读 LoRA-baked 权重，verify graph 根据 `lora_mode` 读 base 或 LoRA 权重。

优化注意：

- 改了 graph capture 逻辑后必须重启 server。
- 改了 `--cuda-graph-bs` 也必须重启 server。
- 如果看到某个 batch size 没走 graph，先检查 server 启动参数是否包含该 batch size，再看 `cuda_graph_runner.can_run()` 条件。

### 8.11 其他 kernel 位置

常见相关目录：

```text
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/triton_ops/
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/triton_backend.py
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/flashattention_backend.py
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/linear/
sglang_dllm/src/sglang/python/sglang/jit_kernel/
/data/home/wly/.conda/envs/nld_sglang/lib/python3.12/site-packages/sgl_kernel/
```

解释：

- `flashinfer_backend.py`：当前 DLLM 主线 backend。
- `triton_ops/`：SGLang 自带 Triton attention kernel，当前 DLLM 不作为主线 backend，但可作为参考。
- `linear/`：线性 attention 相关实现，不等于本文的 LinearSpec；名字相似但不是同一件事。
- `jit_kernel/`：SGLang Python 包内 JIT kernel。
- `site-packages/sgl_kernel/`：已安装的 `sglang-kernel` wheel。这里不是 editable 源码，直接改 site-packages 不利于版本管理；如果要长期改 C++/CUDA kernel，应转为源码构建或在 SGLang fork 中接入新 kernel。

## 9. 推荐的开发实验循环

### 9.1 每次改代码前记录 baseline

先启动原始 LinearSpec + LoRA server，然后跑：

```bash
curl -fsS http://127.0.0.1:30000/health
```

再跑一个小 serving benchmark：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --num-prompts 32 --random-input-len 128 --random-output-len 128 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/baseline_smoke_c1_i128_o128.jsonl --disable-tqdm --seed 1
```

这样每次修改后都有一个可比对的小基线。

### 9.2 打开 LinearSpec profile / TPF stats

如果目标是拆分 LinearSpec 内部耗时，写一个 profile YAML：

```bash
printf 'algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\nprofile: true\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_accept_stats.jsonl\n' > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_profile_host.yaml
```

解释：

- `profile: true`：打开 LinearSpec 和 DLLM scheduler 内部 profile。
- `stats_file`：每个 block 输出 forward 次数、接受 token 数、acceptance rate 等 JSONL。
- 代码里每 500 个 block 会打印一次平均耗时拆分，包括 scheduler、draft forward、verify forward、accept 等。

如果目标只是让 serving benchmark 同轮统计近似 decode-stage TPF，使用下面这个 TPF YAML。它只打开 `stats_file`，不打开 `profile: true`，避免 CUDA synchronize 计时影响吞吐。

```bash
printf 'algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_tpf_stats.jsonl\n' > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_tpf_host.yaml
```

这份 YAML 已在当前项目中写好：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_tpf_host.yaml
```

TPF 统计逻辑来自 SGLang 本地代码：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py
```

每个 LinearSpec decode block 会写一行 JSONL，核心字段是：

```text
forward_passes: 2
tokens: 本 block 实际接受/输出的 token 数
block_gen_positions: 本 block 候选生成位置数
acceptance_rate: tokens / block_gen_positions
```

因此近似 decode-stage TPF 是：

```text
decode_TPF = sum(tokens) / sum(forward_passes)
```

注意这不是完整 HF `evaluate.py` 口径的端到端 TPF，因为它没有把每个 request 的 prefill/seed forward 计入分母；它衡量的是 LinearSpec decode block 内部“每次 forward 平均产出多少 token”。

用 profile YAML 启动 server：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_profile_host.yaml --cuda-graph-bs 1 2 3 4 --context-length 2048 --host 0.0.0.0 --port 30000
```

### 9.3 修改 Python 层后如何验证

例如你改了：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py
```

先做语法检查：

```bash
conda activate nld_sglang && python -m py_compile /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py
```

然后重启 server，再跑：

```bash
curl -fsS http://127.0.0.1:30000/health
```

再跑小 benchmark：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name random --num-prompts 32 --random-input-len 128 --random-output-len 128 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/after_change_smoke_c1_i128_o128.jsonl --disable-tqdm --seed 1
```

解释：

- editable install 意味着 Python 文件修改后重启 server 即可生效。
- 不需要重新 `pip install -e`，除非改了 package metadata、依赖、Rust extension 或安装结构。
- 先用小 benchmark 排除功能错误，再跑完整 matrix。

### 9.4 修改 CUDA graph 或 kernel 后如何验证

如果改的是：

```text
sglang_dllm/src/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
sglang_dllm/src/sglang/python/sglang/srt/layers/attention/triton_ops/
sglang_dllm/src/sglang/python/sglang/jit_kernel/
```

建议流程：

1. 先做 Python 语法检查。
2. 重启 server。
3. 观察启动日志中 CUDA graph capture 是否成功。
4. 跑 health 和 chat smoke。
5. 跑 small random benchmark。
6. 再跑目标并发 benchmark。

如果你怀疑 JIT cache 没有重新编译，可先改 `SGLANG_CACHE_DIR` 到一个新目录，例如：

```bash
export SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache_exp1
```

解释：

- 换新 cache 目录比直接删除旧 cache 更稳妥。
- FlashInfer 自己也会使用 `~/.cache/flashinfer/...`；如果你改的是 FlashInfer 内部 JIT 代码或依赖，需要另外确认 FlashInfer cache 是否影响实验。

## 10. 你是否需要 eval.sh

对于你的当前目标：在 SGLang 推理引擎下优化 Nemotron-Labs-Diffusion 的 Linear SS / LinearSpec 解码效率，并研究 serving 场景下不同 request 和并发度的推理效率，不需要把 `eval.sh` 作为主线。

推荐路径是：

1. 用 `/chat` 和 `evaluate.py` 确认模型权重、HF reference 解码、基础 accuracy 没问题。你已经完成。
2. 用本文的本地 SGLang server 路径确认 LinearSpec + LoRA 可以启动和服务。当前已完成。
3. 用 SGLang OpenAI-compatible eval 跑 GSM8K、MATH-500、HumanEval 等 accuracy smoke，确认 server 输出没有明显格式或质量问题。
4. 用 `sglang.bench_serving` 做 serving benchmark matrix，记录 TTFT、TPOT、吞吐、并发扩展性。
5. 修改本地 SGLang 源码中的 LinearSpec、DLLM scheduler、FlashInfer attention、CUDA graph 逻辑。
6. 重启 server，重复 smoke + benchmark，对比结果。

`eval.sh` 的价值在于 SLURM 集群、NeMo-Skills、容器化、多 benchmark 大规模评测编排。它可以用于论文式大规模复现，但它不会替代你对本地 SGLang engine 内部解码逻辑和 kernel 的修改、profile、压测。

## 11. 常见问题

### 11.1 为什么启动 server 需要 HTTP

因为 SGLang 是 serving engine。它的核心目标就是把模型作为服务运行，并通过 HTTP / OpenAI-compatible API 接收请求、调度 batch、返回流式或非流式结果。HTTP 是外部接口，不是远端黑盒。当前 `sglang` 是本地 editable 源码，内部 scheduler、解码算法、attention backend 都可以改。

### 11.2 为什么 DLLM 必须用 flashinfer

`server_args.py` 中有明确检查：只要 `--dllm-algorithm` 不为空，attention backend 必须是 `flashinfer`。原因是 DLLM block denoising 需要 bidirectional / ENCODER_ONLY attention，当前分支只有 FlashInfer backend 支持所需路径。

### 11.3 为什么 bench_serving 不告诉我 accuracy

`bench_serving` 是压测工具，不是任务评分器。它关心每个请求多快返回、吞吐多少、首 token 多快、token 间隔多少。GSM8K / MATH-500 / HumanEval accuracy 要用第 6 节的 eval client 或自写 scorer。

### 11.4 为什么并发 client 不等于并发 server

client 的 `--max-concurrency` 只表示同时发多少请求。server 是否真的并发执行，取决于启动时的 `--max-running-requests`、显存、scheduler、CUDA graph batch size、当前 request 长度等。做 serving 并发实验时两边要一起设置。

### 11.5 A100 上是否要做 FP8

当前已验证路径是 bf16。A100 没有 Hopper / Blackwell 那类 FP8 tensor core 路径，把 DGX Spark 文档里的 `QUANT=fp8` 直接搬过来不一定有意义。你的主目标是 LinearSpec decoding efficiency 和 serving 并发，建议先在 bf16 下把 scheduler、acceptance、CUDA graph、attention backend 开销分析清楚。

## 12. 以 GSM8K 为例理解 accuracy 和 serving benchmark

本节专门解释第 6.1 节的 `GSM8K accuracy` 和第 7.3 节的 `用 GSM8K prompt 做 serving benchmark`。它们都可以使用 GSM8K 的题目文本，但测的不是同一件事。

### 12.1 两类 benchmark 的测评逻辑不同

`GSM8K accuracy` 的基本单位是“一道题是否答对”：

```text
读取 GSM8K test
  -> 构造数学题 prompt
  -> 调本地 SGLang OpenAI-compatible server 生成答案
  -> 从输出里抽取最后数字
  -> 和 gold answer 比较
  -> 汇总 score / mean / std
```

第 6.1 节命令调用的是：

```text
python -m sglang.test.run_eval --eval-name gsm8k
```

它进入：

```text
sglang_dllm/src/sglang/python/sglang/test/run_eval.py
sglang_dllm/src/sglang/python/sglang/test/simple_eval_gsm8k.py
```

其中 `simple_eval_gsm8k.py` 会：

- 从 OpenAI grade-school-math 仓库的 GSM8K test JSONL 下载 / 读取题目。
- 默认取前 5 题作为 few-shot examples。
- 后续题目逐题生成。
- 用 `get_answer_value()` 从模型输出中找最后一个数字。
- 用同样方法从 gold answer 中取数字。
- 二者相等则记为 1，否则记为 0。

`GSM8K serving benchmark` 的基本单位是“一个请求的服务耗时”：

```text
读取或构造一批请求
  -> 按指定并发和到达率发给本地 SGLang server
  -> 记录每个请求的首 token 延迟、总延迟、token 间隔
  -> 汇总 throughput / TTFT / TPOT / ITL
  -> 不判断答案是否正确
```

第 7.3 节命令调用的是：

```text
python -m sglang.bench_serving --dataset-name openai
```

它进入：

```text
sglang_dllm/src/sglang/python/sglang/bench_serving.py
sglang_dllm/src/sglang/python/sglang/benchmark/datasets/openai_dataset.py
```

`openai_dataset.py` 只要求 JSONL 每行是 OpenAI-compatible request，例如：

```json
{"messages":[{"role":"user","content":"..."}],"max_tokens":512,"temperature":0}
```

它不会读取 gold answer，也不会做正确性判断。它只把这行 request 发给 server，并把返回延迟和 token 数交给 `bench_serving.py` 汇总。

### 12.2 为什么 accuracy 和 efficiency 通常要分开测

可以在一个自写 client 里同时记录 latency 和 score，但不建议把它作为主线。原因是两类指标需要的控制条件不同。

accuracy 需要：

- 任务专用 prompt。
- 任务专用 scorer。
- 可能需要答案抽取、数学等价判断、代码执行等额外逻辑。
- 生成长度应该尊重 EOS / stop，否则模型可能输出多余内容影响解析。
- 更关心最终答对率，而不是每个 token 的精确到达时间。

efficiency / serving benchmark 需要：

- 可控的请求到达率，例如 `--request-rate inf` 或固定 Poisson 到达率。
- 可控的并发上限，例如 `--max-concurrency 1/2/4/8`。
- 可控的输入 / 输出长度，否则不同算法生成长短不一样，吞吐不可比。
- 精确记录 TTFT、TPOT、ITL、request throughput、token throughput。
- 尽量不要把 scorer、正则抽取、HumanEval 代码执行、数据下载这些 CPU 逻辑混进计时。

因此官方工具也基本是分开的：

- `sglang.test.run_eval` / `simple_eval_*`：偏 accuracy。
- `sglang.bench_serving`：偏 serving efficiency。

`run_eval.py` 也会打印总 latency 和粗粒度 output throughput，但那是整个 eval client 的 wall-clock，包含 client 线程调度和 scoring，不等价于 serving benchmark 中的 TTFT / TPOT / ITL。

### 12.3 哪些命令是官方组织形式，哪些是本地 glue code

第 6.1 节的 GSM8K accuracy 命令：

```bash
python -m sglang.test.run_eval --eval-name gsm8k ...
```

这是 SGLang 源码里自带的 eval 入口。它的组织形式是 SGLang 自带的，不是我重写的。

第 7.3 节的 serving benchmark 命令：

```bash
python -m sglang.bench_serving --dataset-name openai ...
```

这也是 SGLang 源码里自带的 serving benchmark 入口。`--dataset-name openai` 也是 SGLang 自带 dataset loader。

但第 7.3 节里“把 GSM8K 转成 OpenAI JSONL”的一行 Python：

```bash
python -c "from datasets import load_dataset; import json; ..."
```

这是我为本地实验写的数据准备 glue code。原因是 `bench_serving` 没有内置 `--dataset-name gsm8k`。它内置的是通用请求格式，例如 `random`、`custom`、`openai`、`sharegpt`。所以如果想用 GSM8K 的真实 prompt 形状压测 serving，需要先把 GSM8K 题目转换为 `openai` JSONL。

总结：

```text
GSM8K accuracy runner:  SGLang 自带
GSM8K scorer:           SGLang 自带 simple_eval_gsm8k
bench_serving:          SGLang 自带
openai dataset loader:  SGLang 自带
GSM8K -> OpenAI JSONL:  本地 glue code
```

### 12.4 batch size=1 在 SGLang serving 里是什么意思

在 `evaluate.py` 中，HF `linear_spec_generate()` 本身要求：

```python
if prompt_ids.shape[0] != 1:
    raise ValueError("Linear speculative decoding requires batch_size == 1")
```

所以 `evaluate.py` 的 Linear SS 天然是单样本逐题跑。

在 SGLang server 中，更准确的说法是“server 同时运行的 request 数”。要做严格的 bs=1 / 单请求 serving baseline，需要同时限制 server 端和 client 端：

```text
server 端: --max-running-requests 1 --cuda-graph-bs 1
client accuracy: --num-threads 1
client efficiency: --max-concurrency 1
```

解释：

- `--max-running-requests 1`：SGLang scheduler 同时只让 1 个 request 进入运行态。
- `--cuda-graph-bs 1`：只捕获 batch size 1 的 CUDA graph，避免把 2/3/4 的捕获和实验混在一起。
- `--num-threads 1`：accuracy eval client 串行发请求，避免多个 client 请求在 server 侧排队。
- `--max-concurrency 1`：serving benchmark client 同时只保留 1 个 in-flight 请求。
- `--request-rate inf` 在 `--max-concurrency 1` 下表示“前一个完成后立刻发下一个”，适合测 bs=1 饱和串行吞吐。

### 12.5 bs=1 LinearSpec + LoRA server 启动命令

如果你要先做 batch size=1 的全量基线，建议用更严格的 server 启动命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_host.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000
```

这个命令和第 4.1 节的区别只有一个关键点：

```text
第 4.1 节: --cuda-graph-bs 1 2 3 4
本节:     --cuda-graph-bs 1
```

如果当前目标是单请求 baseline，先只捕获 bs=1 更清楚。以后要测并发 4，再用 `--max-running-requests 4 --cuda-graph-bs 1 2 3 4` 重启。

如果这轮 serving benchmark 还要同时统计 LinearSpec decode-stage TPF，把 `--dllm-algorithm-config` 改成 TPF YAML：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.65 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_tpf_host.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000
```

这条命令和普通 bs=1 server 的唯一关键区别是：

```text
普通基线: --dllm-algorithm-config .../linearspec_lora_host.yaml
TPF 统计: --dllm-algorithm-config .../linearspec_lora_tpf_host.yaml
```

TPF YAML 会让 server 在每个 LinearSpec block 后向 `bench_results/linearspec_tpf_stats.jsonl` 追加一行统计。这个写文件动作会带来一点 I/O 开销，因此纯吞吐基线和带 TPF instrumentation 的结果最好分开记录。

### 12.6 bs=1 全量 GSM8K accuracy

命令：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH OPENAI_API_KEY=EMPTY /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.test.run_eval --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --eval-name gsm8k --num-threads 1 --max-tokens 512 --temperature 0 --num-shots 5
```

主参数解释：

- `OPENAI_API_KEY=EMPTY`：OpenAI Python client 要求这个变量存在；这里不会访问 OpenAI。
- `--base-url http://127.0.0.1:30000`：指向本地 SGLang server。`run_eval.py` 内部会追加 `/v1`。
- `--model /data1/...`：请求里的 model 字段，和 server 启动模型路径保持一致。
- `--eval-name gsm8k`：选择 SGLang 自带 GSM8K scorer。
- `--num-threads 1`：严格串行请求，对齐 bs=1 baseline。
- 不写 `--num-examples`：跑完整 GSM8K test。SGLang 的 `simple_eval_gsm8k.py` 默认把前 5 题用于 few-shot，实际评分题目是 test 中剩余样本。
- `--max-tokens 512`：和 `evaluate.py` 默认 `--max-new-tokens 512` 对齐。
- `--temperature 0`：greedy 输出，对齐 `evaluate.py` / HF README 的默认 Linear SS。
- `--num-shots 5`：SGLang GSM8K simple eval 默认 5-shot。

注意：这个 accuracy prompt/scorer 和 `evaluate.py --tasks gsm8k` 不完全一样。`evaluate.py` 用的是 `\boxed{}` instruction 和 HF datasets 的 GSM8K；SGLang simple eval 用的是 OpenAI simple-evals 风格 few-shot `Question: ... Answer:` prompt。它能测本地 SGLang server 的 GSM8K 正确性，但不能和 `evaluate.py` 数字无条件逐点对齐。

### 12.7 bs=1 全量 GSM8K serving efficiency

第一步，准备完整 GSM8K OpenAI JSONL：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl'; ds=load_dataset('gsm8k','main',split='test'); f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':'Solve the following math problem. Put the final numerical answer inside \\\\boxed{} at the very end.\\n\\n'+r['question']}], 'max_tokens':512, 'temperature':0})+'\\n') for r in ds]; f.close(); print(out, len(ds))"
```

主参数解释：

- `load_dataset('gsm8k','main',split='test')`：读取 GSM8K test 全量。
- `messages`：构造 OpenAI-compatible chat request。
- `max_tokens: 512`：每个请求的目标输出上限，和 `evaluate.py` 默认一致。
- `temperature: 0`：greedy。
- 输出文件 `gsm8k_openai_full.jsonl`：供 `bench_serving --dataset-name openai` 使用。

第二步，跑 serving benchmark：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl --num-prompts 1319 --request-rate inf --max-concurrency 1 --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_tpf_stats.jsonl --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_gsm8k_full_bs1_efficiency.jsonl --disable-ignore-eos --disable-tqdm
```

主参数解释：

- `--backend sglang-oai-chat`：通过 SGLang 的 OpenAI-compatible chat API 发请求。
- `--base-url http://127.0.0.1:30000`：`bench_serving` 这里不带 `/v1`，它会自己拼 endpoint。
- `--dataset-name openai`：读取 OpenAI-compatible JSONL。
- `--dataset-path ...gsm8k_openai_full.jsonl`：使用上一步生成的 GSM8K request 文件。
- `--num-prompts 1319`：GSM8K test 全量 1319 条。
- `--request-rate inf`：尽快发送；在 `--max-concurrency 1` 下实际效果是串行饱和。
- `--max-concurrency 1`：同一时刻只有一个 in-flight 请求。
- `--decode-stats-file ...linearspec_tpf_stats.jsonl`：读取 server 端 LinearSpec YAML 写出的 stats 文件，只汇总 warmup 之后、本轮 benchmark 新追加的 block 统计，并在终端和 `--output-file` JSONL 中输出 `decode_stats`。
- `--output-file ...jsonl`：保存本次 serving 指标，文件名里写清楚算法、任务、full、bs1。
- `--disable-tqdm`：减少终端进度条干扰。

如果 server 不是用第 12.5 节的 `linearspec_lora_tpf_host.yaml` 启动，`--decode-stats-file` 找不到新增 stats 行，就不会输出 TPF。正确搭配是：

```text
server: --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_tpf_host.yaml
client: --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_tpf_stats.jsonl
```

运行结束后，终端会额外出现：

```text
Decode stat records
Decode tokens
Decode forward passes
Decode TPF
Weighted accept rate
```

输出 JSONL 里会多一个字段：

```text
decode_stats.decode_tpf
```

重要：`bench_serving` 默认会给请求加 `ignore_eos=True`，除非你传 `--disable-ignore-eos`。这有两个后果：

- 优点：每个请求更接近固定输出长度，适合控制变量测 token throughput。
- 缺点：它不等价于真实 accuracy 生成，因为真实生成通常遇到 EOS 就停。

如果你想测“更接近真实服务”的 GSM8K 延迟，可以加：

```text
--disable-ignore-eos
```

但这样不同请求输出长度会差很多，吞吐和 TPOT 的可比性要另外解释。

TPOT 和 ITL 的含义不同，LinearSpec 下两者可能差很多：

```text
TPOT = (整条请求 latency - TTFT) / (output_tokens - 1)
ITL  = 相邻 streaming chunk 到达客户端的时间间隔
```

在普通 AR 解码里，一个 streaming chunk 往往接近一个 token，所以 `TPOT` 和 `ITL` 通常比较接近。LinearSpec 不同：一次 decode block 会做 draft forward + verify forward，然后一次性接受多个 token；客户端看到的是“过了一段时间后收到一坨 token”。因此：

```text
Mean ITL 接近一个 LinearSpec block 的 wall-clock 间隔
Mean TPOT 是这个 block 间隔被本轮接受的多个 token 平摊后的结果
```

例如 `Mean TPOT=2.21ms`、`Mean ITL=32.25ms` 并不矛盾。粗略看，`32.25 / 2.21 ~= 14.6`，说明每个 streaming 间隔平均被十几个输出 token 平摊；这正是 Linear SS / LinearSpec 想利用的并行接受效果。真正判断 LinearSpec 是否高效时，应同时看：

```text
Output token throughput
Mean TPOT
Mean ITL
decode_stats.decode_tpf
decode_stats.weighted_acceptance_rate
```

### 12.8 如果要 bs=1 跑所有本文 benchmark

先保持 server 是第 12.5 节的 bs=1 启动方式。

accuracy 全量：

```text
GSM8K:    第 12.6 节命令，不传 --num-examples，--num-threads 1
MATH-500: 第 6.2 节全量命令，把 MathEval(..., None, 16) 改为 MathEval(..., None, 1)
HumanEval:第 6.3 节全量命令，把 --num-threads 16 改为 --num-threads 1
```

efficiency 全量：

```text
GSM8K:    第 12.7 节命令，--num-prompts 1319 --max-concurrency 1
MATH-500: 第 7.4 节命令，生成 JSONL 时不要 select(range(128))，--num-prompts 500 --max-concurrency 1
HumanEval:第 7.5 节命令，生成 JSONL 时不要 [:128] 或保留全量 164，--num-prompts 164 --max-concurrency 1
```

如果还要比较 `LinearSpec+LoRA / LinearSpec-base / FastDiffuser`，不要在同一个 server 进程里切模式；每个模式都重启一次 server，保持相同：

```text
CUDA_VISIBLE_DEVICES
dtype
context-length
max-running-requests
cuda-graph-bs
benchmark dataset
num-prompts
max-concurrency
```

只改变算法相关参数，这样结果才可比。

### 12.9 当前 SGLang Linear SS 超参和 evaluate.py 是否一致

`evaluate.py` 的 Linear SS 默认配置是：

```text
mode=linear_spec
block_length=32
threshold=0.0
max_new_tokens=512
max_thinking_tokens=6000
temperature=0.0  # HF linear_spec_generate 函数默认值，evaluate.py 没单独暴露
lora=off         # 只有传 --lora 或 --lora-path 才开启
```

当前 SGLang LinearSpec + LoRA server 配置是：

```text
dllm_algorithm=LinearSpec
block_size=32              # DllmConfig 对 NemotronLabsDiffusionModel 的默认值
causal_context=true
lora_path=sglang_dllm/linear_spec_lora
lora_mode=draft_only
request max_tokens=512     # GSM8K accuracy / JSONL 中显式设置
temperature=0              # eval / JSONL 中显式设置
```

对应关系：

- `block_length=32` 和 `block_size=32` 对齐。
- `threshold=0.0` 和当前 SGLang `LinearSpec` 对齐。HF 中 `threshold=0.0` 表示 draft 一次性填满 mask block；当前 SGLang `linear_spec.py` 也是每个 block 一次 bidirectional draft + 一次 causal verify，没有实现 `threshold>0` 的多轮 draft。
- `max_new_tokens=512` 需要通过 request 侧 `--max-tokens 512` 或 JSONL 里的 `max_tokens: 512` 对齐。
- `temperature=0` 已在 eval / JSONL 中显式设置。
- `lora_mode=draft_only` 和 HF `linear_spec_generate` 的 LoRA 语义一致：draft phase LoRA ON，prefill / verify phase LoRA OFF。

主要不一致点：

- `evaluate.py --mode linear_spec` 默认不带 LoRA；当前第 4.1 节 SGLang server 默认带 LoRA。因此它更接近 `evaluate.py --mode linear_spec --lora`，不是 plain `evaluate.py --mode linear_spec`。
- `evaluate.py` 的 GSM8K prompt/scorer 和 SGLang `run_eval --eval-name gsm8k` 不完全一致。
- `bench_serving` 默认 `ignore_eos=True`，这和 accuracy / evaluate.py 的自然 EOS 停止不同。
- SGLang 是 serving scheduler 路径；即使 bs=1，仍包含 HTTP、scheduler、KV pool、CUDA graph、FlashInfer backend 等 serving 开销，不能把它当成纯 HF forward loop。

所以，如果目标是“代码逻辑和超参尽量对齐 evaluate.py 的 LinearSpec+LoRA”，你应使用：

```text
server:  --dllm-algorithm LinearSpec + linearspec_lora_host.yaml
server:  --max-running-requests 1 --cuda-graph-bs 1
request: max_tokens=512 temperature=0
client:  accuracy 用 --num-threads 1，efficiency 用 --max-concurrency 1
```

如果目标是“和 evaluate.py plain LinearSpec 不带 LoRA 对齐”，则需要另写一个不含 `lora_path` 的 YAML，并用它启动：

```bash
printf 'algorithm: LinearSpec\ncausal_context: true\n' > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_host.yaml
```

然后把 server 启动命令中的：

```text
--dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_host.yaml
```

改成：

```text
--dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_host.yaml
```

## 13. bs=1 + TPF 全 benchmark efficiency 命令矩阵

本节只保留当前需要比较的三种 DLLM / Linear SS 解码路径：

```text
benchmark: GSM8K, MATH-500, HumanEval
decode mode: confidence-based DLLM, Linear SS without LoRA, Linear SS with LoRA
server batch: --max-running-requests 1
client concurrency: --max-concurrency 1
cuda graph: --cuda-graph-bs 1
TPF stats: server YAML stats_file + client --decode-stats-file
```

这里的 `batch size=1` 对 serving benchmark 来说分成两层：

- server 端：`--max-running-requests 1`，表示 SGLang scheduler 同时最多运行 1 个 request。
- client 端：`--max-concurrency 1`，表示 `bench_serving` 同时最多发出 1 个未完成请求。

本节所有 server 命令都是前台运行。每次只能让一个 server 占用同一个端口 `30000`。切换解码模式、切换 benchmark、或切换 stats 文件时，先在 server 终端按 `Ctrl-C` 停掉旧 server，再启动下一条 server 命令。下面命令统一使用 `CUDA_VISIBLE_DEVICES=2`；如果 GPU 2 正忙，把命令里的 `CUDA_VISIBLE_DEVICES=2` 改成空闲 GPU 编号即可。

### 13.1 TPF 统计口径

当前本地 `bench_serving.py` 支持本地参数：

```text
--decode-stats-file
```

它不是 server 参数，而是 benchmark client 参数。server 端真正写 stats 的位置是 `--dllm-algorithm-config` 指向的 YAML 里的：

```yaml
stats_file: /path/to/stats.jsonl
```

因此每一组 benchmark 都要保证两边路径一致：

```text
server YAML:   stats_file: /.../xxx_stats.jsonl
client:        --decode-stats-file /.../xxx_stats.jsonl
```

`bench_serving` 会在 benchmark 开始前记录 stats 文件的当前字节 offset，结束后只读取本轮新追加的 JSONL 行，并汇总：

```text
decode_TPF = sum(tokens) / sum(forward_passes)
```

三种模式的分母含义不同：

- confidence-based DLLM / FastDiffuser：`forward_passes` 是每个 request 在当前 block 内实际参与的 denoising forward steps。当前 stats 口径不把最后一次 KV-update forward 算进分母，因此更接近 HF NFE / diffusion-step TPF。
- Linear SS without LoRA：`forward_passes=2`，对应一次 bidirectional draft forward 加一次 causal verify forward。
- Linear SS with LoRA：同样 `forward_passes=2`，区别是 draft pass 使用 LoRA，verify pass 使用 base 权重，也就是 `lora_mode=draft_only`。

旧版文档里曾使用 `--linearspec-stats-file`。这个旧参数现在仍保留为兼容别名，但后续统一使用 `--decode-stats-file`，因为 FastDiffuser / confidence-based DLLM 也使用同一个汇总逻辑。

### 13.2 先准备三个 OpenAI JSONL 数据文件

GSM8K 全量 test split，一共 1319 条：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl'; ds=load_dataset('gsm8k','main',split='test'); f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':'Solve the following math problem. Put the final numerical answer inside \\\\boxed{} at the very end.\\n\\n'+r['question']}], 'max_tokens':512, 'temperature':0})+'\\n') for r in ds]; f.close(); print(out, len(ds))"
```

MATH-500 全量 test split，一共 500 条：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from datasets import load_dataset; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_full.jsonl'; ds=load_dataset('HuggingFaceH4/MATH-500', split='test'); f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':'Solve the following math problem. Put the final answer inside \\\\boxed{} at the very end.\\n\\n'+r['problem']}], 'max_tokens':1024, 'temperature':0})+'\\n') for r in ds]; f.close(); print(out, len(ds))"
```

HumanEval 全量 164 条：

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -c "from human_eval.data import read_problems; import json; out='/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_full.jsonl'; problems=list(read_problems().values()); instruction='Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\\n'; f=open(out,'w'); [f.write(json.dumps({'messages':[{'role':'user','content':instruction+p['prompt']}], 'max_tokens':512, 'temperature':0.2})+'\\n') for p in problems]; f.close(); print(out, len(problems))"
```

这三个 JSONL 只用于 efficiency benchmark。它们不会执行答案评分，也不会判断生成代码是否正确；`bench_serving` 只用它们提供真实任务 prompt 形状、`max_tokens` 和 temperature。本节的 efficiency 命令都显式加了 `--disable-ignore-eos`，所以 JSONL 里的 `max_tokens` 是生成上限，不是强制输出长度。

### 13.3 GSM8K efficiency：1319 requests, bs=1

#### 13.3.1 confidence-based DLLM server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_gsm8k_full_bs1_stats.jsonl && printf "algorithm: FastDiffuser\ncausal_context: true\ntemperature: 0.0\nthreshold: 0.9\nmax_steps: 32\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_gsm8k_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_gsm8k_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.65 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm FastDiffuser --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_gsm8k_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.3.2 confidence-based DLLM efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl --num-prompts 1319 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_gsm8k_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_gsm8k_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.3.3 Linear SS without LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_gsm8k_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_gsm8k_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_gsm8k_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.65 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_gsm8k_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.3.4 Linear SS without LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl --num-prompts 1319 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_gsm8k_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_gsm8k_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.3.5 Linear SS with LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_gsm8k_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_gsm8k_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_gsm8k_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.6 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_gsm8k_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.3.6 Linear SS with LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/gsm8k_openai_full.jsonl --num-prompts 1319 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_gsm8k_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_gsm8k_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

### 13.4 MATH-500 efficiency：500 requests, bs=1

#### 13.4.1 confidence-based DLLM server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_math500_full_bs1_stats.jsonl && printf "algorithm: FastDiffuser\ncausal_context: true\ntemperature: 0.0\nthreshold: 0.9\nmax_steps: 32\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_math500_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_math500_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm FastDiffuser --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_math500_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.4.2 confidence-based DLLM efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_full.jsonl --num-prompts 500 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_math500_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_math500_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.4.3 Linear SS without LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_math500_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_math500_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_math500_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_math500_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.4.4 Linear SS without LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_full.jsonl --num-prompts 500 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_math500_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_math500_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.4.5 Linear SS with LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_math500_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_math500_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_math500_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_math500_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.4.6 Linear SS with LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/math500_openai_full.jsonl --num-prompts 500 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_math500_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_math500_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

### 13.5 HumanEval efficiency：164 requests, bs=1

#### 13.5.1 confidence-based DLLM server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_humaneval_full_bs1_stats.jsonl && printf "algorithm: FastDiffuser\ncausal_context: true\ntemperature: 0.2\nthreshold: 0.9\nmax_steps: 32\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_humaneval_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_humaneval_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm FastDiffuser --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/fastdiffuser_humaneval_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.5.2 confidence-based DLLM efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_full.jsonl --num-prompts 164 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_humaneval_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/fastdiffuser_humaneval_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.5.3 Linear SS without LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_humaneval_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_humaneval_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_humaneval_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_base_humaneval_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.5.4 Linear SS without LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_full.jsonl --num-prompts 164 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_humaneval_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_base_humaneval_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

#### 13.5.5 Linear SS with LoRA server

```bash
bash -lc ': > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_humaneval_full_bs1_stats.jsonl && printf "algorithm: LinearSpec\ncausal_context: true\nlora_path: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linear_spec_lora\nlora_mode: draft_only\nstats_file: /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_humaneval_full_bs1_stats.jsonl\n" > /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_humaneval_bs1.yaml && env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/hf_cache SGLANG_CACHE_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/sglang_cache /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.launch_server --model-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 --mem-fraction-static 0.55 --max-running-requests 1 --attention-backend flashinfer --sampling-backend flashinfer --dllm-algorithm LinearSpec --dllm-algorithm-config /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/linearspec_lora_humaneval_bs1.yaml --cuda-graph-bs 1 --context-length 2048 --host 0.0.0.0 --port 30000'
```

#### 13.5.6 Linear SS with LoRA efficiency

```bash
env PATH=/data/home/wly/.conda/envs/nld_sglang/bin:$PATH /data/home/wly/.conda/envs/nld_sglang/bin/python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --dataset-name openai --dataset-path /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/humaneval_openai_full.jsonl --num-prompts 164 --request-rate inf --max-concurrency 1 --output-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_humaneval_full_bs1_efficiency.jsonl --decode-stats-file /data/home/wly/dLLM/Nemotron-Labs-Diffusion/sglang_dllm/bench_results/linearspec_lora_humaneval_full_bs1_stats.jsonl --disable-ignore-eos --disable-tqdm
```

### 13.6 如何对比结果

每条 `bench_serving` 命令结束后，终端会打印常规 serving 指标：

```text
Request throughput
Output token throughput
Mean TTFT
Mean TPOT
Mean ITL
```

如果 `--decode-stats-file` 路径正确，并且 server YAML 里也配置了同一个 `stats_file`，终端还会打印额外的 decode stats：

```text
Decode stat records
Decode tokens
Decode forward passes
Decode TPF
Weighted accept rate
```

最终看结果时，每种模式都有两类文件：

```text
*_efficiency.jsonl  # bench_serving 的请求级 latency / throughput 汇总，包含 decode_stats 汇总字段
*_stats.jsonl       # server algorithm 每个 decode block 追加的 tokens / forward_passes 记录
```

比较三种模式时，先固定以下变量：

```text
GPU:                  同一张 A100，例如都用 CUDA_VISIBLE_DEVICES=2
dtype:                bfloat16
context length:       2048
server concurrency:   --max-running-requests 1
client concurrency:   --max-concurrency 1
cuda graph bs:        --cuda-graph-bs 1
benchmark JSONL:      同一个 *_openai_full.jsonl
```

然后只改变：

```text
confidence-based DLLM: FastDiffuser config
Linear SS base:        LinearSpec config without lora_path
Linear SS LoRA:        LinearSpec config with lora_path + lora_mode=draft_only
```
