# Nemotron-Labs-Diffusion A100 环境安装与三类入口脚本运行手册

本文档说明如何在当前 A100 服务器上安装运行环境，并在指定 GPU 上依次运行：

1. `chat/` 文件夹下的 smoke test。
2. `evaluate.py` 单进程评测。
3. `eval.sh` SLURM/容器评测。

文档基于以下本地路径：

- 项目目录：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion`
- 模型目录：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`
- LoRA 目录：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora`

## 1. 当前服务器状态

我在当前 shell 中检查到：

- GPU：4 张 `NVIDIA A100 80GB PCIe`，编号 `0,1,2,3`，当前基本空闲。
- Driver：`590.44.01`。
- `nvidia-smi` 显示 CUDA runtime capability：`13.1`。
- 默认 Python：`/opt/anaconda3/bin/python`，版本 `Python 3.13.5`。
- 默认 base 环境没有安装 `torch`。
- 已有 conda 环境：
  - `dinfer`：Python 3.10.19，torch 2.8.0+cu128，transformers 4.57.0。
  - `dinfer_new`：Python 3.10.20，torch 2.8.0+cu128，transformers 4.57.0。
  - `llada` / `sdar`：有 torch，但 transformers 与 huggingface-hub 依赖冲突。
- 当前 shell 中 `srun`、`sbatch` 不在 `PATH`。
- 当前 shell 中 `ACCOUNT`、`CONTAINER_IMAGE` 未设置。

结论：

- 跑 `chat/` 和 `evaluate.py`，建议新建独立 conda 环境，不要用 base。
- 跑 `eval.sh --dry-run` 可以在当前 shell 尝试。
- 真实跑 `eval.sh` 需要进入有 SLURM 命令的环境，并准备 NeMo-Skills-ready `.sqsh` 容器镜像和 SLURM account。

## 2. GPU 指定规则

对 `chat/` 和 `evaluate.py`：

- 用 `CUDA_VISIBLE_DEVICES=<物理GPU编号>` 指定物理 GPU。
- 例如 `CUDA_VISIBLE_DEVICES=2` 时，脚本里的 `cuda` / `cuda:0` 实际对应物理 GPU 2。
- 单 GPU smoke test 推荐先用空闲的 `GPU 0`。

对 `eval.sh`：

- `eval.sh` 是 SLURM/容器路径，核心控制参数是 `--gpus N`。
- 在 SLURM 作业内部，pipeline 会给每个 worker 设置 `CUDA_VISIBLE_DEVICES=$i`。
- 如果你已经在一个手动分配好的交互式 GPU 环境里运行，外层 `CUDA_VISIBLE_DEVICES=2` 可能有用；但在标准 SLURM 提交中，应优先相信调度器分配，而不是手动绑物理卡。

## 3. 推荐新建 Python 环境

当前模型 `config.json` 标记了 `transformers_version = 5.0.0`，`README.md` 也写了 `transformers>=5.0.0`。建议创建独立环境 `nld`。

以下命令均为单行命令。

创建环境：

```bash
conda create -n nld python=3.11 -y
```

升级 pip：

```bash
conda run --no-capture-output -n nld python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch CUDA 12.8 wheel。当前 driver 590 足够运行 cu128 wheel；PyTorch wheel 自带 CUDA runtime，不要求系统 CUDA toolkit 版本等于 12.8：

```bash
conda run --no-capture-output -n nld python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装项目直接入口需要的 Python 包：

```bash
conda run --no-capture-output -n nld python -m pip install "transformers>=5.0.0" datasets peft accelerate safetensors sentencepiece protobuf numpy tqdm
```

如果你还想直接启动 `xp/dlm_api/dlm_batch_server.py` 做本地 HTTP 服务，额外安装服务依赖：

```bash
conda run --no-capture-output -n nld python -m pip install fastapi uvicorn pydantic httpx
```

如果你的 pip 源暂时找不到 `transformers>=5.0.0`，可临时安装 Hugging Face Transformers 源码版：

```bash
conda run --no-capture-output -n nld python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git"
```

## 4. 环境验证命令

验证 PyTorch 能看到指定 GPU。下面示例绑定物理 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python -c "import torch, transformers, datasets, peft; print('torch=', torch.__version__, 'torch_cuda=', torch.version.cuda, 'cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0)); print('transformers=', transformers.__version__)"
```

验证模型远程 config 能通过 `trust_remote_code=True` 正常加载：

```bash
conda run --no-capture-output -n nld python -c "from transformers import AutoConfig; p='/data1/linyewei/models/Nemotron-Labs-Diffusion-8B'; c=AutoConfig.from_pretrained(p, trust_remote_code=True); print(type(c).__name__, c.model_type, c.dlm_paradigm, c.mask_token_id, c.eos_token_id)"
```

可选：验证完整模型能加载到指定 GPU。该命令会占用 GPU 显存，A100 80GB 足够：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python -c "import torch; from transformers import AutoModel; p='/data1/linyewei/models/Nemotron-Labs-Diffusion-8B'; m=AutoModel.from_pretrained(p, trust_remote_code=True).cuda().to(torch.bfloat16).eval(); print(type(m).__name__, 'device=', next(m.parameters()).device)"
```

## 5. 运行 `chat/` smoke tests

进入项目目录：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

下面每条命令都是单行命令，都会把一个问题通过 stdin 传给脚本。默认示例使用物理 GPU 0；要换 GPU，把 `CUDA_VISIBLE_DEVICES=0` 改成 `CUDA_VISIBLE_DEVICES=1/2/3`。

### 5.1 AR smoke test

调用链：`chat/chat_ar.py -> model.ar_generate()`。

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_ar.py
```

### 5.2 dLM / diffusion smoke test

调用链：`chat/chat_dlm.py -> model.generate(..., block_length=32, threshold=0.9)`。

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_dlm.py
```

### 5.3 Linear self-speculation smoke test

调用链：`chat/chat_linear_spec.py -> model.linear_spec_generate(..., block_length=32)`。

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_linear_spec.py
```

### 5.4 Linear self-speculation + LoRA smoke test

调用链：`chat/chat_linear_spec_lora.py -> PeftModel.from_pretrained(..., subfolder="linear_spec_lora") -> model.linear_spec_generate()`。

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_linear_spec_lora.py
```

### 5.5 统一 chat launcher

如果想用 `chat/chat.py` 统一入口，也可以用下面命令做单轮输入后退出。

AR：

```bash
printf 'What is 15%% of 240?\n:q\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat.py --mode ar --max-new-tokens 128
```

dLM：

```bash
printf 'What is 15%% of 240?\n:q\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat.py --mode dlm --max-new-tokens 128 --block-length 32 --threshold 0.9
```

Linear SS：

```bash
printf 'What is 15%% of 240?\n:q\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat.py --mode linear_spec --max-new-tokens 128 --block-length 32
```

Linear SS + LoRA：

```bash
printf 'What is 15%% of 240?\n:q\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat.py --mode linear_spec_lora --max-new-tokens 128 --block-length 32
```

## 6. 运行 `evaluate.py`

`evaluate.py` 是单进程、单 GPU、无需 server 的轻量评测路径。

注意：

- 首次运行 `gsm8k` 或 `math-500` 会通过 `datasets.load_dataset()` 下载数据集；如果服务器不能联网，需要提前缓存 Hugging Face datasets。
- `--limit` 用于 smoke test，建议先设成 2 到 10。
- `--max-new-tokens 128` 用于快速验证；正式评测可改回默认 512 或更大。
- 输出中的 `TPF = total_new_tokens / total_nfe`。

### 6.1 AR evaluate smoke test

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode ar --tasks gsm8k --limit 5 --max-new-tokens 128 --print-every 1
```

### 6.2 dLM evaluate smoke test

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode dlm --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 8 --threshold 0.9 --print-every 1
```

### 6.3 Linear SS evaluate smoke test

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode linear_spec --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 32 --print-every 1
```

### 6.4 Linear SS + LoRA evaluate smoke test

`evaluate.py` 默认 LoRA path 是项目下 `miscs/linear_spec_lora`，但当前本地模型目录已经有 LoRA，所以这里显式传 `--lora-path`：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode linear_spec --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 32 --lora --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --print-every 1
```

### 6.5 多任务 smoke test

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode dlm --tasks gsm8k,math-500 --limit 3 --max-new-tokens 128 --block-length 8 --threshold 0.9 --print-every 1
```

### 6.6 输出 JSON

如果要保存结果：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode dlm --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 8 --threshold 0.9 --output configs/evaluate_dlm_gsm8k_limit5.json --print-every 1
```

## 7. 运行 `eval.sh`

`eval.sh` 是完整 benchmark sweep 的 SLURM + 容器路径，不等价于 `evaluate.py`。它会：

1. 解析 `--mode`。
2. 导出 `SERVER_*` 和 `SEQ_EVAL_*` 环境变量。
3. 调用 `xp/examples/run_dlm_eval_pipeline_gpu_only.sh`。
4. 在容器中启动每 GPU 一个 `dlm_batch_server.py` worker。
5. 启动 `dlm_load_balancer.py`。
6. 用 `xp/nemo-skills/eval_dlm.py` 访问 OpenAI-compatible endpoint。

当前 shell 中 `srun/sbatch` 不在 `PATH`，并且 `ACCOUNT`、`CONTAINER_IMAGE` 没有设置。因此当前可以先跑 dry-run；真实运行需要去有 SLURM 环境的 shell，并准备容器镜像。

### 7.1 eval.sh dry-run

dry-run 不需要容器镜像，不会提交作业，适合检查参数解析。

AR：

```bash
bash eval.sh --mode ar --benchmarks gsm8k:1 --gpus 1 --dry-run
```

dLM：

```bash
bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --dry-run
```

Linear SS：

```bash
bash eval.sh --mode linear_spec --benchmarks gsm8k:1 --gpus 1 --dry-run
```

Linear SS + LoRA：

```bash
bash eval.sh --mode linear_spec --benchmarks gsm8k:1 --gpus 1 --lora --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --dry-run
```

### 7.2 eval.sh 真实运行前置条件

真实运行需要：

- `srun` / `sbatch` 在 `PATH`。
- 已有可用 SLURM account。
- 已有 NeMo-Skills-ready `.sqsh` 容器镜像。
- 容器内要有 PyTorch、Transformers、NeMo-Skills、FastAPI、uvicorn 等运行依赖。
- 如果使用 `--lora`，容器内还需要 `peft`。

如果不知道容器是否带 PEFT，可以先用不带 LoRA 的 `linear_spec` 跑通，再跑 `--lora`。

### 7.3 eval.sh 真实 smoke 命令

把下面命令中的 `<your_slurm_account>` 和 `<path_to_container.sqsh>` 换成真实值。

AR：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode ar --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

dLM：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

Linear SS：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode linear_spec --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

Linear SS + LoRA：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container_with_peft.sqsh> bash eval.sh --mode linear_spec --benchmarks gsm8k:1 --gpus 1 --lora --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --time 01:00:00
```

### 7.4 eval.sh 输出位置

默认输出目录：

```text
/data/home/wly/dLLM/Nemotron-Labs-Diffusion/eval_suit_results
```

指定输出目录的单行命令：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> OUT_DIR=/data/home/wly/dLLM/Nemotron-Labs-Diffusion/eval_suit_results_smoke bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

## 8. 推荐执行顺序

建议按这个顺序排查，避免一开始就进入复杂 SLURM 路径：

1. 检查 GPU：

```bash
nvidia-smi
```

2. 创建并安装 `nld` 环境：

```bash
conda create -n nld python=3.11 -y
```

3. 安装 PyTorch：

```bash
conda run --no-capture-output -n nld python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4. 安装 Transformers 等依赖：

```bash
conda run --no-capture-output -n nld python -m pip install "transformers>=5.0.0" datasets peft accelerate safetensors sentencepiece protobuf numpy tqdm fastapi uvicorn pydantic httpx
```

5. 验证 CUDA：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

6. 跑 AR chat：

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_ar.py
```

7. 跑 dLM chat：

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_dlm.py
```

8. 跑 Linear SS chat：

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_linear_spec.py
```

9. 跑 `evaluate.py` smoke：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode dlm --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 8 --threshold 0.9 --print-every 1
```

10. 跑 `eval.sh` dry-run：

```bash
bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --dry-run
```

11. 准备 SLURM account 和容器后再跑真实 `eval.sh`：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --time 01:00:00
```

## 9. 常见问题

### 9.1 `ModuleNotFoundError: No module named 'torch'`

你用了 base Python。请用 `conda run -n nld ...` 或先激活 `nld` 环境。

验证命令：

```bash
conda run --no-capture-output -n nld python -c "import torch; print(torch.__version__)"
```

### 9.2 `torch.cuda.is_available()` 是 `False`

优先检查：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

如果仍是 `False`，通常是安装了 CPU 版 PyTorch，重新安装 cu128 wheel：

```bash
conda run --no-capture-output -n nld python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 9.3 `ImportError: huggingface-hub...`

不要复用 `llada` 或 `sdar` 这类已有环境；它们当前有 transformers/huggingface-hub 版本冲突。使用新环境：

```bash
conda create -n nld python=3.11 -y
```

### 9.4 `transformers>=5.0.0` 安装不到

先尝试升级 pip：

```bash
conda run --no-capture-output -n nld python -m pip install --upgrade pip setuptools wheel
```

如果仍不行，临时用源码版 Transformers：

```bash
conda run --no-capture-output -n nld python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git"
```

### 9.5 `chat_linear_spec_lora.py` 找不到 LoRA

本地模型目录已经包含 LoRA：

```text
/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora
```

`chat_linear_spec_lora.py` 默认从模型目录的 `linear_spec_lora` subfolder 读取，正常不需要额外参数。

`evaluate.py` 的默认 LoRA path 是项目下 `miscs/linear_spec_lora`，所以建议显式传：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode linear_spec --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 32 --lora --lora-path /data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora --print-every 1
```

### 9.6 `eval.sh` 报 no container image 或 account required

真实运行必须指定：

```bash
ACCOUNT=<your_slurm_account> CONTAINER_IMAGE=<path_to_container.sqsh> bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1
```

如果只是检查参数，用：

```bash
bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --dry-run
```

### 9.7 `srun` 或 `sbatch` 找不到

当前 shell 没有 SLURM 命令。请进入集群登录节点/计算节点的正确环境，或加载 SLURM module。没有 SLURM 时不要跑真实 `eval.sh`，先用 `evaluate.py`。

## 10. 最小可执行命令清单

如果只想最快跑通一遍，按顺序执行这几条单行命令：

```bash
conda create -n nld python=3.11 -y
```

```bash
conda run --no-capture-output -n nld python -m pip install --upgrade pip setuptools wheel
```

```bash
conda run --no-capture-output -n nld python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
conda run --no-capture-output -n nld python -m pip install "transformers>=5.0.0" datasets peft accelerate safetensors sentencepiece protobuf numpy tqdm fastapi uvicorn pydantic httpx
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_ar.py
```

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_dlm.py
```

```bash
printf 'What is 15%% of 240?\n' | CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python chat/chat_linear_spec.py
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nld python evaluate.py --mode dlm --tasks gsm8k --limit 5 --max-new-tokens 128 --block-length 8 --threshold 0.9 --print-every 1
```

```bash
bash eval.sh --mode dlm --benchmarks gsm8k:1 --gpus 1 --dry-run
```

## 11. 参考

- PyTorch 官方安装选择器：https://pytorch.org/get-started/locally/
- Hugging Face Transformers 安装文档：https://huggingface.co/docs/transformers/installation
- 本项目 README：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/README.md`
