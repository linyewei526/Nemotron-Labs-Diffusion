# NLD Chat Benchmarks 评测接入与框架变更记录

> 本地入口：`observations/eval_sglang.sh`、`observations/eval_pytorch_nemo.sh`
>
> 历史 smoke 结果：`/data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/`
>
> 历史结果说明：`chat_benchmark_smoke/mt_bench_sglang_gpu2/eval_20260811_155004/` 是 artifact 收口修复前的问题复现，metrics 中三个路径指向按整理要求已删除的内部工作目录；可审计的修复验证结果是 `chat_benchmark_smoke/mt_bench_sglang_artifact_fix/eval_20260811_155548/`。两轮结果没有互相回填文件。
>
> 官方 SLURM 入口 `eval.sh` 仍保留在项目根目录，不随 observation 接口迁移。

本文从 2026-08-10 起持续记录 NLD 正式评测框架对 Arena-Hard、MT-Bench 和 AlpacaEval 的接入。目标是在不改变原有十项默认 benchmark、也不改变模型解码实现的前提下，为每个 benchmark 固定其正式协议；后端覆盖按实际需求推进，不强求三条推理链路同时接入。

## 1. 当前结论

Arena-Hard 和 MT-Bench 已接入三条既有评测路径；AlpacaEval 2.0 本轮按“先保证一条后端真实跑通”的原则接入 SGLang：

| 入口 | 候选模型后端 | Arena-Hard | MT-Bench | AlpacaEval 2.0 |
|---|---|---|---|---|
| `observations/eval_sglang.sh` | SGLang AR / FastDiffuser / LinearSpec | 支持 | 支持 | 支持；本轮验证后端 |
| `observations/eval_pytorch_nemo.sh` | 原生 PyTorch AR / dLM / LinearSpec | 支持 | 支持 | 支持；AR 已真实 smoke |
| `eval.sh` | SLURM + 容器 + NLD worker | 支持；容器内需包含 NeMo-Skills adapter | 支持；使用独立 runner | 尚未接顶层入口 |
| `evaluate.py` | 单进程轻量 scorer | 不支持 | 不支持 | 不支持 |

当前默认十项列表没有加入这些 chat benchmark。它们会额外调用外部 judge，产生网络依赖和 API 费用。Arena-Hard/MT-Bench 的 NLD 默认 judge 为 GPT-4.1；AlpacaEval 保持官方 `gpt-4-1106-preview` 默认值。必须通过 `--benchmarks arena-hard:1`、`--benchmarks arena-hard-v2:1`、`--benchmarks mt-bench:1` 或 `--benchmarks alpaca-eval:1` 显式启用。

## 2. 评测协议

当前验证环境使用 `nemo-skills 0.7.0`，其中包含两个内置 adapter：

- `arena-hard`：Arena-Hard v0.1，候选回答与 GPT-4-0314 baseline 回答进行比较。
- `arena-hard-v2`：Arena-Hard v2.0；`hard_prompt` 使用 o3-mini-2025-01-31 baseline，`creative_writing` 使用 gemini-2.0-flash-001 baseline。

两个 adapter 默认都使用：

- judge model：`gpt-4.1`；
- judge server type：`openai`；
- judge endpoint：`https://api.openai.com/v1`；
- metric type：`arena`。

每个候选答案会执行两个 judge 请求：一次把候选放在 A、baseline 放在 B，另一次交换 A/B，以降低位置偏差。最终 `metrics.json` 中保留 NeMo-Skills 的 Arena score、置信区间、无效 judgement 数和样本数；NLD 原有的候选模型计时、token、TPS/TPOT/NFE 合并逻辑继续工作。

注意：judge 模型、judge prompt、baseline 答案或 NeMo-Skills 版本任一变化，都会改变评测协议。正式横向对比时必须固定这些条件。

## 3. 调用链

SGLang 路径：

```text
observations/eval_sglang.sh
  -> run_sglang_eval_pipeline_gpu_only.sh
     -> 准备/恢复 arena-hard 数据
     -> SGLang 生成候选回答
     -> eval_dlm.py
        -> NeMo-Skills Arena judge（独立 judge endpoint）
        -> Arena metrics
     -> 合并候选模型的 SGLang timing/decode stats
```

原生 PyTorch 路径：

```text
observations/eval_pytorch_nemo.sh
  -> run_pytorch_nemo_eval_pipeline_gpu_only.sh
     -> 准备/恢复 arena-hard 数据
     -> 原生 PyTorch server 生成候选回答
     -> eval_dlm.py
        -> NeMo-Skills Arena judge
        -> Arena metrics
     -> 合并候选模型 request stats
```

judge 请求不会发送到候选 NLD server，也不会携带 NLD diffusion `extra_body`。候选生成指标与 judge 指标因此保持分离。

## 4. 本次代码改动

### 4.1 `xp/nemo-skills/eval_dlm.py`

新增：

- 识别 `arena-hard` 和 `arena-hard-v2`。
- 在生成候选答案前检查当前 NeMo-Skills 是否包含对应 dataset adapter。
- 默认 OpenAI judge 场景下检查 `OPENAI_API_KEY`。
- 新增 judge CLI 参数，并只对当前 Arena-Hard benchmark 传入 NeMo-Skills；同一命令中的 GSM8K 等普通 benchmark 不会被误设为 judge benchmark。
- 保留 NeMo-Skills adapter 自带的 judge 默认值；只有用户显式传参时才覆盖。

### 4.2 顶层入口

`observations/eval_sglang.sh`、`observations/eval_pytorch_nemo.sh` 和 `eval.sh` 新增统一参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--judge-model NAME` | 覆盖 judge 模型 | adapter 默认，即 GPT-4.1 |
| `--judge-server-address URL` | 覆盖 judge OpenAI-compatible endpoint | adapter 默认，即 OpenAI `/v1` |
| `--judge-server-type TYPE` | 覆盖 NeMo-Skills judge server type | adapter 默认，即 `openai` |
| `--skip-judge-api-key-check` | 跳过 NLD 的提前凭证检查 | 关闭 |

凭证必须通过环境变量提供，不能通过 CLI 传入。`Settings.json` 只记录是否配置了 `OPENAI_API_KEY`，不会记录 key 内容。

### 4.3 数据准备与缓存

SGLang 和原生 PyTorch pipeline 会：

1. 检查活动 NeMo-Skills 安装是否存在 `<dataset>/__init__.py` 和 `<dataset>/prepare.py`。
2. 优先恢复 `--nemo-skills-data-dir/<dataset>/` 中的持久缓存。
3. 对 Arena-Hard 要求缓存必须包含 `test.jsonl`；只有 `question.jsonl` 或 baseline 原始文件不算准备完成。
4. 缓存缺失时运行 `python -m nemo_skills.dataset.prepare <dataset>`。
5. Arena-Hard prepare 失败时立即报错，不再像普通兼容路径那样忽略 prepare 返回码。
6. 准备成功后把完整数据同步回持久缓存。

默认持久数据目录仍为 `/data1/linyewei/datasets/NLD`。

## 5. 运行前检查

激活环境并确认版本：

```bash
conda activate nld_sglang
```

```bash
python -c "import nemo_skills; print(nemo_skills.__version__)"
```

确认 adapter 存在：

```bash
python -c "from pathlib import Path; import nemo_skills.dataset as d; p=Path(d.__file__).resolve().parent; print({n: (p/n/'__init__.py').is_file() and (p/n/'prepare.py').is_file() for n in ('arena-hard','arena-hard-v2')})"
```

使用默认 GPT-4.1 judge 时配置凭证：

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

不要把真实 key 写进脚本、Markdown、shell history 或 `Settings.json`。

## 6. Dry-run

Dry-run 不启动模型、不调用 judge，也不要求 API key：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks arena-hard:1 --max-samples 5 --gpu-devices 0 --dry-run
```

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks arena-hard:1 --max-samples 5 --gpu-device 0 --dry-run
```

```bash
bash eval.sh --mode ar --benchmarks arena-hard:1 --gpus 1 --dry-run
```

## 7. SGLang 运行命令

先做 5 条样本 smoke：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks arena-hard:1 --max-samples 5 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_sglang_smoke
```

Arena-Hard v2 smoke：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks arena-hard-v2:1 --max-samples 5 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_v2_sglang_smoke
```

三种解码 baseline 建议分别运行：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks arena-hard:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_sglang_ar
```

```bash
bash observations/eval_sglang.sh --mode fastdiffuser --benchmarks arena-hard:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_sglang_dlm
```

```bash
bash observations/eval_sglang.sh --mode linearspec_base --benchmarks arena-hard:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_sglang_linearspec
```

正式对比必须保持 `--tokens`、候选 sampling 参数、judge、NeMo-Skills 版本和数据版本一致。

## 8. 原生 PyTorch 运行命令

AR smoke：

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks arena-hard:1 --max-samples 5 --gpu-device 0 --client-concurrency 1 --tokens 8192 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_pytorch_ar_smoke
```

dLM smoke：

```bash
bash observations/eval_pytorch_nemo.sh --mode dlm --benchmarks arena-hard:1 --max-samples 5 --gpu-device 0 --client-concurrency 1 --tokens 8192 --block-length 8 --threshold 0.9 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_pytorch_dlm_smoke
```

LinearSpec smoke：

```bash
bash observations/eval_pytorch_nemo.sh --mode linearspec_base --benchmarks arena-hard:1 --max-samples 5 --gpu-device 0 --client-concurrency 1 --tokens 8192 --block-length 32 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_pytorch_linearspec_smoke
```

原生 PyTorch pipeline 会把显式选择的 `EVAL_PYTHON` 所在目录前置到 `PATH`。这是 NeMo-Skills 本地执行模式的必要约束：顶层 runner 虽由绝对路径 Python 启动，但其 generation、judge 和 summarize 子任务使用 `python -m ...`；若不约束 `PATH`，可能误用系统 Python，导致 ABI 不匹配或找不到 `nemo_skills`。

当前原生 NLD 生成方法只暴露 temperature。本次 smoke 真正需要注意的是：PyTorch OpenAI server 接收 `top_p`，但没有将它传入模型方法。另外，原有 NeMo-Skills 链路和 server 请求模型中确实也存在 `top_k`，但 NeMo-Skills 在本框架中始终强制 `top_k=-1`（即禁用 top-k 截断），它不是本次 smoke 失败原因。为避免任意 API 调用者误以为两个参数均已生效，请求统计和汇总 metrics 仍分别记录 `top_p_requested`/`top_k_requested` 以及 `top_p_applied=false`/`top_k_applied=false`。本次 Arena-Hard 使用 `temperature=0`，MT-Bench 使用 `top_p=1.0`，所以不影响链路验证；后续若设计跨 backend 的随机采样对比，则必须纳入协议边界。

## 9. 自定义 judge

如已有另一个 OpenAI-compatible judge server，可以显式覆盖：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks arena-hard:1 --max-samples 5 --gpu-devices 0 --judge-model my-judge-model --judge-server-address http://127.0.0.1:40000/v1 --judge-server-type openai --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/arena_hard_custom_judge
```

这时 NLD 不强制检查 `OPENAI_API_KEY`。但底层 judge client 或远程服务若仍要求认证，必须按该服务的要求配置环境变量。

如果凭证只会在 SLURM job/container 内注入，可使用：

```bash
bash eval.sh --mode ar --benchmarks arena-hard:1 --gpus 1 --skip-judge-api-key-check
```

`--skip-judge-api-key-check` 只跳过提前检查，不会让 OpenAI API 绕过认证。

## 10. 输出与排错

成功时，时间戳目录中会出现：

```text
Settings.json
metrics_arena-hard.json
```

失败时会保留：

```text
error_arena-hard.json
.<timestamp>_work_<pid>/
```

常见错误：

- `OPENAI_API_KEY is not set`：默认 judge 缺少凭证。
- `does not provide the required Arena-Hard dataset adapter`：活动 NeMo-Skills 太旧或使用了错误 Python。
- `failed to prepare required Arena-Hard data`：检查 GitHub 网络访问、代理和数据目录权限。
- `metrics.json was not produced`：检查候选 benchmark log 和 judge 子任务日志；judge 请求失败也会导致最终 metrics 缺失。
- judge 返回大量 invalid judgement：检查 judge 模型是否严格遵循 Arena 输出格式；自定义 judge 不一定与 GPT-4.1 协议等价。

## 11. 验证状态

本次接入要求至少完成以下无付费验证：

- Python 语法编译。
- 三个顶层 shell 入口和三个 pipeline 的 `bash -n`。
- `eval_dlm.py --help` 包含 judge 参数。
- SGLang、原生 PyTorch、旧 SLURM 三个入口的 Arena-Hard dry-run。
- `arena-hard` 与 `arena-hard-v2` adapter/prepare 文件检查。
- 公共数据 prepare 与持久缓存检查。

2026-08-10 至 2026-08-11 已完成的本地验证：

| 项目 | 结果 |
|---|---|
| 六个 shell 文件 `bash -n` | 通过 |
| `eval_dlm.py` Python 编译 | 通过 |
| 三个顶层入口 Arena-Hard dry-run | 通过 |
| 底层 NeMo-Skills Arena judge dry-run | 通过；能够创建候选、judge、Arena summarize 三阶段任务 |
| 自定义 judge 参数 | 通过；dry-run 中 judge model 正确解析为自定义值 |
| 混合 benchmark 参数隔离 | 通过；judge override 只进入 Arena-Hard，不进入同一命令中的 GSM8K |
| 缺少默认 judge key | 通过；SGLang/PyTorch 均在加载模型前明确失败 |
| `arena-hard` prepare | 通过；500 条，已生成 `test.jsonl` |
| `arena-hard-v2` prepare | 通过；750 条，已生成 `test.jsonl` |
| 持久缓存 | 已同步到 `/data1/linyewei/datasets/NLD/arena-hard` 和 `/data1/linyewei/datasets/NLD/arena-hard-v2`，源文件与缓存 SHA-256 一致 |
| SGLang 真实候选模型 smoke | 通过；Arena-Hard 2 条样本完成 2 次候选请求，失败请求为 0 |
| 原生 PyTorch AR 真实候选模型 smoke | 通过；Arena-Hard 2 条样本完成 2 次候选请求，失败请求为 0，均记录原生 NFE/请求耗时 |
| Arena-Hard judge 链路 smoke | SGLang 与原生 PyTorch 均通过；各自使用本地确定性 mock OpenAI-compatible judge 完成 4 次交换位置评分请求，只验证集成，分数无质量意义 |

这说明 SGLang 和原生 PyTorch AR 下的 Arena-Hard 候选生成、judge 请求、解析和 metrics 收口链路均可行，但正式 Arena score 仍需要有效 judge 凭证。没有凭证时不应通过跳过检查伪装成已完成正式评测。

## 12. MT-Bench 接入（2026-08-10）

### 12.1 结论与实现边界

当前活动环境中的 `nemo-skills 0.7.0` 没有标准 MT-Bench adapter，因此 MT-Bench 没有伪装成普通 NeMo-Skills dataset。新增的 `xp/mt_bench/eval_mt_bench.py` 负责 MT-Bench 特有协议，三套既有 server 仍负责候选模型推理：

| 入口 | 候选模型后端 | MT-Bench 状态 |
|---|---|---|
| `observations/eval_sglang.sh` | SGLang AR / FastDiffuser / LinearSpec | 已接入 |
| `observations/eval_pytorch_nemo.sh` | 原生 PyTorch AR / dLM / LinearSpec | 已接入 |
| `eval.sh` | SLURM + 容器 + NLD worker/load balancer | 已接入 |
| `evaluate.py` | 单进程轻量 scorer | 未接入 |

MT-Bench 仍是显式可选项，没有加入原十项默认 benchmark 列表。它会产生 160 次候选请求和 160 次 judge 请求，使用外部 judge 时会产生 API 费用。

### 12.2 固定的官方数据与协议

runner 固定到 FastChat commit `587d5cfa1609a43d192cedb8441cac3c17db105d`，下载后逐文件检查 SHA-256：

| 文件 | 数量/用途 | SHA-256 |
|---|---|---|
| `question.jsonl` | 80 个问题，每题两轮 | `119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7` |
| `reference_answer_gpt-4.jsonl` | math/reasoning/coding 参考答案 | `f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f` |
| `judge_prompts.jsonl` | 官方 single/multi-turn judge prompt | `fd283293406d024f44c174b094ef48031d0687a4682fd3a56b29b138f80281b6` |

数据默认缓存在 `/data1/linyewei/datasets/NLD/mt-bench`。`protocol.json` 同时记录 commit、URL 和 hash；文件缺失或 hash 不符时才重新下载，`--offline` 下则立即失败。

评分使用 FastChat single-answer grading；候选生成采用 NLD 模型原生 Hugging Face chat template，而不是未识别模型在 FastChat 中会落入的通用 `one_shot` template。当前协议因此准确表述为“固定 FastChat MT-Bench 数据/judge prompt + model-native candidate chat template”。

候选生成与评分细节：

1. 每题第一轮生成后，把模型的完整第一轮原始回答作为 assistant history，再发送第二轮问题。
2. `writing/roleplay=0.7`，`stem/humanities=0.1`，`extraction/math/coding/reasoning=0.0`。
3. 每轮默认最多生成 1024 token，`top_p=1.0`。
4. math、reasoning、coding 使用固定 GPT-4 reference answer。
5. 每题分别执行 first-turn judgment 和包含完整两轮内容的 multi-turn judgment，共 160 条 judgment。
6. judge 必须返回 `[[rating]]`，有效范围为 1–10；最终同时汇总 overall、turn 1、turn 2 和八类 category 分数。

`--strip-thinking` 只控制写入答案文件和发给 judge 的文本；第二轮候选生成仍看到第一轮完整原始回答，避免破坏多轮上下文。

默认 system prompt 固定为 `You are a helpful assistant.`。pipeline 会把实际候选 tokenizer 传给 runner，runner 在生成前对一轮和两轮 sentinel conversation 执行本地 prompt preflight，记录：

- chat template SHA-256；
- Transformers 版本与 tokenizer class/revision；
- rendered prompt SHA-256；
- token IDs SHA-256 与 prompt token 数；
- `enable_thinking` 和 `truncate_history_thinking`。

当前 NLD 模型目录中的 chat template SHA-256 为 `24901e3846b530e3ed20436b26ea1cd7b3768ab2b2645e31c47df1413ab289dc`。可通过 `--expected-chat-template-sha256` 或 pipeline 环境变量 `SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256` 将它设为硬校验；不匹配时会在候选请求前失败。

这项 preflight 验证 runner 本地 tokenizer 的模板和 token 序列，不冒充远程 server 的在线 token dump。正式使用的单一 backend 必须指向同一 tokenizer；如果将来需要做跨 backend 协议验证，再额外比较各 server 的 prompt/token debug 输出即可，不把它设为所有 MT-Bench 运行的强依赖。

### 12.3 judge 默认值与可比性

NLD 当前默认 judge 是 `gpt-4.1`，endpoint 是 `https://api.openai.com/v1`。这与本仓库 Arena-Hard 默认值一致，但 FastChat 固定版本脚本的历史默认名称是 `gpt-4`。因此：

- 使用默认 `gpt-4.1` 得到的是“固定 MT-Bench 数据/prompt + GPT-4.1 judge”的当前基线；
- 若要尽量复现历史 FastChat 分数，应显式传 `--judge-model gpt-4`，并记录服务端实际解析到的模型版本；
- 不同 judge 模型、模型快照、endpoint 或 prompt 版本的分数不能直接混为同一基线。

runner 还会保存 candidate/judge API 响应中的实际 `model` 字段，并在 metrics 中汇总 observed response models。正式基线仍应优先把 `--judge-model` 写成服务商提供的不可变版本名；若 endpoint 只返回别名，结果文件只能证明请求/响应中的名称，不能推断服务端背后的隐藏快照。

### 12.4 新增参数

三个顶层入口统一支持：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--judge-model NAME` | judge 模型 | `gpt-4.1` |
| `--judge-server-address URL` | OpenAI-compatible judge endpoint | `https://api.openai.com/v1` |
| `--judge-server-type TYPE` | 当前只接受 `openai` | `openai` |
| `--judge-concurrency N` | MT-Bench 并发 judge 数 | `4` |
| `--mt-bench-max-tokens N` | 每轮候选回答 token 上限 | `1024` |
| `--skip-judge-api-key-check` | 只跳过 host 侧 key 预检 | 关闭 |

候选生成并发沿用各入口原有的 `--client-concurrency`；legacy `eval.sh` 沿用 `--batch-size`。legacy runner 还显式传递 generation algorithm、steps、block length、threshold、thinking budget、LinearSpec 和 sampler，防止服务端默认值改变当前解码模式。MT-Bench 的 diffusion steps 最大取每轮 token budget，避免继续使用普通长答案任务的 8192-step 默认值。

### 12.5 运行命令

先做无副作用检查：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks mt-bench:1 --max-samples 2 --gpu-devices 0 --dry-run
```

```bash
bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks mt-bench:1 --max-samples 2 --gpu-device 0 --dry-run
```

```bash
bash eval.sh --mode ar --benchmarks mt-bench:1 --gpus 1 --dry-run
```

真实 smoke 会调用 4 次候选生成和 4 次 judge：

```bash
OPENAI_API_KEY=<your-openai-api-key> bash observations/eval_sglang.sh --mode ar --benchmarks mt-bench:1 --max-samples 2 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --mt-bench-max-tokens 1024 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/mt_bench_sglang_smoke
```

完整 80 题：

```bash
OPENAI_API_KEY=<your-openai-api-key> bash observations/eval_sglang.sh --mode ar --benchmarks mt-bench:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --judge-concurrency 4 --mt-bench-max-tokens 1024 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/mt_bench_sglang_full
```

原生 PyTorch 完整入口：

```bash
OPENAI_API_KEY=<your-openai-api-key> bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks mt-bench:1 --gpu-device 0 --client-concurrency 1 --judge-concurrency 4 --mt-bench-max-tokens 1024 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/mt_bench_pytorch_full
```

旧 SLURM 入口示例：

```bash
bash eval.sh --mode ar --benchmarks mt-bench:1 --gpus 8 --account <slurm-account> --judge-concurrency 4 --mt-bench-max-tokens 1024
```

### 12.6 输出与续跑

每个 MT-Bench 工作目录包含：

```text
eval-results/mt-bench/
  model_answers.jsonl
  model_judgments.jsonl
  prompt_preflight.json
  metrics.json
```

`metrics.json` 包含 `mt_bench_score`、turn/category 分数、无效 judgment 数、固定协议信息、资产 hash 和候选生成的 token/NFE 汇总。SGLang 与 PyTorch pipeline 会继续合并各自的 timing/decode/request stats，并在最终输出目录写入 `metrics_mt-bench.json`。

SGLang 与原生 PyTorch 顶层入口在成功后都会删除内部工作目录，因此两个 pipeline 的 compact-output 收口都会先把 benchmark 原生产物复制到最终目录的 `artifacts/<benchmark>/`，再重写 `metrics_<benchmark>.json` 中的 artifact 路径。这保证结果收口后仍可审计，不会出现 metrics 指向已删除临时文件的情况。Arena-Hard 的原生输出也通过同一收口逻辑保留。

pipeline 默认传 `--resume`。runner 分别计算 generation protocol fingerprint 和 judge protocol fingerprint：

- generation fingerprint 覆盖 candidate model、system prompt、chat-template preflight、thinking 参数、seed policy、sampling/解码参数、数据 hash 和 FastChat commit；
- judge fingerprint 覆盖 generation fingerprint、judge model/endpoint、judge sampling 参数、reference answer 和 judge prompt hash；
- 每条 judgment 还记录其完整 system/user judge messages 的输入 hash，候选回答即使被手工改动也不会复用旧 judgment。

只有 fingerprint 和输入 hash 完全一致的记录才会续跑。发现旧版记录或协议变化时会在新请求前明确拒绝 resume，要求使用新输出目录或显式移除旧 artifact，不会把两套协议静默混在同一个分数中。

runner 默认按 `base + question_id * 2 + (turn - 1)` 发送稳定的 per-turn seed；SGLang 与原生 PyTorch 路径保留该请求。旧 batched NLD worker 尚不消费 per-request seed，因此 legacy pipeline 显式使用 `--candidate-seed-mode none`，不会在结果里伪装成确定性 seed 已生效。最终正式 baseline 可以只选择一条 backend，不要求三套 backend 同时提供结果。

### 12.7 已完成验证

| 项目 | 结果 |
|---|---|
| 三个顶层入口 MT-Bench dry-run | 通过；不启动模型、不下载数据、不调用 judge |
| 六个 shell 文件 `bash -n` | 通过 |
| runner 与测试 Python 编译 | 通过 |
| 固定资产离线校验 | 通过；80 题且三个 SHA-256 一致 |
| 本地 mock candidate + judge 端到端 | 通过；验证两轮 history、稳定 seed、两种 judge prompt、分数解析和 metrics |
| 协议一致的 `--resume` | 通过；candidate/judge 请求数均保持不变 |
| 改变 system prompt 后 `--resume` | 通过；在任何新请求前拒绝混用旧答案 |
| NLD tokenizer prompt preflight | 通过；template hash 与一轮/两轮 prompt/token fingerprints 已生成 |
| SGLang 真实候选模型生成 | 通过；2 题完整 smoke 共 4 次候选请求，失败请求为 0 |
| SGLang MT-Bench judge 链路 | 通过；本地 mock judge 产生 4 条有效 judgment，只验证集成，分数无质量意义 |
| 原生 PyTorch AR 真实候选模型生成 | 通过；2 题完整 smoke 共 4 次候选请求，失败请求为 0，chat-template hash 硬校验通过 |
| 原生 PyTorch MT-Bench judge 链路 | 通过；本地 mock judge 产生 4 条有效 judgment，只验证集成，分数无质量意义 |
| compact artifact 留存复验 | SGLang 与原生 PyTorch 均通过；默认清理后 answers、judgments、preflight、原生 benchmark 输出和 backend stats 仍存在，metrics 路径有效 |
| 真实 GPT judge 评分 | 尚未执行，未产生 API 费用 |

无有效 judge 凭证时，只能说明代码路径和协议结构通过验证，不能宣称已经得到正式 MT-Bench baseline 分数。

## 13. Smoke 测试发现与处理汇总（2026-08-11）

| 发现 | 影响 | 处理/当前边界 |
|---|---|---|
| SGLang 在 GPU 0 仅约 24.6 GiB 空闲显存、`mem-fraction=0.28` 时启动失败 | 权重加载后没有足够显存建立运行时缓存；尚未发出候选或 judge 请求 | 换到约 64 GiB 空闲显存的 GPU 2，使用 `mem-fraction=0.45 --disable-cuda-graph` 后通过；这是容量配置问题，不是 benchmark 协议失败 |
| SGLang/PyTorch compact metrics 原先可能指向成功后已删除的内部产物 | answers、judgments、preflight 和 backend stats 不可审计 | 两个 pipeline 都会复制到 `artifacts/<benchmark>/` 并重写 metrics 路径；默认清理模式已复验 |
| 原生 PyTorch 的 NeMo-Skills 子任务误用系统 Python 3.14 | `pydantic_core` ABI 不匹配，judge/summarize 找不到 `nemo_skills` | 将 `EVAL_PYTHON` 目录前置到 `PATH`，子任务统一使用已验证的 Python 3.12；Arena-Hard 重试通过 |
| 原生 PyTorch server 接收 `top_p` 但模型方法只应用 temperature；原有 `top_k` 字段固定为禁用值 `-1` | 非中性截断参数不能宣称已生效；`top_k` 不是本次 smoke 失败原因 | 两个字段的请求值与实际应用状态均写入 metrics；本次 Arena `temperature=0`、MT-Bench `top_p=1.0`，不影响链路 smoke |
| smoke 候选回答使用 64/128/1024 token 的人为短上限，部分或全部请求以 `length` 结束 | 候选质量和速度不能当作完整 baseline | 只用于验证链路；正式运行恢复 benchmark 协议所需 token/context 配置 |
| 本地 mock judge 固定返回 Arena tie 和 MT-Bench 8 分 | 产生的 50.0/8.0 没有模型质量意义 | 只用于验证请求、交换位置、解析和 metrics；正式分数必须使用锁定的真实 judge |

## 14. AlpacaEval 2.0 接入（2026-08-11）

### 14.1 实现边界

当前 `nemo-skills 0.7.0` 没有 AlpacaEval 2.0 adapter。本次新增独立 runner `xp/alpaca_eval/eval_alpaca_eval.py`，接入 `observations/eval_sglang.sh` 和 `observations/eval_pytorch_nemo.sh` 两条本地 GPU 路径。候选答案走对应 backend 的 OpenAI-compatible API，评分不自行复刻，而是调用固定版本官方 AlpacaEval 包。

legacy 入口本轮没有为了形式上的“三后端覆盖”继续扩张；最终 baseline 可以只选择已验证的 SGLang 或原生 PyTorch 后端。独立 runner 的候选接口本身是 OpenAI-compatible，后续若确有需要，可在不改评分协议的前提下接到其他 server。

### 14.2 固定资产与隔离运行时

数据固定到 Hugging Face dataset revision `2edc6fad8be6b14ea7230aabfd08188da6b8b814`：

| 文件 | 数量/用途 | SHA-256 |
|---|---|---|
| `alpaca_eval_gpt4_baseline.json` | 805 条 instruction 与 GPT-4 Turbo reference | `83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4` |
| `df_gamed.csv` | 官方 length-controlled GLM 特征/正则数据 | `97aeec3f1c7de6dee6fd31fe66f1702623f4614c3efe1c6fc5f4927cd5fd674d` |
| `instruction_difficulty.csv` | 官方 instruction difficulty 资产 | `e28d875bb334f75e17acd1e4ed659b261860e3db379cf5de29060301bed0a18b` |

官方 scorer 固定为 `alpaca_eval==0.6.6`（wheel SHA-256 `8f4f218b8a1d7ef379491222e90f38446ca327930b51416d55306663cf85f28c`），补充的 `patsy==1.0.1` wheel SHA-256 为 `751fb38f9e97e62312e921a1954b81e1bb2bcda4f5eeabaf94db251ee791509c`。

这些 wheel 不会 `pip install` 到现有 Conda 环境，而是校验后解压到 `/data1/linyewei/datasets/NLD/alpaca-eval/runtime/site`，仅由 runner 临时加入 `sys.path`。这避免为接 benchmark 改写或污染当前 SGLang/PyTorch 环境。固定数据、wheel URL/hash 和版本均写入 `protocol.json`/`metrics.json`；`--offline` 下任一文件缺失或 hash 不符都会在请求前失败。

### 14.3 候选生成与 chat template

AlpacaEval 接受外部模型输出，本身没有要求所有模型采用同一个采样配置。本实现把候选协议明确固定为：

- 每个 instruction 一轮独立请求，共 805 条；
- 使用模型自己的 Hugging Face chat template；
- 默认不额外注入 system message，只发送一条 user instruction；
- `temperature=0`、`top_p=1.0`、`max_tokens=2048`；
- 默认关闭 thinking，并记录 `enable_thinking=false`、`truncate_history_thinking=true`；
- 默认发送由样本位置导出的稳定 request seed。

pipeline 会把实际模型目录作为 tokenizer 传给 runner，在首个候选请求前执行 prompt preflight。当前 NLD chat template SHA-256 仍为 `24901e3846b530e3ed20436b26ea1cd7b3768ab2b2645e31c47df1413ab289dc`；真实 SGLang smoke 记录的 Transformers 版本为 5.8.1，并保存 rendered prompt/token IDs hash。可通过环境变量 `SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256` 启用硬校验。

SGLang 与原生 PyTorch smoke 得到相同的 template、rendered prompt 和 token IDs hash，说明这两条已验证路径在候选 prompt 层面一致。原生 PyTorch 模型方法仍只应用 temperature；runner 的中性 `top_p=1.0` 会记录为 requested，但 metrics 明确标记 `top_p_applied=false`。原有 `top_k=-1` 同样保持禁用且标记为未应用。

### 14.4 官方 judge 与指标

正式默认使用官方 `weighted_alpaca_eval_gpt4_turbo` evaluator：

- judge model：`gpt-4-1106-preview`；
- 官方 prompt SHA-256：`784227e6dc2832fc08c43d2c8ea3a308e7523780187a1aaad2f85e30bac85f62`；
- `max_tokens=1`、`temperature=1`、`logprobs=true`、`top_logprobs=5`；
- 官方 `logprob_parser` 读取 `m/M` 概率并保留连续 preference；
- 官方 `PairwiseAnnotator` 对 reference/candidate 位置做确定性随机交换；
- 同时输出 raw win rate 与官方 `get_length_controlled_winrate`。

runner 将固定的本地 `df_gamed.csv` 提供给官方 GLM，避免官方 scorer 在运行时从可变的 dataset main 分支重新下载；GLM 算法本身仍调用官方实现。judge model 或 endpoint 可以覆盖，但覆盖后就是另一套协议，不能与官方默认分数直接混用。`gpt-4-1106-preview` 是协议固定的历史模型名；若服务商已不再提供它，应显式记录替代快照，而不能静默假装仍是官方默认 judge。

完整正式运行包含 805 次候选请求和 805 次 judge 请求，会产生外部 API 费用。`--max-samples` 只用于 smoke；极小样本上的 raw/LC 数值都没有质量意义，尤其 LC GLM 可能产生极端结果。`metrics.json` 用 `is_full_formal_run` 明确区分 805 条正式运行与子集 smoke。

### 14.5 运行命令

无副作用顶层检查：

```bash
bash observations/eval_sglang.sh --mode ar --benchmarks alpaca-eval:1 --max-samples 2 --gpu-devices 0 --alpaca-eval-max-tokens 128 --dry-run
```

正式全量命令：

```bash
OPENAI_API_KEY=<your-openai-api-key> bash observations/eval_sglang.sh --mode ar --benchmarks alpaca-eval:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1 --judge-concurrency 4 --alpaca-eval-max-tokens 2048 --tokens 2048 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/alpaca_eval_sglang_full
```

原生 PyTorch 全量命令：

```bash
OPENAI_API_KEY=<your-openai-api-key> bash observations/eval_pytorch_nemo.sh --mode ar --benchmarks alpaca-eval:1 --gpu-device 0 --client-concurrency 1 --judge-concurrency 4 --alpaca-eval-max-tokens 2048 --tokens 2048 --output-path /data/home/wly/dLLM/NLD_results/observations/chat_benchmarks/alpaca_eval_pytorch_full
```

若当前 SGLang fork 的 piecewise CUDA graph 在共享/繁忙 GPU 上启动不稳定，可按本次 smoke 使用：

```bash
--extra-server-args "--disable-cuda-graph --disable-piecewise-cuda-graph"
```

这只改变 SGLang 执行优化，不改变候选 prompt、解码参数或 AlpacaEval judge 协议。

### 14.6 输出、续跑与验证结果

最终 compact 目录保留：`model_outputs.json`、逐请求 `candidate_generations.jsonl`、`annotations.json`、`leaderboard.csv`、generation/judge protocol、chat-template preflight、官方 judge prompt/config 和 SGLang timing/decode stats。`metrics_alpaca-eval.json` 中的 artifact 路径已重写到最终 `artifacts/alpaca-eval/`，内部临时目录清理后仍可审计。

generation 与 judge 分别具有 protocol fingerprint。`--resume` 只在 fingerprint 一致时复用已有结果；协议变化会在发出新请求前拒绝混用。`--overwrite` 同时清理官方 annotator cache，避免更换 judge 后误复用旧 annotation。

| 项目 | 结果 |
|---|---|
| Python 编译、shell `bash -n` | 通过 |
| AlpacaEval adapter 单元测试 | 6/6 通过 |
| 固定资产下载、SHA-256 与离线复验 | 通过；805 条唯一 instruction |
| 隔离运行时导入 | 通过；AlpacaEval 0.6.6 / Patsy 1.0.1 / sklearn 1.9.0 |
| mock candidate + 官方 evaluator + mock logprobs judge | 通过；2 候选、2 judge、raw/LC metrics 均产出 |
| SGLang 真实候选生成 | 通过；2 次请求、失败 0、观察到模型名 `nemotron-labs-diffusion-8b` |
| chat-template preflight | 通过；hash 为 `24901e...a289dc` |
| SGLang 请求统计 | 2 条均生成 128 token 并以 `length` 结束；仅为链路 smoke |
| 原生 PyTorch AR 真实候选生成 | 通过；A100 GPU 0，2 次请求、失败 0、共 256 NFE |
| 原生 PyTorch 请求统计 | 模型生成 4.980 s，约 51.40 output tok/s；2 条均以 `length` 结束 |
| PyTorch `top_p`/`top_k` 审计 | `top_p_requested=[1.0]`、`top_p_applied=false`；`top_k_requested=[-1]`、`top_k_applied=false` |
| compact artifacts 与路径复验 | 通过 |
| 真实 `gpt-4-1106-preview` judge | 尚未执行，未产生 API 费用 |

本次共享 GPU 上前两次 SGLang 启动在 piecewise CUDA graph warmup 的 `FusedAddRMSNorm` 处出现 illegal memory access，均发生在候选请求之前；增加 `--disable-piecewise-cuda-graph` 后同一卡启动并完成端到端 smoke。因此该问题记录为共享 GPU/执行优化启动条件，不归因于 AlpacaEval 请求或评分实现。

## 15. 统一回退边界

Arena-Hard、MT-Bench 和 AlpacaEval 共用一个恢复脚本。它不会做 benchmark 之间的 hunk 级拆分；执行后会把本次 chat benchmark 改造涉及的评估框架文件整文件恢复到 Arena-Hard 开始前的字节级快照，并移走本次新增的 runner/文档。评估框架范围外的 chat、模型实现、实验结果等文件不会被改动。

恢复脚本位置：

```bash
bash /data/home/wly/dLLM/NLD_checkpoints/pre_arena_hard_20260810/restore_pre_arena_hard.sh --restore
```

脚本执行前会把当前待恢复文件保存到时间戳 rescue 目录。注意：由于这是整文件恢复，开始改造之后若继续手工修改这些同一评估文件，那些后续修改也会一并退回；这正是“评估框架回到完全未修改前状态”的约定。
