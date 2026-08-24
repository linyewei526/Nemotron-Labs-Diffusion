# NLD 原生 PyTorch + NeMo-Skills Strict Direct MASK-Redraft LinearSpec 实验手册

> 实现目录：`method/confidence_direct_mask_redraft_linearspec/`
>
> 实验入口：`method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh`
>
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/`
>
> 推荐环境：`conda activate nld_sglang`

## 1. 实验目标与隔离性

本实验实现“自主全 MASK 重起草、只接受触发位置直接修正命中”的新版 overlap。它是独立的第三套方法，不修改模型权重、模型 remote code、原生 baseline、SGLang 实验、`confidence_overlap_linearspec` 固定第二候选方法或 `confidence_mask_redraft_linearspec` 可变后缀复用方法。旧入口和旧结果均保持不变。

每次运行使用方法专用服务、请求统计文件、带 PID 的隐藏工作目录和原子创建的时间戳结果目录。默认 `--port 0` 由服务进程先绑定空闲端口再发布端口号，不会与并行实验发生“先探测、后占用”的端口竞态。同一秒创建多个结果目录时自动增加 `_01`、`_02` 后缀。

## 2. 解码逻辑

设固定 block 长度为 `L`，当前草稿为 `D=[seed,D1,...,D(L-1)]`。位置 0 的 `seed` 是上一轮 verifier 给出、尚未写入 canonical KV 的 token。对草稿已选 token 的置信度记为 `C_i`，其左侧可评分 token 的平均置信度记为 `C_imean`：

```text
token_y_drop_pct(i) = 1 - C_i / C_imean
```

置信度计算从 softmax 分母排除 MASK；seed 不进入历史均值。程序从左向右寻找第一个严格满足 `token_y_drop_pct > drop_pct_threshold` 的位置 `p`。`p` 同时是该 token 的 tensor 下标和它左侧 prefix 的长度。

存在候选且位置/生成预算允许时，一次 fused forward 使用两个长度均为 `p+L` 的 batch row：

```text
row 0 = [D 的 L 个 token] + [padding MASK × p]
row 1 = [D[:p]] + [MASK × L]
```

- row 0 只使用 causal attention，执行当前草稿的 AR verifier；尾部 padding 输出丢弃。
- row 1 的 `D[:p]` 使用 causal attention 重建旧 canonical cache 之后的 prefix；后续 `L` 个 MASK 彼此双向可见，并能看到完整 prefix，自主预测 `R=[R0,...,R(L-1)]`。
- `direct_mask_redraft_lora` 下，普通 dLLM draft 和 row 1 的 MASK suffix 使用 LinearSpec LoRA；prefill、row 0 verifier、row 1 prefix 均关闭 LoRA。实现使用 token 级本地路由，不切换 PEFT 全局 adapter。
- fused forward 中全局 `diffusion_lm` 保持关闭，两行不同注意力语义完全由显式 4D attention mask 表达。
- 输出 token 与 canonical KV cache 永远只取 row 0；row 1 不能自行提交结果。

令 `matched` 为 verifier 按 LinearSpec 一位 shift 接受的 draft token 数，令 `m=matched+1` 为本轮 verifier 发出的 token 数。原触发 token 为 `A=D[p]`，若 verifier 恰在触发位置拒绝，则正确修正为 `C=ar_tokens[p-1]`。新版严格决策为：

|状态|条件|row 1 处理|
|:---:|:---:|:---:|
|`m_lt_p`|`m<p`，在触发位置前已拒绝|丢弃|
|`direct_hit`|`m=p`、`C≠A`、`R0=C`|复用完整 `R`|
|`repeat_a`|`m=p`、`C≠A`、`R0=A`|丢弃，并单独计数“重新预测仍为 A”|
|`wrong_non_a`|`m=p`、`C≠A`、`R0≠A` 且 `R0≠C`|丢弃，并单独计数“改了但仍错”|
|`a_ok_later_reject`|`m>p` 且后续位置拒绝，即 A 在 p 正确|无论 R0 是否等于 A 都丢弃|
|`full_bonus`|整块 draft 通过并产生 bonus|无论 R0 是否等于 A 都丢弃|

因此唯一的复用途径是 `direct_hit`。复用时不截取后缀，下一轮 draft 始终是完整 `L` token 的 `R`，其首 token 已由 causal verifier 确认为 `C`。新 draft 的 confidence 历史重新开始，并继续寻找下一处 drop。EOS、生成预算结束或 thinking budget 强制替换 seed 时，即使形成 direct hit 也不会消费 row 1。

## 3. 文件职责

|文件|职责|
|:---:|:---:|
|`eval_confidence_direct_mask_redraft.sh`|完整用户入口；参数校验、自动选 GPU、创建结果目录并立即写 Settings、启动和收尾、生成报告|
|`run_pipeline.sh`|启动独立原生服务，执行 NeMo-Skills 数据准备、生成、评分、指标合并和紧凑产物复制|
|`server.py`|方法专用 OpenAI-compatible 服务；串行化模型执行并记录请求级统计|
|`generation.py`|confidence 触发、normal/fused forward、六状态决策、full-L 复用、KV/NFE/状态转移统计|
|`hybrid.py`|混合 4D attention mask 与 DynamicCache repeat/select/crop；强制 `verify_length=L`|
|`segmented_lora.py`|加载 bundled `o_proj` LoRA 并按 batch/token 位置路由|
|`merge_metrics.py`|合并 accuracy、物理 NFE、TPF/TPS、状态与下一轮验证统计，并执行硬一致性校验|
|`report_results.py`|比较 B16/B32 baseline，生成中文紧凑居中表格报告|
|`select_gpu.py`|按空闲显存与利用率自动选择物理 GPU|
|`update_settings.py`|原子更新 Settings 的实际端口、GPU 与运行状态|
|`tests/`|attention、LoRA、shift、六分支、KV、full-L、聚合与报告测试|

## 4. 环境与自检

激活环境：

```bash
conda activate nld_sglang
```

检查依赖：

```bash
python -c "import torch,transformers,safetensors,fastapi,uvicorn,nemo_skills; print(torch.__version__,transformers.__version__,nemo_skills.__version__)"
```

运行全部方法级单元测试：

```bash
python -m unittest discover -s method/confidence_direct_mask_redraft_linearspec/tests -v
```

只解析参数，不选 GPU、不加载模型、不创建任何结果目录：

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --block-size 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --dry-run
```

## 5. 推荐命令

本节所有命令均为单行形式。

### 5.1 单样本端到端 smoke

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-min-free-gb 24 --block-size 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --keep-runtime
```

### 5.2 单数据集正式测评

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks human-eval:1 --gpu-device auto --gpu-min-free-gb 24 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

当前 NeMo-Skills/EvalPlus 要求 HumanEval 与 MBPP 输入完整题集，因此不能将 `human-eval` 或 `mbpp` 与 `--max-samples`、`--quick-test` 组合。`human-eval:1` 中的 `:1` 表示 pass@1，而不是只测一题。

### 5.3 多数据集正式测评

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1,math-500:1,human-eval:1 --gpu-device 2 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.4 默认十项 benchmark

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --gpu-device 2 --gpu-memory-reserve-gb 50 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

默认列表为 `gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1`。

### 5.5 base 权重消融

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_base --benchmarks gsm8k:1 --gpu-device auto --gpu-min-free-gb 24 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.6 指定 GPU、限制自动候选或等待 GPU

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --gpu-candidates 1,2,3 --gpu-min-free-gb 28 --gpu-wait-seconds 1800 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.7 预留显存实验

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 20 --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

`--gpu-memory-reserve-gb` 会由独立进程在指定 GPU 上真实持有显存，再加载模型；退出或异常时 trap 会释放它。该参数用于模拟受限显存，不是给模型预留独占空间。

### 5.8 显式端口、结果根目录与运行日志

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device 2 --port 19083 --output-path /data/home/wly/dLLM/NLD_results --block-size 16 --drop-pct-threshold 0.15 --tokens 256 --max-samples 1 --keep-runtime
```

并行实验应优先使用默认 `--port 0`。显式端口已占用时，当前服务会直接绑定失败，不会连接到其他实验。

### 5.9 thinking budget

```bash
bash method/confidence_direct_mask_redraft_linearspec/eval_confidence_direct_mask_redraft.sh --mode direct_mask_redraft_lora --benchmarks gsm8k:1 --gpu-device auto --enable-thinking --max-thinking-tokens 6000 --keep-thinking --block-size 16 --drop-pct-threshold 0.15 --tokens 8192
```

### 5.10 手工重生成结果报告

```bash
python method/confidence_direct_mask_redraft_linearspec/report_results.py --result-dir /data/home/wly/dLLM/NLD_results/confidence_direct_mask_redraft_linearspec_YYYYMMDD_HHMMSS --baseline-b16 /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-b32 /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935 --output /data/home/wly/dLLM/NLD_results/confidence_direct_mask_redraft_linearspec_YYYYMMDD_HHMMSS/report.md
```

## 6. 全部入口参数

### 6.1 模式、模型、数据集与评分

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--mode direct_mask_redraft_lora`|普通 dLLM draft 与 row 1 MASK suffix 使用 LinearSpec LoRA|必填模式之一|
|`--mode direct_mask_redraft_base`|所有阶段仅使用 base 权重|必填模式之一|
|`--model PATH`|本地/HF 模型目录|`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`|
|`--served-model-name NAME`|本地 OpenAI API 的方法专用模型标签|`nemotron-labs-diffusion-8b-direct-mask-redraft`|
|`--lora-path DIR`|lora 模式 adapter 目录|`<model>/linear_spec_lora`|
|`--benchmarks LIST`|逗号分隔 benchmark spec；支持单项/多项|默认十项|
|`--tokens N`|每请求最多返回 completion token|8192|
|`--max-samples N`|每项只测前 N 道；HumanEval/MBPP 禁用|全量|
|`--quick-test`|NeMo-Skills quick test；HumanEval/MBPP 禁用|关闭|
|`--num-chunks N`|NeMo 客户端数据 chunk 数|client concurrency|
|`--client-concurrency N`|并发 HTTP 请求数；GPU generation 仍串行|1|
|`--math-prompt-config NAME`|数学任务 prompt_config 覆盖|空|
|`--keep-thinking`|保留 thinking 文本|关闭|
|`--strip-thinking`|在支持的数据集上剥离 thinking 后重评分|关闭|
|`--disable-thinking`|明确向 NeMo 传递禁用 thinking|关闭|
|`--enable-thinking`|模型 chat template 开启 thinking|关闭|

`--client-concurrency > 1` 只会让 HTTP 请求排队，不等于 SGLang continuous batching，因为 attention/LoRA 路由是进程级状态，服务端用锁串行模型执行。

### 6.2 解码参数

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--block-length N` / `--block-size N`|普通 draft、verifier 和 row 1 的固定 `L`|16，至少 2|
|`--drop-pct-threshold V`|第一处严格满足 confidence drop 大于 V 的阈值|0.15，范围 `[0,1)`|
|`--threshold V`|dLLM 一轮内 unmask threshold|当前必须为 0.0|
|`--temperature V`|解码温度|当前必须为 0，即 greedy|
|`--top-p V`|仅为 OpenAI/NeMo 参数对齐记录；原生方法不应用|0.95|
|`--context-length N`|服务允许的 prompt+生成预算上限|默认 `tokens+2048`|
|`--max-thinking-tokens N`|超预算后按原链路强制 `</think>` seed|空|

该实验固定 greedy 与一次性并行 unmask，以隔离新选择规则并保持 verifier-only 输出等价性。采样或多步 dLLM 应另建实验变体。

### 6.3 GPU、端口、路径与报告

|参数|含义|默认|
|:---:|:---:|:---:|
|`--gpu-device ID`|指定一个物理 GPU|`auto`|
|`--gpu-devices ID`|兼容别名；多 GPU 列表会被拒绝|`auto`|
|`--gpu-min-free-gb V`|auto 选择要求的最低空闲显存|24|
|`--gpu-candidates LIST`|auto 只考虑这些物理 GPU|全部|
|`--gpu-wait-seconds N`|没有合适 GPU 时最长等待秒数|0|
|`--gpu-memory-reserve-gb V`|模型加载前由独立进程真实占用的显存|0|
|`--dtype DTYPE`|`bfloat16`、`float16`、`float32` 及别名|bfloat16|
|`--port N`|0 为 OS 原子分配，也可显式指定|0|
|`--output-path DIR` / `--out-dir DIR`|时间戳结果根目录|`/data/home/wly/dLLM/NLD_results`|
|`--baseline-b16 DIR`|`report.md` 使用的 B16 greedy baseline|`.../eval_20260804_120138`|
|`--baseline-b32 DIR`|`report.md` 使用的 B32 greedy baseline|`.../eval_20260804_114935`|
|`--pytorch-python PATH`|原生模型服务 Python|`nld_sglang` Python|
|`--eval-python PATH`|NeMo-Skills Python|同 PyTorch Python|
|`--nemo-skills-data-dir DIR`|持久数据/cache 根目录|`/data1/linyewei/datasets/NLD`|
|`--google-research-dir DIR`|IFEval google-research checkout|`<data-dir>/google-research`|
|`--keep-runtime`|保留隐藏工作目录、server log 和中间结果|关闭|
|`--dry-run`|只校验并打印参数，不创建目录或加载模型|关闭|

### 6.4 Judge benchmark 参数

|参数|含义|默认|
|:---:|:---:|:---:|
|`--judge-model NAME`|Arena-Hard、MT-Bench、AlpacaEval judge 模型|各数据集默认|
|`--judge-server-address URL`|OpenAI-compatible judge 地址|数据集默认|
|`--judge-server-type TYPE`|judge server 类型|openai-compatible|
|`--judge-concurrency N`|judge 并发数|4|
|`--mt-bench-max-tokens N`|MT-Bench 每轮 completion 预算|1024|
|`--alpaca-eval-max-tokens N`|AlpacaEval candidate 预算|2048|
|`--skip-judge-api-key-check`|跳过入口的 `OPENAI_API_KEY` 预检|关闭|

## 7. 结果目录、Settings 与自动报告

成功运行的紧凑结果目录如下：

```text
/data/home/wly/dLLM/NLD_results/confidence_direct_mask_redraft_linearspec_YYYYMMDD_HHMMSS/
├── Settings.json
├── report.md
├── metrics_<benchmark>.json
└── artifacts/<benchmark>/
    ├── output-rs0.jsonl
    ├── direct_mask_redraft_request_stats.jsonl
    ├── pytorch_confidence_direct_mask_redraft_metrics_summary.json
    └── pytorch_benchmark.log
```

结果目录建立后立即写入 `Settings.json`，在加载模型前就记录实验目标、原始命令、完整超参、row 1 构造、strict-direct/full-L 规则、模型/LoRA、GPU/显存、端口、Python、数据路径、baseline 路径和运行状态。服务成功启动后写入实际原子分配端口；结束时更新为 `completed` 或 `completed_with_errors`。

`report.md` 在 benchmark metrics 合并后自动生成，包括：

- 新方法与 B16/B32 PyTorch+NeMo-Skills+LinearSpec LoRA greedy baseline 的逐数据集 TPF、Accuracy 和差值；
- 六种决策状态的计数与尝试占比，以及 A 正确/bonus 时 `R0=A` 与 `R0≠A` 子类；
- 每种状态当前轮与下一轮 verifier 的平均 Matched、Emitted 及逐事件差值；
- Saved、NFE、Rows、QTok 和 EOS/GenEnd/Think 边界计数；
- ModelTPS、端到端 WallTPS、峰值显存、触发覆盖率、平均触发位置和未尝试边界；
- 状态划分、下一轮配对和 full-length reuse 自动一致性校验。

AIME24 单项结果保留在所有表格中，但因已知精度问题不进入任何平均。其余数据集先各自计算指标，再按数据集等权平均；不会让样本数多的数据集覆盖样本数少的数据集。报告表格全部居中对齐，列宽只按最长字段和 Markdown 最小分隔要求设置。

## 8. 统计变量解释

|短名/字段|含义与例子|
|:---:|:---:|
|`TPF`|`completion_tokens / physical_nfe`；例如返回 100 token、20 次 encoder 调用，则 TPF=5|
|`NFE`|物理 encoder 调用次数；双行 fused batch 仍计一次，但额外工作由 Rows/QTok 暴露|
|`Rows`|所有 encoder forward 实际处理的 batch row 总数；普通 forward 加 1，fused forward 加 2|
|`QTok`|所有 forward 的 `batch_size × query_length` 总和|
|`Saved` / `redraft_saved_draft_forwards`|上一轮完整 R 被下一轮实际消费，因而省去 normal draft forward 的次数|
|`尝试` / `redraft_attempts`|存在候选并真正执行 fused verifier+redraft 的轮数|
|`Matched` / `M`|verifier 按一位 shift 接受的 draft token 数|
|`Emitted` / `E`|本轮 verifier 逻辑产生的 token 数，恒为 `Matched+1`，含修正或 bonus；即使输出随后在 EOS 截断也不改变该验证工作量|
|`本M/本E`|形成某状态的当前轮平均 Matched/Emitted|
|`下M/下E`|该状态之后确实存在下一轮时，下一轮平均 Matched/Emitted|
|`ΔM/ΔE`|对同一事件计算“下一轮−当前轮”再平均；例如本轮 M=3、下一轮 M=7，则 ΔM=+4|
|`下N`|能观察到下一轮并完成配对的事件数|
|`无下轮`|EOS、generation end 等导致不存在下一轮的事件数；不会按 0 混入下轮均值|
|`后拒R=A` / `后拒R≠A`|A 在 p 正确但后面拒绝时，row 1 在 p 保持 A/改成非 A 的次数；两者都丢弃|
|`BonusR=A` / `BonusR≠A`|整块通过时，row 1 在 p 保持 A/改成非 A 的次数；两者都丢弃|
|`EOS` / `GenEnd` / `Think`|已 direct hit，但因输出结束、生成预算结束或 thinking seed 被强制替换而未复用的次数|

聚合器会硬校验：每轮恰好属于一个状态、每次 fused 尝试恰好属于六种尝试状态之一、每个状态的 `下N+无下轮=N`、每轮恰好使用 normal/saved draft 和 normal/fused verifier、每次 reuse 均为完整 L、保存的 row 1 必须在下一轮实际消费、物理 NFE 与 prompt/draft/verify 调用总数一致。任一不变量失败时，本 benchmark 会生成错误文件而不会伪装为成功结果。
