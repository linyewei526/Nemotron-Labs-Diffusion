# NLD PyTorch + NeMo-Skills：固定 margin-risk P1 + conditional-new overlap 实验

> 实验入口：`method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh`  
> 独立代码：`method/margin_risk_p1_conditional_new_overlap_linearspec/`  
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/margin_risk_p1_conditional_new_overlap_results/`  
> 正式配置：LinearSpec LoRA、greedy、block size 16、draft threshold 0、固定 `margin_risk_threshold=0.5`  
> 正式范围：排除 AIME24/MMLU 的八个数据集全量评测，各数据集等权  
> TPF 口径：completion token / decode-only physical encoder forward，不含 prompt prefill，与当前 SGLang 口径对齐

## 1. 目的与隔离边界

本方法从既有 `margin_risk_no_p3_overlap_linearspec` 独立派生，只实际执行 P1 纠错分支，并在 crossing 数不超过 2 时保留 new。目标是进一步减少多 row 和 padding 带来的 dense 计算，同时真实测量去掉 P2 后的 TPF、TPS、dense query token slot 与状态变化。它不修改原 multi-overlap、no-P3、单位置 overlap、论文复现、其他 method 或 observation。

并行安全措施如下：

- 默认 `--port 0`，由操作系统原子分配本次 server 的空闲端口；
- 每次运行在独立结果根目录中新建带时间戳的子目录，同秒冲突自动追加编号；
- 内部工作目录含进程 PID，server、日志、request stats 和显存预留进程均为本次运行私有；
- 退出清理只针对本次入口启动的进程；
- `Settings.json` 与 `report.md` 模板在任何数据集开始前立即建立；每完成一个数据集，报告原子刷新一次。

结果结构如下：

```text
/data/home/wly/dLLM/NLD_results/margin_risk_p1_conditional_new_overlap_results/
└── margin_risk_p1_conditional_new_overlap_YYYYMMDD_HHMMSS/
    ├── Settings.json
    ├── report.md
    ├── metrics_<dataset>.json
    └── artifacts/<dataset>/
        ├── output-rs0.jsonl
        ├── pytorch_request_stats.jsonl
        ├── pytorch_margin_risk_p1_conditional_new_overlap_metrics_summary.json
        └── pytorch_benchmark.log
```

## 2. 策略定义

### 2.1 风险位置

对当前长度为 L 的 draft，从位置 1 到 L−1 左向右扫描；位置 0 是已知 seed，不参与判断。softmax 计算前排除 MASK，然后计算：

```text
margin = 最高 token 概率 - 第二高 token 概率
margin_risk = 1 - margin
```

记录前三个严格满足 `margin_risk > 0.5` 的位置 P1、P2、P3，同时继续扫描完整块以区分 crossing 总数 C 是 0、1、2 还是 3 及以上。严格大于意味着恰好等于 0.5 不触发。

例：最高、第二概率分别为 0.65 和 0.25，则 margin=0.40、margin_risk=0.60，该位置触发。

### 2.2 P1 + conditional-new 分支分配

P1 的实际替代 token 取排除原 draft token、MASK、EOS 后概率最高的 token，可理解为有效二选 token。P2/P3 的同口径 token 仍会计算，但只用于反事实审计，不构造 prospective row。当前 verifier 与 prospective row 一次前向并行，具体规则固定为：

|crossing 数 C|speculative row|加 verifier 后总 row|
|:---:|:---:|:---:|
|0|new|2|
|1|P1 + new|3|
|2|P1 + new|3|
|3 及以上|P1|2|

P2/P3 永远不构造 prospective row。仍记录它们的有效二选 token：当 verifier 首错正好位于 P2 或 P3 时，报告这个被省略的 token 本来能否修正。该统计只衡量潜在损失机会，不进入 fused batch，也绝不计作实际命中、提交或复用。

实际候选 P1 row 形式为：

```text
[当前 draft 的 0..p-1 causal prefix] + [替代 token + L-1 个 MASK]
```

new row 形式为：

```text
[当前完整 L-token draft causal prefix] + [L 个 MASK]
```

只有 verifier row 是当前输出和 canonical KV 的权威来源。P1 分支只有在 verifier 证明首错恰为 P1 且替代 token 正确后才能作为下一轮 draft；new 只在 C≤2 时构造，并且只有整块通过且 verifier bonus token 等于 `new[0]` 时才能复用。被复用的 prospective draft 下一轮仍按同一规则递归分析，所以并非只优化第一轮。

### 2.3 Padding 与 dense slot

本实验保持原始组 batch、不展平，但最多只有 3 row：一个 verifier 加 P1/new 中至多两个 prospective row。row 会 pad 到该 fused forward 的公共 query length Q；attention mask 能隔离语义，但不会跳过 dense QKV/MLP 计算。

报告显式记录：

```text
dense query token slot = row 数 × 公共 Q
总Slot = 所有 decode forward 的 dense query token slot 总和
每FwdSlot = 总Slot / decode physical forward 数
每TokSlot = 总Slot / completion token 数
```

例：100 次 decode forward 共计算 6400 slot、返回 800 completion token，则每FwdSlot=64、每TokSlot=8。`每TokSlot` 越小，表示产出相同 token 所需的 dense query-token 工作越少。它不是完整 FLOPs，因为 attention 成本还受 KV 长度影响。

## 3. 报告内容与统计口径

`report.md` 根据 `--benchmarks` 中实际传入的数据集名称和数量动态调整完成进度、主表和等权平均；AIME24/MMLU 始终不进入主表。默认正式命令恰好传入其余八个数据集。

主要报告包括：

- 新方法与既有 PyTorch+NeMo-Skills B16/B32 greedy baseline 的 decode-only TPF、NFE、TPS；
- 请求尝试数、成功数、失败数、OOM 数与效率覆盖率；
- C0/C1/C2/C3+、实际 R2/R3、P1/new 分支和复用漏斗；
- 首错未命中、实际 P1 二选修正正确或错误、P2/P3 省略但本可修正或二选也错误、整块通过时 new 命中/未命中/不存在；
- 每种互斥状态的本轮接收长度、同 request 下一轮接收长度及逐对差值；
- 每 forward dense token 的均值、Min、P50/P90/P95/P99、Max、有效 slot、padding slot、padding 比例、row 数与 Q；
- decode 总Slot、每FwdSlot、每TokSlot。

八集宏平均先在每个数据集内部计算，再对八个数据集算术平均，不按 sample 数加权。绝对次数另给总计，不能把总计比例误当作等权比例。

TPF 定义为：

```text
TPF = 返回的 completion token 总数 / decode physical encoder forward 总数
```

prompt prefill 单独记为 prefill forward，不进入 TPF。无论 fused forward 是 2 或 3 row，都只计一次 physical forward，所以解读 TPF 时必须同时查看每FwdSlot、每TokSlot和 padding。

入口默认 `--efficiency-only`：accuracy 不作为完成标准；单 request CUDA OOM 会被记录并从效率均值排除，报告通过 Att/OK/Fail/OOM/Cov 披露覆盖率。加 `--require-accuracy` 才恢复严格 scorer/accuracy 与错误处理。

## 4. 推荐单行命令

以下每条命令均为单行。

### 4.1 查看帮助

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --help
```

### 4.2 只检查参数，不创建目录、不加载模型

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --gpu-device 0 --gpu-memory-reserve-gb 0 --dry-run
```

### 4.3 一题全链路 smoke

GSM8K 子集能覆盖模型加载、动态端口、NeMo-Skills 请求、指标合并、Settings 与增量报告；不会启动完整数据集。

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 1 --tokens 64 --context-length 2112 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device auto --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --efficiency-only --keep-runtime
```

### 4.4 正式八数据集全量评测

该命令不传 `--max-samples`，因此八集均为全量；不包含 AIME24/MMLU。默认效率模式正是本实验关注的口径。

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1 --tokens 8192 --context-length 10240 --block-size 16 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --top-p 0.95 --disable-thinking --client-concurrency 1 --num-chunks 1 --gpu-device 3 --gpu-memory-reserve-gb 0 --efficiency-only --output-path /data/home/wly/dLLM/NLD_results/margin_risk_p1_conditional_new_overlap_results
```

### 4.5 单数据集与自定义多数据集

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1,aime25:1 --tokens 8192 --block-size 16 --margin-risk-threshold 0.5 --gpu-device 2
```

`human-eval:1` 和 `mbpp:1` 的 `:1` 表示 pass@1，不是只取一个 sample。默认 efficiency-only 允许对子集做链路验证；严格 accuracy 模式下，当前 EvalPlus scorer 通常要求完整题集。

### 4.6 自定义 block size

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1,math-500:1 --tokens 8192 --block-size 32 --threshold 0 --margin-risk-threshold 0.5 --temperature 0 --gpu-device 2
```

`--block-size` 是 `--block-length` 的别名；所有 draft、verify 和 prospective block 使用同一个 L。正式P1 + conditional-new 对比默认 L=16，但代码支持任意至少为 2 的 L。

### 4.7 指定 GPU、自动选择与等待

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --gpu-device 3
```

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --gpu-device auto --gpu-candidates 1,3,5 --gpu-min-free-gb 28 --gpu-wait-seconds 1800
```

后端是单 GPU。`--gpu-devices` 是兼容别名，但同样只接受一个 ID；逗号列表只用于 `--gpu-candidates` 的自动选择候选集。

### 4.8 模拟显存受限

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --tokens 8192 --block-size 16 --gpu-device 3 --gpu-memory-reserve-gb 20
```

`--gpu-memory-reserve-gb` 会在模型加载前由本次运行的独立进程真实占用显存，并保持到实验退出，用来模拟受限环境；它不是替模型保护可用显存。退出 trap 只终止本次预留进程。

### 4.9 显式端口、结果根目录和 baseline

```bash
bash method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh --mode overlap_lora --benchmarks gsm8k:1 --max-samples 2 --tokens 512 --block-size 16 --gpu-device 1 --port 19081 --output-path /data/home/wly/dLLM/NLD_results/margin_risk_p1_conditional_new_overlap_results --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935 --keep-runtime
```

并行实验优先使用 `--port 0`。显式端口已经被占用时 bind 会失败，不会误连其他 server。

### 4.10 只重建已有结果报告

```bash
python method/margin_risk_p1_conditional_new_overlap_linearspec/report.py --result-dir /data/home/wly/dLLM/NLD_results/margin_risk_p1_conditional_new_overlap_results/margin_risk_p1_conditional_new_overlap_YYYYMMDD_HHMMSS --baseline-block16-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138 --baseline-block32-dir /data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935
```

此命令不启动 GPU；它按现有 `Settings.json` 中的 benchmark 范围读取紧凑 metrics，并原子重写 `report.md`。

## 5. 参数说明

### 5.1 模型、数据集与输出

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--mode overlap_lora`|normal draft 与 prospective suffix 使用 LinearSpec LoRA|正式模式|
|`--mode overlap_base`|不加载 LoRA 的消融|可选|
|`--model PATH`|本地/HF 模型目录|本地 8B checkpoint|
|`--lora-path DIR`|LinearSpec LoRA|`<model>/linear_spec_lora`|
|`--served-model-name NAME`|本地 OpenAI API 模型标签|本方法专用标签|
|`--benchmarks LIST`|逗号分隔，支持单/多数据集|默认八集|
|`--tokens N`|每请求最多返回 completion token|8192|
|`--context-length N`|prompt+补齐后生成预算的上限|未显式传入时为 tokens+2048|
|`--max-samples N`|每数据集最多样本数|默认全量|
|`--quick-test`|NeMo-Skills quick 模式|默认关闭|
|`--output-path DIR`|结果根目录|方法独立目录|
|`--keep-runtime`|保留 server log、完整中间结果|默认成功后清理|

### 5.2 解码与评估

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--block-size N` / `--block-length N`|draft/verify/prospective 的 L|16，至少 2|
|`--threshold V`|LinearSpec draft unmask 阈值|必须为 0|
|`--margin-risk-threshold V`|strict crossing 阈值|0.5，范围 0 到 1|
|`--temperature V`|生成温度|必须为 0|
|`--top-p V`|为接口对齐而记录|native 方法当前不应用，默认 0.95|
|`--disable-thinking`|显式关闭 chat-template thinking|正式命令使用|
|`--enable-thinking`|启用 thinking|与 disable 互斥|
|`--max-thinking-tokens N`|超过预算时强制结束 thinking|默认空|
|`--client-concurrency N`|HTTP 客户端并发|默认 1；模型执行有锁串行|
|`--num-chunks N`|NeMo 客户端 chunk 数|默认等于客户端并发|
|`--efficiency-only`|accuracy/scorer 不作为完成目标，OOM request 可跳过|默认|
|`--require-accuracy`|恢复严格 accuracy/scorer 与 OOM 行为|可选|

### 5.3 GPU、端口与依赖

|参数|含义|默认/限制|
|:---:|:---:|:---:|
|`--gpu-device ID`|指定物理 GPU|默认 auto|
|`--gpu-devices ID`|上项兼容别名|只允许一个 ID|
|`--gpu-candidates LIST`|auto 模式候选 GPU|默认所有可见设备|
|`--gpu-min-free-gb V`|auto 选择最低空闲显存|24 GiB|
|`--gpu-wait-seconds N`|没有合适 GPU 时最多等待|0|
|`--gpu-memory-reserve-gb V`|启动模型前模拟占用 V GiB|0|
|`--dtype DTYPE`|模型 dtype|bfloat16|
|`--port N`|本地 API 端口|0，原子找空闲端口|
|`--pytorch-python PATH`|模型 server Python|nld_sglang 环境|
|`--eval-python PATH`|NeMo-Skills Python|默认同上|
|`--nemo-skills-data-dir DIR`|数据与持久缓存目录|本地 NLD 数据目录|
|`--google-research-dir DIR`|IFEval 依赖目录|数据目录下 google-research|

## 6. 自检与结果解读顺序

建议按以下顺序检查正式结果：

1. `Settings.json` 中 benchmark 是否恰为八集、block size 是否为 16、margin-risk 是否为 0.5；
2. `report.md` 顶部是否最终显示 8/8，Cov 是否接近 100%；
3. 比较新 TPF 与 B16/B32，但同时比较每TokSlot与 fused R2/R3 分布；
4. 查看 `P2本可修` 与 `P3本可修` 比例：它们是省略分支的潜在复用机会，不等于最终 TPF 差值；
5. 查看实际 P1/new 的验中、可复用、已复用，确认剩余分支的真实贡献；
6. 对不同数据集先分别解释，再使用八集等权均值形成总判断。

静态和单元测试命令：

```bash
bash -n method/margin_risk_p1_conditional_new_overlap_linearspec/eval_margin_risk_p1_conditional_new_overlap.sh method/margin_risk_p1_conditional_new_overlap_linearspec/run_pipeline.sh && python -m py_compile method/margin_risk_p1_conditional_new_overlap_linearspec/*.py && python -B -m unittest discover -s method/margin_risk_p1_conditional_new_overlap_linearspec/tests -p 'test_*.py'
```

真实模型 fused-path smoke 命令：

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/wly/.conda/envs/nld_sglang/bin/python method/margin_risk_p1_conditional_new_overlap_linearspec/tests/smoke_fused_multi.py --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --block-size 16 --dtype bfloat16
```

该 smoke 同时核验 C3+ 的 verifier+P1 为 2 row、C2 的 verifier+P1+new 为 3 row、融合 verifier 与普通 causal verifier 的 argmax 一致、候选 seed 正确，以及 canonical prefill KV 未被 prospective row 污染。
