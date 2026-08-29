# NLD PyTorch LinearSpec 多 block size 同状态影子实验手册

> 入口：`observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh`
>
> 实现目录：`observations/pytorch_linearspec_block_size_shadow/`
>
> 默认结果根：`/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/`
>
> 默认正式配置：LinearSpec LoRA、L16 唯一提交、L4/L8/L32 影子、BF16、temperature 0、threshold 0、non-thinking、NeMo-Skills 十项协议。

## 1. 实验要回答什么

第一阶段不测动态策略的端到端加速，也不把 wall time、吞吐或显存作为结论指标。它在每个真实 L16 解码轮的同一 causal prefix、同一 seed token、同一 canonical KV 状态上，分别完整执行 L4、L8、L16、L32 的 draft 和 verify，回答：

1. 大 block 的接收长度优势有多少来自“小 block 容量之外的 token”，有多少来自 lookahead 改变后“小 block 容量以内的接收长度也发生变化”；
2. L4/L8/L16/L32 的接收长度、draft token 和逐位置置信度在同轮是否一致；
3. 当前 request 的历史接收长度/置信度能否预测下一轮的各 `A_L`；
4. 历史信息能否预测不同 block size 之间的“区间内衰减”，为后续按 request 动态选择 block size 提供监督目标。

这条链路保留 NeMo-Skills 的数据、prompt、生成请求和 scorer，所以 L16 输出还能获得与已有 PyTorch 正式评测相同性质的 accuracy。shadow 计算不进入输出文本。

## 2. 最重要的因果隔离约束

每一轮执行顺序如下：

```text
canonical L16 prefix/cache + seed
  ├─ clone cache → L4 draft+verify → 只记结果并丢弃 cache
  ├─ clone cache → L8 draft+verify → 只记结果并丢弃 cache
  ├─ clone cache → L32 draft+verify → 只记结果并丢弃 cache
  └─ clone cache → L16 draft+verify → 唯一提交 token/cache → 下一轮
```

关键保证：

- 四个分支不是把 L32 的 draft 截断成 L4/L8/L16，而是分别运行真实长度的 draft+verify；
- 所有分支从同一 causal KV 状态和 seed 开始；每个 verifier 只修改自己的 cache clone；
- 非 L16 cache 在本轮立即释放，不可能污染下一轮；
- L16 分支最后执行并唯一推进输出与 canonical cache；
- temperature 大于 0 时四个分支从同一轮 RNG 起点开始，结束后恢复 L16 分支的 RNG 终态；正式问题 1 仍应固定 `--temperature 0`，避免随机采样混淆；
- server 对模型调用使用全局锁。`--client-concurrency > 1` 可以制造请求排队，但原生 8B 模型仍串行进入 GPU；第一阶段研究的是逐 request 的 counterfactual，不声称复现 SGLang continuous batching 的吞吐。

新代码不修改 `xp/`、`method/`、SGLang fork 或已有 observation 入口；进程清理只处理本次入口自己创建的 server/reserver PID。默认端口从 `34000+GPU ID` 向上自动搜索，每次结果使用独立时间戳目录。

## 3. 接收长度与成对分解的精确定义

对 block size `L`：

- block position 0 是上一轮产生但尚未写入 causal cache 的 seed；
- draft positions 是 `1..L-1`；
- `M_L` 是 draft 从 position 1 开始与 verifier 连续相同的 token 数，范围 `0..L-1`；
- `A_L=M_L+1`，最后的 `+1` 是 verifier correction/bonus，范围 `1..L`。

例：L8 的前 5 个 draft token 连续匹配，第 6 个不匹配，则 `M_8=5`、`A_8=6`；`A` 不是只统计 accepted draft token。

对每个 `L1<L2`，逐轮定义：

```text
ΔA(L1,L2)=A_L2-A_L1
Tail(L1,L2)=max(A_L2-L1,0)
Decay(L1,L2)=min(A_L2,L1)-A_L1
ΔA=Tail+Decay
```

- `Tail`（容量尾部）：大 block 接收到了超过小 block 上限 `L1` 的部分。例：`L1=8,A_32=11`，则 `Tail=3`。
- `Decay`（区间内衰减/变化）：把大 block 截到小 block 容量后仍与小 block 有多少差异。例：`A_8=5,A_32=7`，则 `Tail=0,Decay=2`，表示 L8 lookahead 缩短后在 8 以内也少接收 2 个。`Decay<0` 表示小 block 反而更好。
- 恒等式在汇总器中逐轮校验；任何失败会令该数据集 summary 为 `invalid`。

本实验生成六组 pair，缺一不可：

```text
4→8,4→16,4→32,8→16,8→32,16→32
```

## 4. 条件生存率 S 的完整口径

每组 `L1<L2` 都生成：

```text
S(L1,L2,k)=P(A_L1≥k | A_L2≥k)=N12/N2
k=1,2,...,L1+1
```

- `N2=count(A_L2≥k)` 是分母；
- `N12=count(A_L1≥k and A_L2≥k)` 是分子；
- `k=L1+1` 也必须存在。由于 `A_L1≤L1`，如果 `N2>0`，该端点结构上为 0；如果 `N2=0`，条件概率没有定义，报告 `NA`；
- JSON 和 Markdown 都同时保留 `N2/N12/S`，不会只给比例掩盖小分母；
- 数据集内先计算 `S`，宏平均只对该格 `N2>0` 的非 AIME24 数据集等权平均，并报告参与数据集数 `D`，绝不池化所有 round。

## 5. 主分析轮与边界轮

raw trace 永远记录所有轮。默认主统计只使用 `analysis_valid=true` 的轮，即同时满足：

- 四个分支各自的 `A_L` 范围内都没有 EOS；
- canonical 剩余 generation budget 至少等于最大 block（默认 32）。

这是为了避免“某个分支只是更早看到 EOS”或最后不足 32 token 的预算边界被误解释为 lookahead 效应。`rounds.raw/analysis/excluded/budget_boundary/any_branch_eos` 会同时报告。

如需敏感性分析，可传 `--include-boundary-rounds`。该参数只改变 summary 主表的纳入范围，不删除 raw trace，也不改变 L16 输出。

## 6. 记录的指标

### 6.1 每个 block size

- `A_L` 与 `M_L`：count、mean、std、min/max、P25/P50/P75/P90/P95/P99、精确整数频数；
- `A_L/L` 均值、full-accept rate、zero-draft-match rate；
- draft forward passes（threshold>0 时可能大于 1）；
- accepted draft confidence 的本轮 mean/min/last 分布；
- 首次 rejected draft 的 confidence、top1-top2 margin、entropy 分布；
- 每个 draft position 的被连续接收率、selected-token confidence、margin、entropy、selected-is-top1 rate。

confidence 使用 draft logits 的 softmax，分母排除 MASK；不会保存全 vocab logits。

### 6.2 每个 block pair

- `ΔA/Tail/Decay` 的完整分布和精确频数；
- `P(Δ>0)、P(Δ=0)、P(Δ<0)`；
- `P(A_L2>L1)`，用于直接量化超出小 block 容量的轮；
- monotonic violation `P(A_L1>A_L2)`；
- 同轮两个 `A` 的 Pearson/Spearman；
- common draft positions 和 verifier positions 的 token agreement、all-prefix equality、first divergence position；
- `count(A_L1 | A_L2)` 条件计数矩阵；
- 上述完整 `S(L1,L2,k)`。

### 6.3 问题 2/3 的历史参照

每轮在看到当前结果之前记录 L16 历史特征：

- `prev_anchor_a`、`prev_anchor_conf`；
- `a_ma1/2/4/8`、`conf_ma1/2/4/8`（窗口可用 `--history-windows` 控制）；
- `a_ewma05`，alpha=0.5。

汇总对这些历史特征计算与以下当前目标的 Pearson/Spearman：

- `A4/A8/A16/A32`；
- 六组 pair 的 `ΔA/Tail/Decay`。

此外用各接收长度历史特征直接预测当前 `A_L`，先裁剪到 `[1,L]`，记录 MAE 分布。这只是无训练的 reference baseline，不能替代后续按 request 分组、train/validation/test 切分的策略实验。

## 7. 结果目录与即时写入语义

每次非 dry-run 都创建：

```text
/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/
├── Settings.json
├── Settings.md
├── report.md
├── benchmark_status.jsonl
├── traces/
│   └── block_size_shadow_<dataset>.jsonl
├── summaries/
│   └── block_size_shadow_<dataset>.json
├── metrics/
│   └── metrics_<dataset>.json
├── eval_runs/<dataset>/eval-results/<dataset>/
└── runtime/<dataset>/
    ├── server.log
    ├── nemo_skills.log
    └── pytorch_request_stats.jsonl
```

创建时间顺序：

1. 建立时间戳目录；
2. 立即原子写 `Settings.json` 和中文 `Settings.md`；
3. 立即建立 `benchmark_status.jsonl` 的 initialized 事件；
4. 立即生成带待运行行、变量解释和表结构的 `report.md`；
5. 每个数据集开始时写 `running`；完成或失败时追加状态并原子刷新 `report.md`；
6. 单项失败不删除已经完成的数据集结果，后续数据集继续运行；最终进程以非零退出提醒检查。

`report.md` 的表格列全部居中，列名采用短变量名；每个变量在报告末尾配中文含义和例子。完整机器可读统计在 summary JSON，Markdown 保留解答问题 1 与支撑问题 2/3 的主表。

## 8. 数据集与宏平均

默认十项与已有 PyTorch+NeMo-Skills 管线一致：

```text
gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1
```

`dataset:1` 是每道题生成一次（pass@1），不是只跑一个样本；单数据 smoke 要加 `--max-samples 1`。

正确率字段：

- GSM8K、MATH-500、AIME24/25、GPQA、MMLU：`symbolic_correct`；
- HumanEval、MBPP：`passing_base_tests`；
- IFEval：`average_score`；
- LiveCodeBench-C++：`accuracy`。

AIME24 的 raw trace、summary、metrics 和报告数据集行全部保留，但不参与任何宏平均。其余数据集的宏平均为“先数据集内统计，再等权平均”；不会因为 GSM8K/MMLU 样本多就覆盖 AIME25/GPQA。报告 `D` 表示当前参与该格平均的数据集数。

## 9. 推荐命令（均为单行）

先激活环境：

```bash
conda activate nld_sglang
```

只解析参数、自动选 GPU 和端口，不创建结果、不加载模型：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --block-sizes 4,8,16,32 --anchor-block-size 16 --gpu-device auto --gpu-candidates 0,1,2,3 --gpu-min-free-gb 24 --tokens 128 --context-length 2176 --max-samples 1 --dry-run
```

真实 GSM8K 一样本 smoke（正式实现自检推荐）：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --block-sizes 4,8,16,32 --anchor-block-size 16 --history-windows 1,2,4,8 --gpu-device auto --gpu-candidates 0,3 --gpu-min-free-gb 24 --tokens 96 --context-length 2144 --max-samples 1 --temperature 0 --threshold 0 --disable-thinking
```

显式选 GPU 3：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device 3 --block-sizes 4,8,16,32 --anchor-block-size 16 --tokens 256 --context-length 2304 --max-samples 5 --temperature 0 --threshold 0 --disable-thinking
```

自动选 GPU，并最多等待 1 小时直到候选中有至少 30 GiB 空闲：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device auto --gpu-candidates 0,3 --gpu-min-free-gb 30 --gpu-wait-timeout-s 3600 --gpu-poll-interval-s 30 --block-sizes 4,8,16,32 --anchor-block-size 16 --max-samples 5
```

在 GPU 0 上先真实占位 10 GiB，再加载模型：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device 0 --gpu-memory-reserve-gb 10 --block-sizes 4,8,16,32 --anchor-block-size 16 --tokens 256 --context-length 2304 --max-samples 5
```

同时跑三个数据集：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1,math-500:1,aime25:1 --gpu-device auto --gpu-candidates 0,3 --gpu-min-free-gb 24 --block-sizes 4,8,16,32 --anchor-block-size 16 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

默认正式十项（AIME24 会运行但不进入宏平均）：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --gpu-device 3 --gpu-memory-reserve-gb 0 --block-sizes 4,8,16,32 --anchor-block-size 16 --history-windows 1,2,4,8 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

LinearSpec base（不用 LoRA）对照：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --mode linearspec_base --benchmarks gsm8k:1 --gpu-device 3 --block-sizes 4,8,16,32 --anchor-block-size 16 --tokens 256 --context-length 2304 --max-samples 5 --temperature 0 --threshold 0 --disable-thinking
```

记录 token ID 以便逐轮人工审计（文件会更大）：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device 3 --block-sizes 4,8,16,32 --anchor-block-size 16 --trace-detail tokens --tokens 256 --context-length 2304 --max-samples 5
```

边界敏感性统计：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device 3 --block-sizes 4,8,16,32 --anchor-block-size 16 --include-boundary-rounds --tokens 256 --context-length 2304 --max-samples 5
```

指定另一个结果根（仍自动建立时间戳子目录）：

```bash
bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh --benchmarks gsm8k:1 --gpu-device 3 --output-path /data/home/wly/dLLM/NLD_results/observations/my_block_shadow_runs --block-sizes 4,8,16,32 --anchor-block-size 16 --max-samples 5
```

## 10. 全部参数说明

### 10.1 实验与统计

|参数|含义|默认|
|:---:|:---:|:---:|
|`--benchmarks LIST`|逗号分隔的一个或多个 NeMo spec|正式十项|
|`--block-sizes LIST`|同轮真实模拟的所有长度|`4,8,16,32`|
|`--anchor-block-size N`|唯一提交输出/cache 的分支|16|
|`--block-size N`/`--block-length N`|anchor 的兼容别名，不替代 `--block-sizes`|16|
|`--history-windows LIST`|保存历史移动平均的窗口|`1,2,4,8`|
|`--trace-detail scalar`|只存每分支/每 pair 标量；不能生成逐位置表|—|
|`--trace-detail position`|存逐位置 confidence/margin/entropy/accepted|默认|
|`--trace-detail tokens`|在 position 基础上另存 draft/verifier token IDs|—|
|`--include-boundary-rounds`|主 summary 纳入 EOS/预算边界|关闭|

问题 1 正式实验必须使用 `--block-sizes 4,8,16,32 --anchor-block-size 16 --trace-detail position`。入口允许研究者做额外长度敏感性分析，但改变这些值就不再是本文定义的主配置。

### 10.2 模型与解码

|参数|含义|默认|
|:---:|:---:|:---:|
|`--mode`|`linearspec_lora` 或 `linearspec_base`|LoRA|
|`--model PATH`|HF remote-code 模型目录|本地 NLD-8B|
|`--served-model-name`|OpenAI API 标签|`nemotron-labs-diffusion-8b`|
|`--lora-path PATH`|draft-only LinearSpec adapter|`<model>/linear_spec_lora`|
|`--dtype`|`bfloat16/bf16/float16/fp16/float32/fp32`|BF16|
|`--threshold V`|draft 并行 unmask 阈值；0 时一轮填满|0|
|`--temperature V`|原生 token sampling temperature|0|
|`--top-p V`|协议对齐记录；当前原生 LinearSpec 不应用|0.95|
|`--tokens N`|API 最大返回 completion tokens|8192|
|`--context-length N`|prompt+内部预算+shadow 安全上限|tokens+2048|
|`--max-thinking-tokens N`|超过预算未闭合时注入 `</think>`|关闭|

server 将可用 context 内部减去最大 shadow block 作为显式 guard，因此长 prompt 不会在最后一轮 L32 shadow 才越界。LinearSpec 内部生成预算按 anchor L16 向上取整，API 最终最多返回 `--tokens`。

### 10.3 GPU、并发与端口

|参数|含义|默认|
|:---:|:---:|:---:|
|`--gpu-device ID/auto`|一个物理 GPU 或自动选择|auto|
|`--gpu-devices`|单 GPU 兼容别名；不接受列表作为直接设备|auto|
|`--gpu-candidates LIST/all`|auto 可选物理 GPU|all|
|`--gpu-min-free-gb V`|auto 的显存最低门槛|24|
|`--gpu-wait-timeout-s N`|没有合适 GPU 时最多等待秒数；0 立即失败|0|
|`--gpu-poll-interval-s N`|等待时查询间隔|30|
|`--gpu-memory-reserve-gb V`|模型前真实分配并保持的显存|0|
|`--port N`|显式端口；不传则从 `34000+GPU` 搜索|auto|
|`--batch-size 1`|接口兼容；原生 LinearSpec 强制 batch 1|1|
|`--client-concurrency N`|NeMo 同时发请求数，模型锁内串行|1|
|`--num-chunks N`|客户端数据 chunk 数|client concurrency|

auto GPU 先过滤达到显存门槛的候选，再按 GPU compute utilization、memory utilization、空闲显存排序，满足“显存够且优先算力占用小”。它不会要求 GPU 完全空闲，也不会杀死或暂停其他任务。

`--gpu-memory-reserve-gb` 是真实占位，不是“至少保留多少空闲”。例：GPU 原有 50 GiB 空闲，传 10 后模型大约只能竞争剩下 40 GiB；值过大会令模型 OOM。

### 10.4 数据、prompt 与输出

|参数|含义|默认|
|:---:|:---:|:---:|
|`--max-samples N`|每个 benchmark 最多问题数|全量|
|`--quick-test`|NeMo quick-test|关闭|
|`--enable-thinking`|server chat template 启用 thinking|关闭|
|`--disable-thinking`|显式记录 non-thinking；server 默认本来即关闭|关闭|
|`--keep-thinking`|NeMo 保留 thinking 文本|关闭|
|`--strip-thinking`|NeMo 去 thinking 并重评分|关闭|
|`--math-prompt-config NAME`|数学题 prompt_config 覆盖|空|
|`--nemo-skills-data-dir DIR`|已准备数据与 cache|`/data1/linyewei/datasets/NLD`|
|`--google-research-dir DIR`|IFEval scorer checkout|数据根下 `google-research`|
|`--prepare-missing-data`|本地无数据时允许 NeMo prepare（可能下载）|关闭|
|`--output-path DIR`|结果根，不是最终 run 目录|固定 observation 根|
|`--pytorch-python PATH`|加载 8B/server 的 Python|`nld_sglang`|
|`--eval-python PATH`|NeMo-Skills Python|同上|
|`--keep-runtime`|兼容已有入口；本观察默认保留审计产物|—|
|`--dry-run`|只校验/解析，不创建目录或加载模型|关闭|

默认不会因缺数据擅自下载。入口先用已安装数据；没有时从 `/data1/.../NLD/<dataset>` 恢复，并用 `.prepare.lock` 降低与并行实验的数据准备冲突；两处都没有才显式报错。只有传 `--prepare-missing-data` 才允许调用 NeMo prepare。

## 11. 查看实时结果（均为单行）

找到最近一次运行：

```bash
find /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results -maxdepth 1 -type d -name 'block_size_shadow_*' | sort | tail -n 1
```

实时看报告：

```bash
less /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/report.md
```

解析状态 JSONL：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/benchmark_status.jsonl'; [print(json.dumps(json.loads(x),ensure_ascii=False,indent=2)) for x in open(p) if x.strip()]"
```

查看 GSM8K 的六组 pair 名和主轮数：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/summaries/block_size_shadow_gsm8k.json'; d=json.load(open(p)); print(d['status'],d['rounds']); print(list(d['pairs']))"
```

查看 `S(4,32,k)` 的全部 `k=1..5`：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/summaries/block_size_shadow_gsm8k.json'; d=json.load(open(p)); [print(x) for x in d['survival'] if x['pair']=='4_32']"
```

检查分解恒等式、接收边界和生存率端点完整性：

```bash
python -c "import json; p='/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/summaries/block_size_shadow_gsm8k.json'; d=json.load(open(p)); print(json.dumps(d['validation'],ensure_ascii=False,indent=2))"
```

抽查 raw trace 第一轮：

```bash
head -n 1 /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results/block_size_shadow_YYYYMMDD_HHMMSS/traces/block_size_shadow_gsm8k.jsonl | python -m json.tool
```

## 12. 如何用第一阶段结果判断问题 1

推荐按以下顺序，不要只比较四个 `A均`：

1. 看六组 pair 的 `Δ`，确认大 block 是否总体更好以及是否有反例；
2. 看 `Tail` 和 `P越界`：若大部分增益来自 `Tail`，说明 L32 的优势主要是少数轮能接收超过 8/16；
3. 看 `Decay`：若 `A_L2≤L1` 的轮中 `Decay` 仍明显为正，说明 lookahead 缩短会让共同容量内也少接收；
4. 看 `S(k)`：它直接回答“大 block 至少能接收 k 时，小 block还能不能至少接收 k”；
5. 看 draft/verifier agreement 与 first divergence：区分是 draft token 本身随长度变化，还是 verifier prediction 随 block 输入变化；
6. 看逐位置 `P收/C/Mg/H`：判断长度影响从哪个位置开始积累；
7. 对数据集内结论与等权宏平均都检查，避免被大数据集 round 数支配。

`A32=6` 之类个例只用于 raw trace 抽查，正式结论必须来自上述分布、条件概率、分解和跨数据集宏平均。

## 13. 如何把结果用于问题 2/3

第一阶段的 history correlation 与 clipped-history MAE 只能说明“历史是否有预测信息”。后续策略实验至少应：

- 以 request 为分组单位切分 train/validation/test，不能随机拆 round 造成同一序列泄漏；
- 预测 `A4/A8/A16/A32` 或直接预测每个 L 的 serving cost-adjusted utility；
- 对问题 3 单独预测六组 `Decay`，因为只预测 L16 历史均值不能判断 lookahead 缩短的衰减；
- 与 always-L4/L8/L16/L32、oracle-per-round、last-A、moving-average 和 EWMA 对照；
- 到第二阶段再纳入真实 batch/concurrency 的吞吐、浪费 token、latency 和调度开销。本阶段不使用 shadow 链路的 wall time 做 serving 结论。

## 14. 自检与失败语义

代码级自检命令：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python -m unittest -v observations.pytorch_linearspec_block_size_shadow.tests.test_observation
```

静态检查命令：

```bash
bash -n observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh && /data/home/wly/.conda/envs/nld_sglang/bin/python -m py_compile observations/pytorch_linearspec_block_size_shadow/*.py
```

数据集 summary 只有在以下校验全部通过时才为 `ok`：至少一个 raw/主分析轮、每轮四个 branch、每轮六个 pair、所有 `A_L` 在合法范围、所有 pair 满足 `Δ=Tail+Decay`、所有 pair 含 `k=1..L1+1`、JSONL 无解码错误。

server 启动、NeMo 生成/scorer、metrics 缺失或 summary invalid 都会把该数据集写为 `failed`，同时保留日志、raw trace、已经完成的其他数据集和即时报告。入口只终止自己记录的 PID，不扫描或杀死任何已有 PyTorch/SGLang 实验。
