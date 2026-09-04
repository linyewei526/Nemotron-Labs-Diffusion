# NLD SGLang + NeMo-Skills 动态 block size 历史信号实验

> 实验入口：`observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh`
>
> 独立实现目录：`observations/sglang_dynamic_block_history_signal/`
>
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/`
>
> 前置设计备忘录：`configs/observations/NLD_PyTorch_LinearSpec_dynamic_block_size_history_signal_design_zh.md`
>
> 参数基线：`configs/observations/NLD_SGLang_NeMoSkills_eval_pipeline_zh.md`

## 1. 实验要回答什么

本实验在真实 SGLang + NeMo-Skills LinearSpec 上检索只依赖同一 request 过去轮信息的动态 block 策略。候选动作固定为 L8、L16、L32，所有 request 第一轮固定 L16。

重点不是预测精确接收长度，而是回答两类 compute-bound 决策：

- S8 目标：通常使用 L8；只有历史信号高置信地表明 L16/L32 能带来显著接收增益时才升级。
- S16 目标：通常使用 L16；只有显著增益信号才升级 L32，只有强无损信号才下调 L8。

最终策略只能有一套跨数据集参数。AIME24 始终排除；其余九个数据集等权，MMLU 默认仅取 2000 个样本并最后运行。

## 2. 为什么这是“真实动态轨迹”，不是固定 L16 的离线拼接

SGLang 调度器每轮按物理 L32 分配 KV 槽。观察算法先根据历史动作把当前 request 分成 L8/L16/L32 动作桶；随后在每个动作桶内部，以完全相同的 request 组成、已提交 prefix、KV 和 seed 依次运行三种真实变长 ForwardBatch。该桶预选的 canonical block 总是最后执行，并直接作为提交结果，因此不再发生“完整 batch shadow、较小动作子组 replay”带来的数值差异。

调用链如下：

```text
NeMo-Skills request
  -> SGLang prompt prefill
  -> H_t 只读取此前 canonical 历史
  -> 当轮动作 L_t（此时看不到当轮 shadow）
  -> 按 L_t 形成动作桶
  -> 每个桶内同 request 组成运行 L8/L16/L32 draft+verify
  -> L_t 分支最后执行并直接提交正确 KV
  -> 仅用 L_t 分支更新 H_(t+1)
```

公共 `sglang_dllm` 源码和原 `observations/eval_sglang.sh` 均不被本实验替换。入口只给本次启动的 Python 进程临时 prepend 独立 `sitecustomize.py`，把该进程注册表中的 `LinearSpec` 指向观察类；不带该环境变量启动的既有 SGLang/PyTorch 实验完全保持原行为。

变长 shadow 必须使用 eager attention metadata，入口会强制附加 `--disable-cuda-graph`。这是观察实验的正确性约束，不是部署结论。

## 3. 数据、切分和权重

默认九数据集顺序为：

```text
gsm8k,human-eval,mbpp,math-500,aime25,gpqa,ifeval,livecodebench-cpp,mmlu
```

即使用户把 MMLU 写在列表中间，入口也会将它移到最后。传入 AIME24 会直接报错。

每条 prompt 直接从 SGLang request 的完整原始 `origin_input_ids` 计算稳定 SHA-256 指纹，并按 `--split-seed` 划分。该值不读取 prefix-cache 后的 prefill suffix，因此请求顺序、batch 和 radix cache 命中不会改变切分：

- 70% train：拟合历史信号模型；
- 15% selection：选择显著增益定义、特征组、模型族和概率阈值；
- 15% test：不参与拟合或阈值选择，只做 held-out 评估。

所有拟合、选择和汇报都显式采用：

```text
数据集等权 -> 数据集内 request 等权 -> request 内轮次等权
```

例如 MMLU 有 2000 个样本而 AIME25 只有几十个样本时，两者对全局目标仍各占九分之一；长 response 也不会因轮数多而盖过短 response。

`大精`、`大浪`、`8安`等条件指标也严格保持数据集等权：先在每个实际发生升级/下调的数据集内部计算条件比例，再对这些数据集做算术宏平均。这样某数据集即使产生很多升级轮也不能盖过另一个数据集；完全没有该类动作的数据集不提供条件观测，不会被虚构成 100% 精度，也不会参与该条件指标的分母。

## 4. 记录的 trace

每轮 JSONL 至少包含：

- request ID、prompt 指纹、数据集、轮号；
- 当轮决策前的完整历史特征、决策块长、决策来源和概率分数；
- L8/L16/L32 的接收长度、匹配 draft 数、full/EOS、输出 token 和下一 seed；
- 每个 draft 位置的 top1 confidence、top1-top2 margin、margin_risk 和 entropy；
- accepted confidence 均值/最小/末位、首个 rejected confidence/margin/entropy；
- canonical 是否由同动作桶 chosen-last 分支直接产生，以及三分支公共接收前缀是否一致。

搜索前会审计 `canonical_replay_match` 和三分支公共接收前缀。存在歧义的轮次不会进入拟合、selection 或 test，并会按数据集写入 `report.md` 的原始/可用/排除/回放异常/跨块异常表；排除后重新计算“数据集等权→request 等权→有效轮次等权”。默认排除比例上限为 5%，超过即拒绝生成策略，防止静默掩盖损坏 trace。新版同动作桶 chosen-last 采集从构造上消除了 canonical 二次 replay。

历史特征包含接收均值/比例/趋势、full/non-full streak、窗口 1/2/4/8、超过 8/16 的删失感知 known rate/count，以及公共前 7 个 draft 位置的 confidence/margin/entropy。小块 full 不会被误记为“能力只能到小块上限”。dataset ID 不进入模型。

## 5. 搜索空间和保守目标

当前完整检索包含：

- 特征组：接收历史、接收+删失、接收+删失+confidence/margin/entropy；
- 可部署模型：L2 逻辑回归、最大深度 3 的浅决策树；
- 信号上界诊断：深度 3/6 的直方图 GBDT，只报告 held-out AUROC/AUPRC，不参与冻结策略和 serving；
- 历史窗口：1/2/4/8 轮及 4 轮趋势；
- S8 的 L16 最小增益、L32 最小增益及各自单位额外 block 成本增益；
- S16 的 L32 最小增益/边际效率，以及 L8 最大允许损失；
- 两个动作概率阈值分别搜索 0.50、0.60、0.70、0.80、0.85、0.90、0.95、0.98。

默认策略选择约束：

- 所有升级轮中，达到所定义“显著接收增益”的等权精度至少 80%；
- 所有升级轮中，实际只多接收不超过 1 token 的浪费比例至多 10%；
- S16 下调 L8 的轮中，A8 不低于 A16 的比例至少 98%。

满足约束后最大化“平均接收 - 0.10 × 平均 block”；若没有可行项，报告按约束违背程度惩罚后的最佳项，而不会静默放宽要求。阈值和约束都可通过命令行控制。

搜索时终端持续输出 `[s8] 已完成/总候选` 和 `[s16] 已完成/总候选` 百分比；`report.md` 也会在每个数据集或阶段完成后原子刷新。

## 6. 五种 stage

`collect`：真实运行探索行为策略。首轮固定 L16，之后分层 Markov 转移，覆盖 8/16/32 的保持、相邻迁移和少量跨级迁移。只采集监督 trace，不宣布最终策略。

`search`：候选评分本身只使用 CPU，读取已有 `traces/explore/*.jsonl`，按九数据集等权完成 S8/S16 全局检索并写出 `policy_s8.json`、`policy_s16.json`。与此同时，主脚本通过现有 baseline pipeline 正常运行九个非 AIME24 数据集的 SGLang+NeMoSkills 复现，使 GPU 不仅保留模型和 KV pool，而且持续执行真实请求；CPU 搜索结束时立即终止 baseline，无论它当时运行到哪个数据集。默认会检查动态 trace 的数据集集合必须恰好是上述九个；缺一个也会拒绝生成“正式全局最优”策略。

`validate`：读取冻结策略，在真实 SGLang 上重新产生动态 canonical；每轮仍保留三分支作为只读标签，但决策看不到当轮标签。最终报告同时列出九数据集全部验证样本的逐集/集等权结果，以及 prompt-hash test request 的严格 held-out 视图。

`remaining`：必须传入已有 `--run-dir`。复用已完成探索 trace，依次执行 search、S8 validate、S16 validate；已有且 JSON 完整的搜索/验证结果会跳过，已有 completed 事件和非空 trace 的数据集也会跳过。未显式覆盖的模型、数据集、GPU、batch、并发和搜索约束都从该 run 的 `settings.json` 恢复。单个数据集遇到临时 502、服务退出或空 trace 时，会清空该次不完整 trace、启动全新 server 自动重试，不会把部分结果误标为完成。

`all`：依次执行 collect、search、S8 validate、S16 validate。正式实验推荐分阶段运行，便于先检查探索 trace，再决定何时占用 GPU 做冻结验证。

## 7. 推荐命令（全部为单行）

进入项目：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

查看完整帮助：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --help
```

只检查参数、MMLU 顺序和资源设置，不创建结果目录、不启动 GPU：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --gpu-devices 0 --batch-size 4 --client-concurrency 4 --dry-run
```

在默认九个非 AIME24 数据集上采集正式探索 trace，MMLU 只取 2000 个样本：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --gpu-devices 0 --batch-size 4 --client-concurrency 4 --mmlu-max-samples 2000
```

在已有 run 上执行完整搜索；候选计算使用 CPU，同时 guardian baseline 使用并占用该 run 的 GPU：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_YYYYMMDD_HHMMSS
```

在九数据集上真实重跑冻结的 S8/S16 策略：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_YYYYMMDD_HHMMSS --gpu-devices 0 --batch-size 4 --client-concurrency 4
```

从探索到两类冻结验证全部连续执行：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage all --gpu-devices 2 --batch-size 4 --client-concurrency 4 --mmlu-max-samples 2000
```

只跑两个数据集做链路检查：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --benchmarks gsm8k:1,math-500:1 --max-samples 2 --tokens 64 --context-length 512 --gpu-devices 0 --batch-size 1 --client-concurrency 1
```

在上述开发性部分 trace 上试跑离线检索时，必须明确声明它不是九集正式策略：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_YYYYMMDD_HHMMSS --allow-partial-search
```

使用两个 GPU 做 TP=2，并在每张卡预留 8 GiB：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --gpu-devices 0,1 --tp-size 2 --gpu-memory-reserve-gb 8 --batch-size 4 --client-concurrency 4
```

使用无 LoRA 的 LinearSpec 对照：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --mode linearspec_base --benchmarks gsm8k:1 --max-samples 10 --gpu-devices 0
```

指定端口；不指定时沿用公共 SGLang pipeline 的空闲端口自动搜索：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --benchmarks gsm8k:1 --port 34100 --proxy-port 35100 --gpu-devices 0
```

提高“只有明确收益才扩大 block”的保守程度：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_YYYYMMDD_HHMMSS --min-large-precision 0.90 --max-large-waste 0.05 --min-safe8 0.99
```

## 8. 参数逐项解释

### 8.1 实验和数据参数

- `--stage`：必填。选择探索采集、离线搜索、冻结验证、从已有探索继续全部剩余阶段，或从头全流程。
- `--run-dir`：`search/validate/remaining` 必填，指向已有时间戳目录；续跑会恢复未显式覆盖的原设置并跳过完整产物，`collect/all` 不传时自动新建。
- `--benchmarks`：NeMo-Skills `name:reps` 逗号列表，支持单/多数据集；拒绝 AIME24，自动把 MMLU 移到最后。
- `--max-samples`：非 MMLU 数据集统一样本上限。正式实验留空即完整数据；smoke 可设 1–10。
- `--mmlu-max-samples`：只控制 MMLU，默认 2000，约为完整 MMLU 的 10%–20%。
- `--tokens`：每个 response 最多生成 token 数，默认 8192，与现有 NeMo-Skills 复现实验一致。
- `--context-length`：prompt+completion 的 SGLang 最大上下文，默认 10240。
- `--temperature`：必须为 0，保证同状态三分支是严格 greedy 反事实。
- `--top-p`：传给 NeMo-Skills 的兼容参数；temperature=0 时不会引入随机采样。

### 8.2 block 和策略参数

- `--block-sizes`：shadow 候选，当前成对反事实协议严格固定 `8,16,32`；传入其他集合会在启动模型前报错。
- `--block-size`：SGLang 调度器实际分配的物理 KV block，默认 32，必须不小于最大候选；它不是固定 canonical 动作。
- `--seed`：探索动作的稳定 hash 随机种子。同 request、同轮在重复运行时得到相同探索动作。
- `--split-seed`：prompt 指纹 train/selection/test 切分种子。
- `--min-large-precision`：升级大块后的显著收益精度下限，默认 0.80。
- `--max-large-waste`：升级但只多接收至多 1 token 的比例上限，默认 0.10。
- `--min-safe8`：S16 下调 L8 时的严格无损率下限，默认 0.98。
- `--max-invalid-row-rate`：canonical replay 或跨 block 公共前缀存在歧义时允许保守排除的总轮次比例上限，默认 0.05；超过即失败，不能用来放宽策略收益约束。
- `--dataset-max-attempts`：每个数据集最多使用全新 SGLang server 尝试的次数，默认 3；设为 1 可关闭自动重试。
- `--dataset-retry-delay-s`：两次数据集尝试之间等待的秒数，默认 10。每次重试都会从头生成该数据集，绝不续接失败尝试的 request 历史或部分 trace。
- `--allow-partial-search`：仅用于早期开发/链路检查，允许不满九数据集时运行 `search/validate`；正式结果绝对不应使用此项。

### 8.3 GPU、并行和 SGLang 参数

- `--gpu-devices`：传具体编号时作为 `CUDA_VISIBLE_DEVICES`，支持单卡或逗号分隔多卡；传 `auto` 时，每个未完成数据集及其重试开始前重新选一张达到空闲显存门槛的卡，先保证显存足够，再优先当前利用率低者。显式编号永远不会被自动替换。
- `--auto-gpu-min-free-gb`：`--gpu-devices auto` 的最低空闲显存门槛，默认 48 GiB，依据本实验实际 SGLang 进程约 46–47 GiB 峰值留出少量余量。当前 auto 只支持单卡 TP=1；多卡 TP 必须显式给出 GPU 列表。
- `--tp-size`：tensor parallel 数；不传时由公共入口按 GPU 数推断。
- `--batch-size`：SGLang `max_running_requests`。动态动作可在同一调度 batch 内不同；观察实现先按动作分桶，再在同一动作桶内用完全相同 request 组成执行三种 block，chosen block 最后执行并提交。
- `--client-concurrency`：NeMo-Skills 和 timing proxy 最大并发 request 数。
- `--gpu-memory-reserve-gb`：加载 SGLang 前在每张选中 GPU 上预占多少 GiB，用于模拟 serving 已有显存占用；0 表示不预留。
- `--mem-fraction`：SGLang KV/static memory fraction，默认 0.55。
- `--dtype`：默认 bfloat16；应与已有 SGLang 复现实验保持一致。
- `--port/--proxy-port`：SGLang/timing proxy 端口；省略时公共 pipeline 自动寻找空闲端口。
- `--lora-path`：`linearspec_lora` 的 draft LoRA 路径；留空使用现有 SGLang work dir 默认 adapter。
- `--lora-mode`：`draft_only` 或 `both`，默认与现有 SGLang LinearSpec 对齐为 `draft_only`。
- `--extra-server-args`：追加 SGLang 参数。入口无论如何都会加 `--disable-cuda-graph`，不能用此项重新开启 graph。
- `--sglang-python/--eval-python`：分别控制模型/搜索 Python 与 NeMo-Skills Python。
- `--sglang-src/--sglang-work-dir`：覆盖 SGLang 源码和缓存目录。

## 9. 输出结构

一次新运行立即建立：

```text
dynamic_block_history_YYYYMMDD_HHMMSS/
  settings.json
  settings.md
  report.md
  run_state.json
  traces/explore/<dataset>.jsonl
  traces/validate_s8/<dataset>.jsonl
  traces/validate_s16/<dataset>.jsonl
  search/search_results.json
  search/policy_s8.json
  search/policy_s16.json
  search/validation_s8.json
  search/validation_s16.json
  eval_runs/...
  runtime/runner.lock
  runtime/attempt_traces/<阶段>/<dataset>/attempt_*.jsonl
  runtime/search_guardian_baseline/...
```

`settings.md` 在模型启动前写入完整超参和原始命令。`report.md` 从目录创建时就是可读模板；每完成一个数据集、搜索或验证阶段就更新，不必等九数据集全部结束。

报告中的“均(集等权)”永远是当前已完成非 AIME24 数据集的算术宏平均；正式搜索和验证默认必须凑齐九集，因而最终表一定含九集逐项结果和九集宏平均。表格前都会就地解释变量并给出例子，列采用居中、紧凑 Markdown 格式。

## 10. 指标含义和结论边界

- `块均`：策略平均选择的 block size，越低代表理论计算 token 越少。
- `接均`：策略分支平均每轮验证通过/发出的 token。
- `TPF代`：等权 `接均/块均`，只用于检索计算收益方向，不是实际 serving TPF。
- `损32`：相对同状态 L32 少接收的 token/轮。
- `损默`：相对 S8 或 S16 默认小块少接收的 token/轮。
- `大精`：所有升级轮中，真实增益达到所选标签显著门槛的比例。
- `大浪`：所有升级轮中，真实增益不超过 1 token 的比例。
- `8安`：S16 下调 L8 时，A8 不低于 A16 的比例。
- `AUROC/AUPRC`：历史信号对“更大 block 是否值得”或“下调是否安全”的排序能力；不是逐轮动作准确率。

为取得同状态标签，本实验在每个实际动作桶内执行 L8/L16/L32 三个分支，chosen 分支本身就是 canonical，不再额外二次 replay。`report.md` 明确排除反事实观察开销，只评价若部署冻结策略时应选择的 block token 数。策略有正结果后，仍需另做按 L8/L16/L32 分桶的连续 serving 实验，才能测真实吞吐、延迟、排队和 padding 收益。

## 11. 各阶段执行顺序与进度条对照

`--stage all` 不会为每个候选策略重新启动一次九数据集推理。它先顺序采集一次动态探索 trace，再对现有 trace 做纯 CPU 候选计算（同时在 GPU 上正常运行九集 baseline 复现），最后终止 baseline，并分别用冻结的 S8、S16 策略各重跑一次九数据集验证。

下表中的“外层进度”由总入口显示；“内层进度”表示当前阶段还能看到的细分进度。例如总体 `2/4` 表示探索和搜索都已完成，即将进入 S8 验证，不表示候选搜索只完成一半。

|总体|阶段|是否GPU|遍历单位|内层进度|
|:---:|:---:|:---:|:---:|:---:|
|1/4|探索采集|是|9数据集|`explore 数据集 i/9`|
|2/4|离线搜索|守护驻留|候选与GBDT|S8候选、S8 GBDT、S16候选、S16 GBDT|
|3/4|S8验证|是|9数据集|`validate_s8 数据集 i/9`|
|4/4|S16验证|是|9数据集|`validate_s16 数据集 i/9`|

探索和验证始终按命令中的数据集顺序运行，同时强制把 MMLU 放在最后。默认顺序是 `gsm8k → human-eval → mbpp → math-500 → aime25 → gpqa → ifeval → livecodebench-cpp → mmlu`。每完成一个数据集，终端进度条和结果目录中的 `report.md` 都会更新。

数据集与总体阶段使用固定宽度进度条，典型输出如下。`3/9` 指已经完整完成三个数据集；“正在运行 math-500”期间分子仍保持 3，直到该数据集的 pipeline 和 trace 检查均成功才变为 4。

```text
[进度][explore 数据集] |##########--------------------| 3/9 ( 33%) 正在运行 math-500
[进度][总体阶段] |###############---------------| 2/4 ( 50%) 搜索完成，准备 S8 验证
```

离线非数据集遍历有四条独立动态进度条：

|进度条|总数|含义|
|:---:|:---:|:---:|
|S8候选|13824|默认L8策略的标签规格、特征组、模型族及双阈值组合|
|S8 GBDT|12|S8所选标签下三特征组、两种树深和两个标签的信号上界|
|S16候选|4608|默认L16策略的标签规格、特征组、模型族及双阈值组合|
|S16 GBDT|12|S16所选标签下三特征组、两种树深和两个标签的信号上界|

交互终端中，候选和 GBDT 进度条会在同一行动态刷新，并显示 `已完成/总数`、百分比和预计剩余时间，例如：

```text
[S8 候选] |###############---------------| 6912/13824 ( 50.0%) ETA 00:18
```

若输出被重定向到文件、`tee` 的非终端端或作业调度日志，程序不会写入回车覆盖控制符，而是每跨过约 1% 写一条完整进度行；因此可以用 `tail -f` 直接查看。GBDT 总数只有 12，每完成一个拟合都会写一行。NeMo-Skills 在单个数据集内部原有的 sample/generation 进度仍会照常显示，与这里的外层数据集进度互不替代。

进度条只反映执行进度，不参与策略评分，也不会改变九数据集等权、MMLU 子采样、探索动作、候选数或最终最优策略。

## 12. CPU 搜索期间并行运行九集 baseline

离线候选和 GBDT 计算仍然全部在 CPU 上完成，但 `search` 不再让实验 GPU 空闲。进入搜索前，主脚本直接调用未修改的 `observations/eval_sglang.sh` baseline 入口，正常运行以下九个非 AIME24 数据集：

```text
gsm8k,human-eval,mbpp,math-500,aime25,gpqa,ifeval,livecodebench-cpp,mmlu
```

baseline 沿用当前 run 的模型、模式、GPU、TP、batch、client concurrency、生成 token 上限、temperature、top-p、context、dtype、block size、LoRA、`mem_fraction` 和 `gpu_memory_reserve_gb`。它不使用动态实验的 `max_samples` 或 MMLU 采样上限，即按 baseline 的完整数据集复现方式持续运行；通常会在九集完成前被 CPU 搜索完成事件终止。

守护启动顺序如下：

```text
探索最后一个数据集结束
  -> baseline pipeline 加载模型、KV pool 和显存预留
  -> 启动九集 SGLang+NeMoSkills 正常复现
  -> 确认已进入真实 benchmark evaluation
  -> baseline GPU 推理与 CPU 的 S8/S16 候选搜索、GBDT 并行
  -> CPU 搜索成功或失败后终止整个 baseline 专属进程组
  -> 释放显存并启动后续 S8 验证
```

baseline 的输出放在当前 run 的 `runtime/search_guardian_baseline/baseline_eval/`。如果 CPU 搜索较快，baseline 可能在某个数据集或样本中途被终止，所以这里的产物允许不完整；它不写入动态探索/验证 trace，也不参与动态策略检索或最终指标。`--keep-server`仍会传给 baseline，但只用于九集意外先完成时继续保持显存；正常路径是在 baseline 仍执行真实推理时由 CPU 搜索完成事件将其终止。

守护状态会实时追加到 `report.md` 的进度表，状态含义如下：

|状态|含义|
|:---:|:---:|
|starting|baseline 正在加载模型、KV pool、显存预留并准备九集测评|
|active|baseline 已进入真实 benchmark evaluation，并与 CPU 搜索并行|
|stopped|CPU 搜索结束，baseline 测评进程组已终止并交还显存|
|failed|baseline 未能进入真实测评，拒绝开始 CPU 搜索|

baseline 主日志首次为 `runtime/search_guardian_baseline/guardian.log`；同一 run 再次搜索时为了保留旧诊断日志，会写入带新时间戳的 `guardian_YYYYMMDD_HHMMSS.log`。其中的 `nemo-run Remaining generations` 是 guardian baseline 的 GPU 生成进度，不是 CPU 策略候选进度。启动超过 30 秒时，终端会每 30 秒提示仍在加载并给出本次日志路径；入口使用系统 `grep` 的固定字符串检测确认进入真实测评，最长等待 1800 秒，随后才启动 CPU 候选进度条。

正常搜索结束、搜索报错、脚本收到中断或主脚本异常退出时都会终止 baseline。先向专属进程组发送 TERM 并最多等待 30 秒；只有该组未退出时才发送 KILL，不会按模型名或 GPU 编号模糊清理其他实验进程。被 CPU 搜索主动终止是预期行为，不代表 baseline 或动态搜索失败。

分阶段运行 `--stage search --run-dir ...` 时，守护默认从该 run 的 `settings.json` 恢复原探索实验的模型和显存设置。GPU 也默认恢复原设置；如果搜索命令显式传入 `--gpu-devices`，则以本次明确指定的 GPU 为准。因而离线搜索阶段现在同样要求目标 GPU 能成功加载 baseline，不能再将它视为无 GPU 阶段。

## 13. 从现有 `20260901_032420` 继续及从头重跑

### 13.1 当前目录实际状态

`dynamic_block_history_20260901_032420` 已完成九个非 AIME24 数据集的唯一一次动态探索，共 699838 轮；`eval_runs/explore_s8` 的名称不代表只观察 L8，每轮 trace 都包含 L8/L16/L32 三分支。九集搜索及 S8/S16 策略已经生成。S8 冻结验证已完成 `gsm8k、human-eval、mbpp、math-500、aime25`，旧实现随后在 `gpqa` 的动态 shadow 分支触发 CUDA 异步索引越界；S16 尚未开始。

旧版采集存在 23274 轮 canonical replay 异常和 228 轮跨 block 公共前缀异常；两者去重后排除 23414 轮，占 3.35%，低于默认 5% 保护上限。已完成的搜索明确排除了这些歧义轮，并使用其余约 96.65% 轮次执行九集等权搜索；本次续跑会直接复用该搜索结果。最终结论以新版同动作桶 chosen-last 逻辑跑出的 S8/S16 真实冻结验证为准。

### 13.2 当前目录一条命令完成所有剩余阶段

在项目根目录执行下面这一条命令即可复用已有搜索和五个已完成的 S8 数据集，从 `gpqa` 重新开始，随后完成剩余 S8 数据集、S8 汇总和全部 S16 验证。batch 4、client concurrency 4 等未写参数仍从旧 `settings.json` 恢复；这里显式使用 `auto`，让每个未完成数据集的新 server 启动前重新选择至少有 48 GiB 空闲显存的卡，避免继续绑定到已经被外部任务抢占的 GPU 2/3。

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420 --gpu-devices auto
```

续跑是幂等的：搜索结果、两类验证汇总和逐数据集 completed trace 会分别检查并跳过；某数据集只有失败/中断的部分 trace 时，会从头启动一个独立 attempt trace，不会把部分历史续接到新 request，也不会覆盖已经完成的数据集。默认每个未完成数据集最多尝试 3 次；公共 pipeline 的 benchmark 错误在本实验内会强制转成非零退出，即使已经写出部分 trace 也不能误报 completed。失败 trace 保留在 `runtime/attempt_traces/<阶段>/<数据集>/`，失败尝试的 `.eval_*_work_*` 目录保留 server/proxy/runtime 日志；只有成功尝试才原子替换 `traces/<阶段>/<数据集>.jsonl`。

真正完成时应同时存在 `search/search_results.json`、`search/policy_s8.json`、`search/policy_s16.json`、`search/validation_s8.json`、`search/validation_s16.json`，并在 `report.md` 看到搜索结果以及 S8/S16 两节的九集真实验证表。guardian 日志里 baseline 自己跑到多少不构成完成条件。

### 13.3 以后从头完整运行

下面命令会新建时间戳目录，并按“九集探索→CPU 搜索与 GPU guardian 并行→S8 九集验证→S16 九集验证”完整执行；MMLU 仍固定最后且最多 2000 个样本。

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage all --gpu-devices auto --batch-size 4 --client-concurrency 4 --mmlu-max-samples 2000
```

从头运行产生的 schema v2 trace 使用同动作桶三分支、chosen-last canonical；如果任务中断，可直接对该时间戳目录执行上一节形式的 `--stage remaining --run-dir ...`，已完成阶段不会被清空或重复执行。

### 13.4 GPQA 连续失败的最终诊断与修复

`20260902_151333` 首次失败时只保留了客户端 502，证据不足以定位上游；加入失败 runtime 保留和自动重试后，`20260902_202245`、`20260902_202721`、`20260902_203243` 三次全新 server 尝试都在真实生成一段时间后复现。这证明它不是一次性的端口、显存抢占或 NeMo-Skills 汇总问题。

三份 `sglang_server.log` 的首个真实错误均为 CUDA `ScatterGatherKernel` 索引越界，随后 SGLang scheduler 退出，timing proxy 才向 NeMo-Skills 返回 HTTP 502。终端中的“没有 metrics”、LiteLLM `BadGatewayError` 和 server `Killed` 都是这次上游 CUDA 失败的连锁症状，不是根因。错误异步延迟到下一次 `_view` 中读取 `seq_lens` 时才被 Python 看见，因此旧堆栈指向的 `seq_lens.sum().item()` 也不是实际产生越界的运算。

根因类别是本观察实现连续执行 L8/L16/L32 反事实分支时，对同一组物理 KV slot 和 FlashInfer batch 元数据的快速重绑定缺少明确的分支完成边界。在 batch 4、client concurrency 4 的持续负载下，前一分支的 CUDA 工作尚未完全结束就规划下一可变长度分支，最终可在后端表现为异步索引越界。为验证这一点，首先使用同一冻结 S8 策略、GPQA 前 16 个样本、batch 4、client concurrency 4、8192 token 进行了独立分段同步诊断：16/16 request 全部完成，共产生 3903 轮三分支 trace，没有 CUDA assert。随后关闭额外诊断同步开关，只保留正式代码中的逐分支 correctness barrier 再跑相同 16 样本：同样完成 16/16 request 和 NeMo-Skills metrics，共产生 3235 轮 trace；每轮都齐备 L8/L16/L32，canonical 异常为 0，跨 block 公共前缀异常为 0。两组通过对照加上正式任务三次无屏障复现，支持上述定位和修复。

当前观察模块已采用以下修复和保护；这些改动只由本目录的 `sitecustomize.py` 激活，不修改共享 SGLang，也不影响其他复现或实验：

|保护|作用|
|:---:|:---:|
|分支屏障|每个 L8/L16/L32 shadow 分支结束后执行 CUDA correctness barrier，确认该分支全部完成再重建下一分支视图；本实验不评价 shadow 采集墙钟效率，因此正确性优先|
|形状检查|每次 `_view` 在 GPU 前向前检查 request 索引和物理 batch 的 input、position、KV location、request-pool 等长度关系；布局异常会给出同步 Python 错误|
|尝试隔离|每次 fresh-server 尝试写入 `runtime/attempt_traces/<阶段>/<数据集>/` 的独立文件；只有 pipeline 与 trace 都成功时才原子移动为 canonical trace|
|单写锁|`runtime/runner.lock` 保证同一结果目录同时只有一个主入口写入；若重复启动同一条 `--stage remaining` 命令，后启动者以退出码 89 立即拒绝|

正式目录曾在 20:21:48 和 20:22:21 被两个 `--stage remaining` 入口同时启动，这不是 CUDA 越界唯一解释，但会同时改写 `run_state.json`、canonical trace 和报告。新增单写锁消除了这类结果目录竞争。旧失败留下的 GPQA 部分 canonical 文件没有 `completed` 事件，因而不会被视作可复用数据；下一次成功尝试会原子替换它。此前已经 completed 的五个 S8 数据集、九集搜索结果和冻结策略均保持不变。

继续时仍执行 13.2 的单行命令，不需要删除、移动或手工编辑现有 trace、策略和结果目录。入口会跳过已完成部分，从 GPQA 开始写独立尝试 trace；成功后再继续 IFEval、LiveCodeBench、MMLU 及全部 S16 验证。分支屏障会降低本观察采集速度，但不改变 chosen action、接受长度、历史特征、九集等权搜索或最终信号结论。
