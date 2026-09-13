# SGLang + NeMoSkills：基于真实 verifier 历史与 B200 延迟的动态 block size 实验

> 代码入口：`observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh`
>
> 默认结果根目录：`/data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/`

## 1. 实验目标

本实验为 NLD-8B 和 NLD-14B 从头采集新的 SGLang+NeMoSkills trace。策略只能读取已经完成轮次的 causal verify 信息，不读取 draft confidence、draft margin 或 draft entropy。

对并发度 `C ∈ {2,4,8,16,32,64,128}` 和候选块长 `B ∈ {8,16,32}`，`T_C(B)`是外部 B200 sweep 给出的满 C 桶 draft+causal-verify forward 时间。实验不模拟分桶等待、桶占用率或调度顺序；每个 request 独立决策，并始终按满名义并发度 C 的延迟计费。

三类策略使用同一份 trace、相同 OOF 划分和相同等权目标进行离线比较：

1. `local_ratio`：估计 `V_B(s)`，选择 `V_B(s)/T_C(B)`最大的 B，不使用 λ。
2. `direct_rank`：不要求 V 的绝对值准确，直接在信号区间内最小化相对同一步 B8/B16/B32 oracle 的条件 regret。
3. `global_fractional`：选择 `V_B(s)-ρ_C×T_C(B)`最大的 B；`ρ_C`由八集等权目标自动求解，不是人为控制大块倾向的参数。

三类策略各保存一份冻结文件，但完整真实 SGLang 验证默认只运行离线全局赢家。

## 2. 数据集、权重和固定基线

正式实验只允许以下八个数据集，全部 sample 参与：

```text
GSM8K、HumanEval、MBPP、MATH-500、AIME25、GPQA、IFEval、LiveCodeBench-C++
```

明确排除：

```text
AIME24、MMLU
```

聚合顺序为：

```text
每轮结果 -> sample内总接收/总B200时间 -> 数据集内sample平均 -> 八数据集等权平均
```

七个并发度在全局策略选择时也等权。不能让 sample 较多的数据集或 decode 轮次较多的 response 获得更高全局权重。

指定固定基线和无历史首轮 block size 为：

|C|固定基线|首轮B|
|:---:|:---:|:---:|
|2/4|32|32|
|8/16/32|16|16|
|64/128|8|8|

注意：本实验遵循满桶抽象，每个 sample 都使用 `T_C(B)`；不因数据集最后不足 C 个 sample 而改用 `T_r(B)`。

## 3. 新 trace 和合法信号

旧8B动态块 trace 没有真正的 verifier confidence/margin，因此本实验不会复用旧 trace。8B和14B都必须从 `collect` 阶段重新采集。

每一轮从同一个 committed prefix、相同 request composition 真实运行 B8/B16/B32 三个 shadow 分支。实际选中分支最后执行并成为 canonical 状态，trace 记录三种 counterfactual 接收长度。

新增 verifier 字段直接来自 causal verify logits，并与被检查的 draft token 对齐：

- verifier top-1 confidence；
- verifier top-1/top-2 margin；
- verifier entropy；
- verifier 对 draft token 分配的概率；
- 首个拒绝位置的 verifier top-1 与 draft token 概率差；
- 整块通过时，verify 最后产生并供下一轮使用的 bonus token 的 confidence、margin 和 entropy；
- 验证通过数量、比例、整块通过状态及连续状态。

策略最多使用两个历史信号：

- 主信号是接收长度/比例/full streak或真正 verifier confidence/margin/entropy的1/2/4/8轮统计之一；
- 可选第二信号固定为“上一轮 block size + 是否整块通过”，用来区分例如“B8接收8且full”和“B16接收8但partial”这两种不同的截断含义。

搜索代码的信号注册表没有任何 draft-logit 字段。trace 中保留的 draft position 数据仅用于分支一致性审计，不能成为策略输入。

接收长度/比例的方向明确为“越大通常表示能力越强”。verifier confidence、margin、entropy和首拒位置统计对下一轮能力的方向不做先验假定：搜索会对同一个原始信号同时检查正向、反向单调映射，这仍然只算一个在线信号，并没有混合第三种特征。

## 4. 8B从头完整运行

下面一行依次完成新trace采集、CPU离线搜索、GPU显存守护释放、唯一赢家的八集×七C真实验证：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage all --model-size 8b --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1 --concurrencies 2,4,8,16,32,64,128 --cost-document configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md --gpu-devices 0 --tp-size 1 --trace-batch-size 1 --trace-client-concurrency 1 --batch-size 1 --client-concurrency 1 --search-gpu-hold-gb 70 --search-gpu-hold-chunk-gb 1 --gpu-memory-reserve-gb 0 --mem-fraction 0.7 --tokens 8192 --context-length 10240 --temperature 0 --top-p 0.95 --cv-folds 5 --signal-bins 24 --rho-grid-size 33 --report-top 20 --dataset-max-attempts 3 --dataset-retry-delay-s 10 --validation-trace-retention delete-after-analysis
```

本协议固定使用`--batch-size 1 --client-concurrency 1`采集动态轨迹。C只是冻结策略选择和B200延迟表`T_C(B)`的索引，不是当前A100上的物理并发度；每个request独立改变B，并按满名义C的外部延迟计费。trace探索阶段同样由`--trace-batch-size 1 --trace-client-concurrency 1`逐request采集。

## 5. 14B从头完整运行

14B默认模型是`/data1/linyewei/models/Nemotron-Labs-Diffusion-14B`，默认LoRA是模型目录下的`linear_spec_lora`，默认单GPU、TP=1。

14B使用已经完成并通过审计的`configs/NLD-14B_B200_forward_sweep.md`作为B200 C1～128、B8/B16/B32延迟表，不能使用8B延迟得出正式14B结论：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage all --model-size 14b --benchmarks human-eval:1,gsm8k:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1 --concurrencies 2,4,8,16,32,64,128 --cost-document configs/NLD-14B_B200_forward_sweep.md --gpu-devices 1 --tp-size 1 --trace-batch-size 1 --trace-client-concurrency 1 --batch-size 1 --client-concurrency 1 --search-gpu-hold-gb 48 --search-gpu-hold-chunk-gb 1 --gpu-memory-reserve-gb 0 --mem-fraction 0.55 --tokens 8192 --context-length 10240 --temperature 0 --top-p 0.95 --cv-folds 5 --signal-bins 24 --rho-grid-size 33 --report-top 20 --dataset-max-attempts 3 --dataset-retry-delay-s 10 --validation-trace-retention delete-after-analysis
```

## 6. 分阶段与断点恢复

只采集新trace：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage collect --model-size 8b --gpu-devices 1 --tp-size 1 --trace-batch-size 1 --trace-client-concurrency 1 --gpu-memory-reserve-gb 0 --mem-fraction 0.55 --tokens 8192 --context-length 10240
```

在已有run目录上执行或恢复CPU搜索：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --search-gpu-hold-gb 48
```

只验证离线赢家：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --mem-fraction 0.70 --policy-family winner --validation-trace-retention delete-after-analysis
```

从任意中断位置自动补齐trace、搜索和赢家验证：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --policy-family winner --validation-trace-retention delete-after-analysis
```

重建实时报告，不运行模型或搜索：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage report --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS
```

采集阶段的断点完成条件仍是非空canonical trace和对应completed事件同时存在。低存储验证阶段则以通过语义校验的单数据集紧凑结果作为断点；原始验证trace删除后再次执行不会重跑该“数据集×C”。失败尝试保存在`runtime/attempt_traces/`，不会追加到或覆盖已经提交的canonical trace。同一个run使用非阻塞文件锁，避免两个入口同时写同一结果目录。

当前8B和14B run正在单独恢复采集。必须等待对应的`--stage collect`进程退出后，才能对同一run执行下面命令；文件锁会拒绝并发写入。以下两行分别从当前8B/14B目录补齐采集、搜索和低存储赢家验证：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_20260912_164023 --gpu-devices 0 --validation-trace-retention delete-after-analysis
```

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_14b_20260912_173452 --gpu-devices 1 --validation-trace-retention delete-after-analysis
```

`--validation-trace-retention delete-after-analysis`会同步写回该run的`settings.json/settings.md`。因此命令中只需指定一次，后续同一run的断点恢复会继续采用低存储模式。

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage collect --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_20260912_164023 --gpu-devices 0
```

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage collect --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_14b_20260912_173452 --gpu-devices 1
```

### 6.1 当前8B/14B：采集完成后手动搜索、再手动验证

如果希望明确分开CPU离线搜索和GPU全量验证，不要使用`--stage remaining`。必须先等待对应run的八集`collect`全部完成，然后分别执行本节命令。

8B纯CPU离线搜索，不启动GPU显存守护：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_20260912_164023 --search-gpu-hold-gb 0
```

14B纯CPU离线搜索，不启动GPU显存守护：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage search --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_14b_20260912_173452 --search-gpu-hold-gb 0
```

`--stage search`完成后会在同一run的`search/`中保存三类冻结策略、唯一winner和离线结果，并自动刷新该run的`report.md`与`progress.md`；不需要额外执行`--stage report`。`--search-gpu-hold-gb 0`表示搜索期间不创建CUDA显存守护进程，也不占用指定GPU。

离线搜索完成后，8B在GPU0执行唯一winner的八数据集×七C低存储验证：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_20260912_164023 --gpu-devices 0 --port 30000 --proxy-port 31000 --batch-size 1 --client-concurrency 1 --mem-fraction 0.70 --gpu-memory-reserve-gb 8 --policy-family winner --validation-trace-retention delete-after-analysis
```

14B在GPU1执行唯一winner的八数据集×七C低存储验证；端口与8B显式隔离，因此两条验证命令允许并行运行：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_14b_20260912_173452 --gpu-devices 1 --port 30001 --proxy-port 31001 --batch-size 1 --client-concurrency 1 --mem-fraction 0.55 --gpu-memory-reserve-gb 8 --policy-family winner --validation-trace-retention delete-after-analysis
```

验证命令显式设置8B的`--mem-fraction 0.70`和14B的`--mem-fraction 0.55`；该命令行参数会覆盖run目录原先保存的值，便于验证启动时重新分配模型权重与KV静态预算。上面以`--gpu-memory-reserve-gb 8`作为可直接运行的保守示例：每个真实验证任务启动时额外占用8GiB显存，并在该任务退出时释放；若启动前GPU剩余显存不足，应减小该值，设为`0`则禁用。它不会改变B200虚拟并发度C的成本查表逻辑。验证仍按“数据集外层、C内层”执行，每个request物理并发度为1；每完成一个“数据集×C”，程序就把统计写入`search/validation_parts/`、更新累计验证JSON和`report.md`，随后删除该组原始验证trace。

验证中断后，原样重新执行对应的单行`--stage validate`命令即可续跑。已经生成并通过校验的紧凑结果会被复用，不会重新推理；尚未完成的组合继续执行。搜索、策略、验证紧凑结果、进度和最终报告始终落在各自原有时间戳run目录中。

## 7. 后续手动验证另外两类策略

搜索完成后会保留：

```text
search/policy_local_ratio.json
search/policy_direct_rank.json
search/policy_global_fractional.json
search/policy_winner.json
```

默认全流程只验证`policy_winner.json`。如之后需要单独全量启动另一个策略，无需重新采集trace或重新搜索：

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --mem-fraction 0.70 --policy-family local_ratio --validation-trace-retention delete-after-analysis
```

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --mem-fraction 0.70 --policy-family direct_rank --validation-trace-retention delete-after-analysis
```

```bash
bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_verify_efficiency_policy_results/b200_verify_efficiency_8b_YYYYMMDD_HHMMSS --gpu-devices 1 --mem-fraction 0.70 --policy-family global_fractional --validation-trace-retention delete-after-analysis
```

各策略使用独立的`traces/validate/<family>/`、`search/validation_parts/<family>/`和`eval_runs/validate/<family>/`目录，不会覆盖赢家结果。

## 8. 真实验证顺序

trace采集和在线验证都严格保留`--benchmarks`中数据集的输入顺序，不再按脚本内置列表重排。在线验证以该数据集顺序为外层、并发度为内层。例如第4节命令把`human-eval:1`放在最前，因此执行顺序为：

```text
HumanEval: C2 -> C4 -> C8 -> C16 -> C32 -> C64 -> C128
GSM8K: C2 -> C4 -> C8 -> C16 -> C32 -> C64 -> C128
MBPP: C2 -> C4 -> C8 -> C16 -> C32 -> C64 -> C128
...
LiveCodeBench-C++: C2 -> C4 -> C8 -> C16 -> C32 -> C64 -> C128
```

每完成一个“数据集×C”组合就重新生成对应C的部分汇总并更新`report.md`。因此跑完首个传入的数据集后，即可查看该数据集七种C的完整趋势。

已有run目录会继续采用其`settings.json`保存的数据集顺序。如需调整旧run后续尚未执行项目的顺序，应在断点恢复命令中显式再次传入`--benchmarks`；入口会采用新顺序并同步更新`settings.json`和`settings.md`。已经完成的项目仍会按断点状态复用，不会重复运行。

### 8.1 验证阶段低存储模式

推荐正式验证使用`--validation-trace-retention delete-after-analysis`。它只作用于在线冻结验证，不删除`traces/explore/`中的探索trace，因为离线搜索和之后重新检索仍需要探索数据。

每个“数据集×C”的执行顺序是：先把canonical原始验证JSONL写完整，再单独读取该文件计算动态策略、固定B8/B16/B32、TPF、块占比、B200理论token/ms与相对基线收益；随后原子写入并校验：

```text
search/validation_parts/<family>/c<C>/<dataset>.json
```

该紧凑文件通过模型规格、策略族、并发度、数据集、policy replay和必需统计字段校验后，程序才删除：

```text
traces/validate/<family>/c<C>/<dataset>.jsonl
```

同时会用当前已有的单数据集紧凑文件重建累计结果`search/validation_<family>_c<C>.json`，所以`report.md`仍会实时显示每集结果和当前已完成数据集的等权平均。八集全部完成后再次执行正式八集等权合并。删除前分析失败、紧凑文件不完整或校验失败时，原始JSONL会保留以便排查；如果进程在canonical trace提交后、紧凑分析前中断，续跑会直接分析已有trace，不会重跑GPU推理。

默认值`keep`完整保留旧行为和全部验证trace。已有run若未记录这个新参数，也会按`keep`处理，除非续跑命令显式传入`delete-after-analysis`。

## 9. GPU显存守护

trace采集完成后，CPU搜索本身不使用GPU。入口会在搜索开始前在`--gpu-devices`指定的每张卡上实际分配`--search-gpu-hold-gb`GiB CUDA显存，防止长时间CPU搜索期间其他任务占走后续验证GPU。

守护状态和日志位于：

```text
runtime/gpu_memory_guard/
```

行为为：

1. 等待显存实际分配完成后才开始CPU搜索；
2. 搜索失败或入口异常退出时由trap兜底释放；
3. 搜索成功后、启动第一项真实验证前主动终止守护；
4. 守护完全释放并额外等待后才加载SGLang模型；
5. `--search-gpu-hold-gb 0`可明确禁用。

`--gpu-memory-reserve-gb`含义不同：它是SGLang加载模型时希望保留的空闲显存，不是CPU搜索守护量。

## 10. 进度和结果目录

新run建立后立即生成：

```text
settings.json
settings.md
report.md
progress.md
run_state.json
```

终端和`progress.md`都会显示三个总阶段：

```text
1/3 新verifier trace采集
2/3 三策略族离线搜索
3/3 唯一赢家真实验证
```

CPU检索进度条按“主信号×是否启用截断状态×三种策略族”计数；在线进度条按“八数据集×七并发度×唯一赢家”计数。低存储模式下，在线进度以通过校验的`validation_parts`计数，不依赖已经删除的原始trace。

`report.md`从实验初始化起就是可读模板，并在每次事件、搜索候选推进、数据集×C完成后更新。表格包括：

- 三类策略各自最优信号、是否使用截断状态、七C平均/最差增益；
- 每个C的ρ、局部oracle一致率、regret和B8/B16/B32占比；
- 离线相对指定固定基线的token/ms提升和吞吐等效省时；
- 在线每个数据集、每个C的decode TPF、token/ms/request和整批纯forward吞吐；
- 当前已完成数据集的临时等权平均及最终八集等权平均。

## 11. 主要参数解释

|参数|含义|
|:---:|:---:|
|`--model-size`|选择8B/14B默认权重、LoRA、服务名和结果目录标签|
|`--cost-document`|B200满C桶的B8/B16/B32 draft+verify时间；14B必须使用14B表|
|`--trace-batch-size`|新探索trace采集时SGLang最大并发，不影响B200虚拟C计费|
|`--batch-size`|当前A100上的物理SGLang max-running-requests；本协议固定为1，与虚拟C无关|
|`--client-concurrency`|当前A100上的物理NeMoSkills请求并发；本协议固定为1，与虚拟C无关|
|`--signal-bins`|条件期望和regret表的主信号分箱数|
|`--rho-grid-size`|全局分式对照自动ρ搜索的确定性补充网格数|
|`--cv-folds`|按prompt fingerprint固定划分的request级OOF折数|
|`--policy-family`|默认winner；也可显式验证另外两类冻结策略|
|`--validation-trace-retention`|`keep`保留验证原始JSONL；`delete-after-analysis`在每个数据集×C的紧凑统计通过校验后立即删除原始验证trace，探索trace不受影响|
|`--search-gpu-hold-gb`|CPU搜索期间每张指定GPU实际预占的GiB数|
|`--gpu-memory-reserve-gb`|SGLang模型运行时额外保留的空闲GiB数|
|`--allow-partial-datasets`|仅供smoke；允许减少数据集/C/sample/搜索行|
|`--max-rows-per-dataset`|仅供CPU smoke的每集trace行数上限；正式必须为0|
|`--max-invalid-row-rate`|counterfactual完整性异常行的全局排除率上限，默认5%；超限直接停止搜索|

正式结论以唯一赢家的真实动态轨迹八集等权结果为准。离线oracle、局部一致率和regret用于解释信号质量，不替代真实在线验证。
