# NLD SGLang + NeMo-Skills 双动作空间接收/计算比策略实验

## 1. 实验目标

本实验只为 S8、S16 两个 serving 档位寻找一套共享历史信号和两套效率最优 λ：

```text
S8：第一轮L8，后续动作属于{8,16,32}
S16：第一轮L16，后续动作属于{16,32}
```

两档共享同一个信号、同一套 V8/V16/V32 条件价值表和同一个决策公式，仅 cold-start、允许动作和 λ 不同：

```text
U(L)=V(L|s)-lambda×L
```

S8 只在 8/16/32 中比较 U，S16 只在 16/32 中比较 U；完全并列时取较小动作。搜索目标不是 accuracy、墙钟速度或单纯接收长度，而是八数据集等权的接收/计算比：

```text
每集效率=该集request等权平均接收长度/该集request等权平均block长度
八集效率=八个每集效率的算术平均
```

S8 报告相对固定 L8 的效率提升，S16 报告相对固定 L16 的效率提升。搜索不会暗中加入“TPF 不下降”等约束，但会完整报告 TPF、接收长度、计算量和截断风险。

## 2. 正式数据协议

正式搜索和冻结验证必须恰好包含：

```text
gsm8k,human-eval,mbpp,math-500,aime25,gpqa,ifeval,livecodebench-cpp
```

AIME24 和 MMLU 被硬性排除，不能通过命令误加入。除非显式使用仅供开发的 `--allow-partial-datasets`，其余八集均使用全部 sample，不允许 `--max-samples`。

权重固定为：

```text
八个数据集等权 -> 数据集内request等权 -> request内有效轮次等权
```

因此 LiveCodeBench-C++ 或 GSM8K 的样本和轮次更多，也不会盖过 AIME25、HumanEval 等小数据集。

## 3. 默认复用的 SGLang trace

默认读取：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore
```

该 trace 来自修复后的 SGLang + NeMo-Skills LinearSpec/LoRA 推理，每轮记录同状态 L8/L16/L32 shadow 接收结果和动态 canonical 历史。本实验读取时显式丢弃其中 MMLU，trace 完整性统计也重新只按八集计算。

旧 schema 中 canonical replay 或三分支公共前缀不一致的歧义轮默认排除；八集总排除率超过 `--max-invalid-row-rate` 时停止，不能静默继续。冻结验证使用当前 schema v2，要求零歧义。

离线 OOF 是 off-policy 同状态反事实评估。由于新策略会改变后续历史，最终必须分别运行 S8、S16 的真实闭环 SGLang 冻结验证。

## 4. 信号和值模型

Tier1 每个候选严格只读取一个标量：

|类别|候选|示例|
|:---:|:---:|:---:|
|接收|`accept_last/ma2/ma4/ma8`|`accept_ma4=7` 表示近4轮平均接收7|
|接收比例|`ratio_last/ma2/ma4/ma8`|按历史实际动态block归一化|
|连续状态|`full_streak/nonfull_streak`|连续整块通过或未通过轮数|
|confidence|`head_conf_last/ma2/ma4/ma8`、拒绝位conf|头部取前7个位置，兼容L8/16/32|
|margin|`head_margin_last/ma2/ma4/ma8`、拒绝位margin|不与confidence拼接|
|entropy|`head_entropy_last/ma2/ma4/ma8`|内部取相反数，使越大统一表示越强|

默认使用 5 折 prompt-hash OOF：同一 prompt 的全部轮次只属于一折，每个 prompt 的价值预测均来自未见过该 prompt 的其余折。

对于每个信号和 block L，拟合 32 个集等权分位 bin 上的单调 survival：

```text
P(A_L>=k|s)，k从1到L
V(L|s)=这些概率之和
```

Tier2 最多允许一个接收类主信号加一个 `full_streak>=1/2/3` 右删失分层。双信号只有同时提高 S8、S16 两档，并使两档较差收益至少改善 `--min-dual-improvement`，才允许替代单信号。不会训练多特征分类器，也没有数据集专属参数。

## 5. 精确 λ 搜索和公共信号选择

程序不依赖粗 λ 网格寻找最优值。固定 V 后，动作只会在以下交点改变：

```text
lambda=(V(L2)-V(L1))/(L2-L1)
```

程序枚举 `[0,--lambda-max]` 内所有交点、交点本身和相邻开区间中点，因此得到当前 value 模型与动作空间内的精确最优 λ 区间。`--diagnostic-lambda-grid` 只控制报告里的人工对照表，不参与替代精确搜索。

对每个公共信号分别求：

```text
G8=Eff(S8动态)/Eff(固定L8)-1
G16=Eff(S16动态)/Eff(固定L16)-1
```

公共信号的全局排序是：

```text
最大化min(G8,G16)
-> 再最大化(G8+G16)/2
-> 再最小化两档regret之和
```

报告会保存所有候选的 λ8、λ16、G8、G16，并导出一个 `policy.json`，其中两档共享信号和值表。

## 6. 代码与隔离边界

新代码全部位于：

```text
observations/sglang_dual_action_efficiency_policy/
```

|文件|作用|
|:---:|:---:|
|`eval_dual_action_efficiency.sh`|时间戳目录、显存守护、搜索、双档验证、重试和续跑总入口|
|`search.py`|八集过滤、OOF、精确λ、公共信号选择和验证汇总|
|`policy_runtime.py`|共享策略在S8/S16动作空间内的纯Python决策|
|`dual_efficiency_algorithm.py`|本实验专属SGLang LinearSpec子类|
|`sitecustomize.py`|仅显式环境变量开启时注册，不修改共享SGLang|
|`gpu_memory_guard.py`|CPU搜索期间预占指定GPU显存|
|`reporting.py`|立即创建settings/report并持续原子更新中文表格|
|`tests/`|动作空间、等权、精确λ、报告和完整搜索测试|

SGLang worker 只有在入口设置 `NLD_DUAL_EFF_ENABLE=1` 时才替换本进程的 LinearSpec 注册。公共 SGLang、`observations/eval_sglang.sh`、PyTorch、此前动态 block 和论文复现实验均不修改。未指定端口时沿用 baseline 自动找空闲端口。

结果根目录为：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/
```

每次新实验建立：

```text
dual_action_efficiency_YYYYMMDD_HHMMSS/
```

目录创建后立即生成 `settings.json`、`settings.md`、`run_state.json` 和 `report.md`，不会等搜索或八集验证结束才写报告。

## 7. 推荐：从现有 trace 连续完成搜索和双档验证

正式单卡、batch size 1 全流程：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --gpu-devices 1 --search-gpu-hold-gb 40 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 20
```

较大 batch 示例：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --gpu-devices 2 --search-gpu-hold-gb 48 --batch-size 4 --client-concurrency 4 --gpu-memory-reserve-gb 0
```

自动选择至少有 52 GiB 空闲且当前计算利用率较低的 GPU：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --gpu-devices auto --auto-gpu-min-free-gb 52 --search-gpu-hold-gb 48 --batch-size 1 --client-concurrency 1
```

多 GPU tensor parallel；显存守护值是每张卡的 GiB：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --gpu-devices 0,1 --tp-size 2 --search-gpu-hold-gb 40 --batch-size 2 --client-concurrency 2
```

## 8. 分阶段、续跑和指定档位

只完成八集离线搜索；搜索期间仍预占 GPU，完成后自动释放：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage search --gpu-devices 2 --search-gpu-hold-gb 48
```

上条命令会打印时间戳目录。随后验证两个档位：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s8,s16
```

只验证 S8：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s8
```

只验证 S16：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s16
```

在冻结策略不变的情况下，人工覆盖两档 λ：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s8,s16 --policy-lambda-s8 0.70 --policy-lambda-s16 0.30
```

中断后幂等完成缺失搜索、数据集和档位；已完成 trace 不会重跑：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS
```

续跑时允许更换执行 GPU 和并发资源，不会改变搜索协议：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --gpu-devices 3 --batch-size 2 --client-concurrency 2
```

只重新渲染报告：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage report --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS
```

## 9. 单/多数据集开发和 smoke test

正式结论必须使用全部八集；以下只用于链路检查。

单数据集、每集最多 2 条、只验证 S8。小样本 smoke 建议使用 GSM8K；HumanEval 的 EvalPlus 评分器会检查完整题集，`--max-samples` 子集会在准确率汇总阶段报 `Missing problems in samples`，即使模型推理和效率 trace 已经成功：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s8 --benchmarks gsm8k:1 --max-samples 2 --allow-partial-datasets --gpu-devices 3
```

多数据集开发验证：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_dual_action_efficiency_policy_results/dual_action_efficiency_YYYYMMDD_HHMMSS --profiles s16 --benchmarks gsm8k:1,math-500:1 --max-samples 2 --allow-partial-datasets --gpu-devices 3
```

纯 CPU 小规模搜索链路；只读取每集前 300 行，不能用于结论：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage search --gpu-devices 3 --search-gpu-hold-gb 0 --allow-partial-datasets --search-max-rows-per-dataset 300
```

不创建目录、不占显存、不启动模型，仅检查参数解析：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --gpu-devices 3 --dry-run
```

## 10. 可选：重新采集 SGLang 探索 trace

本实验默认无需重新采集。如果需要从头获取新的八集 SGLang+NeMo-Skills 三分支探索轨迹，可使用已有且独立的动态 trace 采集入口：

```bash
bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh --stage collect --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1 --allow-partial-search --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 0
```

采集完成后，将打印目录下的 `traces/explore` 传给新搜索：

```bash
bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh --stage all --trace-root /data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_YYYYMMDD_HHMMSS/traces/explore --gpu-devices 2 --search-gpu-hold-gb 48
```

## 11. GPU 显存守护顺序

复用 trace 的 `search/all/remaining` 顺序为：

```text
创建settings/report
-> 在指定GPU启动显存守护并等待ready
-> CPU读取trace、OOF和精确lambda搜索
-> 搜索成功或失败都终止守护
-> wait确认CUDA context退出并记录released
-> 额外等待2秒
-> 才启动S8/S16 SGLang验证
```

|参数|阶段|含义|
|:---:|:---:|:---:|
|`--search-gpu-hold-gb`|CPU搜索|每张指定GPU预占GiB，验证前完全释放|
|`--search-gpu-hold-chunk-gb`|CPU搜索|守护分块申请大小|
|`--gpu-memory-reserve-gb`|SGLang验证|模型/KV存在期间沿用baseline的额外显存预留|
|`--mem-fraction`|SGLang验证|模型和KV pool的静态显存比例|

传 `--search-gpu-hold-gb 0` 可关闭守护，但长时间正式搜索不建议关闭。若指定卡剩余显存不足以安全申请目标值，守护会明确失败，不能静默少占。

## 12. 参数总览

|参数|默认|作用|
|:---:|:---:|:---:|
|`--stage`|必填|`search/validate/remaining/all/report`|
|`--run-dir`|新建时空|续跑或验证已有时间戳目录|
|`--trace-root`|现有动态trace|离线搜索输入|
|`--benchmarks`|正式八集|NeMo-Skills单/多数据集规格|
|`--profiles`|`s8,s16`|需要运行的冻结档位|
|`--policy-lambda-s8/s16`|`auto`|验证时采用搜索值或人工覆盖|
|`--model`|NLD-8B本地权重|SGLang加载的checkpoint路径|
|`--served-model-name`|`nemotron-labs-diffusion-8b`|OpenAI接口与NeMo-Skills请求共用的模型名|
|`--mode`|`linearspec_lora`|`linearspec_lora`启用draft LoRA；也支持`linearspec_base`|
|`--lora-path`|模型目录默认adapter|显式覆盖LinearSpec draft LoRA目录|
|`--lora-mode`|`draft_only`|LoRA只用于draft；其余值须为底层SGLang支持值|
|`--cv-folds`|5|prompt级OOF折数|
|`--signal-bins`|32|单调survival分位bin|
|`--lambda-max`|1|精确λ搜索上界|
|`--diagnostic-lambda-grid`|0到1共13点|报告人工对照，不决定最优λ|
|`--full-gate-grid`|`1,2,3`|受限双信号门槛|
|`--min-dual-improvement`|0.002|两档最差相对效率的最小改善|
|`--min-gain16`|2|S8选L16时显著收益定义|
|`--min-gain32`|4|选L32时显著收益定义|
|`--max-invalid-row-rate`|0.05|旧trace最大排除比例|
|`--split-seed`|20260906|prompt hash OOF划分种子；同prompt不会跨折|
|`--gpu-devices`|0|GPU ID、逗号列表或`auto`|
|`--auto-gpu-min-free-gb`|52|自动选卡要求的最低空闲GiB，再按利用率和空闲量排序|
|`--search-gpu-hold-gb`|48|CPU搜索期间每卡预占GiB|
|`--search-gpu-hold-chunk-gb`|1|守护进程逐块申请显存的chunk大小|
|`--tp-size`|按GPU数推断|SGLang tensor parallel大小|
|`--batch-size`|1|SGLang最大并行request|
|`--client-concurrency`|1|NeMo-Skills客户端并发|
|`--gpu-memory-reserve-gb`|0|模型期附加显存预留|
|`--mem-fraction`|0.55|SGLang静态显存比例|
|`--tokens`|8192|最大生成token|
|`--context-length`|10240|上下文上限|
|`--temperature`|0|同状态贪心反事实必须为0|
|`--top-p`|0.95|与baseline接口对齐；temperature 0时不引入采样|
|`--dtype`|`bfloat16`|模型推理dtype|
|`--port/--proxy-port`|自动|仅需要固定端口时设置|
|`--nemo-skills-data-dir`|baseline默认|数据集与评分依赖缓存目录|
|`--sglang-python`|`nld_sglang`环境|启动SGLang和执行搜索的Python|
|`--eval-python`|同SGLang Python|执行NeMo-Skills的Python，可单独覆盖|
|`--sglang-src`|baseline默认|SGLang源码目录覆盖|
|`--sglang-work-dir`|baseline默认|SGLang运行与缓存目录覆盖|
|`--extra-server-args`|空|追加到底层server的参数字符串；本实验始终禁用CUDA graph|
|`--dataset-max-attempts`|3|每个档位/数据集使用全新server重试次数|
|`--dataset-retry-delay-s`|10|失败重试前等待秒数|
|`--allow-partial-datasets`|关|只供开发的单/多数据集模式|
|`--search-max-rows-per-dataset`|0|0为全部；非0必须配合开发模式|
|`--dry-run`|关|只校验并打印解析结果，不建目录、不占显存、不启模型|

## 13. 进度与结果文件

终端搜索进度分两层：

```text
Tier1共享单信号+精确lambda：24个候选
Tier2接收+full+精确lambda：默认27个候选
```

每完成一个候选即更新：

```text
<run>/search/progress.json
<run>/report.md
```

验证进度由 NeMo-Skills 的 `Remaining generations` 显示 sample 完成数；`report.md` 的实时表区分 `s8/数据集` 和 `s16/数据集`。每个数据集完成后立即重新分析当前已完成集合并追加逐集效率、动作占比、风险、相邻轮 block 转移/振荡和官方 decode-only TPF 表格。

主要文件：

```text
settings.json / settings.md             完整超参数与原始命令
run_state.json / report.md              实时事件与中文报告
search/search_results.json              全候选搜索结果
search/policy.json                      共享信号、价值表、λ8和λ16
search/validation_s8.json               S8冻结汇总
search/validation_s16.json              S16冻结汇总
traces/validate/s8/*.jsonl              S8闭环trace
traces/validate/s16/*.jsonl             S16闭环trace
eval_runs/validate/{s8,s16}/             NeMo-Skills/SGLang原始产物
runtime/gpu_memory_guard/                显存守护ready/released/log
```

报告中所有表格列均居中，变量解释紧邻对应表格。`TPF逻`和官方 decode-only TPF 不含 prompt prefill；由于每轮仍执行三分支 shadow，墙钟吞吐、TTFT、TPOT 和显存峰值不是未来分桶 serving 的性能结论。
