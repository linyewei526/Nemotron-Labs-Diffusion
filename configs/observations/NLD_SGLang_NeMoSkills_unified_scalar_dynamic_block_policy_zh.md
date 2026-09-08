# NLD SGLang + NeMo-Skills 统一标量动态 block size 策略实验

## 1. 实验目标

本实验回答：能否只依赖一个历史标量信号，最多再增加一个仅用于处理“小 block 整块通过导致右删失”的 `full_streak`，用同一套全局参数决定下一轮选择 L8、L16 或 L32。

它严格避免此前两类复杂性：

- 不把 acceptance、confidence、margin、entropy 等多类指标拼成一个特征向量拟合分类器；
- 不再分别训练 S8 的 `worth16/worth32` 和 S16 的 `safe8/worth32`，也没有按数据集配置的策略。

三个动作始终由同一个价值规则决定：

```text
U(L) = V(L | s) - lambda × L
动作 = L8、L16、L32 中 U 最大者；完全并列时选择较小 block
```

其中：

- `s` 是唯一主历史信号；
- `V(L | s)` 是在该信号下使用 block L 的期望接收长度；
- `lambda` 是每计算一个 block token 的惩罚，也是部署时唯一控制大/小 block 倾向的超参；
- `lambda` 越大越保守，越倾向 L8；越小越重视接收潜力，越倾向 L16/L32。

例如 `lambda=0.25` 时，L16 相比 L8 至少需要增加 `0.25×(16-8)=2` 个期望接收 token 才值得升级；L32 相比 L16 至少需要增加 4 个期望接收 token。

## 2. 代码与隔离边界

新代码全部位于：

```text
observations/sglang_unified_scalar_block_policy/
```

主要文件如下。

|文件|作用|
|:---:|:---:|
|`eval_unified_scalar_block_policy.sh`|时间戳目录、GPU 显存守护、离线搜索、九集冻结验证和幂等续跑总入口|
|`search.py`|旧 trace 审计、九集等权 OOF、单调 survival 拟合、单/双信号比较、lambda 扫描及验证汇总|
|`policy_runtime.py`|不依赖 SGLang 的冻结策略动作计算|
|`scalar_policy_algorithm.py`|仅本实验进程使用的 SGLang LinearSpec 子类|
|`sitecustomize.py`|仅在显式环境变量存在时注册新算法，不改共享注册表源码|
|`gpu_memory_guard.py`|CPU 搜索期间按指定 GiB 预占目标 GPU，搜索结束后释放|
|`reporting.py`|立即建立 settings/report，并按阶段原子更新中文表格|
|`tests/`|权重、单调性、lambda、信号数量、导出和报告测试|

新实现复用已经验证过的 `sglang_dynamic_block_history_signal` 同动作桶三分支、chosen-last canonical、逐分支 CUDA barrier 和 schema v2 trace 逻辑，但只通过本实验的 `PYTHONPATH/sitecustomize` 在本次 SGLang worker 内替换动作函数。以下内容均未修改：

- 公共 `sglang_dllm` 源码；
- `observations/eval_sglang.sh`；
- 此前 S8/S16 动态策略实验；
- 任何 PyTorch、method 或论文复现实验入口。

未显式设置 `NLD_UNIFIED_BLOCK_ENABLE=1` 的其他并行进程不会看到新算法。端口未指定时继续由原 SGLang pipeline 自动寻找可用端口。

## 3. 复用的探索 trace

默认直接读取：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore
```

不会重新运行探索 GPU 推理。该目录包含九个非 AIME24 数据集的真实 SGLang + NeMo-Skills 动态轨迹；每轮均有同一状态下的 L8/L16/L32 接收及 confidence/margin/entropy 标签。

当前数据规模：

|项|值|说明|
|:---:|:---:|:---:|
|原始轮|699838|九集全部 shadow round|
|有效轮|676424|进入搜索的轮次|
|排除轮|23414|旧 schema v1 canonical replay 或公共前缀歧义|
|排除率|3.35%|低于默认 5% 保护线|
|提交 request|5583|其中 MMLU pipeline 提交 2000 条|
|MMLU trace request|1989|11 条没有产生可学习的 decode round；不能伪造为训练行|

正式协议仍要求 MMLU 向 NeMo-Skills 提交 2000 条。搜索和验证会报告实际产生 decode trace 的 request 数，默认至少要求 95% 覆盖且不允许超过 2000。

## 4. 九集全局最优与防泄漏协议

正式数据集固定为：

```text
gsm8k,human-eval,mbpp,math-500,aime25,gpqa,ifeval,livecodebench-cpp,mmlu
```

AIME24 会被拒绝；MMLU 自动放在验证遍历的最后，并固定 `--max-samples 2000`。其余数据集默认使用全部样本。

每次拟合、候选选择和指标汇总都使用：

```text
九个数据集等权 -> 数据集内 request 等权 -> request 内有效轮次等权
```

因此 MMLU、GSM8K 或 LiveCodeBench 的样本/轮次数再多，也不会盖过 AIME25 等小数据集。

默认按完整 prompt fingerprint 做 5 折 OOF：同一个 prompt 的全部动态轮只属于一个 fold。每个 prompt 的预测都由没有见过该 prompt 的其余四折拟合，所有有效样本轮都会恰好作为一次 held-out 预测；完成模型选择后，再用九集全部有效历史轮拟合最终部署表。

这里的“全局最优”指：在文档明确列出的单信号和受限双信号候选类、lambda 网格与九集等权 regret 目标内的全局比较，不声称覆盖任意神经网络或任意手工规则。

## 5. 信号候选和受限双信号

Tier1 每个候选严格只读取一项历史标量。

|类别|候选|含义示例|
|:---:|:---:|:---:|
|接收长度|`accept_last/ma2/ma4/ma8`|`accept_ma4=7` 表示近4轮平均接收7|
|接收比例|`ratio_last/ma2/ma4/ma8`|动态 L8/L16/L32 下均按当轮实际 block 归一化|
|连续状态|`full_streak/nonfull_streak`|`full_streak=2` 表示连续两轮整块通过|
|confidence|`head_conf_last/ma2/ma4/ma8`、`reject_conf_last`|只使用 confidence，不和其他类别拼接|
|margin|`head_margin_last/ma2/ma4/ma8`、`reject_margin_last`|只使用 margin，不和接收长度拼接|
|entropy|`head_entropy_last/ma2/ma4/ma8`|内部取相反数，使越大统一代表能力越强|

Tier2 只允许以下结构：

```text
一个 acceptance 类主信号 + full_streak 是否达到全局门槛
```

默认搜索 `full_streak>=1/2/3`。它只用于选择“普通/右删失”两套全局 survival 表，L8、L16、L32 仍共同使用同一个 `lambda` 规则。confidence、margin 和 entropy 不会再与 full_streak 组成双信号。

只有最佳 Tier2 相比最佳 Tier1 的完整 lambda 网格 OOF 平均 regret 至少降低 `--min-dual-improvement`（默认每轮 0.02 token-utility），最终策略才允许使用第二信号；否则强制保留单信号。

## 6. V8/V16/V32 的拟合

对每个候选信号、OOF fold 和 block L，程序拟合：

```text
P(A_L >= k | s)，k 从 1 到 L
V(L | s) = 上述 L 个概率之和
```

实现为集等权加权分位 bin 上的单调 survival 曲线：`s` 越大，接收至少 k 个 token 的估计概率不会下降；同一 bin 内，接收至少 k+1 个的概率不会高于接收至少 k 个。它不是训练神经网络，也没有 dataset id 输入。

默认 32 个 bin。冻结 `policy.json` 保存 survival 表、信号定义、可选 full_streak 门槛和默认 lambda，SGLang worker 不需要 sklearn。

## 7. 默认 lambda 与评价指标

对每个 lambda，程序同时计算同状态 oracle：

```text
Oracle(L) = 实际 A_L - lambda × L
```

并汇报策略相对 oracle 的 regret。默认 lambda 从给定网格中选择：在满足下列约束的候选中，优先获得更大的 L32 计算节省，再以更低 regret 打破并列；若没有候选可行，报告会明确标为“默认不可行”，并输出惩罚最小的诊断点，不会伪称满足目标。

|变量|默认|含义与例子|
|:---:|:---:|:---:|
|`大精`|至少80%|所选 L16 相比 L8 增益至少2，或 L32 相比 L8/L16 中较优者增益至少4的条件比例|
|`大浪`|至多10%|所选较大块的对应增益不超过1 token 的条件比例|
|`损32`|至多2.5|相对同状态 L32 平均少接收 token/轮|
|`Reg均/90/95`|越小越好|策略 utility 与同状态 oracle 的差距|
|`准Or`|越大越好|动作恰好等于同状态 oracle 的概率|
|`截>1/2/4`|越小越好|所选块相对三个块最大接收值损失超过阈值的比例|
|`块均`|越小计算越少|实际动作的平均 block token 数|
|`接均`|越大接收越多|每轮平均提交 token|
|`接/算`|越大越好|平均接收 token / 平均 block token|
|`TPF逻`|接均/2|每个 canonical 解码轮有 draft+verify 两次 forward|
|`8→16`等|相邻动作转移|先在每个 request 内统计，再 request 等权、数据集等权|
|`切换/摆动`|越低越稳定|切换是相邻动作不同；摆动是三轮出现 A→B→A|

冻结验证还从 SGLang `sglang_metrics_summary.json` 读取官方 decode-only `tokens_per_forward_pass`；该口径不含 prompt prefill，也不把为观察反事实而执行的额外 shadow forward 计入 NFE。

由于验证每轮仍运行三个 shadow，墙钟吞吐、TTFT 和 TPOT 会被观察成本污染，不代表未来分桶 serving 的性能。最终策略通过后仍需做只执行所选动作的 L8/L16/L32 分桶 serving benchmark。

## 8. GPU 显存守护

CPU 搜索开始前，入口会先在 `--gpu-devices` 指定的每张 GPU 上启动独立守护进程，并各自分配 `--search-gpu-hold-gb` GiB。默认每卡 48 GiB、按 1 GiB chunk 分配。

顺序严格为：

```text
初始化 settings/report
-> 启动并等待 GPU guard ready
-> 读取 trace 和执行 CPU OOF 搜索
-> 搜索成功或失败均 SIGTERM guard
-> wait 确认 CUDA context 退出并记录 released
-> 额外等待 2 秒
-> 才允许启动 SGLang 冻结验证
```

守护只预占显存，不运行模型或数据集。日志及 ready/released JSON 位于：

```text
<run>/runtime/gpu_memory_guard/
```

两个显存参数不要混淆：

|参数|生效阶段|作用|
|:---:|:---:|:---:|
|`--search-gpu-hold-gb`|纯 CPU 搜索期间|阻止其他任务抢占之后要加载模型的 GPU；验证前完全释放|
|`--gpu-memory-reserve-gb`|每个 SGLang 验证进程期间|沿用 baseline pipeline 的附加显存预留，与模型和 KV pool 同期存在|

例如目标卡当前有 74 GiB 空闲，可先使用 `--search-gpu-hold-gb 48`；若已有进程使空闲低于 48.25 GiB，守护会明确失败，不能静默少占。此时根据现有占用调低数值。传 `0` 可显式关闭，但正式长时间搜索不建议关闭。

`--gpu-devices auto` 默认要求至少 52 GiB 空闲，按“GPU 利用率最低、再选空闲最多”选择一张卡；选定后守护和验证固定使用同一张，不会在 CPU 搜索结束后换卡。

## 9. 推荐的一行全流程命令

从现有 trace 开始，连续完成正式九集搜索和冻结验证：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage all --gpu-devices 2 --search-gpu-hold-gb 48 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 0
```

大 batch 验证示例：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage all --gpu-devices 2 --search-gpu-hold-gb 48 --batch-size 4 --client-concurrency 4 --gpu-memory-reserve-gb 0
```

自动选择 GPU：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage all --gpu-devices auto --auto-gpu-min-free-gb 52 --search-gpu-hold-gb 48 --batch-size 1 --client-concurrency 1
```

多 GPU tensor parallel 示例；`--search-gpu-hold-gb` 是每张卡的数值：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage all --gpu-devices 0,1 --tp-size 2 --search-gpu-hold-gb 40 --batch-size 2 --client-concurrency 2
```

## 10. 分阶段运行、续跑和 lambda 控制

只做正式离线搜索，仍在搜索期间预占 GPU：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage search --gpu-devices 2 --search-gpu-hold-gb 48
```

上条命令结束会打印 `Run directory`。随后对该目录运行默认 lambda 冻结验证：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS
```

在尚未完成验证的搜索目录上指定更保守的 `lambda=0.35`：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS --policy-lambda 0.35
```

中断后幂等完成缺失阶段；已有完整搜索和数据集 trace 会跳过：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS
```

续跑时可显式换到另一张 GPU；搜索模型和数据协议仍从 settings 恢复：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS --gpu-devices 3 --search-gpu-hold-gb 40
```

只重建报告：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage report --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS
```

完整 dry-run，不创建目录、不占 GPU、不加载模型：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage all --gpu-devices 2 --search-gpu-hold-gb 48 --batch-size 4 --client-concurrency 4 --dry-run
```

开发用有限 trace 行搜索；该命令不是正式结论：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage search --gpu-devices 2 --search-gpu-hold-gb 1 --allow-partial-datasets --search-max-rows-per-dataset 1000 --cv-folds 3 --signal-bins 8 --lambda-grid 0,0.25,0.5 --full-gate-grid 1
```

开发用单数据集验证必须在建立搜索目录时就把 benchmark 设为单集，并显式允许部分数据：

```bash
bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh --stage search --benchmarks gsm8k:1 --gpu-devices 2 --search-gpu-hold-gb 1 --allow-partial-datasets --search-max-rows-per-dataset 1000 --max-samples 1 --tokens 64 --context-length 1024
```

## 11. 参数详细说明

### 阶段和数据

- `--stage`：`search/validate/remaining/all/report`，含义见第 9、10 节。
- `--run-dir`：续跑目录；同一目录由 `flock` 保证同一时间只有一个写入者。
- `--trace-root`：离线搜索输入；默认是第 3 节现有九集 trace。
- `--benchmarks`：冻结验证的 NeMo-Skills 数据集列表；正式默认九集，AIME24 被拒绝，MMLU 自动放最后。
- `--max-samples`：限制非 MMLU 数据集，只允许开发检查使用；正式命令留空。
- `--mmlu-max-samples`：正式固定 2000。

### 搜索和策略

- `--cv-folds`：prompt-hash OOF 折数，默认 5。
- `--signal-bins`：单调 survival 分位区间数，默认 32；更多 bin 容量更高但单 bin 样本更少。
- `--lambda-grid`：逗号分隔的非负列表；默认从 0 到 1 的 19 个点。
- `--policy-lambda`：`auto` 或非负数字；只改变冻结验证动作倾向，不重新选择信号。
- `--full-gate-grid`：Tier2 的全局 `full_streak` 门槛列表，默认 `1,2,3`。
- `--min-dual-improvement`：允许第二信号所需的 OOF regret 最小绝对改善，默认 0.02。
- `--min-large-precision`：默认 lambda 的大块显著收益精度下限，默认 0.80。
- `--max-large-waste`：默认 lambda 的大块浪费率上限，默认 0.10。
- `--max-loss-vs-l32`：默认 lambda 相对 L32 的平均接收损失上限，默认 2.5 token/轮。
- `--min-gain16`：L16 相对 L8 的显著增益，默认 2。
- `--min-gain32`：L32 相对 L8/L16 中较优者的显著增益，默认 4。
- `--max-invalid-row-rate`：旧探索 trace 允许显式排除的歧义轮上限，默认 0.05。
- `--split-seed`：fold 哈希种子；不会进入策略特征。
- `--allow-partial-datasets`：仅开发使用；关闭时必须恰好九集。
- `--search-max-rows-per-dataset`：每集最多读取多少轮，默认 0 表示全部；非零必须和 `--allow-partial-datasets` 一起使用。

### GPU、模型和并发

- `--gpu-devices`：GPU ID 或逗号列表；也可为 `auto`。
- `--search-gpu-hold-gb`：CPU 搜索时每张卡预占 GiB，默认 48；0 关闭。
- `--search-gpu-hold-chunk-gb`：显存守护分块大小，默认 1 GiB。
- `--auto-gpu-min-free-gb`：auto 选卡门槛，默认 52 GiB。
- `--tp-size`：tensor parallel 数；省略时由 baseline pipeline 根据 GPU 列表推断。
- `--batch-size`：SGLang `max-running-requests`，默认 1。
- `--client-concurrency`：NeMo-Skills 和 timing proxy 最大并发，默认 1。
- `--gpu-memory-reserve-gb`：模型验证期间 baseline 的额外显存预留，默认 0。
- `--mem-fraction`：SGLang static/KV memory fraction，默认 0.55。
- `--model`：模型 checkpoint，默认 `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`。
- `--served-model-name`：OpenAI API 模型名，默认 `nemotron-labs-diffusion-8b`。
- `--mode`：`linearspec_lora` 或 `linearspec_base`，默认前者。
- `--lora-path`：LoRA 路径；留空沿用 baseline 默认 adapter。
- `--lora-mode`：`draft_only` 或 `both`，默认 `draft_only`；修复后的 SGLang pre-draft hook 会让 LoRA 正确进入 draft。
- `--tokens`：最大生成 token，默认 8192。
- `--context-length`：上下文长度，默认 10240。
- `--temperature`：同状态反事实要求 greedy，必须为 0。
- `--top-p`：默认 0.95；temperature=0 时不引入随机采样。
- `--dtype`：默认 `bfloat16`。
- `--port/--proxy-port`：显式端口；省略时沿用 baseline 的自动避冲突逻辑。
- `--nemo-skills-data-dir`：持久数据目录。
- `--sglang-python/--eval-python`：SGLang 和 NeMo-Skills Python。
- `--sglang-src/--sglang-work-dir`：本地 SGLang 源码和缓存工作目录。
- `--extra-server-args`：附加 server 参数；本观察始终自动添加 `--disable-cuda-graph`，保证变长三分支正确性。
- `--dataset-max-attempts`：单数据集 fresh-server 最大尝试次数，默认 3。
- `--dataset-retry-delay-s`：失败尝试间隔，默认 10 秒。

## 12. 进度条和阶段对应

正式 `all` 流程不是按搜索候选外层重复跑数据集。离线 trace 已存在，因此顺序如下。

|阶段|是否GPU|进度显示|完成产物|
|:---:|:---:|:---:|:---:|
|显存守护|占显存、不计算|`gpu_guard starting/active`|`runtime/gpu_memory_guard/ready_*.json`|
|trace读取|否|每集一行原始/可用/排除|内存中的九集紧凑数组|
|Tier1|否|`Tier1单信号 OOF 0/24...24/24`|每个候选后更新 `search/progress.json` 和 report|
|Tier2|否|默认 `Tier2接收+full OOF 0/27...27/27`|受限双信号排名|
|lambda|否|最终 report 的完整 Pareto 表|`policy.json`、`search_results.json`|
|守护释放|释放GPU|`gpu_guard released`|`released_*.json`|
|冻结验证|是|按文档九集顺序，每集完成更新事件和指标表|`traces/validate/<dataset>.jsonl`|
|最终汇总|否|`validate 九集全局 completed`|`validation.json`、最终 `report.md`|

验证阶段每完成一个数据集就会立即对当前已完成集合生成一次部分 `validation.json/report.md`；九集全部完成后再开启严格九集/MMLU 保护并写 `formal_complete=true`。

## 13. 结果目录

每次新运行创建：

```text
/data/home/wly/dLLM/NLD_results/observations/sglang_unified_scalar_block_policy_results/unified_scalar_block_YYYYMMDD_HHMMSS/
```

结构如下。

```text
settings.json
settings.md
run_state.json
report.md
search/
  progress.json
  policy.json
  search_results.json
  validation.json
traces/validate/<dataset>.jsonl
eval_runs/validate/<dataset>/...
runtime/
  runner.lock
  gpu_memory_guard/
  attempt_traces/validate/<dataset>/...
```

`settings.md` 和 `report.md` 在目录创建时立即存在。失败尝试只保留在 `runtime/attempt_traces`；只有 NeMo-Skills pipeline 成功且 trace 非空才原子提升为正式 `traces/validate` 文件。

## 14. 已完成自检

当前实现已完成：

- 所有 Python 文件 `py_compile`；
- shell `bash -n` 和 dry-run；
- dataset/request/round 三级等权单测；
- survival 信号方向和 token 维单调性单测；
- lambda 增大时动作不增大的单测；
- Tier1/Tier2 全候选合成 OOF 搜索、policy JSON 导出和报告初始化；
- 真实旧 trace 每集前 1000 轮解析及完整候选 smoke；
- GPU 2 上 0.125 GiB 守护 ready/released 生命周期测试；
- GPU 2、GSM8K 1 条、64 token 的真实 SGLang + NeMo-Skills smoke：产生 11 轮 schema v2 trace，canonical replay 和跨 block 公共前缀异常均为 0；trace 逻辑 TPF 3.0 与官方 decode-only TPF 3.0 一致。

未运行九集正式搜索和冻结验证；应使用第 9 节单行命令启动。
