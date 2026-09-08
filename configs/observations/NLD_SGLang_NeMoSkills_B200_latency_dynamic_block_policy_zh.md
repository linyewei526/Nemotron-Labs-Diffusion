# NLD：SGLang + NeMoSkills + B200 延迟感知动态 block 策略实验

## 1. 实验目的

本实验在 `B∈{8,16,32}` 中为每个 request 的每一轮独立选择下一轮 block size，检索目标不是抽象的接收率、`接收/block token` 或“大块是否显著”，而是给定 B200 实测前向时间后的纯 forward 接收吞吐：

`单个 sample 的分数 = 该 sample 全部 decode 轮接收 token 总数 ÷ 对应 B200 两次 forward 时间总和`

每个数据集先对其所有 sample 的分数做算术平均，最后对八个数据集做算术平均。正式实验只包含：

- GSM8K
- HumanEval
- MBPP
- MATH-500
- AIME25
- GPQA
- IFEval
- LiveCodeBench-C++

固定排除 AIME24 和 MMLU。数据集无论 sample 数多少均为 `1/8` 权重；数据集内部每个 sample 等权，长 response 和 decode 轮数多的 sample 不会获得更大权重。

## 2. 与 serving 并发/分桶的关系

本实验分别研究虚拟并发度 `C=2,4,8,16,32,64,128`。这里的 C 有且只有两个作用：

1. 查询 B200 文档中的 `T_C(8)`、`T_C(16)`、`T_C(32)`。
2. 选择该并发度唯一允许不同的超参 `λ_C`。

它不会把请求拆成 `C8/C16/C32`，也不会要求同时并发的 request 采取相同动作。每个 request 都维护自己的动态历史并逐轮独立选择 B8/B16/B32。真实 serving 的分桶等待、桶利用率和调度顺序不属于本轮实验。

如果某数据集 sample 数不能整除 C，最后不足 C 个 sample 构成的虚拟尾组使用实际余数 `r` 对应的 `T_r(B)` 计算理论吞吐。这个尾组修正只影响成本，不耦合各 request 的 block 动作。

## 3. 输入和隔离边界

代码目录：

`observations/sglang_b200_latency_dynamic_block_policy/`

结果根目录：

`/data/home/wly/dLLM/NLD_results/observations/sglang_b200_latency_dynamic_block_policy_results/`

每次新实验创建：

`b200_latency_dynamic_YYYYMMDD_HHMMSS/`

默认复用以下已经完成的 SGLang+NeMoSkills 三分支探索 trace，不重新跑探索：

`/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore`

B200 成本来自：

`configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md`

搜索开始时会严格解析 B8/B16/B32 的 C=1～128 全表，在结果目录写入 `search/b200_latency_costs.json`，同时记录源文件 SHA256。搜索和验证均使用该口径。

新策略通过独立的 `sitecustomize.py` 和环境开关注册，不修改 SGLang 包、基础 `eval_sglang.sh`、旧动态策略或 PyTorch 实现。旧实验的命令和行为不受影响。不同数据集/并发度依次使用独立 server 运行；不指定端口时由基础 pipeline 自行寻找空闲端口。

## 4. 策略形式

### 4.1 信号限制

Tier 1 穷举已有 trace 中的单个历史标量，包括历史接收长度/比例、整块通过连续轮数、头部 confidence、margin、entropy、首拒绝位 confidence/margin 等。候选不会把这些特征拟合成一个多特征模型。

Tier 2 最多使用两个信号，而且严格限制为：

- 一个接收类历史标量；
- 一个全局共享的 `full_streak` 门（默认门值候选 1/2/3）。

Tier 2 的门和价值表在所有并发度间共享。若单/双信号吞吐目标完全相同，选择单信号；双信号只有在 OOF 目标严格更高时才会被采用。

### 4.2 公共价值表与并发度超参

利用 request 级 OOF，在信号分箱内估计三种 block 的期望接收长度：

- `V8(s)`：给定历史信号 s 时，B8 的期望接收长度；
- `V16(s)`：B16 的期望接收长度；
- `V32(s)`：B32 的期望接收长度。

同一套信号、分箱和 `V8/V16/V32` 用于所有 C。每个 C 只单独搜索一个 `λ_C`，动作规则为：

`U_C(B) = V_B(s) - λ_C × T_C(B)`

选择 U 最大的 B；完全并列时选择更小的 B。λ 越大越偏向低延迟 block，λ 越小越偏向高接收长度 block。

每个 request 的第一轮没有历史，不参与上述信号决策，而是按虚拟并发度使用固定映射：

- C=2/4：首轮 B32；
- C=8/16/32：首轮 B16；
- C=64/128：首轮 B8。

离线目标、λ 搜索、block 占比、TPF、理论吞吐和在线冻结验证都计入这次真实首轮动作，并非只在 SGLang 运行时临时替换。

冻结策略 schema 已升级为版本 2。若某个实验目录是在本首轮映射加入前只完成了旧版离线搜索，再执行 `--stage search --run-dir 旧目录` 会识别旧策略并重搜；正式实验更建议直接使用第 6 节命令创建新时间戳目录。旧版策略不会被在线验证静默接受。

λ 不使用手工网格近似：代码枚举预测价值函数之间的全部有效交点及相邻交点的中点，在给定 `--lambda-max` 范围内直接选择八集 sample 宏平均吞吐最高者。

### 4.3 跨并发度选择公共信号

对每个候选信号，先在每个 C 下独立找到最优 λ_C。每个 C 的基准是同一离线 trace 上固定 B8/B16/B32 三者中吞吐最高者。公共信号的主排序指标是：

`七个 C 的“相对各自最佳固定块提升”的等权平均`

次排序指标是七个 C 中最差的相对提升；数值完全相同则优先单信号。这样不同 C 的 token/ms 数值尺度不会让某个并发度支配公共信号选择。

## 5. 离线结果与在线冻结验证

离线搜索完成后，不等待 GPU 验证，`report.md` 会立即给出：

- 最终信号、最佳单信号、最佳双信号；
- 七个 `λ_C` 及稳定区间；
- 每个 C 的固定 B8/B16/B32 分数和最佳固定块；
- 动态策略的 token/ms/request、整批纯 forward token/ms、TPF；
- 相对三个固定 baseline 和最佳固定 baseline 的提升；
- 每个数据集及八集等权平均的 B8/B16/B32 占比和理论吞吐。

随后对七个 C 各自运行八个数据集，共 56 个 SGLang+NeMoSkills 冻结验证任务。对 C 的正式验证会将：

- SGLang `max-running-requests` 设为 C；
- NeMoSkills `client-concurrency` 设为 C；
- 每个 request 的 block 决策仍保持独立。

验证记录真实动态轨迹下的 decode-only TPF、三种 block 占比和基于 B200 成本表计算的理论吞吐，不把当前机器实际 wall-clock TPS 当作 B200 TPS。

### 5.1 旧 strict trace audit 报错已规避

旧统一/双动作实验在生成成功后，把探索 trace 的“跨 block 共同前缀必须一致”审计错误地用于冻结轨迹汇总，少量 shadow 分歧会使完整评估报错。

新实验区分两类审计：

- 离线探索：只有 canonical 回放一致且跨 block 共同前缀一致的行能用于 counterfactual 搜索；排除数量和比例写入报告。
- 在线冻结：实际提交分支只要求 canonical 行完整、决策分支存在且策略回放完全一致。跨 block shadow 前缀差异作为诊断计数，不会否定已经提交的真实轨迹。

冻结验证会逐行用 `policy.json` 和 `history_before_round` 重放动作；任何实际动作与策略不一致都会报错，不会静默生成错误汇总。

## 6. 从头连续执行的正式单行命令

以下一行完成：GPU 显存预占 → 离线搜索 → 立即生成离线结论 → 释放预占 → 七并发度八数据集冻结验证 → 增量/最终报告。

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage all --gpu-devices 1 --search-gpu-hold-gb 48
```

如果模型不在默认路径，显式指定：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage all --gpu-devices 0 --search-gpu-hold-gb 48 --model /path/to/Nemotron-Labs-Diffusion-8B --lora-path /path/to/draft_lora
```

多卡 TP 示例：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage all --gpu-devices 0,1 --tp-size 2 --search-gpu-hold-gb 48
```

`--search-gpu-hold-gb` 是每张可见 GPU 的预占值，不是多卡总量。

## 7. 分阶段和断点恢复命令

只做离线搜索并在完成时查看 baseline 提升：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage search --gpu-devices 0 --search-gpu-hold-gb 48
```

记下终端最后输出的 `Run directory`。从该目录开始运行所有冻结验证：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage validate --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_latency_dynamic_block_policy_results/b200_latency_dynamic_YYYYMMDD_HHMMSS
```

无论中断发生在搜索还是 56 项验证中的任意位置，幂等完成所有缺失阶段：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_latency_dynamic_block_policy_results/b200_latency_dynamic_YYYYMMDD_HHMMSS
```

恢复时更换 GPU，但保留原实验协议：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage remaining --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_latency_dynamic_block_policy_results/b200_latency_dynamic_YYYYMMDD_HHMMSS --gpu-devices 1
```

只重建报告和进度文档，不运行搜索/GPU：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage report --run-dir /data/home/wly/dLLM/NLD_results/observations/sglang_b200_latency_dynamic_block_policy_results/b200_latency_dynamic_YYYYMMDD_HHMMSS
```

## 8. 开发和 smoke 命令

仅检查参数解析，不创建结果目录、不预占 GPU、不启动服务：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage all --allow-partial-datasets --benchmarks gsm8k:1 --concurrencies 2 --max-samples 1 --search-max-rows-per-dataset 200 --gpu-devices 0 --search-gpu-hold-gb 1 --dry-run
```

小规模完整链路会真实预占显存、做离线搜索并运行一个 sample 的 C2 验证：

```bash
bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh --stage all --allow-partial-datasets --benchmarks gsm8k:1 --concurrencies 2 --max-samples 1 --search-max-rows-per-dataset 200 --cv-folds 2 --signal-bins 4 --full-gate-grid 1 --lambda-max 2 --gpu-devices 0 --search-gpu-hold-gb 1
```

开发模式允许单/多数据集和并发度子集；正式结论不要带 `--allow-partial-datasets`、`--max-samples` 或 `--search-max-rows-per-dataset`。

## 9. 参数详解

### 9.1 阶段与输入

- `--stage all`：创建新时间戳目录并连续运行全部阶段。
- `--stage search`：只做成本审计和 CPU 离线检索。
- `--stage validate`：在已有目录执行缺失的在线验证，要求 `--run-dir`。
- `--stage remaining`：断点扫描后完成所有缺失阶段，要求 `--run-dir`。
- `--stage report`：只刷新 `report.md/progress.md/progress.json`。
- `--trace-root`：可复用的 SGLang 三 block 同上下文 trace 根目录。
- `--cost-document`：包含 B8/B16/B32、C=1～128 完整表的 B200 markdown。
- `--concurrencies`：逗号分隔虚拟并发度。正式协议必须是 `2,4,8,16,32,64,128`；开发模式可取它的子集，但不接受其他 C，因为首轮映射只对这七个目标并发度定义。
- `--benchmarks`：格式与基础 pipeline 一致，例如 `gsm8k:1,math-500:1`。正式协议必须是指定八集。
- `--allow-partial-datasets`：只用于开发；允许数据集/C 子集、sample/trace 限制及固定验证并发覆盖。
- `--max-samples`：每个在线验证数据集最多 sample 数；正式实验不允许设置。
- `--search-max-rows-per-dataset`：每个探索 trace 最多读取轮数，0 为全部；非零必须进入开发模式。

### 9.2 搜索

- `--cv-folds`：按 prompt fingerprint 划分的 request 级 OOF 折数，默认 5；同一 request 的轮次不会跨训练/验证折。
- `--signal-bins`：单标量加权分位分箱数，默认 32；每个 block 的 survival/期望接收值在信号轴上做单调校准。
- `--lambda-max`：λ 精确交点枚举上限，默认 16 token/ms，足以覆盖进入固定小块状态的区域。
- `--full-gate-grid`：第二信号允许的全接收连续轮数门候选，默认 `1,2,3`。
- `--max-invalid-row-rate`：旧探索 trace 中可保守排除的异常 counterfactual 行比例上限，默认 5%。
- `--split-seed`：request OOF 的稳定哈希 seed。

### 9.3 GPU 守护与设备

- `--gpu-devices 0`：搜索开始就锁定 GPU 0；多卡用 `0,1`；`auto` 按低利用率、高空闲显存自动选择一张卡。
- `--search-gpu-hold-gb`：CPU 搜索期间每卡实际分配并持有的显存，默认 48 GiB；搜索成功/失败、脚本退出或进入验证前都会释放。
- `--search-gpu-hold-chunk-gb`：显存守护分块分配大小，默认 1 GiB。
- `--auto-gpu-min-free-gb`：自动选择 GPU 的最低空闲显存，默认 52 GiB。
- `--tp-size`：SGLang tensor parallel 大小；省略时由基础 pipeline 根据 GPU 列表推断。
- `--gpu-memory-reserve-gb`：模型加载/运行阶段交给基础 pipeline 的额外保留显存，与 CPU 搜索守护不同。
- `--mem-fraction`：SGLang `mem-fraction-static`，默认 0.55。

### 9.4 验证并发和模型

- `--batch-size auto`：正式默认；对每个 C 自动令 SGLang `max-running-requests=C`。
- `--client-concurrency auto`：正式默认；对每个 C 自动令 NeMoSkills 并发为 C。
- 开发模式可把二者设为固定整数以适配小显存 smoke；这样只验证动作链，不可作为正式 C 吞吐结论。
- `--model`、`--served-model-name`、`--mode`、`--tokens`、`--context-length`、`--temperature`、`--top-p`、`--dtype`、`--lora-path`、`--lora-mode`、`--sglang-src`、`--sglang-work-dir`、`--nemo-skills-data-dir` 与基础 SGLang+NeMoSkills pipeline 含义一致。
- `--temperature` 正式固定为 0；物理 block 固定为 32，以容纳动态 B8/B16/B32 view。
- 动态 view 当前禁用 CUDA Graph；B200 成本表仍作为理论成本模型，不测当前服务器 wall TPS。
- `--port/--proxy-port` 省略时自动规避冲突；显式端口适合调试，不建议与并行实验共用。
- `--dataset-max-attempts` 默认 3。每次重试使用新 server 和独立 attempt trace，只有成功且非空的 trace 才原子提升为正式文件。

## 10. 实时进度和结果文件

终端会依次显示：

1. GPU 守护启动/释放；
2. Tier 1 单信号候选进度条；
3. Tier 2 最多双信号候选进度条；
4. 56 项 `C/数据集` 冻结验证总进度条；
5. 每次失败重试和汇总状态。

结果目录中的以下文件实时、原子更新：

- `settings.md/settings.json`：创建目录时立即写入的完整实验设置。
- `progress.md/progress.json`：阶段、搜索候选、验证任务和最近事件。
- `run_state.json`：可供断点恢复判定的事件流水。
- `runtime/completed_protocol_v2/cC/数据集`：首轮映射版本 2 的成功提交标记；旧协议 trace 不会被恢复流程误当成已完成结果。
- `report.md`：成本表、离线结果、每个已完成 C 的在线结果和解释。
- `search/progress.json`：非数据集搜索进度。
- `search/validation_cC_*_progress.json`：各并发度在线汇总进度；与离线 `search/progress.json` 分离，避免覆盖候选检索进度。
- `search/b200_latency_costs.json`：成本快照及 SHA256。
- `search/policy.json`：冻结信号、价值表和七个 λ_C。
- `search/search_results.json`：完整离线候选、baseline 与选中策略。
- `search/validation_cC.json`：某个 C 当前已完成数据集的增量/正式汇总。
- `traces/validate/cC/数据集.jsonl`：正式 committed 动态轨迹。
- `eval_runs/validate/cC/数据集/`：基础评估 pipeline 产物。

不要在同一 `--run-dir` 上同时启动两个 `validate/remaining` 进程；runner lock 会拒绝第二个写者。不同时间戳目录或其他旧实验可并行运行。

## 11. 指标解释与例子

- `Req吞吐` / `score_token_ms_req`：sample 的总接收 token 除以总 B200 forward ms，再在数据集内和数据集间分层等权。例如某 sample 两轮分别接收 6、10 token，选择 B8、B16；C=16 时分母为 `T_16(8)+T_16(16)`，该 sample 分数为 `16/分母`。
- `首B` / `cold_start_block`：每个 request 第一轮的确定性 block；C2/4 为 B32，C8/16/32 为 B16，C64/128 为 B8。例如 C64 的每个 sample 第一轮都计入一次 B8 接收长度和 `T_64(8)` 成本。
- `批吞吐` / `pure_forward_token_ms_batch`：上述 request 分数乘以该 sample 的有效虚拟并发度；正常组乘 C，尾组乘 r。它是理想满并发纯 forward token/ms，不含分桶等待。
- `TPF`：decode token/physical decode forward。每轮 canonical 路径一次 draft 和一次 verify，因此 trace 口径为 sample 总接收/`2×轮数`；官方汇总同样排除 prefill 和 shadow 额外前向。
- `B8%/B16%/B32%`：先对每个 sample 计算其各 block 轮次占比，再做分层平均。不会让轮数多的 sample 主导。
- `Best分`：固定 B8/B16/B32 中按相同 sample/数据集权重得到的最高分。
- `增Best`：动态分数/Best分−1。例如 4% 表示在该 B200 成本模型和离线 OOF trace 上，相比最佳固定块每 request 每毫秒多接收约 4% token。
- `官方TPF`：从基础 pipeline 的 `sglang_metrics_summary.json` 读取；warmup request 已由更新后的基础 SGLang pipeline 过滤。
- `同态增`：在线动态状态中利用同一轮 shadow 分支得到的固定块诊断，只用于解释，不是独立固定块完整轨迹 baseline。

## 12. 在线结果后的策略修正

在线动态轨迹的 block 历史与离线探索轨迹不同，因此允许在全部 C 验证后做下一阶段校准。当前验证 trace 仍记录三种 block 的同状态 shadow，可以重新估计相同信号下的价值表和 λ_C，但必须遵循：

1. 不增加第三个信号，不改变“所有 C 一套策略、仅 λ_C 不同”的结构。
2. 把校准 trace 和最终无调参验证数据分开。
3. 先报告原始冻结策略在线结果，再报告校准后的独立验证；不能用同一批在线数据既调参又声称泛化提升。

这一步在本轮代码中保留数据和接口基础，但不自动覆盖 `search/policy.json`，避免在线验证过程中偷偷改策略。
