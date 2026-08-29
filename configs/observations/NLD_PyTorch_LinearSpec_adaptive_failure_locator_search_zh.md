# NLD PyTorch LinearSpec 自适应首错位置免训练策略实验手册

> 实验入口：`observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh`
>
> 实现目录：`observations/adaptive_failure_locator_search/`
>
> 默认结果根：`/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/`
>
> 默认正式配置：LinearSpec LoRA、block size 16、BF16、temperature 0、draft threshold 0、non-thinking、历史窗口1/2/4、mean/median/EWMA、标准免训练规则网格、request级60%/20%/20%切分。

## 1. 实验目标和“最优”的严格含义

本实验只研究：能否用当前 draft 的逐位置 confidence 特征与同一 request 过去1/2/4轮的真实验证历史，比固定 `token_y_drop_pct > 0.15` 更准确地定位当前 verifier 的首个不通过位置。

它不执行 verify/draft overlap，不使用预测结果改变 token、cache 或生成轨迹，也不训练神经网络、分类器或回归器。所谓“搜索”是枚举预先声明的可解释公式、窗口、聚合方式、阈值和决策方式；所谓“最优”是该有限规则空间内，在无泄漏 selection 上取得最高非 AIME24 数据集等权宏平均 Exact-F1 的唯一全局策略。

不能声称该策略是所有数学函数中的绝对最优。报告会同时给出：

- 固定0.15原始基线；
- 调过阈值但不用历史的全局 current-only 基线；
- 使用历史1/2/4轮的策略族；
- 各数据集在 test 上事后选择的专属 oracle 及 global-to-oracle regret。oracle 只衡量全局策略距离数据集专属上界多远，绝不进入策略选择或部署。

## 2. 独立性和因果约束

这条链路不读取正在运行或此前任何实验的 trace、summary、report 或 metrics。每个非 dry-run 都重新加载本地 checkpoint、通过 NeMo-Skills 重新发送真实请求，并建立独立时间戳目录。

生成过程为普通 LinearSpec：

```text
当前 causal prefix/cache + seed
  → 完整 draft
  → 记录当前 verify 前已可见的逐位置特征
  → 正常 verifier
  → 记录真实首错 q
  → 正常提交 token/cache
```

定位器不参与最后一步，因此所有候选策略可以共享同一份新 trace 离线重放，不必为1600余个候选各自重新推理。历史特征严格来自当前 request 已经完成的过去轮；不会跨 request，不会使用当前 verifier 或未来轮。

新实现不修改 `method/`、`xp/`、SGLang fork、已有 observation 或模型 remote code。server、GPU显存占位进程、端口锁、partial trace 和结果目录全部属于本次运行；清理只针对本入口创建的PID。默认从 `36000+GPU ID` 起寻找空闲端口。

## 3. 标签、位置和有效轮

对 block size `L`：

- position 0 是上一轮 verifier 给出的 seed，不是待定位 draft token；
- draft positions 为 `1..L-1`；
- `q` 是 verifier 第一个不匹配的 draft position；
- 所有 draft positions 均通过时 `q=NONE`；
- `p` 是某条免训练规则在当前 verify 前预测的位置，未越阈值时 `p=NONE`；
- 只有 `p=q` 才算严格精确命中。

例：position 1、2通过，position 3首次失败，则 `q=3`；若规则给 `p=4`，它既是一次错误尝试（FP），也是对真实错误的一次未精确召回（FN）。

默认主分析仅使用 `analysis_valid=true` 的轮，即：

- 本轮接收 token 中没有 EOS；
- 当前剩余 generation budget 至少能容纳一个完整 block。

raw trace 永远保留全部轮。`--include-boundary-rounds` 可把边界轮加入主统计，但不会改变推理。

## 4. 每轮真实记录

`traces/failure_locator_<dataset>.jsonl` 每行是一轮，至少包含：

- `request_id/benchmark/round_index/generation_offset/cache_length`；
- `block_size/remaining_generation_budget/analysis_valid/eos_hit/budget_boundary`；
- `matched_draft_tokens/accept_length/mismatch_position/full_accept`；
- selected-token confidence，softmax 分母排除 MASK；
- top1-top2 概率 margin、entropy、selected-is-top1；
- 每个位置之前的 prefix mean/median/min；
- `prefix_drop_pct` 和 `local_drop_pct`；
- 每个 draft position 是否被连续验证通过；
- verify 前的过去1/2/4轮接收、首错、good confidence、error confidence 统计；
- `--trace-detail tokens` 时额外保存 draft/verifier token IDs。

原始硬阈值定义保持完全一致：

```text
prefix_drop(i)=1-C_i/mean(C_1,...,C_{i-1})
p=从左到右第一个严格满足 prefix_drop(i)>threshold 的位置
```

position 1 没有前置 draft confidence，因此原始规则结构上不能预测 `q=1`。新搜索空间中的 absolute/history 特征能够覆盖 position 1，报告单列其发生率和召回表现。

## 5. 免训练候选规则空间

### 5.1 Current-only

- `prefix_drop`：当前 confidence 相对当前左侧均值的下降；包含原始严格0.15和全局调阈值版本。
- `prefix_median_drop`：把均值换成稳健中位数。
- `local_drop`：`1-C_i/C_(i-1)`，寻找相邻突降。
- `abs_risk`：`1-C_i`，可以预测 position 1。
- `margin_risk`：`1-(top1-top2 probability margin)`。
- `entropy`：当前 draft 分布熵，越大风险越高。

### 5.2 History-adaptive

过去窗口中，把 verifier 已通过位置的 confidence 记为 `C_good`，首错位置 confidence 记为 `C_err`：

```text
hist_good_drop(i)=1-C_i/aggregate(C_good)
hist_error_drop(i)=1-C_i/aggregate(C_err)
hist_separator(i)=(C_good-C_i)/(C_good-C_err)
```

例：历史 `C_good=0.8,C_err=0.4`，当前 `C_i=0.6`，则 separator=0.5。

搜索维度：

- history window：默认1、2、4轮；
- aggregate：mean、median、EWMA（alpha=0.5）；
- position prior：过去首错位置只作为软加分，不会直接硬编码位置；
- decision：从左到右第一个严格越阈值的 `first`，或全 block 最大风险且越阈值的 `max`；
- threshold：按不同分数尺度设置预声明网格；
- cold start：所需历史不存在或历史从未出现首错时回退原始严格 `prefix_drop>0.15`，不以0填充缺失值。

`--grid compact|standard|extended` 控制阈值和位置先验的网格密度，不改变规则类型。默认窗口和三种聚合下，compact约756项、standard约1612项；extended用于更细的二次搜索。

## 6. 无泄漏全局选择

按 `dataset + request_id + split_seed` 稳定排序，在每个数据集内部切分 request：

```text
search 60% / selection 20% / test 20%
```

同一 request 的所有轮始终位于同一 split。小 smoke 若只有1个request只能落入search；此时报告会明确 test 为NA，不能当正式结论。

选择步骤：

1. 在 search 上评价全部候选；每项指标先在数据集内算，再对所有非 AIME24 数据集等权平均。
2. 取全局前 `--shortlist` 项，并强制补入每个规则族/历史窗口的代表项及原0.15，避免大规则族完全挤掉小规则族。
3. 在 selection 上只比较 shortlist，冻结唯一公式和唯一参数集。
4. 冻结后才计算每个数据集的 test 指标；test 不反馈到策略选择。
5. 每完成一个新数据集重新执行上述过程，因此中途 winner 标记为临时；所有数据集终止后才标记最终。

默认 `--search-max-rounds-per-dataset 50000` 只限制全部候选的粗搜索成本，保持各数据集搜索贡献上限一致；shortlist 的 selection、最终 winner 的 test 和报告统计使用完整相应 split。设为0可取消上限。

AIME24：

- 默认 benchmark 列表根本不运行 AIME24；
- 若用户显式传入，仍保存它的 trace、summary、metrics 和报告数据集行；
- 永不参与 search/selection、候选排名、oracle 或宏平均。

## 7. 指标定义

令 `attempt=(p!=NONE)`，`failure=(q!=NONE)`，`exact=(p=q且q!=NONE)`：

- `Coverage=attempts/valid rounds`；
- `Exact Precision=exact/attempts`；
- `Exact Recall=exact/failure rounds`；
- `Exact-F1`：上述 Precision/Recall 的调和平均，是主选择指标；
- `H100=100*exact/valid rounds`；
- `Full false alarm`：`q=NONE` 的全接收轮中仍预测位置的比例；
- `Miss`：有真实首错但预测NONE的比例；
- `Early/Late`：错误轮上预测早于/晚于真实q的比例；
- `MAE/Bias`：有真实首错且做出预测时 `|p-q|` 和 `p-q` 的均值；
- `±1/±2 Recall`：所有错误轮中预测落在 `q±1/q±2` 的比例；
- `position1 share/recall`：原规则结构性盲点的发生率与覆盖能力；
- request-cluster bootstrap 95% CI：默认200次，同一回答相邻轮作为一个聚类重采样。

全局选择的排序为：

1. 非 AIME24 数据集等权宏 Exact-F1；
2. 最低数据集 Exact-F1 更高，避免相同宏平均只靠少数数据集；
3. H100更高；
4. full false alarm更低；
5. MAE更低。

报告还给出原0.15、current-only、history、各family、H1/H2/H4和matched-coverage参照。任务 accuracy 仅审计生成输出正常，不参与定位规则选择。本实验不把wall time、吞吐或显存作为算法结论指标。

## 8. 结果目录和增量语义

每次非 dry-run 创建：

```text
/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_YYYYMMDD_HHMMSS/
├── Settings.json
├── Settings.md
├── report.md
├── benchmark_status.jsonl
├── traces/failure_locator_<dataset>.jsonl
├── summaries/failure_locator_<dataset>.json
├── analysis/strategy_search.json
├── metrics/metrics_<dataset>.json
├── eval_runs/<dataset>/eval-results/<dataset>/
└── runtime/
    ├── failure_locator_<dataset>.partial.jsonl
    └── <dataset>/
        ├── server.log
        ├── nemo_skills.log
        └── pytorch_request_stats.jsonl
```

创建顺序：

1. 建立唯一时间戳目录；
2. 立即写 `Settings.json`、中文 `Settings.md`；
3. 立即生成带完整变量解释和待运行进度表的 `report.md`；
4. server 先写 runtime 下的 partial trace；只有 NeMo-Skills 与 metrics 成功后才移动到正式 `traces/`，失败的半截数据不会混入策略搜索；
5. 每完成一个数据集就重算已完成数据上的策略、写该数据集summary并原子刷新report；
6. 单数据集失败不会删除其他已完成项，入口继续后续数据集并最终返回非零状态提醒检查。

所有 Markdown 表格使用紧凑列名和居中对齐 `:---:`；所有缩写在报告末尾有中文解释和例子。`analysis/strategy_search.json` 的 `all_search_candidates` 保存全部候选的公式、超参和search宏指标，Markdown展示顶部候选与各规则族/窗口消融，避免把1600余行全部塞进正文。

## 9. 默认数据集和任务指标

默认正式九项与既有 PyTorch+NeMo-Skills 协议对齐，但排除 AIME24：

```text
gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1
```

`dataset:1` 表示每题生成一次，不是只跑1个样本；smoke 需要额外传 `--max-samples 1`。

任务正确率字段：

- GSM8K、MATH-500、AIME25、GPQA、MMLU：`symbolic_correct`；
- HumanEval、MBPP：`passing_base_tests`；
- IFEval：`average_score`；
- LiveCodeBench-C++：`accuracy`。

## 10. 推荐命令（全部单行）

查看帮助：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --help
```

只校验参数、选定GPU/端口，不建结果目录、不加载模型：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --block-size 16 --history-windows 1,2,4 --grid compact --gpu-device 0 --tokens 64 --context-length 2112 --max-samples 1 --dry-run
```

真实GSM8K单样本smoke：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid compact --gpu-device auto --gpu-candidates 0,1,2,3 --gpu-min-free-gb 24 --tokens 96 --context-length 2144 --max-samples 1 --temperature 0 --threshold 0 --disable-thinking --bootstrap-replicates 10
```

早期GSM8K子集搜索，保留完整三段request切分：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --block-size 16 --history-windows 1,2,4 --grid standard --gpu-device auto --gpu-candidates 0,3 --gpu-min-free-gb 24 --tokens 8192 --context-length 10240 --max-samples 100 --temperature 0 --threshold 0 --disable-thinking
```

多个数据集，仍只选一套共享策略：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1,aime25:1,gpqa:1 --block-size 16 --history-windows 1,2,4 --grid standard --gpu-device 3 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

默认正式九项全量运行：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid standard --gpu-device auto --gpu-candidates all --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking --search-ratio 0.6 --selection-ratio 0.2 --bootstrap-replicates 200
```
```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid standard --gpu-device 2 --gpu-memory-reserve-gb 25 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking --search-ratio 0.6 --selection-ratio 0.2 --bootstrap-replicates 200
```

显式GPU 3并预留10 GiB后再加载模型：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1 --gpu-device 3 --gpu-memory-reserve-gb 10 --block-size 16 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

自动GPU，最多等待1小时直到候选GPU至少有30 GiB空闲：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --gpu-device auto --gpu-candidates 0,3 --gpu-min-free-gb 30 --gpu-wait-timeout-s 3600 --gpu-poll-interval-s 30 --block-size 16 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

自定义端口、client并发和NeMo chunks：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1 --gpu-device 3 --port 36888 --client-concurrency 4 --num-chunks 4 --block-size 16 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

注意：client可并发发请求，但原生单模型server用锁串行进入GPU；这不影响每个request的历史定义。本实验不报告serving吞吐。

LinearSpec base对照，不加载LoRA：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --mode linearspec_base --benchmarks gsm8k:1 --gpu-device 3 --block-size 16 --tokens 512 --context-length 2560 --max-samples 20 --temperature 0 --threshold 0 --disable-thinking
```

更换block size；策略仍在本次配置的固定L上跨数据集全局共享：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1 --gpu-device 3 --block-size 32 --history-windows 1,2,4 --grid standard --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

扩大规则网格做二次细搜：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1,aime25:1,gpqa:1 --gpu-device 3 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid extended --shortlist 150 --search-max-rounds-per-dataset 0 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

改变request切分，保留30%最终test：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1,math-500:1,aime25:1,gpqa:1 --gpu-device 3 --block-size 16 --search-ratio 0.5 --selection-ratio 0.2 --split-seed 20260829 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

额外保存token IDs做人工审计：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --gpu-device 3 --block-size 16 --trace-detail tokens --tokens 256 --context-length 2304 --max-samples 5 --temperature 0 --threshold 0 --disable-thinking
```

边界敏感性分析：

```bash
bash observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh --benchmarks gsm8k:1 --gpu-device 3 --block-size 16 --include-boundary-rounds --tokens 256 --context-length 2304 --max-samples 10 --temperature 0 --threshold 0 --disable-thinking
```

仅在同一次新实验自己的trace上重新运行离线规则搜索（不会重新推理）：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python observations/adaptive_failure_locator_search/strategy_search.py --run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_YYYYMMDD_HHMMSS --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid extended --split-seed 20260828 --search-ratio 0.6 --selection-ratio 0.2 --shortlist 150 --report-top 30 --search-max-rounds-per-dataset 0 --bootstrap-replicates 500
```

离线重搜后刷新中文报告：

```bash
/data/home/wly/.conda/envs/nld_sglang/bin/python observations/adaptive_failure_locator_search/run_manager.py report --run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_YYYYMMDD_HHMMSS
```

## 11. 参数逐项解释

### 推理和数据

- `--mode`：`linearspec_lora`（默认）或 `linearspec_base`。
- `--benchmarks`：逗号分隔的NeMo-Skills `dataset:repeats`；支持单/多数据集，数据集名不能重复。
- `--model/--lora-path`：本地基础模型和LinearSpec LoRA路径。
- `--served-model-name`：OpenAI兼容API中暴露的模型名，不改变权重。
- `--dtype`：BF16/FP16/FP32别名。
- `--block-size`：固定解码block；默认16。一次运行不混合不同L。
- `--threshold`：LinearSpec draft迭代unmask阈值，不是待搜索的首错定位阈值；复现默认0。
- `--temperature/--top-p`：与既有管线对齐；原生server应用temperature，不应用top-p，正式定位应固定temperature 0。
- `--tokens/--context-length`：请求最大生成token和server上下文上限；后者需容纳prompt加向上对齐后的生成预算。
- `--max-samples`：每数据集最多样本数；不传为全量。
- `--quick-test`：转交NeMo-Skills快速模式。
- `--prepare-missing-data`：允许准备/下载缺失数据；默认不执行下载，已有缓存按既有管线同步。

### 规则搜索

- `--history-windows`：逗号分隔正整数，默认1,2,4。
- `--aggregations`：mean/median/ewma的非空子集。
- `--grid`：compact用于smoke/早期检查，standard用于正式选择，extended用于更细验证。
- `--split-seed`：稳定request切分种子；不是模型采样seed。
- `--search-ratio/--selection-ratio`：数据集内部request比例；剩余比例自动作为test，二者之和必须小于1。
- `--shortlist`：search全局前若干候选进入selection；代码还会补入各规则族代表和原0.15。
- `--report-top`：Markdown和JSON保留的顶部候选数。
- `--search-max-rounds-per-dataset`：全候选粗搜中每数据集最多轮；0为不限。不会裁剪shortlist selection和winner test。
- `--bootstrap-replicates`：request聚类bootstrap次数；0关闭。
- `--include-boundary-rounds`：把EOS/末端不足block轮纳入主分析。

### GPU、端口和并发

- `--gpu-device ID|auto`：指定一张物理GPU或自动选择；原生8B入口每次只使用一张卡。
- `--gpu-candidates`：auto时允许候选ID列表或all。
- `--gpu-min-free-gb`：auto候选最低空闲显存；满足后优先算力利用率低、显存利用率低、空闲显存高者。
- `--gpu-wait-timeout-s/--gpu-poll-interval-s`：无合格GPU时等待总时长和轮询间隔；timeout 0表示立即失败。
- `--gpu-memory-reserve-gb`：在同卡先实际占用指定显存，再加载模型；独立占位进程在本入口退出时清理。
- `--port`：显式端口；不传则自动搜索并持有本实验端口锁。
- `--batch-size`：为接口对齐保留，只允许1。
- `--client-concurrency/--num-chunks`：客户端并发上限和NeMo数据分块数。

### Prompt、环境和输出

- `--enable-thinking/--disable-thinking`：控制chat template；互斥。
- `--keep-thinking/--strip-thinking`：控制NeMo后处理；互斥。
- `--max-thinking-tokens`：思考token硬上限。
- `--math-prompt-config`：转交既有数学prompt配置。
- `--trace-detail position|tokens`：是否额外保存token IDs。
- `--output-path`：时间戳结果目录的父目录。
- `--pytorch-python/--eval-python`：模型server和NeMo客户端Python；默认均为`nld_sglang`环境。
- `--nemo-skills-data-dir/--google-research-dir`：数据缓存和IFEval scorer目录。
- `--dry-run`：完整参数校验和GPU/端口解析后退出；不建立结果目录、不加载模型、不准备数据。

## 12. 正式结论检查清单

正式报告至少应满足：

- 默认九项或明确列出的全部非 AIME24 数据集均已完成；
- `report.md` 标记“最终”，不是“进行中”；
- 每个数据集test都有足够request，而非单样本smoke的NA；
- winner只由search/selection选出，`selection_contract.test_used_for_selection=false`；
- 全局策略的formula、threshold、window、aggregation、position weight在所有数据集完全相同；
- 同时查看宏Exact-F1、H100、FA、最差数据集表现、win/tie/loss及oracle regret，不能只看平均；
- 原0.15、全局current-only和history策略都有对照；
- position 1、cold-start、history-ready和边界敏感性有单独统计；
- 任务accuracy只作生成质量审计，不与定位F1混淆；
- 不从时间、吞吐或显存推导本观察阶段的算法结论。
