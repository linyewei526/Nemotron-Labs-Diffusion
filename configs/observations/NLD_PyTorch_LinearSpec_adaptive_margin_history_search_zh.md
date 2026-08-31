# NLD PyTorch LinearSpec 历史自适应 margin_risk 首错定位实验手册

> 入口：`observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh`
>
> 独立实现：`observations/adaptive_margin_history_search/`
>
> 默认结果根：`/data/home/wly/dLLM/NLD_results/observations/adaptive_margin_history_search_results/`
>
> 默认已有 trace：`/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527`

## 1. 本实验回答什么

此前固定 `margin_risk>0.5` 已明显优于 `prefix_drop_pct>0.15`，但仍对每个 request、每个解码阶段使用同一阈值。本实验检索一个免训练、可解释的动态阈值：

```text
r_t(i)=1-[P_top1(i)-P_top2(i)]
τ_t=f(同一request在t以前的1/2/4轮状态)
p_t=min{i:r_t(i)>τ_t}
```

`q_t` 是当前 verifier 从左到右第一个不通过的位置；full accept 时为 0。目标不是一般准确率最大化，而是同时满足：

1. `p=q>0` 的 Exact Recall 高于固定 `margin_risk>0.5`；
2. 对 verifier 正确位置的误报更少，即 `0<p<q` 与 `q=0,p>0` 的合计/轮低于固定0.5；
3. 所有非 AIME24 数据集共享完全相同的公式和超参，不按数据集特化。

实验不训练分类器或回归器，不进行 overlap，不改变 token/cache/生成结果，也不评价额外搜索的端到端耗时和显存。

## 2. 两种 trace 模式与独立性

- `--trace-mode offline`：只读取指定运行目录的原始 `traces/failure_locator_*.jsonl`，完全忽略其旧 `analysis/report/summary/metrics`，重新做 request 切分、历史重建、参数搜索和最终统计。这最适合当前已完成九数据集 trace 的正式离线研究。
- `--trace-mode rerun`：在本轮时间戳目录的 `trace_source/` 下启动独立 PyTorch LinearSpec + NeMo-Skills 真实推理，自动选择/检查端口，完成后再运行本实验搜索。它不读取任何旧实验产物。

两种模式均创建新的顶层时间戳目录，立即写 `Settings.json`、中文 `Settings.md` 和增量 `report.md`。旧 observation、method、SGLang fork和既有结果均不会被修改。`rerun` 复用经过验证的旧 observation **trace采集入口代码**，但采集新数据；旧入口只在新目录的子目录写文件。

## 3. 因果历史特征

每个历史轮先压缩成以下状态，再在最近 `H=1/2/4` 轮上做 mean、median 或 EWMA（新轮权重0.5）：

- `good_mean`：该轮 verifier 在首错前通过位置的 margin risk 均值；full accept时使用所有位置。
- `good_q75/good_q90`：正确位置 margin risk 的75%/90%分位数，描述“正确位置也可能有多危险”。
- `error`：该轮真实首错位置的 margin risk；历史 full-accept轮在本项为缺失。
- `sep=error-good_mean`：历史错误和正确位置的风险分离度。例如 good=0.20、error=0.65，则 sep=0.45。
- `accept`：历史接收比例；首错为q时取 `(q-1)/(L-1)`，full accept取1。
- `full`：历史 full-accept 指示的窗口聚合值。

当前轮特征和当前轮 `q` 绝不进入 `τ_t`。同一 request 的历史不会跨到另一 request。候选所需历史不足或历史没有首错导致特征缺失时，严格回退 `margin_risk>0.5`，不会回退旧 drop 0.15，也不会以0填充。

## 4. 免训练公式族

令全局常数 `R(x)` 是“先按数据集求均值、再等权平均”的历史特征参考值；full_data模式用九数据集全部有效轮拟合，split模式只用search拟合。所有阈值裁剪到 `[0.05,0.95]`。

- `good_center`：`τ=0.5+α·(G-R(G))+offset`。历史正确位置本身风险高时提高阈值，避免把正常低margin误报为首错。
- `good_direct`：`τ=(1-s)·0.5+s·(G+δ)`。直接用历史正确风险上沿加安全余量。
- `separator`：先取 `G+λ(E-G)` 作为历史正确/错误分界，再以 `s` 向0.5收缩。
- `accept_center`：`τ=0.5+β(A-R(A))+γ(F-R(F))`，只用历史接收比例和full率表达当前request/阶段能力。
- `joint_center`：同时使用 `good_q90`、接收比例和full率。
- `gap_center`：同时使用历史正确风险和历史 `error-good` 分离度。
- `accept_gate`：历史接收比例相对全局参考低/中/高时，使用三档阈值；它是最容易部署的离散策略。
- `fixed_margin`：严格 `margin_risk>0.5`，既是冷启动回退也是主要对照。

`--grid compact|standard|extended`只改变上述参数取值密度，不改变公式。正式全数据命令使用extended；机器可读JSON在 `all_full_data_candidates` 中保存所有候选及其全数据等权宏指标。

## 5. 选择协议：全数据全局最优与held-out验证

### 5.1 `full_data`：本轮要求的正式协议

`--selection-protocol full_data`严格表示：

1. 九个非AIME24数据集的每个源trace request/sample都被扫描、计数并分配给full_data；不切出held-out集合。
2. 每个request中全部 `analysis_valid=true` 的解码轮都进入每一个候选的指标计算。若某个request只有EOS或不足完整block的边界轮，则它记为 `ZeroReq` 并保留在全样本来源审计中，但没有可合法进入定位指标的轮，不能伪造一轮或改变预声明边界口径。
3. 所有声明的候选公式和参数点都在同一批全数据上评价；不经过shortlist。
4. 禁止每数据集轮数截断。该模式强制要求 `--search-max-rounds-per-dataset 0`，任何非零值都会在建立结果目录前报错。
5. 每个候选先在每个数据集内部汇总全部有效轮的计数和指标，再对九个数据集取算术平均，因此每个数据集权重严格为 `1/9`，与它有多少sample或解码轮无关。
6. AIME24完全排除；九个数据集共用唯一公式和唯一参数集。
7. offline模式会检查源 `Settings.json`：如果源运行使用过 `--max-samples` 或 `--quick-test`，或缺少任一请求数据集的非空trace，直接拒绝。rerun模式的full_data同样禁止这两个子集参数。

全数据模式的数据集内部采用“全部有效轮pooled”：长response在本数据集内会提供更多真实首错事件；数据集之间则严格等权。因此MMLU即使有远多于其他数据集的sample或轮，进入全局目标后仍只占1/9。报告中的 `Req`（源trace request总数）与 `Full`（分配给full_data的源request数）必须逐数据集相等，并满足 `EvalReq+ZeroReq=Req`；其中 `EvalReq` 至少有一轮有效分析，`ZeroReq` 只有预声明排除的边界轮。这样既证明没有按样本抽样，也不会把无有效定位目标的边界轮错误混入指标。

全数据最优的严格定义是：在声明的有限候选网格中，先筛选相对固定 `margin_risk>0.5` 同时满足 `ΔRec>0` 和 `ΔCFP/R<0` 的策略，再取Exact Recall最高者，以CFP/R更低、F1更高依次打破平局。如果不存在严格支配者，固定0.5保持全数据winner，并完整报告Pareto前沿。

它回答“现有九数据集全部数据上的描述性全局最优”。由于同一批数据也参与选择，它不是未见数据泛化估计；以后若需泛化结论，应另采独立trace或使用下面的split协议。

### 5.2 `split`：可选held-out协议

每个数据集内部按 `dataset+request_id+split_seed` 稳定划分request，默认：

```text
search 60% / selection 20% / test 20%
```

同一 request 的所有轮只能落在同一 split：

1. search 拟合历史特征全局参考值并粗评全部候选；`--search-max-rounds-per-dataset` 对每个数据集使用相同轮上限。
2. search Pareto 前沿、主排序候选及各公式/窗口/聚合代表进入 shortlist。
3. selection 上检查相对固定0.5的严格双目标支配：`ΔRec>0` 且 `ΔCFP/R<0`。
4. 若有严格支配者，优先最大 Exact Recall，再最小正确位置误报/轮，再最大F1；若没有，selection winner保持固定0.5，并完整报告 Pareto trade-off。
5. 冻结后才计算 test。只有冻结候选在test仍满足 `ΔRec>0` 且 `ΔCFP/R<0` 时才推荐adaptive；否则部署建议保持固定0.5。消融代表也只在selection选定后做test诊断，test不反向改变冻结候选或调参。

AIME24在两种协议中都不参与候选搜索、宏平均或全局策略选择。所有宏平均都先在每个非AIME24数据集内计算，再让数据集等权平均。

## 6. 指标和误报分母

令报告位置为 `p`、真实首错为 `q`：

- `Rpt=数量(p>0)/轮数`：总报告率。
- `Pre=数量(p=q>0)/数量(p>0)`：Exact Precision。
- `Rec=数量(p=q>0)/数量(q>0)`：Exact Recall，首错命中的主收益。
- `F1`：Pre与Rec调和平均。
- `CFP=数量(0<p<q)+数量(q=0,p>0)`：报告到 verifier 正确位置的次数。
- `CFP/R=CFP/全部轮`：误报主指标，能与Rec共同表达总体收益。
- `CFP/P=CFP/所有报告轮`：一次报告是正确位置误报的条件概率。
- `PosFPR=CFP/[sum(q-1, q>0)+sum(L-1, q=0)]`：以所有实际正确位置为分母。
- `Early/FullFP`：分别拆解错误轮首错前误报和full-accept轮误报。
- `Miss/Late`：有首错却不报告、以及报告晚于首错。
- `MAE/Bias`：有首错且发生报告时的 `|p-q|` 与 `p-q`。
- `±1/±2/P1 Rec`：容差命中和position 1召回。
- `τ均/标/P10/P50/P90`：动态阈值分布；`升/降/等` 是相对0.5的轮占比；`冷/就绪` 是回退/可用占比。
- paired request bootstrap：同一 replicate 对 winner 和参照使用相同 request 重采样，给 `ΔRec/ΔCFP/R/ΔF1` 的95%区间。

报告同时给旧 `prefix_drop_pct>0.15`、固定 `margin_risk>0.5` 和冻结策略，回答用户关心的命中率、正确位置误报率及总报告率三项变化。

## 7. 结果目录与增量报告

```text
/data/home/wly/dLLM/NLD_results/observations/adaptive_margin_history_search_results/adaptive_margin_history_YYYYMMDD_HHMMSS/
├── Settings.json
├── Settings.md
├── report.md
├── analysis/strategy_search.json
├── summaries/trace_<dataset>.json
├── runtime/failure.json（仅失败时）
└── trace_source/（rerun模式的新trace子运行；offline模式为空）
```

`report.md`在目录建立时即生成。full_data模式会在trace校验、历史特征构造、全候选全数据搜索和最终全局最优完成时原子刷新；split模式继续在selection冻结和test后刷新。每张表前就地解释变量与简单例子；所有列使用紧凑中文名与居中对齐。

offline模式不会复制体积很大的 trace；`Settings.json` 记录绝对源路径、每个trace文件的文件名、字节数和mtime，确保来源可审计。不要移动或删除源trace，否则该时间戳结果仍可读，但不能原样重跑。

### 实时进度条

进度条默认开启，无需在推荐命令中增加参数，依次覆盖最耗时的阶段：

- `trace scan`：第一遍按输入trace总字节数校验，右侧显示当前文件名；
- `history features`：第二遍按有效解码轮数构造严格因果历史特征，右侧显示当前数据集；
- `full-data search`：full_data模式全部候选的全数据检索，右侧显示当前公式族；
- `search candidates` 与 `selection shortlist`：split模式的粗搜和selection候选复评；
- `bootstrap vs margin.5/drop.15`：最终两组配对request bootstrap。

每条进度都显示完成数/总数、百分比、处理速率、已用时间和ETA。直接在终端运行时，同一阶段原地单行刷新，刷新频率上限约5次/秒；stderr被重定向到文件或作业调度日志时，每约2%写一条普通日志行，便于 `tail -f` 实时查看且不会产生海量刷新记录。阶段刚开始时没有足够测速样本，ETA显示 `--:--`，随后会逐渐稳定。若确实不需要进度输出，可加 `--no-progress`；该选项只关闭显示，不改变搜索数据、候选、指标或结果文件。

## 8. 推荐命令（全部为单行）

查看全部参数：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --help
```

本轮要求的九数据集全部样本、全部候选、等数据集权重正式离线命令（推荐直接使用；不启动模型或GPU）：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --selection-protocol full_data --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid extended --search-max-rounds-per-dataset 0 --report-top 40 --bootstrap-replicates 500
```

上面命令会让extended网格中的每个候选都使用九数据集全部有效轮；没有split、cap或shortlist。运行时间和CPU内存开销会显著高于此前截断搜索，但不会启动GPU。

旧的request级60/20/20 held-out协议命令（仅在需要泛化估计时使用，不是本轮“全数据最优”命令）：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --selection-protocol split --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid standard --search-ratio 0.6 --selection-ratio 0.2 --split-seed 20260830 --search-max-rounds-per-dataset 30000 --shortlist 120 --bootstrap-replicates 500
```

offline单数据集早期搜索：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid compact --search-max-rounds-per-dataset 10000 --shortlist 40 --bootstrap-replicates 50
```

offline多数据集并取消search轮上限：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1,math-500:1,aime25:1,gpqa:1 --block-size 16 --grid extended --search-max-rounds-per-dataset 0 --shortlist 180 --bootstrap-replicates 1000
```

只校验offline参数和源trace；不建立结果目录：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1 --block-size 16 --grid compact --dry-run
```

只校验九数据集全数据正式命令的强约束；不会开始全搜索：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode offline --selection-protocol full_data --source-run-dir /data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/adaptive_failure_locator_20260828_203527 --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid extended --search-max-rounds-per-dataset 0 --bootstrap-replicates 500 --dry-run
```

真实单样本GPU smoke；新trace与搜索都写入本轮顶层目录：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1 --block-size 16 --history-windows 1,2,4 --grid compact --gpu-device auto --gpu-candidates 0,1,2,3 --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --tokens 96 --context-length 2144 --max-samples 1 --temperature 0 --threshold 0 --disable-thinking --bootstrap-replicates 10
```

真实多数据集重跑，自动选择GPU与空闲端口：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1,math-500:1,aime25:1,gpqa:1 --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid standard --gpu-device auto --gpu-candidates all --gpu-min-free-gb 24 --gpu-wait-timeout-s 3600 --gpu-poll-interval-s 30 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking --search-ratio 0.6 --selection-ratio 0.2
```

指定GPU 3、预留10 GiB并显式使用端口36888：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1,math-500:1 --block-size 16 --gpu-device 3 --gpu-memory-reserve-gb 10 --port 36888 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

真实九数据集重新推理并在新trace上做全数据全局搜索（耗时很长）：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --selection-protocol full_data --benchmarks gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1 --mode linearspec_lora --block-size 16 --history-windows 1,2,4 --aggregations mean,median,ewma --grid extended --search-max-rounds-per-dataset 0 --gpu-device auto --gpu-candidates all --gpu-min-free-gb 24 --gpu-memory-reserve-gb 0 --tokens 8192 --context-length 10240 --temperature 0 --top-p 0.95 --threshold 0 --disable-thinking --bootstrap-replicates 500
```

LinearSpec base、小规模真实对照：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --mode linearspec_base --benchmarks gsm8k:1 --block-size 16 --gpu-device 3 --tokens 512 --context-length 2560 --max-samples 20 --temperature 0 --threshold 0 --disable-thinking --grid compact
```

更换block size为32；一次运行不能混合不同L：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1,math-500:1 --block-size 32 --gpu-device 3 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking --grid standard
```

自定义客户端并发和NeMo chunks（原生server仍以锁保护单模型推理）：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1,math-500:1 --block-size 16 --gpu-device 3 --client-concurrency 4 --num-chunks 4 --tokens 8192 --context-length 10240 --temperature 0 --threshold 0 --disable-thinking
```

允许准备缺失数据；可能访问网络，默认不会执行：

```bash
bash observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh --trace-mode rerun --benchmarks gsm8k:1 --block-size 16 --gpu-device 3 --prepare-missing-data --tokens 256 --context-length 2304 --max-samples 5 --temperature 0 --threshold 0 --disable-thinking
```

## 9. 参数逐项解释

### trace、数据和输出

- `--trace-mode offline|rerun`：读取已有原始trace或重新真实推理；默认offline。
- `--source-run-dir`：offline源运行目录，其下必须有 `traces/failure_locator_*.jsonl`；rerun忽略。
- `--output-path/--out-dir`：新时间戳目录的父目录。
- `--benchmarks`：逗号分隔 `dataset:repeats`；支持单/多数据集。`:1` 是每题生成一次，不是只运行1题。
- `--block-size/--block-length`：offline可省略并从trace推断，若显式给出则严格核验；rerun默认16。
- `--include-boundary-rounds`：将EOS/生成预算末端等 `analysis_valid=false` 轮也纳入分析；默认排除但原trace仍保留。
- `--no-progress`：关闭离线trace扫描、历史特征构造、候选检索和bootstrap的实时进度；默认不传，即开启。进度写到stderr，正常结果摘要仍写stdout。
- `--dry-run`：只做参数与offline源目录校验，不建立结果目录、不选GPU、不占端口、不加载模型。

full_data offline还要求源目录存在可审计的 `Settings.json`，并确认其不是max-samples/quick-test子集且声明了全部请求数据集。

### 搜索与统计

- `--selection-protocol split|full_data`：`full_data`是本轮正式要求，每个候选使用所有request的全部有效轮并直接产生全数据全局最优；`split`保留search/selection/test做held-out验证。默认仍为split以保持旧命令兼容，因此正式全数据命令必须显式写 `full_data`。
- `--history-windows`：逗号分隔正整数，默认1,2,4。
- `--aggregations`：`mean,median,ewma` 的非空子集。
- `--grid`：compact用于smoke，standard用于较小网格，extended是本轮全数据正式命令使用的最密预声明网格。
- `--split-seed`：split模式的request稳定切分种子；full_data模式仅作为bootstrap随机种子，不改变纳入的数据。
- `--search-ratio/--selection-ratio`：只在split模式生效；full_data模式忽略。
- `--search-max-rounds-per-dataset`：split模式的候选粗搜轮上限；0表示不限。full_data模式必须严格为0，否则拒绝启动。
- `--shortlist`：只在split模式生效。full_data模式不会建立shortlist，所有候选直接跑完整数据。
- `--report-top`：full_data模式控制JSON中 `top_full_data` 的展示条数，但 `all_full_data_candidates` 永远保存全部候选；split模式对应search候选。
- `--bootstrap-replicates`：配对request bootstrap次数；0关闭。full_data模式的区间是内部重采样不确定性，不是held-out区间。

### 真实重跑模型参数

- `--mode`：`linearspec_lora`（默认）或 `linearspec_base`。
- `--model/--lora-path`：本地基础checkpoint和LoRA目录。
- `--served-model-name`：OpenAI兼容接口暴露名，不改变权重。
- `--dtype`：模型dtype；与已有管线一致默认bfloat16。
- `--threshold`：LinearSpec draft unmask confidence threshold，**不是**本实验检索的locator阈值；复现默认0。
- `--temperature/--top-p`：采样设置；正式比较应使用temperature 0保证轨迹确定。
- `--tokens/--context-length`：最大生成长度与server上下文；默认8192/10240。
- `--max-samples`：每数据集最多题数；不传为全量。
- `--quick-test`：转交NeMo-Skills快速模式。
- `--trace-detail position|tokens`：默认position；tokens额外保存token IDs供人工审计。

### GPU、端口、显存与并发

- `--gpu-device ID|auto`：指定单张物理GPU或自动选择。
- `--gpu-candidates all|ID列表`：auto允许的GPU集合。
- `--gpu-min-free-gb`：auto候选最低空闲显存。
- `--gpu-wait-timeout-s/--gpu-poll-interval-s`：无合格GPU时的最长等待与轮询周期；timeout 0为立即失败。
- `--gpu-memory-reserve-gb`：先在所选GPU真实占用指定GiB再加载模型，进程只由本轮入口清理。
- `--port`：显式端口；不传由trace采集入口搜索空闲端口并用锁防冲突。
- `--batch-size`：接口对齐参数；原生trace server当前只支持1。
- `--client-concurrency/--num-chunks`：NeMo客户端并发与数据分块数。

### Prompt、环境与数据准备

- `--enable-thinking/--disable-thinking`：互斥的chat-template思考开关。
- `--keep-thinking/--strip-thinking`：互斥的NeMo输出后处理开关。
- `--max-thinking-tokens`：思考token上限。
- `--math-prompt-config`：转交既有数学prompt配置。
- `--pytorch-python/--eval-python`：模型端与NeMo客户端Python；默认均为本地 `nld_sglang` 环境。
- `--nemo-skills-data-dir/--google-research-dir`：数据缓存与IFEval scorer根目录。
- `--prepare-missing-data`：允许准备缺失数据；默认false，避免擅自下载。

## 10. 正式结果判定清单

- `report.md` 状态是“全部候选已在九数据集全部有效轮上做等数据集权重全局搜索”。
- `selection_protocol=full_data`，`full_data_contract.all_candidates_use_all_included_valid_rounds=true`。
- 九个非AIME24数据集逐项满足 `Req=Full`、`EvalReq+ZeroReq=Req`，且 `full_data_datasets` 恰好包含九项；ZeroReq允许非零，但必须确实没有 `analysis_valid=true` 的轮。
- `search_rounds_after_cap=null`、`shortlist_count=null`、`full_data_contract.round_cap=null`。
- `all_full_data_candidates` 条目数等于 `candidate_count`；每个候选的宏指标都是九个数据集等权平均。
- `full_data_dominates_fixed=true` 时adaptive才是双目标全数据winner；否则固定0.5保持winner，Pareto只描述取舍。
- 同时检查 `ΔRec>0` 和 `ΔCFP/R<0`，并结合配对bootstrap区间判断变化是否稳定，不只看F1。
- 查看 `CFP/P`、`PosFPR` 与 `Rpt`，排除单纯少报告制造的低误报。
- 查看各数据集行、窗口/聚合/公式消融，确认同一全局策略不是只靠大数据集或少数数据集。
- 查看动态阈值分布和cold占比；若大量轮回退0.5，应谨慎解释history贡献。
- 旧drop 0.15只作历史对照；自适应策略的主要参照固定为margin 0.5。
- 全数据结果是当前有限网格上的描述性最优，不冒充held-out泛化结论。
- 不从本实验推导吞吐、端到端时延或显存结论。
