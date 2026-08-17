# NLD PyTorch LinearSpec 既有 Trace 离线低置信度分析指南

> 入口：`observations/analyze_pytorch_linearspec_low_confidence_offline.sh`
>
> 默认输入：`/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154/`
>
> 默认输出：`/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_low_confidence_offline_results/`

## 1. 实验边界

这是对已经生成的原生 PyTorch LinearSpec confidence trace 做离线统计的独立实验，不是再次执行模型推理，也不是 `observations/eval_pytorch_linearspec_confidence.sh` 的在线采集阶段。

```text
既有 raw_trace_<benchmark>.jsonl
  → 顺序读取每轮 accepted/rejected confidence
  → 重建 token_x_drop_abs / token_y_drop_pct
  → 扫描阈值、统计 precision/recall/FPR/F1
  → 写 curve、summary、Settings 和 report.md
```

分析程序只使用 Python 标准库，不加载模型、不导入 PyTorch、不占用 GPU，也不会修改输入 trace。默认要求输入 trace 的 LinearSpec block size 为 16。

## 2. 与在线实验的关系

在线采集入口：

```bash
bash observations/eval_pytorch_linearspec_confidence.sh --benchmarks gsm8k:1 --mode linearspec_lora --gpu-device 2 --block-size 16 --threshold 0 --temperature 0 --tokens 8192 --context-length 10240 --disable-thinking --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results
```

在线实验产生 `traces/raw_trace_*.jsonl` 后，本离线入口可以对同一批 trace 反复扫描不同阈值，无需再次推理。两者结果分别位于不同实验目录，不能把离线统计的 TPF/TPS 当作模型性能。

## 3. 环境与检查

从项目根目录执行：

```bash
cd /data/home/wly/dLLM/Nemotron-Labs-Diffusion
```

检查默认输入：

```bash
test -f /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154/Settings.json && test -d /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154/traces
```

查看帮助：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh --help
```

## 4. 推荐命令

以下命令均为单行。

只解析并打印默认命令，不创建结果：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh --dry-run
```

使用默认完整 block=16 trace 和默认阈值：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh
```

只分析部分 benchmark：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh --benchmarks gsm8k:1,math-500:1
```

指定另一轮在线 trace：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh --input-run /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_YYYYMMDD_HHMMSS --require-block-size 16
```

复现 2026-08-09 完整报告的扫描区间：

```bash
bash observations/analyze_pytorch_linearspec_low_confidence_offline.sh --input-run /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154 --abs-start 0.140 --abs-end 0.300 --abs-step 0.005 --pct-start 0.15 --pct-end 0.33 --pct-step 0.01 --output-path /data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_low_confidence_offline_results
```

## 5. 参数说明

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--input-run DIR` | 含 `Settings.json` 和 `traces/` 的在线 confidence run | 已完成的 block=16 十项 run |
| `--output-path DIR` / `--out-dir DIR` | 时间戳离线结果根目录 | observation 离线结果目录 |
| `--benchmarks LIST` | 只分析指定的逗号分隔 benchmark | 输入 run 中全部可用项 |
| `--require-block-size N` | 输入 Settings/trace 必须使用的 block size | 16 |
| `--abs-start/end/step` | `drop_abs=C_imean-C_i` 扫描区间 | 0.140/0.300/0.005 |
| `--pct-start/end/step` | `drop_pct=1-C_i/C_imean` 扫描区间 | 0.15/0.33/0.01 |
| `--python PATH` | 执行标准库分析器的 Python | 当前 `python3` 或 `python` |
| `--dry-run` | 校验输入并打印解析后命令，不写结果 | 关闭 |

阈值端点均包含。步长必须为正，起点不能大于终点。`--benchmarks` 使用在线实验中的名称，例如 `human-eval:1`；不存在、失败或校验不通过的 trace 会在结果状态中明确记录。

## 6. 输出结构与指标

```text
pytorch_linearspec_low_confidence_offline_results/
└── offline_low_confidence_YYYYMMDD_HHMMSS/
    ├── Settings.json
    ├── report.md
    ├── curves/
    └── summaries/
```

核心口径：

- `C_i`：排除 MASK 后，被 draft 选中 token 的 softmax probability。
- `C_imean`：同一轮位置 `i` 之前所有 draft candidate confidence 的均值。
- `token_x_drop_abs`：`C_imean-C_i >= x`。
- `token_y_drop_pct`：`1-C_i/C_imean >= y`。
- precision：阈值标记 token 中实际 rejected 的比例。
- rejection recall：阈值覆盖全部可评估 rejected token 的比例。
- accepted FPR：阈值误覆盖全部可评估 accepted token 的比例。

第一个 draft candidate 没有前缀均值，因此不进入 drop 统计；拒绝点之后未被 verifier 验证的 token 也不计为 accepted/rejected。

## 7. 已迁移历史结果

三轮历史离线结果现位于默认输出目录。其中完整报告是：

```text
/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_low_confidence_offline_results/offline_low_confidence_20260809_112909/report.md
```

历史结果只证明对应 trace 与阈值设置下的离线观察；如果在线 trace、block size 或 confidence 定义变化，应创建新的时间戳结果，不能覆盖旧报告。
