# Observation 实验入口

本目录只存放 `configs/observations/` 所描述实验的用户级 Shell 入口。底层实现仍位于项目根目录的 `xp/`、`sglang_dllm/` 等目录。

所有入口都可以从项目根目录以 `bash observations/<入口>.sh ...` 调用。默认结果根目录为 `/data/home/wly/dLLM/NLD_results/observations/`，可通过各入口的 `--output-path` 覆盖；环境变量 `NLD_OBSERVATION_RESULTS_ROOT` 可以整体覆盖默认 observation 结果根目录。

所有入口均支持 `--help`；正式入口和诊断包装器均支持 `--dry-run` 路径/参数检查。入口使用自身文件位置解析 `PROJECT_DIR`，因此也可以从项目目录之外用绝对路径调用。

| 入口 | 功能 |
|---|---|
| `eval_sglang.sh` | SGLang + NeMo-Skills 正式评测及 chat benchmark |
| `eval_pytorch_nemo.sh` | 原生 PyTorch + NeMo-Skills 正式评测及 chat benchmark |
| `eval_linearspec_confidence.sh` | SGLang confidence/rank trace |
| `eval_pytorch_linearspec_confidence.sh` | PyTorch confidence/rank trace |
| `eval_linearspec_draft_alignment.sh` | SGLang draft/final alignment |
| `eval_linearspec_low_confidence_rejection.sh` | SGLang low-confidence/rejection 阈值观察 |
| `analyze_pytorch_linearspec_low_confidence_offline.sh` | 既有 PyTorch trace 的 CPU 离线阈值分析 |

项目官方 SLURM 入口 `eval.sh` 仍保留在项目根目录；解码优化实验继续位于 `method/`，两者均不属于本目录。
