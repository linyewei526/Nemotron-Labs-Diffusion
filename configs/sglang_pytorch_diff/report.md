# NLD PyTorch 与 SGLang LinearSpec 差异排查报告

> 排查时间：2026-08-31 CST  
> 状态：已完成原因定位、独立诊断验证和正式代码修复；新启动的评估进程使用修复后口径。

## 1. 结论

主要原因已经确认：SGLang 的 LoRA CUDA Graph 双权重捕获流程注册了
`_dllm_pre_draft_hook = set_lora`，但捕获非因果 draft graph 前从未调用该
hook。因此，当前 `linearspec_lora + CUDA Graph` 的 draft graph 实际捕获的
是 base 权重；随后 verify graph 也按预期捕获 base 权重。捕获完成后 LoRA
delta 被释放，运行时又因 `_graphs_baked=true` 不再动态加 LoRA，最终实际
执行成为：

```text
draft = base，verify = base
```

而不是预期的：

```text
draft = base + LoRA，verify = base
```

这是真实推理路径差异，不是 NeMo-Skills 评测或 confidence 汇总口径造成的。

另有两个次要统计问题：SGLang TPF 不计 prompt prefill，而 PyTorch NFE 计
prefill；SGLang confidence trace 还包含 server 启动 warmup request。这些
需要后续统一，但都不能解释原始 confidence/TPF 的大幅差异。

## 2. 隔离诊断设置

- 数据：GSM8K 前 2 个样本，NeMo-Skills 相同 prompt/pipeline。
- block size：16。
- temperature：0。
- 最大生成：512 token；context：1024。
- batch/client concurrency：1。
- GPU：1；SGLang `mem-fraction-static=0.55`，并用
  `max-total-tokens=1024` 限制诊断 KV pool。
- 所有模式串行启动，使用原有自动端口逻辑；每次结束后 server/proxy 均退出。
- 所有新增脚本、日志、trace 和结果均位于本目录，不修改现有实验实现。

有效对照：

|简称|后端与权重|CUDA Graph|结果目录|
|:---:|:---|:---:|:---|
|Base-G|SGLang base draft|开|`results/sglang_base_graph/linearspec_confidence_20260831_174408`|
|LoRA-G|SGLang LoRA 配置（原实现）|开|`results/sglang_lora_graph/linearspec_confidence_20260831_174548`|
|LoRA-E|SGLang LoRA|关，eager|`results/sglang_lora_eager/linearspec_confidence_20260831_174916`|
|PyTorch|原生 PyTorch LoRA|不适用|`results/pytorch_lora/linearspec_confidence_20260831_175158`|
|LoRA-G-H|SGLang LoRA + 仅诊断时调用 pre-draft hook|开|`results/sglang_lora_graph_predraft_hook/linearspec_confidence_20260831_175855`|

`linearspec_confidence_20260831_174116` 和
`linearspec_confidence_20260831_174224` 是在模型推理前因诊断显存预算过低而
退出的无效启动记录，不参与结论。

## 3. 核心实证

下表只统计两个真实 GSM8K request，排除了 SGLang 启动时的三个 warmup
request。`轮`是 draft/verify 轮数；`AccC` 是通过验证的 draft token 平均
confidence；`RejC` 是首个错误 draft token 平均 confidence；`TPF-S` 是当前
SGLang 的 decode-only 口径。

|模式|轮|AccC|RejC|TPF-S|
|:---:|:---:|:---:|:---:|:---:|
|Base-G|37|0.729900|0.406016|3.3243|
|LoRA-G|37|0.729900|0.406016|3.3243|
|LoRA-E|30|0.945712|0.631206|4.1000|
|PyTorch|30|0.943576|0.642876|—|
|LoRA-G-H|31|0.947535|0.631129|3.9677|

### 3.1 Base-G 与原 LoRA-G 完全相同

去掉时间戳、随机 request ID、backend/mode 标签后：

- 44/44 个 trace 轮次完全相同（包含 warmup）；
- 1182 个可比数值叶子的最大绝对差为 0；
- 只看两个真实 request，37/37 轮的接收长度、draft/correct token、rank 和
  confidence 全部完全相同；
- 两个样本最终 generation 也完全相同。

机器结果：

- `results/base_vs_lora_graph.json`
- `results/base_vs_lora_graph_aligned.json`

这证明当前 SGLang `linearspec_lora + graph` 没有让 LoRA 改变 draft 计算。

### 3.2 禁用 Graph 后 LoRA 立即生效，并与 PyTorch 对齐

LoRA eager 相比原 LoRA graph：

- 第一个 warmup draft 位置的 token 从 1044 变为 1046；
- 对应 confidence 从 0.249815 变为 0.472012；
- 真实 request 总轮数从 37 降至 30；
- AccC 从 0.729900 提升到 0.945712；
- TPF-S 从 3.3243 提升到 4.1000。

LoRA eager 与 PyTorch：

- 两边 completion token 都是 246；
- 两边都是 30 个 draft/verify 轮；
- 30 轮的接收长度、首错 draft token 和 verify correct token 一致；
- 只出现两个 correct-token rank 的 2/3 互换，以及两个 EOS 轮的 tracer
  `emitted_tokens` 口径差异；
- 可对齐 accepted confidence 的平均绝对差为 0.00583；
- 可对齐 rejected confidence 的平均绝对差为 0.01706；
- 四种模式的两个最终 generation 文本哈希完全相同。

PyTorch 计入两个 request 的两次 prefill，因此其 TPF 是：

```text
246 / (30 × 2 + 2) = 3.9677
```

若给 SGLang eager 使用相同口径，也正好是 3.9677。当前 SGLang 报告的
4.1000 来自不计 prefill：

```text
246 / (30 × 2) = 4.1000
```

机器结果：

- `results/sglang_graph_vs_eager_aligned.json`
- `results/sglang_eager_vs_pytorch_aligned.json`

### 3.3 临时补调用 hook 后，CUDA Graph 的 LoRA 信号恢复

`diagnostic_site/sitecustomize.py` 只在独立诊断 server 进程中包装
`CudaGraphRunner.capture_one_batch_size`，在非因果 graph 捕获前调用已经注册
但未使用的 `_dllm_pre_draft_hook`。它不修改 SGLang 源码。

日志 `results/patched_graph_hook.log` 明确记录：

```text
capture bs=1 causal=false hook_present=True
pre_draft_hook_invoked
```

补调用后：

- AccC 从 0.729900 恢复到 0.947535；
- RejC 从 0.406016 恢复到 0.631129；
- 两者已接近 PyTorch 的 0.943576 / 0.642876；
- 两个样本 31 轮，只有第二个样本后段因 CUDA Graph/eager BF16 数值差异多
  一轮；系统性的 base-like draft 退化已消失。

机器结果：

- `results/patched_graph_vs_eager_aligned.json`
- `results/patched_graph_vs_pytorch_aligned.json`

## 4. 对应代码路径

SGLang LoRA bake：

```text
sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py
```

- 先构造 `draft_copy = base + delta`；
- 将 `set_lora` 注册为 `_dllm_pre_draft_hook`；
- 调用 `gr.init_capture()`；
- 捕获完成后设置 `_graphs_baked=true` 并释放 delta。

SGLang graph capture：

```text
sglang_dllm/src/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py
```

- 先直接调用 `capture_one_batch_size(...)` 捕获非因果 graph；
- 没有查找或调用 `_dllm_pre_draft_hook`；
- 只在 causal graph 前调用 `_dllm_pre_verify_hook`。

全仓库 `_dllm_pre_draft_hook` 只有赋值，没有调用。缺失调用来自原始提交
`ce4286ed0aa1a242e79f36c493f31140217feff9`，不是后续 confidence tracer 改动。

## 5. 修复方案（已实施）

### 5.1 必需修复

在每个 batch size / stream 的非因果 graph 捕获前，执行：

```text
pre_draft = getattr(self, "_dllm_pre_draft_hook", None)
if pre_draft:
    pre_draft()
```

随后才调用现有的非因果 `capture_one_batch_size(...)`。现有 causal graph 前的
`pre_verify` 保持不变。

为防止 capture 结束后 live parameter 指针停留在错误权重，还应在
`_bake_dual_weights_into_graphs()` 返回前显式恢复期望的 live base 权重；默认
`draft_only` 下应恢复 base。

### 5.2 必需回归测试

1. hook 调用顺序测试：每个 capture shape 都必须是
   `pre_draft → draft capture → pre_verify → causal capture`。
2. 权重效果测试：相同输入下 `linearspec_base` 与 `linearspec_lora graph` 的
   draft logits/confidence 不得完全相同。
3. graph/eager 对齐测试：LoRA graph 和 LoRA eager 的 draft top-1、接收前缀和
   confidence 应在规定容差内。
4. PyTorch 交叉测试：固定 prompt、block=16、temperature=0，逐轮检查 seed、
   draft token、verify token、首错位置和累计生成偏移。
5. 覆盖多个 CUDA graph batch size 和 stream，避免只修 batch size 1。

### 5.3 统计口径同步

建议同时保留两个明确命名的指标：

- `decode_tpf`：completion / (draft + verify forward)，不计 prefill；
- `end_to_end_nfe_tpf`：completion / (prefill + draft + verify forward)。

两端比较时必须选同一个指标。SGLang confidence tracer 还应过滤 server warmup
request；本次小实验中它额外记录了三个 request，原完整 GSM8K 结果中也正好
表现为 confidence trace 比 decode block 多 7 轮。该偏差在全量数据中很小，
但应修正以保证严格口径一致。

## 6. 本目录新增内容

- `compare_traces.py`：去除易变字段后精确比较两份 trace。
- `analyze_trace_alignment.py`：按 request 对齐两端逐轮字段和 confidence。
- `diagnostic_site/sitecustomize.py`：仅诊断进程使用的临时 pre-draft hook。
- `python_with_predraft_hook.sh`：只对 SGLang launch-server 注入上述诊断 hook。
- `results/`：全部独立实验结果、机器比较 JSON 和 hook 日志。

以上诊断产物均来自正式修复前的隔离排查，没有改写历史实验结果。

## 7. 2026-08-31 正式修复状态

- SGLang CUDA Graph 捕获现在对每个 batch size/stream 执行
  `pre_draft → draft capture → pre_verify → causal capture`。
- LinearSpec graph bake 结束后显式恢复 live base 权重，保证后续 prompt prefill
  不继承 LoRA 指针。
- SGLang 正式 benchmark 前会同时清空 confidence、low-confidence 和
  draft-alignment 三类启动 trace，warmup request 不再进入后续分布统计。
- PyTorch 基线、confidence observation、block-size shadow、failure locator 和五套
  method 接口均使用 decode-only NFE 计算默认 TPF，同时保留 prefill/total NFE 和
  含 prefill 的端到端 TPF 字段。
- 用此前两条真实 GSM8K PyTorch stats 做离线回归后，得到
  `decode_forward_passes=60`、`prefill_forward_passes=2`、
  `total_forward_passes=62`、`TPF=4.1`，与 SGLang eager 的 4.1 一致。
- 本轮相关单元/回归测试共 91 项通过，shell 语法检查与 `git diff --check` 通过。
- 正式源码 GPU smoke 尝试在共享 GPU 上加载完模型后被 SGLang 的显存池启动检查
  终止，未进入 benchmark；原因是当时共存任务只留下约 38 GiB，而模型、双权重、
  KV pool 与 CUDA Graph 需要更多空间。正式修复动作与此前已经跑通的隔离注入
  hook 完全一致，因此不把这次资源失败解释为代码回归。
