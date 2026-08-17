# Nemotron-Labs-Diffusion 解码与代码实现导读

本文档面向已经读过 `Nemotron_Diffusion_Tech_Report_v1.pdf` 的读者，目标是把论文中的三模态解码概念对应到本项目和本地模型目录的实际代码实现，帮助你按调用链读懂推理、评测和服务代码。

分析范围：

- 项目目录：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion`
- 论文：`/data/home/wly/dLLM/Nemotron-Labs-Diffusion/Nemotron_Diffusion_Tech_Report_v1.pdf`
- 模型目录：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B`
- 核心远程代码：
  - `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py`
  - `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_ministral.py`
  - `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/configuration_nemotron_labs_diffusion.py`

重要结论先行：

- 公开仓库主要是推理、聊天、评测、服务化包装；完整训练流水线不在本仓库中。
- 真正的三种原生生成方法在模型目录的远程代码中：`generate()`、`ar_generate()`、`linear_spec_generate()`。
- 当前本地模型代码实现了 AR、block-wise diffusion、linear self-speculation；论文中的 quadratic self-speculation、SOL 分析和训练好的 diffusion sampler 在本仓库中没有完整可调用实现。
- 同一个模型切换模式的关键是 attention 层里的 `diffusion_lm` 布尔开关：`True` 表示 bidirectional/diffusion attention，`False` 表示 causal/AR attention。

## 1. 论文概念到代码的映射

| 论文术语 | 代码入口 | 当前仓库状态 | 说明 |
| --- | --- | --- | --- |
| AR decoding | `NemotronLabsDiffusionModel.ar_generate()` | 已实现 | 纯自回归；每步生成 1 个 token；使用 causal attention 和 KV cache。 |
| Diffusion decoding / dLLM decoding | `NemotronLabsDiffusionModel.generate()` | 已实现 | block-wise denoising；每个 block 先放 mask，再按 confidence 并行解 mask。 |
| Linear self-speculation / Linear SS | `NemotronLabsDiffusionModel.linear_spec_generate()` | 已实现 | diffusion draft + AR verify；接受最长匹配前缀，再加一个 AR bonus token。 |
| LoRA-enhanced Linear SS | `linear_spec_lora/` + `linear_spec_generate()` | 已实现 | PEFT LoRA 挂到 `o_proj`，代码在 draft 阶段打开、verify 阶段关闭。 |
| Quadratic self-speculation / Quad SS | 论文附录 C | 本仓库未实现 | 当前无 `quadratic_spec_generate` 或 `quad` mode；SGLang 文档也只列 AR、FastDiffuser、LinearSpec。 |
| Diffusion trained sampler | 论文 3.2 和附录 A | 本仓库未实现完整路径 | `xp/dlm_api/dlm_generate/utils/sampler.py` 是通用/遗留采样辅助；当前本地模型 `generate()` 没有 `sampler` 参数。 |
| SOL 分析 | 论文第 4 节 | 本仓库未实现 | 没有递归 dynamic compaction / SOL evaluation 脚本。 |
| AR+dLM mixed logits | `xp/dlm_api/dlm_generate/nemotron_mixed.py` | 包装层存在，依赖额外 code repo | 需要模型类有 `generate_mixed()`；当前本地模型文件没有该方法。 |

## 2. 项目代码组织

项目根目录的主要文件和目录：

- `README.md`：使用说明，列出三种 mode：`ar`、`dlm`、`linear_spec`。
- `chat/`：最小聊天入口。
  - `chat_ar.py` 调 `model.ar_generate()`。
  - `chat_dlm.py` 调 `model.generate()`。
  - `chat_linear_spec.py` 调 `model.linear_spec_generate()`。
  - `chat_linear_spec_lora.py` 先用 PEFT 挂 LoRA，再调 `linear_spec_generate()`。
  - `chat.py` 是统一多轮 CLI launcher。
- `evaluate.py`：单进程轻量评测，不起 server，不依赖 NeMo-Skills，只支持内置 `gsm8k` 和 `math-500`。
- `eval.sh`：SLURM + 容器 + NeMo-Skills 的大规模评测编排入口。
- `xp/dlm_api/`：OpenAI-compatible 推理服务。
  - `dlm_batch_server.py`：每个 GPU worker，一个 FastAPI server，负责加载模型、batch、NFE 记录、算法分发。
  - `dlm_load_balancer.py`：多 worker 前面的负载均衡器。
  - `dlm_generate/`：生成算法注册表和具体算法类。
- `xp/nemo-skills/eval_dlm.py`：NeMo-Skills eval client 的扩展版，把 diffusion 参数放入 OpenAI `extra_body`。
- `xp/examples/run_dlm_eval_pipeline_gpu_only.sh`：`eval.sh` 实际调用的 GPU-only pipeline。
- `sglang_spark/`：SGLang 部署说明和 launcher。这里使用外部 SGLang fork，不是本仓库原生 Python 解码实现。
- `configs/`：配置和文档目录；本文档新增于此。

模型目录的主要文件：

- `config.json`：模型架构和 diffusion 配置。当前 8B 的关键字段：
  - `architectures = ["NemotronLabsDiffusionModel"]`
  - `vocab_size = 131072`
  - `hidden_size = 4096`
  - `num_hidden_layers = 34`
  - `num_attention_heads = 32`
  - `num_key_value_heads = 8`
  - `block_size = 32`
  - `mask_token_id = 100`
  - `eos_token_id = 11`
  - `dlm_paradigm = "bidirectional"`
  - `max_position_embeddings = 262144`
  - `rope_parameters.rope_type = "yarn"`
- `configuration_nemotron_labs_diffusion.py`：`NemotronLabsDiffusionConfig`。
- `modeling_ministral.py`：Ministral3 transformer backbone，包含 attention 的 AR/diffusion 切换逻辑。
- `modeling_nemotron_labs_diffusion.py`：NLD 模型封装、训练 forward、三种生成方法。
- `linear_spec_lora/adapter_config.json`：Linear SS drafter LoRA 配置，`target_modules = ["o_proj"]`，`r = 128`，`lora_alpha = 512`。
- `model.safetensors`：权重文件。

核心函数定位表：

| 功能 | 文件 | 当前行号 |
| --- | --- | --- |
| 配置类 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/configuration_nemotron_labs_diffusion.py` | `NemotronLabsDiffusionConfig`：25 |
| attention 切换 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_ministral.py` | `Ministral3Attention.forward`：129 |
| backbone forward | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_ministral.py` | `Ministral3Model.forward`：391 |
| 模型封装 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `NemotronLabsDiffusionModel`：157 |
| 随机 noising | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `forward_process`：201 |
| 训练/推理 forward | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `forward`：235 |
| diffusion decode | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `generate`：380 |
| AR decode | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `ar_generate`：534 |
| Linear SS | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `linear_spec_generate`：627 |
| native transfer 选择 | `/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/modeling_nemotron_labs_diffusion.py` | `_get_transfer_index`：835 |
| 轻量评测分发 | `/data/home/wly/dLLM/Nemotron-Labs-Diffusion/evaluate.py` | `generate`：146 |
| 服务算法基类 | `/data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/dlm_api/dlm_generate/base.py` | `GenerationAlgorithm`：16 |
| Nemotron 服务包装 | `/data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/dlm_api/dlm_generate/nemotron.py` | `NemotronGeneration.generate`：118 |
| AR 服务包装 | `/data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/dlm_api/dlm_generate/ar_native.py` | `ArNativeGeneration.generate`：50 |
| OpenAI worker | `/data/home/wly/dLLM/Nemotron-Labs-Diffusion/xp/dlm_api/dlm_batch_server.py` | `_process_algorithm_group`：480 |

## 3. 模型结构和 Attention 切换

### 3.1 `NemotronLabsDiffusionModel`

核心类在 `modeling_nemotron_labs_diffusion.py`：

- `NemotronLabsDiffusionModel.__init__()` 创建一个 `Ministral3Model` 作为 `self.encoder`。
- `self.diffusion_head = nn.Linear(hidden_size, vocab_size, bias=False)` 是输出头。
- `self.mask_token_id` 来自 config。

初始化时会复制 config 得到 `diffusion_config`，并设置：

- `diffusion_config.diffusion_lm = True`：默认用 diffusion/bidirectional attention。
- 如果 `config.dlm_paradigm == "autoregressive"`，则 `diffusion_lm = False`。
- 如果 `config.dlm_paradigm == "block_diff"`，则使用 `NemotronLabsDiffusionFlexAttention`，用于训练时的 dual-stream block attention。
- 当前本地 8B 的 `config.json` 是 `dlm_paradigm = "bidirectional"`，因此默认 attention 类是 `Ministral3Attention`，只靠 `diffusion_lm` 标志在推理时切换。

### 3.2 `Ministral3Attention.diffusion_lm`

`modeling_ministral.py` 的 `Ministral3Attention.forward()` 是模式切换的关键：

- 当 `self.diffusion_lm == True`：
  - 调 attention 时 `attention_mask = None`
  - `is_causal = False`
  - 当前输入内部可以 bidirectional attend
  - 这是 diffusion draft / denoising 的 attention 形式
- 当 `self.diffusion_lm == False`：
  - 使用 `attention_mask`
  - 由 `Ministral3Model.forward(..., use_causal_mask=True)` 构造 causal mask
  - 这是 AR prefill / AR verify / AR generate 的 attention 形式

三种解码方式都复用同一个 encoder 和同一个 output head，只是在不同阶段切换 `diffusion_lm`。

### 3.3 `past_key_values` 的行为

`Ministral3Attention.forward()` 对 cache 有两种处理：

- `past_key_values is not None and use_cache=True`：把当前 key/value 写入 cache。
- `past_key_values is not None and use_cache=False`：不更新 cache，而是把旧 cache 的 key/value 拼到当前 key/value 前面参与 attention。

这正好服务于 diffusion block：

- prefix 的 KV cache 已经由 causal prefill 或上一块 post-block forward 写好。
- 当前 block denoise 时 `use_cache=False`，所以当前 block 可以看 prefix cache，但不会污染 cache。
- block 完成后再用 causal forward 一次写入 cache。

## 4. 训练 forward 在代码中的保留实现

虽然本仓库不包含完整训练流水线，模型远程代码仍保留了训练 forward。

### 4.1 `forward_process()`

`forward_process(input_ids, eps=1e-3, loss_mask=None)` 做 diffusion noising：

1. 对 batch 内每条样本采样 `t ~ Uniform(0, 1)`。
2. 计算 `p_mask = (1 - eps) * t + eps`。
3. 按 `p_mask` 随机选择 masked positions。
4. 如果有 `loss_mask`，把不可训练位置的 mask 清掉。
5. 返回：
   - `noisy_batch`：masked 位置替换成 `mask_token_id`
   - `masked_indices`
   - `p_mask`

这对应论文中的扩散损失随机 mask 和 `1/t` reweighting。

### 4.2 `forward()`

`NemotronLabsDiffusionModel.forward()` 根据 `labels` 和 `dlm_paradigm` 走不同路径：

- `labels is None`：推理路径，不构造 loss。
- `dlm_paradigm == "autoregressive"`：标准 causal LM loss，使用 shifted logits。
- 非 autoregressive 且有 labels：
  - 如果调用方没有传 `masked_indices`，则调用 `forward_process()` 随机 mask。
  - logits 在 masked positions 上做 cross entropy，并除以 `p_mask[masked_indices]`。
  - loss 先 `sum()`，再返回 `(loss, num_mask_tokens)`，由外部按 token 数做 global loss average。
- `dlm_paradigm == "block_diff"`：
  - 训练时把 `noisy_inputs` 和 clean `input_ids` 拼接。
  - 前半段 logits 用于 diffusion loss。
  - 后半段 `causal_logits` 用于 AR loss。
  - 返回 `(diffusion_loss + ar_loss_weight * ar_loss, num_tokens)`。

这与论文中的 joint objective 和 global loss averaging 对应。需要注意：当前本地 8B 推理配置不是 `block_diff`，但代码保留了该训练范式。

## 5. 原生 AR 解码：`ar_generate()`

入口：`modeling_nemotron_labs_diffusion.py` 的 `NemotronLabsDiffusionModel.ar_generate()`。

调用方式示例：

```python
out_ids, nfe = model.ar_generate(
    prompt_ids,
    max_new_tokens=512,
    temperature=0.0,
    eos_token_id=tokenizer.eos_token_id,
)
```

核心流程：

1. 遍历 `self.encoder.layers`，把所有 `layer.self_attn.diffusion_lm` 设为 `False`。
2. 创建 `DynamicCache()`。
3. 对完整 prompt 做一次 causal prefill：
   - 显式传 `position_ids`
   - 显式传 `cache_position`
   - `use_cache=True`
4. 取最后一个 hidden state，经 `diffusion_head` 得到 `next_logit`。
5. 循环 `max_new_tokens` 次：
   - `nfe += 1`
   - `temperature > 0` 时 multinomial sampling；否则 argmax。
   - 如果超过 `max_thinking_tokens` 且还没生成 `</think>`，强制把 next token 改成 `end_think_token_id`。
   - 如果所有 batch 的 next token 都是 EOS，则停止。
   - 否则把刚生成的 token 作为下一步输入，继续调用 encoder 更新 cache。
6. 拼接 `prompt_ids` 和 generated tokens 返回。

实现细节：

- AR 的 prefill 不计入 `nfe`；`nfe` 等于实际生成 token 的步数。
- `eos_token_id` 的停止条件是 `(next_token == eos_token_id).all()`，即 batch 内所有样本同时 EOS 才停。
- 输出 `output_ids` 包含 prompt，所以解码回复时要切掉 `prompt_ids.shape[1]:`。
- 该函数直接调用 `self.encoder`，绕开 `NemotronLabsDiffusionModel.forward()`，避免 diffusion loss/noising 相关逻辑。

## 6. Block-wise diffusion 解码：`generate()`

入口：`NemotronLabsDiffusionModel.generate()`。

调用方式示例：

```python
out_ids, nfe = model.generate(
    prompt_ids,
    max_new_tokens=512,
    block_length=32,
    threshold=0.9,
    eos_token_id=tokenizer.eos_token_id,
)
```

### 6.1 总体结构

`generate()` 是论文中 diffusion mode 的实现，但它是一个实用版 block-wise decoder：

1. 要求 `max_new_tokens % block_length == 0`。
2. `num_blocks = max_new_tokens // block_length`。
3. `steps_per_block = block_length`，即最多每个 token 一个 denoising step。
4. 如果 `causal_context=True`：
   - 先把 attention 设为 causal。
   - 对 prompt 做 prefill，得到 prefix KV cache。
   - 从 prompt 末尾 logits 采样一个 `next_token`，作为第一个 block 的第一个 seed token。
   - 再把 attention 切回 diffusion。
5. 逐 block 生成：
   - 构造 `[MASK] * block_length`。
   - 如果 `causal_context=True`，把 block 的第 0 位填成上一步 AR seed。
   - 对 block 反复 denoise，直到没有 mask 或达到步数。
   - block 完成后，用 causal forward 更新 prefix KV cache，并产生下一 block 的 seed。

### 6.2 Denoising loop

每个 block 内的 denoising 逻辑：

1. `mask_block_idx = x_accum[:, block_slice] == mask_id` 找出还没填的位置。
2. 如果没有 mask，结束当前 block。
3. `nfe += 1`。
4. 调用：

```python
logits_block = self(
    x_accum[:, block_slice],
    past_key_values=past_key_values,
    use_cache=False,
).logits
```

这一步的含义：

- 当前 block 的输入只传 block slice，不传完整 prefix。
- prefix 信息来自 `past_key_values`。
- `use_cache=False`，所以不会把未完成 block 写入 cache。
- attention 此时是 diffusion/bidirectional，因此 block 内 masked positions 可相互 attend。

5. 调 `_get_transfer_index(...)` 选择本轮提交哪些位置。
6. 把选中的位置写回 `x_accum[:, block_slice]`。
7. 如果 block 内出现 EOS，且 EOS 左边没有未填 mask，则提前结束该 block。

### 6.3 `_get_num_transfer_tokens()`

`_get_num_transfer_tokens(mask_index, steps)` 把当前 block 里的 mask 数平均分配到 `steps` 个 denoising step 上，余数前置。

例如一个 block 有 31 个 mask，`steps = 32`，则前 31 步每步 1 个，最后一步 0 个。实际 loop 会在 mask 清空后提前 break。

### 6.4 `_get_transfer_index()`

`_get_transfer_index(logits, temperature, mask_index, x, num_transfer_tokens, threshold)` 做两件事：

1. 产生候选 token：
   - `temperature == 0`：直接 argmax。
   - `temperature > 0`：先加 Gumbel-style noise，再 argmax。
2. 计算 confidence：
   - 对原始 logits softmax。
   - 取候选 token 的概率作为 confidence。
   - 非 mask 位置 confidence 设为 `-inf`。

选择策略分两种：

- `threshold is None`：
  - 使用 `_get_num_transfer_tokens()` 给出的固定预算。
  - 每步提交 top-k confidence 的 masked positions。
- `threshold is not None`：
  - 把本轮候选范围改成所有仍 masked positions。
  - top-1 总会保留，保证每步至少前进一个 token。
  - 从第 2 个候选开始，confidence 低于 threshold 的位置不提交。

因此，`threshold=0.9` 不是“所有低于 0.9 都不动”；最高置信位置仍会被提交，以免死循环。

### 6.5 Post-block causal forward

block denoise 完成后：

```python
if causal_context:
    _set_diffusion_lm(False)
output = self(
    x_accum[:, block_slice],
    past_key_values=past_key_values,
    use_cache=True,
    use_causal_mask=causal_context,
)
past_key_values = output.past_key_values
nfe += 1
```

这一步的作用：

- 把已经完成的 block 写入 KV cache。
- 如果 `causal_context=True`，用 causal logits 产生下一 block 的第一个 seed token。
- 这次 forward 计入 `nfe`，所以 diffusion mode 的 NFE = denoising forward 次数 + 每个 block 的 cache refresh forward 次数。

### 6.6 EOS 裁剪

每个 block 结束后，代码检查所有 batch 是否都已经出现 EOS：

- `gen_so_far = x_accum[:, prompt_len:]`
- 找每条样本第一个 EOS。
- 如果 batch 内每条样本都有 EOS，则裁剪到最晚那个 first EOS 后一位并返回。

## 7. Linear Self-Speculation：`linear_spec_generate()`

入口：`NemotronLabsDiffusionModel.linear_spec_generate()`。

调用方式示例：

```python
out_ids, nfe = model.linear_spec_generate(
    prompt_ids,
    max_new_tokens=512,
    block_length=32,
    eos_token_id=tokenizer.eos_token_id,
)
```

限制：

- 当前实现要求 `batch_size == 1`，否则直接抛 `ValueError`。
- 默认 `threshold = 0.0`，也就是 draft 阶段一次 forward 填满整个 block。

### 7.1 Prefill

Linear SS 先做一次 causal prefill：

1. `_set_diffusion_lm(False)`：AR attention。
2. `_toggle_adapters(False)`：LoRA 关闭。
3. 调 encoder：

```python
enc_out = self.encoder(
    input_ids=prompt_ids,
    past_key_values=DynamicCache(),
    use_cache=True,
    use_causal_mask=True,
)
```

4. 从最后 logits 采样一个 `next_token`。
5. `nfe = 1`。注意这里 prefill 计入 NFE，和 `ar_generate()` 的计数方式不同。
6. `generated = [next_token]`，但这个 seed token 还没有写入 KV cache；它会作为下一次 verify block 的第 0 个输入。

### 7.2 Draft phase

每轮 speculative cycle 先构造一个 block：

```python
block = torch.full((1, block_length), token_mask_id)
block[0, 0] = next_token.item()
```

含义：

- `block[0]` 是已经预测出来但尚未写入 cache 的 seed。
- `block[1:]` 是需要 diffusion drafter 并行起草的位置。

draft 阶段：

1. `_set_diffusion_lm(True)`：切到 bidirectional attention。
2. `_toggle_adapters(True)`：如果挂了 LoRA，则打开 adapter。
3. 调 encoder：

```python
enc_out = self.encoder(
    input_ids=block,
    past_key_values=past_key_values,
    use_cache=False,
)
```

4. `nfe += 1`。
5. `draft_logits = diffusion_head(hidden_states)`。
6. 采样或 argmax 得到 `draft_tokens`。
7. 如果 `threshold > 0`：
   - 只提交 confidence 大于等于 threshold 的 mask。
   - 如果一个都没有，强制提交最高 confidence 的位置，保证进展。
   - 因为可能只填一部分，所以 draft phase 会 while 循环多次。
8. 如果 `threshold == 0`：
   - 一次性填满所有 mask。
   - 退出 draft phase。

默认 `linear_spec` 的 `threshold=0`，所以正常是“一次 diffusion draft + 一次 AR verify”。

### 7.3 Verify phase

verify 阶段：

1. `_set_diffusion_lm(False)`：切回 causal attention。
2. `_toggle_adapters(False)`：关闭 LoRA，保持 AR verifier 不被 drafter LoRA 改变。
3. 调 encoder：

```python
enc_out = self.encoder(
    input_ids=block,
    past_key_values=past_key_values,
    use_cache=True,
    use_causal_mask=True,
)
```

4. 这次 forward 会把完整 `block` 暂时写入 KV cache。
5. `nfe += 1`。
6. `verify_logits` 经过 `diffusion_head` 得到 `ar_tokens`。

### 7.4 接受规则

代码中的接受规则：

```python
accepted = 0
for i in range(block_length - 1):
    if ar_tokens[0, i].item() == block[0, i + 1].item():
        accepted += 1
    else:
        break
accepted += 1
accepted_toks = ar_tokens[:, :accepted]
```

解释：

- causal logits 在位置 `i` 预测的是下一个 token。
- 因此用 `ar_tokens[i]` 和 draft 的 `block[i + 1]` 比较。
- 连续匹配多少个，就接受多少个 draft token。
- 最后 `accepted += 1` 是 speculative decoding 的 bonus token：第一个不匹配位置的 AR 预测本身也可以作为一个已经验证的 token。

示例：

- 如果 `ar_tokens[0] != block[1]`，说明第一个 draft token 就错了；仍接受 `ar_tokens[0]` 这个 bonus token。
- 如果前 5 个 draft token 都匹配，则接受这 5 个 draft token，再加 `ar_tokens[5]` 作为 bonus token。

### 7.5 Cache crop 的意义

verify forward 暂时把整个 block 写进了 cache，但实际只接受了一部分。因此需要：

```python
_crop_dynamic_cache(past_key_values, cache_len + accepted)
next_token = ar_tokens[:, accepted - 1 : accepted]
```

这里容易误解，关键是：

- `cache_len` 是 verify 前的已缓存 prefix 长度。
- verify 输入的 `block[0]` 是上轮留下的 seed，它在 `generated` 中已经存在，但还没进 cache。
- `accepted_toks` 是这轮新增的输出 token，其中最后一个 token 是 bonus token，还没有作为输入进入模型。
- 裁剪到 `cache_len + accepted`，保留的是 seed 和匹配的 draft prefix，不保留最后的 bonus token。
- `next_token` 设为最后的 bonus token，下一轮作为 `block[0]` 输入并写入 cache。

因此 cache 总是保存“已经作为输入验证过的前缀”，而 `next_token` 是“已生成但尚未写入 cache 的下一轮 seed”。

### 7.6 LoRA-aware 行为

`linear_spec_generate()` 内部有：

```python
def _toggle_adapters(enable: bool):
    for module in self.modules():
        if hasattr(module, "_disable_adapters"):
            module._disable_adapters = not enable
```

因此：

- draft phase：adapter ON。
- prefill / verify phase：adapter OFF。

这与论文附录 B 一致：LoRA 只增强 diffusion drafter，不改变 AR verifier。

本地 LoRA 配置：

- 目录：`/data1/linyewei/models/Nemotron-Labs-Diffusion-8B/linear_spec_lora`
- `target_modules = ["o_proj"]`
- `r = 128`
- `lora_alpha = 512`
- `inference_mode = true`

## 8. 三种入口脚本的直接调用链

### 8.1 `chat/`

`chat/chat.py` 的调用链：

```text
parse_args()
  -> load_model_and_tokenizer()
       -> AutoTokenizer.from_pretrained(..., trust_remote_code=True)
       -> AutoModel.from_pretrained(..., trust_remote_code=True)
       -> linear_spec_lora 时 PeftModel.from_pretrained(...).model
  -> tokenizer.apply_chat_template(...)
  -> generate()
       ar               -> model.ar_generate(...)
       dlm              -> model.generate(...)
       linear_spec      -> model.linear_spec_generate(...)
       linear_spec_lora -> model.linear_spec_generate(...)
```

几个单文件脚本只是上述逻辑的固定 mode 版本，便于快速 smoke test。

### 8.2 `evaluate.py`

`evaluate.py` 是最简单的评测入口：

```text
main()
  -> 解析 mode / tasks / lora / generation knobs
  -> load()
       -> AutoTokenizer
       -> AutoModel
       -> 可选 PeftModel.from_pretrained(...)
  -> run_one_task()
       -> datasets.load_dataset(...)
       -> tokenizer.apply_chat_template(...)
       -> generate()
            ar          -> model.ar_generate(...)
            dlm         -> model.generate(...)
            linear_spec -> model.linear_spec_generate(...)
       -> inline scorer
       -> 汇总 acc / avg_tok / avg_nfe / TPF
```

mode 默认值：

- `ar`：`block_length=1`，`threshold=None`
- `dlm`：`block_length=8`，`threshold=0.9`
- `linear_spec`：`block_length=32`，`threshold=0.0`

实现注意点：

- `evaluate.py` 只内置 `gsm8k` 和 `math-500` 两个任务。
- `generate()` 对 `max_new_tokens` 做 `_round_to_block()`，当前实现是向下取整到 block 的倍数，而服务端路径是向上取整；默认 512 不受影响。
- 轻量 `linear_spec` 路径没有显式传 `threshold`，因此使用模型方法默认 `threshold=0.0`。
- 如果传 `--lora` 或 `--lora-path`，代码用 PEFT 包模型后再 `m = wrapped.model`，这样可以直接访问底层 `linear_spec_generate()`。

## 9. 服务化评测调用链

大规模评测路径比 `evaluate.py` 多三层：`eval.sh` 编排、GPU worker server、NeMo-Skills client。

### 9.1 `eval.sh`

`eval.sh` 根据 `--mode` 设置默认参数：

| mode | server engine | generation algorithm | block length | threshold | native call |
| --- | --- | --- | --- | --- | --- |
| `ar` | `ar_native` | `ar_native` | 1 | 空 | `model.ar_generate()` |
| `dlm` | auto / `nemotron` | `nemotron` | 8 | 0.9 | `model.generate()` |
| `linear_spec` | auto / `nemotron` | `nemotron` | 32 | 0 | `model.linear_spec_generate()` |

然后导出环境变量，例如：

- `SERVER_MODEL_PATH`
- `SERVER_ENGINE`
- `SEQ_EVAL_GENERATION_ALGORITHM`
- `SEQ_EVAL_TOKENS_TO_GENERATE`
- `SEQ_EVAL_BLOCK_LENGTH`
- `SEQ_EVAL_THRESHOLD`
- `LINEAR_SPECULATION`
- `SERVER_LORA_PATH`
- `DRAFT_LORA_ONLY`

最后调用 `xp/examples/run_dlm_eval_pipeline_gpu_only.sh`。

### 9.2 GPU-only pipeline

`xp/examples/run_dlm_eval_pipeline_gpu_only.sh` 在一个 GPU SLURM allocation 内做：

1. 准备容器环境。
2. 如有 DCP checkpoint，转换或加载。
3. 每张 GPU 启一个 `dlm_batch_server.py` worker。
4. 启动 `dlm_load_balancer.py`。
5. 等待 `/health`。
6. 启动 `xp/nemo-skills/eval_dlm.py` 访问 load balancer。
7. 评测完成后清理进程。

worker 参数由 `SERVER_*`、`SEQ_EVAL_*`、`LINEAR_SPECULATION` 等环境变量拼出来。eval client 参数同理。

### 9.3 `eval_dlm.py`

`xp/nemo-skills/eval_dlm.py` 的核心作用是把 diffusion 参数塞进 OpenAI request 的 `extra_body`：

```text
++inference.extra_body.steps
++inference.extra_body.block_length
++inference.extra_body.threshold
++inference.extra_body.generation_algorithm
++inference.extra_body.linear_speculation
++inference.extra_body.draft_lora_only
++inference.extra_body.max_thinking_tokens
++inference.extra_body.sampler
++inference.extra_body.benchmark_name
```

这些字段会被 NeMo-Skills/OpenAI client 原样转发给 `dlm_batch_server.py` 的 Pydantic request model。

### 9.4 `dlm_batch_server.py`

worker server 的调用链：

```text
main()
  -> load_model_with_engine()
       -> get_default_algorithm_for_engine(engine)
       -> get_algorithm(...)
       -> algorithm.load_model_from_hf(...) 或 load_model_from_dcp(...)
       -> _load_other_algorithms_same_engine(...)
  -> uvicorn.run(app)

POST /v1/chat/completions
  -> batch_processor.add_request()
  -> BatchProcessor._process_batch()
  -> _process_batch_requests()
       -> 按 request.generation_algorithm 分组
  -> _process_algorithm_group()
       -> algorithm.tokenize_batch(...)
       -> 调 algorithm.generate(...)
       -> algorithm.decode_outputs(...)
       -> 记录 NFE
       -> 返回 OpenAI-compatible response
```

实现注意点：

- `ChatCompletionRequest` 定义在 `dlm_openai_server.py`，`model_config = {"extra": "allow"}`，因此未知字段可以被保留。
- batch 内会按 `generation_algorithm` 分组，但同一 algorithm group 内只使用第一条 request 的 generation config。异构参数混在同一 batch 时，后续 request 的 config 可能被忽略。
- `prompt_lengths` 当前统一使用 padded batch length；非 dInfer 路径的 usage 统计会把 padding 算入 prompt tokens。
- NFE 日志写到 `nfe_log.jsonl`，可带 `benchmark_name`，后续由 `add_nfe_to_metrics.py` 合并进 metrics。

## 10. 生成算法注册表

`xp/dlm_api/dlm_generate/__init__.py` 注册了三个算法实例：

- `NemotronGeneration()`：
  - name: `nemotron`
  - engine: `nemotron`
  - aliases: `nemotron_native`、`nemotron_diffusion`
- `NemotronMixedGeneration()`：
  - name: `nemotron_mixed`
  - engine: `nemotron`
  - aliases: `mix_ar_dlm`
- `ArNativeGeneration()`：
  - name: `ar_native`
  - engine: `ar_native`
  - aliases: `ar-native`

默认 engine 映射：

- `engine="nemotron"` -> `algorithm="nemotron"`
- `engine="ar_native"` -> `algorithm="ar_native"`

## 11. `GenerationAlgorithm` 基类

文件：`xp/dlm_api/dlm_generate/base.py`。

职责：

- 保存模型、tokenizer、device、config。
- 统一 HF / DCP 模型加载。
- 处理 `trust_remote_code=True`。
- 可选挂载 PEFT LoRA。
- 设置 pad token。
- 处理 chat template。
- 对 batch 做 left padding。
- 提供 `decode_outputs()`。

几个重要实现点：

### 11.1 pad token 选择

如果 tokenizer 没有 `pad_token_id`：

- AR mode / `shift_logits=True`：用 `eos_token_id` 做 pad，避免 mask token 的 embedding 污染 causal attention。
- diffusion mode：优先用 `mask_token_id` 做 pad。
- 如果没有 mask id，则 fallback 到 EOS。

### 11.2 LoRA 挂载

`_apply_lora_if_configured()` 使用：

```python
from peft import PeftModel
self.model = PeftModel.from_pretrained(self.model, self.lora_path).eval()
```

后续 `NemotronGeneration` 会 unwrap PEFT wrapper 找到底层 native model。

### 11.3 DCP 加载

`load_model_from_dcp()` 支持：

- structured DCP：`weights/` + `tokenizer/`
- legacy DCP
- 可先加载 base HF 模型，再用 NeMo-RL checkpoint loader 覆盖权重

这部分主要服务训练中间 checkpoint 的评测。

## 12. `NemotronGeneration`

文件：`xp/dlm_api/dlm_generate/nemotron.py`。

这是服务路径中 `dlm` 和 `linear_spec` 的核心包装类。

### 12.1 加载模型

`load_model_class()`：

1. 读取环境变量 `NEMOTRON_DLM_PARADIGM`，默认 `bidirectional`。
2. `AutoConfig.from_pretrained(..., trust_remote_code=True)`。
3. 如果 CLI 传了 `--max-position-embeddings`，覆盖 `config.max_position_embeddings`。
4. 设置 `config.dlm_paradigm = dlm_paradigm`。
5. `AutoModel.from_pretrained(..., config=config, trust_remote_code=True)`。

这意味着服务层可以在加载时改变模型构造使用的 `dlm_paradigm`，但实际三种生成方法仍会在运行时显式切 `diffusion_lm`。

### 12.2 生成分发

`NemotronGeneration.generate()` 先归一化参数，再决定走哪条 native method：

- 如果 `linear_speculation` 为 truthy：
  - 默认方法名是 `linear_spec_generate`。
  - 如果 `draft_lora_only=True` 且 native model 有 `linear_spec_generate_lora`，则优先调用它。
  - 当前本地模型没有 `linear_spec_generate_lora`，所以会调用统一的 `linear_spec_generate()`。
- 否则：
  - 调 `native_model.generate()`。

### 12.3 `_filter_kwargs_for_signature()`

包装层会构造比较多候选参数，例如：

- `steps`
- `block_length`
- `threshold`
- `shift_logits`
- `temperature`
- `causal_context`
- `sampler`
- `end_think_token_id`
- `max_thinking_tokens`

但模型远程代码的签名可能随版本变化，因此 `_filter_kwargs_for_signature()` 会检查目标函数签名，把不支持的 kwargs 自动丢掉。

对当前本地模型很重要：

- `NemotronLabsDiffusionModel.generate()` 没有 `steps` 参数，所以服务里传的 `steps` 会被过滤掉；真实 steps 是 native code 内部的 `steps_per_block = block_length`。
- 当前 native `generate()` 也没有 `sampler` 参数，所以 `sampler` 会被过滤掉。

### 12.4 CLI-level sampler 注意点

`dlm_batch_server.py` 定义了 `--sampler` 并打印日志，但当前 `main()` 调 `load_model_with_engine(...)` 时没有传 `sampler=args.sampler`。因此 worker 启动级别的 sampler 默认值不会进入 algorithm。

不过 request 级别的 `extra_body.sampler` 仍会进入 `_process_algorithm_group()` 的 config，再传给 `NemotronGeneration.generate()`；只是当前 native `generate()` 签名不接收 `sampler`，最终也会被过滤。

## 13. `ArNativeGeneration`

文件：`xp/dlm_api/dlm_generate/ar_native.py`。

职责很直接：

- `load_model_class()` 用 `AutoModel.from_pretrained(..., trust_remote_code=True)` 加载模型。
- 检查模型是否有 `ar_generate()`。
- `generate()` 组装参数后调用：

```python
output_ids, nfe = model.ar_generate(
    prompt_ids=prompt,
    max_new_tokens=gen_length,
    temperature=temperature,
    eos_token_id=eos_token_id,
    end_think_token_id=end_think_token_id,
    max_thinking_tokens=max_thinking_tokens,
)
```

注意：

- `block_length`、`steps`、`threshold` 对 AR native 不起作用。
- 是否传 EOS 取决于 server 启动时 `--eos-early-stop`。

## 14. `NemotronMixedGeneration`

文件：`xp/dlm_api/dlm_generate/nemotron_mixed.py`。

这个类用于 AR+dLM logits mixing：

- 如果 `ar_weight > 0` 且模型有 `generate_mixed()`，调用 `_generate_mixed()`。
- 否则 fallback 到 `NemotronGeneration.generate()`。

加载逻辑支持环境变量 `NEMOTRON_CODE_REPO`：

- code repo 提供 `configuration_ministral_dlm.MinistralDLMConfig` 和 `modeling_ministral_dlm.MinistralDiffEncoderModel`。
- 权重仍来自标准 `model_path`。

当前本地模型远程代码没有 `generate_mixed()`，所以这个路径不是主线。

## 15. Utility 文件的当前作用

### 15.1 `xp/dlm_api/dlm_generate/utils/sampler.py`

该文件定义了多种 confidence / sampling 策略：

- `low_confidence`
- `high_confidence`
- `top_p_margin`
- `random`
- `fixed`
- `confidence_threshold_ref`
- `confidence_threshold_bound`
- `confidence_threshold`
- `cumulative_error`

但当前原生 Nemotron decode 不调用这个文件；模型自己的 `_get_transfer_index()` 在 `modeling_nemotron_labs_diffusion.py` 中。

另外，该文件里 `get_transfer_index()` 定义了两次，后面的定义会覆盖前面的定义。它更像从通用 DLM/Fast-dLLM 路径保留下来的辅助实现。

### 15.2 `xp/dlm_api/dlm_generate/utils/eos_detect.py`

提供 block 内 EOS 检测的多个版本，包括 torch.compile 尝试。当前本地模型原生 `generate()` 自己在函数内做 EOS 检查，没有直接调用这里。

### 15.3 `BatchStaticCache.py`

扩展 HuggingFace `StaticCache`，支持 `[batch_size, seq_len]` 的 per-batch cache positions。当前主线 native decode 使用 `DynamicCache`，没有直接使用该类。

### 15.4 `utils/sliding_window.py`

这是一个滑窗 cache 更新代码片段，不是完整函数，当前没有被主线 import。

## 16. SGLang Spark 路径

`sglang_spark/` 是另一个 deployment path：

- 使用外部 fork：`hutm/sglang @ upstream/2-dllm-lora-ar`
- launcher：`sglang_spark/launch_server.sh`
- 通过 `--dllm-algorithm` 选择：
  - `LinearSpec`
  - `LinearSpec-base`
  - `FastDiffuser`
  - `AR`

几个映射：

- `ALGO=LinearSpec`：生成一个临时 YAML，指定 LoRA path 和 `lora_mode`。
- `ALGO=LinearSpec-base`：不使用 LoRA 的 LinearSpec。
- `ALGO=FastDiffuser`：纯 diffusion denoising。
- `ALGO=AR`：通过 `MODEL_OVERRIDE='{"ar_mode": true}'` 强制模型 attention causal，然后 SGLang 仍以 FastDiffuser 框架启动。

这条路径的核心 scheduler/kernel 实现在 SGLang fork 里，不在当前仓库。

## 17. 参数速查

### `block_length`

- diffusion mode：每个 denoising block 的长度。
- linear_spec：每轮 draft/verify 的 speculative width 加 seed 布局的 block 长度。
- AR mode：无实际意义，通常设为 1。

### `threshold`

- native `generate()`：
  - `None`：固定 top-k 进度，按 `_get_num_transfer_tokens()` 分配。
  - 非 `None`：按 confidence threshold 提交多个位置，但 top-1 永远提交。
- native `linear_spec_generate()`：
  - `0.0`：draft 一次填满 block。
  - `>0`：draft 阶段可能多轮，只提交高置信位置，且每轮强制至少提交一个。

### `steps`

- 服务和评测脚本会传 `steps`。
- 当前本地 native `generate()` 签名不接收 `steps`，包装层会过滤。
- 实际 diffusion steps per block 固定为 `block_length`。

### `temperature`

- AR：`temperature > 0` 时 softmax multinomial；否则 argmax。
- diffusion：`_get_transfer_index()` 里 `temperature > 0` 会用 Gumbel-style noise 后 argmax。
- linear_spec：draft 和 verify 都支持 `temperature > 0` 的 multinomial。

### `causal_context`

- native `generate()` 默认 `True`。
- `True` 时 block 间用 causal KV cache 串起来，并为每个 block 产生第一个 seed token。
- 这是论文中 block-wise diffusion “clean prefix conditioning / causal across blocks”的推理对应。

### `max_thinking_tokens` 和 `end_think_token_id`

- 包装层会用 tokenizer 编码 `</think>`，取最后一个 id 作为 `end_think_token_id`。
- AR 和 Linear SS 超过预算会强制下一 token 为 `</think>`。
- diffusion mode 会在跨过预算的 block 内注入 `</think>`。

### `eos_token_id`

- 直接 chat/evaluate 路径通常显式传 tokenizer EOS。
- 服务路径只有启动 worker 时带 `--eos-early-stop` 才会从 tokenizer 取 EOS 并传给 native method。

### `linear_speculation`

- 在服务路径中，`extra_body.linear_speculation=true` 或 worker 启动 `--linear-speculation` 会让 `NemotronGeneration` 调 `linear_spec_generate()`。
- `eval.sh --mode linear_spec` 会设置 `LINEAR_SPECULATION=true`。

### `draft_lora_only`

- 包装层含义：如果存在 `linear_spec_generate_lora()`，优先走该专用方法。
- 当前本地模型没有该方法；实际 LoRA 控制由统一 `linear_spec_generate()` 内部 `_toggle_adapters()` 完成。

## 18. 典型调用链总览

### 18.1 直接 AR 聊天

```text
chat/chat_ar.py
  -> AutoModel.from_pretrained(model_dir, trust_remote_code=True)
  -> tokenizer.apply_chat_template(...)
  -> model.ar_generate(prompt_ids, max_new_tokens=512)
       -> set diffusion_lm=False
       -> encoder causal prefill with DynamicCache
       -> token-by-token causal decode
       -> diffusion_head
```

### 18.2 直接 dLLM 聊天

```text
chat/chat_dlm.py
  -> model.generate(prompt_ids, block_length=32, threshold=0.9, eos_token_id=...)
       -> optional causal prompt prefill
       -> per block create mask block
       -> repeated diffusion forward with use_cache=False
       -> confidence-based transfer
       -> causal post-block cache refresh
```

### 18.3 Linear SS + LoRA 聊天

```text
chat/chat_linear_spec_lora.py
  -> AutoModel
  -> PeftModel.from_pretrained(model, model_dir, subfolder="linear_spec_lora")
  -> unwrap: model = model.model
  -> model.linear_spec_generate(...)
       -> causal prefill, adapter OFF
       -> diffusion draft, adapter ON
       -> causal verify, adapter OFF
       -> accept prefix + one bonus token
       -> crop cache
```

### 18.4 `eval.sh --mode dlm`

```text
eval.sh
  -> export SERVER_ENGINE unset/auto, SEQ_EVAL_GENERATION_ALGORITHM=nemotron
  -> run_dlm_eval_pipeline_gpu_only.sh
       -> start N workers: dlm_batch_server.py --engine nemotron
       -> start dlm_load_balancer.py
       -> eval_dlm.py --generation-algorithm nemotron
            -> OpenAI extra_body.block_length / threshold / steps
            -> POST /v1/chat/completions
                 -> BatchProcessor
                 -> NemotronGeneration.generate()
                 -> native_model.generate()
```

### 18.5 `eval.sh --mode linear_spec --lora`

```text
eval.sh
  -> LINEAR_SPECULATION=true
  -> SERVER_LORA_PATH=<adapter dir>
  -> run_dlm_eval_pipeline_gpu_only.sh
       -> worker: dlm_batch_server.py --linear-speculation --lora-path ...
       -> GenerationAlgorithm._apply_lora_if_configured()
       -> NemotronGeneration.generate()
            -> resolve_linear_speculation_mode(...)
            -> unwrap PEFT native model
            -> native_model.linear_spec_generate(...)
```

## 19. 当前代码与论文的差异点

1. 论文讨论的 quadratic self-speculation 没有在当前本地模型代码和本仓库入口中实现。
2. 论文附录的 trained sampler 没有作为模型目录中的独立 sampler 模块或 checkpoint 出现；当前 native `generate()` 也没有 `sampler` 参数。
3. 论文 SOL 分析算法没有在仓库中提供脚本。
4. 服务路径保留了 `steps` 参数，但当前 native `generate()` 不使用它。
5. `xp/dlm_api/dlm_generate/utils/sampler.py` 和 `eos_detect.py` 更像通用 DLM/历史代码，不是当前 NLD native decode 主线。
6. `evaluate.py` 是轻量评测，不代表论文全 benchmark 的评测实现；完整 10 benchmark 走 `eval.sh` + NeMo-Skills。
7. `linear_spec_generate()` 当前只支持 batch size 1；服务层 batch 到同一 worker 后如果实际调用 linear spec，仍要求 native method 接收的 prompt batch 为 1，否则会报错。因此 Linear SS 更适合低并发/单流场景。

## 20. 建议阅读顺序

如果要从代码层面掌握整个项目，建议按下面顺序读：

1. `README.md` 的 Modes 和 Quick start，先建立入口印象。
2. 模型目录 `config.json`，确认模型架构、mask id、EOS id、block size。
3. `modeling_ministral.py`：
   - `Ministral3Attention.forward()`
   - `Ministral3Model.forward()`
4. `modeling_nemotron_labs_diffusion.py`：
   - `NemotronLabsDiffusionModel.__init__()`
   - `forward_process()`
   - `forward()`
   - `generate()`
   - `ar_generate()`
   - `linear_spec_generate()`
   - `_get_transfer_index()`
5. `chat/chat.py` 和几个单模式 chat 脚本，理解最短路径。
6. `evaluate.py`，理解单进程评测如何调 native methods。
7. `xp/dlm_api/dlm_generate/base.py` 和 `nemotron.py`，理解服务包装层如何加载、过滤 kwargs、分发到 native methods。
8. `xp/dlm_api/dlm_batch_server.py`，理解 OpenAI request 到 algorithm.generate 的 batch 调用链。
9. `eval.sh`、`xp/examples/run_dlm_eval_pipeline_gpu_only.sh`、`xp/nemo-skills/eval_dlm.py`，理解大规模评测如何把 shell 参数变成 `extra_body`。
10. `sglang_spark/README.md` 和 `launch_server.sh`，仅在需要 SGLang 部署时阅读。

## 21. 一句话总结

Nemotron-Labs-Diffusion 的代码实现可以理解为：同一个 Ministral-style decoder backbone，在 `diffusion_lm=True` 时作为 bidirectional denoiser，在 `diffusion_lm=False` 时作为 causal LM；`generate()` 用 denoiser 并行填 mask 并用 causal post-block forward 维护 prefix cache，`ar_generate()` 只走 causal token-by-token，`linear_spec_generate()` 则先用 denoiser 起草一个 block，再用同一个模型的 causal logits 验证并裁剪 cache，从而把论文中的 AR、diffusion、self-speculation 三种模式落到同一套权重和同一套 attention 切换机制上。
