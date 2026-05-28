"""Single-turn chat in Linear Self-Speculation mode with the bundled LoRA
adapter attached as the diffusion draft. Mirrors the HF README snippet.

The adapter lives in the model repo at subfolder `linear_spec_lora/`. PEFT
attaches it to o_proj; linear_spec_generate toggles the adapter ON during
the bidirectional draft phase and OFF during the causal verify phase, so
LoRA only specializes the draft — AR semantics are preserved.

Requires `peft` (`pip install peft`).
"""
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

REPO = "nvidia/Nemotron-Labs-Diffusion-8B"

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).cuda().to(torch.bfloat16)

# Attach the linear_spec_lora adapter bundled in the model repo.
model = PeftModel.from_pretrained(model, REPO, subfolder="linear_spec_lora").eval()
# Unwrap the PeftModel so we can call linear_spec_generate directly.
model = model.model

user_input = input("User: ").strip()
history = [{"role": "user", "content": user_input}]
prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

out_ids, nfe = model.linear_spec_generate(
    prompt_ids,
    max_new_tokens=512,
    block_length=32,
    eos_token_id=tokenizer.eos_token_id,
)
reply = tokenizer.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
print(f"Model: {reply}")
print(f"[NFE={nfe}]")
