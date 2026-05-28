"""Single-turn chat in Linear Self-Speculation mode. Mirrors the HF README snippet.

linear_spec_generate draws a diffusion draft under bidirectional attention,
then verifies it autoregressively, accepting the longest matching prefix plus
one bonus token per iteration. No LoRA — see chat_linear_spec_lora.py for the
LoRA-enhanced draft variant.
"""
import torch
from transformers import AutoModel, AutoTokenizer

REPO = "nvidia/Nemotron-Labs-Diffusion-8B"

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).cuda().to(torch.bfloat16)

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
