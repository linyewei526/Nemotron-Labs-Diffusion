"""Single-turn chat in dLM mode (block-diffusion sampling). Mirrors the HF README snippet."""
import torch
from transformers import AutoModel, AutoTokenizer

REPO = "nvidia/Nemotron-Labs-Diffusion-8B"

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).cuda().to(torch.bfloat16)

user_input = input("User: ").strip()
history = [{"role": "user", "content": user_input}]
prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

out_ids, nfe = model.generate(
    prompt_ids,
    max_new_tokens=512,
    block_length=32,
    threshold=0.9,
    eos_token_id=tokenizer.eos_token_id,
)
reply = tokenizer.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
print(f"Model: {reply}")
print(f"[NFE={nfe}]")
