"""Interactive chat with a Nemotron-Labs diffusion-LM, mirroring the snippets
in the HF model card (https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B).

Usage:
    python chat.py --mode {ar|dlm|linear_spec|linear_spec_lora}
                   [--model nvidia/Nemotron-Labs-Diffusion-8B]
                   [--max-new-tokens 512]
                   [--block-length 32]
                   [--threshold 0.9]

The four modes mirror the model's four decoding paths:

    ar              -> model.ar_generate(...)
    dlm             -> model.generate(...)            block-diffusion sampling
    linear_spec     -> model.linear_spec_generate(...)  no LoRA
    linear_spec_lora-> model.linear_spec_generate(...) with the bundled
                       `linear_spec_lora/` PEFT adapter attached as the draft

After load, type a message and press Enter. Type :q or Ctrl-C to quit. The
multi-turn chat history is fed back to the model via the chat template each
turn — same behaviour as the HF README snippet.
"""

import argparse
import sys
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["ar", "dlm", "linear_spec", "linear_spec_lora"])
    p.add_argument("--model", default="nvidia/Nemotron-Labs-Diffusion-8B")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--block-length", type=int, default=32, help="dLM / linear_spec block size")
    p.add_argument("--threshold", type=float, default=0.9, help="dLM confidence threshold")
    p.add_argument("--lora-subfolder", default="linear_spec_lora",
                   help="subfolder under --model that holds the PEFT adapter (linear_spec_lora mode)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(args.device).to(dtype)

    # linear_spec_lora: attach the bundled draft-side LoRA adapter and unwrap
    # the PeftModel so we can call linear_spec_generate directly (the method
    # toggles the adapter internally between the diffusion draft and the
    # causal verify pass).
    if args.mode == "linear_spec_lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model, subfolder=args.lora_subfolder).eval()
        model = model.model

    return model, tokenizer


def generate(args, model, tokenizer, prompt_ids: torch.Tensor):
    """Dispatch to the right per-mode call. Returns (out_ids, nfe)."""
    eos = tokenizer.eos_token_id
    if args.mode == "ar":
        return model.ar_generate(prompt_ids, max_new_tokens=args.max_new_tokens)
    if args.mode == "dlm":
        return model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            threshold=args.threshold,
            eos_token_id=eos,
        )
    # linear_spec / linear_spec_lora share the call path; the LoRA attach has
    # already happened in load_model_and_tokenizer if needed.
    return model.linear_spec_generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        eos_token_id=eos,
    )


def main():
    args = parse_args()
    print(f"Loading {args.model} ({args.mode})...", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(args)
    print(f"Ready. Type :q to quit.\n", file=sys.stderr)

    history = []
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input in {":q", ":quit", "exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        out_ids, nfe = generate(args, model, tokenizer, prompt_ids)
        new_tokens = out_ids[0, prompt_ids.shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        history.append({"role": "assistant", "content": reply})
        print(f"Model [{args.mode}, NFE={nfe}]: {reply}\n")


if __name__ == "__main__":
    main()
