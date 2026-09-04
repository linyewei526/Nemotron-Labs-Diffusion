#!/usr/bin/env python3
"""One-file evaluator for Nemotron-Labs-Diffusion.

No server. No SLURM. No eval-framework dependency. One Python process:
    1. Load the HF model + tokenizer once.
    2. Iterate over benchmark datasets via `datasets.load_dataset`.
    3. Call the right `model.X_generate` for the chosen --mode.
    4. Score with an inline task-specific extractor.
    5. Print a per-task pass@1 + TPF table.

    pip install torch transformers datasets peft         # peft only for --lora

    # Smoke (50 problems, ~5 min on 1× H100)
    python evaluate.py --mode dlm --tasks gsm8k --limit 50

    # Full gsm8k (1319 problems), each mode
    python evaluate.py --mode ar           --tasks gsm8k
    python evaluate.py --mode dlm          --tasks gsm8k
    python evaluate.py --mode linear_spec  --tasks gsm8k
    python evaluate.py --mode linear_spec  --tasks gsm8k --lora       # + bundled LoRA draft

    # Multiple tasks in one run
    python evaluate.py --mode dlm --tasks gsm8k,math-500

Supported tasks (extend with TASKS dict below):
    gsm8k     — GSM8K test split, 1319 problems. Score: \\boxed{N} or last
                number in model output equals the gold answer.
    math-500  — Hendrycks MATH-500 test split. Score: \\boxed{N} equality
                with the gold answer.

For the full 10-benchmark suite (HumanEval / MBPP / MMLU / IFEval /
LiveCodeBench / AIME / GPQA — each needs its own scorer) use eval.sh.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# Heavy imports (torch, transformers, datasets, peft) are deferred into
# load() / run_one_task() so `python evaluate.py --help` works in a fresh
# clone before users have run pip install.

# ─── Per-mode decoding defaults (mirror eval.sh) ────────────────────────────

MODE_DEFAULTS = {
    "ar":          dict(block_length=1,  threshold=None),
    "dlm":         dict(block_length=8,  threshold=0.9),
    "linear_spec": dict(block_length=32, threshold=0.0),
}


# ─── Inline scorers ─────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _last_number(text: str) -> Optional[str]:
    """Pull the last number-like token from `text`. Strips commas."""
    cleaned = text.replace(",", "")
    matches = _NUMBER_RE.findall(cleaned)
    return matches[-1] if matches else None


def _boxed_answer(text: str) -> Optional[str]:
    """Return the last `\\boxed{...}` payload, or None."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _numbers_equal(a: str, b: str) -> bool:
    """Float-aware equality (so '18', '18.0', and '18.00' all match)."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a.strip() == b.strip()


def score_gsm8k(model_out: str, gold: str) -> bool:
    pred = _boxed_answer(model_out) or _last_number(model_out)
    return pred is not None and _numbers_equal(pred, gold)


def score_math500(model_out: str, gold: str) -> bool:
    # MATH gold is the literal contents of \boxed{...} in the solution.
    pred = _boxed_answer(model_out)
    if pred is None:
        return False
    # Normalize whitespace + strip surrounding $$.
    norm = lambda s: re.sub(r"\s+", "", s.strip().strip("$"))
    return norm(pred) == norm(gold) or _numbers_equal(pred, gold)


# ─── Task registry ──────────────────────────────────────────────────────────

@dataclass
class Task:
    name: str
    hf_dataset: str         # `datasets.load_dataset` repo id
    hf_split: str           # which split to score
    question_field: str     # column with the problem statement
    gold_extractor: Callable[[dict], str]  # row -> gold answer string
    scorer: Callable[[str, str], bool]
    instruction: str        # appended in front of the question


TASKS = {
    "gsm8k": Task(
        name="gsm8k",
        hf_dataset="gsm8k",
        hf_split="test",
        question_field="question",
        gold_extractor=lambda row: row["answer"].split("####")[-1].strip().replace(",", ""),
        scorer=score_gsm8k,
        instruction=(
            "Solve the following math problem. Put the final numerical answer "
            "inside \\boxed{} at the very end.\n\n"
        ),
    ),
    "math-500": Task(
        name="math-500",
        hf_dataset="HuggingFaceH4/MATH-500",
        hf_split="test",
        question_field="problem",
        gold_extractor=lambda row: row["answer"],
        scorer=score_math500,
        instruction=(
            "Solve the following math problem. Put the final answer inside "
            "\\boxed{} at the very end.\n\n"
        ),
    ),
}


# ─── Generation dispatch ────────────────────────────────────────────────────

def _round_to_block(n: int, block: int) -> int:
    return max(block, (n // block) * block)


def generate(model, tokenizer, prompt_ids, mode: str, max_new_tokens: int,
             block_length: int, threshold: Optional[float],
             max_thinking_tokens: int) -> tuple:
    """Dispatch to the right `model.X_generate` for the chosen mode.
    Returns (output_ids, nfe)."""
    eos = tokenizer.eos_token_id
    if mode == "ar":
        return model.ar_generate(
            prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos,
        )
    if mode == "dlm":
        n = _round_to_block(max_new_tokens, block_length)
        return model.generate(
            prompt_ids, max_new_tokens=n, block_length=block_length,
            threshold=threshold, eos_token_id=eos,
            max_thinking_tokens=max_thinking_tokens,
        )
    if mode == "linear_spec":
        n = _round_to_block(max_new_tokens, block_length)
        return model.linear_spec_generate(
            prompt_ids, max_new_tokens=n, block_length=block_length,
            eos_token_id=eos, max_thinking_tokens=max_thinking_tokens,
        )
    raise ValueError(f"unknown mode {mode!r}")


def run_one_task(model, tokenizer, task: Task, args) -> dict:
    from datasets import load_dataset

    print(f"\n── {task.name} ── loading {task.hf_dataset} [{task.hf_split}]", flush=True)
    if task.hf_dataset == "gsm8k":
        ds = load_dataset(task.hf_dataset, "main", split=task.hf_split)
    else:
        ds = load_dataset(task.hf_dataset, split=task.hf_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0
    total_new_tokens = 0
    total_decode_nfe = 0
    total_model_nfe = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        question = row[task.question_field]
        gold = task.gold_extractor(row)

        messages = [{"role": "user", "content": task.instruction + question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        out_ids, nfe = generate(
            model, tokenizer, prompt_ids,
            mode=args.mode, max_new_tokens=args.max_new_tokens,
            block_length=args.block_length, threshold=args.threshold,
            max_thinking_tokens=args.max_thinking_tokens,
        )
        new_ids = out_ids[0, prompt_ids.shape[1]:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        ok = task.scorer(new_text, str(gold))
        correct += int(ok)
        total += 1
        total_new_tokens += int(new_ids.numel())
        model_nfe = int(nfe) if isinstance(nfe, (int, float)) else 0
        # HF LinearSpec includes one causal prompt prefill; SGLang's TPF starts
        # at draft/verify decode, so exclude that prefill here as well.  The HF
        # AR and block-diffusion methods already match their SGLang NFE paths.
        prefill_nfe = 1 if args.mode == "linear_spec" and model_nfe > 0 else 0
        decode_nfe = max(model_nfe - prefill_nfe, 0)
        total_model_nfe += model_nfe
        total_decode_nfe += decode_nfe

        if (i + 1) % args.print_every == 0:
            acc = 100.0 * correct / total
            tpf = total_new_tokens / max(total_decode_nfe, 1)
            elapsed = time.time() - t0
            print(f"  [{i+1:5d}/{len(ds)}]  acc={acc:5.2f}%  "
                  f"avg_tok={total_new_tokens/total:6.1f}  "
                  f"avg_nfe={total_decode_nfe/total:6.1f}  "
                  f"TPF={tpf:5.2f}  ({elapsed:.0f}s)", flush=True)

    acc = 100.0 * correct / max(total, 1)
    avg_tok = total_new_tokens / max(total, 1)
    avg_nfe = total_decode_nfe / max(total, 1)
    avg_total_nfe = total_model_nfe / max(total, 1)
    tpf = total_new_tokens / max(total_decode_nfe, 1)
    print(f"  ✓ {task.name:<12} acc={acc:5.2f}%  avg_tok={avg_tok:6.1f}  "
          f"avg_nfe={avg_nfe:6.1f}  TPF={tpf:5.2f}  ({total} problems)", flush=True)
    return dict(task=task.name, num_entries=total, accuracy=acc,
                avg_tokens=avg_tok, avg_nfe=avg_nfe,
                avg_total_nfe=avg_total_nfe, tpf=tpf,
                elapsed_seconds=time.time() - t0)


# ─── Model load ─────────────────────────────────────────────────────────────

def load(args) -> tuple:
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading {args.model} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    m = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).to(dtype)

    if args.lora or args.lora_path:
        from peft import PeftModel
        lora_dir = args.lora_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "miscs", "linear_spec_lora")
        if not os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
            sys.exit(f"ERROR: LoRA adapter_config.json not found at {lora_dir}. "
                     f"Run `bash scripts/fetch_bundled_lora.sh` first.")
        print(f"Attaching LoRA from {lora_dir}", file=sys.stderr, flush=True)
        wrapped = PeftModel.from_pretrained(m, lora_dir).eval()
        m = wrapped.model  # unwrap so .linear_spec_generate is reachable

    if args.mode != "linear_spec" and (args.lora or args.lora_path):
        print(f"WARNING: --lora ignored — only meaningful for --mode linear_spec",
              file=sys.stderr)

    m.eval()
    return m, tok


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B",
                   help="HuggingFace model id (default: %(default)s)")
    p.add_argument("--mode", default="dlm", choices=list(MODE_DEFAULTS.keys()),
                   help="Decoding path: ar | dlm | linear_spec")
    p.add_argument("--tasks", default="gsm8k",
                   help=f"Comma-separated task names. Available: {','.join(TASKS.keys())}")
    p.add_argument("--lora", action="store_true",
                   help="(linear_spec only) attach the bundled miscs/linear_spec_lora/ as the draft")
    p.add_argument("--lora-path", default=None,
                   help="Local directory containing adapter_config.json + adapter_model.safetensors")
    p.add_argument("--limit", type=int, default=None, help="Cap problems per task (smoke testing)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--block-length", type=int, default=None,
                   help="Override per-mode default block_length")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override per-mode default confidence threshold (dlm/linear_spec)")
    p.add_argument("--max-thinking-tokens", type=int, default=6000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--print-every", type=int, default=100, help="Progress every N problems")
    p.add_argument("--output", default=None,
                   help="If set, write per-task results to this JSON file")
    args = p.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    if args.block_length is None:
        args.block_length = defaults["block_length"]
    if args.threshold is None:
        args.threshold = defaults["threshold"]

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(task_names) - TASKS.keys()
    if unknown:
        sys.exit(f"ERROR: unknown task(s) {sorted(unknown)}. "
                 f"Available: {sorted(TASKS.keys())}. For the full 10-benchmark "
                 f"suite, use eval.sh.")

    m, tok = load(args)
    results = [run_one_task(m, tok, TASKS[name], args) for name in task_names]

    # Summary
    print("\n── summary ──")
    print(f"  mode={args.mode}  lora={'on' if (args.lora or args.lora_path) else 'off'}  "
          f"model={args.model}")
    print(f"  {'task':<14} {'acc%':>7} {'avg_tok':>8} {'avg_nfe':>8} {'TPF':>6}")
    for r in results:
        print(f"  {r['task']:<14} {r['accuracy']:>7.2f} {r['avg_tokens']:>8.1f} "
              f"{r['avg_nfe']:>8.1f} {r['tpf']:>6.2f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
