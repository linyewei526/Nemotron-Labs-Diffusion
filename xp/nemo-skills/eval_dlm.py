#!/usr/bin/env python3
"""NeMo-Skills eval client, with diffusion-specific glue on top.

The upstream NeMo-Skills runner only knows how to talk to a vanilla
OpenAI chat-completions endpoint. The four decoding modes of these
diffusion models (dlm / ar_native / linear_spec) each need
extra knobs — diffusion steps, block length, confidence threshold,
linear / self-speculation toggles, AR-mix weight, max-thinking budget
— that aren't part of the OpenAI schema.

We piggy-back on NeMo-Skills' "++inference.extra_body.<key>=<value>"
mechanism: any key/value pair set there is forwarded into the OpenAI
request's extra_body, where dlm_batch_server.py unpacks it and routes
to the right algorithm in dlm_generate/.

Diffusion-specific CLI flags added on top of stock NeMo-Skills:
- --generation-algorithm: selects nemotron / nemotron_mixed / ar_native,
  dispatching to the corresponding algorithm class.
- --steps / --block-length / --threshold / --factor / --cfg-scale /
  --remasking: diffusion sampler knobs.
- --linear-speculation / --draft-lora-only: linear self-spec toggles.
- --ar-weight: AR/dLM logit mixing weight for nemotron_mixed.
- --max-thinking-tokens: hard cap on pre-</think> tokens.
- --shift-logits: AR vs diffusion logit shift; auto-defaults from
  generation_algorithm.

Usage:
    # Default GSM8K evaluation (DLM)
    python eval_dlm.py
    
    # Evaluate with Nemotron model
    python eval_dlm.py --generation-algorithm nemotron --model nemotron-labs-diffusion-8b
    
    # Evaluate on a different benchmark
    python eval_dlm.py --benchmark math:2
    
    # Quick test mode
    python eval_dlm.py --quick-test
    
    # Custom settings
    python eval_dlm.py --benchmark gsm8k:1 --temperature 0.8 --max-samples 100
    
    # Different server/model
    python eval_dlm.py --server-address http://my-server:8080/v1 --model my-model
    
    # DLM model with custom settings
    python eval_dlm.py --generation-algorithm dual_cache --steps 128 --cfg-scale 1.5 --remasking random
    
    # DLM generation algorithms (Fast-dLLM acceleration)
    python eval_dlm.py --generation-algorithm basic           # No caching
    python eval_dlm.py --generation-algorithm prefix_cache    # Prefix caching
    python eval_dlm.py --generation-algorithm dual_cache      # Dual caching (default for DLM)
    
    # Nemotron native generation
    python eval_dlm.py --generation-algorithm nemotron        # Native Nemotron generation
    
    # Advanced settings for specific models
    python eval_dlm.py --generation-algorithm dual_cache --threshold 0.8 --factor 2.0  # DLM Fast-dLLM
    python eval_dlm.py --generation-algorithm nemotron --steps 128 --threshold 0.9      # Nemotron native
    
    # Handle truncated outputs (when model cuts off mid-reasoning)
    python eval_dlm.py --keep-thinking                    # Don't remove <think> tags, extract answer from full output
    python eval_dlm.py --keep-thinking --tokens-to-generate 1024  # Increase tokens + keep thinking mode
"""

import os
import re
import argparse
from pathlib import Path

import nemo_skills.dataset
from nemo_skills.pipeline.eval import eval


ARENA_HARD_BENCHMARKS = frozenset({"arena-hard", "arena-hard-v2"})


def _benchmark_names(benchmark_specs: str) -> set[str]:
    """Return normalized benchmark names from NeMo-Skills name:repeats specs."""
    return {
        spec.split(":", 1)[0].strip()
        for spec in benchmark_specs.split(",")
        if spec.strip()
    }


def _validate_arena_hard_runtime(config: dict) -> set[str]:
    """Fail early when an Arena-Hard dataset or its default judge is unavailable."""
    arena_benchmarks = _benchmark_names(config["benchmarks"]) & ARENA_HARD_BENCHMARKS
    if not arena_benchmarks:
        return set()

    dataset_root = Path(nemo_skills.dataset.__file__).resolve().parent
    missing = [
        name
        for name in sorted(arena_benchmarks)
        if not (dataset_root / name / "__init__.py").is_file()
        or not (dataset_root / name / "prepare.py").is_file()
    ]
    if missing:
        raise RuntimeError(
            "The active NeMo-Skills installation does not provide the required "
            f"Arena-Hard dataset adapter(s): {', '.join(missing)}. "
            "Use a NeMo-Skills build that includes arena-hard/arena-hard-v2 "
            "(the validated NLD environment uses nemo-skills 0.7.0)."
        )

    # Both built-in Arena-Hard variants default to api.openai.com + GPT-4.1.
    # A custom OpenAI-compatible judge endpoint may use its own authentication,
    # so only enforce OPENAI_API_KEY for the built-in/default OpenAI endpoint.
    judge_address = config["judge_server_address"] or "https://api.openai.com/v1"
    uses_openai_api = "api.openai.com" in judge_address.lower()
    if (
        uses_openai_api
        and not config["dry_run"]
        and not config["skip_judge_api_key_check"]
        and not os.environ.get("OPENAI_API_KEY")
    ):
        raise RuntimeError(
            "Arena-Hard uses the GPT-4.1 OpenAI judge by default, but "
            "OPENAI_API_KEY is not set. Export OPENAI_API_KEY, configure a "
            "custom judge with --judge-model/--judge-server-address, or use "
            "--skip-judge-api-key-check when credentials are injected downstream."
        )

    return arena_benchmarks


def wrap_arguments(arguments: str):
    """Minimal NeMo-Skills ctx wrapper without importing the full CLI module."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    return MockContext(args=[arg for arg in arguments.split(" ") if arg])


def _str_to_bool(value: str) -> bool:
    """Parse common boolean string values for CLI flags."""
    normalized = value.strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _is_nemotron_nanov2(model_name: str) -> bool:
    """Check if a model name/path refers to Nemotron Nano v2."""
    return bool(re.search(r'Nemotron[-_]?Nano.*v2', model_name, re.IGNORECASE))


def _apply_nanov2_no_think_patch():
    """Monkey-patch litellm to inject /no_think into system prompts.

    Nemotron Nano v2 chat templates recognize the literal string '/no_think'
    in the system prompt to force no-thinking mode.  The template strips
    '/no_think' from the text before formatting input to the model, so the
    injected token never reaches the model itself.
    """
    try:
        import litellm
    except ImportError:
        print("WARNING: litellm not importable; /no_think injection skipped")
        return False

    _original_acompletion = litellm.acompletion

    async def _patched_acompletion(*args, **kwargs):
        messages = kwargs.get("messages")
        if messages is not None:
            found_system = False
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    found_system = True
                    content = msg.get("content", "")
                    if "/no_think" not in content:
                        msg["content"] = f"/no_think {content}" if content else "/no_think"
            if not found_system:
                messages.insert(0, {"role": "system", "content": "/no_think"})
        return await _original_acompletion(*args, **kwargs)

    litellm.acompletion = _patched_acompletion
    print("\n🔇 Nemotron Nano v2: injecting /no_think into system prompts")
    return True


def create_parser():
    """Create argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate DLM/Nemotron diffusion models on various benchmarks using NeMo-Skills with OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Benchmark and evaluation settings
    parser.add_argument(
        "--benchmark", 
        default="gsm8k:4",
        help="Benchmark to evaluate on (format: benchmark_name:num_samples)"
    )
    parser.add_argument(
        "--output-dir", 
        default=".",
        help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--expname",
        default=None,
        help="Experiment name (defaults to 'dlm-{benchmark}-eval')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="NeMo-Skills num_chunks override. In SGLang-proxy eval this is the client-side chunking/concurrency knob.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="NeMo-Skills max_concurrent_requests override. Use with SGLang proxy concurrency to define the client workload.",
    )
    parser.add_argument(
        "--cluster",
        default="local",
        help="If you want to run on a cluster via nemo-skills (defaults to 'local')"
    )
    
    # Server configuration
    parser.add_argument(
        "--server-address",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible server endpoint"
    )
    parser.add_argument(
        "--model",
        default="nemotron-labs-diffusion-8b", 
        help="Model identifier alias (a label, not the HF id; default: nemotron-labs-diffusion-8b)"
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model override for judge-based benchmarks. Arena-Hard defaults to the NeMo-Skills setting (GPT-4.1).",
    )
    parser.add_argument(
        "--judge-server-address",
        default=None,
        help="OpenAI-compatible judge endpoint override. Arena-Hard defaults to https://api.openai.com/v1.",
    )
    parser.add_argument(
        "--judge-server-type",
        default=None,
        help="NeMo-Skills judge server type override (normally openai).",
    )
    parser.add_argument(
        "--skip-judge-api-key-check",
        action="store_true",
        help="Skip the Arena-Hard OPENAI_API_KEY preflight check. This does not disable authentication in the judge client.",
    )
    
    # Inference settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--top-k", 
        type=int,
        default=-1,
        help="Top-k sampling parameter (will be forced to -1 for OpenAI API compatibility)"
    )
    parser.add_argument(
        "--tokens-to-generate",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    # Diffusion model parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Diffusion steps (1-512, higher = better quality but slower) - used by both DLM and Nemotron"
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=8,
        help="Block length for semi-autoregressive generation - used by both DLM and Nemotron"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (0.0-3.0) - primarily used by DLM models"
    )
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy - primarily used by DLM models"
    )
    
    # Generation algorithm selection
    parser.add_argument(
        "--generation-algorithm",
        default=None,
        choices=["basic", "prefix_cache", "dual_cache", "nemotron", "nemotron_mixed", "dinfer_blockwise", "dinfer_hierarchy", "dinfer_credit", "dllm_eval", "ar", "ar_native"],
        help="Generation algorithm: DLM (basic/prefix_cache/dual_cache), Nemotron (nemotron), "
             "Nemotron Mixed AR+dLM (nemotron_mixed), "
             "AR native (ar_native=model.ar_generate()), or AR (ar=standard autoregressive, no diffusion). "
             "Auto-detected from --model when not specified (ar model → ar algorithm, otherwise → nemotron)."
    )
    def _nullable_float(value):
        if value.lower() in ("none", "null", ""):
            return None
        return float(value)

    parser.add_argument(
        "--threshold",
        type=_nullable_float,
        default=None,
        help="Confidence threshold - for DLM parallel decoding (e.g., 0.8), Nemotron (e.g., 0.9), or 'none' to disable"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Factor for DLM dynamic parallel decoding strategy (e.g., 2.0) - not used by Nemotron"
    )
    parser.add_argument(
        "--shift-logits",
        type=_str_to_bool,
        default=None,
        help="Whether to use shifted logits in generation. Auto-default: false for Nemotron, true for AR mode."
    )
    parser.add_argument(
        "--ar-weight",
        type=float,
        default=None,
        help="AR logit mixing weight for nemotron_mixed algorithm (0.0=pure dLM, 1.0=pure AR, e.g. 0.3)"
    )
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=None,
        help="Max tokens for thinking before forcing </think> (e.g. 6000). "
             "When set, the generation loop injects </think> after this many tokens "
             "if the model hasn't produced one yet."
    )
    parser.add_argument(
        "--linear-speculation",
        nargs="?",
        const="true",
        default=None,
        help="Enable linear self-speculation (linear_spec_generate). "
             "Pass the flag with no value to enable; omit to disable."
    )
    parser.add_argument(
        "--draft-lora-only",
        type=_str_to_bool,
        default=None,
        help="When true, request linear_spec_generate_lora instead of linear_spec_generate. "
             "Only applies to linear speculation mode."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        help="Sampler name to pass to model.generate() (default: None, omitted)."
    )
    parser.add_argument(
        "--exclude-unfinished-nfe",
        type=_str_to_bool,
        default=None,
        help="Exclude unfinished samples from NFE/tokens aggregation. "
             "When true, samples with nfe >= tokens_to_generate are excluded."
    )
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Do not send DLM/Nemotron-specific inference.extra_body fields. "
             "Use this when NeMo-Skills should keep its benchmark/prompt/scoring "
             "logic but the backend is a vanilla SGLang OpenAI-compatible server.",
    )
    # Execution settings
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual evaluation"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true", 
        help="Run quick test mode (10 problems, single sample, overrides some settings)"
    )
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Keep <think> tags in generation (don't remove them). Useful when model outputs are truncated."
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable <think> tag injection in the chat template (via chat_template_kwargs.enable_thinking=False). "
             "Use for models trained without thinking tokens."
    )
    parser.add_argument(
        "--math-prompt-config",
        default=None,
        help="Override prompt_config for math benchmarks (gsm8k, math-500, etc.). "
             "E.g. 'qwen/math-cot' adds a 'reason step by step' system message that "
             "prevents models from skipping chain-of-thought reasoning."
    )
    parser.add_argument(
        "--strip-thinking",
        action="store_true",
        help="After each benchmark eval, strip <think>...</think> blocks from "
             "output JSONL files and re-score (e.g. IFEval). Use this when "
             "thinking is enabled but format-sensitive benchmarks should be "
             "scored on the answer only, not the thinking chain."
    )
    parser.add_argument(
        "--nanov2-no-think",
        action="store_true",
        help="Inject '/no_think' into every system prompt for Nemotron Nano v2 models. "
             "The Nano v2 chat template recognizes this token and forces no-thinking "
             "mode, stripping it from the system prompt before formatting input to the "
             "model. Auto-enabled when --disable-thinking is set and the model name "
             "(via SERVER_MODEL_PATH env var) matches Nemotron Nano v2."
    )
    
    return parser


def main():
    """Run benchmark evaluation using DLM or Nemotron models with configurable settings."""
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Auto-detect generation algorithm from --model when not explicitly set
    if args.generation_algorithm is None:
        if args.model == "ar" or args.model.startswith("vllm-ar"):
            args.generation_algorithm = "ar"
            print("ℹ️  Auto-detected AR mode from --model (no diffusion parameters will be sent)")
        else:
            args.generation_algorithm = "nemotron"
    
    is_ar_mode = (args.generation_algorithm == "ar")
    
    # Adjust defaults for Nemotron models
    if args.generation_algorithm == "nemotron":
        # Suggest better defaults for Nemotron if user hasn't specified custom values

        # Nemotron typically works well with these defaults
        if not hasattr(args, '_threshold_set') and args.threshold is None:
            print("ℹ️  Note: Nemotron models typically use threshold=0.9 for good results.")
            print("   Add --threshold 0.9 to optimize Nemotron generation.")
    
    # Build configuration from parsed arguments
    config = {
        # Evaluation settings
        "benchmarks": args.benchmark,
        "output_dir": args.output_dir,
        "expname": args.expname,
        
        # Server configuration for external OpenAI API
        "server_type": "openai",  # Use OpenAI-compatible server
        "server_address": args.server_address,
        "model": args.model,
        "judge_model": args.judge_model,
        "judge_server_address": args.judge_server_address,
        "judge_server_type": args.judge_server_type,
        "skip_judge_api_key_check": args.skip_judge_api_key_check,
        
        # Additional evaluation arguments
        "cluster": None if args.cluster == "local" else args.cluster,
        "dry_run": args.dry_run,
        
        # Inference settings
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "tokens_to_generate": args.tokens_to_generate,
        "max_samples": args.max_samples,
        "num_chunks": args.num_chunks,
        "max_concurrent_requests": args.max_concurrent_requests,
        "quick_test": args.quick_test,
        "keep_thinking": args.keep_thinking,
        "disable_thinking": args.disable_thinking,
        "strip_thinking": args.strip_thinking,
        "nanov2_no_think": args.nanov2_no_think,
        "math_prompt_config": args.math_prompt_config,
        
        # DLM-specific settings
        "steps": args.steps,
        "block_length": args.block_length,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        
        # Generation algorithm selection
        "generation_algorithm": getattr(args, 'generation_algorithm', 'dual_cache'),
        "threshold": args.threshold,
        "factor": args.factor,
        "shift_logits": args.shift_logits,
        "ar_weight": args.ar_weight,
        "max_thinking_tokens": args.max_thinking_tokens,
        "linear_speculation": args.linear_speculation,
        "draft_lora_only": args.draft_lora_only,
        "sampler": args.sampler,
        "exclude_unfinished_nfe": args.exclude_unfinished_nfe,
        "no_extra_body": args.no_extra_body,
    }

    # Model-specific default for shift_logits when user does not set it explicitly.
    if config["shift_logits"] is None:
        if is_ar_mode:
            config["shift_logits"] = True
    elif config["generation_algorithm"] in ("nemotron", "nemotron_mixed", "ar_native"):
        config["shift_logits"] = False
    
    # Set default experiment name if not provided
    if config["expname"] is None:
        benchmark_name = args.benchmark.split(":")[0]  # Extract benchmark name (e.g., "gsm8k" from "gsm8k:4")
        if is_ar_mode:
            model_type = "ar"
        elif config["generation_algorithm"] == "ar_native":
            model_type = "ar_native"
        elif config["generation_algorithm"] == "nemotron":
            model_type = "nemotron"
        elif config["generation_algorithm"] == "nemotron_mixed":
            model_type = "nemotron_mixed"
        else:
            model_type = "dlm"
        config["expname"] = f"{model_type}-{benchmark_name}-eval"
    
    # Override with environment variables if needed (for backward compatibility)
    config["output_dir"] = os.environ.get("EVAL_OUTPUT_DIR", config["output_dir"])
    config["server_address"] = os.environ.get("DLM_SERVER_URL", config["server_address"])
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    arena_benchmarks = _validate_arena_hard_runtime(config)
    
    print("=" * 60)
    benchmark_name = config["benchmarks"].split(":")[0].upper()
    if is_ar_mode:
        model_type_display = "Autoregressive (AR)"
    elif config["generation_algorithm"] == "ar_native":
        model_type_display = "AR Native (model.ar_generate)"
    elif config["generation_algorithm"] in ():
        model_type_display = "Self-Spec"
    elif config["generation_algorithm"] == "nemotron":
        model_type_display = "Nemotron"
    elif config["generation_algorithm"] == "nemotron_mixed":
        model_type_display = "Nemotron Mixed AR+dLM"
    else:
        model_type_display = "DLM"
    print(f"{benchmark_name} Evaluation with {model_type_display} Model")
    print("=" * 60)
    print(f"Server: {config['server_address']}")
    print(f"Model: {config['model']}")
    print(f"Benchmark: {config['benchmarks']}")
    print(f"Output: {config['output_dir']}")
    print(f"Experiment: {config['expname']}")
    print(f"Temperature: {config['temperature']} | Top-p: {config['top_p']} | Top-k: {config['top_k']} (will be set to -1)")
    print(f"Max tokens: {config['tokens_to_generate']}")
    print(f"Generation Algorithm: {config['generation_algorithm']}")
    if arena_benchmarks:
        print(f"Arena judge model: {config['judge_model'] or 'gpt-4.1 (NeMo-Skills default)'}")
        print(
            "Arena judge server: "
            f"{config['judge_server_address'] or 'https://api.openai.com/v1 (NeMo-Skills default)'}"
        )
        print(f"Arena judge server type: {config['judge_server_type'] or 'openai (NeMo-Skills default)'}")
    if config["exclude_unfinished_nfe"] is not None:
        print(f"Exclude unfinished NFE: {config['exclude_unfinished_nfe']}")
    
    # Show model-specific parameters
    if is_ar_mode:
        print("AR mode: no diffusion extra_body parameters will be sent")
    elif config["generation_algorithm"] == "ar_native":
        print(f"AR Native: calls model.ar_generate() with max_new_tokens={config['tokens_to_generate']}")
        print("Steps, block_length, cfg_scale, remasking: Not used by AR Native")
    elif config["generation_algorithm"] in ():
        print(f"Self-Spec Steps: {config['steps']} | Block length: {config['block_length']}")
        if config['threshold'] is not None:
            print(f"Self-Spec Threshold: {config['threshold']}")
        print("CFG scale and remasking: Not used by Self-Spec")
    elif config["generation_algorithm"] == "nemotron":
        print(f"Nemotron Steps: {config['steps']} | Block length: {config['block_length']}")
        if config['threshold'] is not None:
            print(f"Nemotron Threshold: {config['threshold']}")
        print("CFG scale and remasking: Not used by Nemotron")
        if config['factor'] is not None:
            print("⚠️  Factor parameter not used by Nemotron (DLM-specific)")
    elif config["generation_algorithm"] == "nemotron_mixed":
        print(f"Nemotron Mixed Steps: {config['steps']} | Block length: {config['block_length']}")
        if config['threshold'] is not None:
            print(f"Nemotron Mixed Threshold: {config['threshold']}")
        if config['ar_weight'] is not None:
            print(f"AR Weight: {config['ar_weight']} (0.0=pure dLM, 1.0=pure AR)")
        print("CFG scale and remasking: Not used by Nemotron Mixed")
    else:
        print(f"DLM Steps: {config['steps']} | Block length: {config['block_length']}")
        print(f"CFG scale: {config['cfg_scale']} | Remasking: {config['remasking']}")
        if config['threshold'] is not None:
            print(f"Fast-dLLM Threshold: {config['threshold']}")
        if config['factor'] is not None:
            print(f"Fast-dLLM Factor: {config['factor']}")
    
    if config["max_samples"]:
        print(f"Max samples: {config['max_samples']} (for testing)")
    print("=" * 60)
    
    if config["quick_test"] or os.environ.get("EVAL_QUICK_TEST") == "1":
        print("🚀 QUICK TEST MODE enabled")
    else:
        print("⚠️  This evaluation may take some time to complete depending on the benchmark.")
    print("   Use --quick-test for a faster test run, or --max-samples N to limit problems.")
    
    # Nemotron Nano v2 no-thinking: auto-detect from SERVER_MODEL_PATH or
    # the explicit --nanov2-no-think flag.  Injects '/no_think' into every
    # system prompt so the model's chat template enters no-thinking mode.
    nanov2_no_think = config["nanov2_no_think"]
    if not nanov2_no_think and config["disable_thinking"]:
        server_model = os.environ.get("SERVER_MODEL_PATH", "")
        if _is_nemotron_nanov2(server_model):
            nanov2_no_think = True
            print(f"\nAuto-detected Nemotron Nano v2 from SERVER_MODEL_PATH={server_model}")

    # Run evaluation using NeMo-Skills
    try:
        # Only pass generation-specific arguments through wrap_arguments
        # These will be forwarded to the underlying generation script
        generation_args = [
            f"++inference.temperature={config['temperature']}",
            f"++inference.top_p={config['top_p']}", 
            f"++inference.top_k=-1",  # Must be -1 for OpenAI API compatibility
            f"++inference.tokens_to_generate={config['tokens_to_generate']}",
            f"++num_chunks={config['num_chunks']}",
        ]
        if config["max_concurrent_requests"] is not None:
            generation_args.append(f"++max_concurrent_requests={config['max_concurrent_requests']}")
        
        if config["no_extra_body"]:
            print("\nSGLang backend mode: DLM/Nemotron-specific extra_body parameters are suppressed.")
            print("  NeMo-Skills benchmark organization, prompts, datasets, and scoring remain unchanged.")
            print("  SGLang server-side defaults/configuration control decoding behavior.")
        elif is_ar_mode:
            # AR mode: standard autoregressive inference.
            # Explicitly pass huggingface algorithm to avoid server-side defaulting to diffusion algorithms.
            ar_steps = max(1, int(config["tokens_to_generate"]))
            generation_args += [
                "++inference.extra_body.generation_algorithm=huggingface",
                f"++inference.extra_body.steps={ar_steps}",
                "++inference.extra_body.block_length=1",
                f"++inference.extra_body.shift_logits={config['shift_logits']}",
                "++inference.extra_body.causal_context=True",
                "++inference.extra_body.threshold=null",
            ]
            print(f"\n🔧 AR generation parameters:")
            print(f"  temperature={config['temperature']}")
            print(f"  tokens_to_generate={config['tokens_to_generate']}")
            print("  generation_algorithm=huggingface (explicit via extra_body)")
            print(f"  steps={ar_steps}")
            print("  block_length=1")
            print(f"  shift_logits={config['shift_logits']}")
            print("  causal_context=True")
            print("  threshold=None")
            print("  (No cfg_scale/remasking sent)")
        else:
            # Diffusion mode: pass model-specific parameters via extra_body
            generation_args += [
                f"++inference.extra_body.steps={config['steps']}",
                f"++inference.extra_body.block_length={config['block_length']}",
                f"++inference.extra_body.cfg_scale={config['cfg_scale']}",
                f"++inference.extra_body.remasking={config['remasking']}",
                f"++inference.extra_body.generation_algorithm={config['generation_algorithm']}",
            ]
            if config["shift_logits"] is not None:
                generation_args.append(f"++inference.extra_body.shift_logits={config['shift_logits']}")
            
            # Add optional Fast-dLLM parameters if specified
            if config['threshold'] is not None:
                generation_args.append(f"++inference.extra_body.threshold={config['threshold']}")
            if config['factor'] is not None:
                generation_args.append(f"++inference.extra_body.factor={config['factor']}")
            if config['ar_weight'] is not None:
                generation_args.append(f"++inference.extra_body.ar_weight={config['ar_weight']}")
            if config['max_thinking_tokens'] is not None:
                generation_args.append(f"++inference.extra_body.max_thinking_tokens={config['max_thinking_tokens']}")
            if config['linear_speculation']:
                generation_args.append(
                    f"++inference.extra_body.linear_speculation={config['linear_speculation']}"
                )
            if config['draft_lora_only'] is not None:
                generation_args.append(
                    f"++inference.extra_body.draft_lora_only={str(config['draft_lora_only']).lower()}"
                )
            if config['sampler'] is not None:
                generation_args.append(f"++inference.extra_body.sampler={config['sampler']}")
            if config['generation_algorithm'] == "ar_native":
                model_type_display = "AR Native"
            elif config['generation_algorithm'] == "nemotron":
                model_type_display = "Nemotron"
            elif config['generation_algorithm'] == "nemotron_mixed":
                model_type_display = "Nemotron Mixed AR+dLM"
            else:
                model_type_display = "DLM"
            print(f"\n🔧 {model_type_display} generation parameters (via extra_body):")
            print(f"  steps={config['steps']}")
            print(f"  block_length={config['block_length']}")
            
            if config['generation_algorithm'] in ("nemotron", "nemotron_mixed", "ar_native"):
                print(f"  generation_algorithm={config['generation_algorithm']} ({model_type_display} native)")
                print(f"  shift_logits={config['shift_logits']}")
                if config['threshold'] is not None:
                    print(f"  threshold={config['threshold']} ({model_type_display} generation threshold)")
                if config['ar_weight'] is not None:
                    print(f"  ar_weight={config['ar_weight']} (AR logit mixing weight)")
                if config['cfg_scale'] != 0.0 or config['remasking'] != "low_confidence":
                    print(f"  ⚠️  Note: cfg_scale and remasking are not used by {model_type_display}")
                if config['factor'] is not None:
                    print(f"  ⚠️  Note: factor is not used by {model_type_display} (DLM-specific)")
            else:
                print(f"  cfg_scale={config['cfg_scale']}")
                print(f"  remasking={config['remasking']}")
                print(f"  generation_algorithm={config['generation_algorithm']} (DLM Fast-dLLM)")
                if config['shift_logits'] is not None:
                    print(f"  shift_logits={config['shift_logits']}")
                if config['threshold'] is not None:
                    print(f"  threshold={config['threshold']} (Fast-dLLM parallel decoding)")
                if config['factor'] is not None:
                    print(f"  factor={config['factor']} (Fast-dLLM dynamic decoding)")
            
            if config['max_thinking_tokens'] is not None:
                print(f"  max_thinking_tokens={config['max_thinking_tokens']} (thinking budget)")
            if config['linear_speculation']:
                print(f"  linear_speculation={config['linear_speculation']}")
            if config['draft_lora_only'] is not None:
                print(f"  draft_lora_only={config['draft_lora_only']}")
            if config['sampler'] is not None:
                print(f"  sampler={config['sampler']}")
            print("   (Passed via NeMo-Skills extra_body to OpenAI API)")

        if (not config["no_extra_body"]) and config["exclude_unfinished_nfe"] is not None:
            generation_args.append(
                f"++inference.extra_body.exclude_unfinished_nfe={str(config['exclude_unfinished_nfe']).lower()}"
            )
        
        # Add max_samples if specified
        if config["max_samples"]:
            generation_args.append(f"++max_samples={config['max_samples']}")
        
        # Quick test mode for development/testing (can be enabled via --quick-test or environment variable)
        if config["quick_test"] or os.environ.get("EVAL_QUICK_TEST") == "1":
            # Override benchmark to single sample if not already set
            if not config["max_samples"]:
                benchmark_name = config["benchmarks"].split(":")[0]
                config["benchmarks"] = f"{benchmark_name}:1"  # Single sample
                generation_args.append("++max_samples=10")  # Only 10 problems
            print(f"\n🚀 QUICK TEST MODE: Running with {config['benchmarks']} and limited samples")
        
        # Control whether NeMo-Skills strips <think>...</think> from generation output.
        # The installed nemo-skills version uses `parse_reasoning` (not `remove_thinking`).
        # parse_reasoning=True  -> strips everything before </think>, keeps only the answer
        # parse_reasoning=False -> keeps the full generation including thinking (default)
        if config["keep_thinking"]:
            generation_args.append("++parse_reasoning=False")
            print("\n⚠️  Keep-thinking mode enabled: <think> tags will NOT be removed from generations")
            print("   This helps when model outputs are truncated and missing </think> tags")
        elif config["strip_thinking"]:
            generation_args.append("++parse_reasoning=True")
            print("\n🧹 Strip-thinking enabled: NeMo-Skills will remove <think> tags before evaluation")
            print("   Post-eval strip+rescore will also run as a safety net")
        
        # Disable <think> tag injection in the chat template
        if config["disable_thinking"] and not config["no_extra_body"]:
            generation_args.append("++inference.extra_body.chat_template_kwargs.enable_thinking=False")
            print("\n🔇 Thinking disabled: chat template will NOT inject <think> tags")
        elif config["disable_thinking"] and config["no_extra_body"]:
            print("\nNote: --disable-thinking was requested, but --no-extra-body is active; no chat_template_kwargs are sent.")

        if nanov2_no_think:
            print("\n🔇 Nemotron Nano v2: will inject /no_think into system_message per-benchmark")
        
        #print("#### generation_args: ", str(wrap_arguments(" ".join(generation_args))), flush=True)
        #exit(2)

        # NFE log setup
        import subprocess
        add_nfe_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_nfe_to_metrics.py")
        nfe_log_dir = os.environ.get("NFE_LOG_DIR", "")
        nfe_log_file = os.path.join(nfe_log_dir, "nfe_log.jsonl") if nfe_log_dir else ""

        # Split benchmarks so each gets its own NFE tracking
        benchmark_specs = [b.strip() for b in config["benchmarks"].split(",") if b.strip()]
        result = None

        # Math benchmarks where --math-prompt-config override applies
        MATH_BENCHMARKS = {"gsm8k", "math", "math-500", "aime24", "aime25"}

        for bench_spec in benchmark_specs:
            bench_name = bench_spec.split(":")[0].strip()

            # Inject benchmark_name into extra_body so the server/proxy can tag
            # per-request metrics. SGLang proxy mode truncates per-benchmark logs
            # and does not require this tag, so keep the request body clean there.
            if config["no_extra_body"]:
                bench_generation_args = list(generation_args)
            else:
                bench_generation_args = generation_args + [
                    f"++inference.extra_body.benchmark_name={bench_name}",
                ]

            # Override prompt_config for math benchmarks when requested.
            # E.g. --math-prompt-config qwen/math-cot adds a "reason step by
            # step" system message that prevents models from skipping CoT.
            if config["math_prompt_config"] and bench_name in MATH_BENCHMARKS:
                bench_generation_args.append(f"++prompt_config={config['math_prompt_config']}")
                print(f"  Using {config['math_prompt_config']} prompt for {bench_name}")

            # Nemotron Nano v2 no-thinking: inject '/no_think' into the system
            # message via NeMo-Skills config.  The Nano v2 chat template
            # recognises this literal string in the system prompt and forces
            # no-thinking mode, stripping '/no_think' from the text before
            # formatting input to the model.
            # NOTE: the old litellm monkey-patch (_apply_nanov2_no_think_patch)
            # is ineffective because NeMo-Skills uses the OpenAI client directly.
            if nanov2_no_think:
                bench_generation_args.append("++system_message=/no_think")
                print(f"  Nano v2: /no_think injected into system_message for {bench_name}")

            # Clear the NFE log before this benchmark so entries are isolated
            if nfe_log_file:
                try:
                    with open(nfe_log_file, "w") as f:
                        pass  # truncate
                except OSError:
                    pass

            print(f"\n--- Running benchmark: {bench_spec} ---")

            # Call the evaluation function with direct parameters. Judge
            # overrides are passed only when explicitly requested so ordinary
            # benchmarks and NeMo-Skills dataset defaults remain unchanged.
            eval_kwargs = dict(
                ctx=wrap_arguments(" ".join(bench_generation_args)),
                # Core parameters
                benchmarks=bench_spec,
                output_dir=config["output_dir"],
                expname=config["expname"],

                # Server configuration
                server_type=config["server_type"],
                server_address=config["server_address"],
                model=config["model"],

                # Optional parameters
                cluster=config["cluster"],
                dry_run=config["dry_run"],
            )
            if bench_name in arena_benchmarks and config["judge_model"]:
                eval_kwargs["judge_model"] = config["judge_model"]
            if bench_name in arena_benchmarks and config["judge_server_address"]:
                eval_kwargs["judge_server_address"] = config["judge_server_address"]
            if bench_name in arena_benchmarks and config["judge_server_type"]:
                eval_kwargs["judge_server_type"] = config["judge_server_type"]
            result = eval(**eval_kwargs)

            # Process NFE for this benchmark immediately
            if not config["dry_run"]:
                eval_dir = os.path.join(config["output_dir"], "eval-results", bench_name)
                metrics_json = os.path.join(eval_dir, "metrics.json")

                try:
                    # Strategy 1: server-side NFE log (filtered to this benchmark)
                    if nfe_log_file and os.path.isfile(nfe_log_file):
                        nfe_merge_args = []
                        if config["exclude_unfinished_nfe"]:
                            nfe_merge_args.extend([
                                "--exclude-unfinished-nfe",
                                "--nfe-cutoff", str(config["tokens_to_generate"]),
                                "--eval-results-dir", eval_dir,
                            ])
                        r = subprocess.run(
                            [os.environ.get("PYTHON", "python3"), add_nfe_script,
                             "--nfe-log", nfe_log_file,
                             "--filter-benchmark", bench_name,
                             "--metrics-json", metrics_json,
                             *nfe_merge_args],
                            capture_output=True, text=True, timeout=60,
                        )
                        if r.returncode == 0:
                            print(f"Added average_nfe to {bench_name} metrics (from server NFE log)")
                            continue

                    # Strategy 2: output JSONL fallback
                    if os.path.isdir(eval_dir):
                        nfe_merge_args = []
                        if config["exclude_unfinished_nfe"]:
                            nfe_merge_args.extend([
                                "--exclude-unfinished-nfe",
                                "--nfe-cutoff", str(config["tokens_to_generate"]),
                            ])
                        r = subprocess.run(
                            [os.environ.get("PYTHON", "python3"), add_nfe_script, "--eval-results-dir", eval_dir, *nfe_merge_args],
                            capture_output=True, text=True, timeout=60,
                        )
                        if r.returncode == 0:
                            print(f"Added average_nfe to {bench_name} metrics (from output JSONL)")
                        elif "No NFE values found" not in (r.stderr or r.stdout or ""):
                            print(f"Note: add_nfe_to_metrics for {bench_name}: {r.stderr or r.stdout}")
                except Exception as e:
                    print(f"Note: could not add NFE to {bench_name} metrics: {e}")

            # Strip <think>...</think> blocks and re-score if requested.
            # This runs AFTER the initial NeMo-Skills eval so that the
            # output JSONL files exist, then overwrites them with cleaned
            # versions and recomputes metrics (e.g. IFEval rescoring).
            if config["strip_thinking"]:
                eval_dir = os.path.join(config["output_dir"], "eval-results", bench_name)
                strip_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strip_think_and_rescore.py")
                if os.path.isdir(eval_dir) and os.path.isfile(strip_script):
                    try:
                        print(f"\n🧹 Stripping <think> blocks from {bench_name} outputs and re-scoring...")
                        r = subprocess.run(
                            [os.environ.get("PYTHON", "python3"), strip_script, eval_dir],
                            capture_output=True, text=True, timeout=300,
                        )
                        if r.stdout:
                            print(r.stdout)
                        if r.returncode != 0 and r.stderr:
                            print(f"Warning: strip_think_and_rescore for {bench_name}: {r.stderr}")
                        else:
                            print(f"Thinking blocks stripped from {bench_name} outputs")
                    except Exception as e:
                        print(f"Warning: could not strip thinking from {bench_name}: {e}")

            if not config["dry_run"]:
                eval_dir = os.path.join(config["output_dir"], "eval-results", bench_name)
                metrics_json = os.path.join(eval_dir, "metrics.json")
                if not os.path.isfile(metrics_json):
                    raise RuntimeError(
                        f"NeMo-Skills did not produce metrics for {bench_name}: {metrics_json}"
                    )

        print("\n" + "=" * 60)
        if config["dry_run"]:
            print("DRY RUN COMPLETED - No actual evaluation was performed")
        else:
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            for bench_spec in benchmark_specs:
                bench_name = bench_spec.split(":")[0].strip()
                print(f"Results saved to: {config['output_dir']}")
                print(f"  metrics: {config['output_dir']}/eval-results/{bench_name}/metrics.json")
                print(f"  outputs: {config['output_dir']}/eval-results/{bench_name}/output-rs*.jsonl")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your diffusion model server (DLM/Nemotron) is running on localhost:8000")
        print("2. Test the server with: curl http://localhost:8000/v1/models")
        print("3. Check server logs for any errors")
        print("4. Verify the server is OpenAI-compatible")
        print("5. Verify generation algorithm matches your model type:")
        print("   - DLM models: use --generation-algorithm dual_cache (or basic/prefix_cache)")
        print("   - Nemotron models: use --generation-algorithm nemotron")
        print("   - AR models (HuggingFace causal LM): use --generation-algorithm ar (or --model ar)")
        print("6. For quick testing, use: python eval_dlm.py --quick-test")
        raise


if __name__ == "__main__":
    main()
