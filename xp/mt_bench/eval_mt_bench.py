#!/usr/bin/env python3
"""Run pinned FastChat MT-Bench against OpenAI-compatible model and judge APIs.

This runner intentionally stays separate from NeMo-Skills. MT-Bench requires
two sequential model turns per question and its own LLM-as-a-judge protocol,
which nemo-skills 0.7.0 does not provide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import httpx
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("eval_mt_bench.py requires httpx") from exc


FASTCHAT_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"
PROTOCOL_VERSION = 2
FASTCHAT_DATA_BASE = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    f"{FASTCHAT_COMMIT}/fastchat/llm_judge/data"
)
ASSETS = {
    "question.jsonl": {
        "url": f"{FASTCHAT_DATA_BASE}/mt_bench/question.jsonl",
        "sha256": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    },
    "reference_answer_gpt-4.jsonl": {
        "url": f"{FASTCHAT_DATA_BASE}/mt_bench/reference_answer/gpt-4.jsonl",
        "sha256": "f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f",
    },
    "judge_prompts.jsonl": {
        "url": f"{FASTCHAT_DATA_BASE}/judge_prompts.jsonl",
        "sha256": "fd283293406d024f44c174b094ef48031d0687a4682fd3a56b29b138f80281b6",
    },
}

TEMPERATURE_BY_CATEGORY = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}
REFERENCE_CATEGORIES = frozenset({"math", "reasoning", "coding"})
RATING_PATTERN = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")
RATING_PATTERN_FALLBACK = re.compile(r"\[(\d+(?:\.\d+)?)\]")
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-server-address", default="")
    parser.add_argument("--candidate-model", default="nemotron-labs-diffusion-8b")
    parser.add_argument("--candidate-api-key-env", default="NLD_CANDIDATE_API_KEY")
    parser.add_argument("--candidate-concurrency", type=int, default=1)
    parser.add_argument("--candidate-timeout", type=float, default=12000.0)
    parser.add_argument("--candidate-extra-body-json", default="{}")
    parser.add_argument("--candidate-generation-algorithm", default="")
    parser.add_argument("--candidate-steps", type=int, default=None)
    parser.add_argument("--candidate-block-length", type=int, default=None)
    parser.add_argument("--candidate-threshold", type=float, default=None)
    parser.add_argument("--candidate-ar-weight", type=float, default=None)
    parser.add_argument("--candidate-max-thinking-tokens", type=int, default=None)
    parser.add_argument("--candidate-linear-speculation", action="store_true")
    parser.add_argument("--candidate-draft-lora-only", action="store_true")
    parser.add_argument("--candidate-sampler", default="")
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=0,
        help=(
            "Base seed. Request seeds are derived deterministically from the "
            "question id and turn (default: 0)."
        ),
    )
    parser.add_argument(
        "--candidate-seed-mode",
        choices=("request", "none"),
        default="request",
        help=(
            "Send deterministic per-turn OpenAI seed fields, or omit seeds for "
            "a backend that does not support them."
        ),
    )
    parser.add_argument(
        "--candidate-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit chat-template thinking mode (default: disabled).",
    )
    parser.add_argument(
        "--candidate-truncate-history-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the model chat template removes prior-turn thinking.",
    )
    parser.add_argument(
        "--candidate-tokenizer",
        default="",
        help="Optional tokenizer path/name used for local prompt preflight.",
    )
    parser.add_argument("--candidate-tokenizer-revision", default="")
    parser.add_argument(
        "--expected-chat-template-sha256",
        default="",
        help="Fail preflight unless the tokenizer chat template has this SHA-256.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--force-temperature", type=float, default=None)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--strip-thinking", action="store_true")

    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument(
        "--judge-server-address", default="https://api.openai.com/v1"
    )
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--skip-judge-api-key-check", action="store_true")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.candidate_concurrency <= 0 or args.judge_concurrency <= 0:
        parser.error("candidate/judge concurrency must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.candidate_steps is not None and args.candidate_steps <= 0:
        parser.error("--candidate-steps must be positive")
    if args.candidate_seed < 0:
        parser.error("--candidate-seed must be non-negative")
    if args.candidate_block_length is not None and args.candidate_block_length <= 0:
        parser.error("--candidate-block-length must be positive")
    if (
        args.candidate_max_thinking_tokens is not None
        and args.candidate_max_thinking_tokens <= 0
    ):
        parser.error("--candidate-max-thinking-tokens must be positive")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p must be between 0 and 1")
    if args.force_temperature is not None and args.force_temperature < 0:
        parser.error("--force-temperature must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive")
    if args.expected_chat_template_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.expected_chat_template_sha256
    ):
        parser.error("--expected-chat-template-sha256 must contain 64 hex digits")
    if not args.prepare_only and not args.dry_run:
        if not args.candidate_server_address:
            parser.error("--candidate-server-address is required")
        if not args.output_dir:
            parser.error("--output-dir is required")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(item)
    return rows


def ensure_assets(data_dir: Path, offline: bool) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for name, metadata in ASSETS.items():
        destination = data_dir / name
        expected = str(metadata["sha256"])
        if destination.is_file() and sha256(destination) == expected:
            resolved[name] = destination
            continue
        if offline:
            raise RuntimeError(
                f"Missing or invalid pinned MT-Bench asset in offline mode: {destination}"
            )
        print(f"Downloading pinned MT-Bench asset: {name}", flush=True)
        with tempfile.NamedTemporaryFile("wb", dir=data_dir, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(str(metadata["url"]), timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {expected}, downloaded {actual}"
            )
        os.replace(temporary, destination)
        resolved[name] = destination

    protocol = {
        "benchmark": "mt-bench",
        "source": "lm-sys/FastChat",
        "fastchat_commit": FASTCHAT_COMMIT,
        "assets": {
            name: {"sha256": metadata["sha256"], "url": metadata["url"]}
            for name, metadata in ASSETS.items()
        },
    }
    atomic_write_json(data_dir / "protocol.json", protocol)
    return resolved


def build_candidate_extra(args: argparse.Namespace) -> dict[str, Any]:
    try:
        candidate_extra = json.loads(args.candidate_extra_body_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid --candidate-extra-body-json: {exc}") from exc
    if not isinstance(candidate_extra, dict):
        raise RuntimeError("--candidate-extra-body-json must decode to an object")

    raw_template_kwargs = candidate_extra.get("chat_template_kwargs", {})
    if raw_template_kwargs is None:
        raw_template_kwargs = {}
    if not isinstance(raw_template_kwargs, dict):
        raise RuntimeError("candidate chat_template_kwargs must be an object")
    chat_template_kwargs = dict(raw_template_kwargs)
    # CLI values are the protocol source of truth. SGLang consumes these request
    # fields directly; the NLD servers use the same values at server startup.
    chat_template_kwargs.update(
        {
            "enable_thinking": bool(args.candidate_enable_thinking),
            "truncate_history_thinking": bool(
                args.candidate_truncate_history_thinking
            ),
        }
    )
    candidate_extra["chat_template_kwargs"] = chat_template_kwargs

    explicit_extra = {
        "generation_algorithm": args.candidate_generation_algorithm or None,
        "steps": args.candidate_steps,
        "block_length": args.candidate_block_length,
        "threshold": args.candidate_threshold,
        "ar_weight": args.candidate_ar_weight,
        "max_thinking_tokens": args.candidate_max_thinking_tokens,
        "linear_speculation": True if args.candidate_linear_speculation else None,
        "draft_lora_only": True if args.candidate_draft_lora_only else None,
        "sampler": args.candidate_sampler or None,
    }
    candidate_extra.update(
        {key: value for key, value in explicit_extra.items() if value is not None}
    )
    candidate_extra["benchmark_name"] = "mt-bench"
    return candidate_extra


def candidate_seed(base_seed: int, question_id: int, turn: int) -> int:
    """Return a stable per-question/per-turn seed independent of concurrency."""
    if turn not in (1, 2):
        raise ValueError(f"MT-Bench turn must be 1 or 2, got {turn}")
    return base_seed + question_id * 2 + (turn - 1)


def prompt_preflight(
    args: argparse.Namespace,
    question: dict[str, Any],
) -> dict[str, Any]:
    if not args.candidate_tokenizer:
        if args.expected_chat_template_sha256:
            raise RuntimeError(
                "--expected-chat-template-sha256 requires --candidate-tokenizer"
            )
        return {
            "status": "not_requested",
            "tokenizer": None,
            "chat_template_sha256": None,
        }

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Prompt preflight requires transformers when --candidate-tokenizer is set"
        ) from exc

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.candidate_tokenizer_revision:
        tokenizer_kwargs["revision"] = args.candidate_tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.candidate_tokenizer, **tokenizer_kwargs
    )
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise RuntimeError(
            f"Tokenizer has no chat template: {args.candidate_tokenizer}"
        )
    template_text = template if isinstance(template, str) else canonical_json(template)
    template_hash = text_sha256(template_text)
    expected = args.expected_chat_template_sha256.lower()
    if expected and template_hash != expected:
        raise RuntimeError(
            "Chat-template SHA-256 mismatch: "
            f"expected {expected}, found {template_hash}"
        )

    template_kwargs = {
        "enable_thinking": bool(args.candidate_enable_thinking),
        "truncate_history_thinking": bool(
            args.candidate_truncate_history_thinking
        ),
    }
    messages_turn_1 = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": str(question["turns"][0])},
    ]
    messages_turn_2 = messages_turn_1 + [
        {
            "role": "assistant",
            "content": "MT-Bench prompt preflight assistant response.",
        },
        {"role": "user", "content": str(question["turns"][1])},
    ]
    turns: dict[str, dict[str, Any]] = {}
    for turn_name, messages in (
        ("turn_1", messages_turn_1),
        ("turn_2", messages_turn_2),
    ):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        token_ids = tokenizer.encode(rendered)
        turns[turn_name] = {
            "rendered_prompt_sha256": text_sha256(rendered),
            "token_ids_sha256": fingerprint(token_ids),
            "prompt_tokens": len(token_ids),
        }

    return {
        "status": "verified_locally",
        "tokenizer": args.candidate_tokenizer,
        "tokenizer_revision": args.candidate_tokenizer_revision or None,
        "tokenizer_class": (
            f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
        ),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "chat_template_sha256": template_hash,
        "expected_chat_template_sha256": expected or None,
        "chat_template_kwargs": template_kwargs,
        "turns": turns,
    }


def generation_protocol(
    args: argparse.Namespace,
    candidate_extra: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_name": "fastchat_mt_bench_model_native_chat_template",
        "fastchat_commit": FASTCHAT_COMMIT,
        "asset_sha256": {
            name: metadata["sha256"] for name, metadata in ASSETS.items()
        },
        "candidate_model": args.candidate_model,
        "system_prompt": args.system_prompt,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "force_temperature": args.force_temperature,
        "category_temperatures": TEMPERATURE_BY_CATEGORY,
        "strip_thinking": args.strip_thinking,
        "candidate_extra_body": candidate_extra,
        "seed": {
            "base": args.candidate_seed,
            "mode": args.candidate_seed_mode,
            "policy": "base + question_id * 2 + (turn - 1)",
        },
        "prompt_preflight": preflight,
    }


def judge_protocol(
    args: argparse.Namespace,
    generation_protocol_fingerprint: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": "fastchat_single_answer_grading",
        "fastchat_commit": FASTCHAT_COMMIT,
        "generation_protocol_fingerprint": generation_protocol_fingerprint,
        "judge_model": args.judge_model,
        "judge_server_address": args.judge_server_address.rstrip("/"),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "reference_answer_model": "gpt-4",
        "reference_answer_sha256": ASSETS["reference_answer_gpt-4.jsonl"][
            "sha256"
        ],
        "judge_prompts_sha256": ASSETS["judge_prompts.jsonl"]["sha256"],
    }


def completion_url(address: str) -> str:
    value = address.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def response_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed chat completion response: {payload}") from exc
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                pieces.append(str(item["text"]))
        return "".join(pieces)
    return str(content)


def request_chat(
    client: httpx.Client,
    *,
    address: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    extra_body: Optional[dict[str, Any]],
    max_retries: int,
    retry_backoff: float,
) -> tuple[str, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)

    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.post(completion_url(address), headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
            return response_content(payload), payload
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = min(retry_backoff * (2 ** (attempt - 1)), 30.0)
            print(
                f"Request attempt {attempt}/{max_retries} failed: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_error}")


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def usage_from_response(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        key: usage.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "nfe")
        if usage.get(key) is not None
    }


def load_answers(
    path: Path,
    model_id: str,
    protocol_fingerprint: str,
    active_ids: set[int],
) -> dict[int, dict[str, Any]]:
    answers: dict[int, dict[str, Any]] = {}
    incompatible: list[int] = []
    for row in read_jsonl(path):
        try:
            question_id = int(row["question_id"])
            turns = row["choices"][0]["turns"]
        except (KeyError, IndexError, TypeError, ValueError):
            incompatible.append(-1)
            continue
        if question_id not in active_ids:
            continue
        if (
            row.get("model_id") == model_id
            and row.get("generation_protocol_fingerprint")
            == protocol_fingerprint
            and isinstance(turns, list)
            and len(turns) == 2
        ):
            answers[question_id] = row
        else:
            incompatible.append(question_id)
    if incompatible:
        shown = sorted(set(incompatible))[:10]
        raise RuntimeError(
            "Cannot resume MT-Bench candidate generation: existing answers use "
            "a different or legacy generation protocol "
            f"(question ids: {shown}). Use a new output directory or remove "
            f"{path}."
        )
    return answers


def generate_one(
    question: dict[str, Any],
    args: argparse.Namespace,
    client: httpx.Client,
    candidate_api_key: str,
    candidate_extra: dict[str, Any],
    generation_protocol_fingerprint: str,
) -> dict[str, Any]:
    question_id = int(question["question_id"])
    category = str(question["category"])
    temperature = (
        args.force_temperature
        if args.force_temperature is not None
        else TEMPERATURE_BY_CATEGORY.get(category, 0.7)
    )
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": str(question["turns"][0])},
    ]
    first_seed = (
        candidate_seed(args.candidate_seed, question_id, 1)
        if args.candidate_seed_mode == "request"
        else None
    )
    first_extra = dict(candidate_extra)
    if first_seed is not None:
        first_extra["seed"] = first_seed
    first_raw, first_payload = request_chat(
        client,
        address=args.candidate_server_address,
        api_key=candidate_api_key,
        model=args.candidate_model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        extra_body=first_extra,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    first = strip_thinking(first_raw) if args.strip_thinking else first_raw
    messages.extend(
        [
            # Keep the model's exact first response in its second-turn context.
            # --strip-thinking only controls what is stored and shown to the judge.
            {"role": "assistant", "content": first_raw},
            {"role": "user", "content": str(question["turns"][1])},
        ]
    )
    second_seed = (
        candidate_seed(args.candidate_seed, question_id, 2)
        if args.candidate_seed_mode == "request"
        else None
    )
    second_extra = dict(candidate_extra)
    if second_seed is not None:
        second_extra["seed"] = second_seed
    second_raw, second_payload = request_chat(
        client,
        address=args.candidate_server_address,
        api_key=candidate_api_key,
        model=args.candidate_model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        extra_body=second_extra,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    second = strip_thinking(second_raw) if args.strip_thinking else second_raw
    return {
        "question_id": question_id,
        "answer_id": uuid.uuid4().hex,
        "model_id": args.candidate_model,
        "choices": [{"index": 0, "turns": [first, second]}],
        "category": category,
        "temperature": temperature,
        "turn_seeds": [first_seed, second_seed],
        "candidate_response_models": [
            first_payload.get("model"),
            second_payload.get("model"),
        ],
        "generation_protocol_fingerprint": generation_protocol_fingerprint,
        "turn_usage": [
            usage_from_response(first_payload),
            usage_from_response(second_payload),
        ],
        "tstamp": time.time(),
    }


def generate_answers(
    questions: list[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
    client: httpx.Client,
    candidate_extra: dict[str, Any],
    generation_protocol_fingerprint: str,
) -> dict[int, dict[str, Any]]:
    active_ids = {int(question["question_id"]) for question in questions}
    answers = (
        load_answers(
            path,
            args.candidate_model,
            generation_protocol_fingerprint,
            active_ids,
        )
        if args.resume
        else {}
    )
    answers = {
        question_id: answer
        for question_id, answer in answers.items()
        if question_id in active_ids
    }
    pending = [q for q in questions if int(q["question_id"]) not in answers]
    print(
        f"Candidate generation: {len(questions) - len(pending)} resumed, {len(pending)} pending",
        flush=True,
    )
    candidate_api_key = os.environ.get(args.candidate_api_key_env, "")

    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.candidate_concurrency) as executor:
        future_map = {
            executor.submit(
                generate_one,
                q,
                args,
                client,
                candidate_api_key,
                candidate_extra,
                generation_protocol_fingerprint,
            ): int(q["question_id"])
            for q in pending
        }
        completed = len(questions) - len(pending)
        for future in as_completed(future_map):
            question_id = future_map[future]
            try:
                answer = future.result()
            except Exception as exc:  # keep successful work resumable
                failures.append((question_id, str(exc)))
                print(f"Candidate question {question_id} failed: {exc}", file=sys.stderr)
                continue
            answers[question_id] = answer
            completed += 1
            with WRITE_LOCK:
                atomic_write_jsonl(path, (answers[key] for key in sorted(answers)))
            print(f"Candidate question {question_id} complete ({completed}/{len(questions)})")
    if failures:
        summary = "; ".join(f"{qid}: {message}" for qid, message in failures[:5])
        raise RuntimeError(f"{len(failures)} candidate question(s) failed: {summary}")
    return answers


def parse_rating(text: str) -> float:
    match = RATING_PATTERN.search(text) or RATING_PATTERN_FALLBACK.search(text)
    if not match:
        return -1.0
    value = float(match.group(1))
    return value if math.isfinite(value) and 1.0 <= value <= 10.0 else -1.0


def prompt_for_judgment(
    question: dict[str, Any],
    answer: dict[str, Any],
    reference: Optional[dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    turn: int,
) -> tuple[dict[str, Any], str]:
    ref_based = str(question["category"]) in REFERENCE_CATEGORIES
    if turn == 1:
        prompt_name = "single-math-v1" if ref_based else "single-v1"
        values = {
            "question": question["turns"][0],
            "answer": answer["choices"][0]["turns"][0],
        }
        if ref_based:
            if reference is None:
                raise RuntimeError(
                    f"Missing reference answer for question {question['question_id']}"
                )
            values["ref_answer_1"] = reference["choices"][0]["turns"][0]
    else:
        prompt_name = (
            "single-math-v1-multi-turn"
            if ref_based
            else "single-v1-multi-turn"
        )
        values = {
            "question_1": question["turns"][0],
            "question_2": question["turns"][1],
            "answer_1": answer["choices"][0]["turns"][0],
            "answer_2": answer["choices"][0]["turns"][1],
        }
        if ref_based:
            if reference is None:
                raise RuntimeError(
                    f"Missing reference answer for question {question['question_id']}"
                )
            values["ref_answer_1"] = reference["choices"][0]["turns"][0]
            values["ref_answer_2"] = reference["choices"][0]["turns"][1]
    prompt = prompts[prompt_name]
    return prompt, str(prompt["prompt_template"]).format(**values)


def judgment_messages(
    question: dict[str, Any],
    answer: dict[str, Any],
    reference: Optional[dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    turn: int,
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    prompt, user_prompt = prompt_for_judgment(
        question, answer, reference, prompts, turn
    )
    messages = [
        {"role": "system", "content": str(prompt["system_prompt"])},
        {"role": "user", "content": user_prompt},
    ]
    return prompt, messages, fingerprint(messages)


def judgment_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["question_id"]), int(row["turn"])


def judge_one(
    question: dict[str, Any],
    answer: dict[str, Any],
    reference: Optional[dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    turn: int,
    args: argparse.Namespace,
    client: httpx.Client,
    judge_api_key: str,
    judge_protocol_fingerprint: str,
) -> dict[str, Any]:
    prompt, messages, input_fingerprint = judgment_messages(
        question, answer, reference, prompts, turn
    )
    judgment, payload = request_chat(
        client,
        address=args.judge_server_address,
        api_key=judge_api_key,
        model=args.judge_model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        top_p=1.0,
        extra_body=None,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    return {
        "question_id": int(question["question_id"]),
        "model": args.candidate_model,
        "judge": [args.judge_model, prompt["name"]],
        "category": str(question["category"]),
        "user_prompt": messages[1]["content"],
        "judgment": judgment,
        "score": parse_rating(judgment),
        "turn": turn,
        "judge_response_model": payload.get("model"),
        "judge_protocol_fingerprint": judge_protocol_fingerprint,
        "judgment_input_sha256": input_fingerprint,
        "tstamp": time.time(),
    }


def generate_judgments(
    questions: list[dict[str, Any]],
    answers: dict[int, dict[str, Any]],
    references: dict[int, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
    client: httpx.Client,
    judge_protocol_fingerprint: str,
) -> list[dict[str, Any]]:
    active_ids = {int(question["question_id"]) for question in questions}
    questions_by_id = {int(question["question_id"]): question for question in questions}
    existing_rows: list[dict[str, Any]] = []
    incompatible: list[tuple[int, int]] = []
    for row in read_jsonl(path) if args.resume else []:
        try:
            question_id = int(row["question_id"])
            turn = int(row["turn"])
        except (KeyError, TypeError, ValueError):
            incompatible.append((-1, -1))
            continue
        if question_id not in active_ids:
            continue
        if turn not in (1, 2):
            incompatible.append((question_id, turn))
            continue
        question = questions_by_id[question_id]
        _, _, expected_input_fingerprint = judgment_messages(
            question,
            answers[question_id],
            references.get(question_id),
            prompts,
            turn,
        )
        if (
            row.get("model") == args.candidate_model
            and isinstance(row.get("judge"), list)
            and row["judge"]
            and row["judge"][0] == args.judge_model
            and row.get("judge_protocol_fingerprint")
            == judge_protocol_fingerprint
            and row.get("judgment_input_sha256") == expected_input_fingerprint
        ):
            existing_rows.append(row)
        else:
            incompatible.append((question_id, turn))
    if incompatible:
        raise RuntimeError(
            "Cannot resume MT-Bench judging: existing judgments use a different "
            "or legacy judge protocol/input "
            f"(question/turn: {sorted(set(incompatible))[:10]}). Use a new "
            f"output directory or remove {path}."
        )
    judgments = {judgment_key(row): row for row in existing_rows}
    work = [
        (question, turn)
        for question in questions
        for turn in (1, 2)
        if (int(question["question_id"]), turn) not in judgments
    ]
    print(
        f"Judge scoring: {len(judgments)} resumed, {len(work)} pending",
        flush=True,
    )
    judge_api_key = os.environ.get(args.judge_api_key_env, "")
    failures: list[tuple[int, int, str]] = []
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as executor:
        future_map = {}
        for question, turn in work:
            question_id = int(question["question_id"])
            future = executor.submit(
                judge_one,
                question,
                answers[question_id],
                references.get(question_id),
                prompts,
                turn,
                args,
                client,
                judge_api_key,
                judge_protocol_fingerprint,
            )
            future_map[future] = (question_id, turn)
        for future in as_completed(future_map):
            question_id, turn = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                failures.append((question_id, turn, str(exc)))
                print(
                    f"Judge question {question_id} turn {turn} failed: {exc}",
                    file=sys.stderr,
                )
                continue
            judgments[(question_id, turn)] = row
            with WRITE_LOCK:
                atomic_write_jsonl(
                    path, (judgments[key] for key in sorted(judgments))
                )
            print(
                f"Judge question {question_id} turn {turn}: score={row['score']} "
                f"({len(judgments)}/{len(questions) * 2})"
            )
    if failures:
        summary = "; ".join(
            f"{qid}/turn{turn}: {message}" for qid, turn, message in failures[:5]
        )
        raise RuntimeError(f"{len(failures)} judge request(s) failed: {summary}")
    return [judgments[key] for key in sorted(judgments)]


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    clean = [value for value in values if value >= 0]
    return round(sum(clean) / len(clean), 6) if clean else None


def score_summary(
    rows: list[dict[str, Any]], expected_judgments: Optional[int] = None
) -> dict[str, Any]:
    valid_rows = [row for row in rows if float(row.get("score", -1)) >= 0]
    per_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        per_category[str(row["category"])].append(row)
    categories = {}
    for category in sorted(per_category):
        category_rows = per_category[category]
        categories[category] = {
            "overall": mean_or_none(float(row["score"]) for row in category_rows),
            "turn_1": mean_or_none(
                float(row["score"]) for row in category_rows if int(row["turn"]) == 1
            ),
            "turn_2": mean_or_none(
                float(row["score"]) for row in category_rows if int(row["turn"]) == 2
            ),
            "valid_judgments": len(category_rows),
        }
    return {
        "overall": mean_or_none(float(row["score"]) for row in valid_rows),
        "turn_1": mean_or_none(
            float(row["score"]) for row in valid_rows if int(row["turn"]) == 1
        ),
        "turn_2": mean_or_none(
            float(row["score"]) for row in valid_rows if int(row["turn"]) == 2
        ),
        "categories": categories,
        "valid_judgments": len(valid_rows),
        "invalid_judgments": len(rows) - len(valid_rows),
        "expected_judgments": (
            expected_judgments if expected_judgments is not None else len(rows)
        ),
    }


def generation_usage_summary(answers: dict[int, dict[str, Any]]) -> dict[str, Any]:
    usage_rows = [
        usage
        for answer in answers.values()
        for usage in answer.get("turn_usage", [])
        if isinstance(usage, dict)
    ]

    def average(key: str) -> Optional[float]:
        values = [float(row[key]) for row in usage_rows if row.get(key) is not None]
        return round(sum(values) / len(values), 6) if values else None

    response_models = sorted(
        {
            str(model)
            for answer in answers.values()
            for model in answer.get("candidate_response_models", [])
            if model
        }
    )

    return {
        "response_count": len(usage_rows),
        "average_nfe": average("nfe"),
        "average_prompt_tokens": average("prompt_tokens"),
        "average_completion_tokens": average("completion_tokens"),
        "average_total_tokens": average("total_tokens"),
        "response_models": response_models,
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    candidate_extra = build_candidate_extra(args)

    if args.dry_run:
        payload = {
            "benchmark": "mt-bench",
            "candidate_server_address": args.candidate_server_address,
            "candidate_model": args.candidate_model,
            "candidate_concurrency": args.candidate_concurrency,
            "judge_server_address": args.judge_server_address,
            "judge_model": args.judge_model,
            "judge_concurrency": args.judge_concurrency,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir) if output_dir else None,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "strip_thinking": args.strip_thinking,
            "candidate_seed": args.candidate_seed,
            "candidate_seed_mode": args.candidate_seed_mode,
            "candidate_tokenizer": args.candidate_tokenizer or None,
            "expected_chat_template_sha256": (
                args.expected_chat_template_sha256 or None
            ),
            "chat_template_kwargs": candidate_extra["chat_template_kwargs"],
            "candidate_generation_algorithm": args.candidate_generation_algorithm or None,
            "candidate_steps": args.candidate_steps,
            "candidate_block_length": args.candidate_block_length,
            "fastchat_commit": FASTCHAT_COMMIT,
            "network_or_files_changed": False,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    assets = ensure_assets(data_dir, args.offline)
    questions = read_jsonl(assets["question.jsonl"])
    if len(questions) != 80 or any(len(q.get("turns", [])) != 2 for q in questions):
        raise RuntimeError("Pinned MT-Bench questions must contain 80 two-turn items")
    if args.max_samples is not None:
        questions = questions[: args.max_samples]
    if args.prepare_only:
        print(f"Prepared {len(questions)} MT-Bench questions in {data_dir}")
        return 0

    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "model_answers.jsonl"
    judgments_path = output_dir / "model_judgments.jsonl"
    metrics_path = output_dir / "metrics.json"
    prompt_preflight_path = output_dir / "prompt_preflight.json"

    preflight = prompt_preflight(args, questions[0])
    generation_protocol_payload = generation_protocol(
        args, candidate_extra, preflight
    )
    generation_protocol_fingerprint = fingerprint(generation_protocol_payload)
    judge_protocol_payload = judge_protocol(
        args, generation_protocol_fingerprint
    )
    judge_protocol_fingerprint = fingerprint(judge_protocol_payload)
    atomic_write_json(prompt_preflight_path, preflight)

    if (
        "api.openai.com" in args.judge_server_address.lower()
        and not args.generation_only
        and not args.skip_judge_api_key_check
        and not os.environ.get(args.judge_api_key_env)
    ):
        raise RuntimeError(
            f"{args.judge_api_key_env} is required for the default OpenAI MT-Bench judge"
        )

    timeout = httpx.Timeout(args.candidate_timeout)
    limits = httpx.Limits(
        max_connections=max(args.candidate_concurrency, 1),
        max_keepalive_connections=max(args.candidate_concurrency, 1),
    )
    with httpx.Client(timeout=timeout, limits=limits) as candidate_client:
        answers = generate_answers(
            questions,
            answers_path,
            args,
            candidate_client,
            candidate_extra,
            generation_protocol_fingerprint,
        )
    generation_usage = generation_usage_summary(answers)

    if args.generation_only:
        metrics = {
            "status": "generation_only",
            "benchmark": "mt-bench",
            "question_count": len(questions),
            "candidate_response_count": len(questions) * 2,
            "mt_bench_score": None,
            "generation_usage": generation_usage,
            "protocol": {
                "source": "lm-sys/FastChat",
                "fastchat_commit": FASTCHAT_COMMIT,
                "generation_fingerprint": generation_protocol_fingerprint,
                "generation": generation_protocol_payload,
                "observed_candidate_response_models": generation_usage[
                    "response_models"
                ],
            },
            "artifacts": {
                "model_answers": str(answers_path),
                "prompt_preflight": str(prompt_preflight_path),
                "data_dir": str(data_dir),
            },
        }
        atomic_write_json(metrics_path, metrics)
        print(f"Generation-only MT-Bench artifacts: {output_dir}")
        return 0

    references = {
        int(row["question_id"]): row
        for row in read_jsonl(assets["reference_answer_gpt-4.jsonl"])
    }
    prompts = {
        str(row["name"]): row for row in read_jsonl(assets["judge_prompts.jsonl"])
    }
    required_prompts = {
        "single-v1",
        "single-math-v1",
        "single-v1-multi-turn",
        "single-math-v1-multi-turn",
    }
    missing_prompts = required_prompts - prompts.keys()
    if missing_prompts:
        raise RuntimeError(f"Pinned judge prompt file is missing: {sorted(missing_prompts)}")

    judge_timeout = httpx.Timeout(args.judge_timeout)
    judge_limits = httpx.Limits(
        max_connections=max(args.judge_concurrency, 1),
        max_keepalive_connections=max(args.judge_concurrency, 1),
    )
    with httpx.Client(timeout=judge_timeout, limits=judge_limits) as judge_client:
        judgments = generate_judgments(
            questions,
            answers,
            references,
            prompts,
            judgments_path,
            args,
            judge_client,
            judge_protocol_fingerprint,
        )

    scores = score_summary(judgments, expected_judgments=len(questions) * 2)
    observed_judge_models = sorted(
        {
            str(row["judge_response_model"])
            for row in judgments
            if row.get("judge_response_model")
        }
    )
    metrics = {
        "status": "complete" if scores["invalid_judgments"] == 0 else "partial",
        "benchmark": "mt-bench",
        "metric": "fastchat_single_answer_grading",
        "mt_bench_score": scores["overall"],
        "score": scores["overall"],
        "question_count": len(questions),
        "candidate_response_count": len(questions) * 2,
        "average_nfe": generation_usage["average_nfe"],
        "generation_usage": generation_usage,
        "mt_bench": scores,
        "protocol": {
            "source": "lm-sys/FastChat",
            "fastchat_commit": FASTCHAT_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "judge_mode": "single",
            "judge_model": args.judge_model,
            "judge_server_address": args.judge_server_address,
            "reference_answer_model": "gpt-4",
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "force_temperature": args.force_temperature,
            "category_temperatures": TEMPERATURE_BY_CATEGORY,
            "strip_thinking": args.strip_thinking,
            "asset_sha256": {
                name: metadata["sha256"] for name, metadata in ASSETS.items()
            },
            "generation_fingerprint": generation_protocol_fingerprint,
            "judge_fingerprint": judge_protocol_fingerprint,
            "generation": generation_protocol_payload,
            "judge": judge_protocol_payload,
            "observed_candidate_response_models": generation_usage[
                "response_models"
            ],
            "observed_judge_response_models": observed_judge_models,
        },
        "artifacts": {
            "model_answers": str(answers_path),
            "model_judgments": str(judgments_path),
            "prompt_preflight": str(prompt_preflight_path),
            "data_dir": str(data_dir),
        },
    }
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"MT-Bench metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
