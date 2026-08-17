#!/usr/bin/env python3
"""Run pinned AlpacaEval 2.0 against OpenAI-compatible candidate/judge APIs.

Candidate answers are generated through the selected NLD backend. Pairwise
annotation, position randomization, weighted logprob parsing, and the official
length-controlled metric are delegated to the pinned upstream alpaca_eval
package instead of being reimplemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import httpx


ALPACA_EVAL_VERSION = "0.6.6"
ALPACA_EVAL_DATA_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
PROTOCOL_VERSION = 1
DEFAULT_ANNOTATOR = "weighted_alpaca_eval_gpt4_turbo"
DEFAULT_JUDGE_MODEL = "gpt-4-1106-preview"
OFFICIAL_PROMPT_SHA256 = (
    "784227e6dc2832fc08c43d2c8ea3a308e7523780187a1aaad2f85e30bac85f62"
)

HF_RESOLVE_BASE = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/"
    f"{ALPACA_EVAL_DATA_REVISION}"
)
ASSETS = {
    "alpaca_eval_gpt4_baseline.json": {
        "url": f"{HF_RESOLVE_BASE}/alpaca_eval_gpt4_baseline.json?download=true",
        "sha256": "83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4",
    },
    "df_gamed.csv": {
        "url": f"{HF_RESOLVE_BASE}/df_gamed.csv?download=true",
        "sha256": "97aeec3f1c7de6dee6fd31fe66f1702623f4614c3efe1c6fc5f4927cd5fd674d",
    },
    "instruction_difficulty.csv": {
        "url": f"{HF_RESOLVE_BASE}/instruction_difficulty.csv?download=true",
        "sha256": "e28d875bb334f75e17acd1e4ed659b261860e3db379cf5de29060301bed0a18b",
    },
}
RUNTIME_WHEELS = {
    "alpaca_eval-0.6.6-py3-none-any.whl": {
        "url": (
            "https://files.pythonhosted.org/packages/6c/c2/"
            "aa7b54eaca10603efdcac7092c43889e7316db6e5781d300fde57a5ec95e/"
            "alpaca_eval-0.6.6-py3-none-any.whl"
        ),
        "sha256": "8f4f218b8a1d7ef379491222e90f38446ca327930b51416d55306663cf85f28c",
    },
    "patsy-1.0.1-py2.py3-none-any.whl": {
        "url": (
            "https://files.pythonhosted.org/packages/87/2b/"
            "b50d3d08ea0fc419c183a84210571eba005328efa62b6b98bc28e9ead32a/"
            "patsy-1.0.1-py2.py3-none-any.whl"
        ),
        "sha256": "751fb38f9e97e62312e921a1954b81e1bb2bcda4f5eeabaf94db251ee791509c",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-server-address", default="")
    parser.add_argument("--candidate-model", default="nemotron-labs-diffusion-8b")
    parser.add_argument("--candidate-api-key-env", default="NLD_CANDIDATE_API_KEY")
    parser.add_argument("--candidate-concurrency", type=int, default=1)
    parser.add_argument("--candidate-timeout", type=float, default=12000.0)
    parser.add_argument("--candidate-extra-body-json", default="{}")
    parser.add_argument("--candidate-tokenizer", default="")
    parser.add_argument("--candidate-tokenizer-revision", default="")
    parser.add_argument("--expected-chat-template-sha256", default="")
    parser.add_argument("--candidate-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--candidate-truncate-history-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--candidate-seed-mode", choices=("request", "none"), default="request")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--strip-thinking", action="store_true")

    parser.add_argument("--annotator", default=DEFAULT_ANNOTATOR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-server-address", default="https://api.openai.com/v1")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--skip-judge-api-key-check", action="store_true")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.candidate_concurrency <= 0 or args.judge_concurrency <= 0:
        parser.error("candidate/judge concurrency must be positive")
    if args.max_tokens <= 0 or args.max_retries <= 0:
        parser.error("--max-tokens and --max-retries must be positive")
    if args.candidate_seed < 0:
        parser.error("--candidate-seed must be non-negative")
    if args.temperature < 0 or not 0.0 <= args.top_p <= 1.0:
        parser.error("temperature must be non-negative and top_p must be in [0, 1]")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.annotator != DEFAULT_ANNOTATOR:
        parser.error(
            f"only the pinned official AlpacaEval 2 annotator is supported: {DEFAULT_ANNOTATOR}"
        )
    if args.expected_chat_template_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.expected_chat_template_sha256
    ):
        parser.error("--expected-chat-template-sha256 must contain 64 hex digits")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if not (args.prepare_only or args.dry_run):
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
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(item)
    return rows


def download_pinned(destination: Path, metadata: dict[str, str], offline: bool) -> None:
    expected = metadata["sha256"]
    if destination.is_file() and sha256(destination) == expected:
        return
    if offline:
        raise RuntimeError(f"Missing or invalid pinned asset in offline mode: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pinned AlpacaEval asset: {destination.name}", flush=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        with urllib.request.urlopen(metadata["url"], timeout=180) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {destination.name}: expected {expected}, downloaded {actual}"
        )
    os.replace(temporary, destination)


def ensure_assets(data_dir: Path, offline: bool) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for name, metadata in ASSETS.items():
        destination = data_dir / name
        download_pinned(destination, metadata, offline)
        resolved[name] = destination

    references = read_json(resolved["alpaca_eval_gpt4_baseline.json"])
    if not isinstance(references, list) or len(references) != 805:
        raise RuntimeError("Pinned AlpacaEval reference file must contain exactly 805 rows")
    if len({str(row.get("instruction", "")) for row in references}) != 805:
        raise RuntimeError("Pinned AlpacaEval reference instructions are not unique")

    atomic_write_json(
        data_dir / "protocol.json",
        {
            "benchmark": "alpaca-eval-2.0",
            "source": "tatsu-lab/alpaca_eval",
            "alpaca_eval_version": ALPACA_EVAL_VERSION,
            "dataset_revision": ALPACA_EVAL_DATA_REVISION,
            "reference_generator": "gpt4_1106_preview",
            "num_examples": 805,
            "assets": ASSETS,
        },
    )
    return resolved


def safe_extract_wheel(wheel: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"Unsafe wheel member: {member.filename}")
        archive.extractall(destination)


def ensure_official_runtime(data_dir: Path, offline: bool) -> Path:
    runtime_dir = data_dir / "runtime"
    wheels_dir = runtime_dir / "wheels"
    site_dir = runtime_dir / "site"
    for name, metadata in RUNTIME_WHEELS.items():
        download_pinned(wheels_dir / name, metadata, offline)

    marker = site_dir / "nld_alpaca_eval_runtime.json"
    expected_marker = {
        "alpaca_eval_version": ALPACA_EVAL_VERSION,
        "wheels": {name: metadata["sha256"] for name, metadata in RUNTIME_WHEELS.items()},
    }
    marker_ok = marker.is_file() and read_json(marker) == expected_marker
    if not marker_ok:
        temporary = Path(tempfile.mkdtemp(prefix="site.", dir=runtime_dir))
        try:
            for name in RUNTIME_WHEELS:
                safe_extract_wheel(wheels_dir / name, temporary)
            atomic_write_json(temporary / marker.name, expected_marker)
            if site_dir.exists():
                shutil.rmtree(site_dir)
            os.replace(temporary, site_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return site_dir


def load_references(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows = read_json(path)
    if max_samples is None or max_samples >= len(rows):
        return rows
    # Match the official evaluate(max_instances=...) fixed-seed selection policy.
    import pandas as pd

    return (
        pd.DataFrame(rows)
        .sample(frac=1, random_state=123)
        .iloc[:max_samples]
        .to_dict(orient="records")
    )


def build_messages(system_prompt: str, instruction: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    return messages


def build_candidate_extra(args: argparse.Namespace) -> dict[str, Any]:
    try:
        extra = json.loads(args.candidate_extra_body_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid --candidate-extra-body-json: {exc}") from exc
    if not isinstance(extra, dict):
        raise RuntimeError("--candidate-extra-body-json must decode to an object")
    template_kwargs = extra.get("chat_template_kwargs") or {}
    if not isinstance(template_kwargs, dict):
        raise RuntimeError("candidate chat_template_kwargs must be an object")
    template_kwargs = dict(template_kwargs)
    template_kwargs.update(
        {
            "enable_thinking": bool(args.candidate_enable_thinking),
            "truncate_history_thinking": bool(args.candidate_truncate_history_thinking),
        }
    )
    extra["chat_template_kwargs"] = template_kwargs
    extra["benchmark_name"] = "alpaca-eval"
    return extra


def prompt_preflight(args: argparse.Namespace, instruction: str) -> dict[str, Any]:
    if not args.candidate_tokenizer:
        if args.expected_chat_template_sha256:
            raise RuntimeError(
                "--expected-chat-template-sha256 requires --candidate-tokenizer"
            )
        return {"status": "not_requested", "chat_template_sha256": None}

    from transformers import AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.candidate_tokenizer_revision:
        tokenizer_kwargs["revision"] = args.candidate_tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(args.candidate_tokenizer, **tokenizer_kwargs)
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise RuntimeError(f"Tokenizer has no chat template: {args.candidate_tokenizer}")
    template_text = template if isinstance(template, str) else canonical_json(template)
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
    expected = args.expected_chat_template_sha256.lower()
    if expected and expected != template_hash:
        raise RuntimeError(
            f"Chat-template SHA-256 mismatch: expected {expected}, found {template_hash}"
        )
    messages = build_messages(args.system_prompt, instruction)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(args.candidate_enable_thinking),
        truncate_history_thinking=bool(args.candidate_truncate_history_thinking),
    )
    token_ids = tokenizer.encode(rendered)
    return {
        "status": "passed",
        "tokenizer": args.candidate_tokenizer,
        "tokenizer_revision": args.candidate_tokenizer_revision or None,
        "transformers_version": __import__("transformers").__version__,
        "chat_template_sha256": template_hash,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "token_ids_sha256": fingerprint(token_ids),
        "prompt_tokens": len(token_ids),
        "template_kwargs": {
            "enable_thinking": bool(args.candidate_enable_thinking),
            "truncate_history_thinking": bool(args.candidate_truncate_history_thinking),
        },
    }


def normalize_chat_url(address: str) -> str:
    return address.rstrip("/") + "/chat/completions"


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_one(
    args: argparse.Namespace,
    reference: dict[str, Any],
    source_index: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": args.candidate_model,
        "messages": build_messages(args.system_prompt, str(reference["instruction"])),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        **extra_body,
    }
    if args.candidate_seed_mode == "request":
        request["seed"] = args.candidate_seed + source_index
    api_key = os.environ.get(args.candidate_api_key_env, "EMPTY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(1, args.max_retries + 1):
        try:
            with httpx.Client(timeout=args.candidate_timeout) as client:
                response = client.post(
                    normalize_chat_url(args.candidate_server_address),
                    headers=headers,
                    json=request,
                )
                response.raise_for_status()
                payload = response.json()
            output = str(payload["choices"][0]["message"]["content"])
            if args.strip_thinking:
                output = strip_thinking(output)
            return {
                "source_index": source_index,
                "instruction": str(reference["instruction"]),
                "dataset": str(reference.get("dataset", "")),
                "output": output,
                "generator": args.candidate_model,
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "response_model": payload.get("model"),
                "usage": payload.get("usage") or {},
                "elapsed_s": time.perf_counter() - started,
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - request retries are intentional
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_backoff * attempt)
    raise RuntimeError(
        f"Candidate generation failed for sample {source_index}: {last_error}"
    )


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {
        key: sum(int((row.get("usage") or {}).get(key) or 0) for row in rows)
        for key in keys
    }
    return {
        **totals,
        "response_models": sorted(
            {str(row["response_model"]) for row in rows if row.get("response_model")}
        ),
        "finish_reasons": {
            reason: sum(1 for row in rows if str(row.get("finish_reason")) == reason)
            for reason in sorted({str(row.get("finish_reason")) for row in rows})
        },
        "request_elapsed_s_sum": sum(float(row.get("elapsed_s") or 0) for row in rows),
    }


def generation_protocol(
    args: argparse.Namespace,
    preflight: dict[str, Any],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_model": args.candidate_model,
        "candidate_server_address": args.candidate_server_address,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "system_prompt": args.system_prompt,
        "strip_thinking": args.strip_thinking,
        "seed": args.candidate_seed,
        "seed_mode": args.candidate_seed_mode,
        "extra_body": extra_body,
        "chat_template_preflight": preflight,
        "dataset_revision": ALPACA_EVAL_DATA_REVISION,
    }


def generate_candidates(
    args: argparse.Namespace,
    references: list[dict[str, Any]],
    output_dir: Path,
    generation_fp: str,
    extra_body: dict[str, Any],
) -> list[dict[str, Any]]:
    details_path = output_dir / "candidate_generations.jsonl"
    protocol_path = output_dir / "generation_protocol.json"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and protocol_path.is_file():
        previous = read_json(protocol_path)
        if previous.get("fingerprint") != generation_fp:
            raise RuntimeError(
                "Cannot resume AlpacaEval: generation protocol fingerprint changed. "
                "Use a fresh output directory or --overwrite."
            )
        existing = {str(row.get("instruction")): row for row in read_jsonl(details_path)}
    elif args.resume and details_path.is_file():
        raise RuntimeError("Cannot resume AlpacaEval without generation_protocol.json")
    elif args.overwrite:
        details_path.unlink(missing_ok=True)
        protocol_path.unlink(missing_ok=True)
    elif details_path.exists() or protocol_path.exists():
        raise RuntimeError("Output already exists; pass --resume, --overwrite, or use a new directory")

    results: dict[str, dict[str, Any]] = {
        key: row
        for key, row in existing.items()
        if key in {str(reference["instruction"]) for reference in references}
    }
    pending = [
        (index, reference)
        for index, reference in enumerate(references)
        if str(reference["instruction"]) not in results
    ]
    if pending:
        print(f"Generating {len(pending)} AlpacaEval candidate answers...", flush=True)
        with ThreadPoolExecutor(max_workers=args.candidate_concurrency) as pool:
            futures = {
                pool.submit(generate_one, args, reference, index, extra_body): reference
                for index, reference in pending
            }
            for future in as_completed(futures):
                row = future.result()
                results[row["instruction"]] = row
                ordered_partial = [
                    results[str(reference["instruction"])]
                    for reference in references
                    if str(reference["instruction"]) in results
                ]
                atomic_write_jsonl(details_path, ordered_partial)

    ordered = [results[str(reference["instruction"])] for reference in references]
    atomic_write_jsonl(details_path, ordered)
    atomic_write_json(protocol_path, {"fingerprint": generation_fp})
    atomic_write_json(
        output_dir / "model_outputs.json",
        [
            {
                "dataset": row["dataset"],
                "instruction": row["instruction"],
                "output": row["output"],
                "generator": row["generator"],
            }
            for row in ordered
        ],
    )
    return ordered


def write_official_evaluator_config(
    site_dir: Path, output_dir: Path, judge_model: str
) -> Path:
    prompt_source = (
        site_dir
        / "alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt"
    )
    if not prompt_source.is_file() or sha256(prompt_source) != OFFICIAL_PROMPT_SHA256:
        raise RuntimeError("Pinned official AlpacaEval judge prompt is missing or has changed")
    config_root = output_dir / "official_evaluator_config"
    config_dir = config_root / DEFAULT_ANNOTATOR
    prompt_path = config_dir / "alpaca_eval_clf.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_source, prompt_path)
    config = {
        DEFAULT_ANNOTATOR: {
            "prompt_template": f"{DEFAULT_ANNOTATOR}/alpaca_eval_clf.txt",
            "fn_completions": "openai_completions",
            "completions_kwargs": {
                "model_name": judge_model,
                "max_tokens": 1,
                "temperature": 1,
                "logprobs": True,
                "top_logprobs": 5,
            },
            "fn_completion_parser": "logprob_parser",
            "completion_parser_kwargs": {
                "numerator_token": "m",
                "denominator_tokens": ["m", "M"],
                "is_binarize": False,
            },
            "completion_key": "completions_all",
            "batch_size": 1,
        }
    }
    config_path = config_dir / "configs.yaml"
    # JSON is a YAML subset and avoids adding another serializer dependency.
    atomic_write_json(config_path, config)
    shutil.copyfile(prompt_path, output_dir / "official_judge_prompt.txt")
    atomic_write_json(output_dir / "official_evaluator_config.json", config)
    return config_path


def run_official_evaluation(
    args: argparse.Namespace,
    site_dir: Path,
    assets: dict[str, Path],
    references: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    judge_key = os.environ.get(args.judge_api_key_env, "")
    if not judge_key and not args.skip_judge_api_key_check:
        raise RuntimeError(
            f"Judge API key environment variable is empty: {args.judge_api_key_env}"
        )
    os.environ["OPENAI_API_KEY"] = judge_key or "EMPTY"
    os.environ["OPENAI_API_BASE"] = args.judge_server_address.rstrip("/")
    os.environ["OPENAI_MAX_CONCURRENCY"] = str(args.judge_concurrency)
    sys.path.insert(0, str(site_dir))

    import pandas as pd
    from alpaca_eval import main as alpaca_main
    from alpaca_eval import metrics as alpaca_metrics
    from alpaca_eval.metrics import glm_winrate

    glm_winrate.hf_hub_download = lambda *unused_args, **unused_kwargs: str(
        assets["df_gamed.csv"]
    )

    config_path = write_official_evaluator_config(site_dir, output_dir, args.judge_model)
    if args.overwrite:
        (config_path.parent / "annotations_seed0_configs.json").unlink(missing_ok=True)
    candidate_df = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "instruction": row["instruction"],
                "output": row["output"],
                "generator": row["generator"],
            }
            for row in candidates
        ]
    )
    reference_df = pd.DataFrame(references)
    delta_lengths = candidate_df["output"].str.len() - reference_df["output"].str.len()
    use_lc_metric = len(candidate_df) >= 2 and float(delta_lengths.std()) != 0.0
    metric_name = "get_length_controlled_winrate" if use_lc_metric else "get_winrate"
    metric_kwargs = {"save_weights_dir": None} if use_lc_metric else None
    official_results = output_dir / "official_results"
    leaderboard, annotations = alpaca_main.evaluate(
        model_outputs=candidate_df,
        reference_outputs=reference_df,
        annotators_config=config_path,
        name=args.candidate_model,
        output_path=official_results,
        precomputed_leaderboard=None,
        is_overwrite_leaderboard=True,
        leaderboard_mode_to_print=None,
        is_return_instead_of_print=True,
        fn_metric=getattr(alpaca_metrics, metric_name),
        metric_kwargs=metric_kwargs,
        sort_by="length_controlled_winrate" if use_lc_metric else "win_rate",
        is_cache_leaderboard=False,
    )
    shutil.copyfile(official_results / "annotations.json", output_dir / "annotations.json")
    shutil.copyfile(official_results / "leaderboard.csv", output_dir / "leaderboard.csv")
    row = leaderboard.loc[args.candidate_model].to_dict()
    clean = {
        key: (None if isinstance(value, float) and math.isnan(value) else value.item() if hasattr(value, "item") else value)
        for key, value in row.items()
    }
    clean["metric_function"] = metric_name
    clean["length_controlled_metric_status"] = (
        "computed" if use_lc_metric else "skipped_insufficient_or_constant_length_sample"
    )
    clean["annotation_count"] = len(annotations)
    return clean, str(config_path)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    assets = ensure_assets(data_dir, args.offline)
    site_dir = ensure_official_runtime(data_dir, args.offline)
    if args.prepare_only:
        print(f"Prepared pinned AlpacaEval data/runtime in {data_dir}")
        return 0

    references = load_references(
        assets["alpaca_eval_gpt4_baseline.json"], args.max_samples
    )
    extra_body = build_candidate_extra(args)
    preflight = prompt_preflight(args, str(references[0]["instruction"]))
    generation = generation_protocol(args, preflight, extra_body)
    generation_fp = fingerprint(generation)
    judge_protocol = {
        "annotator": args.annotator,
        "judge_model": args.judge_model,
        "judge_server_address": args.judge_server_address,
        "judge_prompt_sha256": OFFICIAL_PROMPT_SHA256,
        "max_tokens": 1,
        "temperature": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "parser": "logprob_parser(m / [m, M], non-binarized)",
        "position_randomization": "official PairwiseAnnotator deterministic randomization",
    }
    judge_fp = fingerprint(judge_protocol)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "benchmark": "alpaca-eval-2.0",
                    "examples": len(references),
                    "generation_fingerprint": generation_fp,
                    "judge_fingerprint": judge_fp,
                    "generation": generation,
                    "judge": judge_protocol,
                    "data_dir": str(data_dir),
                    "official_runtime": str(site_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "chat_template_preflight.json", preflight)
    candidates = generate_candidates(
        args, references, output_dir, generation_fp, extra_body
    )
    atomic_write_json(
        output_dir / "generation_protocol.json",
        {"fingerprint": generation_fp, "protocol": generation},
    )
    if args.generation_only:
        print(f"Generated {len(candidates)} answers in {output_dir}")
        return 0

    judge_protocol_path = output_dir / "judge_protocol.json"
    if args.resume and judge_protocol_path.is_file():
        previous = read_json(judge_protocol_path)
        if previous.get("fingerprint") != judge_fp:
            raise RuntimeError(
                "Cannot resume AlpacaEval: judge protocol fingerprint changed. "
                "Use a fresh output directory or --overwrite."
            )
    elif args.resume and (output_dir / "annotations.json").is_file():
        raise RuntimeError("Cannot resume AlpacaEval annotations without judge_protocol.json")
    elif args.overwrite:
        for path in (
            output_dir / "annotations.json",
            output_dir / "leaderboard.csv",
            output_dir / "metrics.json",
            judge_protocol_path,
        ):
            path.unlink(missing_ok=True)
    elif (output_dir / "annotations.json").exists():
        raise RuntimeError("Judgments already exist; pass --resume or --overwrite")
    atomic_write_json(
        judge_protocol_path,
        {"fingerprint": judge_fp, "protocol": judge_protocol},
    )

    official_metrics, _evaluator_config = run_official_evaluation(
        args, site_dir, assets, references, candidates, output_dir
    )
    metrics = {
        "benchmark": "alpaca-eval-2.0",
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "num_examples": len(candidates),
        "full_dataset_examples": 805,
        "is_full_formal_run": len(candidates) == 805,
        "alpaca_eval_win_rate": official_metrics.get("win_rate"),
        "alpaca_eval_length_controlled_win_rate": official_metrics.get(
            "length_controlled_winrate"
        ),
        "official_metrics": official_metrics,
        "generation": {
            "fingerprint": generation_fp,
            "protocol": generation,
            "usage": aggregate_usage(candidates),
        },
        "judge": {"fingerprint": judge_fp, "protocol": judge_protocol},
        "pinned_upstream": {
            "alpaca_eval_version": ALPACA_EVAL_VERSION,
            "dataset_revision": ALPACA_EVAL_DATA_REVISION,
            "assets": {
                name: {"sha256": metadata["sha256"], "url": metadata["url"]}
                for name, metadata in ASSETS.items()
            },
            "runtime_wheels": RUNTIME_WHEELS,
        },
        "artifacts": {
            "data_dir": str(data_dir),
            "model_outputs": str(output_dir / "model_outputs.json"),
            "candidate_generations": str(output_dir / "candidate_generations.jsonl"),
            "annotations": str(output_dir / "annotations.json"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
            "chat_template_preflight": str(output_dir / "chat_template_preflight.json"),
            "generation_protocol": str(output_dir / "generation_protocol.json"),
            "judge_protocol": str(judge_protocol_path),
            "official_evaluator_config": str(output_dir / "official_evaluator_config.json"),
            "official_judge_prompt": str(output_dir / "official_judge_prompt.txt"),
        },
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    print(
        f"AlpacaEval completed: raw={metrics['alpaca_eval_win_rate']}, "
        f"length_controlled={metrics['alpaca_eval_length_controlled_win_rate']}"
    )
    print(f"Metrics: {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
