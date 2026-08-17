#!/usr/bin/env python3
"""OpenAI-compatible timing proxy for NeMo-Skills-on-SGLang eval.

The proxy accepts ordinary non-streaming /v1/chat/completions requests from
NeMo-Skills, forwards them to SGLang as streaming requests, records TTFT and
TPOT, then returns a normal non-streaming OpenAI-style response to the caller.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - import guard for runtime clarity
    raise SystemExit(
        "openai_timing_proxy.py requires fastapi, uvicorn, and httpx. "
        "Install them in the evaluation environment."
    ) from exc


STANDARD_CHAT_KEYS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "stop",
    "seed",
    "n",
    "logprobs",
    "top_logprobs",
    "response_format",
    "tools",
    "tool_choice",
    "user",
}


def _now() -> float:
    return time.perf_counter()


def _unix_now() -> int:
    return int(time.time())


def _normalize_base_url(url: str) -> tuple[str, str]:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        root = base[:-3]
    else:
        root = base
        base = base + "/v1"
    return root.rstrip("/"), base.rstrip("/")


def _json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _extract_benchmark(body: dict[str, Any], default: str) -> str:
    extra = body.get("extra_body")
    if isinstance(extra, dict) and extra.get("benchmark_name"):
        return str(extra["benchmark_name"])
    if body.get("benchmark_name"):
        return str(body["benchmark_name"])
    return default


def _clean_body_for_sglang(body: dict[str, Any]) -> dict[str, Any]:
    """Keep OpenAI/SGLang-compatible fields and drop custom DLM extras."""
    clean = {k: v for k, v in body.items() if k in STANDARD_CHAT_KEYS}
    clean["stream"] = True
    clean["stream_options"] = {"include_usage": True}
    return clean


def _choice_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    delta = first.get("delta")
    if isinstance(delta, dict):
        value = delta.get("content")
        if value is not None:
            return str(value)
    value = first.get("text")
    return "" if value is None else str(value)


def _finish_reason(chunk: dict[str, Any]) -> Optional[str]:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    reason = choices[0].get("finish_reason")
    return None if reason is None else str(reason)


class TimingProxy:
    def __init__(
        self,
        upstream_base_url: str,
        timing_log: str,
        default_benchmark: str,
        max_concurrency: int,
        timeout: float,
    ) -> None:
        self.upstream_root, self.upstream_base = _normalize_base_url(upstream_base_url)
        self.timing_log = Path(timing_log)
        self.default_benchmark = default_benchmark
        self.timeout = timeout
        self.semaphore: Optional[asyncio.Semaphore]
        self.semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        )
        self.client = httpx.AsyncClient(timeout=None)
        self.timing_log.parent.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> JSONResponse:
        try:
            resp = await self.client.get(f"{self.upstream_root}/health", timeout=5.0)
            if resp.status_code >= 400:
                return JSONResponse(
                    {"status": "upstream_error", "code": resp.status_code},
                    status_code=503,
                )
        except Exception as exc:
            return JSONResponse(
                {"status": "upstream_unreachable", "error": str(exc)},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "upstream": self.upstream_base})

    async def models(self) -> JSONResponse:
        try:
            resp = await self.client.get(f"{self.upstream_base}/models", timeout=30.0)
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    async def chat_completions(self, body: dict[str, Any]) -> JSONResponse:
        if self.semaphore is None:
            return await self._chat_completions_unlocked(body)
        async with self.semaphore:
            return await self._chat_completions_unlocked(body)

    async def _chat_completions_unlocked(self, body: dict[str, Any]) -> JSONResponse:
        benchmark = _extract_benchmark(body, self.default_benchmark)
        forward_body = _clean_body_for_sglang(body)
        start = _now()

        try:
            result = await self._stream_from_upstream(forward_body)
        except Exception as exc:
            end = _now()
            self._append_timing(
                {
                    "benchmark": benchmark,
                    "ok": False,
                    "error": str(exc),
                    "start_time_s": round(start, 6),
                    "end_time_s": round(end, 6),
                    "latency_s": round(end - start, 6),
                    "created": _unix_now(),
                }
            )
            return JSONResponse({"error": {"message": str(exc)}}, status_code=502)

        end = _now()
        content = result["content"]
        usage = result["usage"] or {}
        completion_tokens = _to_int(usage.get("completion_tokens"))
        prompt_tokens = _to_int(usage.get("prompt_tokens"))
        ttft_s = None
        if result["first_token_at"] is not None:
            ttft_s = result["first_token_at"] - start

        tpot_s = None
        decode_s = None
        if ttft_s is not None:
            decode_s = end - result["first_token_at"]
            if completion_tokens and completion_tokens > 1:
                tpot_s = decode_s / (completion_tokens - 1)
            elif completion_tokens == 1:
                tpot_s = 0.0

        output_tokens_per_s = None
        if completion_tokens and end > start:
            output_tokens_per_s = completion_tokens / (end - start)

        record = {
            "benchmark": benchmark,
            "ok": True,
            "request_id": result["id"],
            "created": _unix_now(),
            "model": forward_body.get("model"),
            "start_time_s": round(start, 6),
            "end_time_s": round(end, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": _to_int(usage.get("total_tokens")),
            "latency_s": round(end - start, 6),
            "ttft_s": _round_optional(ttft_s),
            "decode_s": _round_optional(decode_s),
            "tpot_s": _round_optional(tpot_s),
            "output_tokens_per_s": _round_optional(output_tokens_per_s),
            "stream_chunks": result["chunks"],
            "finish_reason": result["finish_reason"],
        }
        self._append_timing(record)

        response = {
            "id": result["id"],
            "object": "chat.completion",
            "created": _unix_now(),
            "model": forward_body.get("model", result.get("model") or "sglang"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": result["finish_reason"] or "stop",
                }
            ],
            "usage": usage,
        }
        return JSONResponse(response)

    async def _stream_from_upstream(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.upstream_base}/chat/completions"
        chunks = 0
        content_parts: list[str] = []
        usage: Optional[dict[str, Any]] = None
        first_token_at: Optional[float] = None
        finish_reason: Optional[str] = None
        response_id = f"chatcmpl-proxy-{uuid.uuid4().hex}"
        response_model: Optional[str] = None

        async with self.client.stream(
            "POST",
            url,
            json=body,
            timeout=self.timeout,
        ) as resp:
            if resp.status_code >= 400:
                text = await resp.aread()
                raise RuntimeError(
                    f"upstream HTTP {resp.status_code}: "
                    f"{text.decode('utf-8', errors='replace')[:500]}"
                )

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if chunk.get("id"):
                    response_id = str(chunk["id"])
                if chunk.get("model"):
                    response_model = str(chunk["model"])
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]

                delta = _choice_delta_text(chunk)
                if delta:
                    if first_token_at is None:
                        first_token_at = _now()
                    chunks += 1
                    content_parts.append(delta)

                reason = _finish_reason(chunk)
                if reason:
                    finish_reason = reason

        return {
            "id": response_id,
            "model": response_model,
            "content": "".join(content_parts),
            "usage": usage,
            "first_token_at": first_token_at,
            "chunks": chunks,
            "finish_reason": finish_reason,
        }

    def _append_timing(self, record: dict[str, Any]) -> None:
        with self.timing_log.open("a", encoding="utf-8") as f:
            f.write(_json_dumps(record) + "\n")


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def create_app(args: argparse.Namespace) -> FastAPI:
    proxy = TimingProxy(
        upstream_base_url=args.upstream_base_url,
        timing_log=args.timing_log,
        default_benchmark=args.default_benchmark,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
    )
    app = FastAPI()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await proxy.close()

    @app.get("/health")
    async def _health() -> JSONResponse:
        return await proxy.health()

    @app.get("/v1/models")
    async def _models() -> JSONResponse:
        return await proxy.models()

    @app.post("/v1/chat/completions")
    async def _chat(request: Request) -> JSONResponse:
        body = await request.json()
        return await proxy.chat_completions(body)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--timing-log", required=True)
    parser.add_argument("--default-benchmark", default="")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent in-flight requests forwarded to SGLang. 0 means unlimited.",
    )
    parser.add_argument("--timeout", type=float, default=12000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
