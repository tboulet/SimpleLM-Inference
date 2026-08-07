"""FastAPI OpenAI-compatible server.

Wraps a single `Backend` instance and exposes `/v1/models` +
`/v1/chat/completions`. Tool-call parsing is dispatched via
`simplelm.tools.get_parser`.
"""
from __future__ import annotations

import inspect
import json as json_module
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from simplelm.backends.base import Backend, ContextOverflowError
from simplelm.schema import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelList,
    Usage,
)
from simplelm.tools import get_parser


# Process-global backend slot. Set by `serve()` before app start.
_BACKEND: dict[str, Optional[Backend]] = {"backend": None}
_TOOL_PARSER_NAME = os.environ.get("SIMPLELM_TOOL_PARSER", "noop")

# Per-request metrics sink (one JSON line per completion). Set SIMPLELM_METRICS=0
# to disable. Path override via SIMPLELM_METRICS_FILE.
_METRICS_FILE = os.environ.get(
    "SIMPLELM_METRICS_FILE", os.path.expanduser("~/.simplelm/metrics.jsonl")
)


def _log_metrics(model: str, result) -> None:
    """Append one metrics line. Fail-safe: never breaks serving."""
    if os.environ.get("SIMPLELM_METRICS", "1") == "0":
        return
    try:
        decode_s = (result.decode_ms or 0) / 1000.0
        rec = {
            "ts": time.time(),
            "model": model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "prefill_ms": round(result.prefill_ms, 1) if result.prefill_ms else None,
            "decode_ms": round(result.decode_ms, 1) if result.decode_ms else None,
            "decode_tok_s": round(result.completion_tokens / decode_s, 2)
            if decode_s and result.completion_tokens else None,
            "finish_reason": result.finish_reason,
        }
        os.makedirs(os.path.dirname(_METRICS_FILE), exist_ok=True)
        with open(_METRICS_FILE, "a") as fh:
            fh.write(json_module.dumps(rec) + "\n")
    except Exception:  # metrics must never break serving
        pass


def _declared_tool_names(tools_arg) -> set[str]:
    names: set[str] = set()
    for t in tools_arg or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if name:
            names.add(name)
    return names


def _extract_tool_calls(parser, text: str, tools_arg, tool_choice) -> tuple[str, list[dict]]:
    """Run `parser` over `text`, returning (content, tool_calls).

    `tool_choice == "none"` disables extraction (OpenAI semantics: the
    caller forbids a tool call). Otherwise the declared tool names are
    threaded to parsers that accept them so a call with inline arguments
    (no ``arguments`` wrapper) can be disambiguated from plain JSON data.
    """
    if tool_choice == "none":
        return text, []
    if "tool_names" in inspect.signature(parser).parameters:
        return parser(text, tool_names=_declared_tool_names(tools_arg))
    return parser(text)


def set_backend(backend: Backend) -> None:
    _BACKEND["backend"] = backend


def _get_backend() -> Backend:
    b = _BACKEND["backend"]
    if b is None:
        raise HTTPException(503, "backend not initialised")
    return b


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


app = FastAPI(title="simplelm", lifespan=_lifespan)


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    b = _get_backend()
    return ModelList(data=[ModelInfo(
        id=b.model_name,
        created=int(time.time()),
        max_model_len=getattr(b, "max_model_len", None),
    )])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    backend = _get_backend()

    # Convert pydantic messages → plain dicts the backend expects.
    plain_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    tools_arg = (
        [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
    )

    # Pass params straight through. None ⇒ the backend uses the model's
    # generation_config default for that field. `stop` is normalised to a list.
    stop = req.stop
    if isinstance(stop, str):
        stop = [stop]
    try:
        result = backend.generate(
            plain_msgs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=stop,
            tools=tools_arg,
            chat_template_kwargs=req.chat_template_kwargs,
        )
    except ContextOverflowError as e:
        # OpenAI-shaped context error (status 400, code context_length_exceeded)
        # so OpenAI-compatible clients (litellm/Alan-Code) recognise it as a
        # context overflow and compact/retry, instead of the engine crashing.
        return JSONResponse(status_code=400, content={"error": {
            "message": str(e),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }})
    _log_metrics(req.model, result)

    # Parse tool calls out of the text. Declared tool names disambiguate a
    # bare JSON object whose arguments are inlined (no ``arguments`` wrapper).
    parser = get_parser(_TOOL_PARSER_NAME)
    content, tool_calls = _extract_tool_calls(
        parser, result.text, tools_arg, req.tool_choice
    )

    message: dict = {"role": "assistant", "content": content}
    finish_reason = result.finish_reason
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        # Pseudo-streaming: the backend is synchronous, so we just emit
        # the full message as a single delta plus the [DONE] sentinel.
        # Real token-by-token streaming would require backend hooks; for
        # now this lets clients that hard-require stream=true work.
        return StreamingResponse(
            _stream_synthetic(completion_id, req.model, message, finish_reason),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _stream_synthetic(
    completion_id: str, model: str, message: dict, finish_reason: str
) -> AsyncGenerator[str, None]:
    """Emit the OpenAI streaming chunk format from a finished completion.

    Two chunks: one carrying the full delta, one with `finish_reason`.
    Clients that wait for `data: [DONE]` get it.
    """
    created = int(time.time())
    delta = {"role": "assistant", "content": message.get("content")}
    if "tool_calls" in message:
        delta["tool_calls"] = message["tool_calls"]
    chunk1 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    chunk2 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json_module.dumps(chunk1)}\n\n"
    yield f"data: {json_module.dumps(chunk2)}\n\n"
    yield "data: [DONE]\n\n"


def serve(
    backend: Backend,
    *,
    host: str = "0.0.0.0",
    port: int = 9876,
    tool_parser: str = "noop",
    log_level: str = "info",
) -> None:
    """Convenience: set the backend + tool parser, start uvicorn.

    Intended as the typical entry point from a Python script. The
    `simplelm.cli.main` CLI uses this too.
    """
    global _TOOL_PARSER_NAME
    _TOOL_PARSER_NAME = tool_parser
    set_backend(backend)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
