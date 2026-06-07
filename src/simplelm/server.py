"""FastAPI OpenAI-compatible server.

Wraps a single `Backend` instance and exposes `/v1/models` +
`/v1/chat/completions`. Tool-call parsing is dispatched via
`simplelm.tools.get_parser`.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from simplelm.backends.base import Backend
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
    return ModelList(data=[ModelInfo(id=b.model_name, created=int(time.time()))])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    backend = _get_backend()

    # Convert pydantic messages → plain dicts the backend expects.
    plain_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    tools_arg = (
        [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
    )

    if req.stream:
        # Streaming not yet supported. Fail fast with a 400 so clients can
        # fall back to non-streaming rather than silently hanging.
        raise HTTPException(
            400, "streaming is not implemented in this SimpleLM version"
        )

    result = backend.generate(
        plain_msgs,
        max_new_tokens=int(req.max_tokens or 512),
        temperature=float(req.temperature or 0.7),
        top_p=float(req.top_p or 1.0),
        tools=tools_arg,
        chat_template_kwargs=req.chat_template_kwargs,
    )

    # Parse tool calls out of the text.
    parser = get_parser(_TOOL_PARSER_NAME)
    content, tool_calls = parser(result.text)

    message: dict = {"role": "assistant", "content": content}
    finish_reason = result.finish_reason
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


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
