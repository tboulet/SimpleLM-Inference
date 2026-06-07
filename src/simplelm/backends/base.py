"""Abstract Backend interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length" | "tool_calls" | "error"


class Backend(Protocol):
    """A backend turns a structured chat request into a single completion.

    Implementations should be safe to call concurrently (FastAPI may run
    multiple requests in parallel under uvicorn). If the underlying
    library is not thread-safe, wrap with an internal lock or asyncio
    semaphore.
    """

    model_name: str

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> GenerationResult:
        ...
