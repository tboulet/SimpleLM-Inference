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
    max_model_len: int | None  # context window, or None if unknown

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> GenerationResult:
        """Generate one completion.

        Sampling params that are ``None`` mean "use the backend/model's own
        default" — the HF backend uses the model's ``generation_config`` value
        for that field (its tuned ``top_p`` / ``repetition_penalty`` /
        ``top_k``). ``stop`` is the OpenAI stop-sequence list.
        """
        ...
