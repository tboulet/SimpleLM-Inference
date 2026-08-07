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
    prefill_ms: float | None = None  # wall time to the first generated token
    decode_ms: float | None = None   # wall time for the remaining tokens


class ContextOverflowError(Exception):
    """prompt + max_new_tokens would exceed the VRAM-safe context.

    Raised before generate() so the server returns a clean 4xx instead of the
    backend crashing on an OOM / position-index error mid-generation.
    """

    def __init__(self, prompt_tokens: int, max_new_tokens: int, safe_context: int):
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.safe_context = safe_context
        # Phrased like OpenAI's context error ("maximum context length" +
        # "context_length_exceeded") so OpenAI-compatible clients (litellm,
        # Alan-Code) classify it as a context overflow and compact/retry, rather
        # than treat it as an unknown hard failure. Alan-Code matches on this
        # text (alancode/api/errors.py _PROMPT_TOO_LONG_PATTERNS).
        super().__init__(
            f"This model's maximum context length is {safe_context} tokens "
            f"(VRAM-safe limit). However, your request has "
            f"{prompt_tokens + max_new_tokens} tokens ({prompt_tokens} in the "
            f"messages, {max_new_tokens} for the completion). Reduce the input "
            f"length. (code: context_length_exceeded)"
        )


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
