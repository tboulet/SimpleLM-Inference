"""Pluggable backends.

A backend translates a structured `ChatCompletionRequest` into one or
more decoded tokens. The server is backend-agnostic; pick the one that
matches your environment.

    HuggingFaceBackend  — torch + transformers .generate()
    VLLMBackend         — TBD
    SGLangBackend       — TBD
"""
from __future__ import annotations

from simplelm.backends.base import Backend
from simplelm.backends.hf import HuggingFaceBackend

__all__ = ["Backend", "HuggingFaceBackend"]
