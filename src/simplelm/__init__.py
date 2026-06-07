"""SimpleLM — OpenAI-compatible LLM inference on top of HuggingFace transformers.

Public API surface (re-exported here):

    from simplelm import HuggingFaceBackend, serve, app

Backends, tool parsers, and vision helpers live in their own submodules.
"""
from __future__ import annotations

from simplelm.backends.hf import HuggingFaceBackend
from simplelm.server import app, serve

__all__ = ["HuggingFaceBackend", "app", "serve"]
__version__ = "0.0.1"
