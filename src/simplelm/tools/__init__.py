"""Tool-call parsing — per-model-family registry.

The parser receives the raw assistant text and returns
`(content_without_tool_calls, list_of_tool_calls_dict)` where each
tool_call dict matches OpenAI's `tool_calls` array entry:

    {"id": "...", "type": "function",
     "function": {"name": "...", "arguments": "<JSON-string>"}}

If the model didn't emit a tool call, parsers return `(raw_text, [])`.

Conventions are aligned with the SGLang cookbook so a model that works
on JZ via SGLang with `--tool-call-parser X` works here via
`simplelm.tools.get_parser("X")`.

Supported (initial cut):
    - "noop"   : never extracts a tool call. Default for text-only.
    - "gemma4" : Gemma 4's `<|tool_call|>` block.
    - "qwen3_coder" : Qwen3-Coder XML-ish format.
    - "kimi_k2" : Kimi K2 `<|tool_call_begin|>` markers.
    - "glm45"  : GLM-4.5/4.7/5.1 `[TOOL_CALLS]` block.
    - "minimax-m2" : MiniMax M2 `<minimax:tool_call>`.
    - "deepseek-v4" : DeepSeek V4 JSON block.
"""
from __future__ import annotations

from typing import Callable

from .parsers import (
    parse_deepseek_v4,
    parse_gemma4,
    parse_glm45,
    parse_kimi_k2,
    parse_minimax_m2,
    parse_noop,
    parse_qwen3_coder,
    parse_simple_call,
)


_REGISTRY: dict[str, Callable] = {
    "noop": parse_noop,
    "gemma4": parse_gemma4,
    "qwen3_coder": parse_qwen3_coder,
    "kimi_k2": parse_kimi_k2,
    "glm45": parse_glm45,
    "minimax-m2": parse_minimax_m2,
    "deepseek-v4": parse_deepseek_v4,
    "simple_call": parse_simple_call,
}


def get_parser(name: str | None) -> Callable[[str], tuple[str, list[dict]]]:
    if not name:
        return parse_noop
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown tool parser {name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_parsers() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = ["get_parser", "list_parsers"]
