"""Per-model-family tool-call parsers.

Each parser takes raw assistant text and returns
`(content_without_tool_calls, list_of_openai_tool_call_dicts)`.

The text-without-tool-calls is what gets sent to the client as
`message.content`. Tool calls are normalised to OpenAI's shape:

    {"id": "call_<uuid>", "type": "function",
     "function": {"name": "<fn>", "arguments": "<json-string>"}}

If a model didn't emit a tool call, parsers return `(raw, [])` rather
than raising.

These are deliberately permissive — we accept a tool call that is
*structurally* recognisable, even if the args aren't valid JSON, because
small models often emit close-enough JSON. The downstream client gets
the raw string and can decide how to recover.
"""
from __future__ import annotations

import json
import re
import uuid


def _mk_call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments)
        except Exception:
            arguments = str(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_noop(raw: str) -> tuple[str, list[dict]]:
    """Never extract a tool call. Default for text-only."""
    return raw, []


def parse_gemma4(raw: str) -> tuple[str, list[dict]]:
    """Gemma 4 emits ``<|tool_call|>\\n```json\\n{name, parameters}\\n``` <|end_tool_call|>``.

    Per SGLang gemma4 cookbook + the gemma-4 chat template.
    """
    pat = re.compile(
        r"<\|tool_call\|>\s*```(?:json)?\s*(\{.*?\})\s*```\s*(?:<\|end_tool_call\|>)?",
        re.DOTALL,
    )
    calls: list[dict] = []
    leftover_parts: list[str] = []
    last = 0
    for m in pat.finditer(raw):
        leftover_parts.append(raw[last:m.start()])
        last = m.end()
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("tool")
            args = obj.get("parameters") or obj.get("arguments") or {}
            if name:
                calls.append(_mk_call(name, args))
        except json.JSONDecodeError:
            # keep as-is in content rather than silently dropping
            leftover_parts.append(m.group(0))
    leftover_parts.append(raw[last:])
    return "".join(leftover_parts).strip(), calls


def parse_qwen3_coder(raw: str) -> tuple[str, list[dict]]:
    """Qwen3-Coder uses XML-ish ``<tool_call>`` blocks with JSON inside."""
    pat = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    calls: list[dict] = []
    leftover_parts: list[str] = []
    last = 0
    for m in pat.finditer(raw):
        leftover_parts.append(raw[last:m.start()])
        last = m.end()
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                calls.append(_mk_call(name, args))
        except json.JSONDecodeError:
            leftover_parts.append(m.group(0))
    leftover_parts.append(raw[last:])
    return "".join(leftover_parts).strip(), calls


def parse_kimi_k2(raw: str) -> tuple[str, list[dict]]:
    """Kimi K2 uses ``<|tool_calls_section_begin|> … <|tool_calls_section_end|>``.

    Inside, each call is ``<|tool_call_begin|><id><|tool_call_argument_begin|><args><|tool_call_end|>``.
    """
    section = re.search(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        raw, re.DOTALL,
    )
    if not section:
        return raw, []
    body = section.group(1)
    leftover = raw[:section.start()] + raw[section.end():]
    calls: list[dict] = []
    call_pat = re.compile(
        r"<\|tool_call_begin\|>(?P<id>.*?)<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
        re.DOTALL,
    )
    for m in call_pat.finditer(body):
        ident = m.group("id").strip()
        args = m.group("args").strip()
        # Kimi puts "<name>:<n>" as the id, name is before the colon.
        name = ident.split(":", 1)[0] or "tool"
        calls.append(_mk_call(name, args))
    return leftover.strip(), calls


def parse_glm45(raw: str) -> tuple[str, list[dict]]:
    """GLM-4.5+ uses ``[TOOL_CALLS] [ {name, arguments}, … ]`` JSON block."""
    pat = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
    m = pat.search(raw)
    if not m:
        return raw, []
    try:
        arr = json.loads(m.group(1))
    except json.JSONDecodeError:
        return raw, []
    calls: list[dict] = []
    for obj in arr:
        if isinstance(obj, dict):
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                calls.append(_mk_call(name, args))
    leftover = (raw[:m.start()] + raw[m.end():]).strip()
    return leftover, calls


def parse_minimax_m2(raw: str) -> tuple[str, list[dict]]:
    """MiniMax M2 uses ``<minimax:tool_call> {name, arguments} </minimax:tool_call>``."""
    pat = re.compile(
        r"<minimax:tool_call>\s*(\{.*?\})\s*</minimax:tool_call>", re.DOTALL,
    )
    calls: list[dict] = []
    leftover_parts: list[str] = []
    last = 0
    for m in pat.finditer(raw):
        leftover_parts.append(raw[last:m.start()])
        last = m.end()
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                calls.append(_mk_call(name, args))
        except json.JSONDecodeError:
            leftover_parts.append(m.group(0))
    leftover_parts.append(raw[last:])
    return "".join(leftover_parts).strip(), calls


def parse_simple_call(raw: str) -> tuple[str, list[dict]]:
    """Permissive parser for ``call:FunctionName{key:value,key:value}`` style.

    Many small / mid models emit Claude-imitating text like
    ``call:get_weather{city:Paris}`` when given an OpenAI tools schema
    without strong format steering. Values may be bare, quoted, or
    contain commas inside strings. We do a best-effort parse and
    fall back to keeping the raw block in `content` if structure is
    unrecognisable.

    Works for: gemma-4 (without `<|tool_call|>`), llama-3, mistral
    family, smaller Qwen variants.
    """
    pat = re.compile(r"call\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{([^{}]*)\}")
    calls: list[dict] = []
    leftover_parts: list[str] = []
    last = 0
    for m in pat.finditer(raw):
        leftover_parts.append(raw[last:m.start()])
        last = m.end()
        name = m.group(1)
        body = m.group(2)
        args = _parse_simple_args(body)
        calls.append(_mk_call(name, args))
    leftover_parts.append(raw[last:])
    return "".join(leftover_parts).strip(), calls


def _parse_simple_args(body: str) -> dict:
    """Turn ``key1:value1,key2:value2`` into a dict, preserving commas
    inside quoted strings. Bare values are kept as strings."""
    args: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        # skip whitespace + commas between entries
        while i < n and body[i] in " ,\t\n":
            i += 1
        # key
        key_start = i
        while i < n and body[i] not in ":":
            i += 1
        if i >= n:
            break
        key = body[key_start:i].strip()
        i += 1  # skip ':'
        # value — may be quoted
        while i < n and body[i] in " \t":
            i += 1
        if i >= n:
            args[key] = ""
            break
        if body[i] in ('"', "'"):
            quote = body[i]
            i += 1
            val_start = i
            while i < n and body[i] != quote:
                # honour simple backslash escapes
                if body[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            value = body[val_start:i]
            if i < n:
                i += 1  # skip closing quote
        else:
            val_start = i
            while i < n and body[i] != ",":
                i += 1
            value = body[val_start:i].strip()
        args[key] = value
    return args


def parse_python_call(raw: str) -> tuple[str, list[dict]]:
    """Permissive parser for ``function_name(arg=value, arg="value")`` syntax.

    Observed on Qwen2.5-VL-7B-Instruct (and many other models) when
    given an OpenAI tools schema: the model emits Python function-call
    text. Values may be bare, double-quoted, or single-quoted.

    Distinguishing from natural prose: we only match when the call
    appears alone on a line or is the sole assistant content (i.e. the
    regex enforces it sits at start-of-line or after specific
    whitespace + the parenthesised args have at least one key=value
    pair).
    """
    pat = re.compile(
        r"(?:^|\n)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*=[^()]*)\)\s*(?=$|\n)",
        re.MULTILINE,
    )
    calls: list[dict] = []
    leftover_parts: list[str] = []
    last = 0
    for m in pat.finditer(raw):
        leftover_parts.append(raw[last:m.start()])
        last = m.end()
        name = m.group(1)
        body = m.group(2)
        args = _parse_python_args(body)
        calls.append(_mk_call(name, args))
    leftover_parts.append(raw[last:])
    return "".join(leftover_parts).strip(), calls


def _parse_python_args(body: str) -> dict:
    """Parse ``key=value, key="value with space"`` into a dict."""
    args: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " ,\t\n":
            i += 1
        key_start = i
        while i < n and body[i] != "=":
            i += 1
        if i >= n:
            break
        key = body[key_start:i].strip()
        i += 1  # skip '='
        while i < n and body[i] in " \t":
            i += 1
        if i >= n:
            args[key] = ""
            break
        if body[i] in ('"', "'"):
            quote = body[i]
            i += 1
            val_start = i
            while i < n and body[i] != quote:
                if body[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            value = body[val_start:i]
            if i < n:
                i += 1
        else:
            val_start = i
            while i < n and body[i] != ",":
                i += 1
            value = body[val_start:i].strip()
        args[key] = value
    return args


def parse_deepseek_v4(raw: str) -> tuple[str, list[dict]]:
    """DeepSeek V4 JSON block: ``<｜tool_calls_begin｜>[{...}]<｜tool_calls_end｜>``.

    The pipe characters are the full-width ｜ used in their chat template.
    """
    pat = re.compile(
        r"[<｜<\|]\s*tool_calls_begin\s*[｜\|>]\s*(\[.*?\])\s*[<｜<\|]\s*tool_calls_end\s*[｜\|>]",
        re.DOTALL,
    )
    m = pat.search(raw)
    if not m:
        return raw, []
    try:
        arr = json.loads(m.group(1))
    except json.JSONDecodeError:
        return raw, []
    calls: list[dict] = []
    for obj in arr:
        if isinstance(obj, dict):
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                calls.append(_mk_call(name, args))
    leftover = (raw[:m.start()] + raw[m.end():]).strip()
    return leftover, calls
