"""Parser sanity tests — runs without torch."""
from __future__ import annotations

import json

import pytest

from simplelm.tools import get_parser, list_parsers


def test_registry_listing():
    names = list_parsers()
    assert "noop" in names
    assert "gemma4" in names
    assert "qwen3_coder" in names
    assert "kimi_k2" in names
    assert "glm45" in names
    assert "minimax-m2" in names
    assert "deepseek-v4" in names


def test_noop_passthrough():
    parse = get_parser("noop")
    text = "hello world"
    content, calls = parse(text)
    assert content == text
    assert calls == []


def test_unknown_parser_raises():
    with pytest.raises(KeyError):
        get_parser("does_not_exist")


def test_gemma4_basic():
    parse = get_parser("gemma4")
    raw = 'Here we go.\n<|tool_call|>\n```json\n{"name": "search", "parameters": {"q": "paris"}}\n```\n<|end_tool_call|>\nDone.'
    content, calls = parse(raw)
    assert "Here we go." in content
    assert "Done." in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "paris"}


def test_qwen3_coder_basic():
    parse = get_parser("qwen3_coder")
    raw = 'thinking… <tool_call>{"name": "exec", "arguments": {"cmd": "ls"}}</tool_call>'
    content, calls = parse(raw)
    assert "thinking" in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec"


def test_kimi_k2_basic():
    parse = get_parser("kimi_k2")
    raw = (
        "Reasoning…\n"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>"
        '{"city": "Paris"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
        " trailing."
    )
    content, calls = parse(raw)
    assert "Reasoning" in content
    assert "trailing" in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_glm45_basic():
    parse = get_parser("glm45")
    raw = 'context.\n[TOOL_CALLS] [{"name": "navigate", "arguments": {"dir": "north"}}]\nafter.'
    content, calls = parse(raw)
    assert "context" in content
    assert "after" in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "navigate"


def test_minimax_m2_basic():
    parse = get_parser("minimax-m2")
    raw = 'before <minimax:tool_call>{"name": "ping", "arguments": {}}</minimax:tool_call> after'
    content, calls = parse(raw)
    assert "before" in content
    assert "after" in content
    assert calls[0]["function"]["name"] == "ping"


def test_no_match_returns_raw():
    """A parser that finds nothing should return `(raw, [])` rather than raise."""
    parse = get_parser("gemma4")
    raw = "plain text with no tool call"
    content, calls = parse(raw)
    assert content == raw
    assert calls == []


def test_simple_call_basic():
    parse = get_parser("simple_call")
    raw = "I'll check the weather. call:get_weather{city:Paris}"
    content, calls = parse(raw)
    assert "I'll check the weather" in content
    assert "call:" not in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"city": "Paris"}


def test_simple_call_multiple():
    """Multiple sequential call:Name{...} blocks all extract."""
    parse = get_parser("simple_call")
    raw = "First call:foo{a:1} then call:bar{b:2,c:hello world}"
    content, calls = parse(raw)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "foo"
    assert calls[1]["function"]["name"] == "bar"
    args = json.loads(calls[1]["function"]["arguments"])
    assert args == {"b": "2", "c": "hello world"}


def test_simple_call_quoted_value_with_comma():
    """Quoted values may contain commas."""
    parse = get_parser("simple_call")
    raw = 'call:say{text:"hello, world"}'
    content, calls = parse(raw)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"text": "hello, world"}


def test_simple_call_gameagents_alan_style():
    """The format GameAgents/Alan-Code seems to nudge models toward."""
    parse = get_parser("simple_call")
    raw = "call:Bash{command:ls -R,purpose:Explore the working directory structure.}"
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "Bash"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["command"] == "ls -R"
    assert "Explore" in args["purpose"]
