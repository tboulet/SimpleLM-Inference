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


def test_python_call_basic():
    parse = get_parser("python_call")
    raw = 'get_weather(location="Paris")'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Paris"}


def test_python_call_multiple_args():
    parse = get_parser("python_call")
    raw = 'send_email(to="alice@example.com", subject="hi", body="hello there, friend")'
    content, calls = parse(raw)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["to"] == "alice@example.com"
    assert args["subject"] == "hi"
    assert "hello there" in args["body"]


def test_python_call_positional():
    """Positional args (no `=`) are stored under 'args' key."""
    parse = get_parser("python_call")
    raw = 'get_weather(Paris)'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"args": ["Paris"]}


def test_python_call_zero_args():
    parse = get_parser("python_call")
    raw = "do_thing()"
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "do_thing"
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_python_call_skip_inline_prose():
    """A function-call-like substring in prose shouldn't be extracted."""
    parse = get_parser("python_call")
    raw = "I would say that get_weather(location='Paris') is a good idea."
    content, calls = parse(raw)
    # No newline boundaries → don't extract
    assert calls == []
    assert content == raw


def test_name_then_json_basic():
    parse = get_parser("name_then_json")
    raw = 'get_weather{"location": "Paris"}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Paris"}


def test_json_object_basic():
    parse = get_parser("json_object")
    raw = 'Sure, I will call {"name": "lookup", "arguments": {"q": "weather"}}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup"


def test_json_object_plain_data_preserved():
    """A data object that merely has a `name` field must stay in content."""
    parse = get_parser("json_object")
    raw = '{"name": "Alice", "age": 30}'
    content, calls = parse(raw)
    assert calls == []
    assert content == raw


def test_json_object_real_call_with_arguments():
    parse = get_parser("json_object")
    raw = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_json_object_arguments_as_json_string():
    """`arguments` given as a JSON string (not an object) is preserved verbatim."""
    parse = get_parser("json_object")
    raw = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_json_object_inline_args_with_declared_tool():
    """Inline args (no `arguments` wrapper) resolve only against a declared tool."""
    parse = get_parser("json_object")
    raw = '{"name": "get_weather", "city": "Paris"}'
    content, calls = parse(raw, tool_names={"get_weather"})
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_json_object_inline_args_without_declared_tool_preserved():
    """The same object is indistinguishable from data when no tool is declared."""
    parse = get_parser("json_object")
    raw = '{"name": "get_weather", "city": "Paris"}'
    content, calls = parse(raw)
    assert calls == []
    assert content == raw


def test_json_object_fenced_call_detected():
    parse = get_parser("json_object")
    raw = '```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"


def test_json_object_multiple_calls():
    parse = get_parser("json_object")
    raw = '{"name": "a", "arguments": {"x": 1}} and {"name": "b", "arguments": {"y": 2}}'
    content, calls = parse(raw)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_json_object_data_object_amid_a_real_call():
    """A plain data object stays in content while a sibling real call extracts."""
    parse = get_parser("json_object")
    raw = '{"name": "Alice", "age": 30} then {"name": "get_weather", "arguments": {"city": "Paris"}}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert '"Alice"' in content


def test_universal_preserves_plain_json_data():
    """Regression guard for the canonical probe case: no tools, JSON data output."""
    parse = get_parser("universal")
    raw = '{"name": "Alice", "age": 30}'
    content, calls = parse(raw)
    assert calls == []
    assert content == raw


def test_universal_inline_call_with_declared_tool():
    parse = get_parser("universal")
    raw = '{"name": "get_weather", "city": "Paris"}'
    content, calls = parse(raw, tool_names={"get_weather"})
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"


def test_universal_falls_through():
    """universal tries each format until one matches."""
    parse = get_parser("universal")
    # name_then_json format
    raw = 'get_weather{"location": "Paris"}'
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"

    # python_call format
    raw = "set_volume(level=5)"
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "set_volume"

    # simple_call format
    raw = "call:fire{at:north}"
    content, calls = parse(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "fire"

    # no match
    content, calls = parse("just normal prose with no calls")
    assert calls == []


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
