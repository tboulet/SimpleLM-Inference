"""Server tests using a mock backend — no torch / no model needed."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from simplelm import app
from simplelm.backends.base import GenerationResult
from simplelm.server import set_backend, _TOOL_PARSER_NAME


@dataclass
class MockBackend:
    """Minimal Backend implementation that returns a fixed string."""

    model_name: str = "mock-model"
    canned_text: str = "The capital of France is Paris."

    def generate(
        self,
        messages,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop=None,
        tools=None,
        chat_template_kwargs=None,
    ) -> GenerationResult:
        # Echo a different canned text per call so tests can assert per-request shape.
        n_prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
        return GenerationResult(
            text=self.canned_text,
            prompt_tokens=max(1, n_prompt_tokens),
            completion_tokens=len(self.canned_text.split()),
            finish_reason="stop",
        )


@pytest.fixture
def client():
    set_backend(MockBackend())
    return TestClient(app)


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "mock-model"


def test_chat_completion_basic(client):
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 60,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Paris" in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


def test_chat_completion_streaming_emits_sse(client):
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    body = r.text
    # Two chunks + [DONE] sentinel
    assert "Paris" in body
    assert "[DONE]" in body
    assert '"finish_reason": "stop"' in body or '"finish_reason":"stop"' in body


def test_tool_call_parsed_via_gemma4():
    # Switch the global parser before constructing the test client.
    from simplelm import server as srv
    srv._TOOL_PARSER_NAME = "gemma4"
    canned = (
        'preface text\n<|tool_call|>\n```json\n'
        '{"name": "lookup", "parameters": {"q": "weather paris"}}\n'
        '```\n<|end_tool_call|>\n'
    )
    set_backend(MockBackend(canned_text=canned))
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "weather paris"}],
    })
    assert r.status_code == 200
    data = r.json()
    msg = data["choices"][0]["message"]
    assert "lookup" not in msg["content"]
    assert msg["tool_calls"]
    assert msg["tool_calls"][0]["function"]["name"] == "lookup"
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    # restore default parser so other tests aren't affected
    srv._TOOL_PARSER_NAME = "noop"


def _universal_client(canned_text: str) -> TestClient:
    from simplelm import server as srv
    srv._TOOL_PARSER_NAME = "universal"
    set_backend(MockBackend(canned_text=canned_text))
    return TestClient(app)


def _restore_parser() -> None:
    from simplelm import server as srv
    srv._TOOL_PARSER_NAME = "noop"


def test_json_data_not_parsed_as_tool_call():
    """No tools declared + a JSON data answer -> content preserved, no phantom call."""
    client = _universal_client('{"name": "Alice", "age": 30}')
    try:
        r = client.post("/v1/chat/completions", json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Return JSON with name and age."}],
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg.get("tool_calls") in (None, [])
        assert '"Alice"' in msg["content"]
    finally:
        _restore_parser()


def test_real_tool_call_with_arguments_extracted():
    client = _universal_client('{"name": "get_weather", "arguments": {"city": "Paris"}}')
    try:
        r = client.post("/v1/chat/completions", json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "weather?"}],
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        _restore_parser()


def test_inline_tool_call_resolved_via_declared_tools():
    """Inline-args call is recognised because the request declared the tool name."""
    client = _universal_client('{"name": "get_weather", "city": "Paris"}')
    try:
        r = client.post("/v1/chat/completions", json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        import json as _json
        assert _json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    finally:
        _restore_parser()


def test_tool_choice_none_disables_extraction():
    client = _universal_client('{"name": "get_weather", "arguments": {"city": "Paris"}}')
    try:
        r = client.post("/v1/chat/completions", json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tool_choice": "none",
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg.get("tool_calls") in (None, [])
        assert "get_weather" in msg["content"]
    finally:
        _restore_parser()


def test_multimodal_content_array_accepted(client):
    """Server should accept (and forward to backend) an OpenAI multimodal content array."""
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAABlBMVEX///8AAABVwtN+AAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAeIhvDMAAAAASUVORK5CYII="
                }},
            ],
        }],
    })
    assert r.status_code == 200
    # MockBackend ignores image content — we just verify the server didn't choke on the schema.
    assert "Paris" in r.json()["choices"][0]["message"]["content"]


def test_context_overflow_returns_openai_context_error():
    """An over-budget request surfaces as an OpenAI-shaped context error (400,
    code context_length_exceeded) so litellm/Alan-Code compact instead of crash."""
    from simplelm.backends.base import ContextOverflowError

    @dataclass
    class OverflowBackend:
        model_name: str = "mock-model"

        def generate(self, messages, **kwargs):
            raise ContextOverflowError(prompt_tokens=9000, max_new_tokens=512, safe_context=8192)

    set_backend(OverflowBackend())
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    # The message must contain phrases Alan-Code matches as prompt-too-long
    # (alancode/api/errors.py) so the agent compacts instead of crashing.
    msg = err["message"].lower()
    assert "maximum context length" in msg
    assert "context_length_exceeded" in msg


def test_metrics_logged_to_jsonl(tmp_path, monkeypatch):
    """Each completion appends one JSONL metrics line with the expected keys."""
    import json
    from simplelm import server as srv

    metrics_file = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(srv, "_METRICS_FILE", str(metrics_file))
    set_backend(MockBackend())
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "mock-model"
    assert set(("ts", "prompt_tokens", "completion_tokens", "prefill_ms", "decode_ms")) <= rec.keys()
