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
