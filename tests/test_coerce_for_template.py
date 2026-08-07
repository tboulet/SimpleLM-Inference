from __future__ import annotations

import simplelm.backends.hf as hf
from simplelm.backends.hf import _coerce_for_template


def test_text_only_content_array_collapses_to_string():
    # OpenAI structured (list) content with only text parts must become a plain
    # string; a text model's chat template concatenates content as a string and
    # raises on a list of parts.
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": " world"},
    ]}]
    out, imgs = _coerce_for_template(msgs)
    assert out == [{"role": "user", "content": "hello world"}]
    assert imgs == []


def test_plain_string_and_none_pass_through():
    out, imgs = _coerce_for_template([
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ])
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1]["content"] is None and out[1]["tool_calls"] == [{"id": "1"}]
    assert imgs == []


def test_content_with_image_keeps_parts_list(monkeypatch):
    # A genuinely multimodal message must retain the list-of-parts form (with an
    # {"type": "image"} placeholder) so the vision processor can inject features.
    sentinel = object()
    monkeypatch.setattr(hf, "load_image", lambda url: sentinel)
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}},
    ]}]
    out, imgs = _coerce_for_template(msgs)
    assert out[0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image"},
    ]
    assert imgs == [sentinel]
