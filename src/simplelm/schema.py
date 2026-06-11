"""OpenAI-compatible request/response schemas (minimal subset)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ----- Chat content -----


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    url: str  # http(s):// OR data:image/...;base64,... OR file:///...
    detail: Literal["auto", "low", "high"] | None = "auto"


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ChatContent = TextContent | ImageContent


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatContent] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


# ----- Tools -----


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


# ----- Request / response -----


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # None ⇒ use the model's own generation_config default for this field.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    tools: list[ToolDef] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class ChatChoice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# ----- Models endpoint -----


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "simplelm"
    created: int
    # Context window. AlanCode (and vLLM/SGLang clients) read this from
    # /v1/models to size conversation compaction. Omitted (None) if the
    # model config doesn't expose it.
    max_model_len: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
