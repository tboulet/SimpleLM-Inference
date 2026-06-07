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
    max_tokens: int | None = 512
    temperature: float | None = 0.7
    top_p: float | None = 1.0
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


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
