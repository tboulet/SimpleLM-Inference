"""HuggingFace transformers backend — text + vision via PIL images.

Single-thread `.generate()`. No paged KV cache, no flash attention.
Functional baseline that runs anywhere torch can see a GPU.
"""
from __future__ import annotations

import threading

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from simplelm.backends.base import Backend, GenerationResult
from simplelm.vision import load_image


def _load_model_auto(model_path: str, *, torch_dtype, device_map, trust_remote_code):
    """Try the right AutoModel for the given config.

    `AutoModelForCausalLM` rejects vision-language models (Qwen2.5-VL,
    Gemma3/4 multimodal, etc.). We probe several heads in order.
    """
    from transformers import AutoModelForCausalLM

    # Pre-load the config so we can ask "is this multimodal?".
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    arch = (getattr(cfg, "architectures", None) or [""])[0] or ""
    is_vlm = (
        getattr(cfg, "vision_config", None) is not None
        or "VL" in arch
        or "Vision" in arch
        or "Multi" in arch
        or "ConditionalGeneration" in arch
    )

    # Try the VLM head first if config looks multimodal, else CausalLM
    # first. Each branch falls through to others on failure.
    candidates = []
    if is_vlm:
        # AutoModelForImageTextToText is the newest umbrella head.
        try:
            from transformers import AutoModelForImageTextToText
            candidates.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        try:
            from transformers import AutoModelForVision2Seq
            candidates.append(AutoModelForVision2Seq)
        except ImportError:
            pass
    candidates.append(AutoModelForCausalLM)

    last_err: Exception | None = None
    for cls in candidates:
        try:
            return cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            # ValueError is what AutoModel raises when the config doesn't
            # match. Keep trying other heads.
            last_err = e
            continue
    raise RuntimeError(
        f"None of the AutoModel heads accept config arch={arch!r}. Last error: {last_err}"
    )


def _coerce_for_template(messages: list[dict]) -> tuple[list[dict], list]:
    """Flatten OpenAI multimodal content arrays into the format the HF
    chat template + processor expects.

    Returns `(template_messages, images_in_order)`. The template messages
    embed `{"type": "image"}` placeholders (matching the HF processor
    convention) while the actual PIL Images are returned separately
    for the processor call.
    """
    out_msgs: list[dict] = []
    imgs: list = []
    for m in messages:
        content = m.get("content")
        role = m.get("role", "user")
        if content is None:
            # tool / assistant tool_call messages — leave as-is
            out_msgs.append(m)
            continue
        if isinstance(content, str):
            out_msgs.append({"role": role, "content": content})
            continue
        # content is a list of {type: text|image_url, …}
        parts: list[dict] = []
        for c in content:
            t = c.get("type")
            if t == "text":
                parts.append({"type": "text", "text": c.get("text", "")})
            elif t == "image_url":
                url = c.get("image_url", {}).get("url", "")
                if not url:
                    continue
                img = load_image(url)
                imgs.append(img)
                parts.append({"type": "image"})
            else:
                # silently drop unknown types — future extensibility
                continue
        out_msgs.append({"role": role, "content": parts})
    return out_msgs, imgs


class HuggingFaceBackend:
    """Default backend — wraps `transformers.AutoModelForCausalLM`.

    Args:
        model_path: HF snapshot path or hub id.
        model_name: served-model id (defaults to basename of `model_path`).
        torch_dtype: "bfloat16" (default) / "float16" / "float32".
        device_map: forwarded to from_pretrained. Default `"auto"`.
        trust_remote_code: forwarded to from_pretrained. Default True
            (needed for many recent models — gemma3, qwen3_next, etc).
        prefer_processor: if True, try `AutoProcessor` first; on
            failure fall back to `AutoTokenizer`. Required for VLMs.
    """

    def __init__(
        self,
        model_path: str,
        *,
        model_name: str | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        prefer_processor: bool = True,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name or model_path.rstrip("/").split("/")[-1]
        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(torch_dtype, torch.bfloat16)

        self._processor = None
        self._tokenizer = None
        if prefer_processor:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
                self._tokenizer = self._processor.tokenizer
            except Exception:
                self._processor = None
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )

        self._model = _load_model_auto(
            model_path,
            torch_dtype=self._dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()
        # `.generate()` is not thread-safe — serialise calls.
        self._lock = threading.Lock()

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> GenerationResult:
        template_msgs, images = _coerce_for_template(messages)
        chat_template_kwargs = chat_template_kwargs or {}
        if tools:
            chat_template_kwargs.setdefault("tools", tools)

        # Render to text once via the tokenizer template, then run the
        # vision processor (if any) to inject image features. Two-step
        # is more reliable than `apply_chat_template(return_tensors=…)`
        # which behaves differently across tokenizer families.
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                template_msgs,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except Exception:
            prompt_text = "\n".join(
                f"{m['role']}: {m.get('content', '')}" for m in template_msgs
            ) + "\nassistant: "

        if images and self._processor is not None:
            inputs = self._processor(
                images=images, text=prompt_text, return_tensors="pt"
            ).to(self._model.device)
        else:
            inputs = self._tokenizer(
                prompt_text, return_tensors="pt"
            ).to(self._model.device)

        prompt_len = inputs["input_ids"].shape[1]
        do_sample = (temperature or 0.0) > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        with self._lock, torch.no_grad():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        completion_ids = out_ids[0][prompt_len:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_len = int(completion_ids.shape[0])
        finish = "length" if completion_len >= max_new_tokens else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=int(prompt_len),
            completion_tokens=completion_len,
            finish_reason=finish,
        )
