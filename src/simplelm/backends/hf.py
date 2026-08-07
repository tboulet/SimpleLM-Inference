"""HuggingFace transformers backend — text + vision via PIL images.

Single-thread `.generate()`. No paged KV cache, no flash attention.
Functional baseline that runs anywhere torch can see a GPU.
"""
from __future__ import annotations

import copy
import os
import threading
import time

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from simplelm.backends.base import Backend, ContextOverflowError, GenerationResult
from simplelm.vision import load_image
from simplelm._logging import logger, warning_once


def _shim_torch_accelerator() -> None:
    """gptqmodel >= 7 calls `torch.accelerator.*` (added in torch 2.6), but the
    pinned ROCm build here is torch 2.5. Provide the minimal surface so GPTQ
    checkpoints load, mapping the accelerator to the (ROCm-masquerading) cuda
    device. No-op once torch ships the real module."""
    if hasattr(torch, "accelerator"):
        return
    import sys
    import types

    acc = types.ModuleType("torch.accelerator")
    acc.is_available = lambda: torch.cuda.is_available()
    acc.device_count = lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0
    acc.current_accelerator = lambda *a, **k: (
        torch.device("cuda") if torch.cuda.is_available() else None)
    torch.accelerator = acc
    sys.modules["torch.accelerator"] = acc


_shim_torch_accelerator()

# Context guard: what to do when prompt + max_new_tokens exceeds the VRAM-safe
# context. "reject" -> 4xx (safe default); "truncate" -> drop oldest prompt
# tokens; "off" -> no guard (legacy). Fraction of free VRAM to budget for KV.
_CTX_POLICY = os.environ.get("SIMPLELM_CONTEXT_POLICY", "reject")
_KV_SAFETY_FRACTION = float(os.environ.get("SIMPLELM_KV_SAFETY_FRACTION", "0.9"))
# On-the-fly weight quantization (bitsandbytes): "4bit" (nf4) / "8bit" halve or
# quarter the weight VRAM so a bigger model fits fewer GCDs. Needs bitsandbytes
# (ROCm build). "" = full precision.
_QUANT = os.environ.get("SIMPLELM_QUANTIZATION", "").lower()
# Fraction of each GPU a quantized model's weights may use when we build its
# device_map (see _quantized_device_map). The weights are spread across the GPUs
# with this as the per-device ceiling, so the rest stays free for the KV cache.
_QUANT_GPU_FRACTION = float(os.environ.get("SIMPLELM_QUANT_GPU_FRACTION", "0.9"))


class _FirstTokenTimer(StoppingCriteria):
    """Records the wall-clock of the first generated token (never stops
    generation), so prefill and decode time can be split for metrics."""

    def __init__(self):
        self.t_first = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.t_first is None:
            self.t_first = time.perf_counter()
        return False


def _build_quant_config():
    """BitsAndBytesConfig for SIMPLELM_QUANTIZATION, or None for full precision.

    4bit (nf4, double-quant, bf16 compute) quarters the weight VRAM; 8bit halves
    it, so a bf16 model that needs N GCDs fits fewer. Requires a ROCm
    bitsandbytes build.
    """
    if _QUANT not in ("4bit", "8bit"):
        return None
    from transformers import BitsAndBytesConfig
    if _QUANT == "4bit":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    logger.info("SIMPLELM_QUANTIZATION=%s -> bitsandbytes", _QUANT)
    return cfg


def _quantized_device_map(cfg, quant: str, fraction: float):
    """Explicit device_map for a bnb-quantized load, sized on the QUANTIZED
    element size (INT4/INT8) and spread across the visible GPUs.

    transformers' auto planner sizes device placement on the *unquantized* (bf16)
    dtype for our bnb build - Bnb4BitHfQuantizer exposes no target-dtype hook - so
    it plans a 462 GiB footprint and offloads layers to CPU/disk (which bnb 4-bit
    rejects) even though the 4-bit weights are ~4x smaller and fit easily. Building
    the map ourselves at the quantized size places it GPU-resident, spread evenly
    so each device keeps room for the KV cache. Returns "auto" (transformers'
    default) if a map cannot be built or would still spill off-GPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "auto"
    try:
        from accelerate import init_empty_weights, infer_auto_device_map
        from accelerate.utils import CustomDtype
        from transformers import AutoModelForCausalLM

        qdtype = CustomDtype.INT4 if quant == "4bit" else CustomDtype.INT8
        bytes_per = 0.5 if quant == "4bit" else 1.0
        with init_empty_weights():
            meta = AutoModelForCausalLM.from_config(cfg)
        nsm = list(getattr(meta, "_no_split_modules", []) or [])
        n = torch.cuda.device_count()
        q_bytes = sum(p.numel() for p in meta.parameters()) * bytes_per
        # Per-GPU ceiling: enough to spread the weights evenly (+30% slack for the
        # no-split granularity), but never above `fraction` of the device.
        even = int(q_bytes / n * 1.3)
        cap = {i: min(even, int(torch.cuda.get_device_properties(i).total_memory * fraction))
               for i in range(n)}
        dmap = infer_auto_device_map(meta, max_memory=cap, dtype=qdtype,
                                     no_split_module_classes=nsm)
        if any(str(d) in ("cpu", "disk") for d in dmap.values()):
            logger.warning("quantized device_map still spills off-GPU; using auto")
            return "auto"
        return dmap
    except Exception as e:  # noqa: BLE001 - any failure -> transformers' default
        logger.warning("could not build quantized device_map (%s); using auto", e)
        return "auto"


def _prequant_bits(cfg):
    """'4bit'/'8bit' if the checkpoint is a pre-serialized **bitsandbytes** model,
    else None. bnb keeps the weights int4 in VRAM, but its quantizer makes the HF
    auto planner size on bf16 and offload, so it needs our explicit int4
    device_map. GPTQ/AWQ size fine on their own; compressed-tensors decompresses to
    bf16 on ROCm (no int4 kernel), so an int4 device_map would under-plan it - both
    return None."""
    qc = getattr(cfg, "quantization_config", None)
    if not isinstance(qc, dict):
        return None
    method = (qc.get("quant_method") or "").lower()
    if method and method != "bitsandbytes":
        return None
    if qc.get("load_in_4bit"):
        return "4bit"
    if qc.get("load_in_8bit"):
        return "8bit"
    return None


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
    logger.info("detected architecture %r (is_vlm=%s)", arch, is_vlm)

    # PyTorch's memory-efficient SDP backend miscomputes *sliding-window*
    # attention on the ROCm/gfx90a stack (gemma-4 → <pad> past the window). For
    # such models disable it so SDPA uses the correct flash/MATH backend. Models
    # without a sliding window keep mem-efficient (the memory-frugal path — MATH
    # is O(seq²) and OOMs on long agent contexts).
    sliding = getattr(cfg, "sliding_window", None) or getattr(
        getattr(cfg, "text_config", None), "sliding_window", None
    )
    # Attention backend: prefer FlashAttention-2 when the flash_attn library is
    # importable. It computes sliding-window attention correctly AND memory-
    # efficiently - the only fast+correct path on ROCm/gfx90a, where SDPA's flash
    # backend rejects the sliding mask, mem-efficient miscomputes it, and MATH is
    # O(seq^2) (OOMs on long contexts). Without flash_attn, fall back to SDPA with
    # the broken mem-efficient backend disabled for sliding-window models.
    attn_impl = None
    try:
        import flash_attn as _flash_attn
        attn_impl = "flash_attention_2"
        logger.info("flash_attn %s present - attn_implementation=flash_attention_2",
                    _flash_attn.__version__)
    except ImportError:
        if sliding and torch.version.hip is not None:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info("sliding_window=%s on ROCm, no flash_attn - disabled mem-efficient SDP",
                        sliding)

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

    quant_cfg = _build_quant_config()
    # Size the device_map on the quantized element size (see _quantized_device_map
    # and _prequant_bits) - for bnb the HF auto planner sizes on bf16 and offloads
    # a model that fits in 4-bit.
    bits = _QUANT if quant_cfg is not None else _prequant_bits(cfg)
    if device_map == "auto" and bits in ("4bit", "8bit"):
        device_map = _quantized_device_map(cfg, bits, _QUANT_GPU_FRACTION)

    def _load(attn):
        kw = dict(torch_dtype=torch_dtype, device_map=device_map,
                  trust_remote_code=trust_remote_code)
        if attn:
            kw["attn_implementation"] = attn
        if quant_cfg is not None:
            kw["quantization_config"] = quant_cfg
        errs: list[Exception] = []
        for cls in candidates:
            try:
                return cls.from_pretrained(model_path, **kw), errs
            except ValueError as e:
                # ValueError = this head rejects the config (or the attn impl).
                logger.info("AutoModel head %s rejected arch=%r: %s - trying next candidate",
                            cls.__name__, arch, e)
                errs.append(e)
        return None, errs

    # Try the preferred attn impl, then the default. flash_attention_2 can fail
    # with ValueError (arch unsupported) or ImportError (a missing flash_attn dep):
    # both must degrade to the default rather than kill serving.
    errors: list[Exception] = []
    for attn in ([attn_impl, None] if attn_impl else [None]):
        try:
            model, errors = _load(attn)
        except ImportError as e:
            logger.warning("attn_implementation=%r unavailable (%s) - falling back to default", attn, e)
            continue
        if model is not None:
            if attn:
                logger.info("loaded with attn_implementation=%s", attn)
            return model
    sep = "\n\n----------\n\n"
    raise RuntimeError(
        f"None of the AutoModel heads accept config arch={arch!r}. "
        f"Errors: {sep.join(str(e) for e in errors)}"
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
        has_image = False
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
                has_image = True
            else:
                warning_once("dropping unsupported content part type %r", t)
                continue
        if has_image:
            out_msgs.append({"role": role, "content": parts})
        else:
            # A text model's chat template concatenates content as a string and
            # raises on a list of parts; only genuinely multimodal messages need
            # the list form, so collapse text-only content back to a string.
            out_msgs.append(
                {"role": role, "content": "".join(p["text"] for p in parts)}
            )
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
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype not in _dtype_map:
            warning_once(
                "unknown torch_dtype %r (expected one of %s); using bfloat16",
                torch_dtype, sorted(_dtype_map),
            )
        self._dtype = _dtype_map.get(torch_dtype, torch.bfloat16)

        self._processor = None
        self._tokenizer = None
        if prefer_processor:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
                self._tokenizer = self._processor.tokenizer
            except Exception as e:
                logger.warning(
                    "AutoProcessor failed for %s (%s); falling back to "
                    "AutoTokenizer — vision will be disabled", model_path, e,
                )
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

        self.max_model_len = self._derive_max_model_len(self._model.config)
        if self.max_model_len:
            logger.info("max_model_len=%d", self.max_model_len)
        else:
            warning_once(
                "could not derive max_model_len from config for %s; /v1/models "
                "will omit it and context-aware clients may assume a default",
                self.model_path,
            )

        self.max_safe_context = self._compute_max_safe_context(self._model.config)
        if self.max_safe_context:
            logger.info("max_safe_context=%d (KV budget from free VRAM)", self.max_safe_context)

    def _compute_max_safe_context(self, config) -> int | None:
        """Largest context (prompt + generation) that fits the free VRAM after
        weights, from the per-token KV-cache size. Approximate: it ignores
        activation / attention workspace, hence the safety fraction. Returns
        None (guard disabled) if it can't be computed, e.g. no CUDA."""
        if not torch.cuda.is_available():
            return None
        tc = getattr(config, "text_config", None) or config
        n_layers = getattr(tc, "num_hidden_layers", None)
        n_kv = getattr(tc, "num_key_value_heads", None) or getattr(tc, "num_attention_heads", None)
        head_dim = getattr(tc, "head_dim", None)
        if head_dim is None:
            hs, nh = getattr(tc, "hidden_size", None), getattr(tc, "num_attention_heads", None)
            head_dim = (hs // nh) if hs and nh else None
        if not (n_layers and n_kv and head_dim):
            return None
        dtype_bytes = torch.empty(0, dtype=self._dtype).element_size()
        per_token_kv = 2 * n_layers * n_kv * head_dim * dtype_bytes  # K + V
        free = sum(torch.cuda.mem_get_info(d)[0] for d in range(torch.cuda.device_count()))
        ctx = int(_KV_SAFETY_FRACTION * free) // per_token_kv
        if self.max_model_len:
            ctx = min(ctx, self.max_model_len)
        return int(ctx)

    @staticmethod
    def _derive_max_model_len(config) -> int | None:
        """Best-effort context window from a HF config.

        Multimodal models nest the LM config under `text_config`, so we look
        there first. Different families name the field differently.
        """
        keys = ("max_position_embeddings", "n_positions",
                "seq_length", "max_sequence_length")
        for cfg in (getattr(config, "text_config", None), config):
            if cfg is None:
                continue
            for key in keys:
                v = getattr(cfg, key, None)
                if v:
                    return int(v)
        return None

    # Per-field coercion. These clamp only *provided* values; a `None` is
    # handled by the caller as "inherit the model's generation_config".
    @staticmethod
    def _coerce_max_new_tokens(v: int | None) -> int:
        if v is None:
            return 512
        if not isinstance(v, int) or v <= 0:
            warning_once("max_tokens=%r invalid; using 512", v)
            return 512
        return v

    @staticmethod
    def _coerce_temperature(v: float) -> float:
        if v < 0:
            warning_once("temperature=%r invalid; using 0.0 (greedy)", v)
            return 0.0
        return float(v)

    @staticmethod
    def _coerce_top_p(v: float) -> float:
        if not (0.0 < v <= 1.0):
            warning_once("top_p=%r out of (0, 1]; using 1.0", v)
            return 1.0
        return float(v)

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
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
        except Exception as e:
            logger.warning(
                "apply_chat_template failed (%s); falling back to a plain "
                "role:content join — the model may not see the proper "
                "chat/tool format", e,
            )
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

        # Start from the model's own GenerationConfig (eos_token_id — the chat
        # turn-end such as Qwen's <|im_end|> — plus repetition_penalty, top_k,
        # top_p, …) and override only the fields the client explicitly set, so
        # the model's tuned sampling stands when the client omits a field.
        gen_cfg = copy.deepcopy(self._model.generation_config)
        gen_cfg.max_new_tokens = self._coerce_max_new_tokens(max_new_tokens)
        if temperature is not None:
            t = self._coerce_temperature(temperature)
            gen_cfg.do_sample = t > 0.0
            if t > 0.0:
                gen_cfg.temperature = t
        if top_p is not None:
            gen_cfg.top_p = self._coerce_top_p(top_p)
        # EOS / pad fallback for the rare model that ships no generation_config
        # eos — without this, generation would run to max_new_tokens.
        if gen_cfg.eos_token_id is None:
            gen_cfg.eos_token_id = self._tokenizer.eos_token_id
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = self._tokenizer.eos_token_id

        # Context guard: check before generate() so an over-budget request
        # returns a clean error (or is truncated) instead of crashing on an OOM
        # / position-index error partway through generation.
        if self.max_safe_context and _CTX_POLICY != "off":
            need = prompt_len + gen_cfg.max_new_tokens
            if need > self.max_safe_context:
                if _CTX_POLICY == "truncate":
                    keep = max(1, self.max_safe_context - gen_cfg.max_new_tokens)
                    for k in ("input_ids", "attention_mask"):
                        if k in inputs:
                            inputs[k] = inputs[k][:, -keep:]
                    prompt_len = inputs["input_ids"].shape[1]
                    warning_once("context %d > safe %d; truncated prompt to %d tokens",
                                 need, self.max_safe_context, prompt_len)
                else:
                    raise ContextOverflowError(prompt_len, gen_cfg.max_new_tokens,
                                               self.max_safe_context)

        gen_kwargs: dict = {}
        if stop:
            # transformers halts when any of these strings is produced; it
            # needs the tokenizer to detokenise the running output.
            gen_kwargs["stop_strings"] = stop
            gen_kwargs["tokenizer"] = self._tokenizer

        # Split prefill vs decode time with a no-op criterion that timestamps
        # the first generated token.
        timer = _FirstTokenTimer()
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([timer])
        t_start = time.perf_counter()
        with self._lock, torch.no_grad():
            out_ids = self._model.generate(
                **inputs, generation_config=gen_cfg, **gen_kwargs
            )
        t_end = time.perf_counter()
        prefill_ms = (timer.t_first - t_start) * 1000.0 if timer.t_first else None
        decode_ms = (t_end - timer.t_first) * 1000.0 if timer.t_first else None

        completion_ids = out_ids[0][prompt_len:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_len = int(completion_ids.shape[0])
        # transformers appends the matched stop string to the output; trim it.
        if stop:
            for s in stop:
                if s and text.endswith(s):
                    text = text[: -len(s)]
                    break
        finish = "length" if completion_len >= gen_cfg.max_new_tokens else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=int(prompt_len),
            completion_tokens=completion_len,
            finish_reason=finish,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
        )
