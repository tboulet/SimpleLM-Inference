# SimpleLM

OpenAI-compatible LLM inference server built on top of HuggingFace
transformers. Designed for clusters where the canonical fast engines
(vLLM, SGLang, TGI) are blocked by toolchain / glibc / GPU-arch
mismatches — Adastra (AMD MI250X) and similar.

The goal is **correctness first, perf later**:
- Drop-in `/v1/chat/completions` and `/v1/completions` endpoints, so
  existing OpenAI / LiteLLM clients work unchanged.
- Tool-call parsing per model family (Gemma-4, Qwen3, GLM, Kimi-K2,
  MiniMax-M2, gpt-oss, DeepSeek-V4). Reuses the parser conventions from
  the SGLang cookbook so a model that works on JZ via SGLang also works
  here.
- Multimodal: text + image content arrays, base64 data URLs or local
  file paths (no compute-node-side HTTPS fetch).
- Backends are pluggable — `simplelm.backends.HuggingFaceBackend` is the
  default; future `VLLMBackend` / `SGLangBackend` slot in without
  touching the server layer.

## When to use SimpleLM

- Your cluster has ROCm + torch but no working vLLM/SGLang wheel.
- You want the same client code working on JZ (SGLang) and Adastra
  (SimpleLM) without if-branches.
- You need a synchronous baseline to compare against a faster engine.

## When NOT to use SimpleLM

- You have vLLM/SGLang running. They're 5–20× faster per token thanks
  to paged KV cache + flash attention + CUDA graphs.
- You need streaming (SSE) — current SimpleLM serves only synchronous
  completions. Streaming support is on the roadmap.
- You're serving high-throughput multi-tenant traffic.

## Quick start

```bash
pip install -e .

# Serve a HF model
simplelm serve --model-path /path/to/Qwen2.5-3B-Instruct --port 9876

# Or as a library
from simplelm import HuggingFaceBackend, serve
serve(HuggingFaceBackend("/path/to/Qwen2.5-3B-Instruct"), port=9876)
```

## Architecture

```
clients ────────► FastAPI server (simplelm.server)
                    │
                    │ ChatCompletionRequest
                    ▼
                 Backend.generate(messages, tools, images, max_tokens, …)
                    │
                    ▼
              ┌─────┴─────┐
              │ HFBackend │ ◄── transformers + torch + tools/parsers + vision/utils
              └───────────┘
```

## Status

- [x] Synchronous `/v1/chat/completions` with chat template
- [x] HuggingFace transformers backend
- [x] Vision: base64 data URLs in `image_url` content
- [x] Pluggable tool-call parser registry
- [ ] Streaming (SSE)
- [ ] Tool-call parsers for: gemma4, kimi_k2, glm45, minimax-m2,
      qwen3_coder, deepseek-v4, gpt-oss
- [ ] vLLM backend
- [ ] SGLang backend
- [ ] Batched concurrency

## License

MIT.
